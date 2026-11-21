"""
Phase 5: a web page where anyone can ask a question about the indexed 10-K filings.

Run (from the project folder, after the pipeline has built the index and trained the reranker):
    pip install streamlit
    $env:GEMINI_API_KEY = "your-key"          # in PowerShell; never put the key in this file
    streamlit run app.py

A browser tab opens at http://localhost:8501. Type a question such as
"What were Apple's total assets in fiscal 2025?" and press Ask.

What happens for each question (the same steps as the experiments):
  1. Search    embed the question and take the top 20 chunks in every configuration
  2. Rerank    compute the same features as rerank_data.py and score the candidates with the saved
               model in data/models/ (trained on all questions, so meant for NEW questions like these)
  3. Answer    send the top 5 chunks of the chosen configuration to the LLM, same prompt as generate.py
  4. Show      the answer, plus the 5 excerpts with company, filing, source, section and page

The sidebar picks the configuration, the reranker (or none) and the LLM. "mock" needs no API key: it
just echoes the first number in the evidence, to try the page without using any requests.
Every answer is cached, so asking the same question again costs no request.
"""
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

import embed_index as ei
import generate as gen
import rerank_data as rd

MODELS = ei.DATA_DIR / "models"
K_CANDIDATES = rd.K           # 20, as in training
K_EVIDENCE = 5                # as in generate.py
CONFIG_NAMES = {
    "html_only": "HTML only (baseline)",
    "html+naive_pdf": "HTML + naive PDF",
    "html+layout_pdf": "HTML + layout PDF",
    "html+real_naive": "HTML + company PDF (naive)",
    "html+real_layout": "HTML + company PDF (layout)",
}


# ---------- cached resources (loaded once per session) ----------
@st.cache_resource(show_spinner="Loading the search index...")
def load_index():
    return ei.Index(ei.EMBED_PROVIDER)


@st.cache_resource
def load_reranker(name):
    path = MODELS / f"reranker_{name}.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_names():
    return rd.company_names()


def available_rerankers():
    found = [p.stem.removeprefix("reranker_") for p in sorted(MODELS.glob("reranker_*.joblib"))]
    return found


def configs_in_index(idx):
    present = set(idx.sources.tolist())
    return [c for c in ei.CONFIGS if set(ei.CONFIGS[c]) - {"html_text", "html_table"} <= present]


# ---------- the pipeline for one question ----------
def retrieve(idx, question, config, reranker, names, k_evidence=K_EVIDENCE):
    """Top evidence chunks for one question: [(chunk, embedding_sim, reranker_score or None)]."""
    qvec = ei.embed_texts([question], idx.provider, is_query=True)[0]
    pool, cfg_rank = {}, {}
    for cfg, sources in ei.CONFIGS.items():       # same candidate pool as rerank_data.build
        for rank, (sim, c) in enumerate(idx.search_vec(qvec, k=K_CANDIDATES, sources=sources), 1):
            pool.setdefault(c["chunk_id"], {"chunk": c, "sim": sim})
            if cfg == config:
                cfg_rank[c["chunk_id"]] = rank
    cands = [e for e in pool.values() if e["chunk"]["chunk_id"] in cfg_rank]
    if not cands:
        return []
    if reranker is None:
        cands.sort(key=lambda e: cfg_rank[e["chunk"]["chunk_id"]])
        return [(e["chunk"], e["sim"], None) for e in cands[:k_evidence]]

    ordered = sorted(pool.values(), key=lambda e: -e["sim"])
    max_sim = ordered[0]["sim"]
    pool_rank = {e["chunk"]["chunk_id"]: i for i, e in enumerate(ordered, 1)}
    q = {"question": question}
    keys = {}
    rows = []
    for e in cands:
        c = e["chunk"]
        if c["ticker"] not in keys:
            keys[c["ticker"]] = rd.name_keys(c["ticker"], names)
        f = rd.features(q, c, e["sim"], max_sim, pool_rank[c["chunk_id"]], keys[c["ticker"]])
        rows.append([f[name] for name in reranker["features"]])
    scores = reranker["model"].predict_proba(np.array(rows, dtype=float))[:, 1]
    order = sorted(range(len(cands)), key=lambda i: (-scores[i], cfg_rank[cands[i]["chunk"]["chunk_id"]]))
    return [(cands[i]["chunk"], cands[i]["sim"], float(scores[i])) for i in order[:k_evidence]]


def answer(llm, question, evidence):
    prompt = gen.build_prompt(question, gen.format_context([c for c, _, _ in evidence]))
    return llm.ask(prompt)


# ---------- page ----------
def main():
    st.set_page_config(page_title="10-K Question Answering", page_icon="📄", layout="wide")
    st.title("Ask a question about a company's 10-K")
    st.caption("Retrieval-augmented question answering over SEC annual reports. "
               "Answers come only from the excerpts shown below each answer.")

    try:
        idx = load_index()
    except FileNotFoundError as e:
        st.error(f"{e}\n\nBuild the index first with `python embed_index.py`.")
        st.stop()

    tickers = sorted({c["ticker"] for c in idx.chunks})
    names = load_names()

    with st.sidebar:
        st.header("Settings")
        cfgs = configs_in_index(idx)
        config = st.selectbox("Evidence sources", cfgs, format_func=lambda c: CONFIG_NAMES.get(c, c))
        rerankers = available_rerankers()
        labels = {"none": "None (embedding order)", "logreg": "Logistic regression", "xgboost": "XGBoost",
                  "histgb": "Gradient boosting (scikit-learn)"}
        choice = st.selectbox("Reranker", rerankers + ["none"], format_func=lambda r: labels.get(r, r))
        if not rerankers:
            st.info("No saved reranker in data/models. Run train_reranker.py to enable reranking.")
        provider = st.selectbox("LLM", ["gemini", "groq", "openai", "ollama", "mock"],
                                help="mock needs no API key: it echoes the first number in the evidence.")
        model = st.text_input("Model name", value=gen.PROVIDERS[provider][1])
        st.divider()
        st.caption(f"Index: {len(idx.chunks):,} chunks from {len(tickers)} companies.")
        with st.expander("Companies in the index"):
            st.write(", ".join(f"{t} ({names.get(t, t)})" for t in tickers))

    with st.form("ask"):
        question = st.text_input("Your question",
                                 placeholder="What were Apple's total assets in fiscal 2025?")
        submitted = st.form_submit_button("Ask")
    st.caption("Works best for figures in the financial statements. Name the company and the fiscal year.")

    if not (submitted and question.strip()):
        return

    try:
        llm = gen.LLM(provider, model=model or None)
    except SystemExit as e:
        st.error(str(e))
        st.stop()
    reranker = None if choice == "none" else load_reranker(choice)

    with st.spinner("Searching and reranking..."):
        evidence = retrieve(idx, question.strip(), config, reranker, names)
    if not evidence:
        st.warning("Search found nothing in these sources.")
        return
    with st.spinner("Asking the LLM..."):
        try:
            reply, cached = answer(llm, question.strip(), evidence)
        except SystemExit as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"The LLM call failed: {e}")
            st.stop()

    st.subheader("Answer")
    if "not found" in reply.lower():
        st.warning("NOT FOUND: the top excerpts don't contain the answer. "
                   "Try naming the company and fiscal year, or another evidence source.")
    else:
        st.markdown(f"### {reply}")
    st.caption(f"{llm.model} via {provider}{' (cached)' if cached else ''} · "
               f"{CONFIG_NAMES.get(config, config)} · "
               f"{labels.get(choice, choice)}. Table figures are usually in millions.")

    st.subheader("Evidence the answer is based on")
    for i, (c, sim, score) in enumerate(evidence, 1):
        name = names.get(c["ticker"], c["ticker"])
        where = " · ".join(b for b in [c["filing"], c["source"], c.get("section", ""),
                                        f"page {c['page']}" if c.get("page") else ""] if b)
        stats = f"similarity {sim:.2f}" + (f" · reranker score {score:.2f}" if score is not None else "")
        with st.expander(f"[{i}] {name}: {c.get('title') or where}", expanded=(i == 1)):
            st.caption(f"{where} · {stats}")
            st.text(c["text"][:gen.MAX_CHARS_PER_CHUNK])


if __name__ == "__main__":
    main()