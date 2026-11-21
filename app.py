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

Question log: every question asked here is saved to data/app_questions.csv with its answer, settings and
evidence, plus the user's 👍 / 👎 feedback and an optional corrected answer. NOT FOUND answers mark gaps in
the system. This log is a separate pool, never the test set: promote_questions.py turns the useful entries
into unverified candidate questions for a person to check before they join questions.csv or the training data.
"""
import csv
import html
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

import embed_index as ei
import generate as gen
import rerank_data as rd

MODELS = ei.DATA_DIR / "models"
LOG_PATH = ei.DATA_DIR / "app_questions.csv"
LOG_FIELDS = ["log_id", "time", "question", "config", "reranker", "model", "answer", "not_found",
              "top_filing", "top_source", "evidence_ids", "feedback", "correct_answer"]
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


# ---------- look ----------
CSS = """
<style>
.block-container {padding-top: 3.5rem; max-width: 1200px;}
button[kind="primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {background: #10B981 !important; border-color: #10B981 !important; color: #fff !important;}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover {background: #059669 !important; border-color: #059669 !important;}
.hero {background: linear-gradient(120deg, #0B1F3A 0%, #13294B 55%, #0F766E 100%); border-radius: 18px;
       padding: 28px 32px; color: #fff; margin-bottom: 18px; position: relative; overflow: hidden;}
.hero:after {content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
       border-radius: 50%; background: rgba(16,185,129,.25);}
.hero h1 {color: #fff; font-size: 2.3rem; margin: 0 0 6px 0; padding: 0;}
.hero p {color: #CBD5E1; margin: 0 0 14px 0; font-size: 1.02rem;}
.chip {display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: .85rem; font-weight: 600;
       margin: 0 6px 6px 0; background: rgba(255,255,255,.12); color: #fff; border: 1px solid rgba(255,255,255,.18);}
.answer {border-radius: 16px; padding: 22px 26px; margin: 8px 0 14px 0; background: #ECFDF5; border: 1px solid #A7F3D0;}
.answer.nf {background: #FFFBEB; border-color: #FDE68A;}
.answer .label {font-size: .8rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #047857;}
.answer.nf .label {color: #B45309;}
.answer .value {font-size: 2.6rem; font-weight: 800; color: #0B1F3A; line-height: 1.15; margin: 4px 0;}
.answer.nf .value {font-size: 1.5rem;}
.answer .sub {color: #475569; font-size: .92rem;}
.stat {border-radius: 14px; padding: 14px 16px; background: #F8FAFC; border: 1px solid #E2E8F0;}
.stat .k {font-size: .75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: .06em;}
.stat .v {font-size: 1.05rem; font-weight: 700; color: #0B1F3A; margin-top: 2px;}
.badge {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 700; color: #fff;}
.meta {color: #64748B; font-size: .85rem;}
.excerpt {background: #F8FAFC; border-radius: 10px; padding: 12px 14px; font-family: ui-monospace, Menlo, Consolas, monospace;
          font-size: .85rem; white-space: pre-wrap; color: #0F172A; max-height: 340px; overflow-y: auto;}
.scorebar {height: 6px; border-radius: 3px; background: #E2E8F0; margin: 6px 0 10px 0;}
.scorebar > div {height: 6px; border-radius: 3px; background: #10B981;}
</style>
"""
SOURCE_STYLE = {"html_table": ("#10B981", "HTML table"), "html_text": ("#059669", "HTML text"),
                "pdf_naive": ("#F59E0B", "PDF, naive"), "pdf_layout": ("#D97706", "PDF, layout"),
                "real_naive": ("#38BDF8", "Company PDF, naive"), "real_layout": ("#0284C7", "Company PDF, layout")}
EXAMPLES = [
    ("Apple net sales, FY2025", "What were Apple's total net sales for the fiscal year ended September 27, 2025, as reported in its Form 10-K (in millions)?"),
    ("Apple net income, FY2025", "What was Apple's net income for the fiscal year ended September 27, 2025, as reported in its Form 10-K (in millions)?"),
    ("Abbott net sales, 2025", "What were Abbott Laboratories' net sales for the year ended December 31, 2025, as reported in its Form 10-K (in millions)?"),
    ("Arch Capital total assets, 2025", "What were Arch Capital's total assets as of December 31, 2025, as reported in its Form 10-K (in millions)?"),
]
RERANKER_LABELS = {"none": "None (embedding order)", "logreg": "Logistic regression", "xgboost": "XGBoost",
                   "histgb": "Gradient boosting (scikit-learn)"}


def badge(source):
    color, label = SOURCE_STYLE.get(source, ("#64748B", source))
    return f'<span class="badge" style="background:{color}">{html.escape(label)}</span>'


def use_example(q):
    st.session_state["question"] = q
    st.session_state["run"] = True


# ---------- page ----------
def main():
    st.set_page_config(page_title="10-K Explorer", page_icon="📊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        idx = load_index()
    except FileNotFoundError as e:
        st.error(f"{e}\n\nBuild the index first with `python embed_index.py`.")
        st.stop()
    tickers = sorted({c["ticker"] for c in idx.chunks})
    names = load_names()

    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        cfgs = configs_in_index(idx)
        config = st.selectbox("📚 Evidence sources", cfgs, format_func=lambda c: CONFIG_NAMES.get(c, c))
        rerankers = available_rerankers()
        preferred = [r for r in ["xgboost", "logreg"] if r in rerankers] + [r for r in rerankers if r not in ("xgboost", "logreg")]
        choice = st.selectbox("🏆 Reranker", preferred + ["none"], format_func=lambda r: RERANKER_LABELS.get(r, r))
        if not rerankers:
            st.info("No saved reranker in data/models. Run train_reranker.py to enable reranking.")
        provider = st.selectbox("🤖 LLM", ["gemini", "groq", "openai", "ollama", "mock"],
                                help="mock needs no API key: it echoes the first number in the evidence.")
        model = st.text_input("Model name", value=gen.PROVIDERS[provider][1])
        st.divider()
        with st.expander(f"🏢 {len(tickers)} companies in the index"):
            st.write(", ".join(f"{t} ({names.get(t, t)})" for t in tickers))
        recent_questions_panel()

    st.markdown(f"""
<div class="hero">
  <h1>📊 10-K Explorer</h1>
  <p>Ask about any figure in a company's annual report. Every answer comes with the excerpts it was found in.</p>
  <span class="chip">🔎 {len(idx.chunks):,} chunks</span>
  <span class="chip">🏢 {len(tickers)} companies</span>
  <span class="chip">🏆 trained reranker</span>
  <span class="chip">🤖 Gemini answers</span>
</div>""", unsafe_allow_html=True)

    st.markdown("**Try an example**")
    cols = st.columns(len(EXAMPLES))
    for col, (label, q) in zip(cols, EXAMPLES):
        col.button(label, on_click=use_example, args=(q,), use_container_width=True)

    st.session_state.setdefault("question", "")
    with st.form("ask"):
        st.text_input("Your question", key="question",
                      placeholder="What were Apple's total net sales for fiscal 2025?")
        submitted = st.form_submit_button("Ask  ➜", type="primary")
    st.caption("Tip: name the company, the line item and the fiscal year. Figures from the financial statements work best.")

    question = st.session_state.get("question", "").strip()
    run = st.session_state.pop("run", False) or submitted
    if run and question:
        result = ask(idx, names, question, config, choice, provider, model)
        if result:
            st.session_state["result"] = result
    if "result" in st.session_state:
        show(st.session_state["result"], names)


def ask(idx, names, question, config, choice, provider, model):
    """Run search, rerank and the LLM once, log the question, and return the result to display."""
    try:
        llm = gen.LLM(provider, model=model or None)
    except SystemExit as e:
        st.error(str(e))
        st.stop()
    reranker = None if choice == "none" else load_reranker(choice)
    with st.spinner("Searching and reranking..."):
        evidence = retrieve(idx, question, config, reranker, names)
    if not evidence:
        st.warning("Search found nothing in these sources.")
        return None
    with st.spinner("Asking the LLM..."):
        try:
            reply, cached = answer(llm, question, evidence)
        except SystemExit as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"The LLM call failed: {e}")
            st.stop()
    result = {"question": question, "config": config, "choice": choice, "model": llm.model, "cached": cached,
              "reply": reply, "evidence": evidence}
    result["log_id"] = log_question(result)
    return result


def show(r, names):
    evidence, reply, top = r["evidence"], r["reply"], r["evidence"][0][0]
    if is_not_found(reply):
        st.markdown("""
<div class="answer nf"><div class="label">⚠️ Not found</div>
<div class="value">The top excerpts don't contain the answer.</div>
<div class="sub">Try naming the company and fiscal year, using the filing's wording (e.g. "net sales"), or another evidence source.
This question was saved so we can see where the system falls short.</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="answer"><div class="label">✅ Answer</div>
<div class="value">{html.escape(reply)}</div>
<div class="sub">Most relevant excerpt: {html.escape(names.get(top["ticker"], top["ticker"]))} · {html.escape(top["filing"])} · {badge(top["source"])}
&nbsp;· table figures are usually in millions</div></div>""", unsafe_allow_html=True)
        feedback_box(r)

    c1, c2, c3, c4 = st.columns(4)
    first = evidence[0]
    stats = [("Evidence sources", CONFIG_NAMES.get(r["config"], r["config"])), ("Reranker", RERANKER_LABELS.get(r["choice"], r["choice"])),
             ("Model", f"{r['model']}{' (cached)' if r['cached'] else ''}"),
             ("Top reranker score" if first[2] is not None else "Top similarity",
              f"{(first[2] if first[2] is not None else first[1]):.2f}")]
    for col, (k, v) in zip((c1, c2, c3, c4), stats):
        col.markdown(f'<div class="stat"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("### 📄 Evidence the answer is based on")
    for i, (c, sim, score) in enumerate(evidence, 1):
        name = names.get(c["ticker"], c["ticker"])
        title = c.get("title") or c.get("section") or c["filing"]
        with st.expander(f"[{i}]  {name}  ·  {title[:110]}", expanded=(i == 1)):
            where = " · ".join(b for b in [c["filing"], c.get("section", ""), f"page {c['page']}" if c.get("page") else ""] if b)
            val = score if score is not None else sim
            st.markdown(f'{badge(c["source"])} &nbsp;<span class="meta">{html.escape(where)} · similarity {sim:.2f}'
                        f'{f" · reranker score {score:.2f}" if score is not None else ""}</span>'
                        f'<div class="scorebar"><div style="width:{max(0.0, min(1.0, val)) * 100:.0f}%"></div></div>'
                        f'<div class="excerpt">{html.escape(c["text"][:gen.MAX_CHARS_PER_CHUNK])}</div>',
                        unsafe_allow_html=True)


# ---------- question log and feedback ----------
def is_not_found(reply):
    return "not found" in reply.lower()


def read_log():
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_log(rows):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)


def log_question(r):
    """Append one asked question to data/app_questions.csv. This is a separate pool: never the test set."""
    rows = read_log()
    log_id = f"app{len(rows) + 1:04d}"
    top = r["evidence"][0][0]
    rows.append({"log_id": log_id, "time": datetime.now().isoformat(timespec="seconds"), "question": r["question"],
                 "config": r["config"], "reranker": r["choice"], "model": r["model"], "answer": r["reply"],
                 "not_found": int(is_not_found(r["reply"])), "top_filing": top["filing"], "top_source": top["source"],
                 "evidence_ids": " ".join(c["chunk_id"] for c, _, _ in r["evidence"]), "feedback": "", "correct_answer": ""})
    write_log(rows)
    return log_id


def set_feedback(log_id, value, correct=""):
    rows = read_log()
    for row in rows:
        if row["log_id"] == log_id:
            row["feedback"] = value
            if correct:
                row["correct_answer"] = correct
    write_log(rows)
    if "result" in st.session_state:
        st.session_state["result"]["feedback"] = value


def save_correction(log_id):
    set_feedback(log_id, "down", st.session_state.get(f"fix_{log_id}", "").strip())
    st.session_state["result"]["corrected"] = True


def feedback_box(r):
    fb = r.get("feedback")
    if fb == "up":
        st.success("Thanks! Marked as correct.")
        return
    if fb == "down":
        if r.get("corrected"):
            st.info("Thanks! Your correction was saved for review.")
            return
        c1, c2 = st.columns([3, 1])
        c1.text_input("What is the correct answer? (optional)", key=f"fix_{r['log_id']}")
        c2.button("Save", on_click=save_correction, args=(r["log_id"],), use_container_width=True)
        return
    c1, c2, _ = st.columns([1, 1, 4])
    c1.button("👍 Correct", on_click=set_feedback, args=(r["log_id"], "up"), use_container_width=True)
    c2.button("👎 Wrong", on_click=set_feedback, args=(r["log_id"], "down"), use_container_width=True)


def recent_questions_panel():
    rows = read_log()
    with st.expander(f"📝 Asked questions ({len(rows)})"):
        if not rows:
            st.caption("Questions asked here are saved to data/app_questions.csv.")
        for row in reversed(rows[-8:]):
            icon = "⚠️" if row["not_found"] == "1" else {"up": "✅", "down": "❌"}.get(row["feedback"], "•")
            st.markdown(f"{icon} {row['question'][:80]}")
        if rows:
            n_nf = sum(r["not_found"] == "1" for r in rows)
            n_up = sum(r["feedback"] == "up" for r in rows)
            st.caption(f"{n_up} marked correct · {n_nf} not found · saved in data/app_questions.csv")


if __name__ == "__main__":
    main()