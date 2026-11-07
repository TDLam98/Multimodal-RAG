"""
Build the labeled pairs and features the reranker learns from.

Run (from the project folder, after embed_index.py has built the index):
    python rerank_data.py                                # reads questions.csv
    python rerank_data.py --questions my.csv --provider local --k 20

For every question it retrieves the top K chunks from each of the three retrieval
configurations, pools them, and writes one row per (question, chunk) pair to
data/rerank/pairs.csv with:
    label       1 if the chunk is relevant (same rule as retrieval_check.py), else 0
    rank_<cfg>  the chunk's rank in that configuration's top K (blank if it was not there)
    features    the numbers the classifier sees (FEATURES below)

THE ANTI-LEAKAGE RULE: features use only the question text and the chunk. They never use the
answer, the row label or the gold filing, which exist only to produce the labels. A model that
could see the answer would look excellent and be useless.

The company-name feature is a heuristic, like a real system would use: it checks whether the
chunk's company name or ticker appears in the question text.
"""
import argparse
import csv
import json
import math
import re
from datetime import date
from pathlib import Path

import embed_index as ei
import retrieval_check as rc

K = 20
OUT = ei.DATA_DIR / "rerank" / "pairs.csv"
NAMES_CACHE = ei.DATA_DIR / "company_names.json"

SOURCES = ["html_text", "html_table", "pdf_naive", "pdf_layout", "real_naive", "real_layout"]
FEATURES = [
    "sim", "sim_gap", "log_pool_rank",
    *[f"src_{s}" for s in SOURCES], "src_other",
    "has_title", "pipe_line_share", "digit_share", "log_chars",
    "word_overlap", "word_jaccard",
    "year_in_chunk", "fiscal_year_match", "company_match", "financial_section",
]

STOP = set("""what was were is are the of a an in for as by to and its it on at from how much many
did does do reported report reports form 10 k fiscal year years ended end ending company s which
during total""".split())
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NAME_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|companies|holdings|holding|group|plc|ltd|"
    r"limited|the|international|technologies|technology)\b\.?")


# ---------- company names ----------
def company_names():
    if NAMES_CACHE.exists():
        return json.loads(NAMES_CACHE.read_text(encoding="utf-8"))
    try:
        from ingest import sec_get
        data = sec_get("https://www.sec.gov/files/company_tickers.json").json()
        names = {row["ticker"].upper(): row["title"].strip() for row in data.values()}
        NAMES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        NAMES_CACHE.write_text(json.dumps(names, indent=1), encoding="utf-8")
        return names
    except Exception as e:
        print(f"Could not fetch company names ({e}); company_match will use tickers only.")
        return {}


def name_keys(ticker, names):
    """Lowercase strings that identify a company in a question."""
    keys = set()
    name = names.get(ticker, "")
    clean = re.sub(r"\s+", " ", NAME_SUFFIXES.sub(" ", name.lower().replace("&", " "))).strip(" ,.")
    if clean:
        keys.add(clean)
        keys.add(clean.replace(" ", ""))          # "exxon mobil" also matches "exxonmobil"
        first = clean.split()[0]
        if len(first) >= 4:
            keys.add(first)
    return keys


def company_match(question, ticker, keys):
    q = question.lower()
    if re.search(rf"\b{re.escape(ticker)}\b", question):  # ticker written in capitals
        return 1
    q_compact = re.sub(r"[^a-z0-9]", "", q)
    return int(any((k in q) or (len(k) >= 4 and k.replace(" ", "") in q_compact) for k in keys))


# ---------- features ----------
def words(s):
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 1}


def fiscal_year_match(q_years, report_date):
    try:
        d = date.fromisoformat(report_date)
    except ValueError:
        return 0
    years = {d.year} | ({d.year - 1} if d.month == 1 else set())   # early-January year ends
    return int(bool(q_years & years))


def features(q, chunk, sim, max_sim, pool_rank, keys):
    text = chunk["text"]
    lines = [l for l in text.splitlines() if l.strip()] or [""]
    qw, cw = words(q["question"]), words(text)
    q_years = set(YEAR_RE.findall(q["question"]))          # strings, e.g. {"2025"}
    src = chunk["source"]
    f = {
        "sim": sim,
        "sim_gap": sim - max_sim,
        "log_pool_rank": math.log(pool_rank),
        **{f"src_{s}": int(src == s) for s in SOURCES},
        "src_other": int(src not in SOURCES),
        "has_title": int(bool(chunk.get("title"))),
        "pipe_line_share": sum(" | " in l for l in lines) / len(lines),
        "digit_share": sum(ch.isdigit() for ch in text) / max(len(text), 1),
        "log_chars": math.log(len(text) + 1),
        "word_overlap": len(qw & cw) / len(qw) if qw else 0.0,
        "word_jaccard": len(qw & cw) / len(qw | cw) if (qw | cw) else 0.0,
        "year_in_chunk": int(any(y in text for y in q_years)),
        "fiscal_year_match": fiscal_year_match({int(y) for y in q_years}, chunk.get("report_date", "")),
        "company_match": company_match(q["question"], chunk["ticker"], keys),
        "financial_section": int(chunk.get("section", "") in {"Item 7", "Item 7A", "Item 8"}),
    }
    return f


# ---------- building the pairs ----------
def build(questions_path, provider, k=K, out=None):
    out = Path(out or OUT)                  # resolved at call time, not import time
    questions = rc.load_questions(questions_path)
    idx = ei.Index(provider)
    in_index = {c["ticker"] for c in idx.chunks}
    kept = [q for q in questions if q["filing"].partition("_")[0] in in_index]
    for q in questions:
        if q not in kept:
            print(f"Skipping {q['id']}: company {q['filing'].partition('_')[0]!r} is not in the index.")
    if not kept:
        raise SystemExit("No usable questions.")

    names = company_names()
    key_cache = {}
    qvecs = ei.embed_texts([q["question"] for q in kept], provider, is_query=True)
    configs = list(ei.CONFIGS)
    rows, no_positive = [], []

    for q, qvec in zip(kept, qvecs):
        pool = {}
        for cfg in configs:
            for rank, (sim, c) in enumerate(idx.search_vec(qvec, k=k, sources=ei.CONFIGS[cfg]), 1):
                entry = pool.setdefault(c["chunk_id"], {"chunk": c, "sim": sim})
                entry[f"rank_{cfg}"] = rank
        ordered = sorted(pool.values(), key=lambda e: -e["sim"])
        max_sim = ordered[0]["sim"] if ordered else 0.0
        any_pos = False
        for pool_rank, e in enumerate(ordered, 1):
            c = e["chunk"]
            if c["ticker"] not in key_cache:
                key_cache[c["ticker"]] = name_keys(c["ticker"], names)
            label = int(rc.is_relevant(c, q))
            any_pos |= bool(label)
            rows.append({
                "qid": q["id"], "group": q["filing"].partition("_")[0], "q_label": q.get("label", "") or "unlabeled",
                "chunk_id": c["chunk_id"], "source": c["source"], "label": label,
                **{f"rank_{cfg}": e.get(f"rank_{cfg}", "") for cfg in configs},
                **features(q, c, e["sim"], max_sim, pool_rank, key_cache[c["ticker"]]),
            })
        if not any_pos:
            no_positive.append(q["id"])

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pos = sum(r["label"] for r in rows)
    groups = len({r["group"] for r in rows})
    print(f"Wrote {len(rows)} pairs for {len(kept)} questions from {groups} companies to {out}")
    print(f"Positives: {pos} ({pos / len(rows):.1%}). Questions with no relevant chunk in any top {k}: "
          f"{len(no_positive)}{' ' + str(no_positive[:10]) if no_positive else ''}")
    if no_positive:
        print("  Reranking cannot help those: the right chunk never reached the candidate list. "
              "Check their answers and row labels, or raise --k.")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build labeled (question, chunk) pairs with features.")
    ap.add_argument("--questions", default=str(Path(__file__).resolve().parent / "questions.csv"))
    ap.add_argument("--provider", choices=["openai", "local", "hash"], default=ei.EMBED_PROVIDER)
    ap.add_argument("--k", type=int, default=K, help="candidates per configuration (default 20)")
    args = ap.parse_args()
    build(args.questions, args.provider, args.k)