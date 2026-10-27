"""
Recall@k smoke test: how often does each retrieval configuration put the right evidence
in its top k results?

Run (from the project folder, after embed_index.py has built the index):
    python retrieval_check.py                          # reads questions.csv next to this file
    python retrieval_check.py --questions my.csv --provider local

questions.csv columns (see questions_template.csv):
    id         any short ID: q01, q02, ...
    filing     the filing the answer comes from, named as in data/html, e.g. MMM_2025-12-31
    question   a standalone question that names the company and the fiscal year
    answer     the answer exactly as printed in the filing, e.g. 359,241
    row_label  (recommended) the row label printed next to it, e.g. Total assets
    label      (optional) where the answer lives: text, table or both

How "relevant" is decided: a retrieved chunk is relevant if it comes from the right filing and
contains the answer (and the row label, if given), ignoring $ signs, commas and capitalization.
That is a quick automatic rule for a smoke test. It can over-count, since the same number can
appear in more than one place in a filing.

Besides recall@k, it reports ANSWER COVERAGE: for each source, the share of questions whose
answer appears in at least one chunk from that source. If the answer never made it into the
pdf_naive chunks, no retriever could have found it there. That is the cleanest evidence of what
each PDF parsing method preserved or lost.

Outputs: data/results/retrieval_summary.csv and data/results/retrieval_details.csv
"""
import argparse
import csv
import re
from pathlib import Path

import embed_index as ei

KS = [1, 3, 5, 10]
REQUIRED = ["id", "filing", "question", "answer"]


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower().replace("$", "").replace(",", "")).strip()


def is_relevant(chunk, q):
    if chunk["filing"] != q["filing"]:
        return False
    text = norm(chunk["text"])
    if norm(q["answer"]) not in text:
        return False
    label = norm(q.get("row_label", ""))
    return (not label) or label in text


def load_questions(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader
                if (row.get("id") or "").strip()]


def evaluate(questions, idx, ks=KS):
    qvecs = ei.embed_texts([q["question"] for q in questions], idx.provider, is_query=True)

    by_filing = {}
    needed = {q["filing"] for q in questions}
    for c in idx.chunks:
        if c["filing"] in needed:
            by_filing.setdefault(c["filing"], []).append(c)
    all_sources = sorted(set(idx.sources.tolist()))

    rows = []
    for q, qvec in zip(questions, qvecs):
        coverage = {s: any(is_relevant(c, q) for c in by_filing[q["filing"]] if c["source"] == s)
                    for s in all_sources}
        for config, sources in ei.CONFIGS.items():
            results = idx.search_vec(qvec, k=max(ks), sources=sources)
            rank = next((r for r, (_, c) in enumerate(results, 1) if is_relevant(c, q)), None)
            rows.append({"id": q["id"], "filing": q["filing"], "label": q.get("label", "") or "unlabeled",
                         "config": config, "first_rank": rank,
                         **{f"in_{s}": coverage[s] for s in all_sources}})
    return rows, all_sources


def summarize(rows, ks=KS):
    out = []
    groups = ["all"] + sorted({r["label"] for r in rows})
    for group in groups:
        for config in ei.CONFIGS:
            sel = [r for r in rows if r["config"] == config and (group == "all" or r["label"] == group)]
            if not sel:
                continue
            n = len(sel)
            rec = {f"R@{k}": sum(1 for r in sel if r["first_rank"] and r["first_rank"] <= k) / n for k in ks}
            mrr = sum((1 / r["first_rank"]) if r["first_rank"] else 0 for r in sel) / n
            out.append({"group": group, "config": config, "n": n, **rec, "MRR": mrr})
    return out


def print_summary(summary, ks=KS):
    for group in dict.fromkeys(s["group"] for s in summary):
        sel = [s for s in summary if s["group"] == group]
        print(f"\n=== {group} (n={sel[0]['n']}) ===")
        print(f"{'config':<18}" + "".join(f"{'R@' + str(k):>7}" for k in ks) + f"{'MRR':>7}")
        for s in sel:
            print(f"{s['config']:<18}" + "".join(f"{s['R@' + str(k)]:>7.2f}" for k in ks) + f"{s['MRR']:>7.2f}")


def print_coverage(rows, all_sources):
    one_per_q = {r["id"]: r for r in rows}.values()
    n = len(one_per_q)
    print(f"\n=== Answer coverage: share of questions whose answer appears in each source (n={n}) ===")
    for s in all_sources:
        share = sum(1 for r in one_per_q if r[f"in_{s}"]) / n
        print(f"  {s:<12}{share:>6.0%}")
    missing = [r["id"] for r in one_per_q if not any(r[f"in_{s}"] for s in all_sources)]
    if missing:
        print(f"  WARNING: no source contains the answer for {missing}. Check the answer text, "
              "the filing name, or the row label.")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run(questions_path, provider):
    questions = load_questions(questions_path)
    idx = ei.Index(provider)
    in_index = {c["filing"] for c in idx.chunks}
    kept = []
    for q in questions:
        if q["filing"] in in_index:
            kept.append(q)
        else:
            print(f"Skipping {q['id']}: filing {q['filing']!r} is not in the index.")
    if not kept:
        raise SystemExit("No usable questions. Filing names must match files in data/html.")

    rows, all_sources = evaluate(kept, idx)
    summary = summarize(rows)
    print_summary(summary)
    print_coverage(rows, all_sources)
    out_dir = ei.DATA_DIR / "results"
    write_csv(out_dir / "retrieval_summary.csv", summary)
    write_csv(out_dir / "retrieval_details.csv", rows)
    print(f"\nSaved to {out_dir}")
    return summary, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Recall@k smoke test for the three configurations.")
    ap.add_argument("--questions", default=str(Path(__file__).resolve().parent / "questions.csv"))
    ap.add_argument("--provider", choices=["openai", "local", "hash"], default=ei.EMBED_PROVIDER)
    args = ap.parse_args()
    run(args.questions, args.provider)