"""
Phase 4, step 2: score the generated answers.

Run (from the project folder, after generate.py):
    python evaluate_answers.py
    python evaluate_answers.py --tol 0.01      # relative tolerance for numbers (default 1%)

Scoring
  correct       the prediction contains a number within the tolerance of the gold answer. Units are
                handled: gold table answers are in millions as printed, so "$416.2 billion" matches
                416,161 (millions). Text answers (employee counts) are plain counts, so
                "166 thousand" matches 166,000. The sign is ignored: "12,715" matches "(12,715)".
  exact         the prediction, stripped of $ signs, commas, spaces and a trailing period, equals the
                gold answer stripped the same way. A stricter check, reported alongside.
  not_found     the model said NOT FOUND (it abstained instead of guessing).
  evidence_hit  the right chunk was among the k chunks the model was given.

The most useful columns:
  accuracy                  share of all questions answered correctly
  accuracy_given_evidence   accuracy on just the questions whose evidence contained the answer,
                            which measures the LLM's reading, separately from retrieval
  wrong_with_evidence       answered wrongly although the evidence contained the answer (a reading error)

Outputs: data/results/answer_summary.csv and data/generation/answers_scored.csv
"""
import argparse
import csv
import re
from pathlib import Path

import embed_index as ei

ANSWERS = ei.DATA_DIR / "generation" / "answers.csv"
RESULTS = ei.DATA_DIR / "results"
SCALE = {"trillion": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mm": 1e6, "m": 1e6,
         "thousand": 1e3, "k": 1e3}
NUM_RE = re.compile(r"\(?-?\$?\s*(\d[\d,]*(?:\.\d+)?)\s*\)?\s*(trillion|billion|bn|million|mm|thousand|[bmk])?\b",
                    re.IGNORECASE)


def to_float(s):
    s = s.replace(",", "").replace("$", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    try:
        v = float(s.strip("()-"))
    except ValueError:
        return None
    return -v if (neg or s.startswith("-")) else v


def numbers_in(text):
    """(value as written, scale multiplier or None) for every number in the text."""
    out = []
    for m in NUM_RE.finditer(text):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        scale = SCALE.get((m.group(2) or "").lower())
        out.append((v, scale))
    return out


def close(a, b, tol):
    return abs(abs(a) - abs(b)) <= tol * max(abs(b), 1e-9)


def is_correct(pred, gold, label, tol):
    g = to_float(gold)
    if g is None:
        return norm(pred) == norm(gold)
    base = 1e6 if label == "table" else 1.0       # table answers are printed in millions
    for v, scale in numbers_in(pred):
        if close(v, g, tol):                       # same units as printed
            return True
        if scale and close(v * scale, g * base, tol):   # e.g. "$416.2 billion"
            return True
    return False


def norm(s):
    return re.sub(r"[\s$,]", "", (s or "").lower()).rstrip(".")


def score(rows, tol):
    for r in rows:
        pred = r["prediction"] or ""
        r["not_found"] = int("not found" in pred.lower())
        r["correct"] = int(not r["not_found"] and is_correct(pred, r["gold"], r["label"], tol))
        r["exact"] = int(norm(pred) == norm(r["gold"]))
        r["evidence_hit"] = int(r["evidence_hit"])
    return rows


def summarize(rows):
    out = []
    groups = ["all"] + sorted({r["label"] for r in rows})
    keys = list(dict.fromkeys((r["config"], r["method"]) for r in rows))
    for g in groups:
        for cfg, method in keys:
            sel = [r for r in rows if r["config"] == cfg and r["method"] == method and (g == "all" or r["label"] == g)]
            if not sel:
                continue
            n = len(sel)
            hit = [r for r in sel if r["evidence_hit"]]
            out.append({
                "group": g, "config": cfg, "method": method, "n": n,
                "accuracy": sum(r["correct"] for r in sel) / n,
                "exact": sum(r["exact"] for r in sel) / n,
                "evidence_hit": len(hit) / n,
                "accuracy_given_evidence": (sum(r["correct"] for r in hit) / len(hit)) if hit else float("nan"),
                "not_found": sum(r["not_found"] for r in sel) / n,
                "wrong_with_evidence": sum(1 for r in hit if not r["correct"]),
            })
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(tol=0.01):
    if not ANSWERS.exists():
        raise SystemExit(f"{ANSWERS} not found. Run generate.py first.")
    with open(ANSWERS, newline="", encoding="utf-8") as f:
        rows = score(list(csv.DictReader(f)), tol)
    summary = summarize(rows)
    write_csv(ei.DATA_DIR / "generation" / "answers_scored.csv", rows)
    write_csv(RESULTS / "answer_summary.csv", summary)

    model = rows[0].get("model", "")
    print(f"Scored {len(rows)} answers (model: {model}, tolerance {tol:.1%}).")
    cols = ["accuracy", "exact", "evidence_hit", "accuracy_given_evidence", "not_found"]
    for g in dict.fromkeys(s["group"] for s in summary):
        sel = [s for s in summary if s["group"] == g]
        print(f"\n=== {g} (n={sel[0]['n']}) ===")
        print(f"{'config':<18}{'method':<11}" + "".join(f"{c[:14]:>15}" for c in cols))
        for s in sel:
            vals = "".join(f"{s[c]:>15.2f}" for c in cols)
            print(f"{s['config']:<18}{s['method']:<11}{vals}")
    wrong = [r for r in rows if r["evidence_hit"] and not r["correct"]]
    if wrong:
        print(f"\n{len(wrong)} answers were wrong although the evidence contained the answer. Examples:")
        for r in wrong[:5]:
            print(f"  {r['qid']} [{r['config']}/{r['method']}] gold={r['gold']!r} predicted={r['prediction'][:80]!r}")
        print("  (see data/generation/answers_scored.csv for all of them)")
    print(f"\nSaved to {RESULTS / 'answer_summary.csv'}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score generated answers against questions.csv.")
    ap.add_argument("--tol", type=float, default=0.01, help="relative tolerance for numbers (default 0.01)")
    main(ap.parse_args().tol)