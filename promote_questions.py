"""
Turn questions asked in the app into candidate test or training questions, for a person to verify.

Run (from the project folder):
    python promote_questions.py

It reads data/app_questions.csv (written by app.py) and writes data/questions_from_app.csv in the
same columns as questions.csv. It keeps questions a user marked correct (👍) or corrected (👎 with an
answer typed in), and skips NOT FOUND answers and duplicates.

Every row is written with verified = n. retrieval_check.py, rerank_data.py and generate.py skip
unverified rows, so nothing reaches an experiment until a person has:
  1. opened the filing and checked the answer,
  2. filled in row_label (the table row the number is on) and, if known, statement and pdf_page,
  3. set verified = y and verified_by.
Only then copy the row into questions.csv (test set) or into a training question file. Questions
from the app were never seen by the trained rerankers, so they are fair test questions once verified.
"""
import csv
from pathlib import Path

import embed_index as ei

LOG = ei.DATA_DIR / "app_questions.csv"
OUT = ei.DATA_DIR / "questions_from_app.csv"
FIELDS = ["id", "filing", "question", "answer", "row_label", "label", "statement", "pdf_page", "verified", "verified_by",
          "source_log_id", "app_answer", "feedback"]


def main():
    if not LOG.exists():
        raise SystemExit(f"{LOG} not found. Ask some questions in the app first.")
    rows = list(csv.DictReader(open(LOG, newline="", encoding="utf-8")))
    out, seen = [], set()
    for r in rows:
        corrected = r.get("correct_answer", "").strip()
        if r["feedback"] == "up" and r["not_found"] != "1":
            answer = r["answer"].strip()
        elif r["feedback"] == "down" and corrected:
            answer = corrected
        else:
            continue
        key = r["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"a{len(out) + 1:03d}", "filing": r["top_filing"], "question": r["question"].strip(),
                    "answer": answer, "row_label": "", "label": "table", "statement": "", "pdf_page": "",
                    "verified": "n", "verified_by": "", "source_log_id": r["log_id"], "app_answer": r["answer"],
                    "feedback": r["feedback"]})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    n_nf = sum(r["not_found"] == "1" for r in rows)
    print(f"{len(rows)} logged questions: {len(out)} candidates written to {OUT}, {n_nf} were NOT FOUND (system gaps).")
    print("Check each candidate against its filing, fill in row_label, and set verified = y before using it.")


if __name__ == "__main__":
    main()