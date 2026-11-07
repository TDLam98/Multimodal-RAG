"""
Phase 4, step 1: answer each question with an LLM, from retrieved evidence.

Run (from the project folder, after rerank_data.py and train_reranker.py):
    python generate.py --llm groq          # free tier, needs GROQ_API_KEY
    python generate.py --llm gemini        # free tier, needs GEMINI_API_KEY
    python generate.py --llm ollama        # free, runs on your computer (install Ollama first)
    python generate.py --llm mock          # offline plumbing test, no LLM at all

    Other options:
      --k 5                 evidence chunks given to the LLM (default 5)
      --configs html_only,html+naive_pdf   which retrieval configurations (default: all with data)
      --methods embedding,logreg           retrieval order: raw embedding rank, or reranked (default both)
      --model NAME          override the provider's default model. For Gemini, check which Flash models
                            your free tier offers (AI Studio lists them) and pass one, e.g.
                            --model gemini-2.5-flash
      --limit 5             only the first 5 questions (quick check)

How the evidence is chosen
  It reads data/rerank/pairs.csv, which already holds each question's top-20 candidates for every
  configuration. "embedding" takes the top k by embedding rank; "logreg" (or the gradient-boosting
  model) takes the top k by the reranker's HELD-OUT score from data/rerank/oof_scores.csv, so no
  question is reranked by a model that saw its company. This keeps Phase 4 consistent with Phase 3.

Each evidence chunk is shown with its source line (filing, source type, section, table title).

Caching: every LLM reply is cached in data/generation/cache.jsonl, keyed by the exact prompt and
model. Re-running never pays for (or waits for) the same call twice, and an interrupted run resumes.

Output: data/generation/answers.csv   one row per question x configuration x method
Then run evaluate_answers.py to score the answers.

API keys: set them in the terminal, never in code. PowerShell:  $env:GROQ_API_KEY = "gsk_..."
"""
import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

import embed_index as ei
import retrieval_check as rc

PROJECT = Path(__file__).resolve().parent
PAIRS = ei.DATA_DIR / "rerank" / "pairs.csv"
OOF = ei.DATA_DIR / "rerank" / "oof_scores.csv"
OUT_DIR = ei.DATA_DIR / "generation"

# Provider presets: (base URL, default model, environment variable holding the key)
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.1-8b-instant", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash", "GEMINI_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", "llama3.1:8b", None),
    "mock": (None, "mock", None),
}
DEFAULT_RPM = {"groq": 25, "gemini": 8, "openai": 60, "ollama": 1000, "mock": 100000}
# Gemini models "think" before answering, and on Gemini 3 models thinking cannot be switched off, so a
# small max_tokens could be used up by thinking and cut the answer off. For Gemini: low thinking effort
# and no max_tokens cap (the prompt already asks for just the value).
EXTRA_BODY = {"gemini": {"reasoning_effort": "low"}}
MAX_TOKENS = {"gemini": None}          # provider -> cap; providers not listed use 60

SYSTEM = (
    "You answer questions about companies' SEC Form 10-K filings using ONLY the numbered excerpts "
    "provided. Financial statement figures are usually printed in millions; give the figure exactly "
    "as printed in the excerpt (for example 359,241), without converting units. "
    "Reply with only the answer value, no explanation. If the excerpts do not contain the answer, "
    "reply exactly: NOT FOUND"
)
MAX_CHARS_PER_CHUNK = 2500


# ---------- evidence ----------
def load_candidates():
    if not PAIRS.exists():
        raise SystemExit(f"{PAIRS} not found. Run rerank_data.py and train_reranker.py first.")
    with open(PAIRS, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    scores = {}
    if OOF.exists():
        with open(OOF, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                scores[(r["qid"], r["chunk_id"])] = r
    return pairs, scores


def load_chunks(provider):
    path = ei.DATA_DIR / "index" / ei.model_tag(provider) / "chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"No index at {path}. Check EMBED_PROVIDER in embed_index.py.")
    with open(path, encoding="utf-8") as f:
        return {c["chunk_id"]: c for c in map(json.loads, f)}


def pick_evidence(pairs, scores, qid, config, method, k):
    """Top k candidates for one question and configuration, in the chosen order."""
    col = f"rank_{config}"
    cands = [p for p in pairs if p["qid"] == qid and p.get(col)]
    if method == "embedding":
        cands.sort(key=lambda p: float(p[col]))
    else:
        key = f"score_{method}"

        def score(p):
            s = scores.get((qid, p["chunk_id"]), {}).get(key, "")
            return float(s) if s not in ("", "nan") else float("-inf")
        cands.sort(key=lambda p: (-score(p), float(p[col])))
    return cands[:k]


def format_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        bits = [c["filing"], c["source"], c.get("section", "")]
        if c.get("page"):
            bits.append(f"page {c['page']}")
        if c.get("title"):
            bits.append(c["title"][:120])
        header = " | ".join(b for b in bits if b)
        parts.append(f"[{i}] {header}\n{c['text'][:MAX_CHARS_PER_CHUNK]}")
    return "\n\n".join(parts)


# ---------- LLM calls ----------
class LLM:
    def __init__(self, provider, model=None, rpm=None):
        if provider not in PROVIDERS:
            raise SystemExit(f"Unknown --llm {provider!r}; choose from {list(PROVIDERS)}")
        self.provider = provider
        self.base, default_model, key_env = PROVIDERS[provider]
        self.model = model or default_model
        self.key = os.environ.get(key_env) if key_env else None
        if key_env and not self.key:
            raise SystemExit(f"{key_env} is not set. In PowerShell: $env:{key_env} = \"your-key\" "
                             f"(or use --llm ollama / --llm mock).")
        self.min_gap = 60.0 / (rpm or DEFAULT_RPM[provider])
        self.last = 0.0
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = OUT_DIR / "cache.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        self.cache[e["key"]] = e["reply"]

    def _key(self, prompt):
        return hashlib.sha1(f"{self.provider}|{self.model}|{SYSTEM}|{prompt}".encode()).hexdigest()

    def ask(self, prompt):
        """Returns (reply, from_cache)."""
        key = self._key(prompt)
        if key in self.cache:
            return self.cache[key], True
        reply = self._mock(prompt) if self.provider == "mock" else self._call(prompt)
        self.cache[key] = reply
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": self.model, "reply": reply}) + "\n")
        return reply, False

    def list_models(self):
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        r = requests.get(f"{self.base}/models", headers=headers, timeout=60)
        if r.status_code >= 400:
            raise SystemExit(f"Could not list models (HTTP {r.status_code}): {r.text[:500]}")
        return sorted(m["id"].removeprefix("models/") for m in r.json().get("data", []))

    def _call(self, prompt):
        wait = self.min_gap - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                **EXTRA_BODY.get(self.provider, {})}
        cap = MAX_TOKENS.get(self.provider, 60)
        if cap:
            body["max_tokens"] = cap
        for attempt in range(6):
            self.last = time.time()
            try:
                r = requests.post(f"{self.base}/chat/completions", headers=headers, json=body, timeout=120)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:          # bad model name, bad key, bad request: retrying won't help
                    hint = (f"The model '{self.model}' was not found. Run  python generate.py --llm {self.provider} "
                            f"--list-models  and pass one of those names with --model."
                            if r.status_code == 404 else
                            f"Check that your API key is correct (HTTP {r.status_code}).")
                    raise SystemExit(f"\n{hint}\nServer said: {r.text[:500]}")
                r.raise_for_status()
                return (r.json()["choices"][0]["message"]["content"] or "").strip()
            except Exception as e:
                if attempt == 5:
                    if "HTTP 429" in str(e):
                        raise SystemExit(
                            "\nThe provider keeps refusing with HTTP 429 (rate limit). On a free tier this usually "
                            "means the DAILY request limit is used up. Every answer so far is cached, so just run "
                            "the same command again later (tomorrow, for a daily limit) and it resumes.")
                    raise
                backoff = min(60, 2 ** (attempt + 1))
                print(f"    LLM call failed ({e}); retrying in {backoff}s")
                time.sleep(backoff)

    @staticmethod
    def _mock(prompt):
        """Offline stand-in: answers with the first number in the first evidence chunk.
        Only for testing the plumbing; its accuracy means nothing."""
        evidence = prompt.split("Excerpts:", 1)[-1]
        m = re.search(r"\(?\d[\d,]*\.?\d*\)?", evidence.split("\n", 2)[-1])
        return m.group(0) if m else "NOT FOUND"


def build_prompt(question, context):
    return f"Question: {question}\n\nExcerpts:\n{context}\n\nAnswer:"


# ---------- main ----------
def run(llm_name, k=5, configs=None, methods=None, model=None, limit=None, questions_path=None, rpm=None):
    questions = rc.load_questions(questions_path or PROJECT / "questions.csv")
    pairs, scores = load_candidates()
    chunks = load_chunks(ei.EMBED_PROVIDER)
    pair_qids = {p["qid"] for p in pairs}
    questions = [q for q in questions if q["id"] in pair_qids]
    if limit:
        questions = questions[:limit]
    if not questions:
        raise SystemExit("None of the questions appear in pairs.csv. Re-run rerank_data.py with this questions file.")

    present = {c["source"] for c in chunks.values()}
    if configs:
        configs = [c.strip() for c in configs.split(",")]
    else:
        configs = [c for c, srcs in ei.CONFIGS.items()
                   if all(s in present or s.startswith("html") for s in srcs)]
    available = {c[len("score_"):] for r in scores.values() for c in r if c.startswith("score_")}
    if methods:
        methods = [m.strip() for m in methods.split(",")]
    else:
        methods = ["embedding"] + (["logreg"] if "logreg" in available else [])
    missing = [m for m in methods if m != "embedding" and m not in available]
    if missing:
        raise SystemExit(f"No held-out scores for {missing}. Run train_reranker.py first.")

    llm = LLM(llm_name, model, rpm)
    total = len(questions) * len(configs) * len(methods)
    print(f"{len(questions)} questions x {len(configs)} configurations x {len(methods)} methods = {total} answers "
          f"(model: {llm.model}, k={k}). Configurations: {configs}")

    rows, n, calls = [], 0, 0
    t0 = time.time()
    for q in questions:
        for cfg in configs:
            for method in methods:
                n += 1
                picked = pick_evidence(pairs, scores, q["id"], cfg, method, k)
                ev = [chunks[p["chunk_id"]] for p in picked if p["chunk_id"] in chunks]
                reply, cached = llm.ask(build_prompt(q["question"], format_context(ev)))
                calls += not cached
                rows.append({
                    "qid": q["id"], "filing": q["filing"], "label": q.get("label", ""), "config": cfg,
                    "method": method, "k": k, "model": llm.model, "prediction": reply,
                    "gold": q["answer"], "evidence_hit": int(any(p["label"] == "1" for p in picked)),
                    "evidence_ids": " ".join(p["chunk_id"] for p in picked),
                })
                if n % 20 == 0 or n == total:
                    print(f"  {n}/{total} done ({calls} new LLM calls, {time.time() - t0:.0f}s)")

    empty = [r for r in rows if not r["prediction"].strip()]
    if empty:
        print(f"\nWARNING: {len(empty)} empty replies (e.g. {empty[0]['qid']}). The model may have run out of "
              "output tokens while thinking; try another --model.")
    out = OUT_DIR / "answers.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} answers to {out}. Next: python evaluate_answers.py")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Answer questions with an LLM from retrieved evidence.")
    ap.add_argument("--llm", required=True, choices=list(PROVIDERS))
    ap.add_argument("--model")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--configs")
    ap.add_argument("--methods")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--questions")
    ap.add_argument("--list-models", action="store_true", help="print the model names this provider offers, then stop")
    ap.add_argument("--rpm", type=float, help="max requests per minute (default depends on provider)")
    a = ap.parse_args()
    if a.list_models:
        print("\n".join(LLM(a.llm, a.model).list_models()))
        raise SystemExit
    run(a.llm, a.k, a.configs, a.methods, a.model, a.limit, a.questions, a.rpm)
