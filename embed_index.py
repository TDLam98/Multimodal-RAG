"""
Embed every chunk and build a searchable index.

Run (from the project folder):
    python embed_index.py                    # embed all chunk files, then build the index
    python embed_index.py --limit 5          # pilot: only the first 5 filings
    python embed_index.py --provider local   # free, runs on your own CPU (no API key)
    python embed_index.py --force            # re-embed everything, ignoring the cache
    python embed_index.py --index-only       # rebuild the index from already-cached embeddings

Providers
    openai   text-embedding-3-small through the OpenAI API. Needs OPENAI_API_KEY in your
             environment.  pip install openai
    local    BAAI/bge-small-en-v1.5 on your own CPU. Free but slower.
             pip install sentence-transformers
    hash     An offline stand-in used only to TEST the pipeline. Its results are not meaningful.

What it does
  1. Reads data/chunks/{html,pdf_naive,pdf_layout}/*.jsonl
  2. Embeds each file's chunks and caches the vectors in data/embeddings/<model>/. Re-running
     skips files that are already embedded, and re-embeds any file whose chunks changed.
  3. Joins everything into data/index/<model>/  (vectors.npy, chunks.jsonl, info.json)

Searching is exact (one matrix product). That is fast enough for a few hundred thousand chunks
and makes filtering by source, which is how the three retrieval configurations work, trivial.

Use it from other code:
    from embed_index import Index, CONFIGS
    idx = Index("openai")
    idx.search("What were Apple's total assets in fiscal 2025?", k=5, sources=CONFIGS["html_only"])

Requires: pip install numpy
"""
import argparse
import hashlib
import json
import os
import re
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from ingest import DATA_DIR

# ------------------------- CONFIG -------------------------
EMBED_PROVIDER = "openai"                  # "openai", "local" or "hash"
OPENAI_MODEL = "text-embedding-3-small"
LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HASH_DIM = 512
BATCH_SIZE = 128
MAX_CHARS = 6000                           # longer chunks are truncated before embedding
CHUNK_FOLDERS = ["html", "pdf_naive", "pdf_layout"]    # folders under data/chunks/
USD_PER_MILLION_TOKENS = 0.02              # list price for text-embedding-3-small; check current pricing
# ----------------------------------------------------------

# The three retrieval configurations from the proposal: each is just a set of sources to search.
CONFIGS = {
    "html_only": ["html_text", "html_table"],
    "html+naive_pdf": ["html_text", "html_table", "pdf_naive"],
    "html+layout_pdf": ["html_text", "html_table", "pdf_layout"],
}

_OPENAI_CLIENT = None
_LOCAL_MODEL = None


# ---------- embedding providers ----------
def model_tag(provider):
    if provider == "openai":
        return f"openai_{OPENAI_MODEL}"
    if provider == "local":
        return "local_" + LOCAL_MODEL.replace("/", "-")
    if provider == "hash":
        return f"hash_{HASH_DIM}"
    raise ValueError(f"Unknown provider {provider!r}")


def _normalize(m):
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype("float32")


def _hash_embed(texts):
    m = np.zeros((len(texts), HASH_DIM), dtype="float32")
    for r, t in enumerate(texts):
        for tok in re.findall(r"[a-z0-9]+", t.lower()):
            m[r, zlib.crc32(tok.encode()) % HASH_DIM] += 1.0
    return m


def _openai_embed(texts):
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI              # lazy import: only needed for this provider
        _OPENAI_CLIENT = OpenAI()              # reads OPENAI_API_KEY from the environment
    for attempt in range(6):
        try:
            resp = _OPENAI_CLIENT.embeddings.create(model=OPENAI_MODEL, input=texts)
            return np.array([d.embedding for d in resp.data], dtype="float32")
        except Exception as e:
            wait = 2 ** attempt
            print(f"    embedding call failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("OpenAI embedding failed after several retries")


def _local_embed(texts, is_query):
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer     # lazy import
        _LOCAL_MODEL = SentenceTransformer(LOCAL_MODEL)
    prefix = BGE_QUERY_PREFIX if is_query else ""
    return np.asarray(_LOCAL_MODEL.encode([prefix + t for t in texts], batch_size=32,
                                          normalize_embeddings=True, show_progress_bar=False),
                      dtype="float32")


def embed_texts(texts, provider, is_query=False):
    """Embed a list of strings; returns a float32 matrix with L2-normalized rows."""
    texts = [(t[:MAX_CHARS] or " ") for t in texts]
    parts = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        if provider == "openai":
            parts.append(_openai_embed(batch))
        elif provider == "local":
            parts.append(_local_embed(batch, is_query))
        elif provider == "hash":
            parts.append(_hash_embed(batch))
        else:
            raise ValueError(f"Unknown provider {provider!r}")
    return _normalize(np.vstack(parts)) if parts else np.zeros((0, 1), dtype="float32")


# ---------- reading chunk files ----------
def read_chunks(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def list_jobs(limit=None):
    """(folder, stem, path) for every chunk file. --limit keeps only the first N filings."""
    stems = set()
    for folder in CHUNK_FOLDERS:
        d = DATA_DIR / "chunks" / folder
        if d.exists():
            stems |= {p.stem for p in d.glob("*.jsonl")}
    stems = sorted(stems)
    if limit:
        stems = stems[:limit]
    jobs = []
    for stem in stems:
        for folder in CHUNK_FOLDERS:
            p = DATA_DIR / "chunks" / folder / f"{stem}.jsonl"
            if p.exists():
                jobs.append((folder, stem, p))
    return jobs


def _sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def _paths(tag, folder, stem):
    out = DATA_DIR / "embeddings" / tag / folder / f"{stem}.npy"
    return out, out.with_suffix(".json")


def _is_current(chunk_path, meta_path):
    if not meta_path.exists():
        return False
    return json.loads(meta_path.read_text())["sha1"] == _sha1(chunk_path)


# ---------- step 1: embed (cached per file) ----------
def embed_all(provider, limit=None, force=False):
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Set it in your terminal, "
                           "or run with --provider local (free).")
    tag = model_tag(provider)
    jobs = list_jobs(limit)
    if not jobs:
        raise FileNotFoundError(f"No chunk files under {DATA_DIR / 'chunks'}. "
                                "Run parse_html.py and parse_pdf.py first.")

    todo = []
    for folder, stem, path in jobs:
        out, meta = _paths(tag, folder, stem)
        if not force and out.exists() and _is_current(path, meta):
            continue
        todo.append((folder, stem, path))
    print(f"{len(jobs)} chunk files; {len(jobs) - len(todo)} already embedded, {len(todo)} to do.")
    if not todo:
        return 0

    approx_tokens = sum(p.stat().st_size for _, _, p in todo) / 4
    msg = f"About {approx_tokens / 1e6:.1f}M tokens to embed."
    if provider == "openai":
        msg += f" Roughly ${approx_tokens / 1e6 * USD_PER_MILLION_TOKENS:.2f} at the list price."
    print(msg)

    for n, (folder, stem, path) in enumerate(todo, 1):
        t0 = time.time()
        chunks = read_chunks(path)
        vecs = embed_texts([c["text"] for c in chunks], provider)
        out, meta = _paths(tag, folder, stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, vecs)
        tmp.replace(out)                                    # never leaves a half-written file
        meta.write_text(json.dumps({"sha1": _sha1(path), "rows": len(chunks)}))
        print(f"[{n}/{len(todo)}] {folder}/{stem}: {len(chunks)} chunks ({time.time() - t0:.0f}s)")
    return len(todo)


# ---------- step 2: join everything into one index ----------
def build_index(provider, limit=None):
    tag = model_tag(provider)
    vectors, chunks = [], []
    for folder, stem, path in list_jobs(limit):
        out, meta = _paths(tag, folder, stem)
        if not out.exists():
            raise FileNotFoundError(f"{folder}/{stem} has not been embedded yet; "
                                    "run without --index-only.")
        if not _is_current(path, meta):
            raise ValueError(f"{folder}/{stem}: chunks changed since embedding; "
                             "run again without --index-only.")
        arr, rows = np.load(out), read_chunks(path)
        if len(rows) != arr.shape[0]:
            raise ValueError(f"{path.name}: {len(rows)} chunks but {arr.shape[0]} vectors; "
                             "re-run with --force.")
        if rows:
            vectors.append(arr)
            chunks.extend(rows)
    if not vectors:
        raise ValueError("No chunks to index.")

    matrix = np.vstack(vectors)
    index_dir = DATA_DIR / "index" / tag
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", matrix)
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    per_source = {}
    for source in sorted({c["source"] for c in chunks}):
        rows = [c for c in chunks if c["source"] == source]
        per_source[source] = {"chunks": len(rows),
                              "filings": len({c["filing"] for c in rows}),
                              "avg_chars": round(sum(len(c["text"]) for c in rows) / len(rows))}
    info = {"provider": provider, "model": tag, "dim": int(matrix.shape[1]),
            "n_chunks": len(chunks), "per_source": per_source}
    (index_dir / "info.json").write_text(json.dumps(info, indent=2))

    print(f"\nIndex written to {index_dir}  ({len(chunks)} chunks, {matrix.shape[1]} dimensions, "
          f"{matrix.nbytes / 1e6:.0f} MB of vectors)")
    print(f"{'source':<12}{'chunks':>9}{'filings':>9}{'avg chars':>11}")
    for s, v in per_source.items():
        print(f"{s:<12}{v['chunks']:>9}{v['filings']:>9}{v['avg_chars']:>11}")
    return info


# ---------- searching ----------
class Index:
    """Loads the index and searches it, optionally restricted to some sources."""

    def __init__(self, provider=EMBED_PROVIDER, data_dir=None):
        self.provider = provider
        d = Path(data_dir or DATA_DIR) / "index" / model_tag(provider)
        if not (d / "vectors.npy").exists():
            raise FileNotFoundError(f"No index at {d}. Run embed_index.py first.")
        self.vectors = np.load(d / "vectors.npy")
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.sources = np.array([c["source"] for c in self.chunks])

    def search_vec(self, qvec, k=10, sources=None):
        """Top-k (score, chunk) pairs for an already-embedded query."""
        scores = self.vectors @ qvec
        if sources is not None:
            scores = np.where(np.isin(self.sources, list(sources)), scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self.chunks[i]) for i in top if np.isfinite(scores[i])]

    def search(self, query, k=10, sources=None):
        qvec = embed_texts([query], self.provider, is_query=True)[0]
        return self.search_vec(qvec, k=k, sources=sources)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Embed chunks and build the search index.")
    ap.add_argument("--provider", choices=["openai", "local", "hash"], default=EMBED_PROVIDER)
    ap.add_argument("--limit", type=int, help="only the first N filings")
    ap.add_argument("--force", action="store_true", help="re-embed everything")
    ap.add_argument("--index-only", action="store_true",
                    help="skip embedding; rebuild the index from cached embeddings")
    args = ap.parse_args()
    if not args.index_only:
        embed_all(args.provider, limit=args.limit, force=args.force)
    build_index(args.provider, limit=args.limit)