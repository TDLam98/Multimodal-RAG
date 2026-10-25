"""
Parse each filing's PDF into retrievable chunks, two ways.

  pdf_naive    Plain reading-order text, page by page (pdfplumber). No table
               detection, so a table arrives as whatever text the PDF yields.
  pdf_layout   Layout-aware parse (Unstructured). Tables come out as structured
               tables and are serialized with the same serializer as the HTML
               tables, so the two sources are directly comparable.

Run (from the project folder):
    python parse_pdf.py --file data/pdf/AAPL_2025-09-27.pdf   # one PDF: prints a report
    python parse_pdf.py --limit 3                             # first 3 filings in the manifest
    python parse_pdf.py --only naive                          # fast: do this for everything first
    python parse_pdf.py --only layout                         # slow: run overnight
    python parse_pdf.py --force                               # redo outputs that already exist
    Other options: --scope table_pages|all    --strategy hi_res|fast

Output: data/chunks/pdf_naive/<filing>.jsonl and data/chunks/pdf_layout/<filing>.jsonl
Chunks use the same fields as parse_html.py, plus "page" (1-based).

PAGE SCOPE (important for speed and for a fair comparison)
Layout-aware parsing is slow, so by default BOTH methods only process the pages
that look like tables (pages dense with numbers). Using the same pages for both
keeps the comparison fair, and narrative text is already available from the
clean HTML. Use --scope all if you have the time to do every page. Each run
prints how many pages were selected, so you can tune MIN_NUMERIC_* below.

Requires: pip install pdfplumber pymupdf beautifulsoup4 lxml "unstructured[pdf]"
(Unstructured is only imported for the layout step, so --only naive works without it.)
"""
import argparse
import csv
import re
import tempfile
import time
from pathlib import Path

import pymupdf
import pdfplumber
from bs4 import BeautifulSoup

from ingest import DATA_DIR
from parse_html import (ITEM_RE, MAX_TEXT_CHARS, MIN_TEXT_CHARS, extract_rows,
                        group_rows, is_data_table, split_long, write_jsonl)
from serialize import clean_cell, serialize_table

# ------------------------- CONFIG -------------------------
PAGE_SCOPE = "table_pages"        # "table_pages" or "all"
UNSTRUCTURED_STRATEGY = "hi_res"  # falls back to "fast" if hi_res fails
MIN_NUMERIC_TOKENS = 25           # a "table page" needs at least this many numbers...
MIN_NUMERIC_RATIO = 0.12          # ...and at least this share of its words to be numbers
SKIP_CATEGORIES = {"Footer", "PageNumber", "Header"}   # Unstructured categories to ignore
# ----------------------------------------------------------

NUM_RE = re.compile(r"^\(?\$?\d[\d,]*\.?\d*\)?%?$")
OUT_DIRS = {
    "naive": DATA_DIR / "chunks" / "pdf_naive",
    "layout": DATA_DIR / "chunks" / "pdf_layout",
}


# ---------- step 1: scan every page quickly (PyMuPDF is fast) ----------
def scan_pages(pdf_path):
    """For each page: its section (from 'Item N' headings), and how number-dense it is."""
    pages, section = [], "Front matter"
    with pymupdf.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            tokens = text.split()
            numeric = sum(1 for t in tokens if NUM_RE.match(t))
            for line in text.splitlines():
                m = ITEM_RE.match(line)
                if m and len(line) < 150:
                    section = f"Item {m.group(1).upper()}"   # last heading on the page wins
            pages.append({"index": i, "section": section, "numeric": numeric,
                          "ratio": numeric / max(len(tokens), 1)})
    return pages


def is_table_page(p):
    return p["numeric"] >= MIN_NUMERIC_TOKENS and p["ratio"] >= MIN_NUMERIC_RATIO


def select_pages(pages, scope):
    if scope == "all":
        return [p["index"] for p in pages]
    return [p["index"] for p in pages if is_table_page(p)]


# ---------- shared helpers ----------
def base_meta(stem, source, section, page_number):
    ticker, _, report_date = stem.partition("_")
    return {"filing": stem, "ticker": ticker, "report_date": report_date,
            "source": source, "section": section, "page": page_number}


def number_chunks(chunks, stem):
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{stem}:{c['source']}:{i:05d}"
    return chunks


# ---------- step 2a: naive extraction ----------
def pack_lines(text, limit=MAX_TEXT_CHARS):
    """Pack lines into pieces of at most `limit` characters, keeping line breaks."""
    pieces, current = [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if current and len(current) + len(line) + 1 > limit:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)
    return pieces


def naive_chunks(pdf_path, stem, pages, selected):
    chunks = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i in selected:
            page = pdf.pages[i]
            text = page.extract_text() or ""
            getattr(page, "flush_cache", lambda: None)()      # keep memory flat
            for piece in pack_lines(text):
                if len(piece) >= MIN_TEXT_CHARS:
                    chunks.append({**base_meta(stem, "pdf_naive", pages[i]["section"], i + 1),
                                   "text": piece})
    return number_chunks(chunks, stem)


# ---------- step 2b: layout-aware extraction ----------
def run_unstructured(pdf_path, selected, strategy):
    """Parse only the selected pages with Unstructured.
    Returns (elements, strategy_used, page_map); page_map[k] is the original page
    index of page k+1 in the temporary subset PDF Unstructured actually saw."""
    from unstructured.partition.pdf import partition_pdf    # lazy: slow import

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        subset = Path(tmp) / "subset.pdf"
        with pymupdf.open(str(pdf_path)) as src:
            dst = pymupdf.open()
            for i in selected:
                dst.insert_pdf(src, from_page=i, to_page=i)
            dst.save(str(subset))
            dst.close()
        try:
            kwargs = {"infer_table_structure": True} if strategy == "hi_res" else {}
            elements = partition_pdf(filename=str(subset), strategy=strategy, **kwargs)
            used = strategy
        except Exception as e:
            print(f"    {strategy} failed ({e}); falling back to 'fast'")
            elements = partition_pdf(filename=str(subset), strategy="fast")
            used = "fast"
    return elements, used, list(selected)


def rows_from_html(html):
    table = BeautifulSoup(html, "lxml").find("table")
    return extract_rows(table) if table else []


def layout_chunks(elements, stem, pages, page_map):
    """Turn Unstructured elements into chunks (pure function, easy to test)."""
    chunks, buffer, buffer_page, last_title = [], [], None, ""

    def flush():
        nonlocal buffer, buffer_page
        if buffer:
            for piece in split_long(" ".join(buffer)):
                if len(piece) >= MIN_TEXT_CHARS:
                    chunks.append({**base_meta(stem, "pdf_layout",
                                               pages[buffer_page]["section"], buffer_page + 1),
                                   "text": piece})
        buffer, buffer_page = [], None

    for el in elements:
        cat = getattr(el, "category", "") or ""
        text = clean_cell(getattr(el, "text", "") or "")
        if cat in SKIP_CATEGORIES or not text:
            continue
        pn = getattr(getattr(el, "metadata", None), "page_number", None) or 1
        page_idx = page_map[pn - 1] if 1 <= pn <= len(page_map) else page_map[0]

        if buffer_page is not None and page_idx != buffer_page:
            flush()

        if cat == "Table":
            html = getattr(el.metadata, "text_as_html", None)
            rows = rows_from_html(html) if html else []
            if rows and is_data_table(rows):
                flush()
                for group in group_rows(rows):
                    chunks.append({**base_meta(stem, "pdf_layout", pages[page_idx]["section"],
                                               page_idx + 1),
                                   "title": last_title,
                                   "text": serialize_table(group, last_title)})
                continue
            # no usable table structure: fall through and keep it as plain text

        if cat == "Title":
            last_title = text[:200]            # nearest heading becomes the next table's title
        if buffer and len(" ".join(buffer)) + len(text) > MAX_TEXT_CHARS:
            flush()
        buffer.append(text)
        buffer_page = page_idx

    flush()
    return number_chunks(chunks, stem)


# ---------- orchestration ----------
def process_filing(pdf_path, needs, scope, strategy):
    """needs is a set containing 'naive' and/or 'layout'.
    Returns ({name: chunks}, info)."""
    stem = Path(pdf_path).stem
    pages = scan_pages(pdf_path)
    selected = select_pages(pages, scope)
    info = {"total_pages": len(pages), "selected": len(selected), "strategy_used": None,
            "selected_pages": [i + 1 for i in selected]}
    if not selected:
        return {name: [] for name in needs}, info

    results = {}
    if "naive" in needs:
        results["naive"] = naive_chunks(pdf_path, stem, pages, selected)
    if "layout" in needs:
        elements, used, page_map = run_unstructured(pdf_path, selected, strategy)
        info["strategy_used"] = used
        results["layout"] = layout_chunks(elements, stem, pages, page_map)
    return results, info


def show_total_assets(chunks, label):
    for c in chunks:
        lines = c["text"].splitlines()
        hit = next((k for k, line in enumerate(lines) if "total assets" in line.lower()), None)
        if hit is not None:
            print(f"\n--- {label}: chunk containing 'Total assets' "
                  f"(page {c['page']}, {c['section']}) ---")
            print("\n".join(lines[max(0, hit - 5): hit + 6]))
            return
    print(f"\n--- {label}: no chunk contains 'Total assets' ---")


def report(pdf_path, only, scope, strategy):
    """Parse one PDF and print a report; writes nothing."""
    needs = {only} if only else {"naive", "layout"}
    t0 = time.time()
    results, info = process_filing(pdf_path, needs, scope, strategy)
    print(f"{pdf_path}: {info['total_pages']} pages, {info['selected']} selected "
          f"(scope={scope}) in {time.time() - t0:.0f}s")
    print("Selected page numbers:", info["selected_pages"][:60],
          "..." if info["selected"] > 60 else "")
    for name, chunks in results.items():
        print(f"{name}: {len(chunks)} chunks "
              f"({sum(1 for c in chunks if 'title' in c)} table chunks)")
        show_total_assets(chunks, name)
    if info["strategy_used"]:
        print(f"\nUnstructured strategy used: {info['strategy_used']}")


def run_manifest(only, scope, strategy, force, limit):
    manifest = DATA_DIR / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found; run ingest.py first")
    for d in OUT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    names = [only] if only else ["naive", "layout"]

    failures, done, skipped = [], 0, 0
    for i, row in enumerate(rows, 1):
        stem = Path(str(row["html_path"]).replace("\\", "/")).stem
        pdf_file = DATA_DIR / "pdf" / f"{stem}.pdf"
        needs = {n for n in names if force or not (OUT_DIRS[n] / f"{stem}.jsonl").exists()}
        if not needs:
            skipped += 1
            continue
        if not pdf_file.exists():
            failures.append({"file": stem, "error": "PDF missing; run convert.py"})
            print(f"[{i}/{len(rows)}] {stem}: PDF missing")
            continue

        t0 = time.time()
        try:
            results, info = process_filing(pdf_file, needs, scope, strategy)
            for name, chunks in results.items():
                write_jsonl(OUT_DIRS[name] / f"{stem}.jsonl", chunks)
            if info["selected"] == 0:
                failures.append({"file": stem, "error": f"no pages selected (scope={scope})"})
            done += 1
            counts = ", ".join(f"{n}={len(c)}" for n, c in results.items())
            print(f"[{i}/{len(rows)}] {stem}: {info['selected']}/{info['total_pages']} pages, "
                  f"{counts} chunks ({time.time() - t0:.0f}s)")
        except Exception as e:
            failures.append({"file": stem, "error": str(e)})
            print(f"[{i}/{len(rows)}] {stem}: FAILED ({e})")

    with open(DATA_DIR / "parse_pdf_failures.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "error"])
        writer.writeheader()
        writer.writerows(failures)
    print(f"\nParsed {done}, skipped {skipped} already done, {len(failures)} problems "
          f"(see parse_pdf_failures.csv).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse filing PDFs into naive and layout-aware chunks.")
    ap.add_argument("--file", help="parse one PDF and print a report (writes nothing)")
    ap.add_argument("--only", choices=["naive", "layout"], help="run just one method")
    ap.add_argument("--scope", choices=["table_pages", "all"], default=PAGE_SCOPE)
    ap.add_argument("--strategy", choices=["hi_res", "fast"], default=UNSTRUCTURED_STRATEGY)
    ap.add_argument("--force", action="store_true", help="redo outputs that already exist")
    ap.add_argument("--limit", type=int, help="only the first N filings in the manifest")
    args = ap.parse_args()
    if args.file:
        report(args.file, args.only, args.scope, args.strategy)
    else:
        run_manifest(args.only, args.scope, args.strategy, args.force, args.limit)
