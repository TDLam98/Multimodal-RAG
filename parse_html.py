"""
Parse downloaded 10-K HTML into retrievable chunks: narrative text and tables.

Run:
    python parse_html.py                  # every filing listed in data/manifest.csv
    python parse_html.py --force          # redo files that already have output
    python parse_html.py --file data/html/AAPL_2025-09-27.html
                                          # one file: prints a quick report so you can
                                          # check it against the real balance sheet

Output: data/chunks/html/<TICKER>_<report_date>.jsonl  (one JSON chunk per line)

Each chunk has: chunk_id, filing, ticker, report_date, source ("html_text" or
"html_table"), section (e.g. "Item 7"), text, and for tables a "title".

Requires: pip install beautifulsoup4 lxml
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

from ingest import DATA_DIR
from serialize import clean_cell, clean_row, serialize_table

MAX_TEXT_CHARS = 1500    # target size of a text chunk
MIN_TEXT_CHARS = 80      # drop text chunks shorter than this (stray headers/footers)
MAX_TABLE_ROWS = 40      # split very long tables into groups of this many rows
MIN_NUMERIC_CELLS = 3    # a table needs at least this many numeric cells to be kept

BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6"]
WALK_TAGS = ["p", "div", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"]
ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[abc]?)\b", re.IGNORECASE)


# ---------- loading ----------
def load_soup(html_path):
    soup = BeautifulSoup(Path(html_path).read_bytes(), "lxml")
    hidden = list(soup(["script", "style"]))
    hidden += soup.find_all("ix:header")  # inline-XBRL metadata block, not visible text
    hidden += soup.find_all(style=re.compile(r"display\s*:\s*none", re.IGNORECASE))
    for tag in hidden:
        if not getattr(tag, "decomposed", False):
            tag.decompose()
    return soup


# ---------- tables ----------
def extract_rows(table):
    rows = []
    for tr in table.find_all("tr"):
        cells = [clean_cell(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
        row = clean_row(cells)
        if row:
            rows.append(row)
    return rows


def is_data_table(rows):
    """Keep tables that hold numbers. Skip layout tables and tables of contents."""
    if len(rows) < 2:
        return False
    if sum(1 for row in rows if ITEM_RE.match(row[0])) >= 3:  # table of contents
        return False
    numeric = sum(1 for row in rows for c in row[1:] if re.search(r"\d", c))
    return numeric >= MIN_NUMERIC_CELLS


def table_context(table, max_chars=200):
    """The two nearest text snippets before the table, usually its heading
    (e.g. 'CONSOLIDATED BALANCE SHEETS / (in millions)')."""
    bits = []
    for s in table.find_all_previous(string=True, limit=60):
        t = clean_cell(s)
        if len(t) >= 4:
            bits.append(t)
        if len(bits) == 2:
            break
    return " / ".join(reversed(bits))[:max_chars]


def group_rows(rows):
    for i in range(0, len(rows), MAX_TABLE_ROWS):
        yield rows[i:i + MAX_TABLE_ROWS]


# ---------- text ----------
def split_long(text, limit=MAX_TEXT_CHARS):
    """Split text longer than limit at sentence boundaries."""
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + len(sentence) + 1 > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


# ---------- main parser ----------
def parse_filing(html_path):
    """Return a list of chunk dicts for one filing, in document order."""
    stem = Path(html_path).stem                      # e.g. AAPL_2025-09-27
    ticker, _, report_date = stem.partition("_")
    soup = load_soup(html_path)

    chunks = []
    section = "Front matter"
    buffer = []                                      # paragraphs waiting to become one chunk

    def meta(source):
        return {"filing": stem, "ticker": ticker, "report_date": report_date,
                "source": source, "section": section}

    def flush_text():
        text = " ".join(buffer).strip()
        buffer.clear()
        for piece in split_long(text):
            if len(piece) >= MIN_TEXT_CHARS:
                chunks.append({**meta("html_text"), "text": piece})

    for el in soup.find_all(WALK_TAGS):              # document order
        if el.name == "table":
            if el.find_parent("table"):              # nested table: the outer one covers it
                continue
            flush_text()
            rows = extract_rows(el)
            if is_data_table(rows):
                title = table_context(el)
                for group in group_rows(rows):
                    chunks.append({**meta("html_table"), "title": title,
                                   "text": serialize_table(group, title)})
            continue

        if el.find_parent("table") or el.find(BLOCK_TAGS):
            continue                                  # inside a table, or not a leaf block
        text = clean_cell(el.get_text(" ", strip=True))
        if not text:
            continue

        m = ITEM_RE.match(text)
        if m and len(text) < 150:                     # an "Item 7. ..." heading
            flush_text()
            section = f"Item {m.group(1).upper()}"
        if buffer and len(" ".join(buffer)) + len(text) > MAX_TEXT_CHARS:
            flush_text()
        buffer.append(text)

    flush_text()
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{stem}:{i:05d}"
    return chunks


def write_jsonl(path, chunks):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.replace(path)                                 # never leaves a half-written file


# ---------- command-line entry points ----------
def report(html_path):
    """Parse one file and print a quick report for eyeballing."""
    chunks = parse_filing(html_path)
    print(f"{html_path}: {len(chunks)} chunks")
    print("By source :", dict(Counter(c["source"] for c in chunks)))
    print("By section:", dict(Counter(c["section"] for c in chunks)))

    for c in chunks:
        if c["source"] == "html_table" and "total assets" in c["text"].lower():
            print("\n--- First table containing 'Total assets' (compare to the PDF) ---")
            print(f"[{c['section']}] title: {c['title']}")
            print("\n".join(c["text"].splitlines()[:30]))
            break
    else:
        print("\nNo table containing 'Total assets' was found.")

    first_text = next((c for c in chunks if c["source"] == "html_text"), None)
    if first_text:
        print("\n--- First text chunk ---")
        print(first_text["text"][:500])


def run_manifest(force=False):
    manifest = DATA_DIR / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found; run ingest.py first")
    out_dir = DATA_DIR / "chunks" / "html"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    failures, done, skipped = [], 0, 0
    for i, row in enumerate(rows, 1):
        # use only the file name, so the manifest still works if the project folder moved
        html_file = DATA_DIR / "html" / Path(str(row["html_path"]).replace("\\", "/")).name
        out_file = out_dir / (html_file.stem + ".jsonl")
        if out_file.exists() and not force:
            skipped += 1
            continue
        try:
            chunks = parse_filing(html_file)
            write_jsonl(out_file, chunks)
            done += 1
            print(f"[{i}/{len(rows)}] {html_file.name}: {len(chunks)} chunks")
        except Exception as e:
            failures.append({"file": html_file.name, "error": str(e)})
            print(f"[{i}/{len(rows)}] {html_file.name}: FAILED ({e})")

    with open(DATA_DIR / "parse_html_failures.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "error"])
        writer.writeheader()
        writer.writerows(failures)
    print(f"\nParsed {done}, skipped {skipped} already done, {len(failures)} failed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse 10-K HTML into text and table chunks.")
    ap.add_argument("--file", help="parse a single HTML file and print a report")
    ap.add_argument("--force", action="store_true", help="redo files that already have output")
    args = ap.parse_args()
    if args.file:
        report(args.file)
    else:
        run_manifest(force=args.force)