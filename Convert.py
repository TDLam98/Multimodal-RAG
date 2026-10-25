"""
Convert downloaded 10-K HTML filings into PDFs using headless Chromium.

Run:
    python convert.py            # convert everything listed in data/manifest.csv
    python convert.py --force    # redo every PDF, even ones that already exist

Behavior
- Reads data/manifest.csv (written by ingest.py), so only filings you actually
  downloaded in the latest run are converted, not stray leftovers in data/html/.
- Skips a PDF that already exists AND is newer than its HTML. If you re-download
  a filing, its PDF is automatically rebuilt.
- One bad filing never stops the batch: failures are logged to
  data/convert_failures.csv and the run continues.
- PDFs are written to a temporary name and renamed when finished, so a crash
  can't leave a half-written PDF that later runs would mistake for a good one.

Requires: pip install playwright && playwright install chromium
"""
import argparse
import csv
from pathlib import Path

from playwright.sync_api import sync_playwright

from ingest import DATA_DIR

PDF_OPTIONS = dict(
    format="Letter",
    print_background=True,
    margin={"top": "0.75in", "bottom": "0.75in", "left": "0.75in", "right": "0.75in"},
)


def convert_manifest(data_dir=DATA_DIR, force=False):
    data_dir = Path(data_dir)
    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found; run ingest.py first")
    pdf_dir = data_dir / "pdf"
    pdf_dir.mkdir(exist_ok=True)

    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    failures, done, skipped = [], 0, 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for i, row in enumerate(rows, 1):
            # use only the file name, so the manifest still works if the project folder moved
            name = Path(str(row["html_path"]).replace("\\", "/")).name
            html_file = data_dir / "html" / name
            pdf_file = pdf_dir / (html_file.stem + ".pdf")

            if not html_file.exists():
                failures.append({"file": name, "error": "HTML file missing"})
                print(f"[{i}/{len(rows)}] {name}: HTML missing")
                continue

            up_to_date = pdf_file.exists() and pdf_file.stat().st_mtime >= html_file.stat().st_mtime
            if up_to_date and not force:
                skipped += 1
                continue

            page = browser.new_page()      # fresh page per file isolates failures
            try:
                page.goto(html_file.resolve().as_uri(), wait_until="load", timeout=120_000)
                page.wait_for_timeout(1000)
                tmp = pdf_file.with_suffix(".pdf.tmp")
                page.pdf(path=str(tmp), **PDF_OPTIONS)
                tmp.replace(pdf_file)
                done += 1
                print(f"[{i}/{len(rows)}] {pdf_file.name}")
            except Exception as e:
                failures.append({"file": name, "error": str(e)})
                print(f"[{i}/{len(rows)}] {name}: FAILED ({e})")
            finally:
                page.close()
        browser.close()

    with open(data_dir / "convert_failures.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "error"])
        writer.writeheader()
        writer.writerows(failures)
    print(f"\nConverted {done}, skipped {skipped} already up to date, {len(failures)} failed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert manifest HTML filings to PDF.")
    ap.add_argument("--force", action="store_true", help="redo PDFs that already exist")
    args = ap.parse_args()
    convert_manifest(force=args.force)