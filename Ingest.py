"""
EDGAR ingestion: download recent 10-K filings for the tickers in tickers.txt.

Run (from the project folder):
    python ingest.py               # every ticker in tickers.txt
    python ingest.py --limit 10    # only the first 10 tickers (good for a pilot run)

Outputs
    <DATA_DIR>/html/<TICKER>_<report_date>.html   one file per filing
    <DATA_DIR>/manifest.csv                        the filings covered by THIS run
    <DATA_DIR>/failures.csv                        anything that failed, and why

Safe to re-run: files that already exist are skipped, so a crash or a
rate-limit block never means starting over. Note that manifest.csv describes
the run that just finished, so a --limit run writes a short manifest; run
again without --limit to cover everything.

Other scripts import DATA_DIR from this file, so the data location is set here once.

Requires: pip install requests
"""
import argparse
import csv
import time
from pathlib import Path

import requests

# ------------------------- CONFIG: edit these -------------------------
USER_AGENT = "Tan-Dat Lam txl7064@mavs.uta.edu"      # SEC requires this on every request
FILINGS_PER_COMPANY = 2                              # ~130 tickers x 2 = ~260 filings
TICKERS_FILE = Path(__file__).resolve().parent / "tickers.txt"   # made by ticker_list.py
DATA_DIR = Path(__file__).resolve().parent / "data"              # inside the project folder
# ----------------------------------------------------------------------

HEADERS = {"User-Agent": USER_AGENT}


def sec_get(url):
    time.sleep(0.2)  # stay well under SEC's 10 requests/second limit
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r


def load_tickers(path=TICKERS_FILE):
    """Read tickers.txt: one ticker per line; blank lines and # comments are ignored."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run make_ticker_list.py first.")
    lines = path.read_text().splitlines()
    return [l.strip().upper() for l in lines if l.strip() and not l.startswith("#")]


def load_ticker_map():
    """Download SEC's ticker->CIK table once, instead of once per ticker."""
    data = sec_get("https://www.sec.gov/files/company_tickers.json").json()
    return {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()}


def list_10ks(cik, n):
    """Return up to n of the company's most recent 10-K filings (newest first)."""
    subs = sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    recent = subs["filings"]["recent"]
    filings = []
    rows = zip(recent["form"], recent["accessionNumber"],
               recent["primaryDocument"], recent["reportDate"])
    for form, acc, doc, report_date in rows:
        if form != "10-K":
            continue
        filings.append({
            "accession": acc,
            "report_date": report_date,
            "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}",
        })
        if len(filings) == n:
            break
    return filings


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_all(tickers=None, per_company=FILINGS_PER_COMPANY, data_dir=DATA_DIR, limit=None):
    if tickers is None:
        tickers = load_tickers()
    if limit:
        tickers = tickers[:limit]

    data_dir = Path(data_dir)
    html_dir = data_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    ticker_map = load_ticker_map()
    manifest, failures = [], []

    for ticker in tickers:
        cik = ticker_map.get(ticker.upper())
        if cik is None:
            failures.append({"ticker": ticker, "filing": "", "error": "ticker not found"})
            print(f"{ticker}: ticker not found")
            continue
        try:
            filings = list_10ks(cik, per_company)
        except Exception as e:
            failures.append({"ticker": ticker, "filing": "", "error": str(e)})
            print(f"{ticker}: could not list filings ({e})")
            continue
        if not filings:
            failures.append({"ticker": ticker, "filing": "", "error": "no 10-K in recent filings"})
            print(f"{ticker}: no 10-K found")
            continue

        for f in filings:
            path = html_dir / f"{ticker}_{f['report_date']}.html"
            try:
                if path.exists():
                    print(f"{ticker} {f['report_date']}: already downloaded")
                else:
                    path.write_bytes(sec_get(f["url"]).content)
                    print(f"{ticker} {f['report_date']}: downloaded")
                manifest.append({"ticker": ticker, "cik": cik, "report_date": f["report_date"],
                                 "accession": f["accession"], "html_path": str(path), "url": f["url"]})
            except Exception as e:
                failures.append({"ticker": ticker, "filing": f["report_date"], "error": str(e)})
                print(f"{ticker} {f['report_date']}: FAILED ({e})")

    write_csv(data_dir / "manifest.csv", manifest,
              ["ticker", "cik", "report_date", "accession", "html_path", "url"])
    write_csv(data_dir / "failures.csv", failures, ["ticker", "filing", "error"])
    print(f"\nDone: {len(manifest)} filings in manifest, {len(failures)} failures "
          f"(see {data_dir / 'failures.csv'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download recent 10-K filings from SEC EDGAR.")
    ap.add_argument("--limit", type=int, help="only process the first N tickers in tickers.txt")
    args = ap.parse_args()
    download_all(limit=args.limit)