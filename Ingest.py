"""
EDGAR ingestion.

Run this file directly to bulk-download recent 10-K filings for a list of tickers:
    python ingest.py

Outputs
    <DATA_DIR>/html/<TICKER>_<report_date>.html   one file per filing
    <DATA_DIR>/manifest.csv                        what was downloaded (for later steps)
    <DATA_DIR>/failures.csv                        anything that failed, and why

Safe to re-run: files that already exist are skipped, so a crash or a
rate-limit block never means starting over.

Requires: pip install requests
"""
import csv
import time
from pathlib import Path

import requests

# ------------------------- CONFIG -------------------------------------
USER_AGENT = "Tan-Dat Lam txl7064@mavs.uta.edu"  # SEC requires this on every request
TICKERS = Path(__file__).resolve().parent / "tickers.txt"   #pulls from list generated from ticker_list
FILINGS_PER_COMPANY = 2                                     
DATA_DIR = DATA_DIR = Path(__file__).resolve().parent / "data"
# ----------------------------------------------------------------------

HEADERS = {"User-Agent": USER_AGENT}


def sec_get(url):
    time.sleep(0.2)  # stay well under SEC's 10 requests/second limit
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r


# ---------- Bulk download ----------
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

def load_tickers(path=TICKERS_FILE):
    lines = Path(path).read_text().splitlines()
    return [l.strip().upper() for l in lines if l.strip() and not l.startswith("#")]

def download_all(tickers=None, per_company=FILINGS_PER_COMPANY, data_dir=DATA_DIR):
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
    download_all()