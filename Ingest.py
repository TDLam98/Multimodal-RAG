"""
EDGAR ingestion: look up a company's CIK and download a 10-K filing's HTML

Requirements: pip install requests
"""
import time
from pathlib import Path
import requests

USER_AGENT = "Tan-Dat Lam txl7064@mavs.uta.edu"     #SEC requires this on every request
HEADERS = {"User-Agent": USER_AGENT}

def sec_get(url):
    """
    Wrapper function to fetch data from the SEC EDGAR API
    """
    time.sleep(0.2)     #stay under SEC's 10 requests/second limit
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r

def ticker_to_cik(ticker):
    """
    Converts a stock ticker symbol (like "AAPL" or "MSFT") into its official
    Central Index Key (CIK) used by the SEC EDGAR system
    """
    data = sec_get("https://www.sec.gov/files/company_tickers.json").json()
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker} not found")

def download_10k_html(cik, out_path, report_year=None):
    """
    Download the primary document of the most recent matching 10-K to out_path
    """
    subs = sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    recent = subs["filings"]["recent"]
    rows = zip(recent["form"], recent["accessionNumber"], 
               recent["primaryDocument"], recent["reportDate"])
    for form, acc, doc, report_date in rows:
        if form != "10-K":
            continue
        if report_year and not report_date.startswith(str(report_year)):
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
        print(f"Downloading 10-K (period ending {report_date}): {url}")
        out_path = Path(out_path)
        out_path.write_bytes(sec_get(url).content)
        return out_path
    raise ValueError("No matching 10-K found in the recent filings list")