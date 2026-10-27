"""
Build tickers.txt: a reproducible random sample of S&P 500 companies.

    python make_ticker_list.py
        First run: samples N_COMPANIES companies and writes tickers.txt.
        Later runs: KEEPS the existing tickers.txt and only ADDS anything new in
        MUST_INCLUDE. 

    python make_ticker_list.py --resample
        Throws away tickers.txt and draws a fresh sample.

Requires (first run only): pip install pandas lxml requests
"""
import argparse
from io import StringIO
from pathlib import Path

# ------------------------- CONFIG -------------------------------------
N_COMPANIES = 130
SEED = 42                      # same seed = same sample (for the same Wikipedia table)
MUST_INCLUDE = ["AAPL", "ABT", "AMAT", "BKR", "DOV", "HON", "HPQ", "JPM", "KHC", "MCD",
                "MDLZ", "MMM", "MU", "NEE", "PCAR", "RTX", "SYK", "TMO", "XEL", "XOM"]    # companies with real pdfs on hand
USER_AGENT = "Tan-Dat Lam txl7064@mavs.uta.edu"
# ----------------------------------------------------------------------

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT_FILE = Path(__file__).resolve().parent / "tickers.txt"


def read_existing():
    if not OUT_FILE.exists():
        return []
    lines = OUT_FILE.read_text().splitlines()
    return [l.strip().upper() for l in lines if l.strip() and not l.startswith("#")]


def write_tickers(tickers):
    OUT_FILE.write_text("\n".join(tickers) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Build or extend tickers.txt.")
    ap.add_argument("--resample", action="store_true",
                    help="discard tickers.txt and sample a fresh list")
    args = ap.parse_args()

    must = [t.upper() for t in MUST_INCLUDE]
    existing = [] if args.resample else read_existing()

    if existing:
        added = [t for t in must if t not in existing]
        write_tickers(existing + added)
        print(f"Kept {len(existing)} existing tickers; added {len(added)}: {added}")
        print("(Use --resample if you really want a brand-new list.)")
        return

    import pandas as pd
    import requests

    html = requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=60).text
    table = pd.read_html(StringIO(html))[0]

    # Wikipedia writes BRK.B; SEC's ticker table writes BRK-B
    table["ticker"] = table["Symbol"].str.replace(".", "-", regex=False)

    sampled = table.sample(n=N_COMPANIES, random_state=SEED)["ticker"].tolist()
    tickers = list(dict.fromkeys(must + sampled))  # dedupe, keep order
    write_tickers(tickers)

    print(f"Wrote {len(tickers)} tickers to {OUT_FILE}\n")
    print("Sector spread:")
    print(table.loc[table["ticker"].isin(tickers), "GICS Sector"].value_counts())


if __name__ == "__main__":
    main()