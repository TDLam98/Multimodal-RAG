"""
Driver file for Checkpoint #1. Reusable logic 

Downloads one 10-K, converts it into a PDF, and parses the balance sheet
page two ways on both a converted PDF and a real PDF.
"""
from pathlib import Path
from Ingest import ticker_to_cik, download_10k_html
from Convert import html_to_pdf
from Parse import (
    find_balance_sheet_page,
    save_single_page,
    naive_parse,
    layout_parse,
    total_assets_row_intact
)

#----------- Config --------------
TICKER = "AAPL"         #company of real PDF
REPORT_YEAR = 2025      #year of real PDF so the EDGAR download matches
REAL_PDF = Path("Apple 10-K Annual Filing (2025).pdf")  #company published PDF, downloaded manually
CONVERTED_PAGE = 52     #set a 0-based page index to override auto-detection, hard-coded for testing
REAL_PAGE = 33          #hard-coded for testing
UNSTRUCTURED_STRATEGY = "hi_res"    #falls back to "fast" if hi_res fails
#---------------------------------

DATA = Path("data")
OUT = Path("out")

def run_both(pdf_path, label, page_override=None):
    pdf_path = Path(pdf_path)
    page_idx = page_override if page_override is not None else find_balance_sheet_page(pdf_path)
    if page_idx is None:
        print(f"{label}: balance sheet page not found; set the page override in Config")
        return {"label": label, "page": None}
    print(f"{label}: balance sheet on page index {page_idx}")

    naive_text, naive_tables = naive_parse(pdf_path, page_idx)
    (OUT / f"{label}_naive_text.txt").write_text(naive_text, encoding="utf-8")
    (OUT / f"{label}_naive_tables.txt").write_text(naive_tables, encoding="utf-8")

    one_page = OUT / f"{label}_page.pdf"
    save_single_page(pdf_path, page_idx, one_page)  #keeps Unstructured fast
    layout_text, strategy = layout_parse(one_page, strategy=UNSTRUCTURED_STRATEGY)
    (OUT / f"{label}_layout.txt").write_text(layout_text, encoding="utf-8")

    return{
        "label": label,
        "page" : page_idx,
        "naive_ok" : total_assets_row_intact(naive_text),
        "layout_ok" : total_assets_row_intact(layout_text),
        "strategy": strategy
    }

def write_summary(results):
    lines = ["| Source | Page idx | Naive: intact? | Layout-aware: intact? | Unstructured strategy |",
             "|---|---|---|---|---|"]
    for r in results:
        if r["page"] is None:
            lines.append(f"| {r['label']} | not found | - | - | - |")
        else:
            lines.append(f"| {r['label']} | {r['page']} | {r['naive_ok']} | " 
                         f"{r['layout_ok']} | {r['strategy']} |")
    summary = "\n".join(lines)
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print("\nThe intact check is only a rough heuristic. Open the files in out/ "
          "and compare them against the balance sheet in the PDF itself")

def main():
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    cik = ticker_to_cik(TICKER)
    html_path = download_10k_html(cik, DATA/ "filing.html", REPORT_YEAR)

    converted = DATA / "converted.pdf"
    print("Converting HTML to PDF...")
    html_to_pdf(html_path, converted)

    results = [run_both(converted, "converted", CONVERTED_PAGE)]
    if REAL_PDF.exists():
        results.append(run_both(REAL_PDF, "real", REAL_PAGE))
    else:
        print(f"Skipping real PDF: {REAL_PDF} not found")
    write_summary(results)


if __name__ == "__main__":
    main()