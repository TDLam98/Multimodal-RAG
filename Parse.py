"""
Parse a PDF page two ways: naive (pdfplumber) and layout-aware (Unstructured)
Also includes the balance-sheet page finder shared by both

Requirements: pip install pdfplumber pymupdf "unstructured[pdf]
"""
from pathlib import Path

import pymupdf  
import pdfplumber
from unstructured.partition.pdf import partition_pdf


DEFAULT_STRATEGY = "hi-res"     #falls back to "fast" if hi_res fails

def find_balance_sheet_page(pdf_path):
    """
    Returns the 0-based index of the first page containing both
    'total assets' and 'total liabilities'.
    Returns None if not found.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").lower()
            if "total assets" in text and "total liabilites" in text:
                return i
    return None

def save_single_page(pdf_path, page_idx, out_path):
    """
    Extract one page into its own PDF, so Unstructured doesn't
    have to process the whole filing
    """
    src = pymupdf.open(pdf_path)
    dst = pymupdf.open()
    dst.insert_pdf(src, from_page=page_idx, to_page=page_idx)
    dst.save(str(out_path))

def naive_parse(pdf_path, page_idx):
    """
    Naive extraction - returns (plain_text, table_text)
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_idx]
        text = page.extract_text() or ""
        tables = page.extract_tables()
    table_text = "\n\n".join("\n".join
                             ("\t".join(cell or "" for cell in row)
                               for row in t) for t in tables)
    return text, table_text

def layout_parse(one_page_pdf, strategy=DEFAULT_STRATEGY):
    """
    Layout extration - returns (text, strategy_used)
    Falls back to 'fast' stratgey if hi_res fails
    """
    kwargs = {"filename": str(one_page_pdf)}
    try:
        if strategy == "hi_res":
            kwargs["infer_table_structure"] = True
        elements = partition_pdf(strategy=strategy, **kwargs)
        used = strategy
    except Exception as e:
        print(f"{strategy} failed ({e}); falling back to 'fast'")
        elements = partition_pdf(strategy="fast", filename=str(one_page_pdf))
        used = "fast"

    lines = []
    for el in elements:
        lines.append(f"[{el.category}] {el}")
        html = getattr(el.metadata, "text_as_html", None)
        if html:
            lines.append(f"     table html: {html}")
    return "\n".join(lines), used

def total_assets_row_intact(text):
    """
    Quick sanity check to see if the "Total Assets" line contains digits
    """
    for line in text.splitlines():
        if "total assets" in line.lower():
            return any(ch.isdigit() for ch in line)
        
    return False
