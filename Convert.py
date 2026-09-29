"""
Convert a local HTML filing into a PDF using headless Chromium

Requirements: pip install playwright & playwright install chromium
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

def html_to_pdf(html_path, pdf_path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(Path(html_path).resolve().as_uri(), wait_until="networkidle", timeout=120_000)
        page.wait_for_timeout(1000)  # small extra buffer for final layout
        page.pdf(path=str(pdf_path), format="Letter", print_background=True,
                 margin={"top": "0.75in", "bottom": "0.75in",
                         "left": "0.75in", "right": "0.75in"})
        browser.close()

def batch_convert(html_dir, pdf_dir):
    """
    Convert every .html file in html_dir to a same-named .pdf in pdf_dir,
    reusing one browser instance. Skips files that already have a PDF in case
    of having to restart the conversion.
    """
    html_dir, pdf_dir = Path(html_dir), Path(pdf_dir)
    pdf_dir.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for html_file in html_dir.glob("*.html"):
            pdf_file = pdf_dir / (html_file.stem + ".pdf")
            if pdf_file.exists():
                continue
            page.goto(html_file.resolve().as_uri(), wait_until="load", timeout=120_000)
            page.pdf(path=str(pdf_file), format="Letter", print_background=True)
        browser.close()