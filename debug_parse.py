import pdfplumber

path = "Apple 10-K Annual Filing (2025).pdf" 

with pdfplumber.open(path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if "total assets" in text.lower():
            print(f"Page {i}: found 'total assets'")
        if "total liabilities" in text.lower():
            print(f"Page {i}: found 'total liabilities'")
    # also check whether extraction is working at all
    print("\n--- Sample of page 10 text ---")
    print((pdf.pages[10].extract_text() or "[EMPTY]")[:500])