"""
Shared table handling for the HTML path and the PDF path.

clean_cell():      tidy one cell of text
clean_row():       tidy one table row (merge split "(" / ")" / "%" cells, drop blanks and "$")
serialize_table(): turn a table (a list of rows) into the text we embed and retrieve

Using the same serializer for tables from every source is what makes the
HTML and PDF versions of a table comparable at retrieval time.
"""
import re


def clean_cell(text):
    """Collapse whitespace (including non-breaking spaces) and strip."""
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def clean_row(cells):
    """EDGAR tables split one number across several cells, e.g.
    ['Net loss', '$', '(1,234', ')', '', '$', '(900', ')'].
    This returns ['Net loss', '(1,234)', '(900)']."""
    out, prefix = [], ""
    for raw in cells:
        c = clean_cell(raw)
        if not c or c == "$":
            continue
        if c in {"(", "$("}:
            prefix = "("
            continue
        if c in {")", "%", "%)", ")%"} and out:
            out[-1] += c
            continue
        out.append(prefix + c)
        prefix = ""
    return out


def serialize_table(rows, title=""):
    """One line per row, cells separated by ' | ', with an optional title line."""
    lines = []
    if title:
        lines.append(f"TABLE: {title}")
    for row in rows:
        row = clean_row(row)
        if row:
            lines.append(" | ".join(row))
    return "\n".join(lines)