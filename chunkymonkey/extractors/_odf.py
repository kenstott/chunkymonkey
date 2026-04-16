# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""OpenDocument Format extractor (ODT, ODS, ODP) using odfpy.

Handles:
  - ODT (Writer / word processor)  → markdown with heading levels
  - ODS (Calc / spreadsheet)       → sheet sections with pipe-delimited rows
  - ODP (Impress / presentation)   → slide sections with text blocks

Requires: odfpy>=1.4
"""

from __future__ import annotations

from io import BytesIO

try:
    import odf  # noqa: F401
    _ODF_AVAILABLE = True
except ImportError:
    _ODF_AVAILABLE = False


def _extract_odt(doc) -> str:
    from odf import teletype

    lines: list[str] = []
    for elem in doc.text.childNodes:
        tag = elem.qname[1] if hasattr(elem, "qname") else ""
        if tag == "h":
            level = int(elem.getAttribute("outlinelevel") or 1)
            text = teletype.extractText(elem).strip()
            if text:
                lines.append(f"{'#' * min(level, 6)} {text}")
        elif tag == "p":
            text = teletype.extractText(elem).strip()
            if text:
                lines.append(text)
    return "\n\n".join(lines)


def _extract_ods(doc) -> str:
    from odf.table import Table, TableRow, TableCell
    from odf import teletype

    sheets: list[str] = []
    for table in doc.spreadsheet.getElementsByType(Table):
        name = table.getAttribute("name") or "Sheet"
        rows: list[str] = []
        for row in table.getElementsByType(TableRow):
            cells = [
                teletype.extractText(cell).strip()
                for cell in row.getElementsByType(TableCell)
            ]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            sheets.append(f"[Sheet: {name}]\n" + "\n".join(rows))
    return "\n\n".join(sheets)


def _extract_odp(doc) -> str:
    from odf.draw import Page
    from odf import teletype

    slides: list[str] = []
    for i, page in enumerate(doc.presentation.getElementsByType(Page), 1):
        # Collect all text from every descendant text node on the page
        texts: list[str] = []
        for child in page.childNodes:
            text = teletype.extractText(child).strip()
            if text:
                texts.append(text)
        if texts:
            slides.append(f"[Slide {i}]\n" + "\n".join(texts))
    return "\n\n".join(slides)


class OdfExtractor:
    """Extract plain text from ODT, ODS, and ODP bytes."""

    HANDLED = {"odt", "ods", "odp"}

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        if not _ODF_AVAILABLE:
            raise ImportError(
                "odfpy is required for ODF extraction. "
                "Install it with: pip install chunkymonkey[odf]"
            )

        from odf.opendocument import load as odf_load

        doc = odf_load(BytesIO(data))
        mime = (doc.mimetype or "").lower()

        if "spreadsheet" in mime:
            return _extract_ods(doc)
        if "presentation" in mime:
            return _extract_odp(doc)
        # Default: treat as text/writer (ODT)
        return _extract_odt(doc)
