"""PDF text extractor using pypdf."""

from __future__ import annotations

from io import BytesIO

try:
    import pypdf
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False


class PdfExtractor:
    """Extract plain text from PDF bytes."""

    def can_handle(self, doc_type: str) -> bool:
        return doc_type == "pdf"

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        if not _PYPDF_AVAILABLE:
            raise ImportError("pip install chunkeymonkey[pdf]")

        reader = pypdf.PdfReader(BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i}]\n{text.strip()}")
        return "\n\n".join(pages)
