# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d861b1a2-717f-492c-8394-98eeb30d08f7
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

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
            raise ImportError("pip install chunkymonkey[pdf]")

        reader = pypdf.PdfReader(BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i}]\n{text.strip()}")
        return "\n\n".join(pages)
