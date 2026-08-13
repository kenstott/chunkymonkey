# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d861b1a2-717f-492c-8394-98eeb30d08f7
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""PDF text extractor using pypdf."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chonk.models import DocumentChunk

try:
    import pypdf as pypdf

    _PYPDF_AVAILABLE = True
except ImportError:
    pypdf = None  # type: ignore[assignment]
    _PYPDF_AVAILABLE = False


class PdfExtractor:
    """Extract plain text from PDF bytes."""

    def can_handle(self, doc_type: str) -> bool:
        return doc_type == "pdf"

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        if not _PYPDF_AVAILABLE:
            raise ImportError("pip install chonk-rag[pdf]")
        assert pypdf is not None

        reader = pypdf.PdfReader(BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i}]\n{text.strip()}")
        return "\n\n".join(pages)

    def annotate(
        self,
        chunks: list[DocumentChunk],
        data: bytes,
        source_path: str | None = None,
    ) -> list[DocumentChunk]:
        if not _PYPDF_AVAILABLE:
            raise ImportError("pip install chonk-rag[pdf]")
        assert pypdf is not None

        reader = pypdf.PdfReader(BytesIO(data))
        page_texts: list[tuple[int, str]] = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                page_texts.append((i, text.strip()))

        for chunk in chunks:
            content = chunk.content
            matched_pages: list[int] = []
            for page_num, page_text in page_texts:
                # Check if any substantial fragment of the chunk appears on this page
                if any(frag in page_text for frag in content.split("\n") if len(frag.strip()) > 20):
                    matched_pages.append(page_num)
            if matched_pages:
                if len(matched_pages) == 1:
                    chunk.source_detail = {"page": matched_pages[0]}
                else:
                    chunk.source_detail = {
                        "page_start": matched_pages[0],
                        "page_end": matched_pages[-1],
                    }  # noqa: E501

        return chunks
