"""Document format extractors."""

from ._protocol import Extractor
from ._mime import normalize_type, detect_type_from_source, is_binary_type
from ._text import TextExtractor
from ._html import HtmlExtractor
from ._pdf import PdfExtractor
from ._docx import DocxExtractor
from ._xlsx import XlsxExtractor
from ._pptx import PptxExtractor

_REGISTRY: list[Extractor] | None = None


def _build_registry() -> list[Extractor]:
    return [
        PdfExtractor(),
        DocxExtractor(),
        XlsxExtractor(),
        PptxExtractor(),
        HtmlExtractor(),
        TextExtractor(),
    ]


def detect_extractor(doc_type: str) -> Extractor:
    """Return the first registered extractor that can handle doc_type."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    for ext in _REGISTRY:
        if ext.can_handle(doc_type):
            return ext
    raise ValueError(f"No extractor found for doc_type={doc_type!r}")


__all__ = [
    "Extractor",
    "TextExtractor",
    "HtmlExtractor",
    "PdfExtractor",
    "DocxExtractor",
    "XlsxExtractor",
    "PptxExtractor",
    "detect_extractor",
    "normalize_type",
    "detect_type_from_source",
    "is_binary_type",
]
