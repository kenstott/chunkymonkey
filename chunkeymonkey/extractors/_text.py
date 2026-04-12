"""Plain-text and text-like format extractor."""

from __future__ import annotations


class TextExtractor:
    """Extract content from text-based formats by decoding bytes."""

    HANDLED = {
        "text", "markdown", "md", "csv", "json", "jsonl",
        "yaml", "yml", "xml", "txt",
    }

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")
