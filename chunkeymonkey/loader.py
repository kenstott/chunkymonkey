"""DocumentLoader — orchestrates Transport → Extractor → chunk_document → enrich_chunks."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .models import DocumentChunk
from .chunking import chunk_document
from .context import enrich_chunks
from .extractors import detect_extractor, detect_type_from_source, normalize_type
from .transports import (
    detect_transport,
    FetchResult,
    LocalTransport,
    HttpTransport,
    S3Transport,
    FtpTransport,
    SftpTransport,
)


class DocumentLoader:
    """Full pipeline: fetch bytes → extract text → chunk → contextual enrichment.

    Usage::

        loader = DocumentLoader(chunk_size=1500)
        chunks = loader.load("/path/to/document.pdf")
        chunks = loader.load("https://example.com/doc.html", name="example")
        chunks = loader.load_text("raw text here", name="my-doc")
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        table_chunk_limit: int = 800,
        context_strategy: str | None = "prefix",
        extra_transports: list | None = None,
        extra_extractors: list | None = None,
    ):
        """Args:
            chunk_size: Target max characters per chunk.
            table_chunk_limit: Max size for table blocks before row-level splitting.
            context_strategy: Enrichment strategy ("prefix", "inline", or None to skip).
            extra_transports: Additional transport backends checked before defaults.
            extra_extractors: Additional extractor backends checked before defaults.
        """
        self.chunk_size = chunk_size
        self.table_chunk_limit = table_chunk_limit
        self.context_strategy = context_strategy
        self._extra_transports = extra_transports or []
        self._extra_extractors = extra_extractors or []

        self._transport_registry = self._extra_transports + [
            LocalTransport(),
            HttpTransport(),
            S3Transport(),
            FtpTransport(),
            SftpTransport(),
        ]

    def _find_transport(self, uri: str):
        for transport in self._transport_registry:
            if transport.can_handle(uri):
                return transport
        raise ValueError(f"No transport found for URI: {uri!r}")

    def _find_extractor(self, doc_type: str):
        for extractor in self._extra_extractors:
            if extractor.can_handle(doc_type):
                return extractor
        return detect_extractor(doc_type)

    def _find_extractor_raw(self, raw_doc_type: str):
        """Try extra extractors first with the raw (un-normalised) type string.

        Falls back to normalising via normalize_type and then detect_extractor.
        This allows custom extractors to register arbitrary doc_type strings
        (e.g. "csv-summary", "confluence-wiki") without having to add them to
        the MIME registry.
        """
        for extractor in self._extra_extractors:
            if extractor.can_handle(raw_doc_type):
                return extractor
        resolved = normalize_type(raw_doc_type)
        return self._find_extractor(resolved)

    def _enrich(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        if self.context_strategy is None:
            return chunks
        return enrich_chunks(chunks, strategy=self.context_strategy)

    def load(self, uri: str, name: str | None = None) -> list[DocumentChunk]:
        """Fetch + extract + chunk + enrich.

        Args:
            uri: File path, file:// URL, http(s):// URL, s3://, ftp://, or sftp://.
            name: Document name for chunk metadata. Defaults to the last path segment.

        Returns:
            List of DocumentChunk objects.
        """
        transport = self._find_transport(uri)
        result: FetchResult = transport.fetch(uri)

        doc_type = detect_type_from_source(result.source_path or uri, result.detected_mime)
        extractor = self._find_extractor(doc_type)
        text = extractor.extract(result.data, result.source_path)

        doc_name = name or Path(result.source_path or uri).stem

        chunks = chunk_document(doc_name, text, self.chunk_size, self.table_chunk_limit)
        return self._enrich(chunks)

    def load_bytes(
        self,
        data: bytes,
        name: str,
        doc_type: str = "auto",
        source_path: str | None = None,
    ) -> list[DocumentChunk]:
        """Skip fetch; extract from raw bytes.

        Args:
            data: Raw document bytes.
            name: Document name for chunk metadata.
            doc_type: Explicit document type (e.g. "pdf", "docx") or "auto" to detect
                      from source_path.
            source_path: Optional path hint for type detection.

        Returns:
            List of DocumentChunk objects.
        """
        if doc_type == "auto":
            resolved_type = detect_type_from_source(source_path or name, None)
            extractor = self._find_extractor(resolved_type)
        else:
            extractor = self._find_extractor_raw(doc_type)
        text = extractor.extract(data, source_path)

        chunks = chunk_document(name, text, self.chunk_size, self.table_chunk_limit)
        return self._enrich(chunks)

    def load_text(self, text: str, name: str) -> list[DocumentChunk]:
        """Skip fetch and extract; chunk and enrich pre-extracted text.

        Args:
            text: Pre-extracted plain text.
            name: Document name for chunk metadata.

        Returns:
            List of DocumentChunk objects.
        """
        chunks = chunk_document(name, text, self.chunk_size, self.table_chunk_limit)
        return self._enrich(chunks)
