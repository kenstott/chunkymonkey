# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 6bdda530-9e8a-4fc9-9c12-941e3197beca
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DocumentLoader — orchestrates Transport → Extractor → chunk_document → enrich_chunks."""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .models import DocumentChunk
from .chunking import chunk_document
from .extractors import detect_extractor, detect_type_from_source, normalize_type
from .transports import (
    detect_transport,
    FetchResult,
    LocalTransport,
    HttpTransport,
    S3Transport,
    FtpTransport,
    SftpTransport,
    WebCrawler,
    DirectoryCrawler,
    Crawler,
)

logger = logging.getLogger(__name__)


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
        min_chunk_size: int = 600,
        max_chunk_size: int = 1500,
        overflow_margin: float = 0.15,
        context_strategy: str | None = "prefix",
        extra_transports: list | None = None,
        extra_extractors: list | None = None,
    ):
        """Args:
            min_chunk_size: Accumulation floor — accumulate across sections until
                this size is reached (default 600).
            max_chunk_size: Hard ceiling — blocks exceeding this + overflow_margin
                are split at natural boundaries (default 1500).
            overflow_margin: Fractional slack above max_chunk_size before a split
                is forced (default 0.15 = 15%).
            context_strategy: "prefix" (default) or "inline" embeds an LCA
                breadcrumb at the start of every chunk's content and sets
                embedding_content = content.  None produces naive chunks with no
                breadcrumb and embedding_content = None.
            extra_transports: Additional transport backends checked before defaults.
            extra_extractors: Additional extractor backends checked before defaults.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overflow_margin = overflow_margin
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
        # Breadcrumb is already embedded in content by chunk_document.
        # Set embedding_content = content so callers can use either field.
        return [dataclasses.replace(c, embedding_content=c.content) for c in chunks]

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

        chunks = chunk_document(
            doc_name, text,
            self.min_chunk_size, self.max_chunk_size, self.overflow_margin,
            include_breadcrumb=(self.context_strategy is not None),
        )
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

        chunks = chunk_document(
            name, text,
            self.min_chunk_size, self.max_chunk_size, self.overflow_margin,
            include_breadcrumb=(self.context_strategy is not None),
        )
        return self._enrich(chunks)

    def load_text(self, text: str, name: str) -> list[DocumentChunk]:
        """Skip fetch and extract; chunk and enrich pre-extracted text.

        Args:
            text: Pre-extracted plain text.
            name: Document name for chunk metadata.

        Returns:
            List of DocumentChunk objects.
        """
        chunks = chunk_document(
            name, text,
            self.min_chunk_size, self.max_chunk_size, self.overflow_margin,
            include_breadcrumb=(self.context_strategy is not None),
        )
        return self._enrich(chunks)

    # ── Multi-document crawl methods ─────────────────────────────────────────

    def load_crawl(
        self,
        uri: str,
        crawler: "Crawler | None" = None,
        **crawler_kwargs,
    ) -> list[DocumentChunk]:
        """Crawl *uri* with *crawler*, then load each discovered document.

        This is the generic entry point.  ``load_site`` and ``load_directory``
        are convenience wrappers that auto-select an appropriate crawler.

        Args:
            uri:            Root URI to crawl (URL, path, ``s3://`` prefix, …).
            crawler:        A ``Crawler`` implementation.  If None, uses
                            ``WebCrawler`` for http(s) and ``DirectoryCrawler``
                            for everything else.
            **crawler_kwargs: Passed to ``crawler.crawl()``.

        Returns:
            Combined list of chunks from all discovered documents.
        """
        if crawler is None:
            if uri.startswith("http://") or uri.startswith("https://"):
                crawler = WebCrawler()
            else:
                crawler = DirectoryCrawler()

        uris = crawler.crawl(uri, **crawler_kwargs)
        logger.info("load_crawl: %d URI(s) discovered from %s", len(uris), uri)

        all_chunks: list[DocumentChunk] = []
        for doc_uri in uris:
            try:
                chunks = self.load(doc_uri)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.warning("load_crawl: skipping %s: %s", doc_uri, exc)
        return all_chunks

    def load_site(
        self,
        url: str,
        max_pages: int = 50,
        max_depth: int = 3,
        same_domain: bool = True,
        exclude_patterns: list[str] | None = None,
        include_pattern: str | None = None,
        crawler: "Crawler | None" = None,
    ) -> list[DocumentChunk]:
        """Crawl a website and load all discovered HTML pages.

        Extend for authenticated services (SharePoint, Confluence, …) by
        passing a custom ``crawler`` that implements the ``Crawler`` protocol::

            class SharePointCrawler:
                def can_handle(self, uri): return "sharepoint.com" in uri
                def crawl(self, uri, **kw): ...  # return list of URIs

            chunks = loader.load_site(url, crawler=SharePointCrawler())

        Args:
            url:              Root URL to start from.
            max_pages:        Maximum pages to fetch (default 50).
            max_depth:        Maximum link-follow depth (default 3).
            same_domain:      Stay on the same hostname (default True).
            exclude_patterns: Regex patterns for URLs to skip.
            include_pattern:  If set, only follow URLs matching this regex.
            crawler:          Custom crawler; overrides all other params.

        Returns:
            Combined list of DocumentChunk objects from all pages.
        """
        if crawler is None:
            crawler = WebCrawler(
                max_pages=max_pages,
                max_depth=max_depth,
                same_domain=same_domain,
                exclude_patterns=exclude_patterns,
                include_pattern=include_pattern,
            )
        return self.load_crawl(url, crawler=crawler)

    def load_directory(
        self,
        path: str,
        extensions: list[str] | None = None,
        recursive: bool = True,
        max_files: int = 1000,
        crawler: "Crawler | None" = None,
    ) -> list[DocumentChunk]:
        """Load all documents in a local directory or S3 prefix.

        Extend for cloud storage (Azure Blob, GCS, …) by passing a custom
        ``crawler`` that implements the ``Crawler`` protocol.

        Args:
            path:       Directory path (local or ``s3://bucket/prefix``).
            extensions: File extensions to include (default: broad document set).
            recursive:  Recurse into subdirectories (default True).
            max_files:  Maximum files to process (default 1000).
            crawler:    Custom crawler; overrides all other params.

        Returns:
            Combined list of DocumentChunk objects from all files.
        """
        if crawler is None:
            crawler = DirectoryCrawler(
                extensions=extensions,
                recursive=recursive,
                max_files=max_files,
            )
        return self.load_crawl(path, crawler=crawler)
