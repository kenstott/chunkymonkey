# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 8a3c2e91-7f4b-4e5d-b6a8-0d1f9c3e5b7a
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Background source indexer with progress callbacks and safe abort."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._types import EmbedModel
from .storage._store import Store

if TYPE_CHECKING:
    from .models import DocumentChunk

# ── Singleton registry ────────────────────────────────────────────────────────
# One Indexer per namespace, keyed by namespace id.  Search sessions always
# open their Store read-only; only the Indexer here holds a write connection.
_registry: dict[str, Indexer] = {}
_registry_lock = threading.Lock()


def get_indexer(
    namespace_id: str,
    store: Store,
    embed_model: EmbedModel,
    **kwargs: Any,  # noqa: ANN401
) -> Indexer:
    """Return the process-wide Indexer for *namespace_id*, creating it once.

    Subsequent calls with the same namespace_id return the cached instance.
    *store* and *embed_model* are used only on first creation.
    """
    with _registry_lock:
        if namespace_id not in _registry:
            _registry[namespace_id] = Indexer(store, embed_model, **kwargs)
        return _registry[namespace_id]


def release_indexer(namespace_id: str) -> None:
    """Remove the Indexer for *namespace_id* from the registry.

    Aborts any in-progress run first. Call when the namespace is deleted or
    the process is shutting down a namespace cleanly.
    """
    with _registry_lock:
        indexer = _registry.pop(namespace_id, None)
    if indexer is not None:
        indexer.abort()


class IndexHandle:
    """Handle returned by index_source_async. Call .join() to wait for completion.

    A background build that dies leaves its exception here. Without it the only
    signal is a traceback on stderr, so a caller that did not pass ``on_error``
    cannot tell a failed run from a successful one.
    """

    def __init__(self, thread: threading.Thread) -> None:
        self._thread = thread
        self.error: BaseException | None = None

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    @property
    def failed(self) -> bool:
        """True once the background run has ended in an exception."""
        return self.error is not None

    def raise_if_failed(self) -> None:
        """Re-raise the background failure in the calling thread, if any."""
        if self.error is not None:
            raise self.error


class Indexer:
    """Background source indexer.

    Runs indexing in a background thread. Safe to abort mid-run — always
    finishes the current embedding batch before stopping, leaving the DB
    in a consistent state.

    Args:
        store: An open Store instance to write chunks into.
        embed_model: SentenceTransformer model name or instance. Required.
        on_progress: Called with (phase: str, done: int, total: int).
                     phase is one of: "crawl", "chunk", "embed", "store".
        on_complete: Called with (chunks: int) when indexing finishes.
        on_error: Called with (phase: str, error: Exception) on non-fatal errors.
        on_abort: Called with (chunks: int) when abort() was called and the
                  thread has stopped cleanly.
        embed_batch_size: Embedding batch size (default 256).
        min_chunk_size: Minimum chunk character size (default 400).
        max_chunk_size: Maximum chunk character size (default 1200).
        embed_content_only: Embed content field only, not breadcrumb (default True).
    """

    def __init__(
        self,
        store: Store,
        embed_model: EmbedModel,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_complete: Callable[[int], None] | None = None,
        on_error: Callable[[str, Exception], None] | None = None,
        on_abort: Callable[[int], None] | None = None,
        embed_batch_size: int = 256,
        min_chunk_size: int = 400,
        max_chunk_size: int = 1200,
        embed_content_only: bool = True,
    ) -> None:
        self._store = store
        self._embed_model = embed_model
        self._on_progress = on_progress or (lambda *_: None)
        self._on_complete = on_complete or (lambda *_: None)
        self._on_error = on_error or (lambda *_: None)
        self._on_abort = on_abort or (lambda *_: None)
        self._batch_size = embed_batch_size
        self._min_chunk = min_chunk_size
        self._max_chunk = max_chunk_size
        self._embed_content_only = embed_content_only
        self._abort_flag = threading.Event()

    def abort(self) -> None:
        """Signal the background thread to stop after the current batch."""
        self._abort_flag.set()

    def index_source(self, source_config: dict[str, Any]) -> int:
        """Blocking index of one source config dict. Returns chunks stored."""
        return self._run(source_config)

    def index_source_async(self, source_config: dict[str, Any]) -> IndexHandle:
        """Non-blocking index. Returns an IndexHandle; call .join() to wait."""
        self._abort_flag.clear()
        t = threading.Thread(target=self._run, args=(source_config,), daemon=True)
        t.start()
        return IndexHandle(t)

    def _run(self, source_config: dict[str, Any]) -> int:
        """Internal: run the full indexing pipeline for one source."""
        import numpy as np

        # Resolve embed model
        if isinstance(self._embed_model, str):
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._embed_model)
        else:
            model = self._embed_model

        source_id = source_config.get("source_id")
        domain_id = source_config.get("domain_id")
        namespace = source_config.get("namespace")

        # ── Phase: crawl + chunk ──────────────────────────────────────────────
        # _crawl uses DocumentLoader which handles extraction and chunking
        # in one pass so we don't need a separate chunk phase.
        all_chunks: list[DocumentChunk] = []
        try:
            all_chunks = self._crawl(source_config)
        except Exception as exc:
            self._on_error("crawl", exc)
            return 0

        total_chunks = len(all_chunks)
        self._on_progress("crawl", total_chunks, total_chunks)
        self._on_progress("chunk", total_chunks, total_chunks)

        if self._abort_flag.is_set():
            self._on_abort(0)
            return 0

        # ── Phase: embed ──────────────────────────────────────────────────────
        texts = [
            c.content
            if self._embed_content_only
            else (c.embedding_content if c.embedding_content else c.content)
            for c in all_chunks
        ]
        total_batches = max(1, (total_chunks + self._batch_size - 1) // self._batch_size)
        embeddings = []
        for batch_idx, i in enumerate(range(0, total_chunks, self._batch_size)):
            if self._abort_flag.is_set():
                self._on_abort(0)
                return 0
            batch = texts[i : i + self._batch_size]
            try:
                vecs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
                embeddings.append(vecs)
            except Exception as exc:
                self._on_error("embed", exc)
                return 0
            self._on_progress("embed", batch_idx + 1, total_batches)

        if not embeddings:
            self._on_complete(0)
            return 0

        emb = np.vstack(embeddings).astype("float32")

        if self._abort_flag.is_set():
            self._on_abort(0)
            return 0

        # ── Phase: store ──────────────────────────────────────────────────────
        try:
            self._store.add_document(
                all_chunks,
                emb,
                namespace=namespace,
                source_id=source_id,
                domain_id=domain_id,
            )
            self._on_progress("store", total_chunks, total_chunks)
        except Exception as exc:
            self._on_error("store", exc)
            return 0

        self._on_complete(total_chunks)
        return total_chunks

    def _crawl(self, source_config: dict[str, Any]) -> list[DocumentChunk]:
        """Dispatch to the correct crawler/transport and return DocumentChunks.

        Uses DocumentLoader so all file types (PDF, DOCX, HTML, …) are handled
        by the extractor registry.  Chunking and enrichment happen here via
        DocumentLoader's pipeline, not in _run.
        """
        from chonk.loader import DocumentLoader
        from chonk.transports import (
            DatabaseSchemaCrawler,
            DirectoryCrawler,
            GitHubCrawler,
            WebCrawler,
        )

        source_type = source_config.get("type", "directory")
        uri = source_config.get("uri", "")

        loader = DocumentLoader(
            min_chunk_size=self._min_chunk,
            max_chunk_size=self._max_chunk,
            enrich_context=True,
            include_doc_name=False,
        )

        if source_type == "directory":
            crawler = DirectoryCrawler(
                extensions=source_config.get("extensions"),
                recursive=source_config.get("recursive", True),
                max_files=source_config.get("max_files", 1000),
            )
            return loader.load_crawl(uri, crawler=crawler)

        if source_type == "github":
            crawler = GitHubCrawler(
                token=source_config.get("token"),
                branch=source_config.get("branch"),
                extensions=source_config.get("extensions"),
                max_files=source_config.get("max_files", 2000),
            )
            return loader.load_crawl(uri, crawler=crawler)

        if source_type == "web":
            crawler = WebCrawler(
                max_pages=source_config.get("max_pages", 100),
                max_depth=source_config.get("max_depth", 3),
                same_domain=source_config.get("same_domain", True),
            )
            return loader.load_crawl(uri, crawler=crawler)

        if source_type == "db_schema":
            crawler = DatabaseSchemaCrawler(
                uri,
                include_procs=source_config.get("include_procs", True),
                include_views=source_config.get("include_views", True),
                schemas=source_config.get("schemas"),
            )
            loader_with_transport = DocumentLoader(
                min_chunk_size=self._min_chunk,
                max_chunk_size=self._max_chunk,
                enrich_context=True,
                include_doc_name=False,
                extra_transports=[crawler],
            )
            return loader_with_transport.load_crawl(uri, crawler=crawler)

        if source_type == "http":
            return loader.load(uri)

        raise ValueError(f"Unknown source type: {source_type!r}")
