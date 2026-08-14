# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 43e91c0f-6376-43c8-ac24-6085a1badbe0
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Namespace lifecycle: async build pipeline and background freshness refresh."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._types import EmbedModel
from .community import build_community
from .indexer import IndexHandle
from .storage._store import GLOBAL_NAMESPACE, Store

_log = logging.getLogger(__name__)


def build_namespace_async(
    namespace_id: str,
    db_path: str | Path,
    embed_model: EmbedModel,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
    on_complete: Callable[[int], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
    force: bool = False,
    run_ner: bool = True,
    run_community: bool = True,
    spacy_model: str = "en_core_web_sm",
    community_alpha: float = 0.2,
    community_sim_threshold: float = 0.6,
    embed_batch_size: int = 256,
    dsn: str | None = None,
) -> IndexHandle:
    """Build the full index pipeline for *namespace_id* in a background thread.

    Phases: crawl → chunk → embed → NER → community → FTS.
    Returns an IndexHandle; call .join() to wait for completion.

    Idempotent when force=False: returns immediately if namespace_cache_valid().
    Safe to call concurrently for different namespaces (separate DB files).

    Args:
        namespace_id: Namespace to build.
        db_path: Path to this namespace's DuckDB file.
        embed_model: SentenceTransformer model name or instance.
        on_progress: Called with (phase, done, total).
        on_complete: Called with total chunk count on success.
        on_error: Called with (phase, exception) on failure.
        force: Rebuild even if namespace_cache_valid() is True.
        run_ner: Run NER after chunking.
        run_community: Build community index after NER.
        spacy_model: spaCy model for NER.
        community_alpha: Breadcrumb weight for community graph.
        community_sim_threshold: Cosine similarity threshold for community edges.
        embed_batch_size: Embedding batch size.
        dsn: PostgreSQL DSN. When set, the build runs against the PG backend and
            ``db_path`` is ignored. Community building is DuckDB-only, so
            ``run_community`` must be False when ``dsn`` is set.

    Raises:
        NotImplementedError: dsn is set together with run_community=True.
    """
    if dsn is not None and run_community:
        raise NotImplementedError(
            "build_namespace_async(dsn=...) cannot build the community index: "
            "build_community() reads a DuckDB file directly. Pass run_community=False."
        )

    _on_progress = on_progress or (lambda *_: None)
    _on_complete = on_complete or (lambda *_: None)
    _on_error = on_error or (lambda *_: None)

    def _run() -> None:
        db_path_ = Path(db_path)
        store = Store(dsn=dsn) if dsn is not None else Store(db_path_, read_only=False)

        try:
            if not force and store.namespace_cache_valid(namespace_id):
                _count_row = store.vector._conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE namespace = ?",
                    [namespace_id],
                ).fetchone()
                if _count_row is None:
                    raise RuntimeError("COUNT(*) returned no rows")
                _on_complete(_count_row[0])
                return

            # ── Phase: crawl / chunk / embed ──────────────────────────────────
            sources = store.vector._conn.execute(
                """
                SELECT s.source_id, s.type, s.uri, s.config, d.domain_id
                FROM sources s
                JOIN domains d ON s.domain_id = d.domain_id
                WHERE d.namespace_id = ?
                """,
                [namespace_id],
            ).fetchall()

            import json as _json

            from .indexer import Indexer

            indexer = Indexer(
                store,
                embed_model,
                on_progress=_on_progress,
                on_error=lambda phase, exc: _on_error(phase, exc),
                embed_batch_size=embed_batch_size,
            )
            total_chunks = 0
            for source_id, stype, uri, config_json, domain_id in sources:
                config = _json.loads(config_json) if config_json else {}
                source_config = {
                    "source_id": source_id,
                    "type": stype,
                    "uri": uri,
                    "domain_id": domain_id,
                    "namespace": namespace_id,
                    **config,
                }
                total_chunks += indexer.index_source(source_config)
                store.vector._conn.execute(
                    "UPDATE sources SET last_crawled = now() WHERE source_id = ?",
                    [source_id],
                ).fetchall()

            store._mark_namespace_built(namespace_id, "chunks")
            _on_progress("embed", total_chunks, total_chunks)

            # ── Phase: NER ────────────────────────────────────────────────────
            if run_ner:
                try:
                    from .ner import build_ner

                    _on_progress("ner", 0, 1)
                    build_ner(store, spacy_model=spacy_model, namespace=namespace_id)
                    store._mark_namespace_built(namespace_id, "ner")
                    _on_progress("ner", 1, 1)
                except Exception as exc:
                    _on_error("ner", exc)
                    raise

            # ── Phase: community ──────────────────────────────────────────────
            if run_community:
                try:
                    _on_progress("community", 0, 1)
                    build_community(
                        db_path_,
                        embed_model,
                        namespace_id=namespace_id,
                        alpha=community_alpha,
                        sim_threshold=community_sim_threshold,
                        force=force,
                    )
                    store._mark_namespace_built(namespace_id, "community")
                    _on_progress("community", 1, 1)
                except Exception as exc:
                    _on_error("community", exc)
                    raise

            # ── Phase: FTS ────────────────────────────────────────────────────
            try:
                _on_progress("fts", 0, 1)
                store.vector.rebuild_fts_index()
                _on_progress("fts", 1, 1)
            except Exception as exc:
                _on_error("fts", exc)
                raise

            _on_complete(total_chunks)

        finally:
            store.close()

    def _guarded() -> None:
        # The thread's exception is otherwise invisible: on_error defaults to a
        # no-op, so a caller that did not pass one sees a normal return.
        try:
            _run()
        except BaseException as exc:
            handle.error = exc
            raise

    t = threading.Thread(target=_guarded, daemon=True)
    handle = IndexHandle(t)
    t.start()
    return handle


class NamespaceRefresher:
    """Periodic background job that re-indexes stale namespaces.

    Each interval, checks all registered namespaces via namespace_cache_valid().
    Stale namespaces are queued for rebuild via build_namespace_async().
    Concurrent builds across different namespaces run in parallel (separate DBs).

    Args:
        db_path_fn: Callable mapping namespace_id to its DB file path. Required
            for the DuckDB backend; ignored (and may be None) when ``dsn`` is set.
        embed_model: SentenceTransformer model name or instance.
        interval_seconds: How often to check all namespaces (default 3600).
        on_rebuild: Called with namespace_id when a rebuild is triggered.
        dsn: PostgreSQL DSN. When set, all namespaces live in one PG database and
            enumeration/validation run against it instead of per-namespace files.
        build_kwargs: Extra kwargs forwarded to build_namespace_async.

    Raises:
        ValueError: neither db_path_fn nor dsn was supplied.
        NotImplementedError: dsn is set without run_community=False.
    """

    def __init__(
        self,
        db_path_fn: Callable[[str], str | Path] | None,
        embed_model: EmbedModel,
        interval_seconds: int = 3600,
        on_rebuild: Callable[[str], None] | None = None,
        dsn: str | None = None,
        **build_kwargs: Any,  # noqa: ANN401
    ) -> None:
        if db_path_fn is None and dsn is None:
            raise ValueError("NamespaceRefresher requires either db_path_fn or dsn")
        if dsn is not None and build_kwargs.get("run_community", True):
            # Fail at construction rather than inside the background loop thread.
            raise NotImplementedError(
                "NamespaceRefresher(dsn=...) cannot rebuild the community index: "
                "build_community() reads a DuckDB file directly. Pass run_community=False."
            )
        self._db_path_fn = db_path_fn
        self._dsn = dsn
        self._embed_model = embed_model
        self._interval = interval_seconds
        self._on_rebuild = on_rebuild or (lambda _: None)
        self._build_kwargs = build_kwargs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active: dict[str, IndexHandle] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background refresh loop (non-blocking)."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the refresh loop and wait for the loop thread to exit."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._check_all()

    def _open_store(self, namespace_id: str) -> Store:
        """Open a read-only Store for *namespace_id* on the configured backend."""
        if self._dsn is not None:
            return Store(dsn=self._dsn)
        assert self._db_path_fn is not None  # guaranteed by __init__
        return Store(self._db_path_fn(namespace_id), read_only=True)

    def _check_all(self) -> None:
        # Namespace enumeration goes through the Store abstraction so it works on
        # every backend. On DuckDB the catalog lives in the "global" namespace file;
        # on PG every namespace shares one database.
        try:
            store = self._open_store(GLOBAL_NAMESPACE)
            try:
                namespace_ids = store.list_namespaces()
            finally:
                store.close()
        except Exception:
            _log.exception("NamespaceRefresher: namespace enumeration failed; skipping this pass")
            return

        for ns_id in namespace_ids:
            with self._lock:
                handle = self._active.get(ns_id)
                if handle is not None and handle.running:
                    continue

            try:
                store = self._open_store(ns_id)
                try:
                    valid = store.namespace_cache_valid(ns_id)
                finally:
                    store.close()
            except Exception:
                _log.exception(
                    "NamespaceRefresher: freshness check failed for namespace %r; skipping", ns_id
                )
                continue

            if not valid:
                self._on_rebuild(ns_id)
                handle = build_namespace_async(
                    ns_id,
                    "" if self._db_path_fn is None else self._db_path_fn(ns_id),
                    self._embed_model,
                    dsn=self._dsn,
                    **self._build_kwargs,
                )
                with self._lock:
                    self._active[ns_id] = handle
