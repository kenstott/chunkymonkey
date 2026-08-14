# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: fb0dc27c-4ed9-413d-ae7f-be8255a0d901
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""VectorBackend protocol — implemented by DuckDBVectorBackend and PgVectorBackend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..models import DocumentChunk


@runtime_checkable
class VectorBackend(Protocol):
    """Contract for vector + FTS storage backends.

    Implementations must be safe for the following call sequence per document::

        backend.add_chunks(chunks, embeddings, namespace=..., ...)
        backend.register_document(document_name, content_hash, ...)

    ``chunk_id`` is the idempotency key — ``add_chunks`` must silently ignore
    duplicate chunk_ids (``ON CONFLICT DO NOTHING`` semantics).
    """

    # Raw backend connection handle and FTS-dirty flag. The graph/search/ingest
    # layers reach into these for DuckDB-specific maintenance; they are part of
    # the implemented contract even though they are private by name. Typed as Any
    # because the underlying handle (a DuckDB connection) ships no type stubs.
    _conn: Any
    _fts_dirty: bool

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: Any,  # noqa: ANN401  # np.ndarray shape (n, embedding_dim)
        namespace: str | None = None,
        source_id: str | None = None,
        domain_id: str | None = None,
        session_fingerprint: str | None = None,
    ) -> None: ...

    def register_document(
        self,
        document_name: str,
        content_hash: str,
        source_uri: str = "",
        chunk_count: int = 0,
    ) -> None: ...

    def delete_by_document(self, document_name: str, *, gc_entities: bool = True) -> int:
        """Delete a document's chunks, its registry row, and every derived row.

        Implementations must remove the ``documents`` registry row along with
        the chunks. A surviving registry row makes the next
        :func:`~chonk.storage.sync_document` report "skipped" for a document
        that is no longer indexed — silent, permanent data loss.

        ``gc_entities=False`` defers the orphaned-entity sweep so batch callers
        can run it once at the end.
        """
        ...

    def gc_orphaned_entities(self) -> int: ...

    def chunk_ids_for_document(self, document_name: str) -> list[str]: ...

    def clear(self) -> None: ...

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: Any,  # noqa: ANN401  # np.ndarray shape (dim,) or (1, dim)
        limit: int = 5,
        query_text: str | None = None,  # None → pure vector; str → hybrid RRF
        include_breadcrumbs: bool = True,
        namespaces: list[str] | None = None,
        chunk_types: list[str] | None = None,
        domain_ids: list[str] | None = None,
        session_fingerprint: str | None = None,
        exclude_chunk_types: list[str] | None = None,
        entity_types: list[str] | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]: ...  # (chunk_id, score, DocumentChunk)

    def get_all_chunks(self) -> list[DocumentChunk]: ...  # used by graph builder

    # ------------------------------------------------------------------
    # Document registry
    # ------------------------------------------------------------------

    def get_document_hash(self, document_name: str) -> str | None: ...

    def list_documents(
        self,
    ) -> list[
        dict[str, Any]
    ]: ...  # keys: document_name, content_hash, source_uri, indexed_at, chunk_count

    def count(self) -> int: ...

    # ------------------------------------------------------------------
    # Lifecycle / optimisation hints
    # ------------------------------------------------------------------

    def rebuild_fts_index(self) -> None: ...

    # Backends with live FTS indexes (PG tsvector) implement as no-op.

    def preload_embeddings(self) -> None: ...

    # Backends with index-backed ANN (pgvector HNSW) implement as no-op.
