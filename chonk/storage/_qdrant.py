# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f9a
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Qdrant vector backend for chonk.

Architecture: Qdrant stores vectors + chunk payload for ANN search.
A DuckDB sidecar (``catalog_path``) stores chunk metadata and all catalog
tables (namespaces, domains, sources, community_cache, entities, etc.)
that ``Store`` accesses via ``_conn``.

The DuckDB catalog is the authority on which chunks are live. Searches join
against the catalog, so chunks deleted via ``Store.delete_domain`` or
``Store.invalidate_community_cache`` (which operate on the catalog directly)
are transparently excluded from results even before Qdrant is cleaned up.

Call ``compact()`` to remove orphaned Qdrant points whose catalog rows are gone.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import DocumentChunk

from . import _cascade

logger = logging.getLogger(__name__)

_MISSING_DEPS_MSG = (
    "qdrant-client is required for QdrantVectorBackend. "
    "Install it with: pip install chonk-rag[qdrant]"
)

# Fixed UUID namespace for deterministic chunk → Qdrant point ID mapping.
_CHUNK_UUID_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

_CATALOG_DDL = [
    # Chunk metadata (no embedding column — vectors live in Qdrant)
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id            TEXT PRIMARY KEY,
        document_name       TEXT NOT NULL,
        section             TEXT,
        chunk_index         INTEGER NOT NULL DEFAULT 0,
        content             TEXT NOT NULL,
        breadcrumb          TEXT,
        chunk_type          TEXT NOT NULL DEFAULT 'document',
        source_offset       INTEGER,
        source_length       INTEGER,
        namespace           TEXT,
        source_detail       TEXT,
        source_id           TEXT,
        domain_id           TEXT,
        session_fingerprint TEXT
    )
    """,
    "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS source_detail TEXT",
    "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS source_id TEXT",
    "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS domain_id TEXT",
    "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS session_fingerprint TEXT",
    """
    CREATE TABLE IF NOT EXISTS documents (
        document_name TEXT PRIMARY KEY,
        content_hash  TEXT NOT NULL,
        source_uri    TEXT NOT NULL DEFAULT '',
        indexed_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        chunk_count   INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS community_cache (
        fingerprint   TEXT PRIMARY KEY,
        domain_ids    JSON,
        chunk_count   INTEGER,
        created_at    TIMESTAMP DEFAULT current_timestamp
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS namespaces (
        namespace_id  TEXT PRIMARY KEY,
        owner         TEXT,
        description   TEXT,
        created_at    TIMESTAMP DEFAULT current_timestamp,
        updated_at    TIMESTAMP DEFAULT current_timestamp
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS domains (
        domain_id     TEXT PRIMARY KEY,
        namespace_id  TEXT,
        name          TEXT,
        description   TEXT,
        parent_id     TEXT,
        created_at    TIMESTAMP DEFAULT current_timestamp,
        updated_at    TIMESTAMP DEFAULT current_timestamp
    )
    """,
    "ALTER TABLE domains ADD COLUMN IF NOT EXISTS parent_id TEXT",
    """
    CREATE TABLE IF NOT EXISTS sources (
        source_id     TEXT PRIMARY KEY,
        domain_id     TEXT,
        type          TEXT,
        uri           TEXT,
        config        JSON,
        last_crawled  TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entities (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        display_name TEXT NOT NULL,
        entity_type  TEXT NOT NULL DEFAULT 'concept',
        description  TEXT,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "ALTER TABLE entities ADD COLUMN IF NOT EXISTS description TEXT",
    """
    CREATE TABLE IF NOT EXISTS chunk_entities (
        chunk_id        TEXT NOT NULL,
        entity_id       TEXT NOT NULL,
        frequency       INTEGER NOT NULL DEFAULT 1,
        positions_json  TEXT NOT NULL DEFAULT '[]',
        score           REAL NOT NULL DEFAULT 0.0,
        namespace       TEXT,
        PRIMARY KEY (chunk_id, entity_id)
    )
    """,
    "ALTER TABLE chunk_entities ADD COLUMN IF NOT EXISTS namespace TEXT",
    """
    CREATE TABLE IF NOT EXISTS entity_aliases (
        alias       TEXT    NOT NULL,
        entity_id   TEXT    NOT NULL,
        namespace   TEXT    NOT NULL DEFAULT 'global',
        source      TEXT    NOT NULL DEFAULT 'llm',
        created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (alias, namespace)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS svo_triples (
        chunk_id   TEXT,
        subject_id TEXT NOT NULL,
        verb       TEXT NOT NULL,
        object_id  TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 1.0,
        namespace  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ner_cache (
        config_fingerprint  TEXT PRIMARY KEY,
        chunk_count         INTEGER NOT NULL,
        created_at          TIMESTAMP DEFAULT current_timestamp
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunk_clusters (
        chunk_id    TEXT NOT NULL,
        cluster_id  INTEGER NOT NULL,
        namespace   TEXT NOT NULL DEFAULT 'global',
        PRIMARY KEY (chunk_id, namespace, cluster_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS context_graph_edges (
        source_entity_id  TEXT NOT NULL,
        target_entity_id  TEXT NOT NULL,
        namespace         TEXT NOT NULL DEFAULT 'global',
        weight            REAL NOT NULL,
        svo_signal        REAL NOT NULL DEFAULT 0.0,
        cooccur_signal    REAL NOT NULL DEFAULT 0.0,
        cluster_signal    REAL NOT NULL DEFAULT 0.0,
        PRIMARY KEY (source_entity_id, target_entity_id, namespace)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS context_graph_cache (
        namespace         TEXT PRIMARY KEY,
        chunk_fingerprint TEXT NOT NULL,
        entity_count      INTEGER NOT NULL,
        edge_count        INTEGER NOT NULL,
        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS namespace_build_log (
        namespace_id       TEXT PRIMARY KEY,
        chunks_built_at    TIMESTAMP,
        ner_built_at       TIMESTAMP,
        svo_built_at       TIMESTAMP,
        community_built_at TIMESTAMP
    )
    """,
]


def _require_deps() -> None:
    try:
        import qdrant_client as _qc  # noqa: F401

        _ = _qc
    except ImportError as exc:
        raise ImportError(_MISSING_DEPS_MSG) from exc


def _chunk_uuid(chunk_id: str) -> str:
    """Deterministic UUID from chunk_id for use as Qdrant point ID."""
    return str(uuid.uuid5(_CHUNK_UUID_NS, chunk_id))


def _deserialize_section(value: Any) -> list[str]:  # noqa: ANN401
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, TypeError):
        pass
    return [value]


class QdrantVectorBackend:
    """Vector operations backed by Qdrant (ANN) + DuckDB catalog (metadata).

    Qdrant stores vectors and chunk payload for fast ANN search.
    A DuckDB sidecar (``catalog_path``) stores chunk metadata and all catalog
    tables that ``Store`` accesses via ``_conn`` — namespaces, domains, sources,
    community_cache, entities, SVO triples, etc.

    The DuckDB catalog is the authoritative list of live chunks. Search results
    are joined against the catalog, so orphaned Qdrant points (e.g. from
    ``Store.delete_domain``) are silently excluded. Call ``compact()`` to remove
    them from Qdrant when convenient.

    Args:
        url: Qdrant server URL, e.g. ``"http://localhost:6333"``.
        collection: Qdrant collection name (created automatically if absent).
        embedding_dim: Embedding vector dimension. Must match your model.
        catalog_path: Path to the DuckDB catalog file, or ``":memory:"``.
            Use a file path for persistence across restarts.
        api_key: Optional Qdrant Cloud API key.
        prefer_grpc: Use gRPC transport (faster for bulk ops, requires port 6334).
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "chonk",
        embedding_dim: int = 1024,
        catalog_path: str = ":memory:",
        api_key: str | None = None,
        prefer_grpc: bool = False,
    ) -> None:
        _require_deps()
        self._url = url
        self._collection = collection
        self._embedding_dim = embedding_dim
        self._catalog_path = catalog_path
        self._fts_dirty = False
        self._global_attached = False

        self._catalog = self._open_catalog(catalog_path)
        self._client = self._connect_qdrant(url, api_key, prefer_grpc)
        self._init_collection()
        self._load_fts_ext()

    # ------------------------------------------------------------------
    # _conn — DuckDB-compatible adapter (used by Store catalog methods)
    # ------------------------------------------------------------------

    @property
    def _conn(self) -> Any:  # noqa: ANN401
        """DuckDB catalog connection. Store catalog methods use this for SQL."""
        return self._catalog

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _open_catalog(self, path: str) -> Any:  # noqa: ANN401
        import duckdb

        conn = duckdb.connect(path)
        for ddl in _CATALOG_DDL:
            try:
                conn.execute(ddl)
            except Exception:
                pass  # ADD COLUMN IF NOT EXISTS may fail on older DuckDB — skip
        return conn

    def _connect_qdrant(self, url: str, api_key: str | None, prefer_grpc: bool) -> Any:  # noqa: ANN401
        from qdrant_client import QdrantClient

        return QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)

    def _load_fts_ext(self) -> None:
        try:
            self._catalog.execute("LOAD fts")
        except Exception:
            logger.debug("DuckDB fts extension unavailable — BM25 hybrid search disabled")

    def _init_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.debug(
                "Created Qdrant collection %r (dim=%d)", self._collection, self._embedding_dim
            )

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
    ) -> None:
        """Upsert chunks into Qdrant and the DuckDB catalog (idempotent)."""
        if not chunks:
            return

        import numpy as np
        from qdrant_client.models import PointStruct

        points: list[PointStruct] = []
        catalog_rows: list[tuple[Any, ...]] = []

        for i, chunk in enumerate(chunks):
            embed_content = (
                chunk.embedding_content
                if hasattr(chunk, "embedding_content") and chunk.embedding_content
                else chunk.content
            )
            chunk_id = _make_chunk_id(chunk.document_name, chunk.chunk_index, embed_content)

            raw_section = getattr(chunk, "section", []) or []
            section_str = json.dumps(raw_section) if isinstance(raw_section, list) else raw_section
            raw_detail = getattr(chunk, "source_detail", None)
            source_detail_str = json.dumps(raw_detail) if raw_detail is not None else None
            _ct = getattr(chunk, "chunk_type", "document")
            chunk_type = getattr(_ct, "value", None) if hasattr(_ct, "value") else _ct or "document"
            breadcrumb = getattr(chunk, "breadcrumb", None)

            vec = np.array(embeddings[i], dtype="float32").tolist()

            points.append(
                PointStruct(
                    id=_chunk_uuid(chunk_id),
                    vector=vec,
                    payload={
                        "chunk_id": chunk_id,
                        "document_name": chunk.document_name,
                        "section": section_str,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "breadcrumb": breadcrumb,
                        "chunk_type": chunk_type,
                        "source_offset": getattr(chunk, "source_offset", None),
                        "source_length": getattr(chunk, "source_length", None),
                        "namespace": namespace,
                        "source_detail": source_detail_str,
                        "source_id": source_id,
                        "domain_id": domain_id,
                        "session_fingerprint": session_fingerprint,
                    },
                )
            )

            catalog_rows.append(
                (
                    chunk_id,
                    chunk.document_name,
                    section_str,
                    chunk.chunk_index,
                    chunk.content,
                    breadcrumb,
                    chunk_type,
                    getattr(chunk, "source_offset", None),
                    getattr(chunk, "source_length", None),
                    namespace,
                    source_detail_str,
                    source_id,
                    domain_id,
                    session_fingerprint,
                )
            )

        # Upsert to Qdrant in batches of 256
        batch_size = 256
        for start in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection,
                points=points[start : start + batch_size],
            )

        # Insert to catalog (ON CONFLICT DO NOTHING — idempotent)
        for row in catalog_rows:
            self._catalog.execute(
                """
                INSERT INTO embeddings
                    (chunk_id, document_name, section, chunk_index, content,
                     breadcrumb, chunk_type, source_offset, source_length,
                     namespace, source_detail, source_id, domain_id, session_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                list(row),
            )
        self._fts_dirty = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: Any,  # noqa: ANN401  # np.ndarray shape (dim,) or (1, dim)
        limit: int = 5,
        query_text: str | None = None,
        include_breadcrumbs: bool = True,
        namespaces: list[str] | None = None,
        chunk_types: list[str] | None = None,
        domain_ids: list[str] | None = None,
        session_fingerprint: str | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Hybrid search: Qdrant ANN + optional BM25 on the DuckDB catalog, merged via RRF.

        When ``query_text`` is provided and the DuckDB fts extension is available,
        runs BM25 on the catalog sidecar and merges with Qdrant ANN results using
        Reciprocal Rank Fusion. Falls back to pure ANN when ``query_text`` is None
        or the fts extension is unavailable.
        """
        import numpy as np

        from ..models import DocumentChunk

        query_vec = np.array(query_embedding, dtype="float32").flatten().tolist()
        qdrant_filter = _build_filter(namespaces, chunk_types, domain_ids, session_fingerprint)

        # Over-fetch so the catalog join can still return `limit` results
        # after filtering orphaned points.
        fetch_limit = limit * 4
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            query_filter=qdrant_filter,
            limit=fetch_limit,
            with_payload=True,
        )
        hits = response.points

        if not hits:
            return []

        # Join against catalog to exclude orphaned Qdrant points
        candidate_ids = [
            h.payload["chunk_id"] for h in hits if h.payload and "chunk_id" in h.payload
        ]
        if not candidate_ids:
            return []

        placeholders = ", ".join("?" * len(candidate_ids))
        live_ids: set[str] = {
            r[0]
            for r in self._catalog.execute(
                f"SELECT chunk_id FROM embeddings WHERE chunk_id IN ({placeholders})",
                candidate_ids,
            ).fetchall()
        }

        vector_results: list[tuple[str, float, DocumentChunk]] = []
        for hit in hits:
            if not hit.payload:
                continue
            chunk_id = hit.payload.get("chunk_id")
            if chunk_id not in live_ids:
                continue

            content = hit.payload.get("content", "")
            breadcrumb = hit.payload.get("breadcrumb")
            displayed = (
                f"{breadcrumb}\n\n{content}" if include_breadcrumbs and breadcrumb else content
            )

            chunk = DocumentChunk(
                document_name=hit.payload.get("document_name", ""),
                content=displayed,
                section=_deserialize_section(hit.payload.get("section")),
                chunk_index=hit.payload.get("chunk_index", 0),
                source_offset=hit.payload.get("source_offset"),
                source_length=hit.payload.get("source_length"),
                breadcrumb=breadcrumb,
                chunk_type=hit.payload.get("chunk_type", "document"),
                source_detail=json.loads(hit.payload["source_detail"])
                if hit.payload.get("source_detail")
                else None,
            )
            vector_results.append((chunk_id, float(hit.score), chunk))

        if query_text:
            bm25_results = self._bm25_search(
                query_text,
                limit=fetch_limit,
                namespaces=namespaces,
                chunk_types=chunk_types,
                domain_ids=domain_ids,
                session_fingerprint=session_fingerprint,
            )
            if bm25_results:
                return self._rrf_merge(vector_results, bm25_results)[:limit]

        return vector_results[:limit]

    def rebuild_fts_index(self) -> None:
        if not self._fts_dirty:
            return
        try:
            self._catalog.execute(
                "PRAGMA create_fts_index('embeddings', 'chunk_id', 'content', overwrite=1)"
            )
            self._fts_dirty = False
        except Exception as e:
            logger.warning("FTS index build failed: %s", e)

    def _bm25_search(
        self,
        query_text: str,
        limit: int = 5,
        namespaces: list[str] | None = None,
        chunk_types: list[str] | None = None,
        domain_ids: list[str] | None = None,
        session_fingerprint: str | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        from ..models import DocumentChunk

        try:
            self.rebuild_fts_index()
            clauses = ["bm25_score IS NOT NULL"]
            params: list[Any] = [query_text]
            for col, values in (
                ("namespace", namespaces),
                ("chunk_type", chunk_types),
                ("domain_id", domain_ids),
            ):
                if values is not None:
                    phs = ", ".join("?" * len(values))
                    clauses.append(f"e.{col} IN ({phs})")
                    params.extend(values)
            if session_fingerprint is not None:
                clauses.append("e.session_fingerprint = ?")
                params.append(session_fingerprint)
            params.append(limit)
            rows = self._catalog.execute(
                f"""
                SELECT e.chunk_id, e.document_name, e.section, e.chunk_index,
                       e.content, e.chunk_type, e.source_offset, e.source_length,
                       e.source_detail,
                       fts_main_embeddings.match_bm25(
                           e.chunk_id, ?, fields:='content') AS bm25_score
                FROM embeddings e
                WHERE {" AND ".join(clauses)}
                ORDER BY bm25_score DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
            return []

        results: list[tuple[str, float, DocumentChunk]] = []
        for row in rows:
            (
                chunk_id,
                doc_name,
                section,
                chunk_idx,
                content,
                chunk_type,
                src_off,
                src_len,
                src_detail_str,
                score,
            ) = row
            results.append(
                (
                    chunk_id,
                    float(score),
                    DocumentChunk(
                        document_name=doc_name,
                        content=content,
                        section=_deserialize_section(section),
                        chunk_index=chunk_idx,
                        source_offset=src_off,
                        source_length=src_len,
                        source_detail=json.loads(src_detail_str) if src_detail_str else None,
                        chunk_type=chunk_type or "document",
                    ),
                )
            )
        return results

    @staticmethod
    def _rrf_merge(
        vector_results: list[tuple[str, float, DocumentChunk]],
        bm25_results: list[tuple[str, float, DocumentChunk]],
        k: int = 60,
    ) -> list[tuple[str, float, DocumentChunk]]:
        scores: dict[str, float] = {}
        chunks: dict[str, DocumentChunk] = {}
        for rank, (chunk_id, _score, chunk) in enumerate(vector_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk_id] = chunk
        for rank, (chunk_id, _score, chunk) in enumerate(bm25_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(chunk_id, chunk)
        return [
            (cid, score, chunks[cid])
            for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]

    # ------------------------------------------------------------------
    # Delete / clear
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cascade helpers
    # ------------------------------------------------------------------

    def _run(self, sql: str, params: list[Any]) -> None:
        self._catalog.execute(sql, params)

    def _scalar(self, sql: str, params: list[Any]) -> Any:  # noqa: ANN401
        rows = self._catalog.execute(sql, params).fetchall()
        return rows[0][0] if rows else None

    def _table_exists(self, name: str) -> bool:
        rows = self._catalog.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [name],
        ).fetchall()
        return bool(rows and rows[0][0] > 0)

    def chunk_ids_for_document(self, document_name: str) -> list[str]:
        """Return every chunk_id currently stored for *document_name*."""
        rows = self._catalog.execute(
            "SELECT chunk_id FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchall()
        return [r[0] for r in rows]

    def gc_orphaned_entities(self) -> int:
        """Delete catalog entities nothing references any more."""
        return _cascade.gc_orphaned_entities(self._run, self._scalar, self._table_exists)

    def delete_by_document(self, document_name: str, *, gc_entities: bool = True) -> int:
        """Delete a document from Qdrant and everything keyed to it in the catalog.

        Cascades to ``chunk_entities``, ``chunk_clusters``, ``svo_triples``, and
        the ``documents`` registry row. Leaving the registry row behind would make
        the next :func:`~chonk.storage.sync_document` report "skipped" for a
        document that is no longer indexed.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        rows = self._catalog.execute(
            "SELECT chunk_id FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchall()
        chunk_ids = [r[0] for r in rows]
        count = len(chunk_ids)

        if count:
            self._client.delete(
                collection_name=self._collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="document_name", match=MatchValue(value=document_name))
                    ]
                ),
            )
            _cascade.delete_chunk_dependents(self._run, self._table_exists, chunk_ids)
            self._catalog.execute(
                "DELETE FROM embeddings WHERE document_name = ?",
                [document_name],
            )

        self._catalog.execute(
            "DELETE FROM documents WHERE document_name = ?",
            [document_name],
        )
        if count and gc_entities:
            self.gc_orphaned_entities()

        return count

    def clear(self) -> None:
        """Delete all chunks from Qdrant and the catalog."""
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._embedding_dim, distance=Distance.COSINE),
        )
        self._catalog.execute("DELETE FROM embeddings")

    # ------------------------------------------------------------------
    # Count / get_all_chunks
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of live chunks (from catalog, not Qdrant)."""
        row = self._catalog.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return row[0] if row else 0

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Return all live chunks from the catalog (no vector fetch needed)."""
        from ..models import DocumentChunk

        rows = self._catalog.execute(
            """
            SELECT document_name, content, section, chunk_index,
                   source_offset, source_length, breadcrumb, source_detail, chunk_type
            FROM embeddings
            ORDER BY document_name, chunk_index
            """
        ).fetchall()

        chunks = []
        for row in rows:
            (
                doc_name,
                content,
                section,
                chunk_idx,
                src_off,
                src_len,
                breadcrumb,
                src_detail,
                chunk_type,
            ) = row
            chunks.append(
                DocumentChunk(
                    document_name=doc_name,
                    content=content,
                    section=_deserialize_section(section),
                    chunk_index=chunk_idx,
                    source_offset=src_off,
                    source_length=src_len,
                    breadcrumb=breadcrumb,
                    embedding_content=f"{breadcrumb}\n\n{content}" if breadcrumb else content,
                    source_detail=json.loads(src_detail) if src_detail else None,
                    chunk_type=chunk_type or "document",
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Document registry
    # ------------------------------------------------------------------

    def register_document(
        self,
        document_name: str,
        content_hash: str,
        source_uri: str = "",
        chunk_count: int = 0,
    ) -> None:
        self._catalog.execute(
            """
            INSERT INTO documents (document_name, content_hash, source_uri, chunk_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (document_name) DO UPDATE SET
                content_hash = excluded.content_hash,
                source_uri   = excluded.source_uri,
                indexed_at   = now(),
                chunk_count  = excluded.chunk_count
            """,
            [document_name, content_hash, source_uri, chunk_count],
        )

    def get_document_hash(self, document_name: str) -> str | None:
        row = self._catalog.execute(
            "SELECT content_hash FROM documents WHERE document_name = ?",
            [document_name],
        ).fetchone()
        return row[0] if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self._catalog.execute(
            "SELECT document_name, content_hash, source_uri, indexed_at, chunk_count "
            "FROM documents ORDER BY document_name"
        ).fetchall()
        return [
            {
                "document_name": r[0],
                "content_hash": r[1],
                "source_uri": r[2],
                "indexed_at": r[3],
                "chunk_count": r[4],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def compact(self) -> int:
        """Remove Qdrant points whose catalog rows have been deleted.

        Call after ``Store.delete_domain`` or ``Store.invalidate_community_cache``
        to reclaim Qdrant storage. Returns the count of points deleted.

        Note: Searches already exclude orphaned points via the catalog join,
        so ``compact()`` is a storage-reclaim operation, not a correctness fix.
        """

        live_ids: set[str] = {
            r[0] for r in self._catalog.execute("SELECT chunk_id FROM embeddings").fetchall()
        }
        live_uuids = {_chunk_uuid(cid) for cid in live_ids}

        # Scroll all Qdrant point IDs
        orphan_uuids: list[str] = []
        offset = None
        while True:
            result, offset = self._client.scroll(
                collection_name=self._collection,
                limit=1000,
                with_payload=False,
                with_vectors=False,
                offset=offset,
            )
            for point in result:
                if str(point.id) not in live_uuids:
                    orphan_uuids.append(str(point.id))
            if offset is None:
                break

        if orphan_uuids:
            self._client.delete(
                collection_name=self._collection,
                points_selector=orphan_uuids,
            )
            logger.info("compact(): removed %d orphaned Qdrant points", len(orphan_uuids))

        return len(orphan_uuids)

    def preload_embeddings(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._catalog.close()
        self._client.close()

    def __enter__(self) -> QdrantVectorBackend:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_chunk_id(document_name: str, chunk_index: int, embed_content: str) -> str:
    import hashlib

    content_hash = hashlib.sha256(
        f"{document_name}:{chunk_index}:{embed_content[:100]}".encode()
    ).hexdigest()[:16]
    return f"{document_name}_{chunk_index}_{content_hash}"


def _build_filter(
    namespaces: list[str] | None,
    chunk_types: list[str] | None,
    domain_ids: list[str] | None,
    session_fingerprint: str | None,
) -> Any:  # noqa: ANN401
    """Build a Qdrant Filter from search parameters, or None if no filters."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    conditions = []
    if namespaces is not None:
        conditions.append(FieldCondition(key="namespace", match=MatchAny(any=namespaces)))
    if chunk_types is not None:
        conditions.append(FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types)))
    if domain_ids is not None:
        conditions.append(FieldCondition(key="domain_id", match=MatchAny(any=domain_ids)))
    if session_fingerprint is not None:
        conditions.append(
            FieldCondition(key="session_fingerprint", match=MatchValue(value=session_fingerprint))
        )

    return Filter(must=conditions) if conditions else None
