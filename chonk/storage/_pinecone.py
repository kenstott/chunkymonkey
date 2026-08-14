# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: f1a2b3c4-d5e6-7f8a-9b0c-d1e2f3a4b5c6
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Pinecone vector backend for chonk.

Architecture: Pinecone stores vectors + chunk metadata for ANN search.
A DuckDB sidecar (``catalog_path``) stores all catalog tables that ``Store``
accesses via ``_conn`` (namespaces, domains, sources, community_cache, etc.)
and a ``pinecone_ids`` shadow table used by ``compact()`` to identify orphaned
vectors (Pinecone has no list-all-IDs API).

The DuckDB catalog is the authority on which chunks are live. Searches join
against the catalog, so chunks deleted via ``Store.delete_domain`` are
transparently excluded even before Pinecone is cleaned up.

Call ``compact()`` to remove orphaned Pinecone vectors.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import DocumentChunk

from . import _cascade

logger = logging.getLogger(__name__)

_MISSING_DEPS_MSG = (
    "pinecone is required for PineconeVectorBackend. "
    "Install it with: pip install chonk-rag[pinecone]"
)

_CATALOG_DDL = [
    # Chunk metadata (no embedding column — vectors live in Pinecone)
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
        session_fingerprint TEXT,
        entity_types        VARCHAR[]
    )
    """,
    "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS source_detail TEXT",
    "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS source_id TEXT",
    "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS domain_id TEXT",
    "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS session_fingerprint TEXT",
    # Shadow table — all chunk_ids ever upserted to Pinecone.
    # compact() = pinecone_ids - live embeddings = orphans to delete.
    """
    CREATE TABLE IF NOT EXISTS pinecone_ids (
        chunk_id TEXT PRIMARY KEY
    )
    """,
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
        PRIMARY KEY (alias, namespace, entity_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS svo_triples (
        id              TEXT PRIMARY KEY,
        subject_id      TEXT NOT NULL,
        verb            TEXT NOT NULL,
        object_id       TEXT NOT NULL,
        chunk_id        TEXT,
        namespace       TEXT NOT NULL DEFAULT 'global',
        confidence      REAL NOT NULL DEFAULT 1.0,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        import pinecone as _pc  # noqa: F401

        _ = _pc
    except ImportError as exc:
        raise ImportError(_MISSING_DEPS_MSG) from exc


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
    exclude_chunk_types: list[str] | None = None,
    entity_types: list[str] | None = None,
) -> dict[str, Any] | None:
    """Build a Pinecone metadata filter dict (MongoDB-style)."""
    clauses: list[dict[str, Any]] = []
    if namespaces:
        clauses.append({"namespace": {"$in": namespaces}})
    if chunk_types:
        clauses.append({"chunk_type": {"$in": chunk_types}})
    if exclude_chunk_types:
        clauses.append({"chunk_type": {"$nin": exclude_chunk_types}})
    if entity_types:
        # metadata value is a list; $in matches when any element overlaps
        clauses.append({"entity_types": {"$in": entity_types}})
    if domain_ids:
        clauses.append({"domain_id": {"$in": domain_ids}})
    if session_fingerprint is not None:
        clauses.append({"session_fingerprint": {"$eq": session_fingerprint}})
    if not clauses:
        return None
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


class PineconeVectorBackend:
    """Vector operations backed by Pinecone (ANN) + DuckDB catalog (metadata).

    Pinecone stores vectors and chunk metadata for fast ANN search.
    A DuckDB sidecar (``catalog_path``) stores chunk metadata and all catalog
    tables that ``Store`` accesses via ``_conn``.

    The DuckDB catalog is the authoritative list of live chunks. Search results
    are joined against the catalog, so orphaned Pinecone vectors are silently
    excluded. Call ``compact()`` to remove them when convenient.

    Args:
        api_key: Pinecone API key.
        index_name: Pinecone index name (created automatically if absent).
        embedding_dim: Embedding vector dimension. Must match your model.
        catalog_path: Path to the DuckDB catalog file, or ``":memory:"``.
        cloud: Cloud provider for serverless spec (default: ``"aws"``).
        region: Cloud region for serverless spec (default: ``"us-east-1"``).
    """

    _fts_dirty: bool
    _global_attached: bool

    def __init__(
        self,
        api_key: str,
        index_name: str = "chonk",
        embedding_dim: int = 1024,
        catalog_path: str = ":memory:",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        _require_deps()
        self._index_name = index_name
        self._embedding_dim = embedding_dim
        self._catalog_path = catalog_path
        self._fts_dirty = False
        self._global_attached = False
        self._cloud = cloud
        self._region = region

        self._catalog = self._open_catalog(catalog_path)
        self._pc, self._index = self._connect_pinecone(
            api_key, index_name, embedding_dim, cloud, region
        )
        self._load_fts_ext()

    # ------------------------------------------------------------------
    # _conn — DuckDB-compatible adapter (used by Store catalog methods)
    # ------------------------------------------------------------------

    @property
    def _conn(self) -> Any:  # noqa: ANN401
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
                pass
        return conn

    def _connect_pinecone(
        self,
        api_key: str,
        index_name: str,
        embedding_dim: int,
        cloud: str,
        region: str,
    ) -> tuple[Any, Any]:
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=api_key)
        existing = {i.name for i in pc.list_indexes()}
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            logger.debug("Created Pinecone index %r (dim=%d)", index_name, embedding_dim)
        return pc, pc.Index(index_name)

    def _load_fts_ext(self) -> None:
        try:
            self._catalog.execute("LOAD fts")
        except Exception:
            logger.debug("DuckDB fts extension unavailable — BM25 hybrid search disabled")

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
        """Upsert chunks into Pinecone and the DuckDB catalog (idempotent)."""
        if not chunks:
            return

        import numpy as np

        vectors: list[dict[str, Any]] = []
        catalog_rows: list[tuple[Any, ...]] = []
        chunk_ids: list[str] = []

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

            # Pinecone metadata values must be str/int/float/bool/list[str]
            metadata: dict[str, Any] = {
                "chunk_id": chunk_id,
                "document_name": chunk.document_name,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "chunk_type": chunk_type,
            }
            if section_str:
                metadata["section"] = section_str
            if breadcrumb:
                metadata["breadcrumb"] = breadcrumb
            if namespace:
                metadata["namespace"] = namespace
            if source_id:
                metadata["source_id"] = source_id
            if domain_id:
                metadata["domain_id"] = domain_id
            if session_fingerprint:
                metadata["session_fingerprint"] = session_fingerprint
            if source_detail_str:
                metadata["source_detail"] = source_detail_str
            src_off = getattr(chunk, "source_offset", None)
            src_len = getattr(chunk, "source_length", None)
            if src_off is not None:
                metadata["source_offset"] = src_off
            if src_len is not None:
                metadata["source_length"] = src_len

            vectors.append({"id": chunk_id, "values": vec, "metadata": metadata})
            chunk_ids.append(chunk_id)
            catalog_rows.append(
                (
                    chunk_id,
                    chunk.document_name,
                    section_str,
                    chunk.chunk_index,
                    chunk.content,
                    breadcrumb,
                    chunk_type,
                    src_off,
                    src_len,
                    namespace,
                    source_detail_str,
                    source_id,
                    domain_id,
                    session_fingerprint,
                )
            )

        # Upsert to Pinecone in batches of 100 (serverless recommendation)
        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            self._index.upsert(vectors=vectors[start : start + batch_size])

        # Insert to catalog and shadow table (idempotent)
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
        for cid in chunk_ids:
            self._catalog.execute(
                "INSERT INTO pinecone_ids (chunk_id) VALUES (?) ON CONFLICT DO NOTHING",
                [cid],
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
        exclude_chunk_types: list[str] | None = None,
        entity_types: list[str] | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Hybrid search: Pinecone ANN + optional BM25 on the DuckDB catalog, merged via RRF.

        When ``query_text`` is provided and the DuckDB fts extension is available,
        runs BM25 on the catalog sidecar and merges with Pinecone ANN results using
        Reciprocal Rank Fusion.
        """
        import numpy as np

        from ..models import DocumentChunk

        query_vec = np.array(query_embedding, dtype="float32").flatten().tolist()
        pine_filter = _build_filter(
            namespaces,
            chunk_types,
            domain_ids,
            session_fingerprint,
            exclude_chunk_types,
            entity_types,
        )

        fetch_limit = limit * 4
        query_kwargs: dict[str, Any] = {
            "vector": query_vec,
            "top_k": fetch_limit,
            "include_metadata": True,
        }
        if pine_filter is not None:
            query_kwargs["filter"] = pine_filter

        response = self._index.query(**query_kwargs)
        matches = response.matches

        if not matches:
            return []

        # Join against catalog to exclude orphaned Pinecone vectors
        candidate_ids = [m.id for m in matches]
        placeholders = ", ".join("?" * len(candidate_ids))
        live_ids: set[str] = {
            r[0]
            for r in self._catalog.execute(
                f"SELECT chunk_id FROM embeddings WHERE chunk_id IN ({placeholders})",
                candidate_ids,
            ).fetchall()
        }

        vector_results: list[tuple[str, float, DocumentChunk]] = []
        for match in matches:
            if match.id not in live_ids:
                continue
            meta = match.metadata or {}
            content = meta.get("content", "")
            breadcrumb = meta.get("breadcrumb")
            displayed = (
                f"{breadcrumb}\n\n{content}" if include_breadcrumbs and breadcrumb else content
            )

            chunk = DocumentChunk(
                document_name=meta.get("document_name", ""),
                content=displayed,
                section=_deserialize_section(meta.get("section")),
                chunk_index=meta.get("chunk_index", 0),
                source_offset=meta.get("source_offset"),
                source_length=meta.get("source_length"),
                breadcrumb=breadcrumb,
                chunk_type=meta.get("chunk_type", "document"),
                source_detail=json.loads(meta["source_detail"])
                if meta.get("source_detail")
                else None,
            )
            vector_results.append((match.id, float(match.score), chunk))

        if query_text:
            bm25_results = self._bm25_search(
                query_text,
                limit=fetch_limit,
                namespaces=namespaces,
                chunk_types=chunk_types,
                exclude_chunk_types=exclude_chunk_types,
                entity_types=entity_types,
                domain_ids=domain_ids,
                session_fingerprint=session_fingerprint,
            )
            if bm25_results:
                return self._rrf_merge(vector_results, bm25_results)[:limit]

        return vector_results[:limit]

    def set_chunk_entity_types(self, mapping: dict[str, list[str]]) -> int:
        """Denormalise entity types onto the catalog rows and Pinecone metadata.

        ``mapping`` is ``{chunk_id: [entity_type, ...]}``.
        """
        if not mapping:
            return 0
        self._catalog.executemany(
            "UPDATE embeddings SET entity_types = ? WHERE chunk_id = ?",
            [[sorted(set(t)), cid] for cid, t in mapping.items()],
        )
        for chunk_id, types in mapping.items():
            self._index.update(id=chunk_id, set_metadata={"entity_types": sorted(set(types))})
        return len(mapping)

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
        exclude_chunk_types: list[str] | None = None,
        entity_types: list[str] | None = None,
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
            if exclude_chunk_types is not None:
                phs = ", ".join("?" * len(exclude_chunk_types))
                clauses.append(f"e.chunk_type NOT IN ({phs})")
                params.extend(exclude_chunk_types)
            if entity_types is not None:
                clauses.append("list_has_any(e.entity_types, ?)")
                params.append(list(entity_types))
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
        """Delete a document from Pinecone and everything keyed to it in the catalog.

        Cascades to ``chunk_entities``, ``chunk_clusters``, ``svo_triples``, and
        the ``documents`` registry row. Leaving the registry row behind would make
        the next :func:`~chonk.storage.sync_document` report "skipped" for a
        document that is no longer indexed.
        """
        rows = self._catalog.execute(
            "SELECT chunk_id FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchall()
        chunk_ids = [r[0] for r in rows]
        count = len(chunk_ids)

        if count:
            # Pinecone delete by IDs (batches of 1000)
            batch_size = 1000
            for start in range(0, len(chunk_ids), batch_size):
                self._index.delete(ids=chunk_ids[start : start + batch_size])
            _cascade.delete_chunk_dependents(self._run, self._table_exists, chunk_ids)
            self._catalog.execute("DELETE FROM embeddings WHERE document_name = ?", [document_name])
            self._catalog.execute(
                f"DELETE FROM pinecone_ids WHERE chunk_id IN ({', '.join('?' * len(chunk_ids))})",
                chunk_ids,
            )

        self._catalog.execute(
            "DELETE FROM documents WHERE document_name = ?",
            [document_name],
        )
        if count and gc_entities:
            self.gc_orphaned_entities()

        return count

    def clear(self) -> None:
        """Delete all chunks and everything derived from them.

        Leaves the namespace/domain/source registries intact — those describe
        where content comes from, not the content itself.
        """
        from pinecone import ServerlessSpec

        self._pc.delete_index(self._index_name)
        self._pc.create_index(
            name=self._index_name,
            dimension=self._embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=self._cloud, region=self._region),
        )
        self._index = self._pc.Index(self._index_name)
        self._catalog.execute("DELETE FROM embeddings")
        self._catalog.execute("DELETE FROM pinecone_ids")
        _cascade.clear_derived(self._run, self._table_exists)

    # ------------------------------------------------------------------
    # Count / get_all_chunks
    # ------------------------------------------------------------------

    def count(self) -> int:
        row = self._catalog.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return row[0] if row else 0

    def get_all_chunks(self) -> list[DocumentChunk]:
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
        """Remove Pinecone vectors whose catalog rows have been deleted.

        Uses the ``pinecone_ids`` shadow table (all ever-inserted IDs) minus
        the live ``embeddings`` catalog to find orphans, then deletes them
        from Pinecone. Returns the count of vectors deleted.
        """
        rows = self._catalog.execute(
            """
            SELECT p.chunk_id FROM pinecone_ids p
            LEFT JOIN embeddings e ON p.chunk_id = e.chunk_id
            WHERE e.chunk_id IS NULL
            """
        ).fetchall()
        orphan_ids = [r[0] for r in rows]

        if orphan_ids:
            batch_size = 1000
            for start in range(0, len(orphan_ids), batch_size):
                self._index.delete(ids=orphan_ids[start : start + batch_size])
            phs = ", ".join("?" * len(orphan_ids))
            self._catalog.execute(f"DELETE FROM pinecone_ids WHERE chunk_id IN ({phs})", orphan_ids)
            logger.info("compact(): removed %d orphaned Pinecone vectors", len(orphan_ids))

        return len(orphan_ids)

    def preload_embeddings(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._catalog.close()

    def __enter__(self) -> PineconeVectorBackend:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
