# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: e2f3a4b5-c6d7-8e9f-a0b1-c2d3e4f5a6b7
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Weaviate vector backend for chonk.

Architecture: Weaviate stores vectors + chunk payload for ANN and native BM25.
A DuckDB sidecar (``catalog_path``) stores all catalog tables that ``Store``
accesses via ``_conn`` (namespaces, domains, sources, community_cache, etc.).

Hybrid search uses Weaviate's own BM25 + ANN, merged via RRF — no DuckDB fts
extension is needed (though the sidecar BM25 fallback is also available).

The DuckDB catalog is the authority on which chunks are live. Search results
are joined against the catalog, so chunks deleted via ``Store.delete_domain``
are transparently excluded even before Weaviate is cleaned up.

Call ``compact()`` to remove orphaned Weaviate objects whose catalog rows
have been deleted.
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
    "weaviate-client>=4 is required for WeaviateVectorBackend. "
    "Install it with: pip install chonk-rag[weaviate]"
)

_COLLECTION_PROPERTIES = [
    # filterable text fields
    ("chunk_id", "TEXT", True, False),
    ("document_name", "TEXT", True, False),
    ("chunk_type", "TEXT", True, False),
    ("namespace", "TEXT", True, False),
    ("domain_id", "TEXT", True, False),
    ("session_fingerprint", "TEXT", True, False),
    # inverted-indexed (BM25) text fields
    ("content", "TEXT", False, True),
    ("breadcrumb", "TEXT", False, False),
    # metadata — stored only, no filter/BM25 index needed
    ("section", "TEXT", False, False),
    ("source_detail", "TEXT", False, False),
    ("source_id", "TEXT", False, False),
]

_CATALOG_DDL = [
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
    # Shadow table — all chunk_ids ever inserted to Weaviate.
    # compact() = weaviate_ids - live embeddings = orphans to delete.
    """
    CREATE TABLE IF NOT EXISTS weaviate_ids (
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
        import weaviate as _wv  # noqa: F401

        _ = _wv
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
) -> Any:  # noqa: ANN401
    """Build a Weaviate v4 Filter from search params, or None."""
    from weaviate.classes.query import Filter

    clauses = []
    if namespaces:
        clauses.append(Filter.by_property("namespace").contains_any(namespaces))
    if chunk_types:
        clauses.append(Filter.by_property("chunk_type").contains_any(chunk_types))
    if entity_types:
        clauses.append(Filter.by_property("entity_types").contains_any(entity_types))
    for excluded in exclude_chunk_types or []:
        # Weaviate has no not_contains_any; AND one not_equal per excluded type.
        clauses.append(Filter.by_property("chunk_type").not_equal(excluded))
    if domain_ids:
        clauses.append(Filter.by_property("domain_id").contains_any(domain_ids))
    if session_fingerprint is not None:
        clauses.append(Filter.by_property("session_fingerprint").equal(session_fingerprint))
    if not clauses:
        return None
    result = clauses[0]
    for c in clauses[1:]:
        result = result & c
    return result


class WeaviateVectorBackend:
    """Vector operations backed by Weaviate (ANN + native BM25) + DuckDB catalog.

    Weaviate stores vectors and chunk payload. Native BM25 runs inside Weaviate
    (no DuckDB fts extension required). Hybrid results are merged via RRF.
    A DuckDB sidecar holds all catalog tables used by ``Store`` via ``_conn``.

    Args:
        cluster_url: Weaviate Cloud cluster URL
            (e.g. ``"https://abc123.c0.us-east-1.aws.weaviate.cloud"``).
        api_key: Weaviate Cloud API key.
        collection: Weaviate collection name (created automatically if absent).
        embedding_dim: Embedding vector dimension. Must match your model.
        catalog_path: DuckDB catalog file path, or ``":memory:"``.
    """

    _fts_dirty: bool
    _global_attached: bool

    def __init__(
        self,
        cluster_url: str,
        api_key: str,
        collection: str = "Chonk",
        embedding_dim: int = 1024,
        catalog_path: str = ":memory:",
    ) -> None:
        _require_deps()
        self._collection_name = collection
        self._embedding_dim = embedding_dim
        self._fts_dirty = False
        self._global_attached = False

        self._catalog = self._open_catalog(catalog_path)
        self._client = self._connect(cluster_url, api_key)
        self._col = self._init_collection(collection)

    # ------------------------------------------------------------------
    # _conn — DuckDB-compatible adapter used by Store catalog methods
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

    def _connect(self, cluster_url: str, api_key: str) -> Any:  # noqa: ANN401
        import weaviate
        from weaviate.classes.init import Auth

        return weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=Auth.api_key(api_key),
        )

    def _init_collection(self, name: str) -> Any:  # noqa: ANN401
        from weaviate.classes.config import Configure, DataType, Property

        existing = self._client.collections.list_all()
        if name not in existing:
            props = []
            for pname, dtype, filterable, searchable in _COLLECTION_PROPERTIES:
                wv_dtype = DataType.TEXT if dtype == "TEXT" else DataType.INT
                props.append(
                    Property(
                        name=pname,
                        data_type=wv_dtype,
                        index_filterable=filterable,
                        index_searchable=searchable,
                    )
                )
            # Also store numeric fields
            props += [
                # Denormalised entity types; filterable so search(entity_types=[...])
                # can push the predicate server-side.
                Property(
                    name="entity_types",
                    data_type=DataType.TEXT_ARRAY,
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(name="chunk_index", data_type=DataType.INT, index_filterable=False),
                Property(name="source_offset", data_type=DataType.INT, index_filterable=False),
                Property(name="source_length", data_type=DataType.INT, index_filterable=False),
            ]
            self._client.collections.create(
                name=name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=props,
            )
            logger.debug("Created Weaviate collection %r (dim=%d)", name, self._embedding_dim)
        return self._client.collections.get(name)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: Any,  # noqa: ANN401
        namespace: str | None = None,
        source_id: str | None = None,
        domain_id: str | None = None,
        session_fingerprint: str | None = None,
    ) -> None:
        """Upsert chunks into Weaviate and the DuckDB catalog (idempotent)."""
        if not chunks:
            return

        import numpy as np
        from weaviate.classes.data import DataObject  # noqa: F401 — imported for runtime use

        objects: list[Any] = []
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
            src_off = getattr(chunk, "source_offset", None)
            src_len = getattr(chunk, "source_length", None)

            vec = np.array(embeddings[i], dtype="float32").tolist()

            props: dict[str, Any] = {
                "chunk_id": chunk_id,
                "document_name": chunk.document_name,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "chunk_type": chunk_type,
            }
            if section_str:
                props["section"] = section_str
            if breadcrumb:
                props["breadcrumb"] = breadcrumb
            if namespace:
                props["namespace"] = namespace
            if source_id:
                props["source_id"] = source_id
            if domain_id:
                props["domain_id"] = domain_id
            if session_fingerprint:
                props["session_fingerprint"] = session_fingerprint
            if source_detail_str:
                props["source_detail"] = source_detail_str
            if src_off is not None:
                props["source_offset"] = src_off
            if src_len is not None:
                props["source_length"] = src_len

            objects.append(DataObject(properties=props, vector=vec))
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

        # Insert to Weaviate in batches of 200
        batch_size = 200
        for start in range(0, len(objects), batch_size):
            self._col.data.insert_many(objects[start : start + batch_size])

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
                "INSERT INTO weaviate_ids (chunk_id) VALUES (?) ON CONFLICT DO NOTHING",
                [cid],
            )
        self._fts_dirty = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: Any,  # noqa: ANN401
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
        """Hybrid search: Weaviate ANN + BM25, merged via RRF.

        When ``query_text`` is provided, runs both ANN and native Weaviate BM25
        and merges via Reciprocal Rank Fusion. Falls back to pure ANN otherwise.
        """
        import numpy as np

        from ..models import DocumentChunk

        query_vec = np.array(query_embedding, dtype="float32").flatten().tolist()
        wv_filter = _build_filter(
            namespaces,
            chunk_types,
            domain_ids,
            session_fingerprint,
            exclude_chunk_types,
            entity_types,
        )
        fetch_limit = limit * 4

        from weaviate.classes.query import MetadataQuery

        # ANN search
        ann_kwargs: dict[str, Any] = {
            "limit": fetch_limit,
            "return_metadata": MetadataQuery(distance=True),
        }
        if wv_filter is not None:
            ann_kwargs["filters"] = wv_filter

        ann_resp = self._col.query.near_vector(query_vec, **ann_kwargs)

        # Build vector_results and get live IDs via catalog join
        candidate_ids = [
            o.properties.get("chunk_id") for o in ann_resp.objects if o.properties.get("chunk_id")
        ]

        live_ids: set[str] = set()
        if candidate_ids:
            phs = ", ".join("?" * len(candidate_ids))
            live_ids = {
                r[0]
                for r in self._catalog.execute(
                    f"SELECT chunk_id FROM embeddings WHERE chunk_id IN ({phs})",
                    candidate_ids,
                ).fetchall()
            }

        def _make_chunk(props: dict[str, Any], inc_bc: bool) -> DocumentChunk:
            content = props.get("content", "")
            breadcrumb = props.get("breadcrumb")
            displayed = f"{breadcrumb}\n\n{content}" if inc_bc and breadcrumb else content
            return DocumentChunk(
                document_name=props.get("document_name", ""),
                content=displayed,
                section=_deserialize_section(props.get("section")),
                chunk_index=props.get("chunk_index", 0),
                source_offset=props.get("source_offset"),
                source_length=props.get("source_length"),
                breadcrumb=breadcrumb,
                chunk_type=props.get("chunk_type", "document"),
                source_detail=json.loads(props["source_detail"])
                if props.get("source_detail")
                else None,
            )

        vector_results: list[tuple[str, float, DocumentChunk]] = []
        for obj in ann_resp.objects:
            cid = obj.properties.get("chunk_id")
            if cid not in live_ids:
                continue
            dist = obj.metadata.distance if obj.metadata else 1.0
            score = 1.0 - (dist or 0.0)
            vector_results.append((cid, score, _make_chunk(obj.properties, include_breadcrumbs)))

        if query_text:
            bm25_kwargs: dict[str, Any] = {"limit": fetch_limit}
            if wv_filter is not None:
                bm25_kwargs["filters"] = wv_filter

            bm25_resp = self._col.query.bm25(query_text, **bm25_kwargs)
            bm25_results: list[tuple[str, float, DocumentChunk]] = []
            for obj in bm25_resp.objects:
                cid = obj.properties.get("chunk_id")
                if cid and cid in live_ids:
                    bm25_results.append(
                        (cid, 1.0, _make_chunk(obj.properties, include_breadcrumbs))
                    )

            if bm25_results:
                return self._rrf_merge(vector_results, bm25_results)[:limit]

        return vector_results[:limit]

    @staticmethod
    def _rrf_merge(
        vector_results: list[tuple[str, float, DocumentChunk]],
        bm25_results: list[tuple[str, float, DocumentChunk]],
        k: int = 60,
    ) -> list[tuple[str, float, DocumentChunk]]:
        scores: dict[str, float] = {}
        chunks: dict[str, DocumentChunk] = {}
        for rank, (cid, _s, chunk) in enumerate(vector_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunks[cid] = chunk
        for rank, (cid, _s, chunk) in enumerate(bm25_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(cid, chunk)
        return [
            (cid, score, chunks[cid])
            for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]

    # Protocol stubs (BM25 runs natively in Weaviate — no local FTS index needed)
    def set_chunk_entity_types(self, mapping: dict[str, list[str]]) -> int:
        """Denormalise entity types onto the catalog rows and Weaviate objects.

        ``mapping`` is ``{chunk_id: [entity_type, ...]}``.
        """
        if not mapping:
            return 0
        self._catalog.executemany(
            "UPDATE embeddings SET entity_types = ? WHERE chunk_id = ?",
            [[sorted(set(t)), cid] for cid, t in mapping.items()],
        )
        from weaviate.classes.query import Filter

        for chunk_id, types in mapping.items():
            # Objects are inserted without an explicit UUID, so resolve by chunk_id.
            found = self._col.query.fetch_objects(
                filters=Filter.by_property("chunk_id").equal(chunk_id), limit=1
            )
            if not found.objects:
                raise RuntimeError(
                    f"cannot set entity_types: chunk_id {chunk_id!r} is absent from the "
                    f"Weaviate collection {self._col.name!r}"
                )
            self._col.data.update(
                uuid=found.objects[0].uuid,
                properties={"entity_types": sorted(set(types))},
            )
        return len(mapping)

    def rebuild_fts_index(self) -> None:
        pass

    def preload_embeddings(self) -> None:
        pass

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
        """Delete a document from Weaviate and everything keyed to it in the catalog.

        Cascades to ``chunk_entities``, ``chunk_clusters``, ``svo_triples``, and
        the ``documents`` registry row. Leaving the registry row behind would make
        the next :func:`~chonk.storage.sync_document` report "skipped" for a
        document that is no longer indexed.
        """
        from weaviate.classes.query import Filter

        rows = self._catalog.execute(
            "SELECT chunk_id FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchall()
        chunk_ids = [r[0] for r in rows]
        count = len(chunk_ids)

        if count:
            self._col.data.delete_many(
                where=Filter.by_property("document_name").equal(document_name)
            )
            _cascade.delete_chunk_dependents(self._run, self._table_exists, chunk_ids)
            self._catalog.execute("DELETE FROM embeddings WHERE document_name = ?", [document_name])
            phs = ", ".join("?" * len(chunk_ids))
            self._catalog.execute(f"DELETE FROM weaviate_ids WHERE chunk_id IN ({phs})", chunk_ids)

        self._catalog.execute(
            "DELETE FROM documents WHERE document_name = ?",
            [document_name],
        )
        if count and gc_entities:
            self.gc_orphaned_entities()

        return count

    def clear(self) -> None:
        self._client.collections.delete(self._collection_name)
        self._col = self._init_collection(self._collection_name)
        self._catalog.execute("DELETE FROM embeddings")
        self._catalog.execute("DELETE FROM weaviate_ids")

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
            doc_name, content, section, chunk_idx, src_off, src_len, bc, src_detail, chunk_type = (
                row
            )
            chunks.append(
                DocumentChunk(
                    document_name=doc_name,
                    content=content,
                    section=_deserialize_section(section),
                    chunk_index=chunk_idx,
                    source_offset=src_off,
                    source_length=src_len,
                    breadcrumb=bc,
                    embedding_content=f"{bc}\n\n{content}" if bc else content,
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
        """Remove Weaviate objects whose catalog rows have been deleted.

        Uses the ``weaviate_ids`` shadow table to identify orphans and deletes
        them from Weaviate in batches. Returns the count of objects deleted.
        """
        from weaviate.classes.query import Filter

        rows = self._catalog.execute(
            """
            SELECT w.chunk_id FROM weaviate_ids w
            LEFT JOIN embeddings e ON w.chunk_id = e.chunk_id
            WHERE e.chunk_id IS NULL
            """
        ).fetchall()
        orphan_ids = [r[0] for r in rows]

        if orphan_ids:
            # Weaviate filter: delete where chunk_id is in orphan_ids
            # Process in batches to avoid oversized filters
            batch_size = 100
            for start in range(0, len(orphan_ids), batch_size):
                batch = orphan_ids[start : start + batch_size]
                self._col.data.delete_many(where=Filter.by_property("chunk_id").contains_any(batch))
            phs = ", ".join("?" * len(orphan_ids))
            self._catalog.execute(f"DELETE FROM weaviate_ids WHERE chunk_id IN ({phs})", orphan_ids)
            logger.info("compact(): removed %d orphaned Weaviate objects", len(orphan_ids))

        return len(orphan_ids)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._catalog.close()
        self._client.close()

    def __enter__(self) -> WeaviateVectorBackend:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
