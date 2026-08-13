# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: b9e1f2a3-4c5d-6e7f-8a9b-0c1d2e3f4a5b
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""PostgreSQL + pgvector backend for chonk."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from ..graph._svo import SVOTriple
    from ..models import DocumentChunk

from . import _cascade

logger = logging.getLogger(__name__)

_MISSING_DEPS_MSG = (
    "psycopg2 and pgvector are required for PgVectorBackend. "
    "Install them with: pip install chonk[pgvector]"
)


def _require_deps() -> None:
    try:
        import pgvector  # type: ignore[import-untyped]  # noqa: F401
        import psycopg2  # type: ignore[import-untyped]  # noqa: F401
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


def _translate_sql(sql: str) -> str:
    """Translate DuckDB ``?`` placeholders to psycopg2 ``%s``."""
    return sql.replace("?", "%s")


class _PgResult:
    """DuckDB cursor-compatible result wrapper for psycopg2 results."""

    __slots__ = ("_rows",)

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def fetchall(self) -> list[Any]:
        return self._rows

    def fetchone(self) -> Any:  # noqa: ANN401
        return self._rows[0] if self._rows else None

    def __iter__(self) -> Iterator[Any]:
        return iter(self._rows)

    def __getitem__(self, idx: int) -> Any:  # noqa: ANN401
        return self._rows[idx]


class _PsycopgAdapter:
    """Thin psycopg2 wrapper presenting a DuckDB-compatible ``.execute()`` API.

    Translates ``?`` placeholders to ``%s``, executes against the underlying
    psycopg2 connection, and returns :class:`_PgResult` objects so callers
    can chain ``.fetchall()`` / ``.fetchone()`` in the same style as DuckDB.

    Each ``execute()`` call auto-commits (matches DuckDB's default behaviour).
    """

    def __init__(self, pgconn: Any) -> None:  # noqa: ANN401
        self._pgconn = pgconn

    def execute(self, sql: str, params: list[Any] | None = None) -> _PgResult:
        pg_sql = _translate_sql(sql)
        param_list = list(params) if params is not None else []
        with self._pgconn.cursor() as cur:
            cur.execute(pg_sql, param_list)
            rows = cur.fetchall() if cur.description else []
        self._pgconn.commit()
        return _PgResult(rows)

    def executemany(self, sql: str, params_list: list[list[Any]]) -> None:
        pg_sql = _translate_sql(sql)
        with self._pgconn.cursor() as cur:
            for row in params_list:
                cur.execute(pg_sql, list(row))
        self._pgconn.commit()


class PgVectorBackend:
    """Vector operations backed by PostgreSQL + pgvector (HNSW cosine index).

    Args:
        dsn: PostgreSQL DSN string, e.g. ``"postgresql://user:pass@host/db"``.
        embedding_dim: Embedding vector dimension. Must match your model.
        table: Table name to use (default: ``"embeddings"``).
    """

    def __init__(
        self,
        dsn: str,
        embedding_dim: int = 1024,
        table: str = "embeddings",
    ) -> None:
        _require_deps()
        self._dsn = dsn
        self._embedding_dim = embedding_dim
        self._table = table
        self._docs_table = "documents"
        self._pgconn = self._connect()
        self._global_attached = False
        self._fts_dirty = False  # tsvector index is live — no manual rebuild needed
        self._init_schema()

    # ------------------------------------------------------------------
    # DuckDB-compatibility shims
    # ------------------------------------------------------------------

    @property
    def _conn(self) -> _PsycopgAdapter:
        """DuckDB-compatible adapter over the psycopg2 connection.

        Allows Store catalog methods written for DuckDB to work
        transparently with the PG backend via placeholder translation.
        """
        return _PsycopgAdapter(self._pgconn)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> Any:  # noqa: ANN401
        import psycopg2
        from pgvector.psycopg2 import register_vector  # type: ignore[import-untyped]

        conn = psycopg2.connect(self._dsn)
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
        return conn

    def _ensure_connection(self) -> None:
        import psycopg2

        try:
            with self._pgconn.cursor() as cur:
                cur.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            self._pgconn = self._connect()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        t = self._table
        dt = self._docs_table
        with self._pgconn.cursor() as cur:
            # Vector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Embeddings table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {t} (
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
                    embedding           vector({self._embedding_dim}),
                    fts_vec             tsvector GENERATED ALWAYS AS
                                        (to_tsvector('english', content)) STORED
                )
            """)
            # Additive migrations (safe to re-run)
            for col in ("source_id", "domain_id", "session_fingerprint"):
                cur.execute(f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS {col} TEXT")

            # HNSW cosine index — works on empty tables (unlike IVFFlat)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {t}_embedding_hnsw_idx
                ON {t} USING hnsw (embedding vector_cosine_ops)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {t}_fts_idx ON {t} USING gin(fts_vec)
            """)

            # Documents registry
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {dt} (
                    document_name TEXT PRIMARY KEY,
                    content_hash  TEXT NOT NULL,
                    source_uri    TEXT NOT NULL DEFAULT '',
                    indexed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    chunk_count   INTEGER NOT NULL DEFAULT 0
                )
            """)

            # Catalog: namespaces
            cur.execute("""
                CREATE TABLE IF NOT EXISTS namespaces (
                    namespace_id TEXT PRIMARY KEY,
                    owner        TEXT,
                    description  TEXT,
                    created_at   TIMESTAMPTZ DEFAULT now(),
                    updated_at   TIMESTAMPTZ DEFAULT now()
                )
            """)

            # Catalog: domains
            cur.execute("""
                CREATE TABLE IF NOT EXISTS domains (
                    domain_id    TEXT PRIMARY KEY,
                    namespace_id TEXT,
                    name         TEXT,
                    description  TEXT,
                    parent_id    TEXT,
                    created_at   TIMESTAMPTZ DEFAULT now(),
                    updated_at   TIMESTAMPTZ DEFAULT now()
                )
            """)

            # Catalog: sources
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    source_id    TEXT PRIMARY KEY,
                    domain_id    TEXT,
                    type         TEXT,
                    uri          TEXT,
                    config       JSONB,
                    last_crawled TIMESTAMPTZ
                )
            """)

            # Community cache
            cur.execute("""
                CREATE TABLE IF NOT EXISTS community_cache (
                    fingerprint TEXT PRIMARY KEY,
                    domain_ids  JSONB,
                    chunk_count INTEGER,
                    created_at  TIMESTAMPTZ DEFAULT now()
                )
            """)

            # Namespace build log
            cur.execute("""
                CREATE TABLE IF NOT EXISTS namespace_build_log (
                    namespace_id       TEXT PRIMARY KEY,
                    chunks_built_at    TIMESTAMPTZ,
                    ner_built_at       TIMESTAMPTZ,
                    svo_built_at       TIMESTAMPTZ,
                    community_built_at TIMESTAMPTZ
                )
            """)

            # Entities vocabulary
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id           TEXT PRIMARY KEY,
                    name         TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    entity_type  TEXT NOT NULL DEFAULT 'concept',
                    description  TEXT,
                    created_at   TIMESTAMPTZ DEFAULT now()
                )
            """)

            # Chunk ↔ entity associations
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunk_entities (
                    chunk_id       TEXT NOT NULL,
                    entity_id      TEXT NOT NULL,
                    frequency      INTEGER NOT NULL DEFAULT 1,
                    positions_json TEXT NOT NULL DEFAULT '[]',
                    score          REAL NOT NULL DEFAULT 0.0,
                    namespace      TEXT,
                    PRIMARY KEY (chunk_id, entity_id)
                )
            """)

            # Entity aliases
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    alias      TEXT NOT NULL,
                    entity_id  TEXT NOT NULL,
                    namespace  TEXT NOT NULL DEFAULT 'global',
                    source     TEXT NOT NULL DEFAULT 'llm',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (alias, namespace)
                )
            """)

            # SVO triples
            cur.execute("""
                CREATE TABLE IF NOT EXISTS svo_triples (
                    chunk_id   TEXT,
                    subject_id TEXT NOT NULL,
                    verb       TEXT NOT NULL,
                    object_id  TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    namespace  TEXT
                )
            """)

            # NER cache
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ner_cache (
                    config_fingerprint TEXT PRIMARY KEY,
                    chunk_count        INTEGER NOT NULL,
                    created_at         TIMESTAMPTZ DEFAULT now()
                )
            """)

            # Chunk clusters (Louvain community detection)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunk_clusters (
                    chunk_id   TEXT NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    namespace  TEXT NOT NULL DEFAULT 'global',
                    PRIMARY KEY (chunk_id, namespace, cluster_id)
                )
            """)

            # Context graph edges
            cur.execute("""
                CREATE TABLE IF NOT EXISTS context_graph_edges (
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    namespace        TEXT NOT NULL DEFAULT 'global',
                    weight           REAL NOT NULL,
                    svo_signal       REAL NOT NULL DEFAULT 0.0,
                    cooccur_signal   REAL NOT NULL DEFAULT 0.0,
                    cluster_signal   REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (source_entity_id, target_entity_id, namespace)
                )
            """)

            # Context graph cache
            cur.execute("""
                CREATE TABLE IF NOT EXISTS context_graph_cache (
                    namespace         TEXT PRIMARY KEY,
                    chunk_fingerprint TEXT NOT NULL,
                    entity_count      INTEGER NOT NULL,
                    edge_count        INTEGER NOT NULL,
                    created_at        TIMESTAMPTZ DEFAULT now()
                )
            """)

            # ── Ingest queue (horizontal scale) ──────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingest_queue (
                    id           BIGSERIAL PRIMARY KEY,
                    source_uri   TEXT NOT NULL,
                    namespace    TEXT NOT NULL,
                    content_hash TEXT,
                    status       TEXT DEFAULT 'pending',
                    worker_id    TEXT,
                    leased_at    TIMESTAMPTZ,
                    created_at   TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS ingest_queue_status_idx
                ON ingest_queue (status)
                WHERE status = 'pending'
            """)

            # ── Coordinator control ───────────────────────────────────────────
            # coordinator sets key='workers_paused', value='1' during graph build
            cur.execute("""
                CREATE TABLE IF NOT EXISTS control (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

        self._pgconn.commit()

    # ------------------------------------------------------------------
    # Chunk ID
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_chunk_id(document_name: str, chunk_index: int, content: str) -> str:
        content_hash = hashlib.sha256(
            f"{document_name}:{chunk_index}:{content[:100]}".encode()
        ).hexdigest()[:16]
        return f"{document_name}_{chunk_index}_{content_hash}"

    # ------------------------------------------------------------------
    # Add chunks
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        namespace: str | None = None,
        source_id: str | None = None,
        domain_id: str | None = None,
        session_fingerprint: str | None = None,
    ) -> None:
        """Insert chunks with embeddings (ON CONFLICT DO NOTHING — idempotent)."""
        if not chunks:
            return

        import numpy as np
        from pgvector.psycopg2 import register_vector  # type: ignore[import-untyped]

        self._ensure_connection()
        register_vector(self._pgconn)

        records = []
        for i, chunk in enumerate(chunks):
            embed_content = (
                chunk.embedding_content
                if hasattr(chunk, "embedding_content") and chunk.embedding_content
                else chunk.content
            )
            chunk_id = self._generate_chunk_id(
                chunk.document_name, chunk.chunk_index, embed_content
            )
            _ct = getattr(chunk, "chunk_type", "document")
            chunk_type = getattr(_ct, "value", None) if hasattr(_ct, "value") else _ct or "document"
            raw_section = getattr(chunk, "section", []) or []
            section_str = json.dumps(raw_section) if isinstance(raw_section, list) else raw_section
            raw_detail = getattr(chunk, "source_detail", None)
            source_detail_str = json.dumps(raw_detail) if raw_detail is not None else None

            vec = np.array(embeddings[i], dtype="float32")
            records.append(
                (
                    chunk_id,
                    chunk.document_name,
                    section_str,
                    chunk.chunk_index,
                    chunk.content,
                    getattr(chunk, "breadcrumb", None),
                    chunk_type,
                    getattr(chunk, "source_offset", None),
                    getattr(chunk, "source_length", None),
                    namespace,
                    source_detail_str,
                    source_id,
                    domain_id,
                    session_fingerprint,
                    vec,
                )
            )

        t = self._table
        with self._pgconn.cursor() as cur:
            for rec in records:
                cur.execute(
                    f"""
                    INSERT INTO {t}
                        (chunk_id, document_name, section, chunk_index, content,
                         breadcrumb, chunk_type, source_offset, source_length,
                         namespace, source_detail, source_id, domain_id,
                         session_fingerprint, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO NOTHING
                    """,
                    rec,
                )
        self._pgconn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        query_text: str | None = None,
        include_breadcrumbs: bool = True,
        namespaces: list[str] | None = None,
        chunk_types: list[str] | None = None,
        domain_ids: list[str] | None = None,
        session_fingerprint: str | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Hybrid (BM25 + vector) or pure vector search via pgvector HNSW index.

        When ``query_text`` is provided a reciprocal-rank fusion of the
        tsvector BM25 results and vector similarity results is returned.
        When omitted, pure cosine-similarity ordering is used.

        Returns:
            List of (chunk_id, score, DocumentChunk).
        """
        import numpy as np
        from pgvector.psycopg2 import register_vector  # type: ignore[import-untyped]

        from ..models import DocumentChunk

        self._ensure_connection()
        register_vector(self._pgconn)

        query_vec = np.array(query_embedding, dtype="float32").flatten()

        t = self._table
        clauses: list[str] = []
        filter_params: list[Any] = []

        if namespaces is not None:
            clauses.append("namespace = ANY(%s)")
            filter_params.append(namespaces)
        if chunk_types is not None:
            clauses.append("chunk_type = ANY(%s)")
            filter_params.append(chunk_types)
        if domain_ids is not None:
            clauses.append("domain_id = ANY(%s)")
            filter_params.append(domain_ids)
        if session_fingerprint is not None:
            clauses.append("session_fingerprint = %s")
            filter_params.append(session_fingerprint)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        if query_text is not None:
            # Hybrid: BM25 + vector RRF
            rows = self._search_hybrid(query_vec, query_text, where, filter_params, limit, t)
        else:
            # Pure vector
            all_params = [query_vec] + filter_params + [query_vec, limit]
            with self._pgconn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        chunk_id, document_name, section, chunk_index, content,
                        breadcrumb, chunk_type, source_offset, source_length,
                        source_detail,
                        1.0 - (embedding <=> %s::vector) AS similarity
                    FROM {t}
                    {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    all_params,
                )
                rows = cur.fetchall()

        results = []
        for row in rows:
            (
                chunk_id,
                doc_name,
                section,
                chunk_idx,
                content,
                breadcrumb,
                chunk_type_str,
                source_offset,
                source_length,
                source_detail_str,
                similarity,
            ) = row
            displayed = (
                f"{breadcrumb}\n\n{content}" if include_breadcrumbs and breadcrumb else content
            )
            chunk = DocumentChunk(
                document_name=doc_name,
                content=displayed,
                section=_deserialize_section(section),
                chunk_index=chunk_idx,
                source_offset=source_offset,
                source_length=source_length,
                breadcrumb=breadcrumb,
                chunk_type=chunk_type_str or "document",
                source_detail=json.loads(source_detail_str) if source_detail_str else None,
            )
            results.append((chunk_id, float(similarity), chunk))

        return results

    def _search_hybrid(
        self,
        query_vec: np.ndarray,
        query_text: str,
        where: str,
        filter_params: list[Any],
        limit: int,
        t: str,
    ) -> list[tuple[Any, ...]]:
        """RRF merge of BM25 and vector results."""
        rrf_k = 60
        candidate_limit = limit * 4

        # Vector ranking
        vec_params = [query_vec] + filter_params + [query_vec, candidate_limit]
        with self._pgconn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
                FROM {t} {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                vec_params,
            )
            vec_ranks = {row[0]: row[1] for row in cur.fetchall()}

        # BM25 ranking
        safe_query = query_text.replace("'", "''")
        bm25_where = (
            where + " AND " if where else "WHERE "
        ) + f"fts_vec @@ plainto_tsquery('english', '{safe_query}')"
        bm25_params = filter_params + [candidate_limit]
        with self._pgconn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_id,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank(
                               fts_vec, plainto_tsquery('english', '{safe_query}')
                           ) DESC
                       ) AS rank
                FROM {t} {bm25_where}
                LIMIT %s
                """,
                bm25_params,
            )
            bm25_ranks = {row[0]: row[1] for row in cur.fetchall()}

        # RRF merge
        all_ids = set(vec_ranks) | set(bm25_ranks)
        scores = {
            cid: 1.0 / (rrf_k + vec_ranks.get(cid, candidate_limit + rrf_k))
            + 1.0 / (rrf_k + bm25_ranks.get(cid, candidate_limit + rrf_k))
            for cid in all_ids
        }
        top_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]

        if not top_ids:
            return []

        placeholders = ",".join(["%s"] * len(top_ids))
        with self._pgconn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_id, document_name, section, chunk_index, content,
                       breadcrumb, chunk_type, source_offset, source_length,
                       source_detail, 0.0 AS similarity
                FROM {t}
                WHERE chunk_id IN ({placeholders})
                """,
                top_ids,
            )
            rows_by_id = {row[0]: row for row in cur.fetchall()}

        return [rows_by_id[cid] for cid in top_ids if cid in rows_by_id]

    # ------------------------------------------------------------------
    # Delete / clear
    # ------------------------------------------------------------------

    def _run(self, sql: str, params: list[Any]) -> None:
        with self._pgconn.cursor() as cur:
            cur.execute(sql, params)

    def _scalar(self, sql: str, params: list[Any]) -> Any:  # noqa: ANN401
        with self._pgconn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return row[0] if row else None

    def _table_exists(self, name: str) -> bool:
        return bool(
            self._scalar(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s",
                [name],
            )
        )

    def chunk_ids_for_document(self, document_name: str) -> list[str]:
        """Return every chunk_id currently stored for *document_name*."""
        self._ensure_connection()
        with self._pgconn.cursor() as cur:
            cur.execute(
                f"SELECT chunk_id FROM {self._table} WHERE document_name = %s",  # noqa: S608
                [document_name],
            )
            return [r[0] for r in cur.fetchall()]

    def gc_orphaned_entities(self) -> int:
        """Delete entities nothing references any more."""
        self._ensure_connection()
        deleted = _cascade.gc_orphaned_entities(self._run, self._scalar, self._table_exists)
        self._pgconn.commit()
        return deleted

    def delete_by_document(self, document_name: str, *, gc_entities: bool = True) -> int:
        """Delete all chunks for a document and everything keyed to them.

        Cascades to ``chunk_entities``, ``chunk_clusters``, ``svo_triples``, and
        the ``documents`` registry row. Leaving the registry row behind would make
        the next :func:`~chonk.storage.sync_document` report "skipped" for a
        document that is no longer indexed.

        Returns:
            Number of chunks deleted.
        """
        self._ensure_connection()
        t = self._table
        dt = self._docs_table
        with self._pgconn.cursor() as cur:
            cur.execute(
                f"SELECT chunk_id FROM {t} WHERE document_name = %s",  # noqa: S608
                [document_name],
            )
            chunk_ids = [r[0] for r in cur.fetchall()]

        _cascade.delete_chunk_dependents(self._run, self._table_exists, chunk_ids, placeholder="%s")

        with self._pgconn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {t} WHERE document_name = %s",  # noqa: S608
                [document_name],
            )
            cur.execute(
                f"DELETE FROM {dt} WHERE document_name = %s",  # noqa: S608
                [document_name],
            )

        if chunk_ids and gc_entities:
            _cascade.gc_orphaned_entities(self._run, self._scalar, self._table_exists)

        self._pgconn.commit()
        return len(chunk_ids)

    def clear(self) -> None:
        """Delete all chunks from the table."""
        self._ensure_connection()
        with self._pgconn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table}")
        self._pgconn.commit()

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of stored chunks."""
        self._ensure_connection()
        with self._pgconn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            result = cur.fetchone()
        return result[0] if result else 0

    # ------------------------------------------------------------------
    # get_all_chunks
    # ------------------------------------------------------------------

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Return all stored chunks as DocumentChunk objects (server-side cursor)."""
        from ..models import DocumentChunk

        self._ensure_connection()
        chunks = []
        t = self._table
        with self._pgconn.cursor("get_all_chunks") as cur:
            cur.execute(
                f"""
                SELECT document_name, content, section, chunk_index,
                       source_offset, source_length, breadcrumb, source_detail, chunk_type
                FROM {t}
                ORDER BY document_name, chunk_index
                """
            )
            for row in cur:
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
        self._ensure_connection()
        dt = self._docs_table
        with self._pgconn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {dt} (document_name, content_hash, source_uri, indexed_at, chunk_count)
                VALUES (%s, %s, %s, now(), %s)
                ON CONFLICT (document_name) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    source_uri   = EXCLUDED.source_uri,
                    indexed_at   = EXCLUDED.indexed_at,
                    chunk_count  = EXCLUDED.chunk_count
                """,
                [document_name, content_hash, source_uri, chunk_count],
            )
        self._pgconn.commit()

    def get_document_hash(self, document_name: str) -> str | None:
        self._ensure_connection()
        dt = self._docs_table
        with self._pgconn.cursor() as cur:
            cur.execute(
                f"SELECT content_hash FROM {dt} WHERE document_name = %s",
                [document_name],
            )
            row = cur.fetchone()
        return row[0] if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        self._ensure_connection()
        dt = self._docs_table
        with self._pgconn.cursor() as cur:
            cur.execute(
                f"SELECT document_name, content_hash, source_uri, indexed_at, chunk_count "
                f"FROM {dt} ORDER BY document_name"
            )
            rows = cur.fetchall()
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
    # SVO triples
    # ------------------------------------------------------------------

    def store_svo_triples(self, triples: list[SVOTriple], namespace: str | None = None) -> int:
        """Insert SVO triples into the svo_triples table. Returns count inserted.

        Args:
            triples: List of :class:`~chonk.graph.SVOTriple` objects.
            namespace: Override namespace; if None uses the triple's source chunk namespace.
        """
        if not triples:
            return 0
        self._ensure_connection()
        rows = []
        for t in triples:
            ns = namespace
            if ns is None and t.source_chunk_id:
                with self._pgconn.cursor() as cur:
                    cur.execute(
                        f"SELECT namespace FROM {self._table} WHERE chunk_id = %s",
                        [t.source_chunk_id],
                    )
                    row = cur.fetchone()
                    ns = row[0] if row else None
            rows.append(
                (
                    t.source_chunk_id,
                    t.subject_id,
                    t.verb,
                    t.object_id,
                    float(t.confidence),
                    ns,
                )
            )
        with self._pgconn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO svo_triples
                        (chunk_id, subject_id, verb, object_id, confidence, namespace)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    row,
                )
        self._pgconn.commit()
        return len(rows)

    def get_svo_triples(self, namespace: str | None = None) -> list[tuple[Any, ...]]:
        """Return all svo_triples rows, optionally filtered by namespace."""
        self._ensure_connection()
        with self._pgconn.cursor() as cur:
            if namespace is not None:
                cur.execute(
                    "SELECT chunk_id, subject_id, verb, object_id, confidence, namespace "
                    "FROM svo_triples WHERE namespace = %s ORDER BY subject_id, verb",
                    [namespace],
                )
            else:
                cur.execute(
                    "SELECT chunk_id, subject_id, verb, object_id, confidence, namespace "
                    "FROM svo_triples ORDER BY subject_id, verb"
                )
            return cur.fetchall()

    # ------------------------------------------------------------------
    # Lifecycle hints (no-ops for PG)
    # ------------------------------------------------------------------

    def rebuild_fts_index(self) -> None:
        pass  # tsvector index is live — no manual rebuild needed

    def preload_embeddings(self) -> None:
        pass  # pgvector HNSW index handles ANN — no RAM preload needed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the PostgreSQL connection."""
        if self._pgconn and not self._pgconn.closed:
            self._pgconn.close()

    def __enter__(self) -> PgVectorBackend:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
