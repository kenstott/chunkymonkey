# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 3ce49853-1b78-46af-9a67-7fab26f10d28
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DuckDB VSS + FTS vector backend for chonk."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import duckdb  # noqa: F401 — availability checked at runtime

    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False

if TYPE_CHECKING:
    from ..models import DocumentChunk

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore

from . import _cascade
from ._protocol import VectorBackend

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a sync_document() or prune_documents() call.

    Attributes:
        action:               "added" | "updated" | "skipped" | "deleted"
        document_name:        Name of the document.
        content_hash:         SHA-256 hex of the raw bytes that were checked.
                              Empty for "deleted" — nothing was hashed.
        chunk_count:          Chunks in the new version (0 if skipped or deleted).
        previous_chunk_count: Chunks in the old version (0 if new/skipped).
    """

    action: str
    document_name: str
    content_hash: str = ""
    chunk_count: int = 0
    previous_chunk_count: int = 0


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


_MISSING_DEPS_MSG = (
    "duckdb and numpy are required for storage. Install them with: pip install chonk-rag[storage]"
)


def _require_deps() -> None:
    if not _DUCKDB_AVAILABLE or not _NUMPY_AVAILABLE:
        raise ImportError(_MISSING_DEPS_MSG)


class DuckDBVectorBackend:
    """Vector operations backed by DuckDB VSS (HNSW) + FTS extensions."""

    def __init__(self, db: Any, embedding_dim: int = 1024) -> None:  # noqa: ANN401
        _require_deps()
        self._db = db
        self._embedding_dim = embedding_dim
        # Read-only stores cannot create/rebuild the FTS index; the persisted
        # index from build time is queried as-is. Marking clean avoids a DROP
        # attempt that DuckDB rejects on a read-only database.
        self._fts_dirty = not getattr(db, "_read_only", False)
        self._global_attached: bool = False
        self._np_embeddings = None  # preloaded (n, dim) float32
        self._np_chunk_rows = None  # preloaded metadata rows
        self._init_schema()

    def preload_embeddings(self) -> None:
        """Load all embeddings into RAM for fast batched numpy search."""
        rows = self._conn.execute(
            f"""
            SELECT chunk_id, document_name, section, chunk_index,
                   content, breadcrumb, chunk_type, source_offset, source_length,
                   namespace, source_detail, source_id, domain_id, embedding
            FROM {self._read_table}
            """
        ).fetchall()
        if not rows:
            return
        self._np_chunk_rows = rows
        assert _np is not None, "numpy required for preload_embeddings"
        self._np_embeddings = _np.array([r[13] for r in rows], dtype="float32")

    @property
    def _conn(self) -> Any:  # noqa: ANN401
        return self._db.conn

    @property
    def _read_table(self) -> str:
        return "all_embeddings" if getattr(self, "_global_attached", False) else "embeddings"

    # ------------------------------------------------------------------
    # Schema init
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        from ._schema import VSS_INDEX_DDL, get_ddl

        if getattr(self._db, "_read_only", False):
            # Read-only indexes cannot be stamped, but a version mismatch is still
            # fatal — the chunk_ids and cache keys inside would be misread.
            self._verify_schema_version()
            return

        # Verify before any DDL runs so an incompatible index is left untouched.
        self._verify_schema_version()

        # Load extensions — INSTALL may fail when already installed or on read-only
        # filesystems; attempt LOAD regardless. If LOAD fails, vss/fts is unavailable
        # and callers must know — log WARNING and re-raise.
        for ext in ("vss", "fts"):
            try:
                self._conn.execute(f"INSTALL {ext}").fetchall()
            except Exception as e:
                logger.debug(f"Extension {ext} install skipped (may already be installed): {e}")
            try:
                self._conn.execute(f"LOAD {ext}").fetchall()
            except Exception as e:
                logger.warning(f"Extension {ext} LOAD failed — {ext} features unavailable: {e}")
                raise

        for ddl in get_ddl(self._embedding_dim):
            # All DDL uses IF NOT EXISTS — failure is not idempotent; re-raise.
            self._conn.execute(ddl).fetchall()

        self._stamp_schema_version()

        # Always drop and recreate HNSW index to ensure cosine metric.
        # In file-backed databases, HNSW requires hnsw_enable_experimental_persistence;
        # log at WARNING (not silent) but do not raise — vector search falls back to
        # sequential scan, which is functionally correct though slower.
        from ._schema import VSS_DROP_INDEX_DDL

        try:
            self._conn.execute(VSS_DROP_INDEX_DDL).fetchall()
            self._conn.execute(VSS_INDEX_DDL).fetchall()
            logger.debug("VSS HNSW cosine index ready")
        except Exception as e:
            logger.warning(
                f"VSS HNSW index creation skipped (vector search uses sequential scan): {e}"
            )

    # ------------------------------------------------------------------
    # Schema version
    # ------------------------------------------------------------------

    def _table_exists(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [name],
        ).fetchone()
        if row is None:
            raise RuntimeError("COUNT(*) returned no rows")
        return row[0] > 0

    def _stored_schema_version(self) -> int | None:
        """Version stamped on this index, or None if never stamped."""
        if not self._table_exists("schema_meta"):
            return None
        row = self._conn.execute("SELECT version FROM schema_meta WHERE id = 1").fetchone()
        return row[0] if row else None

    def _verify_schema_version(self) -> None:
        """Raise if this index was written by an incompatible schema version.

        An unstamped index is accepted only when it holds no chunks — that is a
        freshly created database, not a stale one.
        """
        from ._schema import SCHEMA_VERSION, SchemaVersionError

        stored = self._stored_schema_version()
        if stored == SCHEMA_VERSION:
            return

        if stored is None:
            if not self._table_exists("embeddings"):
                return
            row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            if row is None:
                raise RuntimeError("COUNT(*) returned no rows")
            if row[0] == 0:
                return

        found = "unstamped (pre-2)" if stored is None else str(stored)
        raise SchemaVersionError(
            f"Index schema version {found} is incompatible with this build of chonk "
            f"(expects version {SCHEMA_VERSION}). Chunk IDs and cache keys changed, and "
            f"no in-place migration is possible. Rebuild the index from source "
            f"(delete the database file, or re-run indexing with --force)."
        )

    def _stamp_schema_version(self) -> None:
        from ._schema import SCHEMA_VERSION

        self._conn.execute(
            """
            INSERT INTO schema_meta (id, version, updated_at) VALUES (1, ?, now())
            ON CONFLICT (id) DO UPDATE SET version = excluded.version,
                                           updated_at = excluded.updated_at
            """,
            [SCHEMA_VERSION],
        ).fetchall()

    # ------------------------------------------------------------------
    # Chunk ID
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_chunk_id(document_name: str, chunk_index: int, content: str) -> str:
        # Hashes the FULL content: a truncated hash collides for chunks that share
        # a prefix, which silently rebinds stale entity/cluster rows to new content
        # across an incremental update.
        content_hash = hashlib.sha256(
            f"{document_name}:{chunk_index}:{content}".encode()
        ).hexdigest()[:16]
        return f"{document_name}_{chunk_index}_{content_hash}"

    # ------------------------------------------------------------------
    # Add chunks
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
        """Insert chunks with embeddings into the embeddings table.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: np.ndarray of shape (n, embedding_dim).
            namespace: Optional partition key. None means no namespace filter at search time.
            source_id: Optional source registry ID for the originating source.
            domain_id: Optional domain registry ID (denormalization of source_id → domain).
            session_fingerprint: Optional fingerprint tagging community summary chunks.
        """
        if not chunks:
            return

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
            embedding = embeddings[i].tolist()
            _ct = getattr(chunk, "chunk_type", "document")
            chunk_type = getattr(_ct, "value", None) if hasattr(_ct, "value") else _ct or "document"
            raw_section = getattr(chunk, "section", []) or []
            section_str = json.dumps(raw_section) if isinstance(raw_section, list) else raw_section
            raw_detail = getattr(chunk, "source_detail", None)
            source_detail_str = json.dumps(raw_detail) if raw_detail is not None else None
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
                    embedding,
                )
            )

        self._conn.executemany(
            """
            INSERT INTO embeddings
                (chunk_id, document_name, section, chunk_index, content,
                 breadcrumb, chunk_type, source_offset, source_length, namespace,
                 source_detail, source_id, domain_id, session_fingerprint, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            records,
        )
        self._fts_dirty = True

    # ------------------------------------------------------------------
    # FTS / BM25
    # ------------------------------------------------------------------

    def rebuild_fts_index(self) -> None:
        """Rebuild the BM25 full-text search index."""
        self._rebuild_fts_index()

    def _rebuild_fts_index(self) -> None:
        if not self._fts_dirty:
            return
        try:
            _count_row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            if _count_row is None:
                raise RuntimeError("COUNT(*) returned no rows")
            count = _count_row[0]
            if count == 0:
                self._fts_dirty = False
                return
            self._conn.execute(
                "PRAGMA create_fts_index('embeddings', 'chunk_id', 'content', "
                "stemmer='none', stopwords='none', ignore='[^a-zA-Z0-9]', lower=1, overwrite=1)"
            ).fetchall()
            self._fts_dirty = False
        except Exception as e:
            # Keep _fts_dirty=True so the next call retries.
            # Attempt ROLLBACK but do not suppress the original error.
            logger.warning(f"FTS index rebuild failed: {e}")
            try:
                self._conn.execute("ROLLBACK").fetchall()
            except Exception:
                pass
            raise

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
            self._rebuild_fts_index()
            # Apply the same filters as the vector branch so hybrid search does not
            # leak rows from outside the requested namespaces/domains/chunk_types.
            clauses = ["bm25_score IS NOT NULL"]
            params: list[Any] = [query_text]
            for col, values in (
                ("namespace", namespaces),
                ("chunk_type", chunk_types),
                ("domain_id", domain_ids),
            ):
                if values is not None:
                    placeholders = ", ".join("?" * len(values))
                    clauses.append(f"e.{col} IN ({placeholders})")
                    params.extend(values)
            if session_fingerprint is not None:
                clauses.append("e.session_fingerprint = ?")
                params.append(session_fingerprint)
            params.append(limit)
            rows = self._conn.execute(
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

            results = []
            for row in rows:
                (
                    chunk_id,
                    doc_name,
                    section,
                    chunk_idx,
                    content,
                    _chunk_type,
                    source_offset,
                    source_length,
                    source_detail_str,
                    score,
                ) = row
                chunk = DocumentChunk(
                    document_name=doc_name,
                    content=content,
                    section=_deserialize_section(section),
                    chunk_index=chunk_idx,
                    source_offset=source_offset,
                    source_length=source_length,
                    source_detail=json.loads(source_detail_str) if source_detail_str else None,
                )
                results.append((chunk_id, float(score), chunk))
            return results
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            raise

    @staticmethod
    def _rrf_merge(
        vector_results: list[tuple[str, float, DocumentChunk]],
        bm25_results: list[tuple[str, float, DocumentChunk]],
        k: int = 60,
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Reciprocal Rank Fusion of vector and BM25 result lists."""
        max_rrf = 2.0 / (k + 1)
        scores: dict[str, float] = {}
        chunks: dict[str, tuple[str, DocumentChunk]] = {}

        for rank, (chunk_id, _score, chunk) in enumerate(vector_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk_id] = (chunk_id, chunk)

        for rank, (chunk_id, _score, chunk) in enumerate(bm25_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunks:
                chunks[chunk_id] = (chunk_id, chunk)

        merged = []
        for chunk_id, rrf_score in scores.items():
            normalized = rrf_score / max_rrf
            cid, chunk = chunks[chunk_id]
            merged.append((cid, normalized, chunk))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

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
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Search by vector similarity, with optional BM25 hybrid re-ranking.

        Args:
            query_embedding: np.ndarray of shape (dim,) or (1, dim).
            limit: Maximum number of results to return.
            query_text: If provided, perform hybrid vector + BM25 RRF search.
            include_breadcrumbs: If True (default), prepend stored breadcrumb to
                returned chunk content. If False, return raw content only.
            namespaces: If provided, restrict search to rows whose namespace is in
                this list. None searches all namespaces (backwards-compatible default).
            chunk_types: If provided, restrict search to rows whose chunk_type is in
                this list. None searches all chunk types (backwards-compatible default).
            domain_ids: If provided, restrict search to rows whose domain_id is in
                this list. Independent of namespaces filter; when both set both apply (AND).

        Returns:
            List of (chunk_id, score, DocumentChunk).
        """
        from ..models import DocumentChunk

        query_vec = query_embedding.flatten().astype("float32")
        fetch_limit = limit * 3 if query_text else limit

        if self._np_embeddings is not None and _np is not None:
            assert (
                self._np_chunk_rows is not None
            )  # set together with _np_embeddings in preload_embeddings
            # Fast numpy path: single matmul, no DuckDB round-trip
            sims = self._np_embeddings @ query_vec  # (n,)
            ns_set = set(namespaces) if namespaces is not None else None
            ct_set = set(chunk_types) if chunk_types is not None else None
            di_set = set(domain_ids) if domain_ids is not None else None
            top_idx = _np.argpartition(sims, -fetch_limit)[-fetch_limit:]
            top_idx = top_idx[_np.argsort(sims[top_idx])[::-1]]
            if ns_set is not None:
                top_idx = [i for i in top_idx if self._np_chunk_rows[i][9] in ns_set]
            if ct_set is not None:
                top_idx = [i for i in top_idx if self._np_chunk_rows[i][6] in ct_set]
            if di_set is not None:
                # domain_id is at index 12 in preloaded rows (added after source_id at 11)
                top_idx = [
                    i
                    for i in top_idx
                    if len(self._np_chunk_rows[i]) > 12 and self._np_chunk_rows[i][12] in di_set
                ]
            # indices: 0=chunk_id,1=doc_name,2=section,3=chunk_idx,4=content,
            #          5=breadcrumb,6=chunk_type,7=source_offset,8=source_length,
            #          9=namespace,10=source_detail
            rows = [
                (*self._np_chunk_rows[i][:9], self._np_chunk_rows[i][10], float(sims[i]))
                for i in top_idx
            ]
        else:
            query = query_vec.tolist()
            clauses: list[str] = []
            params: list[Any] = [query]
            if namespaces is not None:
                placeholders = ", ".join("?" * len(namespaces))
                clauses.append(f"e.namespace IN ({placeholders})")
                params.extend(namespaces)
            if chunk_types is not None:
                placeholders = ", ".join("?" * len(chunk_types))
                clauses.append(f"e.chunk_type IN ({placeholders})")
                params.extend(chunk_types)
            if domain_ids is not None:
                placeholders = ", ".join("?" * len(domain_ids))
                clauses.append(f"e.domain_id IN ({placeholders})")
                params.extend(domain_ids)
            if session_fingerprint is not None:
                clauses.append("e.session_fingerprint = ?")
                params.append(session_fingerprint)
            where_clause = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params += [query, fetch_limit]

            rows = self._conn.execute(
                f"""
                SELECT
                    e.chunk_id,
                    e.document_name,
                    e.section,
                    e.chunk_index,
                    e.content,
                    e.breadcrumb,
                    e.chunk_type,
                    e.source_offset,
                    e.source_length,
                    e.source_detail,
                    1.0 - array_cosine_distance(
                        e.embedding, ?::FLOAT[{self._embedding_dim}]) AS similarity
                FROM {self._read_table} e
                {where_clause}
                ORDER BY array_cosine_distance(e.embedding, ?::FLOAT[{self._embedding_dim}]) ASC
                LIMIT ?
                """,
                params,
            ).fetchall()

        vector_results = []
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
            vector_results.append((chunk_id, float(similarity), chunk))

        if not query_text:
            return vector_results

        bm25_results = self._bm25_search(
            query_text,
            limit=fetch_limit,
            namespaces=namespaces,
            chunk_types=chunk_types,
            domain_ids=domain_ids,
            session_fingerprint=session_fingerprint,
        )
        if not bm25_results:
            return vector_results[:limit]

        merged = self._rrf_merge(vector_results, bm25_results)
        return merged[:limit]

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Return all stored chunks as DocumentChunk objects."""
        from ..models import DocumentChunk

        rows = self._conn.execute(
            f"""
            SELECT document_name, content, section, chunk_index,
                   source_offset, source_length, breadcrumb, source_detail, chunk_type
            FROM {self._read_table}
            ORDER BY document_name, chunk_index
            """
        ).fetchall()
        return [
            DocumentChunk(
                document_name=row[0],
                content=row[1],
                section=_deserialize_section(row[2]),
                chunk_index=row[3],
                source_offset=row[4],
                source_length=row[5],
                breadcrumb=row[6],
                embedding_content=f"{row[6]}\n\n{row[1]}" if row[6] else row[1],
                source_detail=json.loads(row[7]) if row[7] else None,
                chunk_type=row[8] or "document",
            )
            for row in rows
        ]

    def migrate_extract_breadcrumbs(self) -> int:
        """One-time migration: extract breadcrumb prefix from content into breadcrumb column.

        For databases created before the breadcrumb column was added, the breadcrumb
        was baked into content as ``[doc > section]\\n\\ncontent``. This method
        extracts it into the breadcrumb column and strips it from content.

        Returns:
            Number of rows updated.
        """
        from ._schema import EMBEDDINGS_MIGRATE_BREADCRUMB

        try:
            self._conn.execute(EMBEDDINGS_MIGRATE_BREADCRUMB).fetchall()
        except Exception:
            pass  # column already exists

        self._conn.execute("""
            UPDATE embeddings
            SET
                breadcrumb = regexp_extract(content, '^(\\[[^\\]]+\\])', 1),
                content    = regexp_replace(content, '^\\[[^\\]]+\\]\\n\\n', '')
            WHERE breadcrumb IS NULL
              AND content LIKE '[%'
        """).fetchall()

        _row = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE breadcrumb IS NOT NULL"
        ).fetchone()
        if _row is None:
            raise RuntimeError("COUNT(*) returned no rows")
        return _row[0]

    def count(self) -> int:
        """Return the total number of stored chunks."""
        result = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return result[0] if result else 0

    # ------------------------------------------------------------------
    # Delete / clear
    # ------------------------------------------------------------------

    def _run(self, sql: str, params: list[Any]) -> None:
        self._conn.execute(sql, params).fetchall()

    def _scalar(self, sql: str, params: list[Any]) -> Any:  # noqa: ANN401
        rows = self._conn.execute(sql, params).fetchall()
        return rows[0][0] if rows else None

    def chunk_ids_for_document(self, document_name: str) -> list[str]:
        """Return every chunk_id currently stored for *document_name*."""
        return [
            r[0]
            for r in self._conn.execute(
                "SELECT chunk_id FROM embeddings WHERE document_name = ?",
                [document_name],
            ).fetchall()
        ]

    def delete_chunk_dependents(self, chunk_ids: list[str]) -> None:
        """Delete every chunk-keyed row referencing *chunk_ids*."""
        _cascade.delete_chunk_dependents(self._run, self._table_exists, chunk_ids)

    def gc_orphaned_entities(self) -> int:
        """Delete entities no chunk or triple references, and their dependents."""
        return _cascade.gc_orphaned_entities(self._run, self._scalar, self._table_exists)

    def delete_by_document(self, document_name: str, *, gc_entities: bool = True) -> int:
        """Delete all chunks for a document and everything keyed to them.

        Cascades to ``chunk_entities``, ``chunk_clusters``, ``svo_triples``, and
        the ``documents`` registry row, then garbage-collects entities left with
        no references.

        Args:
            gc_entities: Set False to skip the entity sweep — callers deleting
                many documents in a batch should run :meth:`gc_orphaned_entities`
                once at the end instead.

        Returns:
            Number of chunks deleted.
        """
        chunk_ids = self.chunk_ids_for_document(document_name)
        self.delete_chunk_dependents(chunk_ids)

        self._conn.execute(
            "DELETE FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchall()
        self._conn.execute(
            "DELETE FROM documents WHERE document_name = ?",
            [document_name],
        ).fetchall()

        if gc_entities and chunk_ids:
            self.gc_orphaned_entities()

        self._fts_dirty = True
        return len(chunk_ids)

    def clear(self) -> None:
        """Delete all chunks and everything derived from them.

        Leaves the namespace/domain/source registries intact — those describe
        where content comes from, not the content itself.
        """
        self._conn.execute("DELETE FROM embeddings").fetchall()
        self._conn.execute("DELETE FROM documents").fetchall()
        for table in (
            *_cascade.CHUNK_KEYED_TABLES,
            "entities",
            "entity_aliases",
            "context_graph_edges",
        ):
            if self._table_exists(table):
                self._conn.execute(f"DELETE FROM {table}").fetchall()  # noqa: S608
        self._fts_dirty = True

    # ------------------------------------------------------------------
    # Document registry
    # ------------------------------------------------------------------

    def get_document_hash(self, document_name: str) -> str | None:
        """Return the stored content hash for *document_name*, or None if unknown."""
        row = self._conn.execute(
            "SELECT content_hash FROM documents WHERE document_name = ?",
            [document_name],
        ).fetchone()
        return row[0] if row else None

    def register_document(
        self,
        document_name: str,
        content_hash: str,
        source_uri: str = "",
        chunk_count: int = 0,
    ) -> None:
        """Upsert a document record after indexing.

        Call this after :meth:`add_chunks` to record that *document_name* is
        up-to-date at *content_hash*.
        """
        self._conn.execute(
            """
            INSERT INTO documents (document_name, content_hash, source_uri, indexed_at, chunk_count)
            VALUES (?, ?, ?, now(), ?)
            ON CONFLICT (document_name) DO UPDATE SET
                content_hash = excluded.content_hash,
                source_uri   = excluded.source_uri,
                indexed_at   = excluded.indexed_at,
                chunk_count  = excluded.chunk_count
            """,
            [document_name, content_hash, source_uri, chunk_count],
        ).fetchall()

    def list_documents(self) -> list[dict[str, Any]]:
        """Return all registered documents as a list of dicts.

        Each dict has keys: ``document_name``, ``content_hash``,
        ``source_uri``, ``indexed_at``, ``chunk_count``.
        """
        rows = self._conn.execute(
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


def sync_document(
    backend: VectorBackend,
    document_name: str,
    raw: bytes | None = None,
    *,
    content_hash: str | None = None,
    source_uri: str = "",
) -> SyncResult:
    """Check whether a document needs reindexing and delete stale chunks if so.

    Supports two calling patterns:

    **1. Raw bytes supplied** — hash is computed from the bytes (sha256).
    Use this when the document has already been downloaded::

        result = sync_document(backend, "nvd-feed", raw_bytes)

    **2. Hash supplied upfront** — no bytes needed.  Use this when the source
    provides an ETag, ``Last-Modified`` header, or its own version identifier,
    so you can skip downloading an unchanged document entirely::

        etag = http_head(url).headers["ETag"]
        result = sync_document(backend, "nvd-feed", content_hash=etag)
        if result.action != "skipped":
            raw_bytes = http_get(url).content
            # ... embed and add_chunks ...

    Either *raw* or *content_hash* must be provided.

    Return values for ``action``:

    * **"skipped"** — hash matches stored value; index is current, nothing changed.
    * **"added"**   — document not previously indexed; no stale chunks existed.
    * **"updated"** — document previously indexed with a different hash; all old
      chunks have been deleted.

    On any non-skipped result the caller must re-embed and call
    :meth:`~DuckDBVectorBackend.add_chunks` then
    :meth:`~DuckDBVectorBackend.register_document`.  Pass
    ``result.content_hash`` to ``register_document`` — no need to hash again.
    """
    if content_hash is None:
        if raw is None:
            raise ValueError("Either raw or content_hash must be provided.")
        content_hash = hashlib.sha256(raw).hexdigest()

    stored_hash = backend.get_document_hash(document_name)

    if stored_hash == content_hash:
        return SyncResult(
            action="skipped",
            document_name=document_name,
            content_hash=content_hash,
        )

    prev_count = backend.delete_by_document(document_name)
    action = "updated" if stored_hash is not None else "added"
    return SyncResult(
        action=action,
        document_name=document_name,
        content_hash=content_hash,
        previous_chunk_count=prev_count,
    )


def prune_documents(
    backend: VectorBackend,
    present: Iterable[str],
    *,
    dry_run: bool = False,
) -> list[SyncResult]:
    """Delete every registered document that is absent from *present*.

    This is the deletion half of an incremental sync.  :func:`sync_document`
    only ever sees documents the source still has, so a document removed at the
    source stays indexed and searchable forever unless this is called::

        present = set()
        for doc_id, raw in crawl_source():
            present.add(doc_id)
            result = sync_document(backend, doc_id, raw)
            ...
        removed = prune_documents(backend, present)

    Args:
        backend: The vector backend to prune.
        present: **Every** document name the source currently holds.  A partial
            set deletes live data, so this must be the complete enumeration —
            not a page, a filtered view, or the results of a failed crawl.
        dry_run: Report what would be deleted without touching the index.

    Returns:
        One :class:`SyncResult` per removed document with ``action="deleted"``
        and ``previous_chunk_count`` set, ordered by document name.

    Raises:
        ValueError: If *present* is empty while the registry is not.  An empty
            source enumeration is far more often a failed crawl than an intent
            to wipe the index; call :meth:`~DuckDBVectorBackend.clear` to do
            that deliberately.
    """
    registered = [doc["document_name"] for doc in backend.list_documents()]
    present_set = set(present)

    if not present_set and registered:
        raise ValueError(
            f"prune_documents() received an empty 'present' set while {len(registered)} "
            f"documents are registered — this would delete the entire index. If that is "
            f"intended, call backend.clear() instead."
        )

    stale = sorted(name for name in registered if name not in present_set)

    results: list[SyncResult] = []
    for name in stale:
        if dry_run:
            prev_count = len(backend.chunk_ids_for_document(name))
        else:
            # Entity GC is deferred to a single sweep after the batch.
            prev_count = backend.delete_by_document(name, gc_entities=False)
        results.append(
            SyncResult(
                action="deleted",
                document_name=name,
                previous_chunk_count=prev_count,
            )
        )

    if results and not dry_run:
        backend.gc_orphaned_entities()

    return results
