"""DuckDB VSS + FTS vector backend for chunkeymonkey."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

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
    import numpy as _np
    from .._pool import ThreadLocalDuckDB  # only for type hints

logger = logging.getLogger(__name__)

_MISSING_DEPS_MSG = (
    "duckdb and numpy are required for storage. "
    "Install them with: pip install chunkeymonkey[storage]"
)


def _require_deps() -> None:
    if not _DUCKDB_AVAILABLE or not _NUMPY_AVAILABLE:
        raise ImportError(_MISSING_DEPS_MSG)


class DuckDBVectorBackend:
    """Vector operations backed by DuckDB VSS (HNSW) + FTS extensions."""

    def __init__(self, db, embedding_dim: int = 1024):
        _require_deps()
        self._db = db
        self._embedding_dim = embedding_dim
        self._fts_dirty = True
        self._init_schema()

    @property
    def _conn(self):
        return self._db.conn

    # ------------------------------------------------------------------
    # Schema init
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        from ._schema import get_ddl, VSS_INDEX_DDL

        # Load extensions (best-effort — may already be loaded)
        for ext in ("vss", "fts"):
            try:
                self._conn.execute(f"INSTALL {ext}").fetchall()
                self._conn.execute(f"LOAD {ext}").fetchall()
            except Exception as e:
                logger.debug(f"Extension {ext} load skipped: {e}")

        for ddl in get_ddl(self._embedding_dim):
            try:
                self._conn.execute(ddl).fetchall()
            except Exception as e:
                logger.debug(f"DDL skipped: {e}")

        # VSS HNSW index — best-effort (requires empty table or existing data)
        try:
            self._conn.execute(VSS_INDEX_DDL).fetchall()
        except Exception as e:
            logger.debug(f"VSS index creation skipped: {e}")

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

    def add_chunks(self, chunks: list, embeddings) -> None:
        """Insert chunks with embeddings into the embeddings table.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: np.ndarray of shape (n, embedding_dim).
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
            chunk_type = (
                chunk.chunk_type.value
                if hasattr(chunk, "chunk_type") and hasattr(chunk.chunk_type, "value")
                else getattr(chunk, "chunk_type", "document") or "document"
            )
            records.append((
                chunk_id,
                chunk.document_name,
                getattr(chunk, "section", None),
                chunk.chunk_index,
                chunk.content,
                chunk_type,
                getattr(chunk, "source_offset", None),
                getattr(chunk, "source_length", None),
                embedding,
            ))

        self._conn.executemany(
            """
            INSERT INTO embeddings
                (chunk_id, document_name, section, chunk_index, content,
                 chunk_type, source_offset, source_length, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            count = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            if count == 0:
                self._fts_dirty = False
                return
            self._conn.execute(
                "PRAGMA create_fts_index('embeddings', 'chunk_id', 'content', "
                "stemmer='porter', overwrite=1)"
            ).fetchall()
            self._fts_dirty = False
        except Exception as e:
            logger.debug(f"FTS index rebuild failed (vector-only mode): {e}")
            self._fts_dirty = False

    def _bm25_search(
        self,
        query_text: str,
        limit: int = 5,
    ) -> list[tuple[str, float, object]]:
        from ..models import DocumentChunk
        try:
            self._rebuild_fts_index()
            rows = self._conn.execute(
                """
                SELECT e.chunk_id, e.document_name, e.section, e.chunk_index,
                       e.content, e.chunk_type, e.source_offset, e.source_length,
                       fts_main_embeddings.match_bm25(e.chunk_id, ?) AS bm25_score
                FROM embeddings e
                WHERE bm25_score IS NOT NULL
                ORDER BY bm25_score DESC
                LIMIT ?
                """,
                [query_text, limit],
            ).fetchall()

            results = []
            for row in rows:
                (chunk_id, doc_name, section, chunk_idx, content,
                 chunk_type_str, source_offset, source_length, score) = row
                chunk = DocumentChunk(
                    document_name=doc_name,
                    content=content,
                    section=section,
                    chunk_index=chunk_idx,
                    source_offset=source_offset,
                    source_length=source_length,
                )
                results.append((chunk_id, float(score), chunk))
            return results
        except Exception as e:
            logger.debug(f"BM25 search failed (vector-only mode): {e}")
            return []

    @staticmethod
    def _rrf_merge(
        vector_results: list[tuple],
        bm25_results: list[tuple],
        k: int = 60,
    ) -> list[tuple]:
        """Reciprocal Rank Fusion of vector and BM25 result lists."""
        max_rrf = 2.0 / (k + 1)
        scores: dict[str, float] = {}
        chunks: dict[str, tuple] = {}

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
        query_embedding,
        limit: int = 5,
        query_text: str | None = None,
    ) -> list[tuple[str, float, object]]:
        """Search by vector similarity, with optional BM25 hybrid re-ranking.

        Args:
            query_embedding: np.ndarray of shape (dim,) or (1, dim).
            limit: Maximum number of results to return.
            query_text: If provided, perform hybrid vector + BM25 RRF search.

        Returns:
            List of (chunk_id, score, DocumentChunk).
        """
        from ..models import DocumentChunk

        query = query_embedding.flatten().tolist()
        fetch_limit = limit * 3 if query_text else limit

        rows = self._conn.execute(
            f"""
            SELECT
                e.chunk_id,
                e.document_name,
                e.section,
                e.chunk_index,
                e.content,
                e.chunk_type,
                e.source_offset,
                e.source_length,
                array_cosine_similarity(e.embedding, ?::FLOAT[{self._embedding_dim}]) AS similarity
            FROM embeddings e
            ORDER BY similarity DESC
            LIMIT ?
            """,
            [query, fetch_limit],
        ).fetchall()

        vector_results = []
        for row in rows:
            (chunk_id, doc_name, section, chunk_idx, content,
             chunk_type_str, source_offset, source_length, similarity) = row
            chunk = DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section,
                chunk_index=chunk_idx,
                source_offset=source_offset,
                source_length=source_length,
            )
            vector_results.append((chunk_id, float(similarity), chunk))

        if not query_text:
            return vector_results

        bm25_results = self._bm25_search(query_text, limit=fetch_limit)
        if not bm25_results:
            return vector_results[:limit]

        merged = self._rrf_merge(vector_results, bm25_results)
        return merged[:limit]

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_all_chunks(self) -> list:
        """Return all stored chunks as DocumentChunk objects."""
        from ..models import DocumentChunk

        rows = self._conn.execute(
            """
            SELECT document_name, content, section, chunk_index,
                   source_offset, source_length
            FROM embeddings
            ORDER BY document_name, chunk_index
            """
        ).fetchall()
        return [
            DocumentChunk(
                document_name=row[0],
                content=row[1],
                section=row[2],
                chunk_index=row[3],
                source_offset=row[4],
                source_length=row[5],
            )
            for row in rows
        ]

    def count(self) -> int:
        """Return the total number of stored chunks."""
        result = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return result[0] if result else 0

    # ------------------------------------------------------------------
    # Delete / clear
    # ------------------------------------------------------------------

    def delete_by_document(self, document_name: str) -> int:
        """Delete all chunks for a document. Returns the number deleted."""
        count_before = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchone()[0]
        self._conn.execute(
            "DELETE FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchall()
        self._fts_dirty = True
        return count_before

    def clear(self) -> None:
        """Delete all chunks from the embeddings table."""
        self._conn.execute("DELETE FROM embeddings").fetchall()
        self._fts_dirty = True
