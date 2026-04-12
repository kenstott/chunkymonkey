"""Store: composed facade over DuckDBVectorBackend and RelationalStore."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ._pool import ThreadLocalDuckDB
from ._vector import DuckDBVectorBackend
from ._relational import RelationalStore


class Store:
    """Composed storage facade backed by DuckDB.

    Provides a high-level interface to both vector search and
    relational entity storage via a single DuckDB file.

    Usage::

        with Store("index.duckdb") as store:
            store.add_document(chunks, embeddings)
            results = store.search(query_vec, limit=5)
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedding_dim: int = 1024,
    ):
        """Create a Store backed by DuckDB.

        Args:
            db_path: Path to DuckDB file, or ":memory:" for an in-memory store.
            embedding_dim: Embedding vector dimension. Must match your model.
        """
        db_path = str(db_path)
        self._db = ThreadLocalDuckDB(db_path)
        self.vector = DuckDBVectorBackend(self._db, embedding_dim=embedding_dim)
        relational_url = (
            f"duckdb:///{db_path}" if db_path != ":memory:" else "duckdb://"
        )
        try:
            self.relational = RelationalStore(relational_url)
            self.relational.init_schema()
        except Exception as e:
            # duckdb-engine may not be installed; relational features are optional.
            # Install with: pip install duckdb-engine
            import logging
            logging.getLogger(__name__).warning(
                f"RelationalStore init failed (entity features unavailable): {e}. "
                "Install duckdb-engine for full SQLAlchemy+DuckDB support."
            )
            self.relational = None  # type: ignore

    def add_document(self, chunks: list, embeddings) -> None:
        """Add chunks with embeddings. embeddings is np.ndarray shape (n, dim)."""
        self.vector.add_chunks(chunks, embeddings)

    def search(
        self,
        query_embedding,
        limit: int = 5,
        query_text: str | None = None,
    ) -> list:
        """Hybrid or pure vector search.

        Returns:
            List of (chunk_id, score, DocumentChunk).
        """
        return self.vector.search(query_embedding, limit=limit, query_text=query_text)

    def delete_document(self, document_name: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        return self.vector.delete_by_document(document_name)

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self.vector.count()

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_) -> None:
        self.close()
