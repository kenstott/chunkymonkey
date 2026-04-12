"""VectorBackend protocol — implement this to plug in alternative vector stores."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    """Protocol for vector store backends.

    Implement this to provide a custom vector store for chunkeymonkey.
    The :class:`~chunkeymonkey.storage.DuckDBVectorBackend` is the
    default implementation.
    """

    def add_chunks(self, chunks: list, embeddings) -> None: ...

    def search(
        self,
        query_embedding,
        limit: int = 5,
        query_text: str | None = None,
    ) -> list: ...

    def delete_by_document(self, document_name: str) -> int: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...
