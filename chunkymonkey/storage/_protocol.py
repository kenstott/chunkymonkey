# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: fb0dc27c-4ed9-413d-ae7f-be8255a0d901
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""VectorBackend protocol — implement this to plug in alternative vector stores."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    """Protocol for vector store backends.

    Implement this to provide a custom vector store for chunkymonkey.
    The :class:`~chunkymonkey.storage.DuckDBVectorBackend` is the
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
