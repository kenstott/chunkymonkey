# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: c9f6fe35-26e5-4e0a-bb6c-77278a38c5ed
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""chunkymonkey storage — DuckDB vector store + SQLAlchemy relational store."""
from ._protocol import VectorBackend
from ._store import Store
from ._vector import DuckDBVectorBackend
from ._relational import RelationalStore
from ._schema import get_ddl, EMBEDDINGS_DDL, ENTITIES_DDL, CHUNK_ENTITIES_DDL

__all__ = [
    "Store",
    "DuckDBVectorBackend",
    "RelationalStore",
    "VectorBackend",
    "get_ddl",
    "EMBEDDINGS_DDL",
    "ENTITIES_DDL",
    "CHUNK_ENTITIES_DDL",
]
