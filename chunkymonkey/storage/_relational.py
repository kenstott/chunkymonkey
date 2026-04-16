# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 28f11018-de8c-4abf-827a-877a08d66160
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Relational store for chunk metadata and entity links.

Uses SQLAlchemy Core (not ORM) for compatibility with any SQLAlchemy-supported database.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

try:
    from sqlalchemy import create_engine, text
    _SA_AVAILABLE = True
except ImportError:
    _SA_AVAILABLE = False

logger = logging.getLogger(__name__)

_MISSING_SA_MSG = (
    "sqlalchemy is required for relational storage. "
    "Install it with: pip install chunkymonkey[storage]"
)

_ENTITIES_DDL = """
CREATE TABLE IF NOT EXISTS entities (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    display_name TEXT NOT NULL,
    entity_type  TEXT NOT NULL DEFAULT 'concept',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_CHUNK_ENTITIES_DDL = """
CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id   TEXT NOT NULL,
    entity_id  TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    PRIMARY KEY (chunk_id, entity_id)
)
"""


class RelationalStore:
    """Minimal CRUD for entities and chunk-entity links via SQLAlchemy Core."""

    def __init__(self, connection_string: str):
        """Args:
            connection_string: Any SQLAlchemy connection string,
                e.g. 'duckdb:///index.duckdb' or 'sqlite:///index.db'.
        """
        if not _SA_AVAILABLE:
            raise ImportError(_MISSING_SA_MSG)
        self._engine = create_engine(connection_string)

    def init_schema(self) -> None:
        """Create entities and chunk_entities tables if they do not exist."""
        with self._engine.connect() as conn:
            conn.execute(text(_ENTITIES_DDL))
            conn.execute(text(_CHUNK_ENTITIES_DDL))
            conn.commit()

    def add_entity(
        self,
        id: str,
        name: str,
        display_name: str,
        entity_type: str = "concept",
    ) -> None:
        """Insert an entity row (ignored on conflict)."""
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT OR IGNORE INTO entities (id, name, display_name, entity_type) "
                    "VALUES (:id, :name, :display_name, :entity_type)"
                ),
                {"id": id, "name": name, "display_name": display_name, "entity_type": entity_type},
            )
            conn.commit()

    def get_entities_for_chunk(self, chunk_id: str) -> list[dict]:
        """Return all entities linked to a chunk_id."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT e.id, e.name, e.display_name, e.entity_type, ce.confidence "
                    "FROM entities e "
                    "JOIN chunk_entities ce ON e.id = ce.entity_id "
                    "WHERE ce.chunk_id = :chunk_id"
                ),
                {"chunk_id": chunk_id},
            ).fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "display_name": row[2],
                "entity_type": row[3],
                "confidence": row[4],
            }
            for row in rows
        ]

    def add_chunk_entity_link(
        self,
        chunk_id: str,
        entity_id: str,
        confidence: float = 1.0,
    ) -> None:
        """Insert a chunk-entity link (ignored on conflict)."""
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id, confidence) "
                    "VALUES (:chunk_id, :entity_id, :confidence)"
                ),
                {"chunk_id": chunk_id, "entity_id": entity_id, "confidence": confidence},
            )
            conn.commit()

    def delete_entities_by_document(self, chunk_ids: list[str]) -> int:
        """Delete chunk_entities rows for the given chunk IDs. Returns count deleted."""
        if not chunk_ids:
            return 0
        with self._engine.connect() as conn:
            placeholders = ", ".join(f":id{i}" for i in range(len(chunk_ids)))
            params = {f"id{i}": cid for i, cid in enumerate(chunk_ids)}
            result = conn.execute(
                text(f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})"),
                params,
            )
            conn.commit()
            return result.rowcount
