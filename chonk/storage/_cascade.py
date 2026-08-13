# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 1c8a4f60-7d92-4e3b-95af-3b70e2c81d64
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Shared cascade-delete logic for every vector backend.

Deleting a document must remove everything keyed to its chunks, or the derived
rows outlive the content they describe: stale entities rebind to reused chunk
ids, the NER gate treats changed chunks as already processed, and community
summaries stay built on text that no longer exists.

Backends differ only in their parameter placeholder (``?`` for DuckDB,
``%s`` for PostgreSQL) and in how a statement is run, so both are injected.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Tables whose rows are keyed by chunk_id. svo_triples is created lazily by the
# graph builder in some backends, so presence is checked rather than assumed.
CHUNK_KEYED_TABLES = ("chunk_entities", "chunk_clusters", "svo_triples")

# Bound parameters per statement are capped by every engine here; IN lists are
# chunked at this size.
IN_BATCH = 1000

Run = Callable[[str, list[Any]], None]
Scalar = Callable[[str, list[Any]], Any]
TableExists = Callable[[str], bool]


def delete_chunk_dependents(
    run: Run,
    table_exists: TableExists,
    chunk_ids: list[str],
    placeholder: str = "?",
) -> None:
    """Delete every chunk-keyed row referencing *chunk_ids*.

    Call before removing the chunks themselves — the ids are read from the
    embeddings table, so they are unrecoverable afterwards.
    """
    if not chunk_ids:
        return
    for table in CHUNK_KEYED_TABLES:
        if not table_exists(table):
            continue
        for start in range(0, len(chunk_ids), IN_BATCH):
            batch = chunk_ids[start : start + IN_BATCH]
            placeholders = ", ".join([placeholder] * len(batch))
            run(f"DELETE FROM {table} WHERE chunk_id IN ({placeholders})", batch)  # noqa: S608


def gc_orphaned_entities(run: Run, scalar: Scalar, table_exists: TableExists) -> int:
    """Delete entities nothing references any more, and their dependents.

    Entities are only ever written alongside a ``chunk_entities`` row, so an
    entity with no remaining chunk link and no remaining triple is dead.

    Returns:
        Number of entities deleted.
    """
    if not table_exists("entities"):
        return 0

    predicate = "id NOT IN (SELECT entity_id FROM chunk_entities)"
    if table_exists("svo_triples"):
        predicate += (
            " AND id NOT IN (SELECT subject_id FROM svo_triples)"
            " AND id NOT IN (SELECT object_id FROM svo_triples)"
        )

    deleted = scalar(f"SELECT COUNT(*) FROM entities WHERE {predicate}", [])  # noqa: S608
    if deleted is None:
        raise RuntimeError("COUNT(*) returned no rows")
    if deleted == 0:
        return 0

    run(f"DELETE FROM entities WHERE {predicate}", [])  # noqa: S608

    if table_exists("entity_aliases"):
        run("DELETE FROM entity_aliases WHERE entity_id NOT IN (SELECT id FROM entities)", [])
    if table_exists("context_graph_edges"):
        run(
            "DELETE FROM context_graph_edges "
            "WHERE source_entity_id NOT IN (SELECT id FROM entities) "
            "   OR target_entity_id NOT IN (SELECT id FROM entities)",
            [],
        )
    return int(deleted)
