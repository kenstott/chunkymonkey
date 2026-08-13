# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 5a1d7c93-4f60-42be-9b18-6c0e2a7d4e51

"""Shared test fixtures and invariant helpers."""

from __future__ import annotations

import pytest


def assert_no_orphans(conn) -> None:
    """No chunk-keyed row may reference a chunk_id absent from embeddings.

    Asserted at the end of every incremental-update test: an orphan means a
    delete path missed a table, which is how stale entities silently rebind to
    new content after a document is updated.
    """
    live = {r[0] for r in conn.execute("SELECT chunk_id FROM embeddings").fetchall()}

    for table in ("chunk_entities", "chunk_clusters", "svo_triples"):
        exists = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchall()[0][0]
        if not exists:
            continue
        referenced = {
            r[0]
            for r in conn.execute(f"SELECT DISTINCT chunk_id FROM {table}").fetchall()  # noqa: S608
            if r[0] is not None
        }
        orphans = referenced - live
        assert not orphans, (
            f"{table} references {len(orphans)} dead chunk_id(s): {sorted(orphans)[:5]}"
        )


@pytest.fixture()
def no_orphans():
    """Return :func:`assert_no_orphans`, bound as a fixture.

    Exposed as a fixture rather than imported so tests do not depend on ``tests``
    being importable as a package.
    """
    return assert_no_orphans
