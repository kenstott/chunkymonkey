# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: b73e9d04-2c85-41fa-9e6d-5a0c8b1f7d32

"""Tests for prune_documents — removal of deleted source documents (fixed.md phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

from chonk.storage import prune_documents


@pytest.fixture()
def store():
    from chonk.storage import Store

    with Store(":memory:", embedding_dim=4) as s:
        yield s


@pytest.fixture()
def conn(store):
    return store.vector._conn


def _index(store, document_name: str, contents: list[str]) -> list[str]:
    from chonk.models import DocumentChunk

    chunks = [
        DocumentChunk(document_name=document_name, content=text, chunk_index=i)
        for i, text in enumerate(contents)
    ]
    store.add_document(chunks, np.zeros((len(chunks), 4), dtype="float32"))
    store.vector.register_document(document_name, f"hash-{document_name}")
    return [
        store.vector._generate_chunk_id(document_name, c.chunk_index, c.content) for c in chunks
    ]


def _link_entity(conn, chunk_id: str, entity_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO entities(id, name, display_name, entity_type) VALUES (?,?,?,?)",
        [entity_id, entity_id, entity_id, "concept"],
    )
    conn.execute(
        "INSERT OR REPLACE INTO chunk_entities"
        "(chunk_id, entity_id, frequency, positions_json, score) VALUES (?,?,?,?,?)",
        [chunk_id, entity_id, 1, "[]", 1.0],
    )


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]  # noqa: S608


def _names(store) -> list[str]:
    return [d["document_name"] for d in store.vector.list_documents()]


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


class TestPruneDocuments:
    def test_prune_removes_absent_document(self, store, conn, no_orphans):
        """Regression: defect #1 — a document removed at the source stayed indexed
        forever because sync_document only ever sees documents that still exist."""
        _index(store, "keep", ["kept content"])
        _index(store, "gone", ["removed content"])

        prune_documents(store.vector, {"keep"})

        assert _names(store) == ["keep"]
        assert store.vector.get_document_hash("gone") is None
        rows = conn.execute("SELECT DISTINCT document_name FROM embeddings").fetchall()
        assert [r[0] for r in rows] == ["keep"]
        no_orphans(conn)

    def test_prune_keeps_present_documents(self, store, conn, no_orphans):
        _index(store, "a", ["alpha"])
        _index(store, "b", ["beta"])

        assert prune_documents(store.vector, {"a", "b"}) == []
        assert _names(store) == ["a", "b"]
        no_orphans(conn)

    def test_prune_returns_deleted_results(self, store):
        _index(store, "keep", ["kept"])
        _index(store, "gone", ["one", "two", "three"])

        results = prune_documents(store.vector, {"keep"})

        assert len(results) == 1
        assert results[0].action == "deleted"
        assert results[0].document_name == "gone"
        assert results[0].previous_chunk_count == 3
        assert results[0].content_hash == ""

    def test_prune_results_sorted_by_name(self, store):
        for name in ("zeta", "alpha", "mid"):
            _index(store, name, ["content of " + name])

        results = prune_documents(store.vector, {"keep-nothing", "mid"})

        assert [r.document_name for r in results] == ["alpha", "zeta"]

    def test_prune_accepts_any_iterable(self, store):
        _index(store, "a", ["alpha"])
        _index(store, "b", ["beta"])

        assert prune_documents(store.vector, ["a", "b"]) == []

    def test_prune_ignores_unknown_present_names(self, store):
        _index(store, "a", ["alpha"])

        assert prune_documents(store.vector, {"a", "never-indexed"}) == []
        assert _names(store) == ["a"]

    def test_prune_is_idempotent(self, store):
        _index(store, "keep", ["kept"])
        _index(store, "gone", ["removed"])

        assert len(prune_documents(store.vector, {"keep"})) == 1
        assert prune_documents(store.vector, {"keep"}) == []

    def test_prune_on_empty_index_is_noop(self, store):
        assert prune_documents(store.vector, {"anything"}) == []


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class TestPruneGuardrails:
    def test_prune_empty_present_raises(self, store):
        """An empty source enumeration is a failed crawl far more often than an
        intent to wipe the index."""
        _index(store, "a", ["alpha"])

        with pytest.raises(ValueError) as exc:
            prune_documents(store.vector, set())

        assert "clear()" in str(exc.value)
        assert _names(store) == ["a"]

    def test_prune_empty_present_on_empty_registry_is_noop(self, store):
        assert prune_documents(store.vector, set()) == []

    def test_prune_dry_run_deletes_nothing(self, store, conn, no_orphans):
        _index(store, "keep", ["kept"])
        gone = _index(store, "gone", ["one", "two"])
        _link_entity(conn, gone[0], "ent-gone")

        results = prune_documents(store.vector, {"keep"}, dry_run=True)

        assert [r.document_name for r in results] == ["gone"]
        assert results[0].previous_chunk_count == 2
        assert results[0].action == "deleted"
        assert _names(store) == ["gone", "keep"]
        assert _count(conn, "chunk_entities") == 1
        no_orphans(conn)

    def test_dry_run_matches_real_run(self, store):
        _index(store, "keep", ["kept"])
        _index(store, "gone", ["one", "two"])

        dry = prune_documents(store.vector, {"keep"}, dry_run=True)
        real = prune_documents(store.vector, {"keep"})

        assert [(r.document_name, r.previous_chunk_count) for r in dry] == [
            (r.document_name, r.previous_chunk_count) for r in real
        ]


# ---------------------------------------------------------------------------
# Cascade behaviour
# ---------------------------------------------------------------------------


class TestPruneCascade:
    def test_prune_cascades_to_chunk_entities(self, store, conn, no_orphans):
        keep = _index(store, "keep", ["kept"])
        gone = _index(store, "gone", ["removed"])
        _link_entity(conn, keep[0], "ent-keep")
        _link_entity(conn, gone[0], "ent-gone")

        prune_documents(store.vector, {"keep"})

        remaining = {r[0] for r in conn.execute("SELECT chunk_id FROM chunk_entities").fetchall()}
        assert remaining == {keep[0]}
        no_orphans(conn)

    def test_prune_gcs_orphaned_entities_once(self, store, conn, no_orphans):
        keep = _index(store, "keep", ["kept"])
        for name in ("gone-a", "gone-b"):
            ids = _index(store, name, [f"content of {name}"])
            _link_entity(conn, ids[0], f"ent-{name}")
        _link_entity(conn, keep[0], "ent-keep")

        prune_documents(store.vector, {"keep"})

        assert [r[0] for r in conn.execute("SELECT id FROM entities").fetchall()] == ["ent-keep"]
        no_orphans(conn)

    def test_prune_keeps_entities_shared_with_surviving_documents(self, store, conn, no_orphans):
        keep = _index(store, "keep", ["kept"])
        gone = _index(store, "gone", ["removed"])
        _link_entity(conn, keep[0], "ent-shared")
        _link_entity(conn, gone[0], "ent-shared")

        prune_documents(store.vector, {"keep"})

        assert [r[0] for r in conn.execute("SELECT id FROM entities").fetchall()] == ["ent-shared"]
        no_orphans(conn)

    def test_prune_marks_fts_dirty(self, store):
        _index(store, "keep", ["kept"])
        _index(store, "gone", ["removed"])
        store.vector._fts_dirty = False

        prune_documents(store.vector, {"keep"})

        assert store.vector._fts_dirty is True
