# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 8f42c1d7-6b09-4e35-a7c2-19d8e05b3f6a

"""Cascade-delete tests for the incremental update path (fixed.md phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from chonk.storage._vector import sync_document


@pytest.fixture()
def store():
    from chonk.storage import Store

    with Store(":memory:", embedding_dim=4) as s:
        yield s


@pytest.fixture()
def conn(store):
    return store.vector._conn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _index(store, document_name: str, contents: list[str]) -> list[str]:
    """Index *contents* as chunks of *document_name*; return their chunk_ids."""
    from chonk.models import DocumentChunk

    chunks = [
        DocumentChunk(document_name=document_name, content=text, chunk_index=i)
        for i, text in enumerate(contents)
    ]
    embeddings = np.zeros((len(chunks), 4), dtype="float32")
    store.add_document(chunks, embeddings)
    store.vector.register_document(document_name, f"hash-of-{document_name}-{contents}")
    return [
        store.vector._generate_chunk_id(document_name, c.chunk_index, c.content) for c in chunks
    ]


def _link_entity(conn, chunk_id: str, entity_id: str) -> None:
    """Create an entity and its chunk link, mirroring what build_ner writes."""
    conn.execute(
        "INSERT OR IGNORE INTO entities(id, name, display_name, entity_type) VALUES (?,?,?,?)",
        [entity_id, entity_id, entity_id, "concept"],
    )
    conn.execute(
        "INSERT OR REPLACE INTO chunk_entities"
        "(chunk_id, entity_id, frequency, positions_json, score) VALUES (?,?,?,?,?)",
        [chunk_id, entity_id, 1, "[]", 1.0],
    )
    conn.execute(
        "INSERT OR IGNORE INTO entity_aliases(alias, entity_id, source) VALUES (?,?,?)",
        [f"alias-{entity_id}", entity_id, "test"],
    )


def _cluster(conn, chunk_id: str, cluster_id: int = 1) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO chunk_clusters(chunk_id, cluster_id, namespace) VALUES (?,?,?)",
        [chunk_id, cluster_id, "global"],
    )


def _edge(conn, source_id: str, target_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO context_graph_edges"
        "(source_entity_id, target_entity_id, namespace, weight) VALUES (?,?,?,?)",
        [source_id, target_id, "global", 1.0],
    )


def _create_svo_table(conn) -> None:
    """svo_triples is created lazily by the graph builder, not by get_ddl."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS svo_triples (
            chunk_id    VARCHAR,
            subject_id  VARCHAR NOT NULL,
            verb        VARCHAR NOT NULL,
            object_id   VARCHAR NOT NULL,
            confidence  FLOAT   NOT NULL DEFAULT 1.0,
            namespace   VARCHAR,
            description TEXT    NOT NULL DEFAULT ''
        )
    """)


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]  # noqa: S608


# ---------------------------------------------------------------------------
# Delete cascades
# ---------------------------------------------------------------------------


class TestDeleteCascade:
    def test_delete_removes_chunk_entities(self, store, conn, no_orphans):
        """Regression: defect #2 — delete_by_document touched only embeddings."""
        ids = _index(store, "doc", ["alpha content", "beta content"])
        for cid in ids:
            _link_entity(conn, cid, "ent-shared")

        store.vector.delete_by_document("doc")

        assert _count(conn, "chunk_entities") == 0
        no_orphans(conn)

    def test_delete_removes_chunk_clusters(self, store, conn, no_orphans):
        ids = _index(store, "doc", ["alpha content"])
        _cluster(conn, ids[0])

        store.vector.delete_by_document("doc")

        assert _count(conn, "chunk_clusters") == 0
        no_orphans(conn)

    def test_delete_removes_svo_triples(self, store, conn, no_orphans):
        ids = _index(store, "doc", ["alpha content"])
        _create_svo_table(conn)
        conn.execute(
            "INSERT INTO svo_triples(chunk_id, subject_id, verb, object_id) VALUES (?,?,?,?)",
            [ids[0], "ent-a", "uses", "ent-b"],
        )

        store.vector.delete_by_document("doc")

        assert _count(conn, "svo_triples") == 0
        no_orphans(conn)

    def test_delete_removes_registry_row(self, store):
        _index(store, "doc", ["alpha content"])
        store.vector.delete_by_document("doc")
        assert store.vector.get_document_hash("doc") is None

    def test_delete_returns_chunk_count(self, store):
        _index(store, "doc", ["one", "two", "three"])
        assert store.vector.delete_by_document("doc") == 3

    def test_delete_unknown_document_returns_zero(self, store, conn, no_orphans):
        assert store.vector.delete_by_document("never-indexed") == 0
        no_orphans(conn)

    def test_delete_leaves_other_documents_intact(self, store, conn, no_orphans):
        keep = _index(store, "keep", ["kept content"])
        drop = _index(store, "drop", ["dropped content"])
        _link_entity(conn, keep[0], "ent-keep")
        _link_entity(conn, drop[0], "ent-drop")

        store.vector.delete_by_document("drop")

        remaining = {r[0] for r in conn.execute("SELECT chunk_id FROM chunk_entities").fetchall()}
        assert remaining == {keep[0]}
        assert store.vector.get_document_hash("keep") is not None
        no_orphans(conn)

    def test_delete_batches_large_chunk_id_lists(self, store, conn, no_orphans):
        """More chunks than the IN-list batch size must still cascade fully."""
        n = 1050
        ids = _index(store, "big", [f"content number {i}" for i in range(n)])
        for cid in ids:
            _cluster(conn, cid)

        assert store.vector.delete_by_document("big") == n
        assert _count(conn, "chunk_clusters") == 0
        no_orphans(conn)


# ---------------------------------------------------------------------------
# Update cascades
# ---------------------------------------------------------------------------


class TestUpdateCascade:
    def test_update_leaves_no_orphaned_chunk_entities(self, store, conn, no_orphans):
        """Regression: defect #2 on the update path."""
        v1 = _index(store, "doc", ["version one content", "second chunk of v1"])
        for cid in v1:
            _link_entity(conn, cid, "ent-v1")

        result = sync_document(store.vector, "doc", b"version two")
        assert result.action == "updated"
        _index(store, "doc", ["version two content", "second chunk of v2"])

        no_orphans(conn)

    def test_update_same_chunk_count_does_not_rebind_stale_entities(self, store, conn, no_orphans):
        """Regression: defects #2 + #3 together.

        v1 and v2 chunk to the same count at the same indices with a shared
        100-char prefix — the exact shape that used to collide on chunk_id and
        silently rebind v1's entities to v2's content.
        """
        prefix = "P" * 100
        v1 = _index(store, "doc", [prefix + " tail one", prefix + " tail two"])
        for cid in v1:
            _link_entity(conn, cid, "ent-stale")

        sync_document(store.vector, "doc", b"v2")
        v2 = _index(store, "doc", [prefix + " revised one", prefix + " revised two"])

        assert set(v1).isdisjoint(v2)
        assert _count(conn, "chunk_entities") == 0
        no_orphans(conn)

    def test_update_then_relink_yields_only_new_entities(self, store, conn, no_orphans):
        v1 = _index(store, "doc", ["version one content"])
        _link_entity(conn, v1[0], "ent-old")

        sync_document(store.vector, "doc", b"v2")
        v2 = _index(store, "doc", ["version two content"])
        _link_entity(conn, v2[0], "ent-new")

        rows = conn.execute("SELECT chunk_id, entity_id FROM chunk_entities").fetchall()
        assert rows == [(v2[0], "ent-new")]
        no_orphans(conn)


# ---------------------------------------------------------------------------
# Entity garbage collection
# ---------------------------------------------------------------------------


class TestEntityGC:
    def test_orphaned_entity_removed(self, store, conn):
        ids = _index(store, "doc", ["alpha content"])
        _link_entity(conn, ids[0], "ent-only-here")

        store.vector.delete_by_document("doc")

        assert _count(conn, "entities") == 0

    def test_shared_entity_survives(self, store, conn, no_orphans):
        a = _index(store, "doc-a", ["alpha content"])
        b = _index(store, "doc-b", ["beta content"])
        _link_entity(conn, a[0], "ent-shared")
        _link_entity(conn, b[0], "ent-shared")

        store.vector.delete_by_document("doc-a")

        ids = [r[0] for r in conn.execute("SELECT id FROM entities").fetchall()]
        assert ids == ["ent-shared"]
        no_orphans(conn)

    def test_entity_aliases_follow_entity_gc(self, store, conn):
        ids = _index(store, "doc", ["alpha content"])
        _link_entity(conn, ids[0], "ent-gone")

        store.vector.delete_by_document("doc")

        assert _count(conn, "entity_aliases") == 0

    def test_graph_edges_follow_entity_gc(self, store, conn):
        a = _index(store, "doc-a", ["alpha content"])
        b = _index(store, "doc-b", ["beta content"])
        _link_entity(conn, a[0], "ent-a")
        _link_entity(conn, b[0], "ent-b")
        _edge(conn, "ent-a", "ent-b")

        store.vector.delete_by_document("doc-a")

        assert _count(conn, "context_graph_edges") == 0
        assert _count(conn, "entities") == 1

    def test_entity_referenced_only_by_triple_survives(self, store, conn, no_orphans):
        """A triple from a surviving chunk still holds its entities alive."""
        a = _index(store, "doc-a", ["alpha content"])
        b = _index(store, "doc-b", ["beta content"])
        _link_entity(conn, a[0], "ent-a")
        _create_svo_table(conn)
        conn.execute(
            "INSERT INTO svo_triples(chunk_id, subject_id, verb, object_id) VALUES (?,?,?,?)",
            [b[0], "ent-a", "relates-to", "ent-a"],
        )

        store.vector.delete_by_document("doc-a")

        assert [r[0] for r in conn.execute("SELECT id FROM entities").fetchall()] == ["ent-a"]
        no_orphans(conn)

    def test_gc_entities_false_defers_sweep(self, store, conn):
        ids = _index(store, "doc", ["alpha content"])
        _link_entity(conn, ids[0], "ent-only-here")

        store.vector.delete_by_document("doc", gc_entities=False)
        assert _count(conn, "entities") == 1

        assert store.vector.gc_orphaned_entities() == 1
        assert _count(conn, "entities") == 0

    def test_gc_is_idempotent(self, store, conn):
        ids = _index(store, "doc", ["alpha content"])
        _link_entity(conn, ids[0], "ent-only-here")
        store.vector.delete_by_document("doc")

        assert store.vector.gc_orphaned_entities() == 0

    def test_gc_keeps_linked_entities(self, store, conn):
        ids = _index(store, "doc", ["alpha content"])
        _link_entity(conn, ids[0], "ent-live")

        assert store.vector.gc_orphaned_entities() == 0
        assert _count(conn, "entities") == 1


# ---------------------------------------------------------------------------
# Store facade
# ---------------------------------------------------------------------------


class TestStoreDeleteDocument:
    def test_store_delete_document_cascades(self, store, conn, no_orphans):
        ids = _index(store, "doc", ["alpha content"])
        _link_entity(conn, ids[0], "ent-only-here")
        _cluster(conn, ids[0])

        assert store.delete_document("doc") == 1
        assert _count(conn, "chunk_entities") == 0
        assert _count(conn, "chunk_clusters") == 0
        assert _count(conn, "entities") == 0
        no_orphans(conn)

    def test_clear_removes_everything(self, store, conn, no_orphans):
        ids = _index(store, "doc", ["alpha content"])
        _link_entity(conn, ids[0], "ent-only-here")

        store.vector.clear()

        assert _count(conn, "embeddings") == 0
        assert _count(conn, "documents") == 0
        no_orphans(conn)
