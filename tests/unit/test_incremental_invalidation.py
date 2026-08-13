# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 4d17b5e2-9a63-40c8-b1f7-c62e83a09d45

"""Cache-invalidation tests for the incremental update path (fixed.md phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from chonk.ner._build import _check_cache
from chonk.storage._vector import sync_document

_FP = "config-fingerprint"


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


def _index(store, document_name: str, contents: list[str], domain_id: str | None = None):
    from chonk.models import DocumentChunk

    chunks = [
        DocumentChunk(document_name=document_name, content=text, chunk_index=i)
        for i, text in enumerate(contents)
    ]
    store.add_document(chunks, np.zeros((len(chunks), 4), dtype="float32"), domain_id=domain_id)
    store.vector.register_document(document_name, f"hash-{document_name}-{contents}")
    return [
        store.vector._generate_chunk_id(document_name, c.chunk_index, c.content) for c in chunks
    ]


def _mark_processed(conn, chunk_ids: list[str]) -> None:
    """Record chunk_entities rows as build_ner would, plus the config cache row."""
    for cid in chunk_ids:
        conn.execute(
            "INSERT OR IGNORE INTO entities(id, name, display_name, entity_type) VALUES (?,?,?,?)",
            ["ent-x", "ent-x", "ent-x", "concept"],
        )
        conn.execute(
            "INSERT OR REPLACE INTO chunk_entities"
            "(chunk_id, entity_id, frequency, positions_json, score) VALUES (?,?,?,?,?)",
            [cid, "ent-x", 1, "[]", 1.0],
        )
    conn.execute(
        "INSERT OR REPLACE INTO ner_cache(config_fingerprint, chunk_count) VALUES (?, ?)",
        [_FP, len(chunk_ids)],
    )


# ---------------------------------------------------------------------------
# NER gate
# ---------------------------------------------------------------------------


class TestNerGate:
    def test_ner_skips_when_nothing_changed(self, store, conn):
        ids = _index(store, "doc", ["alpha content"])
        _mark_processed(conn, ids)

        early, incremental, skip_ids = _check_cache(conn, _FP, force=False)

        assert early is True
        assert incremental is False
        assert skip_ids == set()

    def test_ner_reruns_after_document_update(self, store, conn):
        """Regression: defect #4 — an updated document's chunks were treated as
        already processed, so entities were never re-extracted."""
        v1 = _index(store, "doc", ["version one content"])
        _mark_processed(conn, v1)

        sync_document(store.vector, "doc", b"v2")
        v2 = _index(store, "doc", ["version two content"])

        early, incremental, skip_ids = _check_cache(conn, _FP, force=False)

        assert early is False
        assert incremental is True
        assert skip_ids.isdisjoint(v2)

    def test_ner_reruns_when_update_keeps_chunk_count_and_prefix(self, store, conn):
        """The collision shape from defect #3: same count, same 100-char prefix."""
        prefix = "P" * 100
        v1 = _index(store, "doc", [prefix + " tail one", prefix + " tail two"])
        _mark_processed(conn, v1)

        sync_document(store.vector, "doc", b"v2")
        v2 = _index(store, "doc", [prefix + " revised one", prefix + " revised two"])

        early, incremental, skip_ids = _check_cache(conn, _FP, force=False)

        assert early is False
        assert skip_ids == set()
        assert set(v1).isdisjoint(v2)

    def test_ner_processes_only_changed_chunks(self, store, conn):
        stable = _index(store, "stable", ["unchanged content"])
        v1 = _index(store, "churn", ["version one content"])
        _mark_processed(conn, stable + v1)

        sync_document(store.vector, "churn", b"v2")
        v2 = _index(store, "churn", ["version two content"])

        early, incremental, skip_ids = _check_cache(conn, _FP, force=False)

        assert early is False
        assert incremental is True
        assert skip_ids == set(stable)
        assert set(v2).isdisjoint(skip_ids)

    def test_ner_processes_new_document(self, store, conn):
        first = _index(store, "first", ["alpha content"])
        _mark_processed(conn, first)
        _index(store, "second", ["beta content"])

        early, incremental, skip_ids = _check_cache(conn, _FP, force=False)

        assert (early, incremental) == (False, True)
        assert skip_ids == set(first)

    def test_ner_full_build_when_config_changed(self, store, conn):
        ids = _index(store, "doc", ["alpha content"])
        _mark_processed(conn, ids)

        early, incremental, skip_ids = _check_cache(conn, "different-config", force=False)

        assert (early, incremental, skip_ids) == (False, False, set())

    def test_force_bypasses_cache(self, store, conn):
        ids = _index(store, "doc", ["alpha content"])
        _mark_processed(conn, ids)

        assert _check_cache(conn, _FP, force=True) == (False, False, set())

    def test_orphaned_chunk_entities_raise(self, store, conn):
        """A missed cascade must fail loudly rather than silently skip chunks."""
        ids = _index(store, "doc", ["alpha content"])
        _mark_processed(conn, ids)
        conn.execute("DELETE FROM embeddings WHERE document_name = ?", ["doc"])

        with pytest.raises(RuntimeError) as exc:
            _check_cache(conn, _FP, force=False)

        assert "absent from embeddings" in str(exc.value)

    def test_prune_leaves_ner_gate_consistent(self, store, conn, no_orphans):
        from chonk.storage import prune_documents

        keep = _index(store, "keep", ["kept content"])
        gone = _index(store, "gone", ["removed content"])
        _mark_processed(conn, keep + gone)

        prune_documents(store.vector, {"keep"})

        early, incremental, skip_ids = _check_cache(conn, _FP, force=False)
        assert early is True
        assert skip_ids == set()
        no_orphans(conn)


# ---------------------------------------------------------------------------
# Community cache
# ---------------------------------------------------------------------------


class TestCommunityCache:
    def test_community_cache_valid_when_unchanged(self, store):
        _index(store, "doc", ["alpha content"], domain_id="dom")
        store.write_community_cache("sess", ["dom"])

        assert store.community_cache_valid("sess", ["dom"]) is True

    def test_community_cache_invalidated_when_content_changes_but_count_does_not(self, store):
        """Regression: defect #5 — validity compared chunk counts, so an update
        producing the same number of chunks left stale community summaries."""
        _index(store, "doc", ["version one a", "version one b"], domain_id="dom")
        store.write_community_cache("sess", ["dom"])
        assert store.community_cache_valid("sess", ["dom"]) is True

        sync_document(store.vector, "doc", b"v2")
        _index(store, "doc", ["version two a", "version two b"], domain_id="dom")

        assert store.count() == 2  # same chunk count as before
        assert store.community_cache_valid("sess", ["dom"]) is False

    def test_community_cache_invalidated_when_document_added(self, store):
        _index(store, "doc", ["alpha content"], domain_id="dom")
        store.write_community_cache("sess", ["dom"])

        _index(store, "extra", ["beta content"], domain_id="dom")

        assert store.community_cache_valid("sess", ["dom"]) is False

    def test_community_cache_invalidated_when_document_pruned(self, store):
        from chonk.storage import prune_documents

        _index(store, "keep", ["alpha content"], domain_id="dom")
        _index(store, "gone", ["beta content"], domain_id="dom")
        store.write_community_cache("sess", ["dom"])

        prune_documents(store.vector, {"keep"})

        assert store.community_cache_valid("sess", ["dom"]) is False

    def test_community_cache_unknown_fingerprint_is_invalid(self, store):
        _index(store, "doc", ["alpha content"], domain_id="dom")

        assert store.community_cache_valid("never-written", ["dom"]) is False

    def test_community_cache_rewrite_revalidates(self, store):
        _index(store, "doc", ["version one"], domain_id="dom")
        store.write_community_cache("sess", ["dom"])

        sync_document(store.vector, "doc", b"v2")
        _index(store, "doc", ["version two"], domain_id="dom")
        assert store.community_cache_valid("sess", ["dom"]) is False

        store.write_community_cache("sess", ["dom"])
        assert store.community_cache_valid("sess", ["dom"]) is True

    def test_community_cache_is_domain_scoped(self, store):
        _index(store, "a", ["alpha content"], domain_id="dom-a")
        _index(store, "b", ["beta content"], domain_id="dom-b")
        store.write_community_cache("sess", ["dom-a"])

        sync_document(store.vector, "b", b"v2")
        _index(store, "b", ["beta revised"], domain_id="dom-b")

        assert store.community_cache_valid("sess", ["dom-a"]) is True


# ---------------------------------------------------------------------------
# Context graph cache
# ---------------------------------------------------------------------------


class TestContextGraphCache:
    def test_context_graph_fingerprint_changes_on_update(self, store, conn):
        from chonk.graph._context_graph import _chunk_fingerprint

        _index(store, "doc", ["version one a", "version one b"])
        before = _chunk_fingerprint(
            [r[0] for r in conn.execute("SELECT chunk_id FROM embeddings").fetchall()]
        )

        sync_document(store.vector, "doc", b"v2")
        _index(store, "doc", ["version two a", "version two b"])
        after = _chunk_fingerprint(
            [r[0] for r in conn.execute("SELECT chunk_id FROM embeddings").fetchall()]
        )

        assert before != after

    def test_context_graph_fingerprint_stable_when_unchanged(self, store, conn):
        from chonk.graph._context_graph import _chunk_fingerprint

        _index(store, "doc", ["alpha content"])
        ids = [r[0] for r in conn.execute("SELECT chunk_id FROM embeddings").fetchall()]

        assert _chunk_fingerprint(ids) == _chunk_fingerprint(list(reversed(ids)))
