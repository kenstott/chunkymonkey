# Copyright (c) 2025 Kenneth Stott. MIT License.
"""get_entity_namespace_evidence — which namespaces actually mention an entity."""

from __future__ import annotations

import numpy as np
import pytest

from chonk.models import DocumentChunk
from chonk.storage._store import NamespaceEvidence, Store

DIM = 4
EID = "customer:acme_corp"


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "idx.duckdb", embedding_dim=DIM)
    yield store
    store.close()


def _seed(store, plan: list[tuple[str, str]]) -> list[str]:
    """plan is [(namespace, content), ...]; returns chunk_ids in insertion order."""
    for i, (ns, content) in enumerate(plan):
        store.add_document(
            [DocumentChunk(document_name=f"d{i}", content=content, chunk_index=0, section=[])],
            np.ones((1, DIM), dtype="float32"),
            namespace=ns,
        )
    rows = store.vector._conn.execute(
        "SELECT chunk_id FROM embeddings ORDER BY document_name"
    ).fetchall()
    return [r[0] for r in rows]


def _associate(store, pairs: list[tuple[str, str, float]]) -> None:
    """pairs is [(chunk_id, namespace, score), ...]."""
    for chunk_id, ns, score in pairs:
        store.vector._conn.execute(
            "INSERT INTO chunk_entities(chunk_id, entity_id, frequency, positions_json, "
            "score, namespace) VALUES (?, ?, 1, '[]', ?, ?)",
            [chunk_id, EID, score, ns],
        )


class TestEvidence:
    def test_ranks_namespaces_and_normalizes_by_corpus_size(self, store):
        # retail: 2 of 2 chunks mention it. support: 1 of 4.
        ids = _seed(
            store,
            [
                ("retail", "a"),
                ("retail", "b"),
                ("support", "c"),
                ("support", "d"),
                ("support", "e"),
                ("support", "f"),
            ],
        )
        _associate(
            store, [(ids[0], "retail", 0.8), (ids[1], "retail", 0.8), (ids[2], "support", 0.4)]
        )

        ev = store.get_entity_namespace_evidence(EID)
        assert [e.namespace for e in ev] == ["retail", "support"]
        assert [e.chunk_count for e in ev] == [2, 1]
        assert ev[0].share == pytest.approx(1.0)
        assert ev[1].share == pytest.approx(0.25)
        assert ev[0].score == pytest.approx(1.6)

    def test_share_beats_raw_count_for_a_small_focused_namespace(self, store):
        # boutique: 2 of 2 (all about it). bulk: 3 of 100 (barely mentions it).
        plan = [("boutique", "a"), ("boutique", "b")] + [("bulk", f"n{i}") for i in range(100)]
        _seed(store, plan)
        rows = store.vector._conn.execute("SELECT chunk_id, namespace FROM embeddings").fetchall()
        boutique = [c for c, n in rows if n == "boutique"]
        bulk = [c for c, n in rows if n == "bulk"]
        _associate(store, [(c, "boutique", 0.5) for c in boutique])
        _associate(store, [(c, "bulk", 0.5) for c in bulk[:3]])

        ev = {e.namespace: e for e in store.get_entity_namespace_evidence(EID)}
        # Raw count says bulk; share says boutique. Both are exposed.
        assert ev["bulk"].chunk_count > ev["boutique"].chunk_count
        assert ev["boutique"].share > ev["bulk"].share
        assert ev["boutique"].share == pytest.approx(1.0)
        assert ev["bulk"].share == pytest.approx(0.03)

    def test_unknown_entity_returns_empty(self, store):
        _seed(store, [("retail", "a")])
        assert store.get_entity_namespace_evidence("customer:nobody") == []

    def test_null_namespace_counts_as_global(self, store):
        ids = _seed(store, [(None, "a"), (None, "b")])
        _associate(store, [(ids[0], None, 0.5)])
        ev = store.get_entity_namespace_evidence(EID)
        assert [e.namespace for e in ev] == ["global"]
        assert ev[0].share == pytest.approx(0.5)

    def test_returns_namespace_evidence_objects(self, store):
        ids = _seed(store, [("retail", "a")])
        _associate(store, [(ids[0], "retail", 0.5)])
        (ev,) = store.get_entity_namespace_evidence(EID)
        assert isinstance(ev, NamespaceEvidence)
        assert (ev.namespace, ev.chunk_count) == ("retail", 1)

    def test_ordering_is_deterministic_on_ties(self, store):
        ids = _seed(store, [("bbb", "a"), ("aaa", "b")])
        _associate(store, [(ids[0], "bbb", 0.5), (ids[1], "aaa", 0.5)])
        assert [e.namespace for e in store.get_entity_namespace_evidence(EID)] == ["aaa", "bbb"]


class TestDeclarationVsEvidence:
    def test_declaration_and_evidence_answer_different_questions(self, store):
        ids = _seed(store, [("retail", "a"), ("support", "b")])
        _associate(store, [(ids[0], "retail", 0.5), (ids[1], "support", 0.5)])
        # Declared once, centrally.
        store.add_entity_alias("acme corp", EID, source="vocab_source", namespace="global")

        assert store.get_entity_namespaces(EID) == ["global"]
        assert [e.namespace for e in store.get_entity_namespace_evidence(EID)] == [
            "retail",
            "support",
        ]
