# Copyright (c) 2025 Kenneth Stott. MIT License.
"""search(entity_types=[...]) — the denormalized entity_types chunk field."""

from __future__ import annotations

import numpy as np
import pytest

from chonk.models import DocumentChunk
from chonk.storage._store import Store

DIM = 4


def _chunk(name: str, content: str) -> DocumentChunk:
    return DocumentChunk(document_name=name, content=content, chunk_index=0, section=[])


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "idx.duckdb", embedding_dim=DIM)
    chunks = [_chunk("doc_a", "john doe bought a drill"), _chunk("doc_b", "acme corp filed")]
    store.add_document(chunks, np.ones((2, DIM), dtype="float32"))
    yield store
    store.close()


def _ids(store):
    return [c.document_name for _, _, c in store.search(np.ones(DIM, dtype="float32"), limit=10)]


class TestSetChunkEntityTypes:
    def test_written_types_are_filterable(self, store):
        chunk_ids = [
            r[0]
            for r in store.vector._conn.execute(
                "SELECT chunk_id FROM embeddings ORDER BY document_name"
            ).fetchall()
        ]
        store.vector.set_chunk_entity_types(
            {chunk_ids[0]: ["customer", "employee"], chunk_ids[1]: ["org"]}
        )

        q = np.ones(DIM, dtype="float32")
        assert [
            c.document_name for _, _, c in store.search(q, limit=10, entity_types=["customer"])
        ] == ["doc_a"]
        assert [c.document_name for _, _, c in store.search(q, limit=10, entity_types=["org"])] == [
            "doc_b"
        ]
        # Any-overlap semantics.
        assert sorted(
            c.document_name
            for _, _, c in store.search(q, limit=10, entity_types=["employee", "org"])
        ) == ["doc_a", "doc_b"]

    def test_unknown_type_matches_nothing(self, store):
        chunk_ids = [
            r[0] for r in store.vector._conn.execute("SELECT chunk_id FROM embeddings").fetchall()
        ]
        store.vector.set_chunk_entity_types({chunk_ids[0]: ["customer"]})
        assert store.search(np.ones(DIM, dtype="float32"), limit=10, entity_types=["ghost"]) == []

    def test_chunks_without_ner_match_nothing(self, store):
        # entity_types is NULL until build_ner runs.
        assert (
            store.search(np.ones(DIM, dtype="float32"), limit=10, entity_types=["customer"]) == []
        )

    def test_none_is_no_restriction(self, store):
        assert len(_ids(store)) == 2

    def test_empty_mapping_is_a_noop(self, store):
        assert store.vector.set_chunk_entity_types({}) == 0

    def test_types_are_deduped_and_sorted(self, store):
        chunk_ids = [
            r[0] for r in store.vector._conn.execute("SELECT chunk_id FROM embeddings").fetchall()
        ]
        store.vector.set_chunk_entity_types({chunk_ids[0]: ["b", "a", "b"]})
        stored = store.vector._conn.execute(
            "SELECT entity_types FROM embeddings WHERE chunk_id = ?", [chunk_ids[0]]
        ).fetchone()
        assert list(stored[0]) == ["a", "b"]

    def test_filter_applies_to_hybrid_lane(self, store):
        chunk_ids = [
            r[0]
            for r in store.vector._conn.execute(
                "SELECT chunk_id FROM embeddings ORDER BY document_name"
            ).fetchall()
        ]
        store.vector.set_chunk_entity_types({chunk_ids[0]: ["customer"], chunk_ids[1]: ["org"]})
        hits = store.search(
            np.ones(DIM, dtype="float32"), limit=10, query_text="drill", entity_types=["customer"]
        )
        assert [c.document_name for _, _, c in hits] == ["doc_a"]


class TestDenormalizeFromAssociations:
    def test_types_come_from_the_typed_entity_id(self, store):
        from chonk.ner._build import _denormalize_entity_types

        chunk_ids = [
            r[0]
            for r in store.vector._conn.execute(
                "SELECT chunk_id FROM embeddings ORDER BY document_name"
            ).fetchall()
        ]
        data = {
            "associations": [
                {"chunk_id": chunk_ids[0], "entity_id": "customer:john_doe"},
                {"chunk_id": chunk_ids[0], "entity_id": "employee:john_doe"},
                {"chunk_id": chunk_ids[1], "entity_id": "org:acme_corp"},
            ]
        }
        assert _denormalize_entity_types(store, data) == 2

        q = np.ones(DIM, dtype="float32")
        assert [
            c.document_name for _, _, c in store.search(q, limit=10, entity_types=["employee"])
        ] == ["doc_a"]
        stored = store.vector._conn.execute(
            "SELECT entity_types FROM embeddings WHERE chunk_id = ?", [chunk_ids[0]]
        ).fetchone()
        assert list(stored[0]) == ["customer", "employee"]

    def test_untyped_legacy_id_contributes_nothing(self, store):
        from chonk.ner._build import _denormalize_entity_types

        chunk_ids = [
            r[0] for r in store.vector._conn.execute("SELECT chunk_id FROM embeddings").fetchall()
        ]
        assert (
            _denormalize_entity_types(
                store, {"associations": [{"chunk_id": chunk_ids[0], "entity_id": "john_doe"}]}
            )
            == 0
        )


class TestEndToEndThroughBuildNer:
    def test_build_ner_populates_entity_types(self, store):
        pytest.importorskip("spacy")
        pytest.importorskip("en_core_web_sm")
        from chonk.ner._build import build_ner

        store.vector._conn.execute("DELETE FROM embeddings")
        store.add_document(
            [
                _chunk("d1", "John Doe bought a drill at Walmart."),
                _chunk("d2", "Nothing relevant here at all."),
            ],
            np.ones((2, DIM), dtype="float32"),
        )
        build_ner(
            store,
            vocab_entities=[
                {
                    "type": "static",
                    "entity_type": "customer",
                    "names": ["John Doe"],
                    "namespace": "walmart",
                },
                {
                    "type": "static",
                    "entity_type": "employee",
                    "names": ["John Doe"],
                    "namespace": "walmart",
                },
            ],
        )

        q = np.ones(DIM, dtype="float32")
        # Both declared types reached the chunk row from one mention.
        for etype in ("customer", "employee"):
            assert [
                c.document_name for _, _, c in store.search(q, limit=10, entity_types=[etype])
            ] == ["d1"]
        # A chunk with no entities keeps a NULL entity_types and matches nothing.
        assert "d2" not in [
            c.document_name for _, _, c in store.search(q, limit=10, entity_types=["customer"])
        ]
