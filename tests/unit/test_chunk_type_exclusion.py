# Copyright (c) 2025 Kenneth Stott. MIT License.
"""exclude_chunk_types: a bare search includes synthetic rows; callers opt out."""

from __future__ import annotations

import numpy as np
import pytest

from chonk.models import DocumentChunk
from chonk.storage._schema import SYNTHETIC_CHUNK_TYPES
from chonk.storage._store import Store

DIM = 4


def _chunk(name: str, content: str, chunk_type: str) -> DocumentChunk:
    return DocumentChunk(
        document_name=name,
        content=content,
        chunk_index=0,
        section=[],
        chunk_type=chunk_type,
    )


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "idx.duckdb", embedding_dim=DIM)
    chunks = [
        _chunk("doc_a", "quarterly revenue grew", "document"),
        _chunk("__entity__customer:acme", "acme corp. a customer", "entity"),
        _chunk("__community__1", "summary of the revenue cluster", "community_summary"),
    ]
    vecs = np.ones((3, DIM), dtype="float32")
    store.add_document(chunks, vecs)
    yield store
    store.close()


def test_bare_search_includes_synthetic_rows(store):
    """Per design: a bare search is inclusive — filtering is the caller's job."""
    hits = store.search(np.ones(DIM, dtype="float32"), limit=10)
    assert {c.chunk_type for _, _, c in hits} == {"document", "entity", "community_summary"}


def test_exclude_removes_only_named_types(store):
    hits = store.search(
        np.ones(DIM, dtype="float32"), limit=10, exclude_chunk_types=SYNTHETIC_CHUNK_TYPES
    )
    assert {c.chunk_type for _, _, c in hits} == {"document"}


def test_exclude_single_type(store):
    hits = store.search(np.ones(DIM, dtype="float32"), limit=10, exclude_chunk_types=["entity"])
    assert {c.chunk_type for _, _, c in hits} == {"document", "community_summary"}


def test_exclude_none_is_no_restriction(store):
    hits = store.search(np.ones(DIM, dtype="float32"), limit=10, exclude_chunk_types=None)
    assert len(hits) == 3


def test_include_and_exclude_compose(store):
    hits = store.search(
        np.ones(DIM, dtype="float32"),
        limit=10,
        chunk_types=["document", "entity"],
        exclude_chunk_types=["entity"],
    )
    assert {c.chunk_type for _, _, c in hits} == {"document"}


def test_exclusion_applies_to_hybrid_lane(store):
    hits = store.search(
        np.ones(DIM, dtype="float32"),
        limit=10,
        query_text="revenue",
        exclude_chunk_types=SYNTHETIC_CHUNK_TYPES,
    )
    assert all(c.chunk_type == "document" for _, _, c in hits)


def test_synthetic_types_are_the_generated_ones():
    assert set(SYNTHETIC_CHUNK_TYPES) == {"entity", "community_summary"}
