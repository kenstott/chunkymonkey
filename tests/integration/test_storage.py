# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 9c3c753b-dc25-4887-93b9-efc45c77f98b
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Integration tests for chunkymonkey storage (requires chunkymonkey[storage])."""

import pytest

try:
    import duckdb
    import numpy as np
    from chunkymonkey.storage import Store
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not STORAGE_AVAILABLE,
    reason="chunkymonkey[storage] not installed — pip install chunkymonkey[storage]",
)

from chunkymonkey.models import DocumentChunk


def make_chunks(n=3):
    return [
        DocumentChunk(
            document_name="test_doc",
            content=f"Content of chunk {i}",
            chunk_index=i,
            section=f"Section {i}",
        )
        for i in range(n)
    ]


def make_embeddings(n=3, dim=4):
    np.random.seed(42)
    return np.random.rand(n, dim).astype(np.float32)


class TestStore:
    @pytest.fixture
    def store(self):
        s = Store(db_path=":memory:", embedding_dim=4)
        yield s
        s.close()

    def test_add_and_count(self, store):
        store.add_document(make_chunks(3), make_embeddings(3))
        assert store.count() == 3

    def test_empty_store_count_zero(self, store):
        assert store.count() == 0

    def test_search_returns_results(self, store):
        chunks = make_chunks(3)
        embeddings = make_embeddings(3)
        store.add_document(chunks, embeddings)
        query = embeddings[0]
        results = store.search(query, limit=3)
        assert len(results) > 0

    def test_search_result_structure(self, store):
        chunks = make_chunks(3)
        embeddings = make_embeddings(3)
        store.add_document(chunks, embeddings)
        query = embeddings[0]
        results = store.search(query, limit=3)
        chunk_id, score, chunk = results[0]
        assert isinstance(score, float)
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(chunk_id, str)

    def test_delete_by_document(self, store):
        store.add_document(make_chunks(3), make_embeddings(3))
        store.delete_document("test_doc")
        assert store.count() == 0

    def test_delete_nonexistent_document(self, store):
        store.add_document(make_chunks(3), make_embeddings(3))
        store.delete_document("no_such_doc")
        assert store.count() == 3

    def test_clear(self, store):
        store.add_document(make_chunks(3), make_embeddings(3))
        store.vector.clear()
        assert store.count() == 0

    def test_add_multiple_documents(self, store):
        chunks_a = [
            DocumentChunk(document_name="doc_a", content=f"A chunk {i}", chunk_index=i)
            for i in range(2)
        ]
        chunks_b = [
            DocumentChunk(document_name="doc_b", content=f"B chunk {i}", chunk_index=i)
            for i in range(3)
        ]
        np.random.seed(0)
        emb_a = np.random.rand(2, 4).astype(np.float32)
        emb_b = np.random.rand(3, 4).astype(np.float32)
        store.add_document(chunks_a, emb_a)
        store.add_document(chunks_b, emb_b)
        assert store.count() == 5

    def test_delete_one_document_leaves_other(self, store):
        chunks_a = [DocumentChunk(document_name="doc_a", content="A", chunk_index=0)]
        chunks_b = [DocumentChunk(document_name="doc_b", content="B", chunk_index=0)]
        np.random.seed(1)
        emb_a = np.random.rand(1, 4).astype(np.float32)
        emb_b = np.random.rand(1, 4).astype(np.float32)
        store.add_document(chunks_a, emb_a)
        store.add_document(chunks_b, emb_b)
        store.delete_document("doc_a")
        assert store.count() == 1

    def test_context_manager(self):
        with Store(db_path=":memory:", embedding_dim=4) as store:
            store.add_document(make_chunks(2), make_embeddings(2))
            assert store.count() == 2

    def test_search_limit_respected(self, store):
        store.add_document(make_chunks(5), make_embeddings(5))
        np.random.seed(99)
        query = np.random.rand(4).astype(np.float32)
        results = store.search(query, limit=2)
        assert len(results) <= 2

    def test_add_empty_chunks_no_error(self, store):
        np.random.seed(0)
        store.add_document([], np.random.rand(0, 4).astype(np.float32))
        assert store.count() == 0
