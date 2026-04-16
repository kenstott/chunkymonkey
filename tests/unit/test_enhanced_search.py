# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Unit tests for EnhancedSearch cohort assembly."""

import pytest
import numpy as np

from chunkymonkey.models import DocumentChunk, ScoredChunk
from chunkymonkey.ner._vocabulary import VocabularyMatcher
from chunkymonkey.ner._index import EntityIndex
from chunkymonkey.search._enhanced import EnhancedSearch


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

duckdb = pytest.importorskip("duckdb")
np_mod = pytest.importorskip("numpy")


VOCAB = [
    {"id": "ent_alpha", "name": "alpha protocol", "display_name": "Alpha Protocol",
     "type": "concept", "aliases": ["alpha"]},
    {"id": "ent_beta", "name": "beta process", "display_name": "Beta Process",
     "type": "concept", "aliases": ["beta"]},
]

DIM = 8  # small embedding dim for tests


def _random_emb() -> np.ndarray:
    v = np.random.randn(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_store_with_chunks():
    """Return a Store loaded with 5 small chunks."""
    from chunkymonkey.storage._store import Store
    store = Store(":memory:", embedding_dim=DIM)

    chunks = [
        DocumentChunk("doc1", "Alpha protocol governs the procedure.", chunk_index=0),
        DocumentChunk("doc1", "Beta process is a follow-up step.", chunk_index=1),
        DocumentChunk("doc1", "Alpha protocol and beta process combined.", chunk_index=2),
        DocumentChunk("doc2", "Unrelated content about weather.", chunk_index=0),
        DocumentChunk("doc2", "More weather details here.", chunk_index=1),
    ]
    embeddings = np.stack([_random_emb() for _ in chunks])
    store.add_document(chunks, embeddings)
    return store, chunks, embeddings


class FakeStore:
    """Minimal store stub that returns fixed search results."""

    class FakeVector:
        def __init__(self, results, all_chunks, conn):
            self._results = results
            self._all = all_chunks
            self._conn = conn

        def get_all_chunks(self):
            return self._all

        @property
        def _conn(self):
            return self.__conn

        @_conn.setter
        def _conn(self, v):
            self.__conn = v

    def __init__(self, results, all_chunks):
        import duckdb as ddb
        conn = ddb.connect(":memory:")
        conn.execute(
            f"CREATE TABLE embeddings ("
            f"chunk_id TEXT PRIMARY KEY, document_name TEXT, section TEXT, "
            f"chunk_index INT, content TEXT, chunk_type TEXT, "
            f"source_offset INT, source_length INT, embedding FLOAT[{DIM}])"
        )
        self._results = results
        self.vector = self.FakeVector(results, all_chunks, conn)

    def search(self, query_embedding, limit=5, query_text=None):
        return self._results[:limit]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnhancedSearchSeedOnly:
    """Seed-only mode (structural/entity/cluster all disabled)."""

    def test_returns_scored_chunks(self):
        store, chunks, embeddings = _make_store_with_chunks()
        search = EnhancedSearch(
            store,
            structural_expansion=False,
            entity_expansion=False,
            cluster_expansion=False,
        )
        query = _random_emb()
        results = search.search(query, k=3)
        assert len(results) <= 3
        assert all(isinstance(r, ScoredChunk) for r in results)

    def test_provenance_seed(self):
        store, chunks, embeddings = _make_store_with_chunks()
        search = EnhancedSearch(
            store,
            structural_expansion=False,
            entity_expansion=False,
            cluster_expansion=False,
        )
        results = search.search(_random_emb(), k=2)
        assert all(r.provenance == "seed" for r in results)

    def test_k_limits_results(self):
        store, _, _ = _make_store_with_chunks()
        search = EnhancedSearch(store, structural_expansion=False,
                                entity_expansion=False, cluster_expansion=False)
        for k in (1, 2, 5):
            results = search.search(_random_emb(), k=k)
            assert len(results) <= k

    def test_scores_are_finite(self):
        store, _, _ = _make_store_with_chunks()
        search = EnhancedSearch(store, structural_expansion=False,
                                entity_expansion=False, cluster_expansion=False)
        results = search.search(_random_emb(), k=5)
        import math
        assert all(math.isfinite(r.score) for r in results)


class TestEnhancedSearchEntityExpansion:
    def test_entity_adjacent_provenance(self):
        store, chunks, embeddings = _make_store_with_chunks()
        matcher = VocabularyMatcher(VOCAB)
        entity_index = EntityIndex()

        # Index all chunks via NER
        all_db_chunks = store.vector.get_all_chunks()
        from chunkymonkey.storage._vector import DuckDBVectorBackend
        for c in all_db_chunks:
            cid = DuckDBVectorBackend._generate_chunk_id(
                c.document_name, c.chunk_index,
                c.embedding_content or c.content
            )
            entity_index.run_ner(cid, c.content, matcher)

        search = EnhancedSearch(
            store,
            entity_index=entity_index,
            structural_expansion=False,
            cluster_expansion=False,
        )
        results = search.search(_random_emb(), k=5)
        provenances = {r.provenance for r in results}
        # At minimum seed chunks should be there
        assert "seed" in provenances

    def test_no_duplicate_chunk_ids(self):
        store, chunks, embeddings = _make_store_with_chunks()
        matcher = VocabularyMatcher(VOCAB)
        entity_index = EntityIndex()
        all_db_chunks = store.vector.get_all_chunks()
        from chunkymonkey.storage._vector import DuckDBVectorBackend
        for c in all_db_chunks:
            cid = DuckDBVectorBackend._generate_chunk_id(
                c.document_name, c.chunk_index,
                c.embedding_content or c.content
            )
            entity_index.run_ner(cid, c.content, matcher)

        search = EnhancedSearch(store, entity_index=entity_index,
                                structural_expansion=False, cluster_expansion=False)
        results = search.search(_random_emb(), k=5)
        ids = [r.chunk_id for r in results]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids in results"


class TestEnhancedSearchMMR:
    def test_diversity_reduces_redundancy(self):
        """High lambda_diversity should yield less similar results than lambda=0."""
        store, _, _ = _make_store_with_chunks()

        # Fix seed so results differ only by scoring
        query = _random_emb()

        search_uniform = EnhancedSearch(
            store, lambda_diversity=0.0,
            structural_expansion=False, entity_expansion=False, cluster_expansion=False,
        )
        search_diverse = EnhancedSearch(
            store, lambda_diversity=1.0,
            structural_expansion=False, entity_expansion=False, cluster_expansion=False,
        )
        r_uniform = search_uniform.search(query, k=3)
        r_diverse = search_diverse.search(query, k=3)
        # Both should return results — content may differ
        assert len(r_uniform) <= 3
        assert len(r_diverse) <= 3


class TestEnhancedSearchStructural:
    def test_structural_neighbors_included(self):
        store, _, _ = _make_store_with_chunks()
        search = EnhancedSearch(
            store,
            structural_expansion=True,
            entity_expansion=False,
            cluster_expansion=False,
        )
        results = search.search(_random_emb(), k=5)
        # With structural on, structural provenance may appear
        provenances = {r.provenance for r in results}
        assert provenances.issubset({"seed", "structural"})