# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Unit tests for co-occurrence matrix and cluster map."""

import pytest

from chunkymonkey.ner._vocabulary import VocabularyMatcher
from chunkymonkey.ner._index import EntityIndex
from chunkymonkey.cluster._cooccurrence import CooccurrenceMatrix
from chunkymonkey.cluster._map import ClusterMap
from chunkymonkey.models import ClusterRecord


SAMPLE_ENTITIES = [
    {"id": "ent_a", "name": "entity a", "display_name": "Entity A", "type": "concept", "aliases": []},
    {"id": "ent_b", "name": "entity b", "display_name": "Entity B", "type": "concept", "aliases": []},
    {"id": "ent_c", "name": "entity c", "display_name": "Entity C", "type": "concept", "aliases": []},
    {"id": "ent_d", "name": "entity d", "display_name": "Entity D", "type": "concept", "aliases": []},
]


def _build_index() -> EntityIndex:
    """Build an index with predictable co-occurrence patterns.

    A+B co-occur 3x, C+D co-occur 3x, A+C co-occur 1x.
    Expected clusters: {A,B} and {C,D}.
    """
    matcher = VocabularyMatcher(SAMPLE_ENTITIES)
    idx = EntityIndex()

    # A+B chunks
    idx.run_ner("c1", "entity a entity b together", matcher)
    idx.run_ner("c2", "entity a entity b again", matcher)
    idx.run_ner("c3", "entity a and entity b", matcher)
    # C+D chunks
    idx.run_ner("c4", "entity c entity d together", matcher)
    idx.run_ner("c5", "entity c entity d again", matcher)
    idx.run_ner("c6", "entity c and entity d", matcher)
    # Weak A+C link
    idx.run_ner("c7", "entity a entity c weak link", matcher)

    return idx


class TestCooccurrenceMatrix:
    def test_raw_cooccurrence_counts(self):
        idx = _build_index()
        matrix = CooccurrenceMatrix(idx, normalization="raw", min_cooccurrence=1)
        result = matrix.build()
        ab = result.get(("ent_a", "ent_b")) or result.get(("ent_b", "ent_a"), 0)
        cd = result.get(("ent_c", "ent_d")) or result.get(("ent_d", "ent_c"), 0)
        assert ab == 3.0
        assert cd == 3.0

    def test_min_cooccurrence_filter(self):
        idx = _build_index()
        matrix = CooccurrenceMatrix(idx, normalization="raw", min_cooccurrence=3)
        result = matrix.build()
        # A+C only co-occur once — should be filtered out
        ac = result.get(("ent_a", "ent_c")) or result.get(("ent_c", "ent_a"), 0)
        assert ac == 0

    def test_jaccard_normalisation(self):
        idx = _build_index()
        matrix = CooccurrenceMatrix(idx, normalization="jaccard", min_cooccurrence=1)
        result = matrix.build()
        # Jaccard scores should be in [0, 1]
        for score in result.values():
            assert 0.0 <= score <= 1.0

    def test_pmi_normalisation(self):
        idx = _build_index()
        matrix = CooccurrenceMatrix(idx, normalization="pmi", min_cooccurrence=1)
        result = matrix.build()
        # PMI can be negative for rare co-occurrences; just check it's a float
        for score in result.values():
            assert isinstance(score, float)

    def test_symmetric_pairs(self):
        idx = _build_index()
        matrix = CooccurrenceMatrix(idx, normalization="raw", min_cooccurrence=1)
        result = matrix.build()
        # All keys should be (a, b) where a < b (lexicographic)
        for a, b in result.keys():
            assert a < b


class TestClusterMap:
    def test_build_produces_clusters(self):
        pytest.importorskip("sklearn")
        idx = _build_index()
        cmap = ClusterMap.build(idx, algorithm="agglomerative", min_cooccurrence=2)
        assert cmap.cluster_count() >= 1
        assert cmap.entity_count() >= 1

    def test_ab_in_same_cluster(self):
        pytest.importorskip("sklearn")
        idx = _build_index()
        cmap = ClusterMap.build(idx, algorithm="agglomerative",
                                distance_threshold=0.5, min_cooccurrence=2)
        cluster_a = cmap.get_cluster("ent_a")
        cluster_b = cmap.get_cluster("ent_b")
        assert cluster_a is not None
        assert cluster_a == cluster_b

    def test_neighbors_excludes_self(self):
        pytest.importorskip("sklearn")
        idx = _build_index()
        cmap = ClusterMap.build(idx, algorithm="agglomerative", min_cooccurrence=2)
        neighbors = cmap.get_neighbors("ent_a")
        assert "ent_a" not in neighbors

    def test_no_clusters_on_empty_index(self):
        idx = EntityIndex()
        cmap = ClusterMap.build(idx)
        assert cmap.cluster_count() == 0
        assert cmap.entity_count() == 0
        assert cmap.get_cluster("ent_a") is None
        assert cmap.get_neighbors("ent_a") == []

    def test_serialisation_roundtrip(self):
        pytest.importorskip("sklearn")
        idx = _build_index()
        cmap = ClusterMap.build(idx, algorithm="agglomerative", min_cooccurrence=2)
        data = cmap.to_dict()
        cmap2 = ClusterMap.from_dict(data)
        assert cmap2.cluster_count() == cmap.cluster_count()
        assert cmap2.get_cluster("ent_a") == cmap.get_cluster("ent_a")

    def test_all_clusters_returns_records(self):
        pytest.importorskip("sklearn")
        idx = _build_index()
        cmap = ClusterMap.build(idx, algorithm="agglomerative", min_cooccurrence=2)
        records = cmap.all_clusters()
        assert all(isinstance(r, ClusterRecord) for r in records)

    def test_cohesion_score_non_negative(self):
        pytest.importorskip("sklearn")
        idx = _build_index()
        cmap = ClusterMap.build(idx, algorithm="agglomerative",
                                normalization="raw", min_cooccurrence=2)
        for rec in cmap.all_clusters():
            assert rec.cohesion_score >= 0.0

    def test_dbscan_algorithm(self):
        pytest.importorskip("sklearn")
        idx = _build_index()
        # DBSCAN with relaxed threshold
        cmap = ClusterMap.build(idx, algorithm="dbscan",
                                distance_threshold=0.8, min_cooccurrence=2)
        assert cmap.entity_count() >= 1