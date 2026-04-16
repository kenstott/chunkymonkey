# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""ClusterMap: entity -> cluster assignment and cluster neighbour lookup.

Built from the output of ``cluster_entities()``. Provides the two-hop path
used by the enhanced search: query chunk -> entity -> cluster -> neighbour
entity -> neighbour chunks.
"""

from __future__ import annotations

import json
from collections import defaultdict

from ..models import ClusterRecord
from ._cooccurrence import CooccurrenceMatrix
from ._clusterer import cluster_entities
from ..ner._index import EntityIndex


class ClusterMap:
    """In-memory cluster assignment map with neighbour lookup.

    Build via ``ClusterMap.build()`` rather than the constructor.

    Args:
        assignments: Dict mapping entity_id -> cluster_id.
        cooccurrence: The co-occurrence matrix used to compute cohesion.
    """

    def __init__(self) -> None:
        self._entity_to_cluster: dict[str, str] = {}
        self._cluster_to_entities: dict[str, list[str]] = defaultdict(list)
        self._clusters: dict[str, ClusterRecord] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        entity_index: EntityIndex,
        algorithm: str = "agglomerative",
        distance_threshold: float = 0.5,
        normalization: str = "pmi",
        min_cooccurrence: int = 2,
    ) -> "ClusterMap":
        """Compute clusters from an EntityIndex and return a ClusterMap.

        Args:
            entity_index: Populated EntityIndex.
            algorithm: Clustering algorithm (see ``_clusterer.py``).
            distance_threshold: Merge / neighbourhood threshold.
            normalization: Co-occurrence normalisation method.
            min_cooccurrence: Minimum raw co-occurrence to form an edge.
        """
        entities = entity_index.entity_ids()
        if not entities:
            return cls()

        matrix = CooccurrenceMatrix(
            entity_index,
            normalization=normalization,
            min_cooccurrence=min_cooccurrence,
        )
        cooccurrence = matrix.build()

        assignments = cluster_entities(
            cooccurrence,
            entities,
            algorithm=algorithm,
            distance_threshold=distance_threshold,
        )

        instance = cls()
        instance._build_from_assignments(assignments, cooccurrence)
        return instance

    def _build_from_assignments(
        self,
        assignments: dict[str, str],
        cooccurrence: dict[tuple[str, str], float],
    ) -> None:
        self._entity_to_cluster = dict(assignments)
        self._cluster_to_entities = defaultdict(list)
        for entity_id, cluster_id in assignments.items():
            self._cluster_to_entities[cluster_id].append(entity_id)

        # Compute cohesion score per cluster as mean intra-cluster co-occurrence
        for cluster_id, members in self._cluster_to_entities.items():
            intra_scores = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pair = (min(members[i], members[j]), max(members[i], members[j]))
                    if pair in cooccurrence:
                        intra_scores.append(cooccurrence[pair])
            cohesion = sum(intra_scores) / len(intra_scores) if intra_scores else 0.0
            self._clusters[cluster_id] = ClusterRecord(
                cluster_id=cluster_id,
                entities=list(members),
                cohesion_score=round(cohesion, 4),
            )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_cluster(self, entity_id: str) -> str | None:
        """Return the cluster ID for *entity_id*, or None if unassigned."""
        return self._entity_to_cluster.get(entity_id)

    def get_neighbors(self, entity_id: str) -> list[str]:
        """Return all other entities in the same cluster as *entity_id*."""
        cluster_id = self._entity_to_cluster.get(entity_id)
        if cluster_id is None:
            return []
        return [e for e in self._cluster_to_entities[cluster_id] if e != entity_id]

    def get_cluster_record(self, cluster_id: str) -> ClusterRecord | None:
        """Return the ClusterRecord for *cluster_id*."""
        return self._clusters.get(cluster_id)

    def all_clusters(self) -> list[ClusterRecord]:
        """Return all ClusterRecord objects."""
        return list(self._clusters.values())

    def entity_count(self) -> int:
        return len(self._entity_to_cluster)

    def cluster_count(self) -> int:
        return len(self._clusters)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "assignments": self._entity_to_cluster,
            "clusters": {
                cid: {
                    "cluster_id": rec.cluster_id,
                    "entities": rec.entities,
                    "cohesion_score": rec.cohesion_score,
                }
                for cid, rec in self._clusters.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClusterMap":
        instance = cls()
        instance._entity_to_cluster = data.get("assignments", {})
        instance._cluster_to_entities = defaultdict(list)
        for cluster_data in data.get("clusters", {}).values():
            cid = cluster_data["cluster_id"]
            members = cluster_data["entities"]
            instance._cluster_to_entities[cid] = members
            instance._clusters[cid] = ClusterRecord(
                cluster_id=cid,
                entities=members,
                cohesion_score=cluster_data.get("cohesion_score", 0.0),
            )
        return instance