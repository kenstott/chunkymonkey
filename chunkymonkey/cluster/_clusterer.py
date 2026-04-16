# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Clustering algorithms over the entity co-occurrence matrix.

Supported algorithms:
- ``agglomerative``: scikit-learn AgglomerativeClustering (distance_threshold).
- ``dbscan``: scikit-learn DBSCAN (eps derived from distance_threshold).
- ``leiden``: python-igraph + leidenalg community detection.

All algorithms produce a mapping of entity_id -> cluster_id (string label).
Singleton entities (no co-occurrence neighbours) get their own cluster.
"""

from __future__ import annotations

import uuid
from collections import defaultdict


def cluster_entities(
    cooccurrence: dict[tuple[str, str], float],
    entities: list[str],
    algorithm: str = "agglomerative",
    distance_threshold: float = 0.5,
) -> dict[str, str]:
    """Assign cluster IDs to entities.

    Args:
        cooccurrence: Output of ``CooccurrenceMatrix.build()``.
        entities: All entity IDs to cluster (including singletons).
        algorithm: ``"agglomerative"``, ``"dbscan"``, or ``"leiden"``.
        distance_threshold: Merge threshold for agglomerative;
            neighbourhood radius for DBSCAN; resolution for Leiden.

    Returns:
        Dict mapping entity_id -> cluster_id string.
    """
    if not entities:
        return {}

    if algorithm == "agglomerative":
        return _agglomerative(cooccurrence, entities, distance_threshold)
    if algorithm == "dbscan":
        return _dbscan(cooccurrence, entities, distance_threshold)
    if algorithm == "leiden":
        return _leiden(cooccurrence, entities, distance_threshold)
    raise ValueError(f"Unknown clustering algorithm: {algorithm!r}")


# ------------------------------------------------------------------
# Agglomerative
# ------------------------------------------------------------------

def _agglomerative(
    cooccurrence: dict[tuple[str, str], float],
    entities: list[str],
    distance_threshold: float,
) -> dict[str, str]:
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as e:
        raise ImportError(
            "scikit-learn and numpy are required for agglomerative clustering. "
            "Install with: pip install chunkymonkey[cluster]"
        ) from e

    n = len(entities)
    idx = {e: i for i, e in enumerate(entities)}

    # Build distance matrix (1 - normalised_score clamped to [0, 1])
    dist = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(dist, 0.0)

    # Normalise scores to [0, 1] using min-max
    if cooccurrence:
        vals = list(cooccurrence.values())
        lo, hi = min(vals), max(vals)
        span = hi - lo
    else:
        lo, span = 0.0, 0.0

    for (a, b), score in cooccurrence.items():
        if a not in idx or b not in idx:
            continue
        # When all scores are identical, treat as maximally similar (distance 0)
        sim = (score - lo) / span if span > 0 else 1.0
        d = max(0.0, 1.0 - sim)
        dist[idx[a], idx[b]] = d
        dist[idx[b], idx[a]] = d

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = model.fit_predict(dist)

    return {e: f"cluster_{labels[i]:04d}" for i, e in enumerate(entities)}


# ------------------------------------------------------------------
# DBSCAN
# ------------------------------------------------------------------

def _dbscan(
    cooccurrence: dict[tuple[str, str], float],
    entities: list[str],
    distance_threshold: float,
) -> dict[str, str]:
    try:
        import numpy as np
        from sklearn.cluster import DBSCAN
    except ImportError as e:
        raise ImportError(
            "scikit-learn and numpy are required for DBSCAN clustering. "
            "Install with: pip install chunkymonkey[cluster]"
        ) from e

    n = len(entities)
    idx = {e: i for i, e in enumerate(entities)}

    dist = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(dist, 0.0)

    if cooccurrence:
        vals = list(cooccurrence.values())
        lo, hi = min(vals), max(vals)
        span = hi - lo
    else:
        lo, span = 0.0, 0.0

    for (a, b), score in cooccurrence.items():
        if a not in idx or b not in idx:
            continue
        sim = (score - lo) / span if span > 0 else 1.0
        d = max(0.0, 1.0 - sim)
        dist[idx[a], idx[b]] = d
        dist[idx[b], idx[a]] = d

    model = DBSCAN(eps=distance_threshold, min_samples=2, metric="precomputed")
    labels = model.fit_predict(dist)

    # DBSCAN labels -1 as noise — give each noise entity its own singleton cluster
    cluster_counter = max((l for l in labels if l >= 0), default=-1) + 1
    result = {}
    for i, e in enumerate(entities):
        lbl = labels[i]
        if lbl == -1:
            result[e] = f"cluster_{cluster_counter:04d}"
            cluster_counter += 1
        else:
            result[e] = f"cluster_{lbl:04d}"
    return result


# ------------------------------------------------------------------
# Leiden
# ------------------------------------------------------------------

def _leiden(
    cooccurrence: dict[tuple[str, str], float],
    entities: list[str],
    resolution: float,
) -> dict[str, str]:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        raise ImportError(
            "igraph and leidenalg are required for Leiden clustering. "
            "Install with: pip install chunkymonkey[leiden]"
        ) from e

    idx = {e: i for i, e in enumerate(entities)}
    edges = []
    weights = []
    if cooccurrence:
        vals = list(cooccurrence.values())
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi != lo else 1.0
    else:
        lo, span = 0.0, 1.0

    for (a, b), score in cooccurrence.items():
        if a in idx and b in idx:
            edges.append((idx[a], idx[b]))
            weights.append((score - lo) / span)

    g = ig.Graph(n=len(entities), edges=edges)
    g.es["weight"] = weights if weights else []

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight" if weights else None,
        resolution_parameter=resolution,
    )

    result = {}
    for cluster_idx, members in enumerate(partition):
        for member_idx in members:
            result[entities[member_idx]] = f"cluster_{cluster_idx:04d}"
    return result