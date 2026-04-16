# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Co-occurrence matrix construction from an EntityIndex.

Two entities co-occur when they appear in the same chunk. The raw count is
normalised by PMI, Jaccard, or left as raw counts.
"""

from __future__ import annotations

import math
from collections import defaultdict

from ..ner._index import EntityIndex


class CooccurrenceMatrix:
    """Build a normalised entity co-occurrence matrix.

    Args:
        entity_index: A populated EntityIndex.
        normalization: ``"pmi"`` (default), ``"jaccard"``, or ``"raw"``.
        min_cooccurrence: Edges with raw count below this are dropped.
    """

    def __init__(
        self,
        entity_index: EntityIndex,
        normalization: str = "pmi",
        min_cooccurrence: int = 2,
    ):
        self._index = entity_index
        self._normalization = normalization
        self._min_cooccurrence = min_cooccurrence

    def build(self) -> dict[tuple[str, str], float]:
        """Compute and return the co-occurrence matrix.

        Returns:
            Dict mapping (entity_id_a, entity_id_b) -> normalised score,
            where entity_id_a < entity_id_b (lower-triangular, no duplicates).
        """
        # Count co-occurrences per chunk
        raw: dict[tuple[str, str], int] = defaultdict(int)
        entity_df: dict[str, int] = {}  # entity -> number of chunks it appears in

        for chunk_id in self._index.chunk_ids():
            entities_in_chunk = [eid for eid, _ in self._index.get_entities_for_chunk(chunk_id)]
            for eid in entities_in_chunk:
                entity_df[eid] = entity_df.get(eid, 0) + 1
            # All pairs in this chunk
            for i in range(len(entities_in_chunk)):
                for j in range(i + 1, len(entities_in_chunk)):
                    pair = (
                        min(entities_in_chunk[i], entities_in_chunk[j]),
                        max(entities_in_chunk[i], entities_in_chunk[j]),
                    )
                    raw[pair] += 1

        total = max(self._index.total_chunks(), 1)

        result: dict[tuple[str, str], float] = {}
        for (a, b), count in raw.items():
            if count < self._min_cooccurrence:
                continue
            score = self._normalise(count, entity_df.get(a, 1), entity_df.get(b, 1), total)
            result[(a, b)] = score

        return result

    def _normalise(self, count: int, df_a: int, df_b: int, total: int) -> float:
        if self._normalization == "raw":
            return float(count)
        if self._normalization == "jaccard":
            union = df_a + df_b - count
            return count / union if union > 0 else 0.0
        # PMI (default)
        p_ab = count / total
        p_a = df_a / total
        p_b = df_b / total
        if p_a == 0 or p_b == 0 or p_ab == 0:
            return 0.0
        return math.log2(p_ab / (p_a * p_b))