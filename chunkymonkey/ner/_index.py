# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Bidirectional entity-chunk association index.

Stores entity-chunk associations with association scores computed from
frequency, positional centrality, and entity specificity (IDF-like).

All operations are O(1) lookup after the index is built.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

from ._vocabulary import EntityMatch, VocabularyMatcher
from ..models import EntityAssociation


class EntityIndex:
    """Bidirectional index: entity -> chunks and chunk -> entities.

    Association score formula (tunable weights)::

        score = freq_w  * log(1 + frequency)
              + pos_w   * (1 - first_position / chunk_length)
              + spec_w  * log(total_chunks / chunks_with_entity)

    Default weights: frequency=0.4, position=0.3, specificity=0.3.

    Args:
        score_weights: (frequency_weight, position_weight, specificity_weight).
    """

    def __init__(self, score_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)):
        self._fw, self._pw, self._sw = score_weights
        # entity_id -> {chunk_id -> EntityAssociation}
        self._entity_to_chunks: dict[str, dict[str, EntityAssociation]] = defaultdict(dict)
        # chunk_id -> {entity_id -> EntityAssociation}
        self._chunk_to_entities: dict[str, dict[str, EntityAssociation]] = defaultdict(dict)
        self._total_chunks: int = 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunk(
        self,
        chunk_id: str,
        chunk_content: str,
        matches: list[EntityMatch],
    ) -> list[EntityAssociation]:
        """Record entity associations for one chunk.

        Args:
            chunk_id: Stable chunk identifier (from DuckDB store).
            chunk_content: Raw text of the chunk (used for length).
            matches: EntityMatch objects from VocabularyMatcher.match().

        Returns:
            List of EntityAssociation objects created for this chunk.
        """
        self._total_chunks += 1
        chunk_length = max(len(chunk_content), 1)
        associations: list[EntityAssociation] = []

        for match in matches:
            score = self._compute_score(
                frequency=match.frequency,
                first_position=match.positions[0] if match.positions else 0,
                chunk_length=chunk_length,
                entity_id=match.entity_id,
            )
            assoc = EntityAssociation(
                entity_id=match.entity_id,
                chunk_id=chunk_id,
                frequency=match.frequency,
                positions=match.positions,
                score=score,
                chunk_length=chunk_length,
            )
            self._entity_to_chunks[match.entity_id][chunk_id] = assoc
            self._chunk_to_entities[chunk_id][match.entity_id] = assoc
            associations.append(assoc)

        return associations

    def run_ner(
        self,
        chunk_id: str,
        chunk_content: str,
        matcher: VocabularyMatcher,
    ) -> list[EntityAssociation]:
        """Run vocabulary NER on a chunk and index results.

        Convenience wrapper around ``matcher.match()`` + ``index_chunk()``.
        """
        matches = matcher.match(chunk_content)
        return self.index_chunk(chunk_id, chunk_content, matches)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_chunks_for_entity(
        self,
        entity_id: str,
        top_n: int | None = None,
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, score) pairs for an entity, ranked by score desc."""
        assocs = self._entity_to_chunks.get(entity_id, {})
        ranked = sorted(assocs.values(), key=lambda a: a.score, reverse=True)
        result = [(a.chunk_id, a.score) for a in ranked]
        return result[:top_n] if top_n is not None else result

    def get_entities_for_chunk(
        self,
        chunk_id: str,
    ) -> list[tuple[str, float]]:
        """Return (entity_id, score) pairs for a chunk, ranked by score desc."""
        assocs = self._chunk_to_entities.get(chunk_id, {})
        ranked = sorted(assocs.values(), key=lambda a: a.score, reverse=True)
        return [(a.entity_id, a.score) for a in ranked]

    def get_association(self, entity_id: str, chunk_id: str) -> EntityAssociation | None:
        """Return the association for a specific (entity, chunk) pair."""
        return self._entity_to_chunks.get(entity_id, {}).get(chunk_id)

    def chunks_containing_entity(self, entity_id: str) -> int:
        """Number of indexed chunks that contain *entity_id*."""
        return len(self._entity_to_chunks.get(entity_id, {}))

    def total_chunks(self) -> int:
        """Total number of chunks indexed."""
        return self._total_chunks

    def entity_ids(self) -> list[str]:
        """All entity IDs that appear in at least one chunk."""
        return list(self._entity_to_chunks.keys())

    def chunk_ids(self) -> list[str]:
        """All chunk IDs that contain at least one entity."""
        return list(self._chunk_to_entities.keys())

    # ------------------------------------------------------------------
    # Incremental removal
    # ------------------------------------------------------------------

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove all associations for *chunk_id* from the index."""
        if chunk_id not in self._chunk_to_entities:
            return
        for entity_id in list(self._chunk_to_entities[chunk_id].keys()):
            self._entity_to_chunks.get(entity_id, {}).pop(chunk_id, None)
            if not self._entity_to_chunks.get(entity_id):
                self._entity_to_chunks.pop(entity_id, None)
        del self._chunk_to_entities[chunk_id]
        self._total_chunks = max(0, self._total_chunks - 1)

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        frequency: int,
        first_position: int,
        chunk_length: int,
        entity_id: str,
    ) -> float:
        """Compute association score for one entity-chunk pair.

        Scores are recomputed after indexing, so specificity uses the
        current corpus size at the time of the call. For large corpora,
        recompute via ``recompute_scores()`` after bulk indexing.
        """
        df = self.chunks_containing_entity(entity_id) + 1  # +1 for current chunk
        n = self._total_chunks + 1
        freq_score = math.log1p(frequency)
        pos_score = 1.0 - (first_position / chunk_length)
        spec_score = math.log(n / df) if df > 0 else 0.0
        return (
            self._fw * freq_score
            + self._pw * pos_score
            + self._sw * spec_score
        )

    def recompute_scores(self) -> None:
        """Recompute all association scores with the current corpus size.

        Call after bulk-indexing a large corpus to get accurate specificity
        scores that reflect the full document frequency distribution.
        """
        n = self._total_chunks
        for entity_id, chunk_map in self._entity_to_chunks.items():
            df = len(chunk_map)
            for chunk_id, assoc in chunk_map.items():
                freq_score = math.log1p(assoc.frequency)
                pos_score = 1.0 - (assoc.positions[0] / max(1, assoc.chunk_length))
                spec_score = math.log(n / df) if df > 0 and n > 0 else 0.0
                assoc.score = (
                    self._fw * freq_score
                    + self._pw * pos_score
                    + self._sw * spec_score
                )
                self._chunk_to_entities[chunk_id][entity_id] = assoc

    # ------------------------------------------------------------------
    # Serialisation (simple JSON, no external deps)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise index to a plain dict (JSON-compatible)."""
        assocs = []
        for entity_id, chunk_map in self._entity_to_chunks.items():
            for chunk_id, a in chunk_map.items():
                assocs.append({
                    "entity_id": a.entity_id,
                    "chunk_id": a.chunk_id,
                    "frequency": a.frequency,
                    "positions": a.positions,
                    "score": a.score,
                    "chunk_length": a.chunk_length,
                })
        return {
            "total_chunks": self._total_chunks,
            "score_weights": [self._fw, self._pw, self._sw],
            "associations": assocs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityIndex":
        """Restore an index from a dict produced by ``to_dict()``."""
        weights = tuple(data.get("score_weights", [0.4, 0.3, 0.3]))
        idx = cls(score_weights=weights)  # type: ignore[arg-type]
        idx._total_chunks = data.get("total_chunks", 0)
        for a in data.get("associations", []):
            assoc = EntityAssociation(
                entity_id=a["entity_id"],
                chunk_id=a["chunk_id"],
                frequency=a["frequency"],
                positions=a["positions"],
                score=a["score"],
                chunk_length=a.get("chunk_length", 1),
            )
            idx._entity_to_chunks[a["entity_id"]][a["chunk_id"]] = assoc
            idx._chunk_to_entities[a["chunk_id"]][a["entity_id"]] = assoc
        return idx