# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Vocabulary-based entity matcher for NER.

Supports JSON and plain-text vocabulary files. Matching is case-insensitive
by default. Alias expansion maps short forms to canonical entity IDs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EntityMatch:
    """A single entity matched within a chunk of text."""
    entity_id: str
    name: str
    display_name: str
    entity_type: str
    frequency: int
    positions: list[int]        # character offsets of each match start
    spans: list[tuple[int, int]] = field(default_factory=list)  # (start, end) pairs


class VocabularyMatcher:
    """Match entities against a pre-defined vocabulary.

    Vocabulary JSON format (list of objects)::

        [
          {
            "id": "ent_hca",
            "name": "hca healthcare",
            "display_name": "HCA Healthcare",
            "type": "company",
            "aliases": ["hca", "hospital corporation of america"]
          }
        ]

    Plain-text format (one display name per line) auto-generates ``id`` and
    ``name`` from the display name.

    Args:
        entities: Pre-loaded list of entity dicts with keys
            ``id``, ``name``, ``display_name``, ``type``, ``aliases``.
        match_mode: ``"exact"`` or ``"case_insensitive"`` (default).
        min_entity_length: Minimum character length for an entity to be matched.
    """

    def __init__(
        self,
        entities: list[dict],
        match_mode: str = "case_insensitive",
        min_entity_length: int = 2,
    ):
        self._match_mode = match_mode
        self._min_len = min_entity_length
        # Map from normalised surface form -> (entity_id, display_name, type)
        self._lookup: dict[str, tuple[str, str, str]] = {}
        # Track canonical entity metadata
        self._entities: dict[str, dict] = {}

        for ent in entities:
            eid = ent["id"]
            dtype = ent.get("type", ent.get("entity_type", "concept"))
            self._entities[eid] = {
                "id": eid,
                "name": ent["name"],
                "display_name": ent["display_name"],
                "type": dtype,
            }
            surfaces = [ent["name"], ent["display_name"]] + list(ent.get("aliases", []))
            for surface in surfaces:
                if len(surface) < self._min_len:
                    continue
                key = surface if match_mode == "exact" else surface.lower()
                self._lookup[key] = (eid, ent["display_name"], dtype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, text: str) -> list[EntityMatch]:
        """Find all vocabulary entities in *text*.

        Returns one ``EntityMatch`` per unique entity found, aggregating
        frequency and positions across all surface forms.
        """
        check_text = text if self._match_mode == "exact" else text.lower()
        # entity_id -> {positions: list, display_name, type}
        found: dict[str, dict] = {}

        for surface, (eid, display_name, etype) in self._lookup.items():
            start = 0
            while True:
                pos = check_text.find(surface, start)
                if pos == -1:
                    break
                # Require word boundaries to avoid substring false positives
                before_ok = pos == 0 or not check_text[pos - 1].isalnum()
                after_pos = pos + len(surface)
                after_ok = after_pos >= len(check_text) or not check_text[after_pos].isalnum()
                if before_ok and after_ok:
                    if eid not in found:
                        found[eid] = {
                            "spans": [],
                            "display_name": display_name,
                            "type": etype,
                        }
                    found[eid]["spans"].append((pos, pos + len(surface)))
                start = pos + 1

        results = []
        for eid, info in found.items():
            meta = self._entities[eid]
            spans = sorted(info["spans"])
            results.append(EntityMatch(
                entity_id=eid,
                name=meta["name"],
                display_name=info["display_name"],
                entity_type=info["type"],
                frequency=len(spans),
                positions=[s[0] for s in spans],
                spans=spans,
            ))
        return results

    def entity_ids(self) -> list[str]:
        """Return all entity IDs in the vocabulary."""
        return list(self._entities.keys())

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        match_mode: str = "case_insensitive",
        min_entity_length: int = 2,
    ) -> "VocabularyMatcher":
        """Load a VocabularyMatcher from a JSON or plain-text file.

        JSON files must contain a list of entity objects (see class docstring).
        Plain-text files list one display name per line; IDs are auto-generated
        as ``ent_<normalised_name>``.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            raw = json.loads(text)
            entities = []
            for item in raw:
                entities.append({
                    "id": item.get("id", _auto_id(item.get("name", item.get("display_name", "")))),
                    "name": item.get("name", item.get("display_name", "")).lower(),
                    "display_name": item.get("display_name", item.get("name", "")),
                    "type": item.get("type", item.get("entity_type", "concept")),
                    "aliases": item.get("aliases", []),
                })
        else:
            # Plain text: one display name per line
            entities = []
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                entities.append({
                    "id": _auto_id(line),
                    "name": line.lower(),
                    "display_name": line,
                    "type": "concept",
                    "aliases": [],
                })
        return cls(entities, match_mode=match_mode, min_entity_length=min_entity_length)


def _auto_id(display_name: str) -> str:
    """Generate a stable entity ID from a display name."""
    slug = re.sub(r"[^a-z0-9]+", "_", display_name.lower()).strip("_")
    return f"ent_{slug}"