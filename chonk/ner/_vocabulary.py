# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: af2da028-0858-459c-afa0-be0fa46ec83f
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
from typing import Any


@dataclass
class EntityMatch:
    """A single entity matched within a chunk of text."""

    entity_id: str
    name: str
    display_name: str
    entity_type: str
    frequency: int
    positions: list[int]  # character offsets of each match start
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
        normalize_separators: Collapse runs of whitespace and connector
            punctuation (``-``, ``_``, ``/``) to a single space on both the
            vocabulary surfaces and the searched text, so ``"Acme-Corp"`` and
            ``"Acme  Corp"`` match a surface stored as ``"Acme Corp"``.
            Reported spans are always offsets into the original text.
    """

    def __init__(
        self,
        entities: list[dict[str, Any]],
        match_mode: str = "case_insensitive",
        min_entity_length: int = 2,
        normalize_separators: bool = True,
    ) -> None:
        self._match_mode = match_mode
        self._min_len = min_entity_length
        self._normalize = normalize_separators
        # Map from normalised surface form -> [(entity_id, display_name, type), ...].
        # A surface can name more than one entity: "John Doe" may be both
        # customer:john_doe and employee:john_doe. Every mapping is kept, so a
        # mention counts as evidence for each.
        self._lookup: dict[str, list[tuple[str, str, str]]] = {}
        # Track canonical entity metadata
        self._entities: dict[str, dict[str, Any]] = {}

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
                if self._normalize:
                    key = normalize_surface(key)
                    if not key:
                        continue
                mapping = (eid, ent["display_name"], dtype)
                bucket = self._lookup.setdefault(key, [])
                if mapping not in bucket:
                    bucket.append(mapping)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, text: str) -> list[EntityMatch]:
        """Find all vocabulary entities in *text*.

        Returns one ``EntityMatch`` per unique entity found, aggregating
        frequency and positions across all surface forms.
        """
        check_text = text if self._match_mode == "exact" else text.lower()
        # Scan normalised text so separator variants ("Acme-Corp", "Acme  Corp")
        # hit the same surface; index_map carries spans back to the original
        # offsets the caller sees.
        if self._normalize:
            check_text, index_map = normalize_separators(check_text)
        else:
            index_map = list(range(len(check_text)))
        # entity_id -> {positions: list, display_name, type}
        found: dict[str, dict[str, Any]] = {}

        for surface, mappings in self._lookup.items():
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
                    span_start = index_map[pos]
                    span_end = index_map[after_pos - 1] + 1
                    for eid, display_name, etype in mappings:
                        if eid not in found:
                            found[eid] = {
                                "spans": [],
                                "display_name": display_name,
                                "type": etype,
                            }
                        found[eid]["spans"].append((span_start, span_end))
                start = pos + 1

        results = []
        for eid, info in found.items():
            meta = self._entities[eid]
            spans = sorted(info["spans"])
            results.append(
                EntityMatch(
                    entity_id=eid,
                    name=meta["name"],
                    display_name=info["display_name"],
                    entity_type=info["type"],
                    frequency=len(spans),
                    positions=[s[0] for s in spans],
                    spans=spans,
                )
            )
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
    ) -> VocabularyMatcher:
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
                item_type = item.get("type", item.get("entity_type", "concept"))
                item_name = item.get("name", item.get("display_name", ""))
                entities.append(
                    {
                        "id": item.get("id", _typed_id(item_name, item_type)),
                        "name": item_name.lower(),
                        "display_name": item.get("display_name", item.get("name", "")),
                        "type": item_type,
                        "aliases": item.get("aliases", []),
                    }
                )
        else:
            # Plain text: one display name per line
            entities = []
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                entities.append(
                    {
                        "id": _typed_id(line, "concept"),
                        "name": line.lower(),
                        "display_name": line,
                        "type": "concept",
                        "aliases": [],
                    }
                )
        return cls(entities, match_mode=match_mode, min_entity_length=min_entity_length)


# Characters that separate the words of a name without being part of it.
# Sentence punctuation (. , ; : ! ?) is deliberately excluded: collapsing it
# would make "Acme Corp" match across the sentence break in "…Acme. Corp…".
_SEPARATORS = frozenset(" \t\n\r\f\v-‐‑‒–—_/\\| ")


def normalize_separators(text: str) -> tuple[str, list[int]]:
    """Collapse every run of separator characters to a single space.

    Vocabulary matching is a literal substring scan, so ``"Acme  Corp"``,
    ``"Acme-Corp"``, and ``"Acme\\nCorp"`` would each miss a surface stored as
    ``"Acme Corp"``. Normalising both sides makes them match.

    Returns:
        ``(normalized_text, index_map)`` where ``index_map[i]`` is the offset in
        *text* of ``normalized_text[i]``, so match spans map back to offsets in
        the original string.
    """
    out: list[str] = []
    index_map: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] in _SEPARATORS:
            run_start = i
            while i < n and text[i] in _SEPARATORS:
                i += 1
            out.append(" ")
            index_map.append(run_start)
        else:
            out.append(text[i])
            index_map.append(i)
            i += 1
    return "".join(out), index_map


def normalize_surface(surface: str) -> str:
    """Normalise a vocabulary surface form the same way :func:`normalize_separators` does."""
    normalized, _ = normalize_separators(surface)
    return normalized.strip()


def _auto_id(display_name: str) -> str:
    """Generate a stable name slug from a display name.

    Not an entity ID on its own — see :func:`_typed_id`. Every non-alphanumeric
    character collapses to ``_``, so the result can never contain ``:``.
    """
    return re.sub(r"[^a-z0-9]+", "_", display_name.lower()).strip("_")


def _typed_id(display_name: str, entity_type: str) -> str:
    """Generate a stable entity ID from a display name and its entity type.

    IDs are ``"{entity_type}:{name_slug}"``. The type is part of the identity so
    that ``customer:mercury`` and ``element:mercury`` are distinct entities.
    ``_auto_id`` collapses every non-alphanumeric character to ``_``, so ``:``
    is an unambiguous separator: :func:`split_typed_id` always recovers both
    halves.
    """
    return f"{_auto_id(entity_type)}:{_auto_id(display_name)}"


def split_typed_id(entity_id: str) -> tuple[str, str]:
    """Split ``"{type}:{name_slug}"`` into ``(type, name_slug)``.

    An ID with no ``:`` predates typed IDs and yields ``("", entity_id)``.
    """
    entity_type, sep, name_slug = entity_id.partition(":")
    if not sep:
        return "", entity_id
    return entity_type, name_slug
