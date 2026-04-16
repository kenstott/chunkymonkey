# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""spaCy-backed entity matcher — drop-in complement to VocabularyMatcher.

Requires the ``spacy`` optional dependency group::

    pip install chunkymonkey[spacy]

A spaCy language model must also be downloaded, e.g.::

    python -m spacy download en_core_web_sm

"""

from __future__ import annotations

from ._vocabulary import EntityMatch, _auto_id
from ._spacy_labels import SpacyLabel, ALL_SPACY_LABELS


class SpacyMatcher:
    """Run spaCy NER and return results in the same ``EntityMatch`` format
    as ``VocabularyMatcher``.

    Args:
        model: spaCy model name (default ``"en_core_web_sm"``).
        entity_types: Entity label whitelist.  Defaults to
            ``ALL_SPACY_LABELS`` (all 18 standard English labels).
            Pass a subset — e.g. ``[SpacyLabel.ORG, SpacyLabel.PERSON]``
            — to restrict output.
        strip_numeric: Drop entities whose text is purely numeric
            (e.g. bare years, counts).  Default ``False`` (all entities
            pass through).  Set to ``True`` to suppress bare numbers.
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        entity_types: list[SpacyLabel] | list[str] | None = None,
        strip_numeric: bool = False,
    ):
        try:
            import spacy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "spaCy is required for SpacyMatcher. "
                "Install with: pip install chunkymonkey[spacy]"
            ) from exc
        import spacy as _spacy
        self._nlp = _spacy.load(model)
        # Default to all standard labels; accept None as alias for same.
        self._types: set[str] = {
            str(t) for t in (entity_types if entity_types is not None else ALL_SPACY_LABELS)
        }
        self._strip_numeric = strip_numeric

    # ------------------------------------------------------------------
    # Public API (mirrors VocabularyMatcher)
    # ------------------------------------------------------------------

    def match(self, text: str) -> list[EntityMatch]:
        """Run spaCy NER on *text* and return entity matches.

        Multiple occurrences of the same surface form are grouped under
        one ``EntityMatch`` (same entity_id), with all spans recorded.
        """
        doc = self._nlp(text)

        # entity_id -> accumulated data
        found: dict[str, dict] = {}

        for ent in doc.ents:
            if ent.label_ not in self._types:
                continue
            surface = ent.text
            if self._strip_numeric and surface.strip().lstrip("-+").replace(".", "").isdigit():
                continue

            eid = _auto_id(surface)
            if eid not in found:
                found[eid] = {
                    "name": surface.lower(),
                    "display_name": surface,
                    "type": ent.label_.lower(),
                    "spans": [],
                }
            found[eid]["spans"].append((ent.start_char, ent.end_char))

        results = []
        for eid, info in found.items():
            spans = sorted(info["spans"])
            results.append(EntityMatch(
                entity_id=eid,
                name=info["name"],
                display_name=info["display_name"],
                entity_type=info["type"],
                frequency=len(spans),
                positions=[s[0] for s in spans],
                spans=spans,
            ))
        return results