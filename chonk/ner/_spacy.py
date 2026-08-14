# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 745e92bf-be5f-455a-aaba-ec6b975ea6e3
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""spaCy-backed entity matcher — drop-in complement to VocabularyMatcher.

Requires the ``spacy`` optional dependency group::

    pip install chonk-rag[spacy]

A spaCy language model must also be downloaded, e.g.::

    python -m spacy download en_core_web_sm

"""

from __future__ import annotations

from typing import Any

from ._spacy_labels import ALL_SPACY_LABELS, SpacyLabel
from ._vocabulary import EntityMatch, _typed_id


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
    ) -> None:
        try:
            import spacy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "spaCy is required for SpacyMatcher. Install with: pip install chonk-rag[spacy]"
            ) from exc
        import spacy as _spacy

        self._nlp = _spacy.load(model)
        # Default to all standard labels; accept None as alias for same.
        # isinstance narrows SpacyLabel from str in union; .value gives e.g. "ORG" not "SpacyLabel.ORG"  # noqa: E501
        self._types: set[str] = {
            t.value if isinstance(t, SpacyLabel) else t
            for t in (entity_types if entity_types is not None else ALL_SPACY_LABELS)
        }
        self._strip_numeric = strip_numeric

    # ------------------------------------------------------------------
    # Public API (mirrors VocabularyMatcher)
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_surface(text: str) -> str:
        """Strip leading/trailing non-alphanumeric characters."""
        import re

        return re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", text).strip()

    def match(self, text: str) -> list[EntityMatch]:
        """Run spaCy NER on *text* and return entity matches.

        Multiple occurrences of the same surface form are grouped under
        one ``EntityMatch`` (same entity_id), with all spans recorded.
        """
        doc = self._nlp(text)

        # entity_id -> accumulated data
        found: dict[str, dict[str, Any]] = {}

        for ent in doc.ents:
            if ent.label_ not in self._types:
                continue

            # Skip stop-word-only spans
            if all(t.is_stop for t in ent):
                continue

            surface = self._clean_surface(ent.text)
            if not surface:
                continue

            if self._strip_numeric and surface.lstrip("-+").replace(".", "").isdigit():
                continue

            # Canonical form: join token lemmas, lowercase, strip symbols
            # (handles plural/inflected forms and multi-word spans correctly)
            canonical = self._clean_surface(
                " ".join(t.lemma_.lower() for t in ent if not t.is_punct and not t.is_space)
            )
            if not canonical:
                continue

            etype = ent.label_.lower()
            eid = _typed_id(canonical, etype)
            if eid not in found:
                found[eid] = {
                    "name": canonical,
                    "display_name": surface,
                    "type": etype,
                    "spans": [],
                }
            found[eid]["spans"].append((ent.start_char, ent.end_char))

        results = []
        for eid, info in found.items():
            spans = sorted(info["spans"])
            results.append(
                EntityMatch(
                    entity_id=eid,
                    name=info["name"],
                    display_name=info["display_name"],
                    entity_type=info["type"],
                    frequency=len(spans),
                    positions=[s[0] for s in spans],
                    spans=spans,
                )
            )
        return results
