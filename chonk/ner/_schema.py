# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 6d8b03f6-c113-495f-9a5e-759b47cc35df
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Schema-aware vocabulary matcher for table/column/API/business term NER.

Pure Python — no spaCy dependency required.
"""

from __future__ import annotations

import re

from ._vocabulary import EntityMatch, _typed_id, normalize_separators, normalize_surface

# ---------------------------------------------------------------------------
# Normalisation helper (exported from chonk.ner)
# ---------------------------------------------------------------------------


def normalize_schema_term(term: str, to_singular: bool = False) -> str:
    """Convert a schema/API term to a normalized space-separated string.

    Handles underscores, camelCase, PascalCase, and kebab-case.

    Examples::

        normalize_schema_term("performance_reviews")  -> "performance reviews"
        normalize_schema_term("performanceReviews")   -> "performance reviews"
        normalize_schema_term("HTMLParser")            -> "html parser"
        normalize_schema_term("order_items", to_singular=True) -> "order item"

    Args:
        term: Raw schema term (table name, column name, endpoint, etc.).
        to_singular: If True, strip a trailing "s" from the result.

    Returns:
        Lowercase, space-separated string.
    """
    # Split camelCase / PascalCase: insert space before uppercase following lowercase
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", term)
    # Split run of capitals before a capital+lowercase pair: HTMLParser → HTML Parser
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    # Replace underscores and hyphens with spaces
    s = s.replace("_", " ").replace("-", " ")
    # Collapse whitespace and lowercase
    s = " ".join(s.split()).lower()
    if to_singular and s.endswith("s"):
        s = s[:-1]
    return s


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------


def _variants(term: str) -> list[str]:
    """Return all surface-form variants for a term, all lowercase."""
    normalized = normalize_schema_term(term)  # "performance reviews"
    singular = normalize_schema_term(term, to_singular=True)  # "performance review"
    result = {normalized, singular}
    # Underscore form — only if the original contains an underscore
    if "_" in term:
        result.add(term.lower())  # "performance_reviews"
    # Joined forms (no spaces)
    result.add(normalized.replace(" ", ""))  # "performancereviews"
    result.add(singular.replace(" ", ""))  # "performancereview"
    # Filter out empty strings and single-char noise
    return [v for v in result if len(v) > 1]


# ---------------------------------------------------------------------------
# SchemaMatcher
# ---------------------------------------------------------------------------


class SchemaMatcher:
    """Vocabulary-based matcher for schema, API, and business terms.

    Generates surface-form variants per term and matches them
    case-insensitively with word-boundary checks.  No spaCy required.

    Usage::

        matcher = SchemaMatcher(
            schema_terms=["performance_reviews", "employee_id"],
            api_terms=["/api/v1/users"],
            business_terms=["PII", "GDPR"],
        )
        matches = matcher.match("The performance review score is stored in employee_id.")

    Args:
        schema_terms: Table and column names → entity_type ``"schema"``.
        api_terms: Endpoint / operation names → entity_type ``"api"``.
        business_terms: Glossary / business terms → entity_type ``"term"``.
        normalize_separators: Collapse runs of whitespace and connector
            punctuation to a single space on both the stored variants and the
            searched text, so separator differences do not cause a miss.
            Reported spans are always offsets into the original text.
    """

    def __init__(
        self,
        schema_terms: list[str] | None = None,
        api_terms: list[str] | None = None,
        business_terms: list[str] | None = None,
        normalize_separators: bool = True,
    ) -> None:
        self._normalize = normalize_separators
        # variant_lowercase -> (entity_id, display_name, entity_type)
        self._lookup: dict[str, tuple[str, str, str]] = {}
        # entity_id -> (canonical_name, display_name, entity_type)
        self._entities: dict[str, tuple[str, str, str]] = {}

        for terms, etype in (
            (schema_terms or [], "schema"),
            (api_terms or [], "api"),
            (business_terms or [], "term"),
        ):
            for term in terms:
                canonical = normalize_schema_term(term, to_singular=True)
                eid = _typed_id(canonical, etype)
                self._entities[eid] = (canonical, term, etype)
                for variant in _variants(term):
                    key = normalize_surface(variant) if self._normalize else variant
                    if not key:
                        continue
                    # First registration wins on collision
                    if key not in self._lookup:
                        self._lookup[key] = (eid, term, etype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, text: str) -> list[EntityMatch]:
        """Find all schema/API/business terms in *text*.

        Returns one ``EntityMatch`` per unique canonical term, aggregating
        frequency and spans across all surface forms.

        Args:
            text: The text to search.

        Returns:
            List of ``EntityMatch`` objects.
        """
        check = text.lower()
        # Scan normalised text so separator variants hit the same term;
        # index_map carries spans back to offsets in the original text.
        if self._normalize:
            check, index_map = normalize_separators(check)
        else:
            index_map = list(range(len(check)))
        found: dict[str, list[tuple[int, int]]] = {}  # entity_id -> spans

        for variant, (eid, _display, _etype) in self._lookup.items():
            start = 0
            while True:
                pos = check.find(variant, start)
                if pos == -1:
                    break
                before_ok = pos == 0 or not check[pos - 1].isalnum()
                after_pos = pos + len(variant)
                after_ok = after_pos >= len(check) or not check[after_pos].isalnum()
                if before_ok and after_ok:
                    if eid not in found:
                        found[eid] = []
                    found[eid].append((index_map[pos], index_map[after_pos - 1] + 1))
                start = pos + 1

        results = []
        for eid, spans in found.items():
            canonical, display_name, etype = self._entities[eid]
            deduped = sorted({s: None for s in spans})  # stable unique spans
            results.append(
                EntityMatch(
                    entity_id=eid,
                    name=canonical,
                    display_name=display_name,
                    entity_type=etype,
                    frequency=len(deduped),
                    positions=[s[0] for s in deduped],
                    spans=deduped,
                )
            )
        return results
