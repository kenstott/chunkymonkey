# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Utilities for merging entity matches from multiple matchers.

Typical use::

    from chunkymonkey.ner import VocabularyMatcher, SpacyMatcher, merge_matches

    vocab   = VocabularyMatcher.from_file("entities.json")
    generic = SpacyMatcher(entity_types=["ORG", "PERSON", "GPE"])

    combined = merge_matches(
        vocab.match(text),
        generic.match(text),
        source_text=text,
    )
    entity_index.index_chunk(chunk_id, text, combined)

Merge semantics
---------------
1. **Boundary strip** (default on): trim leading/trailing punctuation and
   whitespace from generic-matcher spans before overlap testing.
2. **Span-level suppression**: vocab wins on any character overlap.
   A generic entity's occurrence is dropped only if *that occurrence*
   overlaps a vocab span; surviving occurrences keep the entity alive
   with an adjusted frequency.
3. **Entity-level dedup**: if both matchers produce the same ``entity_id``
   (rare but possible), their spans are unioned and frequency summed.
"""

from __future__ import annotations

from ._vocabulary import EntityMatch

# Characters to strip from the leading/trailing edges of generic spans.
# Includes paired delimiters, operators, and common punctuation that
# NER models often absorb into entity boundaries.
_BOUNDARY_CHARS = frozenset(
    '()[]{}<>*&^%$#@!+=~`\'"\\|/?:;,.—–-'
)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _strip_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Return adjusted (start, end) after stripping boundary chars.

    Returns ``None`` if the span becomes empty after stripping.
    """
    span_text = text[start:end]
    # Strip from left
    new_start = start
    for ch in span_text:
        if ch in _BOUNDARY_CHARS or ch.isspace():
            new_start += 1
        else:
            break
    # Strip from right
    new_end = end
    for ch in reversed(text[new_start:end]):
        if ch in _BOUNDARY_CHARS or ch.isspace():
            new_end -= 1
        else:
            break
    if new_end <= new_start:
        return None
    return new_start, new_end


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if spans share at least one character."""
    return a[0] < b[1] and b[0] < a[1]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def merge_matches(
    vocab_matches: list[EntityMatch],
    generic_matches: list[EntityMatch],
    source_text: str = "",
    boundary_strip: bool = True,
) -> list[EntityMatch]:
    """Merge vocabulary and generic (e.g. spaCy) entity matches.

    Args:
        vocab_matches: Output of ``VocabularyMatcher.match()``.
        generic_matches: Output of ``SpacyMatcher.match()`` or any matcher
            with the same interface.
        source_text: The original text string (required when
            ``boundary_strip=True``; used to re-read stripped characters).
        boundary_strip: Strip leading/trailing punctuation and whitespace
            from generic spans before overlap testing (default ``True``).

    Returns:
        Deduplicated, merged list of ``EntityMatch`` objects.  Vocab entries
        always take precedence; generic entries survive only where their
        spans do not overlap any vocab span.
    """
    # Build a flat set of all vocab spans for O(1) overlap testing.
    vocab_span_set: list[tuple[int, int]] = []
    for m in vocab_matches:
        vocab_span_set.extend(m.spans)

    # Filter generic matches: suppress occurrences that overlap vocab spans.
    surviving_generic: list[EntityMatch] = []
    for gm in generic_matches:
        kept_spans: list[tuple[int, int]] = []
        for span in gm.spans:
            if boundary_strip and source_text:
                stripped = _strip_span(source_text, span[0], span[1])
                if stripped is None:
                    continue  # span is pure punctuation
                span = stripped
            # Keep only if no overlap with any vocab span
            if not any(_overlaps(span, vs) for vs in vocab_span_set):
                kept_spans.append(span)
        if kept_spans:
            surviving_generic.append(EntityMatch(
                entity_id=gm.entity_id,
                name=gm.name,
                display_name=gm.display_name,
                entity_type=gm.entity_type,
                frequency=len(kept_spans),
                positions=[s[0] for s in kept_spans],
                spans=kept_spans,
            ))

    # Combine and dedup by entity_id (handles rare case where both matchers
    # independently produce the same canonical ID).
    merged: dict[str, EntityMatch] = {}
    for m in (*vocab_matches, *surviving_generic):
        if m.entity_id not in merged:
            merged[m.entity_id] = EntityMatch(
                entity_id=m.entity_id,
                name=m.name,
                display_name=m.display_name,
                entity_type=m.entity_type,
                frequency=0,
                positions=[],
                spans=[],
            )
        existing = merged[m.entity_id]
        # Union spans (avoid duplicates by position)
        existing_starts = {s[0] for s in existing.spans}
        new_spans = [s for s in m.spans if s[0] not in existing_starts]
        all_spans = sorted(existing.spans + new_spans)
        merged[m.entity_id] = EntityMatch(
            entity_id=existing.entity_id,
            name=existing.name,
            display_name=existing.display_name,
            entity_type=existing.entity_type,
            frequency=len(all_spans),
            positions=[s[0] for s in all_spans],
            spans=all_spans,
        )

    return list(merged.values())