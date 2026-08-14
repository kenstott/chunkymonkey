# Copyright (c) 2025 Kenneth Stott. MIT License.
"""Separator normalization in vocabulary matching."""

from __future__ import annotations

import pytest

from chonk.ner._merge import merge_matches
from chonk.ner._schema_vocab import SchemaVocabBuilder
from chonk.ner._vocabulary import (
    VocabularyMatcher,
    normalize_separators,
    normalize_surface,
)

ENTITY = {
    "id": "customer:acme_corp",
    "name": "acme corp",
    "display_name": "Acme Corp",
    "type": "customer",
    "aliases": [],
}


class TestNormalizePrimitives:
    def test_collapses_separator_runs_to_one_space(self):
        assert normalize_surface("Acme  Corp") == "Acme Corp"
        assert normalize_surface("Acme-Corp") == "Acme Corp"
        assert normalize_surface("Acme\nCorp") == "Acme Corp"
        assert normalize_surface("Acme_Corp") == "Acme Corp"
        assert normalize_surface("Acme / Corp") == "Acme Corp"

    def test_sentence_punctuation_is_preserved(self):
        # Collapsing "." would make "Acme Corp" match across "…Acme. Corp…".
        assert normalize_surface("Acme. Corp") == "Acme. Corp"
        assert normalize_surface("Acme, Corp") == "Acme, Corp"

    def test_index_map_points_at_original_offsets(self):
        text = "a--b"
        norm, index_map = normalize_separators(text)
        assert norm == "a b"
        assert [text[i] for i in index_map] == ["a", "-", "b"]

    def test_index_map_length_matches_normalized_text(self):
        for text in ["", "abc", "a  b", "  ", "a-b_c/d"]:
            norm, index_map = normalize_separators(text)
            assert len(norm) == len(index_map)


class TestVocabularyMatcherNormalization:
    @pytest.fixture()
    def matcher(self):
        return VocabularyMatcher([ENTITY])

    @pytest.mark.parametrize(
        "text",
        [
            "Acme Corp filed.",
            "Acme-Corp filed.",
            "Acme  Corp filed.",
            "Acme\nCorp filed.",
            "Acme_Corp filed.",
            "ACME   corp filed.",
        ],
    )
    def test_separator_variants_all_match(self, matcher, text):
        results = matcher.match(text)
        assert [r.entity_id for r in results] == ["customer:acme_corp"]

    @pytest.mark.parametrize(
        "text",
        [
            "Acme-Corp filed.",
            "Acme  Corp filed.",
            "Acme\nCorp filed.",
        ],
    )
    def test_spans_index_the_original_text(self, matcher, text):
        (result,) = matcher.match(text)
        start, end = result.spans[0]
        assert text[start:end].lower().replace("-", " ").replace("\n", " ").split() == [
            "acme",
            "corp",
        ]

    def test_word_boundaries_still_enforced(self, matcher):
        assert matcher.match("Acme Corporation filed.") == []
        assert matcher.match("XAcme Corp filed.") == []

    def test_sentence_break_does_not_match(self, matcher):
        assert matcher.match("Bought Acme. Corp filed later.") == []

    def test_normalization_can_be_disabled(self):
        matcher = VocabularyMatcher([ENTITY], normalize_separators=False)
        assert matcher.match("Acme Corp filed.") != []
        assert matcher.match("Acme-Corp filed.") == []


class TestNoTypeSplitAfterNormalization:
    """The vocab/spaCy entity split these variants used to cause is gone."""

    @pytest.mark.parametrize(
        "text",
        [
            "Acme Corp filed. Later, Acme-Corp filed again.",
            "Acme Corp filed. Also Acme  Corp filed.",
        ],
    )
    def test_variant_occurrence_no_longer_leaks_to_spacy(self, text):
        pytest.importorskip("spacy")
        pytest.importorskip("en_core_web_sm")
        from chonk.ner._spacy import SpacyMatcher

        builder = SchemaVocabBuilder()
        builder.add_entities(["Acme Corp"], entity_type="customer")
        vocab_hits = builder.build_data_matcher().match(text)
        spacy_hits = SpacyMatcher(model="en_core_web_sm", strip_numeric=True).match(text)
        combined = merge_matches(vocab_hits, spacy_hits, source_text=text)

        assert {m.entity_id for m in combined} == {"customer:acme_corp"}
        # Both occurrences are attributed to the declared entity.
        assert next(m for m in combined).frequency == 2
