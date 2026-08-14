# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 0c55a823-cad3-4e33-9a2d-598abd1eb3fb
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""chonk NER — vocabulary-based entity matching and chunk association index."""

from ._build import build_ner
from ._index import EntityIndex
from ._merge import merge_matches
from ._normalizer import EntityNormalizer, canonical_key, normalize_entity
from ._pipeline import NerPipeline
from ._schema import SchemaMatcher, normalize_schema_term
from ._schema_vocab import SchemaVocabBuilder
from ._spacy import SpacyMatcher
from ._spacy_labels import ALL_SPACY_LABELS, SpacyLabel
from ._vocabulary import EntityMatch, VocabularyMatcher, normalize_separators, normalize_surface

__all__ = [
    "build_ner",
    "VocabularyMatcher",
    "EntityMatch",
    "EntityIndex",
    "SpacyMatcher",
    "SpacyLabel",
    "ALL_SPACY_LABELS",
    "merge_matches",
    "SchemaMatcher",
    "normalize_schema_term",
    "normalize_separators",
    "normalize_surface",
    "SchemaVocabBuilder",
    "NerPipeline",
    "EntityNormalizer",
    "normalize_entity",
    "canonical_key",
]
