# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""chunkymonkey NER — vocabulary-based entity matching and chunk association index."""
from ._vocabulary import VocabularyMatcher, EntityMatch
from ._index import EntityIndex
from ._spacy import SpacyMatcher
from ._spacy_labels import SpacyLabel, ALL_SPACY_LABELS
from ._merge import merge_matches

__all__ = [
    "VocabularyMatcher",
    "EntityMatch",
    "EntityIndex",
    "SpacyMatcher",
    "SpacyLabel",
    "ALL_SPACY_LABELS",
    "merge_matches",
]