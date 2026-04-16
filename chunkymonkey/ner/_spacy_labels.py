# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""spaCy named-entity label constants for standard English models.

Labels match those produced by ``en_core_web_sm``, ``en_core_web_md``,
``en_core_web_lg``, and ``en_core_web_trf``.  Other language models may
use different label sets; pass ``entity_types`` explicitly in that case.

Usage::

    from chunkymonkey.ner import SpacyLabel, SpacyMatcher

    # All entity types (default)
    matcher = SpacyMatcher()

    # Only organisations and people
    matcher = SpacyMatcher(entity_types=[SpacyLabel.ORG, SpacyLabel.PERSON])
"""

from __future__ import annotations

from enum import Enum


class SpacyLabel(str, Enum):
    """Named-entity label strings for standard spaCy English models.

    Inherits from ``str`` so values can be passed directly wherever a
    plain string label is expected (e.g. ``ent.label_``).
    """

    CARDINAL   = "CARDINAL"    # Numerals not covered by another type
    DATE       = "DATE"        # Absolute or relative dates / periods
    EVENT      = "EVENT"       # Hurricanes, battles, sports events, etc.
    FAC        = "FAC"         # Buildings, airports, highways, bridges
    GPE        = "GPE"         # Geopolitical entities: countries, cities, states
    LANGUAGE   = "LANGUAGE"    # Any named language
    LAW        = "LAW"         # Named documents enacted as laws
    LOC        = "LOC"         # Non-GPE locations: mountain ranges, bodies of water
    MONEY      = "MONEY"       # Monetary values, including unit
    NORP       = "NORP"        # Nationalities, religious or political groups
    ORDINAL    = "ORDINAL"     # "first", "second", etc.
    ORG        = "ORG"         # Companies, agencies, institutions
    PERCENT    = "PERCENT"     # Percentage values
    PERSON     = "PERSON"      # People, including fictional
    PRODUCT    = "PRODUCT"     # Objects, vehicles, foods (not services)
    QUANTITY   = "QUANTITY"    # Measurements of weight, distance, etc.
    TIME       = "TIME"        # Times smaller than a day
    WORK_OF_ART = "WORK_OF_ART"  # Titles of books, songs, etc.


# Convenience constant — all labels in a single list, passed as the
# default ``entity_types`` argument so the behaviour is explicit rather
# than relying on a None sentinel.
ALL_SPACY_LABELS: list[SpacyLabel] = list(SpacyLabel)