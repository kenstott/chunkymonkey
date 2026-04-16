# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: e08c8002-37c7-4dbf-80e4-378690bc7749
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Extractor protocol definition."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Extractor(Protocol):
    def extract(self, data: bytes, source_path: str | None = None) -> str:
        """Extract plain text from raw bytes. Returns UTF-8 string."""
        ...

    def can_handle(self, doc_type: str) -> bool:
        """Return True if this extractor handles the given doc_type short alias."""
        ...
