# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 9ccf08ef-ae1c-4864-a753-a99ab0300447
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Transport protocol and FetchResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class FetchResult:
    """Result of fetching a document via any transport."""

    data: bytes
    detected_mime: str | None = None
    source_path: str | None = None


@runtime_checkable
class Transport(Protocol):
    def fetch(self, uri: str, **kwargs) -> FetchResult: ...
    def can_handle(self, uri: str) -> bool: ...
