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
