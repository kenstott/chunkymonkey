"""Local filesystem transport."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from ._protocol import FetchResult


def _is_local_uri(uri: str) -> bool:
    """Return True for file:// URIs and bare paths (no scheme, or Windows drive letter)."""
    if uri.startswith("file://"):
        return True
    parsed = urlparse(uri)
    # No scheme, or single-letter scheme (Windows drive letter like C:/)
    if not parsed.scheme or len(parsed.scheme) == 1:
        return True
    return False


class LocalTransport:
    """Fetch documents from the local filesystem."""

    def can_handle(self, uri: str) -> bool:
        return _is_local_uri(uri)

    def fetch(self, uri: str, **kwargs) -> FetchResult:
        if uri.startswith("file://"):
            path = Path(uri[len("file://"):])
        else:
            path = Path(uri)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return FetchResult(
            data=path.read_bytes(),
            detected_mime=None,
            source_path=str(path),
        )
