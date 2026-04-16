# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 4c810be5-e7cc-4678-829c-a06667ea9b1f
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""HTTP/HTTPS transport using requests with a persistent session."""

from __future__ import annotations

from ._protocol import FetchResult

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_USER_AGENT = "chunkymonkey/0.1"

# Module-level session for cookie persistence across requests
_http_session: "_requests.Session | None" = None


def _get_http_session() -> "_requests.Session":
    """Get or create a module-level requests session with cookie persistence."""
    if not _REQUESTS_AVAILABLE:
        raise ImportError("pip install chunkymonkey[http]")

    global _http_session
    if _http_session is None:
        _http_session = _requests.Session()
        _http_session.headers["User-Agent"] = _USER_AGENT
    return _http_session


class HttpTransport:
    """Fetch documents via HTTP or HTTPS."""

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("http://") or uri.startswith("https://")

    def fetch(self, uri: str, **kwargs) -> FetchResult:
        session = _get_http_session()
        headers = kwargs.get("headers", {})
        timeout = kwargs.get("timeout", 30)

        response = session.get(uri, headers=headers, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get("content-type")

        return FetchResult(
            data=response.content,
            detected_mime=content_type,
            source_path=uri,
        )
