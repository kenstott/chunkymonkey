# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 4f94be99-69f0-4a47-9d83-7f6779659b02
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Crawler protocol — multi-document traversal contract.

A ``Crawler`` discovers a *list of URIs* from a root location (a website,
directory, or cloud storage prefix) and returns them so that a
``DocumentLoader`` can fetch and process each one individually.

Third-party crawlers (SharePoint, Google Drive, Confluence, Notion, …) can
satisfy this protocol by implementing two methods::

    class SharePointCrawler:
        def can_handle(self, uri: str) -> bool:
            return uri.startswith("sharepoint://") or "sharepoint.com" in uri

        def crawl(self, uri: str, **kwargs) -> list[str]:
            # authenticate, walk the document library, return URIs
            ...

Then pass it to ``DocumentLoader.load_crawl(uri, crawler=SharePointCrawler())``.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Crawler(Protocol):
    """Protocol for multi-document source traversal.

    Implementations discover and return a flat list of fetchable URIs from a
    root location.  Each returned URI is then processed independently by the
    caller (typically via a ``Transport``).
    """

    def can_handle(self, uri: str) -> bool:
        """Return True if this crawler can traverse the given root URI."""
        ...

    def crawl(self, uri: str, **kwargs) -> list[str]:
        """Traverse *uri* and return all discovered document URIs.

        Args:
            uri:     Root location to crawl (URL, path, storage prefix, …).
            **kwargs: Crawler-specific options (max_pages, max_depth, …).

        Returns:
            Ordered list of document URIs (root first, then discovered pages).
        """
        ...
