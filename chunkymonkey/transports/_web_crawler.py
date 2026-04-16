# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: f6794903-923a-4d9e-a02b-de16a198eb8b
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""WebCrawler — BFS HTTP/HTTPS site crawler using stdlib only.

No third-party dependencies.  Fetches HTML via ``urllib.request``, extracts
links with regex, and performs breadth-first traversal with a configurable
page/depth budget.

Extend for authenticated services (SharePoint, Confluence, …) by implementing
the ``Crawler`` protocol — you do **not** need to subclass ``WebCrawler``.
"""
from __future__ import annotations

import logging
import re
import ssl
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

# Use certifi root certificates if available (required on macOS with stock Python).
# Falls back to the default SSL context, which works on most Linux/Windows installs.
try:
    import certifi as _certifi
    _SSL_CONTEXT: ssl.SSLContext | None = ssl.create_default_context(
        cafile=_certifi.where()
    )
except ImportError:
    _SSL_CONTEXT = None  # use urllib default

logger = logging.getLogger(__name__)

# ── URL filtering ────────────────────────────────────────────────────────────

_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp",
        ".mp4", ".mp3", ".wav", ".ogg", ".avi", ".mov", ".mkv",
        ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
        ".exe", ".dmg", ".pkg", ".deb", ".rpm",
        ".woff", ".woff2", ".ttf", ".eot",
        ".css",  # rarely useful for RAG
    }
)

_SKIP_PATH_PREFIXES: tuple[str, ...] = (
    "/cdn-cgi/", "/wp-json/", "/api/", "/static/", "/assets/",
    "/media/", "/uploads/", "/fonts/",
)

# Wiki sites generate hundreds of non-content pages under these path patterns.
# When wiki_mode=True (the default for detected wikis), these are filtered out.
_WIKI_SKIP_PREFIXES: tuple[str, ...] = (
    "/wiki/Special:", "/wiki/Help:", "/wiki/Talk:", "/wiki/File:",
    "/wiki/Category:", "/wiki/Template:", "/wiki/Wikipedia:",
    "/wiki/User:", "/wiki/User_talk:", "/wiki/Project:",
    "/wiki/Portal:", "/wiki/Module:",
    "/w/index.php",   # MediaWiki action URLs (?action=edit etc.)
)

_WIKI_SKIP_PARAMS: tuple[str, ...] = (
    "action=edit", "action=history", "action=raw",
    "action=info", "printable=yes", "diff=",
)

_HREF_RE = re.compile(r'href=["\']([^"\'#?][^"\']*)["\']', re.IGNORECASE)
_MD_LINK_RE = re.compile(r'\[(?:[^\]]+)\]\((https?://[^)]+)\)')

_USER_AGENT = "chunkymonkey/0.1 (+https://github.com/chunkymonkey/chunkymonkey)"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Strip fragment and trailing slash for deduplication."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _extract_links(html: str, base_url: str) -> list[str]:
    """Extract and resolve all href links from an HTML page."""
    raw = _HREF_RE.findall(html)
    seen: set[str] = set()
    result: list[str] = []
    for href in raw:
        href = href.strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        resolved = urljoin(base_url, href)
        parsed = urlparse(resolved)
        if parsed.scheme not in ("http", "https"):
            continue
        ext = "." + parsed.path.rsplit(".", 1)[-1].lower() if "." in parsed.path.rsplit("/", 1)[-1] else ""
        if ext in _BINARY_EXTENSIONS:
            continue
        if any(parsed.path.startswith(p) for p in _SKIP_PATH_PREFIXES):
            continue
        norm = _normalize_url(resolved)
        if norm not in seen:
            seen.add(norm)
            result.append(resolved)
    return result


def _http_get(url: str, timeout: int) -> tuple[bytes, str | None]:
    """Fetch *url* with stdlib urllib.  Returns (body, content_type)."""
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, context=_SSL_CONTEXT, timeout=timeout) as resp:
        content_type: str | None = resp.headers.get("Content-Type")
        return resp.read(), content_type


# ── WebCrawler ───────────────────────────────────────────────────────────────

class WebCrawler:
    """BFS HTTP/HTTPS site crawler.

    Satisfies the ``Crawler`` protocol — can be passed directly to
    ``DocumentLoader.load_crawl()``.

    Wiki sites generate hundreds of non-content pages (Special:, Talk:,
    action=edit, etc.).  Set ``wiki_mode=True`` to automatically filter these.
    Pass ``exclude_patterns`` for custom exclusions (Confluence, Notion, etc.).

    Args:
        max_pages:    Maximum number of pages to return (default 50).
        max_depth:    Maximum link-follow depth from the root (default 3).
        same_domain:  If True (default), only follow links on the same host.
        max_workers:  Parallel fetch threads per BFS wave (default 8).
        timeout:      Per-request timeout in seconds (default 20).
        exclude_patterns: List of regex strings; matching URLs are skipped.
        include_pattern:  If set, only URLs matching this regex are followed.
        wiki_mode:    Filter MediaWiki-style non-content paths (default False).
                      Automatically activated when the root URL contains
                      ``/wiki/`` or query params suggest a MediaWiki instance.

    Usage::

        from chunkymonkey.transports import WebCrawler

        crawler = WebCrawler(max_pages=100, max_depth=4)
        urls = crawler.crawl("https://docs.example.com/")

        # Wikipedia / MediaWiki:
        crawler = WebCrawler(max_pages=200, wiki_mode=True, include_pattern=r"/wiki/[A-Z]")
        urls = crawler.crawl("https://en.wikipedia.org/wiki/Python_(programming_language)")

        # Via DocumentLoader:
        chunks = loader.load_site("https://docs.example.com/")
    """

    def __init__(
        self,
        max_pages: int = 50,
        max_depth: int = 3,
        same_domain: bool = True,
        max_workers: int = 8,
        timeout: int = 20,
        exclude_patterns: list[str] | None = None,
        include_pattern: str | None = None,
        wiki_mode: bool = False,
    ):
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.same_domain = same_domain
        self.max_workers = max_workers
        self.timeout = timeout
        self.wiki_mode = wiki_mode
        self._exclude_res = [re.compile(p) for p in (exclude_patterns or [])]
        self._include_re = re.compile(include_pattern) if include_pattern else None

    # ── Crawler Protocol ────────────────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("http://") or uri.startswith("https://")

    def crawl(self, uri: str, **kwargs) -> list[str]:
        """BFS crawl starting at *uri*.  Returns list of discovered page URLs.

        All kwargs are ignored (use constructor params).  The signature
        accepts **kwargs for protocol compatibility.
        """
        root_domain = urlparse(uri).netloc
        visited: set[str] = {_normalize_url(uri)}
        results: list[str] = []

        # Fetch root
        try:
            body, content_type = _http_get(uri, self.timeout)
        except (URLError, OSError) as exc:
            logger.warning("WebCrawler: failed to fetch root %s: %s", uri, exc)
            return []

        results.append(uri)
        logger.info("WebCrawler: fetched root %s", uri)

        # Auto-detect wiki mode from root URL
        if not self.wiki_mode and ("/wiki/" in uri or "mediawiki" in uri.lower()):
            logger.info("WebCrawler: wiki_mode auto-enabled for %s", uri)
            self.wiki_mode = True

        # Seed queue from root links
        queue: deque[tuple[str, int]] = deque()
        try:
            html = body.decode("utf-8", errors="replace")
        except Exception:
            return results

        for link in _extract_links(html, uri):
            norm = _normalize_url(link)
            if norm not in visited:
                visited.add(norm)
                queue.append((link, 1))

        # BFS waves
        while queue and len(results) < self.max_pages:
            budget = self.max_pages - len(results)
            wave: list[tuple[str, int]] = []
            while queue and len(wave) < budget:
                url, depth = queue.popleft()
                if depth > self.max_depth:
                    continue
                if not self._should_visit(url, root_domain):
                    continue
                wave.append((url, depth))

            if not wave:
                break

            workers = min(len(wave), self.max_workers)
            logger.info("WebCrawler: fetching wave of %d URL(s)", len(wave))

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_http_get, url, self.timeout): (url, depth)
                    for url, depth in wave
                }
                for future in as_completed(futures):
                    url, depth = futures[future]
                    if len(results) >= self.max_pages:
                        break
                    try:
                        body, content_type = future.result()
                    except Exception as exc:
                        logger.warning("WebCrawler: failed %s: %s", url, exc)
                        continue

                    # Skip non-HTML responses (PDFs etc. go to their own extractor)
                    ct = (content_type or "").lower()
                    if ct and "html" not in ct and "text/" not in ct and not ct.startswith("application/xhtml"):
                        results.append(url)  # still include as document
                        continue

                    results.append(url)
                    logger.info("WebCrawler: fetched %s (%d/%d)", url, len(results), self.max_pages)

                    if depth < self.max_depth:
                        try:
                            page_html = body.decode("utf-8", errors="replace")
                        except Exception:
                            continue
                        for link in _extract_links(page_html, url):
                            norm = _normalize_url(link)
                            if norm not in visited:
                                visited.add(norm)
                                queue.append((link, depth + 1))

        logger.info("WebCrawler: done. %d pages from %s", len(results), uri)
        return results

    # ── Internal ────────────────────────────────────────────────────────────

    def _should_visit(self, url: str, root_domain: str) -> bool:
        if self.same_domain and urlparse(url).netloc != root_domain:
            return False
        if self._include_re and not self._include_re.search(url):
            return False
        if any(ep.search(url) for ep in self._exclude_res):
            logger.debug("WebCrawler: excluded by pattern: %s", url)
            return False
        if self.wiki_mode:
            parsed = urlparse(url)
            if any(parsed.path.startswith(p) for p in _WIKI_SKIP_PREFIXES):
                return False
            qs = parsed.query
            if any(param in qs for param in _WIKI_SKIP_PARAMS):
                return False
        return True
