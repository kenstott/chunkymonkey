# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 7f3a9b2e-4c81-4d5f-a8e2-1b6c0d9f3a47
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GitHubCrawler — incremental working-tree traversal via GitHub REST API.

Satisfies the ``Crawler`` protocol. Fetches file content via
``raw.githubusercontent.com``, so ``HttpTransport`` handles all blob
fetching — no new transport required.

Supports incremental indexing: pass ``since_sha`` to receive only files
added or modified since that commit. Read ``crawler.current_sha`` after
``crawl()`` to persist the watermark.

Usage::

    from chonk.transports import GitHubCrawler, HttpTransport
    from chonk.loader import DocumentLoader

    crawler = GitHubCrawler(token="ghp_...")  # or set GITHUB_TOKEN env var
    loader = DocumentLoader()

    # Full index
    chunks = loader.load_crawl("https://github.com/org/repo", crawler=crawler)
    sha = crawler.current_sha  # persist this

    # Incremental update
    chunks = loader.load_crawl(
        "https://github.com/org/repo",
        crawler=crawler,
        since_sha=sha,
    )
    sha = crawler.current_sha
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

_log = logging.getLogger(__name__)

_API_BASE = "https://api.github.com"
_RAW_BASE = "https://raw.githubusercontent.com"

_DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".txt",
        ".rst",
        ".html",
        ".htm",
        ".pdf",
        ".docx",
        ".xlsx",
        ".pptx",
        ".csv",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".toml",
        ".py",
        ".pyw",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".java",
        ".sql",
    }
)


def _parse_repo_url(uri: str) -> tuple[str, str]:
    """Return (owner, repo) from a github.com URL."""
    parsed = urlparse(uri)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"GitHubCrawler: cannot parse owner/repo from {uri!r}")
    return parts[0], parts[1].removesuffix(".git")


class GitHubCrawler:
    """Crawl a GitHub repository working tree via the GitHub REST API.

    Satisfies the ``Crawler`` protocol — pass to ``DocumentLoader.load_crawl()``.

    Args:
        token:      GitHub personal access token. Falls back to ``GITHUB_TOKEN``
                    env var. Public repos work without a token but are rate-limited
                    to 60 requests/hour.
        extensions:   File extensions to include. Defaults to common document and
                      source-code formats.
        branch:       Branch or tag to crawl. Defaults to the repo's default branch.
        max_files:    Maximum number of files to return per repo (default 2000).
        repo_include: Regex pattern — only repos whose URL matches are crawled.
                      Applied by ``list_repos()`` and ``crawl_all()``.
        repo_exclude: Regex pattern — repos whose URL matches are skipped.
                      Applied after ``repo_include``.

    After calling ``crawl()``, read ``current_sha`` to get the HEAD commit SHA
    for use as ``since_sha`` on the next call.
    """

    def __init__(
        self,
        token: str | None = None,
        extensions: list[str] | None = None,
        branch: str | None = None,
        max_files: int = 2000,
        repo_include: str | None = None,
        repo_exclude: str | None = None,
    ) -> None:
        self._token = token or os.environ.get("GITHUB_TOKEN")
        self.extensions: frozenset[str] = (
            frozenset(e if e.startswith(".") else f".{e}" for e in extensions)
            if extensions is not None
            else _DEFAULT_EXTENSIONS
        )
        self.branch = branch
        self.max_files = max_files
        self.current_sha: str | None = None
        self._repo_include = re.compile(repo_include) if repo_include else None
        self._repo_exclude = re.compile(repo_exclude) if repo_exclude else None

    # ── Crawler Protocol ─────────────────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("https://github.com/") or uri.startswith("http://github.com/")

    def list_repos(self) -> list[str]:
        """Return all repo URLs accessible to the configured token.

        Paginates ``/user/repos`` with ``affiliation=owner,collaborator,
        organization_member`` — covers personal repos, org repos, and repos
        shared with the authenticated user.

        ``repo_include`` and ``repo_exclude`` regex filters (set at construction)
        are applied to each ``https://github.com/{owner}/{repo}`` URL.

        Returns:
            Sorted list of ``https://github.com/{owner}/{repo}`` URLs.

        Raises:
            ImportError: if ``requests`` is not installed.
            requests.HTTPError: on API failure (e.g. bad token, no scope).
        """
        try:
            import requests as _requests
        except ImportError as exc:
            raise ImportError(
                "pip install chonk-rag[http]  # requests required for GitHubCrawler"
            ) from exc  # noqa: E501

        session = self._make_session(_requests)
        urls: list[str] = []
        page = 1
        while True:
            resp = session.get(
                f"{_API_BASE}/user/repos",
                params={
                    "affiliation": "owner,collaborator,organization_member",
                    "per_page": 100,
                    "page": page,
                },
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for repo in batch:
                url = repo["html_url"]
                if self._repo_include and not self._repo_include.search(url):
                    continue
                if self._repo_exclude and self._repo_exclude.search(url):
                    continue
                urls.append(url)
            page += 1
        _log.info("GitHubCrawler.list_repos: %d repo(s) found", len(urls))
        return sorted(urls)

    def crawl_all(
        self,
        since_shas: dict[str, str] | None = None,
    ) -> tuple[list[str], dict[str, str]]:
        """Crawl every repo accessible to the configured token.

        Args:
            since_shas: Map of ``{repo_url: sha}`` watermarks from a previous
                        run. Files unchanged since their watermark SHA are skipped.

        Returns:
            ``(urls, current_shas)`` where ``urls`` is the combined list of raw
            content URLs across all repos, and ``current_shas`` is the updated
            watermark map to persist for the next run.
        """
        since_shas = since_shas or {}
        all_urls: list[str] = []
        current_shas: dict[str, str] = {}
        for repo_url in self.list_repos():
            try:
                urls = self.crawl(repo_url, since_sha=since_shas.get(repo_url))
                all_urls.extend(urls)
                if self.current_sha:
                    current_shas[repo_url] = self.current_sha
            except Exception as exc:
                _log.warning("GitHubCrawler.crawl_all: skipping %s: %s", repo_url, exc)
        return all_urls, current_shas

    def crawl(self, uri: str, since_sha: object = None, **_kw: object) -> list[str]:  # noqa: ARG002
        """Return raw.githubusercontent.com URLs for files in the repo.

        Args:
            uri:       ``https://github.com/{owner}/{repo}`` URL.
            since_sha: If given, return only files added or modified since this
                       commit SHA. Read ``crawler.current_sha`` after this call
                       to get the new watermark.
            **kwargs:  Ignored (protocol compatibility).

        Returns:
            List of raw content URLs, each fetchable by ``HttpTransport``.
        """
        _since: str | None = since_sha if isinstance(since_sha, str) else None
        try:
            import requests as _requests
        except ImportError as exc:
            raise ImportError(
                "pip install chonk-rag[http]  # requests required for GitHubCrawler"
            ) from exc  # noqa: E501

        owner, repo = _parse_repo_url(uri)
        session = self._make_session(_requests)

        branch = self.branch or self._default_branch(session, owner, repo)
        head_sha = self._resolve_sha(session, owner, repo, branch)
        self.current_sha = head_sha

        if _since and _since != head_sha:
            paths = self._changed_paths(session, owner, repo, _since, head_sha)
        else:
            paths = self._tree_paths(session, owner, repo, head_sha)

        urls = [f"{_RAW_BASE}/{owner}/{repo}/{head_sha}/{p}" for p in paths if self._accept(p)]
        _log.info("GitHubCrawler: %d file(s) from %s@%s", len(urls), uri, head_sha[:7])
        return urls[: self.max_files]

    # ── Internal ─────────────────────────────────────────────────────────────

    def _make_session(self, requests_mod: Any) -> Any:  # noqa: ANN401
        session = requests_mod.Session()
        session.headers["Accept"] = "application/vnd.github+json"
        session.headers["X-GitHub-Api-Version"] = "2022-11-28"
        if self._token:
            session.headers["Authorization"] = f"Bearer {self._token}"
        return session

    def _default_branch(self, session: Any, owner: str, repo: str) -> str:  # noqa: ANN401
        resp = session.get(f"{_API_BASE}/repos/{owner}/{repo}", timeout=15)
        resp.raise_for_status()
        return resp.json()["default_branch"]

    def _resolve_sha(self, session: Any, owner: str, repo: str, branch: str) -> str:  # noqa: ANN401
        resp = session.get(
            f"{_API_BASE}/repos/{owner}/{repo}/commits/{branch}",
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["sha"]

    def _tree_paths(self, session: Any, owner: str, repo: str, sha: str) -> list[str]:  # noqa: ANN401
        """Return all blob paths in the repo tree."""
        resp = session.get(
            f"{_API_BASE}/repos/{owner}/{repo}/git/trees/{sha}",
            params={"recursive": "1"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("truncated"):
            _log.warning(
                "GitHubCrawler: tree truncated at GitHub's limit for %s/%s — "
                "use extensions filter to reduce scope",
                owner,
                repo,
            )
        return [item["path"] for item in data.get("tree", []) if item["type"] == "blob"]

    def _changed_paths(
        self,
        session: Any,  # noqa: ANN401
        owner: str,
        repo: str,
        base_sha: str,
        head_sha: str,
    ) -> list[str]:
        """Return paths added or modified between base_sha and head_sha."""
        resp = session.get(
            f"{_API_BASE}/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            f["filename"]
            for f in data.get("files", [])
            if f.get("status") in ("added", "modified", "renamed", "copied")
        ]

    def _accept(self, path: str) -> bool:
        dot = path.rfind(".")
        if dot == -1:
            return False
        return path[dot:].lower() in self.extensions
