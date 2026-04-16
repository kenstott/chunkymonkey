# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: e0e29635-1d0f-4b13-9765-7bfd1073e44a
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DirectoryCrawler — recursive file discovery for local paths and S3 prefixes.

Satisfies the ``Crawler`` protocol.  Extend for other blob stores (Azure Blob,
GCS, …) by implementing ``can_handle`` and ``crawl`` on a new class.

Supported URI schemes:
  - Local filesystem: ``/abs/path``, ``./rel/path``, ``file:///abs/path``
  - Amazon S3:        ``s3://bucket/prefix``
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Default file extensions treated as loadable documents
_DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md", ".txt", ".rst", ".html", ".htm",
        ".pdf", ".docx", ".xlsx", ".pptx",
        ".csv", ".json", ".xml", ".yaml", ".yml",
    }
)


class DirectoryCrawler:
    """Recursively discover documents in a local directory or S3 prefix.

    Satisfies the ``Crawler`` protocol — pass to ``DocumentLoader.load_crawl()``.

    Args:
        extensions:   Iterable of extensions to include (e.g. ``[".md", ".pdf"]``).
                      Defaults to a broad set of common document formats.
        recursive:    Recurse into subdirectories (default True).
        max_files:    Maximum files to return (default 1000).
        exclude_dirs: Directory names to skip (default: hidden dirs, __pycache__).

    Usage::

        from chunkymonkey.transports import DirectoryCrawler

        crawler = DirectoryCrawler(extensions=[".md", ".pdf"])
        paths = crawler.crawl("/path/to/knowledge-base/")

        # S3:
        paths = crawler.crawl("s3://my-bucket/docs/")

        # Via DocumentLoader:
        chunks = loader.load_directory("/path/to/docs/")
    """

    def __init__(
        self,
        extensions: list[str] | None = None,
        recursive: bool = True,
        max_files: int = 1000,
        exclude_dirs: list[str] | None = None,
    ):
        self.extensions: frozenset[str] = (
            frozenset(e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions)
            if extensions is not None
            else _DEFAULT_EXTENSIONS
        )
        self.recursive = recursive
        self.max_files = max_files
        self._exclude_dirs: frozenset[str] = frozenset(
            exclude_dirs or ["__pycache__", ".git", ".svn", "node_modules", ".venv", "venv"]
        )

    # ── Crawler Protocol ────────────────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        if uri.startswith("s3://"):
            return True
        if uri.startswith("file://"):
            return True
        # Bare path: absolute or relative
        parsed = urlparse(uri)
        return parsed.scheme in ("", "file") or uri.startswith(("/", "./", "../"))

    def crawl(self, uri: str, **kwargs) -> list[str]:
        """Return sorted list of document paths/URIs under *uri*.

        Args:
            uri:     Root directory path or ``s3://`` prefix.
            **kwargs: Ignored (protocol compatibility).
        """
        if uri.startswith("s3://"):
            return self._crawl_s3(uri)
        return self._crawl_local(uri)

    # ── Local ────────────────────────────────────────────────────────────────

    def _crawl_local(self, uri: str) -> list[str]:
        path_str = uri.removeprefix("file://") if uri.startswith("file://") else uri
        root = Path(path_str).expanduser().resolve()

        if not root.exists():
            raise FileNotFoundError(f"DirectoryCrawler: path not found: {root}")
        if root.is_file():
            return [str(root)] if self._accept(root) else []

        results: list[str] = []
        self._walk_local(root, results)
        results.sort()
        return results[: self.max_files]

    def _walk_local(self, directory: Path, out: list[str]) -> None:
        if len(out) >= self.max_files:
            return
        try:
            entries = sorted(directory.iterdir())
        except PermissionError as exc:
            logger.warning("DirectoryCrawler: permission denied: %s", exc)
            return
        for entry in entries:
            if len(out) >= self.max_files:
                return
            if entry.is_dir():
                if entry.name in self._exclude_dirs or entry.name.startswith("."):
                    continue
                if self.recursive:
                    self._walk_local(entry, out)
            elif entry.is_file() and self._accept(entry):
                out.append(str(entry))

    def _accept(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    # ── S3 ───────────────────────────────────────────────────────────────────

    def _crawl_s3(self, uri: str) -> list[str]:
        try:
            import boto3  # type: ignore[import]
        except ImportError:
            raise ImportError("pip install chunkymonkey[s3]  # boto3 required for S3 crawling")

        parsed = urlparse(uri)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")

        results: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if key.endswith("/"):
                    continue  # skip directory markers
                ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key.rsplit("/", 1)[-1] else ""
                if ext not in self.extensions:
                    continue
                if not self.recursive and "/" in key.removeprefix(prefix):
                    continue
                results.append(f"s3://{bucket}/{key}")
                if len(results) >= self.max_files:
                    return results

        return sorted(results)
