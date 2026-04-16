# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 19441863-65bd-40e7-8cdb-0871d596ab60
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Integration tests for WebCrawler, DirectoryCrawler, and DocumentLoader crawl methods."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chunkymonkey.transports import Crawler, DirectoryCrawler, WebCrawler
from chunkymonkey import DocumentLoader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_ROOT_HTML = textwrap.dedent("""
    <html><body>
    <a href="/page1">Page 1</a>
    <a href="/page2.html">Page 2</a>
    <a href="https://other-domain.com/external">External</a>
    <a href="https://example.com/image.png">Image (binary)</a>
    </body></html>
""").encode()

_PAGE1_HTML = textwrap.dedent("""
    <html><body><h1>Page 1</h1><p>Content here.</p></body></html>
""").encode()

_PAGE2_HTML = textwrap.dedent("""
    <html><body><h1>Page 2</h1><p>More content.</p></body></html>
""").encode()


def _fake_http_get(url: str, timeout: int):
    responses = {
        "https://example.com/": (_ROOT_HTML, "text/html; charset=utf-8"),
        "https://example.com/page1": (_PAGE1_HTML, "text/html; charset=utf-8"),
        "https://example.com/page2.html": (_PAGE2_HTML, "text/html; charset=utf-8"),
    }
    if url in responses:
        return responses[url]
    raise OSError(f"Not mocked: {url}")


# ─────────────────────────────────────────────────────────────────────────────
# Crawler Protocol
# ─────────────────────────────────────────────────────────────────────────────

class TestCrawlerProtocol:
    def test_web_crawler_satisfies_protocol(self):
        assert isinstance(WebCrawler(), Crawler)

    def test_directory_crawler_satisfies_protocol(self):
        assert isinstance(DirectoryCrawler(), Crawler)

    def test_custom_crawler_satisfies_protocol(self):
        class MyCustomCrawler:
            def can_handle(self, uri: str) -> bool:
                return uri.startswith("custom://")
            def crawl(self, uri: str, **kwargs) -> list[str]:
                return [f"{uri}/doc1", f"{uri}/doc2"]

        assert isinstance(MyCustomCrawler(), Crawler)


# ─────────────────────────────────────────────────────────────────────────────
# WebCrawler
# ─────────────────────────────────────────────────────────────────────────────

class TestWebCrawler:
    def test_can_handle_http(self):
        c = WebCrawler()
        assert c.can_handle("http://example.com")
        assert c.can_handle("https://example.com/path")
        assert not c.can_handle("s3://bucket/key")
        assert not c.can_handle("/local/path")

    @patch("chunkymonkey.transports._web_crawler._http_get", side_effect=_fake_http_get)
    def test_crawl_same_domain_only(self, mock_get):
        c = WebCrawler(max_pages=10, same_domain=True)
        urls = c.crawl("https://example.com/")
        assert "https://example.com/" in urls
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2.html" in urls
        # External domain excluded
        assert not any("other-domain.com" in u for u in urls)

    @patch("chunkymonkey.transports._web_crawler._http_get", side_effect=_fake_http_get)
    def test_crawl_binary_urls_excluded(self, mock_get):
        c = WebCrawler(max_pages=10)
        urls = c.crawl("https://example.com/")
        assert not any(u.endswith(".png") for u in urls)

    @patch("chunkymonkey.transports._web_crawler._http_get", side_effect=_fake_http_get)
    def test_crawl_max_pages(self, mock_get):
        c = WebCrawler(max_pages=1)
        urls = c.crawl("https://example.com/")
        assert len(urls) == 1
        assert urls[0] == "https://example.com/"

    @patch("chunkymonkey.transports._web_crawler._http_get", side_effect=_fake_http_get)
    def test_crawl_max_depth_zero(self, mock_get):
        c = WebCrawler(max_pages=50, max_depth=0)
        urls = c.crawl("https://example.com/")
        # Depth 0 means root only (no child links followed)
        assert urls == ["https://example.com/"]

    @patch("chunkymonkey.transports._web_crawler._http_get", side_effect=_fake_http_get)
    def test_crawl_include_pattern(self, mock_get):
        c = WebCrawler(max_pages=10, include_pattern=r"page1")
        urls = c.crawl("https://example.com/")
        # Root is always included; only page1 from links
        assert "https://example.com/" in urls
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2.html" not in urls

    @patch("chunkymonkey.transports._web_crawler._http_get", side_effect=_fake_http_get)
    def test_crawl_exclude_pattern(self, mock_get):
        c = WebCrawler(max_pages=10, exclude_patterns=[r"page2"])
        urls = c.crawl("https://example.com/")
        assert "https://example.com/page2.html" not in urls

    def test_crawl_root_fetch_failure_returns_empty(self):
        from urllib.error import URLError
        with patch("chunkymonkey.transports._web_crawler._http_get", side_effect=URLError("timeout")):
            c = WebCrawler()
            urls = c.crawl("https://unreachable.example.com/")
            assert urls == []

    @patch("chunkymonkey.transports._web_crawler._http_get", side_effect=_fake_http_get)
    def test_crawl_no_duplicates(self, mock_get):
        c = WebCrawler(max_pages=50)
        urls = c.crawl("https://example.com/")
        assert len(urls) == len(set(urls))


# ─────────────────────────────────────────────────────────────────────────────
# DirectoryCrawler — local
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectoryCrawler:
    def test_can_handle_local_path(self):
        c = DirectoryCrawler()
        assert c.can_handle("/abs/path")
        assert c.can_handle("./rel/path")
        assert c.can_handle("file:///abs/path")
        assert not c.can_handle("https://example.com")

    def test_can_handle_s3(self):
        c = DirectoryCrawler()
        assert c.can_handle("s3://my-bucket/prefix/")

    def test_crawl_single_file(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Hello")
        c = DirectoryCrawler()
        result = c.crawl(str(f))
        assert result == [str(f)]

    def test_crawl_directory(self, tmp_path):
        (tmp_path / "a.md").write_text("# A")
        (tmp_path / "b.txt").write_text("B content")
        (tmp_path / "c.png").write_bytes(b"\x89PNG")
        c = DirectoryCrawler()
        result = c.crawl(str(tmp_path))
        names = [Path(p).name for p in result]
        assert "a.md" in names
        assert "b.txt" in names
        assert "c.png" not in names  # not in default extensions

    def test_crawl_recursive(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.md").write_text("root")
        (sub / "nested.md").write_text("nested")
        c = DirectoryCrawler(recursive=True)
        result = c.crawl(str(tmp_path))
        names = [Path(p).name for p in result]
        assert "root.md" in names
        assert "nested.md" in names

    def test_crawl_non_recursive(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.md").write_text("root")
        (sub / "nested.md").write_text("nested")
        c = DirectoryCrawler(recursive=False)
        result = c.crawl(str(tmp_path))
        names = [Path(p).name for p in result]
        assert "root.md" in names
        assert "nested.md" not in names

    def test_crawl_custom_extensions(self, tmp_path):
        (tmp_path / "doc.md").write_text("# Hello")
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "other.log").write_text("log entry")
        c = DirectoryCrawler(extensions=[".md"])
        result = c.crawl(str(tmp_path))
        names = [Path(p).name for p in result]
        assert "doc.md" in names
        assert "data.csv" not in names

    def test_crawl_max_files(self, tmp_path):
        for i in range(10):
            (tmp_path / f"doc{i}.md").write_text(f"# Doc {i}")
        c = DirectoryCrawler(max_files=3)
        result = c.crawl(str(tmp_path))
        assert len(result) <= 3

    def test_crawl_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.md").write_text("secret")
        (tmp_path / "visible.md").write_text("visible")
        c = DirectoryCrawler()
        result = c.crawl(str(tmp_path))
        names = [Path(p).name for p in result]
        assert "visible.md" in names
        assert "secret.md" not in names

    def test_crawl_skips_excluded_dirs(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "module.md").write_text("cached")
        (tmp_path / "real.md").write_text("real")
        c = DirectoryCrawler()
        result = c.crawl(str(tmp_path))
        names = [Path(p).name for p in result]
        assert "real.md" in names
        assert "module.md" not in names

    def test_crawl_nonexistent_raises(self):
        c = DirectoryCrawler()
        with pytest.raises(FileNotFoundError):
            c.crawl("/nonexistent/path/that/does/not/exist")

    def test_crawl_file_uri_scheme(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Hello")
        c = DirectoryCrawler()
        result = c.crawl(f"file://{f}")
        assert str(f) in result


# ─────────────────────────────────────────────────────────────────────────────
# DocumentLoader crawl methods
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentLoaderCrawl:
    def test_load_directory(self, tmp_path):
        (tmp_path / "doc1.md").write_text("# Title\n\nSome content here.")
        (tmp_path / "doc2.txt").write_text("Plain text content.")
        loader = DocumentLoader()
        chunks = loader.load_directory(str(tmp_path))
        assert len(chunks) > 0
        doc_names = {c.document_name for c in chunks}
        assert "doc1" in doc_names
        assert "doc2" in doc_names

    def test_load_directory_with_extensions_filter(self, tmp_path):
        (tmp_path / "doc.md").write_text("# Markdown doc")
        (tmp_path / "doc.txt").write_text("Text doc")
        loader = DocumentLoader()
        chunks = loader.load_directory(str(tmp_path), extensions=[".md"])
        doc_names = {c.document_name for c in chunks}
        assert "doc" in doc_names
        # txt should not appear
        txt_chunks = [c for c in chunks if c.document_name == "doc" and c.content.startswith("Text")]
        assert not txt_chunks

    def test_load_directory_with_custom_crawler(self, tmp_path):
        (tmp_path / "custom.md").write_text("# Custom crawl")

        class FixedCrawler:
            def can_handle(self, uri): return True
            def crawl(self, uri, **kw): return [str(tmp_path / "custom.md")]

        loader = DocumentLoader()
        chunks = loader.load_directory(str(tmp_path), crawler=FixedCrawler())
        assert any(c.document_name == "custom" for c in chunks)

    @patch("chunkymonkey.transports._web_crawler._http_get", side_effect=_fake_http_get)
    def test_load_site_returns_chunks(self, mock_get):
        from chunkymonkey.transports._protocol import FetchResult

        def fake_transport_fetch(uri, **kwargs):
            body, ct = _fake_http_get(uri, 20)
            return FetchResult(data=body, detected_mime=ct, source_path=uri)

        with patch("chunkymonkey.transports._http.HttpTransport.fetch", side_effect=fake_transport_fetch):
            loader = DocumentLoader()
            chunks = loader.load_site("https://example.com/", max_pages=5)
        assert len(chunks) > 0

    @patch("chunkymonkey.transports._web_crawler._http_get", side_effect=_fake_http_get)
    def test_load_site_with_custom_crawler(self, mock_get):
        from chunkymonkey.transports._protocol import FetchResult

        def fake_transport_fetch(uri, **kwargs):
            body, ct = _fake_http_get(uri, 20)
            return FetchResult(data=body, detected_mime=ct, source_path=uri)

        class FixedCrawler:
            def can_handle(self, uri): return True
            def crawl(self, uri, **kw): return ["https://example.com/page1"]

        with patch("chunkymonkey.transports._http.HttpTransport.fetch", side_effect=fake_transport_fetch):
            loader = DocumentLoader()
            chunks = loader.load_site("https://example.com/", crawler=FixedCrawler())
        assert len(chunks) > 0

    def test_load_crawl_skips_failed_uris(self, tmp_path):
        good = tmp_path / "good.md"
        good.write_text("# Good document")

        class MixedCrawler:
            def can_handle(self, uri): return True
            def crawl(self, uri, **kw):
                return [str(good), "/nonexistent/bad.md"]

        loader = DocumentLoader()
        # Should not raise; bad URI is logged and skipped
        chunks = loader.load_crawl(str(tmp_path), crawler=MixedCrawler())
        assert any(c.document_name == "good" for c in chunks)

    def test_load_crawl_auto_selects_web_crawler_for_http(self):
        from urllib.error import URLError
        with patch(
            "chunkymonkey.transports._web_crawler._http_get",
            side_effect=URLError("connection refused"),
        ):
            loader = DocumentLoader()
            # Should auto-select WebCrawler and return [] on connection failure
            chunks = loader.load_crawl("https://unreachable.example.com/")
            assert chunks == []

    def test_load_crawl_auto_selects_dir_crawler_for_local(self, tmp_path):
        (tmp_path / "auto.md").write_text("# Auto-detected")
        loader = DocumentLoader()
        chunks = loader.load_crawl(str(tmp_path))
        assert any(c.document_name == "auto" for c in chunks)
