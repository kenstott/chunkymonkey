"""Tests for chunkeymonkey transport backends."""

import pytest

from chunkeymonkey.transports import LocalTransport, HttpTransport, S3Transport, detect_transport
from chunkeymonkey.transports._protocol import FetchResult


# =============================================================================
# LocalTransport
# =============================================================================

class TestLocalTransport:
    def test_reads_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello from chunkeymonkey")
        result = LocalTransport().fetch(str(f))
        assert result.data == b"hello from chunkeymonkey"

    def test_returns_fetch_result(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")
        result = LocalTransport().fetch(str(f))
        assert isinstance(result, FetchResult)

    def test_source_path_set(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")
        result = LocalTransport().fetch(str(f))
        assert result.source_path is not None

    def test_can_handle_bare_path(self, tmp_path):
        assert LocalTransport().can_handle(str(tmp_path / "file.txt"))

    def test_can_handle_file_uri(self):
        assert LocalTransport().can_handle("file:///tmp/test.txt")

    def test_cannot_handle_http(self):
        assert not LocalTransport().can_handle("https://example.com/doc.pdf")

    def test_cannot_handle_s3(self):
        assert not LocalTransport().can_handle("s3://bucket/key")

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LocalTransport().fetch(str(tmp_path / "nonexistent.txt"))

    def test_reads_via_file_uri(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"file uri content")
        uri = f"file://{f}"
        result = LocalTransport().fetch(uri)
        assert result.data == b"file uri content"

    def test_reads_binary_file(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(bytes(range(256)))
        result = LocalTransport().fetch(str(f))
        assert result.data == bytes(range(256))


# =============================================================================
# HttpTransport
# =============================================================================

class TestHttpTransport:
    def test_can_handle_https(self):
        assert HttpTransport().can_handle("https://example.com/doc.pdf")

    def test_can_handle_http(self):
        assert HttpTransport().can_handle("http://internal/api")

    def test_cannot_handle_s3(self):
        assert not HttpTransport().can_handle("s3://bucket/key")

    def test_cannot_handle_local_path(self):
        assert not HttpTransport().can_handle("/usr/local/file.txt")

    def test_cannot_handle_file_uri(self):
        assert not HttpTransport().can_handle("file:///tmp/test.txt")

    def test_cannot_handle_ftp(self):
        assert not HttpTransport().can_handle("ftp://example.com/file.txt")


# =============================================================================
# S3Transport
# =============================================================================

class TestS3Transport:
    def test_can_handle_s3(self):
        assert S3Transport().can_handle("s3://my-bucket/path/file.pdf")

    def test_can_handle_s3a(self):
        assert S3Transport().can_handle("s3a://my-bucket/path/file.pdf")

    def test_cannot_handle_http(self):
        assert not S3Transport().can_handle("https://example.com/file.pdf")

    def test_cannot_handle_local(self):
        assert not S3Transport().can_handle("/local/file.pdf")

    def test_cannot_handle_ftp(self):
        assert not S3Transport().can_handle("ftp://example.com/file.txt")


# =============================================================================
# detect_transport
# =============================================================================

class TestDetectTransport:
    def test_detects_local_for_bare_path(self, tmp_path):
        p = str(tmp_path / "report.md")
        assert isinstance(detect_transport(p), LocalTransport)

    def test_detects_file_uri(self):
        assert isinstance(detect_transport("file:///tmp/test.txt"), LocalTransport)

    def test_detects_http(self):
        assert isinstance(detect_transport("https://example.com/doc.pdf"), HttpTransport)

    def test_detects_http_scheme(self):
        assert isinstance(detect_transport("http://internal.host/file"), HttpTransport)

    def test_detects_s3(self):
        assert isinstance(detect_transport("s3://bucket/key.pdf"), S3Transport)

    def test_detects_s3a(self):
        assert isinstance(detect_transport("s3a://bucket/key.pdf"), S3Transport)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            detect_transport("unknownscheme://blah/blah")
