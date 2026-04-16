# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 8695d94f-ed02-46c6-bf6e-ce109870096a
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for chunkymonkey transport backends."""

import pytest

from chunkymonkey.transports import LocalTransport, HttpTransport, S3Transport, detect_transport
from chunkymonkey.transports._protocol import FetchResult


# =============================================================================
# LocalTransport
# =============================================================================

class TestLocalTransport:
    def test_reads_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello from chunkymonkey")
        result = LocalTransport().fetch(str(f))
        assert result.data == b"hello from chunkymonkey"

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


# =============================================================================
# ImapTransport
# =============================================================================

class TestImapTransport:
    """Tests for ImapTransport — network calls are mocked via unittest.mock."""

    def test_can_handle_imap(self):
        from chunkymonkey.transports._imap import ImapTransport
        assert ImapTransport().can_handle("imap://user:pass@mail.example.com/INBOX")

    def test_can_handle_imaps(self):
        from chunkymonkey.transports._imap import ImapTransport
        assert ImapTransport().can_handle("imaps://user:pass@imap.gmail.com/INBOX")

    def test_cannot_handle_http(self):
        from chunkymonkey.transports._imap import ImapTransport
        assert not ImapTransport().can_handle("https://example.com/doc.pdf")

    def test_cannot_handle_local(self):
        from chunkymonkey.transports._imap import ImapTransport
        assert not ImapTransport().can_handle("/local/path/to/file.eml")

    def test_cannot_handle_s3(self):
        from chunkymonkey.transports._imap import ImapTransport
        assert not ImapTransport().can_handle("s3://bucket/key")

    def _mock_imap(self, raw_messages: list[bytes], search_ids: list[str] | None = None):
        """Return a mock IMAP4_SSL instance that returns the given raw messages."""
        from unittest.mock import MagicMock
        ids = search_ids or [str(i + 1) for i in range(len(raw_messages))]
        id_string = " ".join(ids).encode()

        conn = MagicMock()
        conn.login.return_value = ("OK", [b"Logged in"])
        conn.select.return_value = ("OK", [b"1"])
        conn.search.return_value = ("OK", [id_string])

        def _fetch(msg_id, spec):
            idx = ids.index(msg_id.decode())
            return ("OK", [(b"1 (RFC822 {N})", raw_messages[idx])])

        conn.fetch.side_effect = _fetch
        conn.close.return_value = ("OK", [])
        conn.logout.return_value = ("OK", [b"Bye"])
        return conn

    def test_fetch_messages_yields_fetch_results(self):
        from unittest.mock import patch
        from chunkymonkey.transports._imap import ImapTransport
        from chunkymonkey.transports._protocol import FetchResult

        raw = b"From: a@b.com\r\nSubject: Hi\r\n\r\nBody"
        conn = self._mock_imap([raw])

        with patch("imaplib.IMAP4_SSL", return_value=conn):
            results = list(ImapTransport().fetch_messages(
                "imaps://user:pass@imap.example.com/INBOX"
            ))

        assert len(results) == 1
        assert isinstance(results[0], FetchResult)
        assert results[0].data == raw
        assert results[0].detected_mime == "message/rfc822"

    def test_fetch_messages_limit(self):
        from unittest.mock import patch
        from chunkymonkey.transports._imap import ImapTransport

        msgs = [f"From: a@b.com\r\n\r\nMsg {i}".encode() for i in range(5)]
        conn = self._mock_imap(msgs)

        with patch("imaplib.IMAP4_SSL", return_value=conn):
            results = list(ImapTransport().fetch_messages(
                "imaps://user:pass@imap.example.com/INBOX", limit=2
            ))

        assert len(results) == 2

    def test_fetch_messages_most_recent_first(self):
        from unittest.mock import patch
        from chunkymonkey.transports._imap import ImapTransport

        msgs = [
            b"From: a@b.com\r\n\r\nOlder",
            b"From: a@b.com\r\n\r\nNewer",
        ]
        conn = self._mock_imap(msgs)

        with patch("imaplib.IMAP4_SSL", return_value=conn):
            results = list(ImapTransport().fetch_messages(
                "imaps://user:pass@imap.example.com/INBOX", limit=1
            ))

        # Should return the last (highest UID = most recent) message
        assert b"Newer" in results[0].data

    def test_fetch_returns_single_result(self):
        from unittest.mock import patch
        from chunkymonkey.transports._imap import ImapTransport

        raw = b"From: a@b.com\r\nSubject: Only\r\n\r\nSingle"
        conn = self._mock_imap([raw])

        with patch("imaplib.IMAP4_SSL", return_value=conn):
            result = ImapTransport().fetch("imaps://user:pass@imap.example.com/INBOX")

        assert result.data == raw

    def test_fetch_no_messages_raises(self):
        from unittest.mock import patch, MagicMock
        from chunkymonkey.transports._imap import ImapTransport

        conn = MagicMock()
        conn.login.return_value = ("OK", [])
        conn.select.return_value = ("OK", [b"0"])
        conn.search.return_value = ("OK", [b""])
        conn.close.return_value = ("OK", [])
        conn.logout.return_value = ("OK", [])

        with patch("imaplib.IMAP4_SSL", return_value=conn):
            with pytest.raises(ValueError):
                ImapTransport().fetch("imaps://user:pass@imap.example.com/INBOX")

    def test_uses_imap4_for_plain_imap(self):
        from unittest.mock import patch
        from chunkymonkey.transports._imap import ImapTransport

        raw = b"From: a@b.com\r\n\r\nPlain"
        conn = self._mock_imap([raw])

        with patch("imaplib.IMAP4", return_value=conn) as mock_cls:
            list(ImapTransport().fetch_messages("imap://user:pass@mail.example.com/INBOX"))
            mock_cls.assert_called_once_with("mail.example.com", 143)
