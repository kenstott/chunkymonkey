"""FTP transport — stdlib only."""

from __future__ import annotations

from ftplib import FTP
from io import BytesIO
from urllib.parse import urlparse

from ._protocol import FetchResult


class FtpTransport:
    """Fetch documents via FTP."""

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("ftp://")

    def fetch(self, uri: str, **kwargs) -> FetchResult:
        parsed = urlparse(uri)
        host = parsed.hostname
        port = kwargs.get("port") or parsed.port or 21
        remote_path = parsed.path
        username = kwargs.get("username") or parsed.username or "anonymous"
        password = kwargs.get("password") or parsed.password or ""

        ftp = FTP()
        ftp.connect(host, port)
        ftp.login(username, password)

        buf = BytesIO()
        ftp.retrbinary(f"RETR {remote_path}", buf.write)
        ftp.quit()

        return FetchResult(
            data=buf.getvalue(),
            detected_mime=None,
            source_path=uri,
        )
