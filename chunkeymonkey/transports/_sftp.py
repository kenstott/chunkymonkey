"""SFTP transport using paramiko."""

from __future__ import annotations

from io import BytesIO
from urllib.parse import urlparse

from ._protocol import FetchResult

try:
    import paramiko as _paramiko
    _PARAMIKO_AVAILABLE = True
except ImportError:
    _PARAMIKO_AVAILABLE = False


class SftpTransport:
    """Fetch documents via SFTP."""

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("sftp://")

    def fetch(self, uri: str, **kwargs) -> FetchResult:
        if not _PARAMIKO_AVAILABLE:
            raise ImportError("pip install chunkeymonkey[sftp]")

        parsed = urlparse(uri)
        host = parsed.hostname
        port = kwargs.get("port") or parsed.port or 22
        remote_path = parsed.path

        client = _paramiko.SSHClient()
        client.set_missing_host_key_policy(_paramiko.AutoAddPolicy())

        connect_kwargs: dict = {
            "hostname": host,
            "port": port,
        }
        username = kwargs.get("username") or parsed.username
        password = kwargs.get("password") or parsed.password
        key_path = kwargs.get("key_path")

        if username:
            connect_kwargs["username"] = username
        if password:
            connect_kwargs["password"] = password
        if key_path:
            connect_kwargs["key_filename"] = key_path

        client.connect(**connect_kwargs)
        sftp = client.open_sftp()

        buf = BytesIO()
        sftp.getfo(remote_path, buf)

        sftp.close()
        client.close()

        return FetchResult(
            data=buf.getvalue(),
            detected_mime=None,
            source_path=uri,
        )
