# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 691bd007-a58c-4f0e-8776-198a2c5c02c3
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SFTP transport using paramiko."""

from __future__ import annotations

from io import BytesIO
from typing import Any
from urllib.parse import urlparse

from ._protocol import FetchOptions, FetchResult

try:
    import paramiko as _paramiko

    _PARAMIKO_AVAILABLE = True
except ImportError:
    _paramiko = None  # type: ignore[assignment]
    _PARAMIKO_AVAILABLE = False


class SftpTransport:
    """Fetch documents via SFTP."""

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("sftp://")

    def fetch(self, uri: str, options: FetchOptions | None = None) -> FetchResult:
        if not _PARAMIKO_AVAILABLE:
            raise ImportError("pip install chonk-rag[sftp]")

        parsed = urlparse(uri)
        host = parsed.hostname
        port = (options.port if options else None) or parsed.port or 22
        remote_path = parsed.path

        assert _paramiko is not None  # guarded by _PARAMIKO_AVAILABLE check above
        client = _paramiko.SSHClient()
        client.set_missing_host_key_policy(_paramiko.AutoAddPolicy())  # nosec B507

        connect_kwargs: dict[str, Any] = {
            "hostname": host,
            "port": port,
        }
        username = (options.username if options else None) or parsed.username
        password = (options.password if options else None) or parsed.password
        key_path = options.key_path if options else None

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
