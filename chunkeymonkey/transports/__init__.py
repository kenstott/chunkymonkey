"""Document transport backends."""

from ._protocol import Transport, FetchResult
from ._local import LocalTransport
from ._http import HttpTransport
from ._s3 import S3Transport
from ._ftp import FtpTransport
from ._sftp import SftpTransport

_DEFAULT_REGISTRY = [
    LocalTransport(),
    HttpTransport(),
    S3Transport(),
    FtpTransport(),
    SftpTransport(),
]


def detect_transport(uri: str) -> Transport:
    """Return the first registered transport that can handle the given URI."""
    for t in _DEFAULT_REGISTRY:
        if t.can_handle(uri):
            return t
    raise ValueError(f"No transport found for URI: {uri!r}")


__all__ = [
    "Transport",
    "FetchResult",
    "LocalTransport",
    "HttpTransport",
    "S3Transport",
    "FtpTransport",
    "SftpTransport",
    "detect_transport",
]
