# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 4b2ec232-173b-4c2f-bf13-5bed0a1172d0
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Document transport backends."""

from ._protocol import Transport, FetchResult
from ._crawler_protocol import Crawler
from ._local import LocalTransport
from ._http import HttpTransport
from ._s3 import S3Transport
from ._ftp import FtpTransport
from ._sftp import SftpTransport
from ._web_crawler import WebCrawler
from ._directory_crawler import DirectoryCrawler
from ._sqlalchemy import SqlAlchemyTransport
from ._imap import ImapTransport

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
    "Crawler",
    "LocalTransport",
    "HttpTransport",
    "S3Transport",
    "FtpTransport",
    "SftpTransport",
    "WebCrawler",
    "DirectoryCrawler",
    "SqlAlchemyTransport",
    "ImapTransport",
    "detect_transport",
]
