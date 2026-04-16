# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 6992c26b-27f2-4fff-92ef-21ffeb441a23
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""S3 transport using boto3."""

from __future__ import annotations

from urllib.parse import urlparse

from ._protocol import FetchResult

try:
    import boto3 as _boto3
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False


class S3Transport:
    """Fetch documents from Amazon S3 (s3:// and s3a:// URIs)."""

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("s3://") or uri.startswith("s3a://")

    def fetch(self, uri: str, **kwargs) -> FetchResult:
        if not _BOTO3_AVAILABLE:
            raise ImportError("pip install chunkymonkey[s3]")

        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        session_kwargs: dict = {}
        if kwargs.get("profile"):
            session_kwargs["profile_name"] = kwargs["profile"]
        if kwargs.get("region"):
            session_kwargs["region_name"] = kwargs["region"]

        session = _boto3.Session(**session_kwargs)
        s3 = session.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)

        return FetchResult(
            data=obj["Body"].read(),
            detected_mime=obj.get("ContentType"),
            source_path=uri,
        )
