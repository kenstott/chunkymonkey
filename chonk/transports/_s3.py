# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 6992c26b-27f2-4fff-92ef-21ffeb441a23
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""S3 transport using boto3."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ._protocol import FetchOptions, FetchResult

try:
    import boto3 as _boto3

    _BOTO3_AVAILABLE = True
except ImportError:
    _boto3 = None  # type: ignore[assignment]
    _BOTO3_AVAILABLE = False


class S3Transport:
    """Fetch documents from Amazon S3 (s3:// and s3a:// URIs)."""

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("s3://") or uri.startswith("s3a://")

    def fetch(self, uri: str, options: FetchOptions | None = None) -> FetchResult:
        if not _BOTO3_AVAILABLE:
            raise ImportError("pip install chonk-rag[s3]")

        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        session_kwargs: dict[str, Any] = {}
        if options and options.profile:
            session_kwargs["profile_name"] = options.profile
        if options and options.region:
            session_kwargs["region_name"] = options.region

        import os

        if not session_kwargs.get("region_name"):
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            if region:
                session_kwargs["region_name"] = region

        assert _boto3 is not None  # guarded by _BOTO3_AVAILABLE check above
        session = _boto3.Session(**session_kwargs)
        client_kwargs: dict[str, Any] = {}
        opt_endpoint = options.endpoint_url if options else None
        endpoint = opt_endpoint or os.environ.get("AWS_ENDPOINT_OVERRIDE")
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
        s3 = session.client("s3", **client_kwargs)
        obj = s3.get_object(Bucket=bucket, Key=key)

        return FetchResult(
            data=obj["Body"].read(),
            detected_mime=obj.get("ContentType"),
            source_path=uri,
        )
