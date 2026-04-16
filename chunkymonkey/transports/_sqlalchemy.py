# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SQLAlchemy transport — executes a SQL query and returns the result as CSV bytes.

The query can be supplied in two ways:

**URI query-parameter** (simple queries, no bind params)::

    loader = DocumentLoader(extra_transports=[SqlAlchemyTransport("postgresql://...")])
    chunks = loader.load("sqlalchemy://articles?query=SELECT+title%2C+body+FROM+articles")

**Constructor** (complex queries, bind parameters)::

    transport = SqlAlchemyTransport(
        connection_url="postgresql://user:pass@host/db",
        query="SELECT title, body FROM articles WHERE published = :pub",
        params={"pub": True},
    )
    loader = DocumentLoader(extra_transports=[transport])
    chunks = loader.load("sqlalchemy://articles")

In both cases the URI path after ``sqlalchemy://`` is used as the document name.
The ``?query=`` parameter takes precedence over the constructor *query* when both
are present.

Requires: sqlalchemy>=2.0 (included in the ``storage`` extra)
"""

from __future__ import annotations

import csv
import io
from urllib.parse import urlparse, parse_qs, unquote_plus

from ._protocol import FetchResult


class SqlAlchemyTransport:
    """Fetch rows from a SQL query and return them as CSV bytes."""

    SCHEME = "sqlalchemy"

    def __init__(
        self,
        connection_url: str,
        query: str | None = None,
        params: dict | None = None,
    ):
        """Args:
            connection_url: SQLAlchemy connection URL (e.g. ``postgresql://...``).
            query:          Default SQL query.  Overridden by ``?query=`` in the URI.
            params:         Optional dict of named bind parameter values.
        """
        self._url = connection_url
        self._default_query = query
        self._params = params or {}

    def can_handle(self, uri: str) -> bool:
        return uri.startswith(f"{self.SCHEME}://")

    def fetch(self, uri: str, **kwargs) -> FetchResult:
        try:
            import sqlalchemy as sa
        except ImportError as exc:
            raise ImportError(
                "sqlalchemy is required for SqlAlchemyTransport. "
                "Install it with: pip install sqlalchemy"
            ) from exc

        parsed = urlparse(uri)
        qs = parse_qs(parsed.query)

        # URI ?query= wins over constructor default
        if "query" in qs:
            sql = unquote_plus(qs["query"][0])
        elif self._default_query:
            sql = self._default_query
        else:
            raise ValueError(
                f"No SQL query provided. Pass ?query=... in the URI or set query= in the constructor."
            )

        # Document name comes from the URI path (strip leading slash)
        doc_name = parsed.netloc or (parsed.path.lstrip("/") or "query")

        engine = sa.create_engine(self._url)
        with engine.connect() as conn:
            result = conn.execute(sa.text(sql), self._params)
            columns = list(result.keys())
            rows = result.fetchall()

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(columns)
        writer.writerows(rows)
        csv_bytes = buf.getvalue().encode("utf-8")

        return FetchResult(
            data=csv_bytes,
            detected_mime="text/csv",
            source_path=doc_name,
        )