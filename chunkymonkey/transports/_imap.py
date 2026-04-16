# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""IMAP transport — fetches email messages from a mailbox using stdlib imaplib.

URI format::

    imap://user:password@host:143/INBOX
    imaps://user:password@host:993/INBOX
    imaps://user:password@host:993/INBOX?search=UNSEEN&limit=50
    imaps://user:password@host:993/INBOX?search=FROM%20boss%40example.com

Search criteria follow IMAP RFC 3501 syntax (ALL, UNSEEN, FROM x, SUBJECT y,
SINCE 01-Jan-2025, etc.).  ``limit`` caps the number of messages returned,
taking the most-recent N by UID descending.
"""

from __future__ import annotations

import imaplib
import logging
from typing import Iterator
from urllib.parse import urlparse, parse_qs, unquote

from ._protocol import FetchResult

logger = logging.getLogger(__name__)

_DEFAULT_IMAP_PORT = 143
_DEFAULT_IMAPS_PORT = 993


class ImapTransport:
    """Fetch email messages from an IMAP mailbox.

    Implements the standard ``Transport`` protocol (``can_handle`` / ``fetch``)
    plus ``fetch_messages`` which yields one ``FetchResult`` per message.

    ``fetch()`` returns only the most-recent message; use ``fetch_messages()``
    (or ``DocumentLoader.load_imap()``) to retrieve multiple.
    """

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("imap://") or uri.startswith("imaps://")

    def fetch(self, uri: str, **kwargs) -> FetchResult:
        """Return the single most-recent message matching the URI."""
        results = list(self.fetch_messages(uri, limit=1))
        if not results:
            raise ValueError(f"No messages found for URI: {uri!r}")
        return results[0]

    def fetch_messages(
        self,
        uri: str,
        limit: int | None = None,
    ) -> Iterator[FetchResult]:
        """Yield one FetchResult per message (raw RFC 822 bytes).

        Args:
            uri:   ``imap[s]://user:pass@host[:port]/mailbox[?search=...&limit=N]``
            limit: Cap on messages returned (most-recent first).  Overrides the
                   ``limit`` query parameter in the URI if both are provided.
        """
        parsed = urlparse(uri)
        ssl = parsed.scheme == "imaps"
        host = parsed.hostname or "localhost"
        port = parsed.port or (_DEFAULT_IMAPS_PORT if ssl else _DEFAULT_IMAP_PORT)
        user = unquote(parsed.username or "")
        password = unquote(parsed.password or "")
        mailbox = unquote(parsed.path.lstrip("/")) or "INBOX"

        qs = parse_qs(parsed.query)
        search_criteria = qs.get("search", ["ALL"])[0]
        uri_limit = qs.get("limit", [None])[0]
        effective_limit = limit if limit is not None else (int(uri_limit) if uri_limit else None)

        conn = imaplib.IMAP4_SSL(host, port) if ssl else imaplib.IMAP4(host, port)
        try:
            conn.login(user, password)
            conn.select(mailbox, readonly=True)

            typ, data = conn.search(None, search_criteria)
            if typ != "OK" or not data or not data[0]:
                return

            msg_ids: list[bytes] = data[0].split()
            # Most-recent first (IMAP IDs are ascending by arrival)
            msg_ids = list(reversed(msg_ids))
            if effective_limit is not None:
                msg_ids = msg_ids[:effective_limit]

            for msg_id in msg_ids:
                try:
                    typ2, msg_data = conn.fetch(msg_id, "(RFC822)")
                    if typ2 != "OK" or not msg_data or msg_data[0] is None:
                        continue
                    raw: bytes = msg_data[0][1]  # type: ignore[index]
                    source = f"imap://{host}/{mailbox}/{msg_id.decode()}"
                    yield FetchResult(
                        data=raw,
                        detected_mime="message/rfc822",
                        source_path=source,
                    )
                except Exception as exc:
                    logger.warning("ImapTransport: skipping message %s: %s", msg_id, exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass
            try:
                conn.logout()
            except Exception:
                pass
