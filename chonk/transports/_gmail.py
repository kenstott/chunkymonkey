# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 5b8e2a4f-9c1d-4e7b-a3f6-0d2c8e5b1a7f
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GmailCrawler — index Gmail messages via the Gmail REST API.

Implements both the ``Crawler`` and ``Transport`` protocols. Pass the same
instance as both ``crawler=`` and in ``extra_transports=``.

Authentication uses OAuth2 with a stored token file. On the first run the
browser opens for the consent flow and the token is saved to ``token_path``
for all subsequent runs.

Requires: ``google-api-python-client``, ``google-auth-oauthlib``
    pip install chonk-rag[gmail]

Usage::

    from chonk.transports import GmailCrawler
    from chonk.loader import DocumentLoader

    crawler = GmailCrawler(
        client_id="...",
        client_secret="...",
        # token_path defaults to ~/.chonk/gmail_token.json
    )
    loader = DocumentLoader(extra_transports=[crawler])

    # All inbox messages
    chunks = loader.load_crawl("gmail://me/INBOX", crawler=crawler)

    # Filtered
    chunks = loader.load_crawl(
        "gmail://me/INBOX",
        crawler=crawler,
        query="is:unread after:2025/01/01",
        limit=100,
    )
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._protocol import FetchOptions, FetchResult

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
_DEFAULT_TOKEN_PATH = Path.home() / ".chonk" / "gmail_token.json"

# Gmail label → IMAP-style mailbox name mapping
_LABEL_MAP = {
    "INBOX": "INBOX",
    "SENT": "[Gmail]/Sent Mail",
    "DRAFTS": "[Gmail]/Drafts",
    "SPAM": "[Gmail]/Spam",
    "TRASH": "[Gmail]/Trash",
    "ALL": "[Gmail]/All Mail",
}


def _decode_body(part: dict[str, Any]) -> str:
    """Decode a base64url-encoded message body part to text."""
    data = part.get("body", {}).get("data", "")
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_text(payload: dict[str, Any]) -> str:
    """Recursively extract text/plain (preferred) or text/html from payload."""
    mime = payload.get("mimeType", "")
    parts = payload.get("parts", [])

    if mime == "text/plain":
        return _decode_body(payload)
    if mime == "text/html" and not parts:
        return _decode_body(payload)

    # Prefer text/plain part; fall back to text/html
    plain = next((p for p in parts if p.get("mimeType") == "text/plain"), None)
    if plain:
        return _decode_body(plain)
    html = next((p for p in parts if p.get("mimeType") == "text/html"), None)
    if html:
        return _decode_body(html)

    # Recurse into multipart
    for part in parts:
        text = _extract_text(part)
        if text:
            return text
    return ""


def _message_to_text(msg: dict[str, Any]) -> str:
    """Serialize a Gmail message to plain text for chunking."""
    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
    lines = []
    for field in ("From", "To", "Cc", "Subject", "Date"):
        if field in headers:
            lines.append(f"{field}: {headers[field]}")
    body = _extract_text(msg.get("payload", {}))
    if body:
        lines.append("")
        lines.append(body.strip())
    return "\n".join(lines)


class GmailCrawler:
    """Crawl Gmail messages via the Gmail REST API.

    Implements both ``Crawler`` and ``Transport`` — pass the same instance to both::

        crawler = GmailCrawler(client_id="...", client_secret="...")
        loader = DocumentLoader(extra_transports=[crawler])
        chunks = loader.load_crawl("gmail://me/INBOX", crawler=crawler)

    On the first run a browser window opens for OAuth2 consent. The resulting
    token is saved to ``token_path`` (default ``~/.chonk/gmail_token.json``)
    and reused on subsequent runs with automatic refresh.

    Args:
        client_id:     Google OAuth2 client ID.
        client_secret: Google OAuth2 client secret.
        token_path:    Path to store the OAuth2 token. Defaults to
                       ``~/.chonk/gmail_token.json``.
        user_id:       Gmail user ID (default ``"me"`` = authenticated user).
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_path: str | Path | None = None,
        user_id: str = "me",
        redirect_port: int = 8000,
    ) -> None:
        self._client_id = client_id or os.environ.get("GOOGLE_EMAIL_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get("GOOGLE_EMAIL_CLIENT_SECRET", "")
        self._token_path = Path(token_path) if token_path else _DEFAULT_TOKEN_PATH
        self._user_id = user_id
        self._redirect_port = redirect_port
        self._service: Any = None
        self._cache: dict[str, FetchResult] = {}
        self._url_key = hashlib.md5(self._client_id.encode(), usedforsecurity=False).hexdigest()[:8]

    # ── Transport + Crawler Protocol ─────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("gmail://") or uri.startswith(f"gmsg://{self._url_key}/")

    def fetch(self, uri: str, options: FetchOptions | None = None) -> FetchResult:
        if uri in self._cache:
            return self._cache[uri]
        # Lazy fetch by message ID
        if uri.startswith(f"gmsg://{self._url_key}/"):
            msg_id = uri.split("/")[-1]
            return self._fetch_message(msg_id, uri)
        raise KeyError(f"GmailCrawler: unknown URI {uri!r} — call crawl() first")

    def crawl(
        self,
        _uri: str = "",
        query: str = "",
        limit: int = 100,
        **__: object,
    ) -> list[str]:
        """List Gmail messages and return their ``gmsg://`` URIs.

        Args:
            _uri:   ``gmail://me/INBOX`` or similar (label parsed from path;
                    ignored if ``query`` is supplied).
            query:  Gmail search query (e.g. ``"is:unread after:2025/01/01"``).
                    Defaults to all messages in INBOX.
            limit:  Maximum number of messages to return (default 100).
            **__:   Ignored (protocol compatibility).

        Returns:
            List of ``gmsg://`` URIs, one per message.
        """
        svc = self._get_service()
        label = self._label_from_uri(_uri)

        kwargs: dict[str, Any] = {"userId": self._user_id, "maxResults": min(limit, 500)}
        if query:
            kwargs["q"] = query
        else:
            kwargs["labelIds"] = [label]

        uris: list[str] = []
        page_token: str | None = None

        while len(uris) < limit:
            if page_token:
                kwargs["pageToken"] = page_token
            resp = svc.users().messages().list(**kwargs).execute()
            messages = resp.get("messages", [])
            for m in messages:
                if len(uris) >= limit:
                    break
                uri = f"gmsg://{self._url_key}/{m['id']}"
                uris.append(uri)
            page_token = resp.get("nextPageToken")
            if not page_token or not messages:
                break

        _log.info("GmailCrawler: %d message(s) found", len(uris))
        return uris

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_service(self) -> Any:  # noqa: ANN401
        if self._service is not None:
            return self._service
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "pip install google-api-python-client google-auth-oauthlib  "
                "# required for GmailCrawler"
            ) from None

        from google.auth.credentials import Credentials as _BaseCredentials

        creds: _BaseCredentials | None = None
        if self._token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self._token_path), _SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                client_config = {
                    "installed": {
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                }
                flow = InstalledAppFlow.from_client_config(client_config, _SCOPES)
                creds = flow.run_local_server(port=self._redirect_port)

            if creds is None:
                raise RuntimeError("GmailCrawler: OAuth flow produced no credentials")
            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_path.write_text(creds.to_json())
            _log.info("GmailCrawler: token saved to %s", self._token_path)

        self._service = build("gmail", "v1", credentials=creds)
        return self._service

    def _fetch_message(self, msg_id: str, uri: str) -> FetchResult:
        svc = self._get_service()
        msg = svc.users().messages().get(userId=self._user_id, id=msg_id, format="full").execute()
        text = _message_to_text(msg)
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        result = FetchResult(
            data=text.encode("utf-8"),
            detected_mime="text/plain",
            source_path=headers.get("Subject", uri),
        )
        self._cache[uri] = result
        return result

    @staticmethod
    def _label_from_uri(uri: str) -> str:
        """Extract Gmail label ID from a gmail:// URI path."""
        if not uri.startswith("gmail://"):
            return "INBOX"
        path = uri.split("/", 3)
        # Gmail API accepts label names like INBOX, SENT, etc.
        return path[3].upper() if len(path) > 3 else "INBOX"
