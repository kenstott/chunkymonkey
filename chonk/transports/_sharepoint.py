# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 2a9f4c1e-8b3d-4e7f-a5c2-9d6e0f1b3a8c
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SharePointCrawler — index SharePoint document libraries, lists, calendars, and pages.

Implements both the ``Crawler`` and ``Transport`` protocols. Pass the same
instance as both ``crawler=`` and in ``extra_transports=``.

Three authentication modes are supported:

- ``"azure_ad"``  — Azure AD app registration (client credentials). Uses
                    Microsoft Graph API. Requires ``msal`` package.
- ``"legacy"``    — SharePoint Add-in (client_id + client_secret registered via
                    appregnew.aspx). Uses SharePoint REST API (``/_api/``).
                    No extra packages required beyond ``requests``.
- ``"ntlm"``      — NTLM/Kerberos for on-premises SharePoint Server. Uses
                    SharePoint REST API. Requires ``requests-ntlm`` package.

Artifact types crawled (all enabled by default):

- ``"documents"`` — Document libraries (files fetched lazily on demand)
- ``"lists"``     — Generic SharePoint lists (items serialized as text)
- ``"calendars"`` — Calendar/Events lists (items serialized as text)
- ``"pages"``     — Site Pages library (HTML content)

Usage::

    from chonk.transports import SharePointCrawler
    from chonk.loader import DocumentLoader

    # Azure AD (modern)
    crawler = SharePointCrawler(
        site_url="https://contoso.sharepoint.com/sites/mysite",
        auth_mode="azure_ad",
        tenant_id="...",
        client_id="...",
        client_secret="...",
    )
    loader = DocumentLoader(extra_transports=[crawler])
    chunks = loader.load_crawl(
        "https://contoso.sharepoint.com/sites/mysite",
        crawler=crawler,
    )

    # Legacy Add-in auth
    crawler = SharePointCrawler(
        site_url="https://contoso.sharepoint.com/sites/mysite",
        auth_mode="legacy",
        tenant_id="contoso.onmicrosoft.com",
        client_id="...",
        client_secret="...",
    )

    # On-premises NTLM
    crawler = SharePointCrawler(
        site_url="https://sharepoint.corp.example.com/sites/mysite",
        auth_mode="ntlm",
        username="DOMAIN\\\\user",
        password="...",
    )
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from ._protocol import FetchOptions, FetchResult

if TYPE_CHECKING:
    pass  # keeps TYPE_CHECKING used

_log = logging.getLogger(__name__)

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# SharePoint list base templates
_TEMPLATE_DOCUMENT_LIBRARY = 101
_TEMPLATE_CALENDAR = 106
_TEMPLATE_SITE_PAGES = 119
_TEMPLATE_GENERIC_LIST = 100

_LIST_TEMPLATES = {
    _TEMPLATE_DOCUMENT_LIBRARY,
    _TEMPLATE_CALENDAR,
    _TEMPLATE_SITE_PAGES,
    _TEMPLATE_GENERIC_LIST,
}


def _item_to_text(fields: dict[str, object], list_title: str, item_type: str = "List item") -> str:
    """Serialize a SharePoint list item's fields to plain text."""
    skip = {"@odata.etag", "odata.etag", "odata.id", "odata.type", "odata.editLink"}
    lines = [f"{item_type}: {list_title}"]
    for key, value in fields.items():
        if key in skip or key.startswith("@odata") or key.startswith("_"):
            continue
        if value is None:
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _event_to_text(fields: dict[str, Any], list_title: str) -> str:
    """Serialize a SharePoint calendar event to plain text."""
    lines = [f"Calendar: {list_title}"]
    priority = [
        "Title",
        "EventDate",
        "EndDate",
        "Location",
        "Description",
        "fAllDayEvent",
        "fRecurrence",
        "RecurrenceData",
        "Category",
    ]
    seen: set[str] = set()
    for key in priority:
        if key in fields and fields[key] is not None:
            lines.append(f"{key}: {fields[key]}")
            seen.add(key)
    for key, value in fields.items():
        if key in seen or key.startswith("@") or key.startswith("_") or value is None:
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


class SharePointCrawler:
    """Crawl a SharePoint site via REST API or Microsoft Graph.

    Implements both ``Crawler`` and ``Transport`` — pass the same instance to both::

        crawler = SharePointCrawler(site_url="https://...", auth_mode="azure_ad", ...)
        loader = DocumentLoader(extra_transports=[crawler])
        chunks = loader.load_crawl("https://...", crawler=crawler)

    Args:
        site_url:      Full URL of the SharePoint site to crawl.
        auth_mode:     ``"azure_ad"``, ``"legacy"``, or ``"ntlm"``.
        tenant_id:     Azure AD tenant ID or domain (azure_ad and legacy modes).
        client_id:     App client ID (azure_ad and legacy modes).
        client_secret: App client secret (azure_ad and legacy modes).
        username:      Windows username including domain, e.g. ``DOMAIN\\\\user`` (ntlm).
        password:      Password (ntlm).
        artifacts:     List of artifact types to crawl. Any of ``"documents"``,
                       ``"lists"``, ``"calendars"``, ``"pages"``. Defaults to all.
        max_items:     Maximum list items to fetch per list (default 5000).
    """

    def __init__(
        self,
        site_url: str,
        auth_mode: str = "azure_ad",
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        username: str | None = None,
        password: str | None = None,
        artifacts: list[str] | None = None,
        max_items: int = 5000,
    ) -> None:
        self._site_url = site_url.rstrip("/")
        self._auth_mode = auth_mode
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._username = username
        self._password = password
        self._artifacts = set(artifacts or ["documents", "lists", "calendars", "pages"])
        self._max_items = max_items
        self._url_key = hashlib.md5(site_url.encode(), usedforsecurity=False).hexdigest()[:8]

        # Pre-fetched structured content (list items, events, pages)
        self._cache: dict[str, FetchResult] = {}
        # Lazy document downloads: uri → (download_url, mime_type)
        self._pending: dict[str, tuple[str, str | None]] = {}
        # Shared requests session (populated during crawl)
        self._session: Any = None

    # ── Transport Protocol ────────────────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith(f"spitem://{self._url_key}/") or uri == self._site_url

    def fetch(self, uri: str, options: FetchOptions | None = None) -> FetchResult:
        if uri in self._cache:
            return self._cache[uri]
        if uri in self._pending:
            download_url, mime = self._pending[uri]
            resp = self._session.get(download_url, timeout=60)
            resp.raise_for_status()
            result = FetchResult(
                data=resp.content,
                detected_mime=mime or resp.headers.get("content-type"),
                source_path=uri,
            )
            self._cache[uri] = result
            del self._pending[uri]
            return result
        raise KeyError(f"SharePointCrawler: unknown URI {uri!r} — call crawl() first")

    # ── Crawler Protocol ──────────────────────────────────────────────────────

    def crawl(self, _uri: str = "", **__: object) -> list[str]:
        """Connect to SharePoint and index all configured artifact types.

        Returns:
            List of ``spitem://`` URIs, one per artifact.
        """
        try:
            import requests as _requests
        except ImportError:
            raise ImportError(
                "pip install chonk-rag[http]  # requests required for SharePointCrawler"
            ) from None

        self._cache.clear()
        self._pending.clear()

        if self._auth_mode == "azure_ad":
            self._session = self._make_graph_session(_requests)
            self._crawl_graph()
        elif self._auth_mode == "legacy":
            self._session = self._make_legacy_session(_requests)
            self._crawl_rest()
        elif self._auth_mode == "ntlm":
            self._session = self._make_ntlm_session(_requests)
            self._crawl_rest()
        else:
            raise ValueError(f"SharePointCrawler: unknown auth_mode {self._auth_mode!r}")

        total = len(self._cache) + len(self._pending)
        _log.info("SharePointCrawler: %d artifact(s) indexed from %s", total, self._site_url)
        return list(self._cache.keys()) + list(self._pending.keys())

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _make_graph_session(self, requests_mod: Any) -> Any:  # noqa: ANN401
        try:
            import msal  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "pip install msal  # required for SharePointCrawler auth_mode='azure_ad'"
            ) from None
        app = msal.ConfidentialClientApplication(
            self._client_id,
            authority=f"https://login.microsoftonline.com/{self._tenant_id}",
            client_credential=self._client_secret,
        )
        result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        if result is None:
            raise RuntimeError("SharePointCrawler: MSAL acquire_token_for_client returned None")
        if "access_token" not in result:
            raise RuntimeError(
                f"SharePointCrawler: MSAL auth failed: {result.get('error_description')}"
            )
        session = requests_mod.Session()
        session.headers["Authorization"] = f"Bearer {result['access_token']}"
        session.headers["Accept"] = "application/json"
        return session

    def _make_legacy_session(self, requests_mod: Any) -> Any:  # noqa: ANN401
        """SharePoint Add-in OAuth via Azure ACS token endpoint."""
        realm = self._get_realm(requests_mod)
        resource = f"00000003-0000-0ff1-ce00-000000000000/{self._site_url.split('/')[2]}@{realm}"
        token_url = f"https://accounts.accesscontrol.windows.net/{realm}/tokens/OAuth/2"
        resp = requests_mod.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": f"{self._client_id}@{realm}",
                "client_secret": self._client_secret,
                "resource": resource,
            },
            timeout=30,
        )
        resp.raise_for_status()
        token = resp.json()["access_token"]
        session = requests_mod.Session()
        session.headers["Authorization"] = f"Bearer {token}"
        session.headers["Accept"] = "application/json;odata=verbose"
        return session

    def _get_realm(self, requests_mod: Any) -> str:  # noqa: ANN401
        """Fetch tenant realm GUID from SharePoint WWW-Authenticate header."""
        resp = requests_mod.get(
            f"{self._site_url}/_vti_bin/client.svc",
            headers={"Authorization": "Bearer"},
            timeout=15,
        )
        www_auth = resp.headers.get("WWW-Authenticate", "")
        for part in www_auth.split(","):
            part = part.strip()
            if part.startswith("realm="):
                return part.split("=", 1)[1].strip('"')
        # Fall back to tenant_id if realm auto-discovery fails
        return self._tenant_id or ""

    def _make_ntlm_session(self, requests_mod: Any) -> Any:  # noqa: ANN401
        try:
            from requests_ntlm import HttpNtlmAuth  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "pip install requests-ntlm  # required for SharePointCrawler auth_mode='ntlm'"
            ) from None
        session = requests_mod.Session()
        session.auth = HttpNtlmAuth(self._username, self._password)
        session.headers["Accept"] = "application/json;odata=verbose"
        return session

    # ── Graph API crawl (azure_ad) ────────────────────────────────────────────

    def _crawl_graph(self) -> None:
        site_id = self._get_graph_site_id()
        if "documents" in self._artifacts:
            self._graph_document_libraries(site_id)
        lists = self._graph_get_lists(site_id)
        for lst in lists:
            tmpl = lst.get("list", {}).get("template", "")
            title = lst.get("displayName", "")
            list_id = lst["id"]
            if tmpl == "genericList" and "lists" in self._artifacts:
                self._graph_list_items(site_id, list_id, title, "List item")
            elif tmpl == "events" and "calendars" in self._artifacts:
                self._graph_list_items(site_id, list_id, title, "Calendar event")
            elif tmpl == "sitepages" and "pages" in self._artifacts:
                self._graph_site_pages(site_id, list_id, title)

    def _get_graph_site_id(self) -> str:
        parsed = self._site_url.replace("https://", "").replace("http://", "")
        hostname = parsed.split("/")[0]
        path = "/" + "/".join(parsed.split("/")[1:]) if "/" in parsed else ""
        resp = self._session.get(
            f"{_GRAPH_BASE}/sites/{hostname}:{path}",
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def _graph_get_lists(self, site_id: str) -> list[dict[str, Any]]:
        resp = self._session.get(
            f"{_GRAPH_BASE}/sites/{site_id}/lists",
            params={"$top": 500},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("value", [])

    def _graph_document_libraries(self, site_id: str) -> None:
        resp = self._session.get(
            f"{_GRAPH_BASE}/sites/{site_id}/drives",
            timeout=30,
        )
        resp.raise_for_status()
        for drive in resp.json().get("value", []):
            self._graph_walk_drive(drive["id"], f"{_GRAPH_BASE}/drives/{drive['id']}/root/children")

    def _graph_walk_drive(self, drive_id: str, url: str) -> None:
        while url:
            resp = self._session.get(url, params={"$top": 200}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("value", []):
                if "folder" in item:
                    child_url = f"{_GRAPH_BASE}/drives/{drive_id}/items/{item['id']}/children"
                    self._graph_walk_drive(drive_id, child_url)
                elif "file" in item:
                    uri = f"spitem://{self._url_key}/documents/{item['id']}"
                    download_url = (
                        item.get("@microsoft.graph.downloadUrl")
                        or f"{_GRAPH_BASE}/drives/{drive_id}/items/{item['id']}/content"
                    )
                    mime = item.get("file", {}).get("mimeType")
                    self._pending[uri] = (download_url, mime)
            url = data.get("@odata.nextLink")

    def _graph_list_items(self, site_id: str, list_id: str, title: str, item_type: str) -> None:
        url = f"{_GRAPH_BASE}/sites/{site_id}/lists/{list_id}/items"
        count = 0
        while url and count < self._max_items:
            resp = self._session.get(url, params={"$expand": "fields", "$top": 200}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("value", []):
                fields = item.get("fields", {})
                text = (
                    _event_to_text(fields, title)
                    if item_type == "Calendar event"
                    else _item_to_text(fields, title, item_type)
                )
                uri = f"spitem://{self._url_key}/lists/{list_id}/{item['id']}"
                self._cache[uri] = FetchResult(
                    data=text.encode("utf-8"),
                    detected_mime="text/plain",
                    source_path=f"{title} / {fields.get('Title', item['id'])}",
                )
                count += 1
            url = data.get("@odata.nextLink")

    def _graph_site_pages(self, site_id: str, list_id: str, title: str) -> None:
        url = f"{_GRAPH_BASE}/sites/{site_id}/pages"
        while url:
            resp = self._session.get(url, params={"$top": 100}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for page in data.get("value", []):
                page_id = page["id"]
                html = page.get("webHtml") or ""
                if not html:
                    # Fetch full page content
                    detail = self._session.get(
                        f"{_GRAPH_BASE}/sites/{site_id}/pages/{page_id}/microsoft.graph.sitePage/webParts",
                        timeout=15,
                    )
                    if detail.ok:
                        parts = detail.json().get("value", [])
                        html = "\n".join(
                            p.get("innerHtml", "")
                            for p in parts
                            if p.get("@odata.type") == "#microsoft.graph.textWebPart"
                        )
                uri = f"spitem://{self._url_key}/pages/{page_id}"
                self._cache[uri] = FetchResult(
                    data=html.encode("utf-8"),
                    detected_mime="text/html",
                    source_path=f"{title} / {page.get('title', page_id)}",
                )
            url = data.get("@odata.nextLink")

    # ── REST API crawl (legacy / ntlm) ────────────────────────────────────────

    def _crawl_rest(self) -> None:
        lists = self._rest_get_lists()
        for lst in lists:
            tmpl = lst.get("BaseTemplate", 0)
            title = lst.get("Title", "")
            list_id = lst.get("Id", "")
            if tmpl == _TEMPLATE_DOCUMENT_LIBRARY and "documents" in self._artifacts:
                self._rest_document_library(list_id, title)
            elif tmpl == _TEMPLATE_CALENDAR and "calendars" in self._artifacts:
                self._rest_list_items(list_id, title, "Calendar event")
            elif tmpl == _TEMPLATE_SITE_PAGES and "pages" in self._artifacts:
                self._rest_list_items(list_id, title, "Page")
            elif tmpl == _TEMPLATE_GENERIC_LIST and "lists" in self._artifacts:
                self._rest_list_items(list_id, title, "List item")

    def _rest_get_lists(self) -> list[dict[str, Any]]:
        resp = self._session.get(
            f"{self._site_url}/_api/web/lists",
            params={
                "$filter": f"Hidden eq false and BaseTemplate in ({_TEMPLATE_DOCUMENT_LIBRARY},"
                f"{_TEMPLATE_CALENDAR},{_TEMPLATE_SITE_PAGES},{_TEMPLATE_GENERIC_LIST})",
                "$select": "Id,Title,BaseTemplate",
                "$top": 500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return self._rest_results(resp)

    def _rest_document_library(self, list_id: str, title: str) -> None:
        url: str | None = (
            f"{self._site_url}/_api/web/lists(guid'{list_id}')/items"
            "?$select=Id,FileRef,FileLeafRef,File_x0020_Type"
            "&$filter=FSObjType eq 0&$top=200"
        )
        count = 0
        while url and count < self._max_items:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = self._rest_results_raw(data)
            for item in results:
                file_ref = item.get("FileRef", "")
                item_id = item.get("Id", "")
                if not file_ref:
                    continue
                download_url = (
                    f"{self._site_url}/_api/web/GetFileByServerRelativeUrl('{file_ref}')/$value"
                )
                uri = f"spitem://{self._url_key}/documents/{list_id}/{item_id}"
                self._pending[uri] = (download_url, None)
                count += 1
            url = self._rest_next(data)

    def _rest_list_items(self, list_id: str, title: str, item_type: str) -> None:
        url: str | None = f"{self._site_url}/_api/web/lists(guid'{list_id}')/items?$top=200"
        count = 0
        while url and count < self._max_items:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for item in self._rest_results_raw(data):
                item_id = item.get("Id", "")
                text = (
                    _event_to_text(item, title)
                    if item_type == "Calendar event"
                    else _item_to_text(item, title, item_type)
                )
                uri = f"spitem://{self._url_key}/lists/{list_id}/{item_id}"
                self._cache[uri] = FetchResult(
                    data=text.encode("utf-8"),
                    detected_mime="text/plain",
                    source_path=f"{title} / {item.get('Title', item_id)}",
                )
                count += 1
            url = self._rest_next(data)

    # ── REST response helpers ─────────────────────────────────────────────────

    @staticmethod
    def _rest_results(resp: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        data = resp.json()
        return SharePointCrawler._rest_results_raw(data)

    @staticmethod
    def _rest_results_raw(data: dict[str, Any]) -> list[dict[str, Any]]:
        # verbose OData: d.results; minimal OData: value
        d = data.get("d", data)
        return d.get("results", d.get("value", []))

    @staticmethod
    def _rest_next(data: dict[str, Any]) -> str | None:
        # verbose OData pagination uses __next
        d = data.get("d", data)
        return d.get("__next") or data.get("@odata.nextLink")
