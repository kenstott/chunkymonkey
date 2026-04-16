# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""RFC 5322 email extractor.

Handles ``message/rfc822`` (short type ``"email"`` / ``"eml"``).

Output format::

    From: Alice <alice@example.com>
    To: Bob <bob@example.com>
    Date: Mon, 14 Apr 2025 10:00:00 +0000
    Subject: Q1 Review

    Body text here...

    [Attachment: report.pdf]
    <extracted text from attachment>

Attachments are extracted recursively using the existing extractor registry.
Pass ``include_attachments=False`` (the default) to skip them.
"""

from __future__ import annotations

import email
import email.policy
import logging
from email.message import Message

logger = logging.getLogger(__name__)

_HEADER_FIELDS = ("From", "To", "Cc", "Date", "Subject")


def _decode_header_value(value: str | None) -> str:
    """Decode a potentially RFC 2047-encoded header value to a plain string."""
    if not value:
        return ""
    from email.header import decode_header as _dh
    parts = []
    for raw, charset in _dh(value):
        if isinstance(raw, bytes):
            parts.append(raw.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(raw)
    return "".join(parts)


def _get_body(msg: Message) -> str:
    """Return the best available plain-text body."""
    if msg.is_multipart():
        # Prefer text/plain; fall back to text/html
        for preferred_type in ("text/plain", "text/html"):
            for part in msg.walk():
                if part.get_content_type() != preferred_type:
                    continue
                if "attachment" in part.get("Content-Disposition", ""):
                    continue
                payload = part.get_payload(decode=True)
                if not payload:
                    continue
                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
                if preferred_type == "text/html":
                    from ._html import HtmlExtractor
                    return HtmlExtractor().extract(payload)
                return text
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                from ._html import HtmlExtractor
                return HtmlExtractor().extract(payload)
            return text
    return ""


def _extract_attachments(msg: Message) -> list[str]:
    """Return extracted text blocks from each attachment."""
    from ._mime import detect_type_from_source
    from . import detect_extractor

    blocks: list[str] = []
    for part in msg.walk():
        # Only process actual attachments (Content-Disposition: attachment OR
        # inline parts that have a filename but are not the body).
        disposition = part.get("Content-Disposition", "")
        filename = part.get_filename()
        if not filename and "attachment" not in disposition:
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        filename = filename or "attachment"
        content_type = part.get_content_type()
        doc_type = detect_type_from_source(filename, content_type)
        try:
            extractor = detect_extractor(doc_type)
            text = extractor.extract(payload, source_path=filename)
            if text.strip():
                blocks.append(f"[Attachment: {filename}]\n{text.strip()}")
        except Exception as exc:
            logger.debug("EmailExtractor: skipping attachment %s: %s", filename, exc)
    return blocks


class EmailExtractor:
    """Extract readable text from a raw RFC 5322 email message.

    Args:
        include_attachments: When True, recursively extract and append text
            from all attachments using the extractor registry (default False).
    """

    HANDLED = {"email", "eml"}

    def __init__(self, *, include_attachments: bool = False):
        self.include_attachments = include_attachments

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        msg = email.message_from_bytes(data, policy=email.policy.compat32)

        # Headers
        header_lines: list[str] = []
        for field in _HEADER_FIELDS:
            raw = msg.get(field)
            if raw:
                header_lines.append(f"{field}: {_decode_header_value(raw)}")

        # Body
        body = _get_body(msg).strip()

        # Attachments
        attachment_blocks: list[str] = []
        if self.include_attachments:
            attachment_blocks = _extract_attachments(msg)

        parts: list[str] = []
        if header_lines:
            parts.append("\n".join(header_lines))
        if body:
            parts.append(body)
        parts.extend(attachment_blocks)

        return "\n\n".join(parts)
