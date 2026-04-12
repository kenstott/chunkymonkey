"""MIME type normalization and detection for document loading."""

from __future__ import annotations

# Full MIME type -> short alias
MIME_TO_SHORT: dict[str, str] = {
    "application/pdf": "pdf",
    "text/html": "html",
    "text/markdown": "markdown",
    "text/plain": "text",
    "text/csv": "csv",
    "application/json": "json",
    "application/x-ndjson": "jsonl",
    "application/x-yaml": "yaml",
    "text/yaml": "yaml",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "image/png": "image",
    "image/jpeg": "image",
    "image/tiff": "image",
    "image/webp": "image",
    "image/bmp": "image",
    "image/gif": "image",
    "image/svg+xml": "image",
    "audio/mpeg": "audio",
    "audio/mp4": "audio",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/ogg": "audio",
    "audio/flac": "audio",
    "audio/webm": "audio",
    "video/mp4": "audio",
    "video/webm": "audio",
    "video/x-matroska": "audio",
}

# File extension -> short alias
EXTENSION_TO_SHORT: dict[str, str] = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".pptx": "pptx",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".tif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".gif": "image",
    ".svg": "image",
    ".mp3": "audio",
    ".mp4": "audio",
    ".m4a": "audio",
    ".wav": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    ".webm": "audio",
    ".mkv": "audio",
    ".aac": "audio",
}

# All known short aliases
_KNOWN_SHORT: set[str] = {
    "auto", "pdf", "html", "markdown", "text", "csv", "json", "jsonl",
    "yaml", "xml", "docx", "xlsx", "pptx", "image", "audio",
}

# Legacy transport types — treated as "auto" (transport is now inferred from fields)
_LEGACY_TRANSPORT_TYPES: set[str] = {"file", "http", "inline", "confluence", "notion"}

# Binary types that need special extraction (not decodable as UTF-8)
_BINARY_TYPES: set[str] = {"pdf", "docx", "xlsx", "pptx", "image", "audio"}


def normalize_type(user_type: str) -> str:
    """Normalize a user-provided type string to a short alias.

    Accepts: short aliases ("pdf"), full MIME ("application/pdf"), or "auto".
    Returns the short alias.
    Raises ValueError on unknown type.
    """
    if not user_type:
        return "auto"

    lower = user_type.strip().lower()

    if lower in _KNOWN_SHORT:
        return lower

    # Legacy transport types (type: file, http, inline) -> treat as auto
    if lower in _LEGACY_TRANSPORT_TYPES:
        return "auto"

    if lower in MIME_TO_SHORT:
        return MIME_TO_SHORT[lower]

    raise ValueError(
        f"Unknown document type: {user_type!r}. "
        f"Use one of: {sorted(_KNOWN_SHORT - {'auto'})} or a MIME type."
    )


def detect_type_from_source(
    source_path: str | None,
    detected_mime: str | None,
) -> str:
    """Auto-detect document type from transport metadata and file extension.

    Priority: detected_mime (from HTTP Content-Type / S3 metadata) > file extension > "text" fallback.
    """
    # Try MIME from transport first
    if detected_mime:
        lower_mime = detected_mime.lower().split(";")[0].strip()
        if lower_mime in MIME_TO_SHORT:
            return MIME_TO_SHORT[lower_mime]
        # Partial match for common patterns
        if "pdf" in lower_mime:
            return "pdf"
        if "html" in lower_mime:
            return "html"
        if "markdown" in lower_mime:
            return "markdown"
        if "wordprocessingml" in lower_mime:
            return "docx"
        if "spreadsheetml" in lower_mime:
            return "xlsx"
        if "presentationml" in lower_mime:
            return "pptx"

    # Try file extension
    if source_path:
        import os
        ext = os.path.splitext(source_path)[1].lower()
        if ext in EXTENSION_TO_SHORT:
            return EXTENSION_TO_SHORT[ext]

    return "text"


def is_binary_type(doc_type: str) -> bool:
    """Return True for types that are binary (PDF, Office docs)."""
    return doc_type.lower() in _BINARY_TYPES


# MIME prefixes we can actually load (text/*, plus specific binary document types)
_LOADABLE_MIME_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/pdf",
    "application/x-yaml",
    "application/x-ndjson",
    "application/vnd.openxmlformats",
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/octet-stream",
    "image/",
    "audio/",
    "video/",
)


def is_loadable_mime(mime: str | None) -> bool:
    """Return True if we can extract text from this MIME type.

    Returns True for None (unknown MIME — let downstream handle it).
    Returns False for images, audio, video, fonts, etc.
    """
    if mime is None:
        return True
    lower = mime.lower().split(";")[0].strip()
    return any(lower.startswith(prefix) for prefix in _LOADABLE_MIME_PREFIXES)
