# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: eb796e37-fb9a-4f42-9af2-18cda35e6338
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Pure chunking functions — table/list detection, block merging, section extraction."""

from __future__ import annotations

import re
from .models import DocumentChunk


def is_table_line(line: str) -> bool:
    """Detect pipe-separated table rows (DOCX/XLSX/PPTX/HTML/MD formats)."""
    return line.count("|") >= 2


def is_list_line(line: str) -> bool:
    """Detect markdown list items."""
    stripped = line.lstrip()
    if stripped[:2] in ("- ", "* ", "+ "):
        return True
    if stripped and stripped[0].isdigit():
        dot_pos = stripped.find(". ")
        if 0 < dot_pos <= 4:
            return stripped[:dot_pos].isdigit()
    return False


def merge_blocks(paragraphs: list[str], separator: str) -> list[str]:
    """Merge consecutive table lines and list lines into atomic blocks."""
    merged: list[str] = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        lines = para.split("\n") if "\n" in para else [para]

        if any(is_table_line(l) for l in lines if l.strip()):
            block_parts = [para]
            j = i + 1
            while j < len(paragraphs):
                next_lines = paragraphs[j].split("\n") if "\n" in paragraphs[j] else [paragraphs[j]]
                if any(is_table_line(l) for l in next_lines if l.strip()):
                    block_parts.append(paragraphs[j])
                    j += 1
                else:
                    break
            merged.append(separator.join(block_parts))
            i = j
            continue

        if any(is_list_line(l) for l in lines if l.strip()):
            block_parts = [para]
            j = i + 1
            while j < len(paragraphs):
                next_lines = paragraphs[j].split("\n") if "\n" in paragraphs[j] else [paragraphs[j]]
                if any(is_list_line(l) for l in next_lines if l.strip()):
                    block_parts.append(paragraphs[j])
                    j += 1
                else:
                    break
            merged.append(separator.join(block_parts))
            i = j
            continue

        merged.append(para)
        i += 1
    return merged


def extract_markdown_sections(content: str, doc_format: str) -> list[str]:
    """Extract section headers from markdown content."""
    if doc_format not in ("markdown", "md"):
        return []
    return [line.lstrip("#").strip() for line in content.split("\n") if line.startswith("#")]


# ─────────────────────────────────────────────────────────────────────────────
# Split helpers
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_END_RE = re.compile(r"(?<=[.?!])\s+")


def _split_at_sentences(text: str, max_size: int) -> list[str]:
    """Split prose at sentence boundaries so each piece ≤ max_size.

    A single sentence that exceeds max_size is kept whole rather than
    split mid-sentence.
    """
    if len(text) <= max_size:
        return [text]
    raw_parts = _SENTENCE_END_RE.split(text)
    pieces: list[str] = []
    current = ""
    for part in raw_parts:
        candidate = (current + " " + part).strip() if current else part
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = part
    if current:
        pieces.append(current)
    return pieces or [text]


def _split_at_list_items(text: str, max_size: int) -> list[str]:
    """Split a list block at item boundaries so each piece ≤ max_size."""
    if len(text) <= max_size:
        return [text]
    lines = text.split("\n")
    pieces: list[str] = []
    current_lines: list[str] = []
    for line in lines:
        candidate = "\n".join(current_lines + [line])
        if len(candidate) <= max_size:
            current_lines.append(line)
        else:
            if current_lines:
                pieces.append("\n".join(current_lines))
            current_lines = [line]
    if current_lines:
        pieces.append("\n".join(current_lines))
    return pieces or [text]


def _split_at_table_rows(text: str, max_size: int) -> list[str]:
    """Split a table block at row boundaries so each piece ≤ max_size."""
    if len(text) <= max_size:
        return [text]
    lines = text.split("\n")
    pieces: list[str] = []
    current_lines: list[str] = []
    for line in lines:
        candidate = "\n".join(current_lines + [line])
        if len(candidate) <= max_size:
            current_lines.append(line)
        else:
            if current_lines:
                pieces.append("\n".join(current_lines))
            current_lines = [line]
    if current_lines:
        pieces.append("\n".join(current_lines))
    return pieces or [text]


# ─────────────────────────────────────────────────────────────────────────────
# LCA breadcrumb
# ─────────────────────────────────────────────────────────────────────────────

def _lca_path(paths: list[list[str]]) -> list[str]:
    """Lowest common ancestor of a list of heading paths."""
    non_empty = [p for p in paths if p]
    if not non_empty:
        return []
    if len(non_empty) == 1:
        return list(non_empty[0])
    result: list[str] = []
    for parts in zip(*non_empty):
        if len(set(parts)) == 1:
            result.append(parts[0])
        else:
            break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# chunk_document
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(
    name: str,
    content: str,
    min_chunk_size: int,
    max_chunk_size: int,
    overflow_margin: float = 0.15,
    include_breadcrumb: bool = True,
) -> list[DocumentChunk]:
    """Split a document into chunks bounded by min_chunk_size and max_chunk_size.

    **Accumulation (Rule 1)**
    Paragraphs accumulate across section boundaries until the chunk reaches
    *min_chunk_size*.  Headings never force a break while below the floor.

    **Flush at section boundary (Rule 2)**
    Once a chunk is ≥ *min_chunk_size*, the next same-level-or-shallower heading
    triggers a flush.  The heading opens the new chunk.

    **Hard split at max (Rule 3)**
    If a single block would push the chunk past *max_chunk_size × (1 + overflow_margin)*
    it is split at the finest natural boundary for its type:
      - Table  → row boundary  → ``[TABLE:start/cont/end]``
      - List   → item boundary → ``[LIST:start/cont/end]``
      - Prose  → sentence end  → ``[PARA:start/cont/end]``

    **Breadcrumb (Rule 4)**
    Every chunk (including continuations) begins with the LCA breadcrumb::

        [doc_name > Ancestor > Section]

    **Sheet breaks (Rule 5)**
    ``[Sheet: ...]`` markers always flush the current chunk.

    **Last-chunk exception (Rule 6)**
    The final chunk may be smaller than *min_chunk_size*.

    Args:
        name: Document name for metadata and breadcrumb.
        content: Full document text.
        min_chunk_size: Accumulation floor (chars).
        max_chunk_size: Hard ceiling before splitting (chars).
        overflow_margin: Fractional slack above max before a split is forced
            (default 0.15 = 15%).
        include_breadcrumb: Prepend LCA breadcrumb to content (default True).
            Pass False for naive/baseline chunking.
    """
    hard_max = int(max_chunk_size * (1.0 + overflow_margin))

    chunks: list[DocumentChunk] = []
    heading_stack: list[tuple[int, str]] = []

    if "\n\n" in content:
        paragraphs = content.split("\n\n")
        separator = "\n\n"
    else:
        paragraphs = content.split("\n")
        separator = "\n"

    paragraphs = merge_blocks(paragraphs, separator)

    chunk_index = 0
    current_chunk = ""
    chunk_start_offset: int | None = None
    current_section_paths: list[list[str]] = []
    current_path: list[str] = []

    # ── inner helpers ─────────────────────────────────────────────────────────

    def _snapshot() -> None:
        if not current_section_paths or current_section_paths[-1] != current_path:
            current_section_paths.append(list(current_path))

    def _build_content(text: str) -> str:
        if not include_breadcrumb or not text:
            return text
        lca = _lca_path(current_section_paths)
        crumb = f"[{name} > {' > '.join(lca)}]" if lca else f"[{name}]"
        return f"{crumb}\n\n{text}"

    def _flush(text: str, idx: int, offset: int | None) -> DocumentChunk:
        lca = _lca_path(current_section_paths)
        section_str = " > ".join(lca) if lca else None
        full_text = _build_content(text)
        src_len = len(full_text.encode("utf-8")) if full_text else 0
        return DocumentChunk(
            document_name=name,
            content=full_text,
            section=section_str,
            chunk_index=idx,
            source_offset=offset,
            source_length=src_len,
        )

    def _reset() -> None:
        nonlocal current_chunk, chunk_start_offset
        current_chunk = ""
        chunk_start_offset = None
        current_section_paths.clear()
        _snapshot()

    def _emit_splits(
        pieces: list[str],
        marker_start: str,
        marker_cont: str,
        marker_end: str,
        base_offset: int | None,
    ) -> None:
        nonlocal chunk_index
        n = len(pieces)
        for i, piece in enumerate(pieces):
            if n > 1:
                if i == 0:
                    marked = f"{marker_start}\n{piece}\n{marker_cont}"
                elif i < n - 1:
                    marked = f"{marker_cont}\n{piece}\n{marker_cont}"
                else:
                    marked = f"{marker_cont}\n{piece}\n{marker_end}"
            else:
                marked = piece
            chunks.append(_flush(marked, chunk_index, base_offset))
            chunk_index += 1
            _reset()

    # seed tracking with empty initial path
    _snapshot()

    # ── main loop ─────────────────────────────────────────────────────────────

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_byte_offset = content.find(para)
        if para_byte_offset >= 0:
            para_byte_offset = len(content[:para_byte_offset].encode("utf-8"))
        else:
            para_byte_offset = None

        is_heading = para.startswith("#")
        is_sheet = para.startswith("[Sheet:") and para.endswith("]")

        # ── Rule 2: flush at section boundary once past min ──────────────────
        if is_heading:
            level = len(para) - len(para.lstrip("#"))
            heading_text = para.lstrip("#").strip()

            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(_flush(current_chunk, chunk_index, chunk_start_offset))
                chunk_index += 1
                _reset()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_path = [t for _, t in heading_stack]
            _snapshot()

        # ── Rule 5: sheet markers always break ───────────────────────────────
        elif is_sheet:
            if current_chunk:
                chunks.append(_flush(current_chunk, chunk_index, chunk_start_offset))
                chunk_index += 1
                _reset()
            sheet_name = para[len("[Sheet:"):-1].strip()
            heading_stack = [(0, sheet_name)]
            current_path = [sheet_name]
            _snapshot()

        # ── Try to accumulate ─────────────────────────────────────────────────
        potential = (current_chunk + separator + para).strip() if current_chunk else para

        if len(potential) <= hard_max:
            # Fits within hard ceiling — accumulate
            if not current_chunk:
                chunk_start_offset = para_byte_offset
            current_chunk = potential

        else:
            # ── Rule 3: hard split ────────────────────────────────────────────
            if current_chunk:
                chunks.append(_flush(current_chunk, chunk_index, chunk_start_offset))
                chunk_index += 1
                _reset()

            lines = para.split("\n")
            is_table = any(is_table_line(l) for l in lines if l.strip())
            is_list  = any(is_list_line(l)  for l in lines if l.strip())

            if is_table:
                pieces = _split_at_table_rows(para, max_chunk_size)
                _emit_splits(pieces, "[TABLE:start]", "[TABLE:cont]", "[TABLE:end]", para_byte_offset)
            elif is_list:
                pieces = _split_at_list_items(para, max_chunk_size)
                _emit_splits(pieces, "[LIST:start]", "[LIST:cont]", "[LIST:end]", para_byte_offset)
            else:
                pieces = _split_at_sentences(para, max_chunk_size)
                _emit_splits(pieces, "[PARA:start]", "[PARA:cont]", "[PARA:end]", para_byte_offset)

    if current_chunk:
        chunks.append(_flush(current_chunk, chunk_index, chunk_start_offset))

    return chunks
