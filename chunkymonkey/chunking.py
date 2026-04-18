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

# ── Plain-text header promotion ───────────────────────────────────────────────

# Default structural-level patterns for prose corpora (novels, non-fiction).
# Each tuple: (regex_pattern, heading_level) where level 1→# and level 2→##.
# Pass as ``structural_levels`` to override completely.
NOVEL_STRUCTURAL_LEVELS: list[tuple[str, int]] = [
    # PART/BOOK/SCENE only when ALL-CAPS (no re.IGNORECASE) — avoids false
    # positives from common words "part", "book", "scene" in running prose.
    (r"(?:PART|BOOK|SCENE)\s+(?:[IVXLCDM]+|THE\s+[A-Z]+|\d+)", 1),
    # CHAPTER is unambiguous regardless of case.
    (r"(?i:CHAPTER)\s+(?:[IVXLCDM]+|\d+)", 2),
]

_VERB_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|can|will|may|should|would|could|"
    r"do|does|did|been|be|being|occur|form|cause|help|provide|include|"
    r"involve|develop|affect|require|contain|consist|allow|make|take|"
    r"give|show|use|need|want|seem|appear|become|remain|stay|go|come)\b",
    re.IGNORECASE,
)

# Sentence-boundary: after . ! ? followed by whitespace
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+")

# Short phrase: 1–N words with no ending punctuation, followed by a new capital
_SHORT_PHRASE_RE = re.compile(
    r"(?<=[.!?] )"           # after a sentence end + space
    r"([A-Z][^\n.!?]{2,}?)" # candidate: starts capital, no sentence-end punct
    r"(?= [A-Z][a-z])",     # immediately followed by capital word (new sentence)
    re.MULTILINE,
)


def _extract_chapter_title(text: str, pos: int) -> tuple[str, int]:
    """Extract ALL-CAPS title words from *text* starting at *pos*.

    Skips separator chars (periods, dashes, spaces) then consumes words whose
    alphabetic characters are all uppercase.  Stops at the first prose word
    (has a lowercase letter) or a bare digit (page-number in a TOC entry).

    Returns ``(title_string, end_pos)`` where *end_pos* is the start of prose.
    """
    while pos < len(text) and text[pos] in ". -\t\u2013\u2014":
        pos += 1
    words: list[str] = []
    i = pos
    while i < len(text):
        while i < len(text) and text[i] == " ":
            i += 1
        j = i
        while j < len(text) and text[j] not in " \t\n":
            j += 1
        word = text[i:j]
        if not word:
            break
        alpha = [c for c in word if c.isalpha()]
        if not alpha:
            if word.isdigit():
                break  # page number → TOC entry, stop
            i = j
            continue
        if all(c.isupper() for c in alpha):
            words.append(word.rstrip(".,;:"))
            i = j
        else:
            break
    return " ".join(words).strip().rstrip(".,;:\u2013\u2014-"), i


def promote_plain_text_headers(
    text: str,
    promote_questions: bool = True,
    promote_short_phrases: bool = True,
    max_header_words: int = 6,
    max_header_chars: int = 80,
    structural_levels: list[tuple[str, int]] | None = None,
    toc_proximity: int = 300,
) -> str:
    """Insert markdown headers into flat plain text by detecting header-like patterns.

    Three heuristics (each independently togglable):

    * **Structural levels** (``structural_levels``): A list of
      ``(regex_pattern, heading_level)`` pairs applied in a single pass.
      Level 1 emits ``#``, level 2 emits ``##``, etc.  Each regex should match
      the structural marker (e.g. ``CHAPTER I``); the function extracts any
      following ALL-CAPS title words automatically.  TOC clusters (same-level
      markers within *toc_proximity* chars of each other) are skipped.
      Pass ``NOVEL_STRUCTURAL_LEVELS`` for the built-in PART/CHAPTER defaults,
      or supply your own list.  ``None`` (default) disables this heuristic.

    * **Questions** (``promote_questions``): A standalone question sentence
      (ends with ``?``) that is *not* followed immediately by another question
      is promoted to a ``##`` heading.  Common in patient-education guides and
      FAQ-style documents where section titles are phrased as questions.

    * **Short phrases** (``promote_short_phrases``): A capitalised phrase of
      ≤ *max_header_words* words and ≤ *max_header_chars* characters that
      (a) contains no finite verb, (b) carries no sentence-ending punctuation,
      and (c) appears between two proper sentences is promoted to ``##``.
      Targets fused PDF headers such as "Signs and symptoms Basal cell …".

    Args:
        text:                Raw plain text (may be a single long string).
        promote_questions:   Promote standalone question sentences.
        promote_short_phrases: Promote short verbless phrases as headers.
        max_header_words:    Phrase word-count ceiling (short-phrase heuristic).
        max_header_chars:    Phrase char-length ceiling (short-phrase heuristic).
        structural_levels:   List of ``(pattern, level)`` pairs for structural
                             promotion, or ``None`` to disable.  Use the exported
                             ``NOVEL_STRUCTURAL_LEVELS`` constant as a starting
                             point.
        toc_proximity:       Max chars between same-level markers before they are
                             treated as a TOC cluster and skipped (default 300).

    Returns:
        Text with qualifying patterns replaced by heading blocks.
    """
    # Normalise whitespace but preserve paragraph breaks if present
    has_paras = "\n\n" in text
    text = re.sub(r"[ \t]+", " ", text).strip()

    if structural_levels:
        # Collect all matches across every (pattern, level) pair.
        all_hits: list[tuple[int, int, str, int]] = []  # (pos, end, marker, level)
        for pattern_str, level in structural_levels:
            for m in re.compile(pattern_str).finditer(text):
                all_hits.append((m.start(), m.end(), m.group(0), level))
        all_hits.sort(key=lambda x: x[0])

        # TOC detection: same-level markers within toc_proximity chars → skip.
        toc_idx: set[int] = set()
        by_level: dict[int, list[int]] = {}
        for idx, (_, _, _, lvl) in enumerate(all_hits):
            by_level.setdefault(lvl, []).append(idx)
        for level_indices in by_level.values():
            for k in range(len(level_indices) - 1):
                ia, ib = level_indices[k], level_indices[k + 1]
                if all_hits[ib][0] - all_hits[ia][0] < toc_proximity:
                    toc_idx.add(ia)
                    toc_idx.add(ib)

        # Replace right-to-left so earlier positions stay valid.
        for i in range(len(all_hits) - 1, -1, -1):
            if i in toc_idx:
                continue
            pos, end, marker, level = all_hits[i]
            hashes = "#" * level
            title, prose_start = _extract_chapter_title(text, end)
            heading = f"\n\n{hashes} {marker}"
            if title:
                heading += f": {title}"
            heading += "\n\n"
            text = text[:pos] + heading + text[prose_start:]

    if promote_questions:
        sentences = _SENT_END_RE.split(text)
        out: list[str] = []
        i = 0
        while i < len(sentences):
            s = sentences[i].strip()
            next_s = sentences[i + 1].strip() if i + 1 < len(sentences) else ""
            if (
                s.endswith("?")
                and not next_s.endswith("?")           # not a run of questions
                and len(s.split()) <= max_header_words * 2  # not a paragraph
                and len(s) <= max_header_chars * 2
            ):
                out.append(f"\n\n## {s}\n\n")
            else:
                out.append((" " if out and not out[-1].endswith("\n") else "") + s)
            i += 1
        text = "".join(out).strip()

    if promote_short_phrases:
        def _replace_phrase(m: re.Match) -> str:
            phrase = m.group(1).strip().rstrip(".,;:")
            words = phrase.split()
            if (
                len(words) < 2
                or len(words) > max_header_words
                or len(phrase) > max_header_chars
                or _VERB_RE.search(phrase)
                or not phrase[0].isupper()
            ):
                return m.group(0)  # leave unchanged
            # Reconstruct: sentence-end + newlines + ## header + newline + next sentence
            # The lookbehind consumed the sentence-end punct+space; we must keep it
            return f"\n\n## {phrase}\n\n"

        # We need to rewrite using sub with the full match context
        # Pattern: after [.!?][space], capture candidate phrase, lookahead for new sentence
        FULL_RE = re.compile(
            r"([.!?] )"
            r"([A-Z][^\n.!?]{2,}?)"
            r"(?= [A-Z][a-z])",
            re.MULTILINE,
        )

        def _full_replace(m: re.Match) -> str:
            sent_end = m.group(1)
            phrase = m.group(2).strip().rstrip(".,;:")
            words = phrase.split()
            if (
                len(words) < 2
                or len(words) > max_header_words
                or len(phrase) > max_header_chars
                or _VERB_RE.search(phrase)
                or not phrase[0].isupper()
            ):
                return m.group(0)
            return f"{sent_end}\n\n## {phrase}\n\n"

        text = FULL_RE.sub(_full_replace, text)

    return text


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
    """Lowest common ancestor of a list of heading paths.

    When sibling paths share no common ancestor (LCA would be empty), falls
    back to the *last* non-empty path so every chunk retains at least the
    section it ends in rather than losing all context.
    """
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
    return result if result else list(non_empty[-1])


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
    promote_headings: bool = False,
    promote_questions: bool = True,
    promote_short_phrases: bool = True,
    max_header_words: int = 6,
    max_header_chars: int = 80,
    structural_levels: list[tuple[str, int]] | None = None,
    toc_proximity: int = 300,
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
        promote_headings: Run ``promote_plain_text_headers()`` on *content* before
            chunking.  Useful for flat plain-text corpora (e.g. PDF-extracted prose)
            where section titles are fused with body text.  Default False.
        promote_questions: Passed to ``promote_plain_text_headers``.
        promote_short_phrases: Passed to ``promote_plain_text_headers``.
        max_header_words: Passed to ``promote_plain_text_headers``.
        max_header_chars: Passed to ``promote_plain_text_headers``.
        structural_levels: Passed to ``promote_plain_text_headers``.  Use
            ``NOVEL_STRUCTURAL_LEVELS`` for PART/CHAPTER promotion.
        toc_proximity: Passed to ``promote_plain_text_headers``.
    """
    hard_max = int(max_chunk_size * (1.0 + overflow_margin))

    if promote_headings:
        content = promote_plain_text_headers(
            content,
            promote_questions=promote_questions,
            promote_short_phrases=promote_short_phrases,
            max_header_words=max_header_words,
            max_header_chars=max_header_chars,
            structural_levels=structural_levels,
            toc_proximity=toc_proximity,
        )

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
        crumb: str | None = None
        if include_breadcrumb and text:
            crumb = f"[{name} > {' > '.join(lca)}]" if lca else f"[{name}]"
        full_text = _build_content(text)
        src_len = len(full_text.encode("utf-8")) if full_text else 0
        return DocumentChunk(
            document_name=name,
            content=full_text,
            section=section_str,
            chunk_index=idx,
            source_offset=offset,
            source_length=src_len,
            breadcrumb=crumb,
            embedding_content=full_text if include_breadcrumb else None,
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
                # Clear the old-path seed so the new chunk's LCA starts
                # fresh from the incoming heading, not from the prior section.
                current_section_paths.clear()

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
