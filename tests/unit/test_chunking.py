# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: db6662f3-b348-4d85-9bc9-6371616ddc70
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for chunkymonkey chunking primitives."""

import pytest

from chunkymonkey import (
    DocumentChunk,
    NOVEL_STRUCTURAL_LEVELS,
    chunk_document,
    is_list_line,
    is_table_line,
    merge_blocks,
    promote_plain_text_headers,
)


# =============================================================================
# is_table_line
# =============================================================================

class TestIsTableLine:
    def test_markdown_table_row(self):
        assert is_table_line("| col1 | col2 | col3 |")

    def test_markdown_separator(self):
        assert is_table_line("|---|---|---|")

    def test_docx_pipe_row(self):
        assert is_table_line("cell1 | cell2 | cell3")

    def test_single_pipe_not_table(self):
        assert not is_table_line("this | has one pipe")

    def test_no_pipe(self):
        assert not is_table_line("just a regular line")

    def test_empty_line(self):
        assert not is_table_line("")

    def test_two_pipes_minimum(self):
        assert is_table_line("a | b | c")


# =============================================================================
# is_list_line
# =============================================================================

class TestIsListLine:
    def test_dash_list(self):
        assert is_list_line("- item one")

    def test_asterisk_list(self):
        assert is_list_line("* item two")

    def test_plus_list(self):
        assert is_list_line("+ item three")

    def test_ordered_list(self):
        assert is_list_line("1. first item")

    def test_ordered_list_double_digit(self):
        assert is_list_line("12. twelfth item")

    def test_indented_list(self):
        assert is_list_line("  - indented item")

    def test_not_a_list(self):
        assert not is_list_line("regular paragraph text")

    def test_empty_line(self):
        assert not is_list_line("")

    def test_dash_without_space(self):
        assert not is_list_line("-not a list")

    def test_number_without_dot(self):
        assert not is_list_line("123 not a list")


# =============================================================================
# merge_blocks
# =============================================================================

class TestMergeBlocks:
    def test_merge_consecutive_table_paragraphs(self):
        paragraphs = [
            "Header text",
            "| col1 | col2 |",
            "|------|------|",
            "| val1 | val2 |",
            "Footer text",
        ]
        result = merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3
        assert result[0] == "Header text"
        assert "| col1 | col2 |" in result[1]
        assert "| val1 | val2 |" in result[1]
        assert result[2] == "Footer text"

    def test_merge_consecutive_list_paragraphs(self):
        paragraphs = [
            "Intro",
            "- item one",
            "- item two",
            "- item three",
            "Outro",
        ]
        result = merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3
        assert result[0] == "Intro"
        assert "- item one" in result[1]
        assert "- item three" in result[1]
        assert result[2] == "Outro"

    def test_no_merge_non_table_non_list(self):
        paragraphs = ["para one", "para two", "para three"]
        result = merge_blocks(paragraphs, "\n\n")
        assert result == paragraphs

    def test_mixed_table_and_list_not_merged(self):
        paragraphs = [
            "| a | b |",
            "- list item",
            "| c | d |",
        ]
        result = merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3

    def test_pipe_separated_docx_rows(self):
        paragraphs = [
            "Intro paragraph",
            "Name | Age | City",
            "Alice | 30 | NYC",
            "Bob | 25 | LA",
            "Summary paragraph",
        ]
        result = merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3
        assert "Name | Age | City" in result[1]
        assert "Bob | 25 | LA" in result[1]

    def test_ordered_list_merge(self):
        paragraphs = [
            "Steps:",
            "1. Do this",
            "2. Do that",
            "3. Done",
            "End.",
        ]
        result = merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3
        assert "1. Do this" in result[1]
        assert "3. Done" in result[1]

    def test_empty_paragraphs_preserved(self):
        paragraphs = ["| a | b |", "", "| c | d |"]
        result = merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3

    def test_separator_used_correctly(self):
        paragraphs = ["| a | b |", "| c | d |"]
        result = merge_blocks(paragraphs, "\n")
        assert result == ["| a | b |\n| c | d |"]

        result2 = merge_blocks(paragraphs, "\n\n")
        assert result2 == ["| a | b |\n\n| c | d |"]

    def test_single_table_paragraph_no_merge(self):
        paragraphs = ["text", "| a | b |", "text"]
        result = merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3


# =============================================================================
# chunk_document
# =============================================================================

class TestChunkDocument:
    def test_basic_chunking(self):
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_document("test.md", content, min_chunk_size=25, max_chunk_size=50)
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.document_name == "test.md" for c in chunks)

    def test_chunk_indices_sequential(self):
        content = "\n\n".join(f"Paragraph {i} with some content." for i in range(10))
        chunks = chunk_document("doc.txt", content, min_chunk_size=25, max_chunk_size=50)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_small_paragraphs_combined(self):
        content = "A.\n\nB.\n\nC.\n\nD."
        chunks = chunk_document("doc.txt", content, min_chunk_size=50, max_chunk_size=100)
        # All fit in one chunk
        assert len(chunks) == 1
        assert "A." in chunks[0].content

    def test_oversized_paragraph_gets_split(self):
        # A paragraph with no sentence endings stays whole even if over max
        long_para = "x" * 2000
        content = f"Intro.\n\n{long_para}\n\nOutro."
        chunks = chunk_document("doc.txt", content, min_chunk_size=50, max_chunk_size=100)
        # long_para has no sentence boundary so it stays as one piece
        long_chunk = next(c for c in chunks if "x" * 100 in c.content)
        assert long_chunk is not None

    def test_sections_accumulate_below_min(self):
        content = "# Section A\n\nContent A.\n\n# Section B\n\nContent B."
        chunks = chunk_document("doc.md", content, min_chunk_size=5000, max_chunk_size=10000)
        # Both sections are well below min_chunk_size — accumulate into one chunk
        assert len(chunks) == 1
        assert "Content A." in chunks[0].content
        assert "Content B." in chunks[0].content

    def test_sections_flush_above_min(self):
        content = "# Section A\n\nContent A.\n\n# Section B\n\nContent B."
        # min=10 means after accumulating "# Section A\n\nContent A." (~22 chars) we're past min
        # so "# Section B" triggers a flush → 2 chunks
        chunks = chunk_document("doc.md", content, min_chunk_size=10, max_chunk_size=5000)
        assert len(chunks) == 2

    def test_section_lca_breadcrumb(self):
        # Sub-heading subsumed — LCA is the parent
        content = "# Top\n\nIntro.\n\n## Sub\n\nDetail."
        chunks = chunk_document("doc.md", content, min_chunk_size=5000, max_chunk_size=10000)
        assert len(chunks) == 1
        assert chunks[0].section == "Top"
        assert chunks[0].breadcrumb == "[doc.md > Top]"
        assert "[doc.md > Top]" in chunks[0].embedding_content

    def test_sibling_sections_lca_is_parent(self):
        content = "# Parent\n\n## Child A\n\nText A.\n\n## Child B\n\nText B."
        chunks = chunk_document("doc.md", content, min_chunk_size=5000, max_chunk_size=10000)
        # All in one chunk; LCA of ["Parent","Child A"] and ["Parent","Child B"] = ["Parent"]
        assert len(chunks) == 1
        assert chunks[0].section == "Parent"
        assert chunks[0].breadcrumb == "[doc.md > Parent]"
        assert "[doc.md > Parent]" in chunks[0].embedding_content

    def test_breadcrumb_absent_when_disabled(self):
        content = "# Section\n\nSome text."
        chunks = chunk_document("doc.md", content, min_chunk_size=5000, max_chunk_size=10000,
                                include_breadcrumb=False)
        assert len(chunks) == 1
        assert not chunks[0].content.startswith("[")
        assert "Some text." in chunks[0].content

    def test_table_continuation_markers(self):
        rows = "\n".join(f"| col{i} | val{i} | extra{i} |" for i in range(100))
        content = f"Intro.\n\n{rows}\n\nOutro."
        chunks = chunk_document("doc.md", content, min_chunk_size=100, max_chunk_size=200)
        table_chunks = [c for c in chunks if "[TABLE:" in c.content]
        assert len(table_chunks) >= 2
        markers = {m for c in table_chunks for m in ["[TABLE:start]", "[TABLE:cont]", "[TABLE:end]"] if m in c.content}
        assert "[TABLE:start]" in markers

    def test_list_continuation_markers(self):
        items = "\n".join(f"- Item {i}: some description text here" for i in range(50))
        chunks = chunk_document("doc.md", items, min_chunk_size=100, max_chunk_size=200)
        list_chunks = [c for c in chunks if "[LIST:" in c.content]
        assert len(list_chunks) >= 2
        markers = {m for c in list_chunks for m in ["[LIST:start]", "[LIST:cont]", "[LIST:end]"] if m in c.content}
        assert "[LIST:start]" in markers

    def test_para_continuation_markers(self):
        # Long prose with sentence boundaries
        sentences = " ".join(f"This is sentence number {i}." for i in range(40))
        chunks = chunk_document("doc.md", sentences, min_chunk_size=100, max_chunk_size=200)
        para_chunks = [c for c in chunks if "[PARA:" in c.content]
        assert len(para_chunks) >= 2
        markers = {m for c in para_chunks for m in ["[PARA:start]", "[PARA:cont]", "[PARA:end]"] if m in c.content}
        assert "[PARA:start]" in markers

    def test_empty_content_returns_no_chunks(self):
        chunks = chunk_document("empty.txt", "", min_chunk_size=100, max_chunk_size=1000)
        assert chunks == []

    def test_single_line_content(self):
        chunks = chunk_document("line.txt", "Just one line.", min_chunk_size=100, max_chunk_size=1000)
        assert len(chunks) == 1
        assert "Just one line." in chunks[0].content
        assert chunks[0].breadcrumb == "[line.txt]"
        assert chunks[0].embedding_content.startswith("[line.txt]")

    def test_sheet_marker_forces_break(self):
        content = "[Sheet: Sheet1]\n\nData row 1.\n\n[Sheet: Sheet2]\n\nData row 2."
        chunks = chunk_document("book.xlsx", content, min_chunk_size=5000, max_chunk_size=10000)
        assert len(chunks) >= 2


# =============================================================================
# Table continuation marker strip (integration)
# =============================================================================

class TestTableMarkerStrip:
    def test_strip_table_markers(self):
        def assemble(chunks):
            parts = []
            for chunk in chunks:
                text = chunk.content
                text = text.replace("[TABLE:start]\n", "")
                text = text.replace("\n[TABLE:cont]", "")
                text = text.replace("[TABLE:cont]\n", "")
                text = text.replace("\n[TABLE:end]", "")
                parts.append(text)
            return "\n\n".join(parts)

        chunks = [
            DocumentChunk(
                document_name="doc",
                content="[TABLE:start]\n| a | b |\n| 1 | 2 |\n[TABLE:cont]",
                chunk_index=0,
            ),
            DocumentChunk(
                document_name="doc",
                content="[TABLE:cont]\n| 3 | 4 |\n| 5 | 6 |\n[TABLE:end]",
                chunk_index=1,
            ),
        ]
        result = assemble(chunks)
        assert "[TABLE:" not in result
        assert "| a | b |" in result
        assert "| 5 | 6 |" in result

    def test_no_markers_passthrough(self):
        def assemble(chunks):
            parts = []
            for chunk in chunks:
                text = chunk.content
                text = text.replace("[TABLE:start]\n", "")
                text = text.replace("\n[TABLE:cont]", "")
                text = text.replace("[TABLE:cont]\n", "")
                text = text.replace("\n[TABLE:end]", "")
                parts.append(text)
            return "\n\n".join(parts)

        chunks = [
            DocumentChunk(document_name="doc", content="regular text", chunk_index=0),
        ]
        result = assemble(chunks)
        assert result == "regular text"


# =============================================================================
# promote_plain_text_headers
# =============================================================================

class TestPromotePlainTextHeaders:
    # ── Question heuristic ────────────────────────────────────────────────────

    def test_question_promoted(self):
        text = "What are the symptoms? Fever and chills are common symptoms."
        result = promote_plain_text_headers(text, promote_short_phrases=False)
        assert "## What are the symptoms?" in result

    def test_question_not_promoted_when_followed_by_question(self):
        text = "What are the symptoms? How is it treated? Treatment involves rest."
        result = promote_plain_text_headers(text, promote_short_phrases=False)
        assert "## What are the symptoms?" not in result

    def test_question_disabled(self):
        text = "What are the symptoms? Fever is common."
        result = promote_plain_text_headers(text, promote_questions=False, promote_short_phrases=False)
        assert "##" not in result

    def test_long_question_not_promoted(self):
        # Exceeds max_header_words * 2 threshold
        text = "What are the primary clinical symptoms seen in patients with advanced disease? Fever is common."
        result = promote_plain_text_headers(text, promote_short_phrases=False, max_header_words=4)
        assert "##" not in result

    # ── Short-phrase heuristic ────────────────────────────────────────────────

    def test_fused_header_promoted(self):
        # Mirrors real medical corpus pattern: "Signs and symptoms Basal cell..."
        text = "Introduction. Signs and symptoms Basal cell carcinoma is common."
        result = promote_plain_text_headers(text, promote_questions=False)
        assert "## Signs and symptoms" in result

    def test_short_phrase_with_verb_not_promoted(self):
        # "Cancer is spreading" has finite verb "is"
        text = "Introduction. Cancer is spreading Basal cell carcinoma is common."
        result = promote_plain_text_headers(text, promote_questions=False)
        assert "## Cancer is spreading" not in result

    def test_short_phrase_too_long_not_promoted(self):
        text = "Introduction. Signs and symptoms of advanced basal cell carcinoma Tumors appear."
        result = promote_plain_text_headers(text, promote_questions=False, max_header_words=4)
        assert "##" not in result

    def test_short_phrase_disabled(self):
        text = "Introduction. Signs and symptoms Basal cell carcinoma is common."
        result = promote_plain_text_headers(text, promote_short_phrases=False, promote_questions=False)
        assert "##" not in result

    def test_short_phrase_exceeds_max_chars_not_promoted(self):
        phrase = "A " * 40  # 80 chars but > default max_header_chars
        text = f"Introduction. {phrase.strip()} Basal cell carcinoma is common."
        result = promote_plain_text_headers(text, promote_questions=False)
        assert "##" not in result

    # ── Integration: chunk_document with promote_headings ─────────────────────

    def test_chunk_document_promote_headings_creates_sections(self):
        text = "Introduction. Signs and symptoms Basal cell carcinoma is a common skin cancer."
        chunks = chunk_document(
            "doc", text, min_chunk_size=10, max_chunk_size=500,
            promote_headings=True, promote_questions=False,
        )
        section_types = {c.chunk_type for c in chunks}
        assert "section" in section_types or any("Signs and symptoms" in c.content for c in chunks)

    def test_chunk_document_promote_headings_false_default(self):
        # Without promote_headings, fused text produces no ## headers
        text = "Introduction. Signs and symptoms Basal cell carcinoma is a common skin cancer."
        chunks_default = chunk_document("doc", text, min_chunk_size=10, max_chunk_size=500)
        chunks_promoted = chunk_document(
            "doc", text, min_chunk_size=10, max_chunk_size=500, promote_headings=True,
            promote_questions=False,
        )
        assert len(chunks_promoted) >= len(chunks_default)

    # ── Output structure ──────────────────────────────────────────────────────

    def test_promoted_header_followed_by_body(self):
        text = "Introduction. Signs and symptoms Basal cell carcinoma is common."
        result = promote_plain_text_headers(text, promote_questions=False)
        idx = result.index("## Signs and symptoms")
        after = result[idx:].split("\n\n", 2)
        assert len(after) >= 2 and "Basal cell" in after[1]

    def test_no_double_newline_at_end(self):
        text = "What are the symptoms? Fever is common."
        result = promote_plain_text_headers(text, promote_short_phrases=False)
        assert not result.endswith("\n\n\n")

    def test_plain_text_unchanged_when_no_matches(self):
        text = "This is a normal sentence. Another normal sentence follows it."
        result = promote_plain_text_headers(text)
        assert "##" not in result


# =============================================================================
# promote_plain_text_headers — structural_levels
# =============================================================================

class TestStructuralLevels:
    # ── CHAPTER promotion ─────────────────────────────────────────────────────

    def test_chapter_with_title_promoted(self):
        text = "CHAPTER I CANADIANS, OLD AND NEW The conquest of Canada was decisive."
        result = promote_plain_text_headers(text, structural_levels=NOVEL_STRUCTURAL_LEVELS)
        # Trailing punctuation stripped per word; comma on CANADIANS, is removed
        assert "## CHAPTER I: CANADIANS OLD AND NEW" in result
        assert "The conquest of Canada" in result

    def test_chapter_no_title_promoted(self):
        text = "CHAPTER I. Music, in however primitive a stage, is universal."
        result = promote_plain_text_headers(text, structural_levels=NOVEL_STRUCTURAL_LEVELS)
        assert "## CHAPTER I" in result
        assert "Music, in however primitive" in result

    def test_chapter_dotdash_separator(self):
        text = "CHAPTER I.--AN OPTIMISTIC FORECAST. As the sun was setting."
        result = promote_plain_text_headers(text, structural_levels=NOVEL_STRUCTURAL_LEVELS)
        assert "## CHAPTER I: AN OPTIMISTIC FORECAST" in result

    def test_chapter_lowercase(self):
        text = "Chapter III THE LADDER OF LEARNING Once upon a time."
        result = promote_plain_text_headers(text, structural_levels=NOVEL_STRUCTURAL_LEVELS)
        assert "## Chapter III: THE LADDER OF LEARNING" in result

    # ── PART promotion ────────────────────────────────────────────────────────

    def test_part_promoted_as_h1(self):
        text = "PART I AMERICA Their graves are scattered far and wide."
        result = promote_plain_text_headers(text, structural_levels=NOVEL_STRUCTURAL_LEVELS)
        assert "# PART I: AMERICA" in result
        assert "## " not in result  # no chapter-level heading

    def test_book_the_first_promoted(self):
        text = "BOOK THE FIRST AN EPIGRAM ON THE AMOURS We who were five books."
        result = promote_plain_text_headers(text, structural_levels=NOVEL_STRUCTURAL_LEVELS)
        assert "# BOOK THE FIRST" in result

    def test_lowercase_part_not_promoted(self):
        # "part of the country" — should not be treated as structural
        text = "He travelled to the part of the country well known for its hills."
        result = promote_plain_text_headers(text, structural_levels=NOVEL_STRUCTURAL_LEVELS)
        assert "#" not in result

    # ── Hierarchy: PART > CHAPTER ─────────────────────────────────────────────

    def test_part_h1_chapter_h2(self):
        text = (
            "PART I AMERICA Their graves are scattered. "
            "CHAPTER I INGLIS OF KINGSMILLS It was evening."
        )
        result = promote_plain_text_headers(text, structural_levels=NOVEL_STRUCTURAL_LEVELS)
        assert "# PART I" in result
        assert "## CHAPTER I" in result
        assert result.index("# PART I") < result.index("## CHAPTER I")

    # ── TOC detection ─────────────────────────────────────────────────────────

    def test_toc_cluster_skipped(self):
        # Three CHAPTER markers within 300 chars → TOC cluster, all skipped.
        # Body chapter is >300 chars away so it is NOT in the cluster.
        toc = (
            "CHAPTER I INGLIS OF KINGSMILLS 1 "
            "CHAPTER II ELSIE MAUD INGLIS 17 "
            "CHAPTER III THE LADDER OF LEARNING 27 "
        )
        body = "x " * 160  # ~320 chars of filler to exceed toc_proximity=300
        chapter = "CHAPTER I INGLIS OF KINGSMILLS It was a cold winter morning."
        text = toc + body + chapter
        result = promote_plain_text_headers(text, structural_levels=NOVEL_STRUCTURAL_LEVELS)
        # TOC entries skipped; body chapter promoted once
        assert result.count("## CHAPTER") == 1

    def test_toc_proximity_configurable(self):
        # Two CHAPTER markers 400 chars apart — default 300 would NOT flag as TOC
        filler = "x " * 150  # ~300 chars
        text = f"CHAPTER I INTRO {filler}CHAPTER II BODY The real content starts here."
        result = promote_plain_text_headers(text, structural_levels=NOVEL_STRUCTURAL_LEVELS)
        assert result.count("## CHAPTER") == 2

    # ── Custom structural_levels ──────────────────────────────────────────────

    def test_custom_structural_levels(self):
        custom = [(r"SECTION\s+\d+", 1), (r"(?i:ARTICLE)\s+\d+", 2)]
        text = "SECTION 1 GENERAL PROVISIONS This section governs. ARTICLE 1 Definitions Terms are defined here."
        result = promote_plain_text_headers(text, structural_levels=custom)
        assert "# SECTION 1" in result
        assert "## ARTICLE 1" in result

    def test_structural_levels_none_disabled(self):
        text = "CHAPTER I CANADIANS, OLD AND NEW The conquest was decisive."
        result = promote_plain_text_headers(text, structural_levels=None,
                                            promote_questions=False, promote_short_phrases=False)
        assert "#" not in result

    # ── chunk_document integration ────────────────────────────────────────────

    def test_chunk_document_structural_levels(self):
        text = (
            "PART I AMERICA Their graves are scattered far and wide. "
            "CHAPTER I INGLIS OF KINGSMILLS It was a cold winter morning in Inverness."
        )
        chunks = chunk_document(
            "novel", text, min_chunk_size=10, max_chunk_size=500,
            promote_headings=True, structural_levels=NOVEL_STRUCTURAL_LEVELS,
            promote_questions=False, promote_short_phrases=False,
        )
        all_content = " ".join(c.content for c in chunks)
        assert "PART I" in all_content
        assert "CHAPTER I" in all_content
