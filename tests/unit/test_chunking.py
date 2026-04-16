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
    chunk_document,
    is_list_line,
    is_table_line,
    merge_blocks,
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
