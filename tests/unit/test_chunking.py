"""Tests for chunkeymonkey chunking primitives."""

import pytest

from chunkeymonkey import (
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
        chunks = chunk_document("test.md", content, chunk_size=50, table_chunk_limit=500)
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.document_name == "test.md" for c in chunks)

    def test_chunk_indices_sequential(self):
        content = "\n\n".join(f"Paragraph {i} with some content." for i in range(10))
        chunks = chunk_document("doc.txt", content, chunk_size=50, table_chunk_limit=500)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_small_paragraphs_combined(self):
        content = "A.\n\nB.\n\nC.\n\nD."
        chunks = chunk_document("doc.txt", content, chunk_size=100, table_chunk_limit=500)
        # All fit in one chunk
        assert len(chunks) == 1
        assert "A." in chunks[0].content

    def test_oversized_paragraph_not_split(self):
        long_para = "x" * 2000
        content = f"Intro.\n\n{long_para}\n\nOutro."
        chunks = chunk_document("doc.txt", content, chunk_size=100, table_chunk_limit=500)
        long_chunk = next(c for c in chunks if len(c.content) > 100)
        assert long_chunk is not None

    def test_markdown_heading_forces_break(self):
        content = "# Section A\n\nContent A.\n\n# Section B\n\nContent B."
        chunks = chunk_document("doc.md", content, chunk_size=5000, table_chunk_limit=500)
        # Heading forces a break — should produce at least 2 chunks
        assert len(chunks) >= 2

    def test_section_breadcrumbs_set(self):
        content = "# Top\n\nIntro.\n\n## Sub\n\nDetail."
        chunks = chunk_document("doc.md", content, chunk_size=5000, table_chunk_limit=500)
        sections = [c.section for c in chunks if c.section]
        assert any("Top" in s for s in sections)

    def test_table_continuation_markers(self):
        rows = "\n".join(f"| col{i} | val{i} | extra{i} |" for i in range(100))
        content = f"Intro.\n\n{rows}\n\nOutro."
        chunks = chunk_document("doc.md", content, chunk_size=200, table_chunk_limit=200)
        table_chunks = [c for c in chunks if "[TABLE:" in c.content]
        assert len(table_chunks) >= 2
        markers = {m for c in table_chunks for m in ["[TABLE:start]", "[TABLE:cont]", "[TABLE:end]"] if m in c.content}
        assert "[TABLE:start]" in markers

    def test_empty_content_returns_no_chunks(self):
        chunks = chunk_document("empty.txt", "", chunk_size=1000, table_chunk_limit=500)
        assert chunks == []

    def test_single_line_content(self):
        chunks = chunk_document("line.txt", "Just one line.", chunk_size=1000, table_chunk_limit=500)
        assert len(chunks) == 1
        assert chunks[0].content == "Just one line."

    def test_sheet_marker_forces_break(self):
        content = "[Sheet: Sheet1]\n\nData row 1.\n\n[Sheet: Sheet2]\n\nData row 2."
        chunks = chunk_document("book.xlsx", content, chunk_size=5000, table_chunk_limit=500)
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
