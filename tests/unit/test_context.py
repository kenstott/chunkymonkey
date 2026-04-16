# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 75e44fdf-178e-4130-b329-f5e639aa0819
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for chunkymonkey.context — enrich_chunk and enrich_chunks."""

import pytest

from chunkymonkey.context import enrich_chunk, enrich_chunks
from chunkymonkey.models import DocumentChunk


def _make_chunk(section=None, content="Some chunk content.", doc_name="test_doc"):
    return DocumentChunk(
        document_name=doc_name,
        content=content,
        section=section,
        chunk_index=0,
    )


# =============================================================================
# enrich_chunk — prefix strategy
# =============================================================================

class TestEnrichChunkPrefix:
    def test_doc_and_section_both_present(self):
        chunk = _make_chunk(section="Methods > Table 1", doc_name="my_report")
        result = enrich_chunk(chunk, strategy="prefix")
        assert result.embedding_content == (
            "Document: my_report\nSection: Methods > Table 1\n\nSome chunk content."
        )

    def test_doc_only_no_section(self):
        chunk = _make_chunk(section=None, content="Plain content.", doc_name="my_report")
        result = enrich_chunk(chunk, strategy="prefix")
        assert result.embedding_content == "Document: my_report\n\nPlain content."

    def test_section_only_no_doc_name(self):
        chunk = DocumentChunk(
            document_name="",
            content="Content here.",
            section="Results",
            chunk_index=0,
        )
        result = enrich_chunk(chunk, strategy="prefix")
        assert result.embedding_content == "Section: Results\n\nContent here."

    def test_neither_doc_nor_section_passthrough(self):
        chunk = DocumentChunk(
            document_name="",
            content="Plain content.",
            section=None,
            chunk_index=0,
        )
        result = enrich_chunk(chunk, strategy="prefix")
        assert result.embedding_content == "Plain content."

    def test_original_content_not_mutated(self):
        chunk = _make_chunk(section="Intro", content="Original text.")
        original_content = chunk.content
        enrich_chunk(chunk, strategy="prefix")
        assert chunk.content == original_content

    def test_returns_new_instance(self):
        chunk = _make_chunk(section="Intro")
        result = enrich_chunk(chunk, strategy="prefix")
        assert result is not chunk

    def test_unknown_strategy_raises(self):
        chunk = _make_chunk(section="Intro")
        with pytest.raises(ValueError, match="magic"):
            enrich_chunk(chunk, strategy="magic")

    def test_all_other_fields_preserved(self):
        chunk = DocumentChunk(
            document_name="my_doc",
            content="Content here.",
            section="Sec A",
            chunk_index=5,
            source_offset=100,
            source_length=50,
            chunk_type="schema",
        )
        result = enrich_chunk(chunk, strategy="prefix")
        assert result.document_name == "my_doc"
        assert result.chunk_index == 5
        assert result.source_offset == 100
        assert result.source_length == 50
        assert result.chunk_type == "schema"

    def test_content_appears_after_header_block(self):
        chunk = _make_chunk(section="Results", content="The p-value was 0.03.")
        result = enrich_chunk(chunk, strategy="prefix")
        assert result.embedding_content.endswith("The p-value was 0.03.")
        assert "\n\n" in result.embedding_content

    def test_document_name_included_with_section(self):
        """Document name must appear even when section is present — primary test of the fix."""
        chunk = _make_chunk(section="Indemnification", doc_name="techcorp_msa")
        result = enrich_chunk(chunk, strategy="prefix")
        assert "techcorp_msa" in result.embedding_content
        assert "Indemnification" in result.embedding_content

    def test_document_name_disambiguates_identical_sections(self):
        """Two chunks with the same section but different documents get different embedding_content."""
        chunk_a = DocumentChunk(
            document_name="techcorp_msa", content="…cap is 12 months fees…",
            section="Limitation of Liability", chunk_index=0,
        )
        chunk_b = DocumentChunk(
            document_name="cloudsolutions_agreement", content="…cap is 12 months fees…",
            section="Limitation of Liability", chunk_index=0,
        )
        result_a = enrich_chunk(chunk_a, strategy="prefix")
        result_b = enrich_chunk(chunk_b, strategy="prefix")
        assert result_a.embedding_content != result_b.embedding_content
        assert "techcorp_msa" in result_a.embedding_content
        assert "cloudsolutions_agreement" in result_b.embedding_content


# =============================================================================
# enrich_chunk — inline strategy
# =============================================================================

class TestEnrichChunkInline:
    def test_doc_and_section_both_present(self):
        chunk = _make_chunk(section="Discussion", content="We observed a trend.", doc_name="my_doc")
        result = enrich_chunk(chunk, strategy="inline")
        assert result.embedding_content == "[my_doc > Discussion] We observed a trend."

    def test_doc_only_no_section(self):
        chunk = _make_chunk(section=None, content="Plain.", doc_name="my_doc")
        result = enrich_chunk(chunk, strategy="inline")
        assert result.embedding_content == "[my_doc] Plain."

    def test_section_only_no_doc_name(self):
        chunk = DocumentChunk(
            document_name="", content="Text.", section="Methods", chunk_index=0,
        )
        result = enrich_chunk(chunk, strategy="inline")
        assert result.embedding_content == "[Methods] Text."

    def test_neither_passthrough(self):
        chunk = DocumentChunk(
            document_name="", content="Text.", section=None, chunk_index=0,
        )
        result = enrich_chunk(chunk, strategy="inline")
        assert result.embedding_content == "Text."

    def test_document_name_included_with_section(self):
        chunk = _make_chunk(section="Indemnification", doc_name="cloudsolutions_agreement")
        result = enrich_chunk(chunk, strategy="inline")
        assert "cloudsolutions_agreement" in result.embedding_content
        assert "Indemnification" in result.embedding_content


# =============================================================================
# enrich_chunks
# =============================================================================

class TestEnrichChunks:
    def test_batch_all_get_embedding_content(self):
        chunks = [_make_chunk(section=f"Sec {i}") for i in range(5)]
        results = enrich_chunks(chunks, strategy="prefix")
        assert all(r.embedding_content is not None for r in results)

    def test_empty_list(self):
        results = enrich_chunks([], strategy="prefix")
        assert results == []

    def test_mixed_sections(self):
        chunks = [
            _make_chunk(section="Intro", content="Has a section.", doc_name="doc_a"),
            _make_chunk(section=None, content="No section here.", doc_name="doc_a"),
            _make_chunk(section="Methods", content="Another section.", doc_name="doc_a"),
        ]
        results = enrich_chunks(chunks, strategy="prefix")
        assert all(r.embedding_content is not None for r in results)
        assert "doc_a" in results[0].embedding_content
        assert "Intro" in results[0].embedding_content
        assert "doc_a" in results[1].embedding_content
        assert "Methods" in results[2].embedding_content

    def test_returns_new_list(self):
        chunks = [_make_chunk(section="A"), _make_chunk(section="B")]
        results = enrich_chunks(chunks, strategy="prefix")
        assert results is not chunks

    def test_originals_not_mutated(self):
        chunks = [_make_chunk(section="Sec", content="Text.")]
        enrich_chunks(chunks, strategy="inline")
        assert chunks[0].embedding_content is None

    def test_length_preserved(self):
        chunks = [_make_chunk() for _ in range(7)]
        results = enrich_chunks(chunks, strategy="prefix")
        assert len(results) == 7

    def test_all_chunks_include_document_name(self):
        """Every chunk in a batch should carry the document name in embedding_content."""
        chunks = [
            DocumentChunk(
                document_name="techcorp_msa", content=f"Clause {i} text.",
                section=f"Section {i}", chunk_index=i,
            )
            for i in range(5)
        ]
        results = enrich_chunks(chunks, strategy="prefix")
        for r in results:
            assert "techcorp_msa" in r.embedding_content
