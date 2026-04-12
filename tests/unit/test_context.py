"""Tests for chunkeymonkey.context — enrich_chunk and enrich_chunks."""

import pytest

from chunkeymonkey.context import enrich_chunk, enrich_chunks
from chunkeymonkey.models import DocumentChunk


def _make_chunk(section=None, content="Some chunk content."):
    return DocumentChunk(
        document_name="test_doc",
        content=content,
        section=section,
        chunk_index=0,
    )


# =============================================================================
# enrich_chunk
# =============================================================================

class TestEnrichChunk:
    def test_prefix_strategy_prepends_section(self):
        chunk = _make_chunk(section="Methods > Table 1")
        result = enrich_chunk(chunk, strategy="prefix")
        assert result.embedding_content.startswith("Section: Methods > Table 1\n\n")
        assert chunk.content in result.embedding_content

    def test_inline_strategy_brackets_section(self):
        chunk = _make_chunk(section="Methods > Table 1")
        result = enrich_chunk(chunk, strategy="inline")
        assert result.embedding_content.startswith("[Methods > Table 1]")
        assert chunk.content in result.embedding_content

    def test_no_section_passthrough(self):
        chunk = _make_chunk(section=None, content="Plain content.")
        result = enrich_chunk(chunk, strategy="prefix")
        assert result.embedding_content == chunk.content

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

    def test_prefix_content_structure(self):
        chunk = _make_chunk(section="Results", content="The p-value was 0.03.")
        result = enrich_chunk(chunk, strategy="prefix")
        assert result.embedding_content == "Section: Results\n\nThe p-value was 0.03."

    def test_inline_content_structure(self):
        chunk = _make_chunk(section="Discussion", content="We observed a trend.")
        result = enrich_chunk(chunk, strategy="inline")
        assert result.embedding_content == "[Discussion] We observed a trend."

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
            _make_chunk(section="Intro", content="Has a section."),
            _make_chunk(section=None, content="No section here."),
            _make_chunk(section="Methods", content="Another section."),
        ]
        results = enrich_chunks(chunks, strategy="prefix")
        assert all(r.embedding_content is not None for r in results)
        assert results[0].embedding_content.startswith("Section: Intro")
        assert results[1].embedding_content == "No section here."
        assert results[2].embedding_content.startswith("Section: Methods")

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
