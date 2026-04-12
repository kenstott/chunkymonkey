"""Integration tests for DocumentLoader using LocalTransport + TextExtractor."""

import pytest
from pathlib import Path

from chunkeymonkey import DocumentLoader

SAMPLE_MARKDOWN = """# Introduction

This is an introduction to the topic.

## Background

The background section explains context.

### Historical notes

Some historical context here.

## Methods

We used a specific method.
"""


class TestDocumentLoaderMarkdown:
    def test_loads_markdown_file(self, tmp_path):
        f = tmp_path / "sample.md"
        f.write_text(SAMPLE_MARKDOWN)
        loader = DocumentLoader()
        chunks = loader.load(str(f))
        assert len(chunks) > 0

    def test_chunks_are_document_chunks(self, tmp_path):
        from chunkeymonkey import DocumentChunk
        f = tmp_path / "sample.md"
        f.write_text(SAMPLE_MARKDOWN)
        chunks = DocumentLoader().load(str(f))
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunks_have_sections(self, tmp_path):
        f = tmp_path / "sample.md"
        f.write_text(SAMPLE_MARKDOWN)
        chunks = DocumentLoader().load(str(f))
        sections = [c.section for c in chunks if c.section]
        assert len(sections) > 0

    def test_enrichment_sets_embedding_content(self, tmp_path):
        f = tmp_path / "sample.md"
        f.write_text(SAMPLE_MARKDOWN)
        chunks = DocumentLoader(context_strategy="prefix").load(str(f))
        assert all(c.embedding_content is not None for c in chunks)

    def test_no_enrichment_when_strategy_none(self, tmp_path):
        f = tmp_path / "sample.md"
        f.write_text(SAMPLE_MARKDOWN)
        chunks = DocumentLoader(context_strategy=None).load(str(f))
        assert all(c.embedding_content is None for c in chunks)

    def test_document_name_from_filename(self, tmp_path):
        f = tmp_path / "my_report.md"
        f.write_text(SAMPLE_MARKDOWN)
        chunks = DocumentLoader().load(str(f))
        assert all(c.document_name == "my_report" for c in chunks)

    def test_chunk_indices_sequential(self, tmp_path):
        f = tmp_path / "sample.md"
        f.write_text(SAMPLE_MARKDOWN)
        chunks = DocumentLoader().load(str(f))
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_inline_enrichment_strategy(self, tmp_path):
        f = tmp_path / "sample.md"
        f.write_text(SAMPLE_MARKDOWN)
        chunks = DocumentLoader(context_strategy="inline").load(str(f))
        enriched = [c for c in chunks if c.section]
        assert all("[" in c.embedding_content for c in enriched)


class TestDocumentLoaderText:
    def test_load_text_returns_chunks(self):
        loader = DocumentLoader()
        chunks = loader.load_text("Paragraph one.\n\nParagraph two.\n\nParagraph three.", "test.txt")
        assert len(chunks) >= 1

    def test_load_text_content_present(self):
        loader = DocumentLoader()
        chunks = loader.load_text("Hello world content.", "test.txt")
        full_text = " ".join(c.content for c in chunks)
        assert "Hello world content." in full_text

    def test_load_bytes(self):
        loader = DocumentLoader()
        chunks = loader.load_bytes(b"Hello world content.", "test.txt", doc_type="text")
        assert len(chunks) == 1
        assert "Hello" in chunks[0].content

    def test_load_bytes_auto_type_from_name(self):
        loader = DocumentLoader()
        chunks = loader.load_bytes(b"# Title\n\nSome content.", "readme.md")
        assert len(chunks) >= 1

    def test_load_bytes_with_enrichment(self):
        loader = DocumentLoader(context_strategy="prefix")
        chunks = loader.load_bytes(
            b"# Intro\n\nSection content here.", "doc.md", doc_type="text"
        )
        assert all(c.embedding_content is not None for c in chunks)

    def test_load_html_bytes(self):
        loader = DocumentLoader()
        chunks = loader.load_bytes(b"<h1>Title</h1><p>Body text here.</p>", "page.html", doc_type="html")
        assert len(chunks) >= 1
        full_text = " ".join(c.content for c in chunks)
        assert "Title" in full_text or "Body" in full_text

    def test_load_text_no_enrichment(self):
        loader = DocumentLoader(context_strategy=None)
        chunks = loader.load_text("Some text content.", "test.txt")
        assert all(c.embedding_content is None for c in chunks)


class TestCustomExtractor:
    def test_custom_extractor_registered(self, tmp_path):
        class FakeExtractor:
            def can_handle(self, doc_type):
                return doc_type == "fake"

            def extract(self, data, source_path=None):
                return "fake content extracted"

        f = tmp_path / "test.fake"
        f.write_bytes(b"raw")
        loader = DocumentLoader(extra_extractors=[FakeExtractor()])
        chunks = loader.load_bytes(b"raw", "test.fake", doc_type="fake")
        assert chunks[0].content == "fake content extracted"

    def test_custom_extractor_takes_priority(self, tmp_path):
        class OverrideHtmlExtractor:
            def can_handle(self, doc_type):
                return doc_type == "html"

            def extract(self, data, source_path=None):
                return "override extracted"

        loader = DocumentLoader(extra_extractors=[OverrideHtmlExtractor()])
        chunks = loader.load_bytes(b"<h1>ignored</h1>", "page.html", doc_type="html")
        assert chunks[0].content == "override extracted"

    def test_custom_transport_used(self, tmp_path):
        from chunkeymonkey.transports._protocol import FetchResult

        class MemTransport:
            def can_handle(self, uri):
                return uri.startswith("mem://")

            def fetch(self, uri, **kwargs):
                return FetchResult(
                    data=b"# Memory Doc\n\nIn-memory content.",
                    detected_mime="text/markdown",
                    source_path="memory.md",
                )

        loader = DocumentLoader(extra_transports=[MemTransport()])
        chunks = loader.load("mem://memory.md")
        assert len(chunks) >= 1
        full_text = " ".join(c.content for c in chunks)
        assert "In-memory content." in full_text
