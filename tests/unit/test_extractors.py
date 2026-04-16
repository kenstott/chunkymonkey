# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 90ffcf48-53ce-4e1d-b958-bbb627326d6f
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for stdlib-only extractors (HtmlExtractor, TextExtractor) and detect_extractor."""

import pytest

from chunkymonkey.extractors._html import HtmlExtractor
from chunkymonkey.extractors._text import TextExtractor
from chunkymonkey.extractors import detect_extractor


# =============================================================================
# HtmlExtractor
# =============================================================================

class TestHtmlExtractor:
    def test_h1_to_markdown_heading(self):
        data = b"<h1>Title</h1>"
        result = HtmlExtractor().extract(data)
        assert "# Title" in result

    def test_h2_to_markdown_heading(self):
        data = b"<h2>Subtitle</h2>"
        result = HtmlExtractor().extract(data)
        assert "## Subtitle" in result

    def test_strips_nav_element(self):
        data = b"<nav>Nav junk here</nav><p>Good content</p>"
        result = HtmlExtractor().extract(data)
        assert "Nav junk here" not in result
        assert "Good content" in result

    def test_strips_script_element(self):
        data = b"<script>alert('bad')</script><p>Main text</p>"
        result = HtmlExtractor().extract(data)
        assert "alert" not in result
        assert "Main text" in result

    def test_strips_style_element(self):
        data = b"<style>.body{color:red}</style><p>Styled content</p>"
        result = HtmlExtractor().extract(data)
        assert ".body" not in result
        assert "Styled content" in result

    def test_table_to_pipe_format(self):
        data = b"<table><tr><td>A</td><td>B</td></tr></table>"
        result = HtmlExtractor().extract(data)
        assert "A" in result and "B" in result
        assert "|" in result

    def test_unordered_list(self):
        data = b"<ul><li>Item one</li><li>Item two</li></ul>"
        result = HtmlExtractor().extract(data)
        assert "Item one" in result
        assert "Item two" in result
        assert "-" in result

    def test_ordered_list(self):
        data = b"<ol><li>First</li><li>Second</li></ol>"
        result = HtmlExtractor().extract(data)
        assert "First" in result
        assert "Second" in result
        assert "1." in result

    def test_paragraph_text(self):
        data = b"<p>Hello world</p>"
        result = HtmlExtractor().extract(data)
        assert "Hello world" in result

    def test_can_handle_html(self):
        assert HtmlExtractor().can_handle("html")

    def test_can_handle_htm(self):
        assert HtmlExtractor().can_handle("htm")

    def test_cannot_handle_pdf(self):
        assert not HtmlExtractor().can_handle("pdf")

    def test_cannot_handle_text(self):
        assert not HtmlExtractor().can_handle("text")

    def test_cannot_handle_docx(self):
        assert not HtmlExtractor().can_handle("docx")

    def test_returns_string(self):
        result = HtmlExtractor().extract(b"<p>test</p>")
        assert isinstance(result, str)

    def test_empty_html(self):
        result = HtmlExtractor().extract(b"")
        assert isinstance(result, str)

    def test_bold_conversion(self):
        data = b"<strong>Important</strong>"
        result = HtmlExtractor().extract(data)
        assert "**Important**" in result

    def test_italic_conversion(self):
        data = b"<em>Italic</em>"
        result = HtmlExtractor().extract(data)
        assert "*Italic*" in result


# =============================================================================
# TextExtractor
# =============================================================================

class TestTextExtractor:
    def test_utf8_decode(self):
        data = b"hello world"
        result = TextExtractor().extract(data)
        assert result == "hello world"

    def test_fallback_latin1(self):
        # bytes that are invalid UTF-8
        data = bytes([0xE9, 0xE0, 0xF9])  # Latin-1 accented chars
        result = TextExtractor().extract(data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_utf8_with_unicode(self):
        text = "Hello, 世界! Привет!"
        result = TextExtractor().extract(text.encode("utf-8"))
        assert result == text

    def test_can_handle_text(self):
        assert TextExtractor().can_handle("text")

    def test_can_handle_markdown(self):
        assert TextExtractor().can_handle("markdown")

    def test_cannot_handle_json(self):
        assert not TextExtractor().can_handle("json")

    def test_can_handle_csv(self):
        assert TextExtractor().can_handle("csv")

    def test_can_handle_yaml(self):
        assert TextExtractor().can_handle("yaml")

    def test_cannot_handle_xml(self):
        assert not TextExtractor().can_handle("xml")

    def test_cannot_handle_jsonl(self):
        assert not TextExtractor().can_handle("jsonl")

    def test_cannot_handle_pdf(self):
        assert not TextExtractor().can_handle("pdf")

    def test_cannot_handle_docx(self):
        assert not TextExtractor().can_handle("docx")

    def test_returns_string(self):
        result = TextExtractor().extract(b"some bytes")
        assert isinstance(result, str)

    def test_empty_bytes(self):
        result = TextExtractor().extract(b"")
        assert result == ""


# =============================================================================
# detect_extractor
# =============================================================================

class TestDetectExtractor:
    def test_detect_html(self):
        assert isinstance(detect_extractor("html"), HtmlExtractor)

    def test_detect_text(self):
        assert isinstance(detect_extractor("text"), TextExtractor)

    def test_detect_markdown(self):
        assert isinstance(detect_extractor("markdown"), TextExtractor)

    def test_detect_csv(self):
        assert isinstance(detect_extractor("csv"), TextExtractor)

    def test_detect_json(self):
        from chunkymonkey.extractors._json import JsonExtractor
        assert isinstance(detect_extractor("json"), JsonExtractor)

    def test_detect_pdf(self):
        from chunkymonkey.extractors._pdf import PdfExtractor
        assert isinstance(detect_extractor("pdf"), PdfExtractor)

    def test_detect_docx(self):
        from chunkymonkey.extractors._docx import DocxExtractor
        assert isinstance(detect_extractor("docx"), DocxExtractor)

    def test_detect_xlsx(self):
        from chunkymonkey.extractors._xlsx import XlsxExtractor
        assert isinstance(detect_extractor("xlsx"), XlsxExtractor)

    def test_detect_pptx(self):
        from chunkymonkey.extractors._pptx import PptxExtractor
        assert isinstance(detect_extractor("pptx"), PptxExtractor)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="No extractor found"):
            detect_extractor("totally_unknown_format_xyz")
