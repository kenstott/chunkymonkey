# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 8a4af826-045e-442b-853e-228c1f714e18
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for chunkymonkey.extractors._mime MIME normalization and detection."""

import pytest

from chunkymonkey.extractors._mime import (
    detect_type_from_source,
    is_binary_type,
    normalize_type,
)


# =============================================================================
# normalize_type
# =============================================================================

class TestNormalizeType:
    def test_pdf_mime(self):
        assert normalize_type("application/pdf") == "pdf"

    def test_docx_mime(self):
        assert normalize_type(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ) == "docx"

    def test_xlsx_mime(self):
        assert normalize_type(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ) == "xlsx"

    def test_pptx_mime(self):
        assert normalize_type(
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        ) == "pptx"

    def test_short_alias_pdf(self):
        assert normalize_type("pdf") == "pdf"

    def test_short_alias_html(self):
        assert normalize_type("html") == "html"

    def test_short_alias_markdown(self):
        assert normalize_type("markdown") == "markdown"

    def test_html(self):
        assert normalize_type("text/html") == "html"

    def test_plain_text(self):
        assert normalize_type("text/plain") == "text"

    def test_csv(self):
        assert normalize_type("text/csv") == "csv"

    def test_json(self):
        assert normalize_type("application/json") == "json"

    def test_auto_passthrough(self):
        assert normalize_type("auto") == "auto"

    def test_empty_string_returns_auto(self):
        assert normalize_type("") == "auto"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown document type"):
            normalize_type("application/x-totally-unknown")

    def test_case_insensitive(self):
        assert normalize_type("PDF") == "pdf"
        assert normalize_type("text/HTML") == "html"

    def test_legacy_file_type_returns_auto(self):
        assert normalize_type("file") == "auto"

    def test_legacy_http_type_returns_auto(self):
        assert normalize_type("http") == "auto"


# =============================================================================
# detect_type_from_source
# =============================================================================

class TestDetectTypeFromSource:
    def test_pdf_extension(self):
        assert detect_type_from_source("report.pdf", None) == "pdf"

    def test_docx_extension(self):
        assert detect_type_from_source("doc.docx", None) == "docx"

    def test_xlsx_extension(self):
        assert detect_type_from_source("data.xlsx", None) == "xlsx"

    def test_pptx_extension(self):
        assert detect_type_from_source("slides.pptx", None) == "pptx"

    def test_html_extension(self):
        assert detect_type_from_source("page.html", None) == "html"

    def test_htm_extension(self):
        assert detect_type_from_source("page.htm", None) == "html"

    def test_md_extension(self):
        result = detect_type_from_source("readme.md", None)
        assert result in ("markdown", "md", "text", "auto")

    def test_txt_extension(self):
        assert detect_type_from_source("notes.txt", None) == "text"

    def test_csv_extension(self):
        assert detect_type_from_source("data.csv", None) == "csv"

    def test_json_extension(self):
        assert detect_type_from_source("config.json", None) == "json"

    def test_mime_overrides_extension(self):
        # MIME wins over extension
        result = detect_type_from_source("file.txt", "application/pdf")
        assert result == "pdf"

    def test_html_mime_overrides_extension(self):
        result = detect_type_from_source("file.pdf", "text/html")
        assert result == "html"

    def test_both_none_returns_text(self):
        assert detect_type_from_source(None, None) == "text"

    def test_path_none_but_mime_present(self):
        assert detect_type_from_source(None, "application/pdf") == "pdf"

    def test_unknown_extension_falls_back_to_text(self):
        assert detect_type_from_source("file.unknown_ext_xyz", None) == "text"

    def test_mime_with_charset_param(self):
        result = detect_type_from_source("file.txt", "text/html; charset=utf-8")
        assert result == "html"


# =============================================================================
# is_binary_type
# =============================================================================

class TestIsBinaryType:
    def test_pdf_is_binary(self):
        assert is_binary_type("pdf")

    def test_docx_is_binary(self):
        assert is_binary_type("docx")

    def test_xlsx_is_binary(self):
        assert is_binary_type("xlsx")

    def test_pptx_is_binary(self):
        assert is_binary_type("pptx")

    def test_image_is_binary(self):
        assert is_binary_type("image")

    def test_audio_is_binary(self):
        assert is_binary_type("audio")

    def test_text_not_binary(self):
        assert not is_binary_type("text")

    def test_markdown_not_binary(self):
        assert not is_binary_type("markdown")

    def test_html_not_binary(self):
        assert not is_binary_type("html")

    def test_csv_not_binary(self):
        assert not is_binary_type("csv")

    def test_json_not_binary(self):
        assert not is_binary_type("json")
