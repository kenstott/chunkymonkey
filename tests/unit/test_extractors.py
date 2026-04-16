# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 90ffcf48-53ce-4e1d-b958-bbb627326d6f
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for stdlib-only extractors (HtmlExtractor, TextExtractor, MarkdownExtractor, YamlExtractor) and detect_extractor."""

import pytest

from chunkymonkey.extractors._html import HtmlExtractor
from chunkymonkey.extractors._text import TextExtractor
from chunkymonkey.extractors._markdown import MarkdownExtractor
from chunkymonkey.extractors._yaml import YamlExtractor
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

    def test_cannot_handle_markdown(self):
        assert not TextExtractor().can_handle("markdown")

    def test_cannot_handle_json(self):
        assert not TextExtractor().can_handle("json")

    def test_can_handle_csv(self):
        assert TextExtractor().can_handle("csv")

    def test_cannot_handle_yaml(self):
        assert not TextExtractor().can_handle("yaml")

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
        assert isinstance(detect_extractor("markdown"), MarkdownExtractor)

    def test_detect_csv(self):
        from chunkymonkey.extractors._csv import CsvExtractor
        assert isinstance(detect_extractor("csv"), CsvExtractor)

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

    def test_detect_yaml(self):
        assert isinstance(detect_extractor("yaml"), YamlExtractor)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="No extractor found"):
            detect_extractor("totally_unknown_format_xyz")


# =============================================================================
# MarkdownExtractor
# =============================================================================

class TestMarkdownExtractor:
    def test_passthrough_plain_markdown(self):
        data = b"# Hello\n\nSome paragraph."
        result = MarkdownExtractor().extract(data)
        assert result == "# Hello\n\nSome paragraph."

    def test_strips_frontmatter(self):
        data = b"---\ntitle: My Doc\ndate: 2025-01-01\n---\n# Actual Content"
        result = MarkdownExtractor().extract(data)
        assert "title:" not in result
        assert "# Actual Content" in result

    def test_no_frontmatter_unchanged(self):
        data = b"# Title\n\nParagraph without frontmatter."
        result = MarkdownExtractor().extract(data)
        assert "# Title" in result
        assert "Paragraph without frontmatter." in result

    def test_frontmatter_only_returns_empty(self):
        data = b"---\ntitle: Only Frontmatter\n---\n"
        result = MarkdownExtractor().extract(data)
        assert result.strip() == ""

    def test_can_handle_markdown(self):
        assert MarkdownExtractor().can_handle("markdown")

    def test_cannot_handle_text(self):
        assert not MarkdownExtractor().can_handle("text")

    def test_cannot_handle_html(self):
        assert not MarkdownExtractor().can_handle("html")

    def test_latin1_fallback(self):
        data = "---\ntitle: Test\n---\n# Héllo".encode("latin-1")
        result = MarkdownExtractor().extract(data)
        assert "# H" in result
        assert "title:" not in result


# =============================================================================
# YamlExtractor
# =============================================================================

class TestYamlExtractor:
    def test_simple_mapping(self):
        data = b"name: Alice\nage: 30"
        result = YamlExtractor().extract(data)
        assert "name" in result
        assert "Alice" in result
        assert "age" in result
        assert "30" in result

    def test_nested_mapping(self):
        data = b"person:\n  name: Bob\n  city: Austin"
        result = YamlExtractor().extract(data)
        assert "Bob" in result
        assert "Austin" in result

    def test_list_values(self):
        data = b"fruits:\n  - apple\n  - banana\n  - cherry"
        result = YamlExtractor().extract(data)
        assert "apple" in result
        assert "banana" in result
        assert "cherry" in result

    def test_multi_document(self):
        data = b"name: Doc1\n---\nname: Doc2"
        result = YamlExtractor().extract(data)
        assert "Doc1" in result
        assert "Doc2" in result
        assert "---" in result

    def test_invalid_yaml_returns_raw(self):
        data = b"key: [unclosed"
        result = YamlExtractor().extract(data)
        assert isinstance(result, str)

    def test_can_handle_yaml(self):
        assert YamlExtractor().can_handle("yaml")

    def test_cannot_handle_json(self):
        assert not YamlExtractor().can_handle("json")

    def test_cannot_handle_text(self):
        assert not YamlExtractor().can_handle("text")

    def test_empty_doc_skipped(self):
        data = b"---\n---\nname: Real"
        result = YamlExtractor().extract(data)
        assert "Real" in result


# =============================================================================
# CsvExtractor
# =============================================================================

class TestCsvExtractor:
    def test_renders_markdown_table(self):
        from chunkymonkey.extractors._csv import CsvExtractor
        data = b"name,age\nAlice,30\nBob,25"
        result = CsvExtractor().extract(data)
        assert "| name |" in result
        assert "| age |" in result
        assert "Alice" in result
        assert "30" in result
        assert "---" in result

    def test_pipe_chars_escaped(self):
        from chunkymonkey.extractors._csv import CsvExtractor
        data = b"col\nval|ue"
        result = CsvExtractor().extract(data)
        assert "\\|" in result

    def test_empty_csv_returns_empty(self):
        from chunkymonkey.extractors._csv import CsvExtractor
        result = CsvExtractor().extract(b"")
        assert result == ""

    def test_can_handle_csv(self):
        from chunkymonkey.extractors._csv import CsvExtractor
        assert CsvExtractor().can_handle("csv")

    def test_can_handle_tsv(self):
        from chunkymonkey.extractors._csv import CsvExtractor
        assert CsvExtractor().can_handle("tsv")

    def test_cannot_handle_text(self):
        from chunkymonkey.extractors._csv import CsvExtractor
        assert not CsvExtractor().can_handle("text")

    def test_tab_separated(self):
        from chunkymonkey.extractors._csv import CsvExtractor
        data = b"name\tage\nAlice\t30"
        result = CsvExtractor().extract(data)
        assert "Alice" in result
        assert "30" in result


# =============================================================================
# OdfExtractor
# =============================================================================

def _make_odt(paragraphs: list[str], headings: list[tuple[int, str]] | None = None) -> bytes:
    """Build a minimal ODT document in memory using odfpy."""
    pytest.importorskip("odf")
    from odf.opendocument import OpenDocumentText
    from odf.text import P, H
    from odf.style import Style, TextProperties
    import io

    doc = OpenDocumentText()
    for level, text in (headings or []):
        h = H(outlinelevel=level, text=text)
        doc.text.addElement(h)
    for text in paragraphs:
        p = P(text=text)
        doc.text.addElement(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_ods(sheets: dict[str, list[list[str]]]) -> bytes:
    """Build a minimal ODS document in memory using odfpy."""
    pytest.importorskip("odf")
    from odf.opendocument import OpenDocumentSpreadsheet
    from odf.table import Table, TableRow, TableCell
    from odf.text import P
    import io

    doc = OpenDocumentSpreadsheet()
    for sheet_name, rows in sheets.items():
        table = Table(name=sheet_name)
        for row_data in rows:
            row = TableRow()
            for cell_text in row_data:
                cell = TableCell()
                cell.addElement(P(text=cell_text))
                row.addElement(cell)
            table.addElement(row)
        doc.spreadsheet.addElement(table)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_odp(slides: list[list[str]]) -> bytes:
    """Build a minimal ODP document in memory using odfpy."""
    pytest.importorskip("odf")
    from odf.opendocument import OpenDocumentPresentation
    from odf.draw import Page, TextBox, Frame
    from odf.text import P
    import io

    doc = OpenDocumentPresentation()
    for slide_texts in slides:
        page = Page(masterpagename="Default")
        for text in slide_texts:
            frame = Frame(width="20cm", height="2cm", x="2cm", y="2cm")
            tb = TextBox()
            tb.addElement(P(text=text))
            frame.addElement(tb)
            page.addElement(frame)
        doc.presentation.addElement(page)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class TestOdfExtractor:
    def test_odt_paragraph_text(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        data = _make_odt(["Hello world", "Second paragraph"])
        result = OdfExtractor().extract(data)
        assert "Hello world" in result
        assert "Second paragraph" in result

    def test_odt_heading_levels(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        data = _make_odt([], headings=[(1, "Chapter One"), (2, "Section")])
        result = OdfExtractor().extract(data)
        assert "# Chapter One" in result
        assert "## Section" in result

    def test_ods_sheet_rows(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        data = _make_ods({"Sales": [["Name", "Amount"], ["Alice", "100"]]})
        result = OdfExtractor().extract(data)
        assert "Sales" in result
        assert "Alice" in result
        assert "100" in result

    def test_ods_multiple_sheets(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        data = _make_ods({
            "Sheet1": [["a", "b"]],
            "Sheet2": [["c", "d"]],
        })
        result = OdfExtractor().extract(data)
        assert "Sheet1" in result
        assert "Sheet2" in result

    def test_odp_slide_text(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        data = _make_odp([["Title of slide", "Bullet point"]])
        result = OdfExtractor().extract(data)
        assert "Slide 1" in result

    def test_can_handle_odt(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        assert OdfExtractor().can_handle("odt")

    def test_can_handle_ods(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        assert OdfExtractor().can_handle("ods")

    def test_can_handle_odp(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        assert OdfExtractor().can_handle("odp")

    def test_cannot_handle_docx(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        assert not OdfExtractor().can_handle("docx")

    def test_detect_odt(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        assert isinstance(detect_extractor("odt"), OdfExtractor)

    def test_detect_ods(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        assert isinstance(detect_extractor("ods"), OdfExtractor)

    def test_detect_odp(self):
        from chunkymonkey.extractors._odf import OdfExtractor
        assert isinstance(detect_extractor("odp"), OdfExtractor)


# =============================================================================
# EmailExtractor
# =============================================================================

def _make_email(
    subject: str = "Test Subject",
    from_: str = "alice@example.com",
    to: str = "bob@example.com",
    body: str = "Hello, world.",
    body_type: str = "plain",
    attachments: list[tuple[str, bytes, str]] | None = None,
) -> bytes:
    """Build a minimal RFC 5322 email in memory using stdlib email."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    if attachments:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = from_
        msg["To"] = to
        msg["Date"] = "Mon, 14 Apr 2025 10:00:00 +0000"
        msg.attach(MIMEText(body, body_type))
        for filename, data, mime_type in attachments:
            maintype, subtype = mime_type.split("/", 1)
            part = MIMEBase(maintype, subtype)
            part.set_payload(data)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename=filename)
            msg.attach(part)
    else:
        msg = MIMEText(body, body_type)
        msg["Subject"] = subject
        msg["From"] = from_
        msg["To"] = to
        msg["Date"] = "Mon, 14 Apr 2025 10:00:00 +0000"

    return msg.as_bytes()


class TestEmailExtractor:
    def test_extracts_body(self):
        from chunkymonkey.extractors._email import EmailExtractor
        data = _make_email(body="This is the email body.")
        result = EmailExtractor().extract(data)
        assert "This is the email body." in result

    def test_extracts_subject_header(self):
        from chunkymonkey.extractors._email import EmailExtractor
        data = _make_email(subject="Weekly Report")
        result = EmailExtractor().extract(data)
        assert "Subject: Weekly Report" in result

    def test_extracts_from_header(self):
        from chunkymonkey.extractors._email import EmailExtractor
        data = _make_email(from_="sender@example.com")
        result = EmailExtractor().extract(data)
        assert "From:" in result
        assert "sender@example.com" in result

    def test_extracts_to_header(self):
        from chunkymonkey.extractors._email import EmailExtractor
        data = _make_email(to="recipient@example.com")
        result = EmailExtractor().extract(data)
        assert "To:" in result
        assert "recipient@example.com" in result

    def test_html_body_stripped(self):
        from chunkymonkey.extractors._email import EmailExtractor
        data = _make_email(body="<h1>Hello</h1><p>World</p>", body_type="html")
        result = EmailExtractor().extract(data)
        assert "<h1>" not in result
        assert "Hello" in result
        assert "World" in result

    def test_can_handle_email(self):
        from chunkymonkey.extractors._email import EmailExtractor
        assert EmailExtractor().can_handle("email")

    def test_can_handle_eml(self):
        from chunkymonkey.extractors._email import EmailExtractor
        assert EmailExtractor().can_handle("eml")

    def test_cannot_handle_text(self):
        from chunkymonkey.extractors._email import EmailExtractor
        assert not EmailExtractor().can_handle("text")

    def test_cannot_handle_html(self):
        from chunkymonkey.extractors._email import EmailExtractor
        assert not EmailExtractor().can_handle("html")

    def test_no_attachments_by_default(self):
        from chunkymonkey.extractors._email import EmailExtractor
        attachment_data = b"Name,Score\nAlice,90"
        data = _make_email(
            body="See attached.",
            attachments=[("results.csv", attachment_data, "text/csv")],
        )
        result = EmailExtractor(include_attachments=False).extract(data)
        assert "[Attachment:" not in result

    def test_includes_attachments_when_enabled(self):
        from chunkymonkey.extractors._email import EmailExtractor
        attachment_data = b"Name,Score\nAlice,90"
        data = _make_email(
            body="See attached.",
            attachments=[("results.csv", attachment_data, "text/csv")],
        )
        result = EmailExtractor(include_attachments=True).extract(data)
        assert "[Attachment: results.csv]" in result
        assert "Alice" in result

    def test_detect_email(self):
        from chunkymonkey.extractors._email import EmailExtractor
        assert isinstance(detect_extractor("email"), EmailExtractor)

    def test_detect_eml(self):
        from chunkymonkey.extractors._email import EmailExtractor
        assert isinstance(detect_extractor("eml"), EmailExtractor)
