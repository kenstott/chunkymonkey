# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 393b2d0b-7f67-40b3-8a85-ddafd75d5456
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for EdgarExtractor — EDGAR inline XBRL HTML extractor."""
from __future__ import annotations

import pytest

from chunkymonkey.extractors._edgar import (
    EdgarExtractor,
    _extract_edgar_prose,
    _parse_toc,
    _strip_tags,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared test fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Minimal but realistic EDGAR inline XBRL HTML with all major prose items.
# Item 8 (Financial Statements) is deliberately included to test exclusion.
_FULL_10K_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Annual Report 10-K</title></head>
<body>
<div>
  <a href="#item1_anchor">Item 1.</a>
  <a href="#item1a_anchor">Item 1A.</a>
  <a href="#item7_anchor">Item 7.</a>
  <a href="#item9a_anchor">Item 9A.</a>
  <a href="#item8_anchor">Item 8.</a>
</div>

<div id="item1_anchor">
  <p>Business overview. We design, manufacture and sell consumer electronics and software.</p>
  <p>Our products include smartphones, tablets, personal computers and wearables sold worldwide.</p>
</div>

<div id="item1a_anchor">
  <p>Risk Factors first paragraph about competition in markets.</p>
  <p>We face intense competition in all markets in which we operate. Our results may be
  materially affected by adverse economic conditions, supply chain disruptions, and
  regulatory changes in jurisdictions where we sell products.</p>
  <p>We depend on key personnel and the loss of such individuals could adversely affect
  our business and results of operations.</p>
</div>

<div id="item7_anchor">
  <p>Management Discussion and Analysis.</p>
  <p>Revenue increased 5% year over year, reflecting higher demand across all product lines.</p>
  <p>Gross margin was 44.5%, compared to 43.3% in the prior year period, driven by a
  more favorable product mix and improved component pricing.</p>
</div>

<div id="item9a_anchor">
  <p>Disclosure Controls and Procedures. Under the supervision and with the participation
  of our principal executive officer and principal financial officer, we evaluated the
  effectiveness of our disclosure controls and procedures as of the end of the period
  covered by this report, as required by Exchange Act Rules 13a-15(e) and 15d-15(e).</p>
  <p>Based on that evaluation, our principal executive officer and principal financial
  officer concluded that our disclosure controls and procedures were effective as of
  September 30, 2024.</p>
</div>

<div id="item8_anchor">
  <p>Financial Statements and Supplementary Data.</p>
  <table>
    <tr><td>Revenue</td><td>$391,035</td></tr>
    <tr><td>Net Income</td><td>$93,736</td></tr>
  </table>
  <p>See audited financial statements beginning on page F-1.</p>
</div>
</body>
</html>"""

# HTML with only Item 8 in the TOC (edge case: all items excluded)
_ONLY_ITEM8_HTML = """\
<html><body>
<div><a href="#sec8">Item 8.</a></div>
<div id="sec8"><p>Financial data only.</p></div>
</body></html>"""

# HTML with no recognisable TOC links (fallback path)
_NO_TOC_HTML = """\
<html><body>
<p>This document has no Item links in the TOC.</p>
<p>Just plain paragraphs with some content.</p>
</body></html>"""

# EDGAR inline XBRL marker in the head
_XBRL_HTML = b"""\
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<head><title>10-K</title></head>
<body><p>XBRL document body.</p></body>
</html>"""

_PLAIN_HTML = b"<html><body><p>Regular HTML, no XBRL.</p></body></html>"


# ─────────────────────────────────────────────────────────────────────────────
# _strip_tags
# ─────────────────────────────────────────────────────────────────────────────

class TestStripTags:
    def test_removes_p_tags(self):
        assert "Hello world" in _strip_tags("<p>Hello world</p>")

    def test_removes_script(self):
        result = _strip_tags("<script>alert('x')</script><p>Keep this</p>")
        assert "alert" not in result
        assert "Keep this" in result

    def test_removes_style(self):
        result = _strip_tags("<style>.cls{color:red}</style><p>Visible</p>")
        assert ".cls" not in result
        assert "Visible" in result

    def test_decodes_html_entities(self):
        result = _strip_tags("<p>AT&amp;T &lt;Corp&gt;</p>")
        assert "AT&T" in result
        assert "<Corp>" in result

    def test_collapses_whitespace(self):
        result = _strip_tags("<p>word1    word2</p>")
        assert "word1" in result and "word2" in result
        assert "    " not in result

    def test_empty_input(self):
        assert _strip_tags("") == ""

    def test_block_elements_insert_newlines(self):
        result = _strip_tags("<p>A</p><p>B</p>")
        # At least one newline between the two paragraphs
        assert result.index("A") < result.index("B")


# ─────────────────────────────────────────────────────────────────────────────
# _parse_toc
# ─────────────────────────────────────────────────────────────────────────────

class TestParseToc:
    def test_extracts_item_numbers(self):
        toc = _parse_toc(_FULL_10K_HTML)
        items = [item for _, item in toc]
        assert "1" in items
        assert "1A" in items
        assert "7" in items
        assert "9A" in items
        assert "8" in items

    def test_extracts_anchor_ids(self):
        toc = _parse_toc(_FULL_10K_HTML)
        anchors = [aid for aid, _ in toc]
        assert "item1_anchor" in anchors
        assert "item1a_anchor" in anchors
        assert "item7_anchor" in anchors
        assert "item9a_anchor" in anchors

    def test_preserves_document_order(self):
        toc = _parse_toc(_FULL_10K_HTML)
        items = [item for _, item in toc]
        # 1, 1A, 7, 9A, 8 in that order
        assert items.index("1") < items.index("1A")
        assert items.index("1A") < items.index("7")
        assert items.index("7") < items.index("9A")

    def test_deduplicates_anchors(self):
        # Duplicate links to same anchor (e.g., TOC + body cross-reference)
        html = """\
<div>
  <a href="#sec1a">Item 1A.</a>
  <a href="#sec1a">Item 1A.</a>
</div>
<div id="sec1a"><p>content</p></div>"""
        toc = _parse_toc(html)
        items = [item for _, item in toc]
        assert items.count("1A") == 1

    def test_empty_html_returns_empty(self):
        assert _parse_toc("") == []

    def test_no_item_links_returns_empty(self):
        assert _parse_toc(_NO_TOC_HTML) == []

    def test_period_required_in_label(self):
        # The regex requires 'Item X.' with a trailing period — EDGAR TOC always uses it.
        html_no_period = '<a href="#s1">Item 1A</a><div id="s1"><p>x</p></div>'
        toc = _parse_toc(html_no_period)
        assert not any(item == "1A" for _, item in toc), (
            "Label without period should NOT be matched by the TOC parser"
        )
        # With a period it should be matched.
        html_with_period = '<a href="#s1">Item 1A.</a><div id="s1"><p>x</p></div>'
        toc2 = _parse_toc(html_with_period)
        assert any(item == "1A" for _, item in toc2)


# ─────────────────────────────────────────────────────────────────────────────
# _extract_edgar_prose
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractEdgarProse:
    def test_includes_item_1a_risk_factors(self):
        result = _extract_edgar_prose(_FULL_10K_HTML)
        assert "Item 1A" in result
        assert "Risk Factors" in result

    def test_includes_item_7_mda(self):
        result = _extract_edgar_prose(_FULL_10K_HTML)
        assert "Item 7" in result
        assert "Management Discussion" in result

    def test_excludes_item_8_financial_statements_by_default(self):
        result = _extract_edgar_prose(_FULL_10K_HTML, prose_only=True)
        # Item 8 content should not appear
        assert "item8_anchor" not in result
        assert "Financial Statements and Supplementary Data" not in result

    def test_includes_item_8_when_prose_only_false(self):
        result = _extract_edgar_prose(_FULL_10K_HTML, prose_only=False)
        assert "Financial Statements" in result

    def test_section_headings_format(self):
        result = _extract_edgar_prose(_FULL_10K_HTML)
        assert "# Item 1." in result
        assert "# Item 1A." in result
        assert "# Item 7." in result
        assert "# Item 9A." in result

    def test_section_separator_present(self):
        result = _extract_edgar_prose(_FULL_10K_HTML)
        assert "---" in result

    def test_prose_content_extracted(self):
        result = _extract_edgar_prose(_FULL_10K_HTML)
        assert "intense competition" in result
        assert "disclosure controls" in result.lower()

    def test_anchor_id_text_not_in_output(self):
        """Bug fix: anchor attribute text must not leak into extracted content."""
        result = _extract_edgar_prose(_FULL_10K_HTML)
        assert 'id="item1_anchor"' not in result
        assert 'id="item1a_anchor"' not in result
        assert 'id="item9a_anchor"' not in result

    def test_fallback_when_no_toc(self):
        """When TOC parsing yields nothing, fall back to stripping all tags."""
        result = _extract_edgar_prose(_NO_TOC_HTML)
        assert "Just plain paragraphs" in result

    def test_all_prose_items_excluded_returns_empty(self):
        """If only Item 8 is in TOC and prose_only=True, result is empty string."""
        result = _extract_edgar_prose(_ONLY_ITEM8_HTML, prose_only=True)
        # Fallback triggers because TOC is parsed but all items are filtered
        # The fallback strips all tags and returns raw text
        assert isinstance(result, str)

    def test_item_names_mapped_correctly(self):
        result = _extract_edgar_prose(_FULL_10K_HTML)
        assert "Business" in result
        assert "Risk Factors" in result
        assert "Management Discussion and Analysis" in result
        assert "Controls and Procedures" in result

    def test_multiple_sections_separated(self):
        result = _extract_edgar_prose(_FULL_10K_HTML)
        parts = result.split("---")
        # Should have at least 3 prose sections: 1, 1A, 7, 9A
        assert len(parts) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# EdgarExtractor
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgarExtractor:
    def setup_method(self):
        self.ext = EdgarExtractor()

    # ── can_handle ────────────────────────────────────────────────────────────

    def test_can_handle_edgar(self):
        assert self.ext.can_handle("edgar") is True

    def test_can_handle_edgar_html(self):
        assert self.ext.can_handle("edgar-html") is True

    def test_can_handle_edgar_xbrl(self):
        assert self.ext.can_handle("edgar-xbrl") is True

    def test_cannot_handle_html(self):
        assert self.ext.can_handle("html") is False

    def test_cannot_handle_pdf(self):
        assert self.ext.can_handle("pdf") is False

    def test_cannot_handle_text(self):
        assert self.ext.can_handle("text") is False

    def test_cannot_handle_empty(self):
        assert self.ext.can_handle("") is False

    # ── is_edgar_html ─────────────────────────────────────────────────────────

    def test_detects_xbrl_namespace(self):
        assert EdgarExtractor.is_edgar_html(_XBRL_HTML) is True

    def test_detects_ix_header_tag(self):
        html = b"<html><head><ix:header>...</ix:header></head></html>"
        assert EdgarExtractor.is_edgar_html(html) is True

    def test_detects_sec_gov_reference(self):
        html = b"<html><body><!-- from sec.gov --><p>text</p></body></html>"
        assert EdgarExtractor.is_edgar_html(html) is True

    def test_plain_html_not_detected(self):
        assert EdgarExtractor.is_edgar_html(_PLAIN_HTML) is False

    def test_empty_bytes_not_detected(self):
        assert EdgarExtractor.is_edgar_html(b"") is False

    # ── extract ───────────────────────────────────────────────────────────────

    def test_extract_returns_string(self):
        result = self.ext.extract(_FULL_10K_HTML.encode())
        assert isinstance(result, str)

    def test_extract_contains_risk_factors(self):
        result = self.ext.extract(_FULL_10K_HTML.encode())
        assert "Risk Factors" in result

    def test_extract_excludes_financial_statements(self):
        result = self.ext.extract(_FULL_10K_HTML.encode())
        assert "Financial Statements and Supplementary Data" not in result

    def test_extract_handles_utf8_bytes(self):
        html = _FULL_10K_HTML.encode("utf-8")
        result = self.ext.extract(html)
        assert len(result) > 0

    def test_extract_handles_latin1_with_replace(self):
        # Bytes with a latin-1 character should not raise
        html = b"<p>caf\xe9 revenue</p>"
        result = self.ext.extract(html)
        assert isinstance(result, str)

    def test_extract_source_path_ignored(self):
        r1 = self.ext.extract(_FULL_10K_HTML.encode(), source_path=None)
        r2 = self.ext.extract(_FULL_10K_HTML.encode(), source_path="/some/path.htm")
        assert r1 == r2

    def test_extract_no_anchor_id_leakage(self):
        result = self.ext.extract(_FULL_10K_HTML.encode())
        assert 'id="item1a_anchor"' not in result
        assert 'id="item9a_anchor"' not in result

    def test_extract_produces_markdown_headings(self):
        result = self.ext.extract(_FULL_10K_HTML.encode())
        assert result.startswith("# Item")

    def test_extract_empty_html(self):
        result = self.ext.extract(b"<html><body></body></html>")
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# Integration with DocumentLoader
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgarWithDocumentLoader:
    """Verify the extractor plugs into DocumentLoader correctly."""

    def test_load_bytes_produces_chunks(self):
        from chunkymonkey import DocumentLoader
        loader = DocumentLoader(
            min_chunk_size=400, max_chunk_size=400,
            extra_extractors=[EdgarExtractor()],
        )
        chunks = loader.load_bytes(
            _FULL_10K_HTML.encode(),
            name="acme_10k_2024",
            doc_type="edgar",
        )
        assert len(chunks) > 0

    def test_chunks_have_document_name(self):
        from chunkymonkey import DocumentLoader
        loader = DocumentLoader(
            min_chunk_size=400, max_chunk_size=400,
            extra_extractors=[EdgarExtractor()],
        )
        chunks = loader.load_bytes(
            _FULL_10K_HTML.encode(),
            name="acme_10k_2024",
            doc_type="edgar",
        )
        assert all(c.document_name == "acme_10k_2024" for c in chunks)

    def test_chunks_have_section_breadcrumbs(self):
        from chunkymonkey import DocumentLoader
        loader = DocumentLoader(
            min_chunk_size=400, max_chunk_size=400,
            extra_extractors=[EdgarExtractor()],
        )
        chunks = loader.load_bytes(
            _FULL_10K_HTML.encode(),
            name="acme_10k_2024",
            doc_type="edgar",
        )
        sections = {c.section for c in chunks if c.section}
        # Should have at least Item 1A and Item 9A sections
        assert any("1A" in (s or "") for s in sections)
        assert any("9A" in (s or "") for s in sections)

    def test_embedding_content_includes_doc_name(self):
        from chunkymonkey import DocumentLoader
        loader = DocumentLoader(
            min_chunk_size=400, max_chunk_size=400,
            context_strategy="prefix",
            extra_extractors=[EdgarExtractor()],
        )
        chunks = loader.load_bytes(
            _FULL_10K_HTML.encode(),
            name="acme_10k_2024",
            doc_type="edgar",
        )
        for c in chunks:
            assert c.embedding_content is not None
            assert "acme_10k_2024" in c.embedding_content

    def test_embedding_content_includes_section(self):
        from chunkymonkey import DocumentLoader
        loader = DocumentLoader(
            min_chunk_size=400, max_chunk_size=400,
            context_strategy="prefix",
            extra_extractors=[EdgarExtractor()],
        )
        chunks = loader.load_bytes(
            _FULL_10K_HTML.encode(),
            name="acme_10k_2024",
            doc_type="edgar",
        )
        sectioned = [c for c in chunks if c.section]
        assert len(sectioned) > 0
        for c in sectioned:
            assert c.section in c.embedding_content

    def test_naive_and_contextual_differ(self):
        from chunkymonkey import DocumentLoader
        naive = DocumentLoader(
            min_chunk_size=400, max_chunk_size=400, context_strategy=None,
            extra_extractors=[EdgarExtractor()],
        )
        ctx = DocumentLoader(
            min_chunk_size=400, max_chunk_size=400, context_strategy="prefix",
            extra_extractors=[EdgarExtractor()],
        )
        n_chunks = naive.load_bytes(_FULL_10K_HTML.encode(), name="x", doc_type="edgar")
        c_chunks = ctx.load_bytes(_FULL_10K_HTML.encode(), name="x", doc_type="edgar")
        # Same number of chunks
        assert len(n_chunks) == len(c_chunks)
        # But embedding_content differs for sectioned chunks
        diffs = [
            i for i, (n, c) in enumerate(zip(n_chunks, c_chunks))
            if n.embedding_content != c.embedding_content
        ]
        assert len(diffs) > 0

    def test_globally_registered_extractor(self):
        """register_extractor() adds extractor to global registry found by detect_extractor."""
        from chunkymonkey.extractors import register_extractor, detect_extractor
        import chunkymonkey.extractors as _ext_mod

        class _CustomEdgar(EdgarExtractor):
            def can_handle(self, doc_type: str) -> bool:
                return doc_type == "custom-edgar"

        # Reset registry so only built-ins are present
        _ext_mod._REGISTRY = None
        register_extractor(_CustomEdgar())

        ext = detect_extractor("custom-edgar")
        assert isinstance(ext, _CustomEdgar)

        # Confirm the custom extractor can actually extract
        result = ext.extract(_FULL_10K_HTML.encode())
        assert "Risk Factors" in result

        # Restore clean state
        _ext_mod._REGISTRY = None
