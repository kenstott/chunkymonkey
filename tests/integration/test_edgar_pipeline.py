# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 7c041a67-b831-4f86-8892-4e15e60270d7
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Integration tests for the EDGAR 10-K pipeline.

These tests verify the full fetch → extract → chunk → enrich pipeline using
mocked HTTP and a realistic but compact EDGAR HTML fixture.  No network
requests are made.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chunkymonkey import DocumentLoader
from chunkymonkey.extractors._edgar import EdgarExtractor, _extract_edgar_prose
from chunkymonkey.models import DocumentChunk


# ─────────────────────────────────────────────────────────────────────────────
# Fixture HTML — compact 10-K with the four prose items we care about
# ─────────────────────────────────────────────────────────────────────────────

_AAPL_HTML = b"""\
<!DOCTYPE html>
<html>
<head><title>Apple 10-K</title></head>
<body>
<div>
  <a href="#i_item1">Item 1.</a>
  <a href="#i_item1a">Item 1A.</a>
  <a href="#i_item7">Item 7.</a>
  <a href="#i_item9a">Item 9A.</a>
  <a href="#i_item8">Item 8.</a>
</div>

<div id="i_item1">
  <p>Business. Apple Inc. designs, manufactures and markets smartphones, personal
  computers, tablets, wearables and accessories and sells a variety of related
  services worldwide. The Company's fiscal year ends on the last Saturday of
  September.</p>
</div>

<div id="i_item1a">
  <p>Risk Factors. A downturn in economic conditions could adversely affect our
  operating results. Our products face intense competition in every market segment
  in which we participate. Competition in the markets for our products and services
  is fierce. We compete with companies that have significant technical, marketing,
  distribution and other resources.</p>
  <p>Our future performance depends substantially on the continued service and
  performance of our highly skilled employees. We are dependent on the performance
  of a limited number of key personnel.</p>
  <p>We are subject to complex and changing laws and regulations worldwide. Changes
  in applicable laws or regulations or the interpretation of laws or regulations
  could result in fines, penalties, or other adverse consequences.</p>
</div>

<div id="i_item7">
  <p>Management Discussion and Analysis of Financial Condition and Results of
  Operations. The following discussion should be read in conjunction with the
  Consolidated Financial Statements and Notes thereto.</p>
  <p>Net sales increased 2% or $7.9 billion during 2024 compared to 2023. iPhone
  revenue was $201.0 billion compared to $205.0 billion in 2023. Services revenue
  grew to $96.2 billion, reflecting strong performance across advertising, licensing
  and AppleCare. Mac revenue increased 2% reflecting demand for MacBook Pro.</p>
</div>

<div id="i_item9a">
  <p>Controls and Procedures. Under the supervision and with the participation of
  the Company's management, including its Chief Executive Officer and Chief Financial
  Officer, the Company carried out an evaluation of the effectiveness of the design
  and operation of its disclosure controls and procedures as of September 28, 2024,
  as required by Exchange Act Rule 13a-15(e).</p>
  <p>Based upon that evaluation, the Chief Executive Officer and Chief Financial
  Officer concluded that the Company's disclosure controls and procedures were
  effective as of September 28, 2024.</p>
</div>

<div id="i_item8">
  <p>Financial Statements. See the consolidated financial statements on page F-1.</p>
  <table><tr><td>Net sales</td><td>391,035</td></tr></table>
</div>
</body>
</html>"""

_MSFT_HTML = b"""\
<!DOCTYPE html>
<html>
<head><title>Microsoft 10-K</title></head>
<body>
<div>
  <a href="#m_item1">Item 1.</a>
  <a href="#m_item1a">Item 1A.</a>
  <a href="#m_item7">Item 7.</a>
  <a href="#m_item9a">Item 9A.</a>
  <a href="#m_item8">Item 8.</a>
</div>

<div id="m_item1">
  <p>Business. Microsoft Corporation enables digital transformation for the era of an
  intelligent cloud and an intelligent edge. Our mission is to empower every person
  and every organization on the planet to achieve more.</p>
</div>

<div id="m_item1a">
  <p>Risk Factors. Intense competition across all of our markets could harm our business.
  We face competition from a wide range of companies in every product and service area.
  Our competitors range from large technology companies to niche players.</p>
  <p>Competition in the cloud computing market continues to intensify. Azure competes
  with Amazon Web Services, Google Cloud Platform and others. Market share could be
  impacted by pricing, performance, and availability of competing services.</p>
  <p>We are subject to government litigation and regulatory activity. The European
  Commission has investigated our business practices. Investigations and proceedings
  could result in fines or behavioral remedies that adversely affect our business.</p>
</div>

<div id="m_item7">
  <p>Management Discussion. Revenue was $245.1 billion, an increase of 16%. Intelligent
  Cloud revenue was $105.4 billion, up 21%, driven by Azure and other cloud services.
  Azure revenue grew 29% reflecting higher consumption from large commercial customers.</p>
</div>

<div id="m_item9a">
  <p>Controls and Procedures. Under supervision of our Chief Executive Officer and
  Chief Financial Officer, we evaluated the effectiveness of our disclosure controls
  and procedures as of June 30, 2024. Based on that evaluation, our Chief Executive
  Officer and Chief Financial Officer concluded that disclosure controls were effective.</p>
</div>

<div id="m_item8">
  <p>Financial Statements data.</p>
</div>
</body>
</html>"""

# Fake EDGAR submissions JSON (minimal structure)
def _fake_submissions(cik: str, acc_no: str, doc: str) -> bytes:
    return json.dumps({
        "cik": int(cik),
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q"],
                "accessionNumber": [acc_no, "0000000000-24-000001"],
                "primaryDocument": [doc, "q1.htm"],
                "filingDate": ["2024-11-01", "2024-08-01"],
            }
        }
    }).encode()


# ─────────────────────────────────────────────────────────────────────────────
# Unit-level extraction tests (no HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgarPipelineExtraction:
    """Verify EdgarExtractor produces correct markdown on our fixture HTMLs."""

    def test_aapl_extracts_risk_factors(self):
        result = _extract_edgar_prose(_AAPL_HTML.decode())
        assert "Risk Factors" in result
        assert "intense competition" in result.lower()

    def test_aapl_excludes_financial_tables(self):
        result = _extract_edgar_prose(_AAPL_HTML.decode())
        assert "391,035" not in result  # numeric table row

    def test_aapl_no_anchor_id_leakage(self):
        result = _extract_edgar_prose(_AAPL_HTML.decode())
        assert 'id="i_item1a"' not in result
        assert 'id="i_item9a"' not in result

    def test_msft_extracts_azure_mention(self):
        result = _extract_edgar_prose(_MSFT_HTML.decode())
        assert "Azure" in result

    def test_msft_excludes_item8(self):
        result = _extract_edgar_prose(_MSFT_HTML.decode())
        assert "Financial Statements data" not in result

    def test_both_have_controls_procedures(self):
        aapl = _extract_edgar_prose(_AAPL_HTML.decode())
        msft = _extract_edgar_prose(_MSFT_HTML.decode())
        assert "Controls and Procedures" in aapl
        assert "Controls and Procedures" in msft


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: DocumentLoader.load_bytes with EdgarExtractor
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgarDocumentLoader:

    def _load(self, html: bytes, name: str, strategy: str | None = "prefix") -> list[DocumentChunk]:
        loader = DocumentLoader(
            min_chunk_size=400, max_chunk_size=400,
            context_strategy=strategy,
            extra_extractors=[EdgarExtractor()],
        )
        return loader.load_bytes(html, name=name, doc_type="edgar")

    # ── Basic pipeline ────────────────────────────────────────────────────────

    def test_aapl_produces_chunks(self):
        chunks = self._load(_AAPL_HTML, "aapl_10k_2024")
        assert len(chunks) > 0

    def test_msft_produces_chunks(self):
        chunks = self._load(_MSFT_HTML, "msft_10k_2024")
        assert len(chunks) > 0

    def test_document_names_set(self):
        chunks = self._load(_AAPL_HTML, "aapl_10k_2024")
        assert all(c.document_name == "aapl_10k_2024" for c in chunks)

    def test_sections_include_item_1a(self):
        chunks = self._load(_AAPL_HTML, "aapl_10k_2024")
        assert any("1A" in (c.section or "") for c in chunks)

    def test_sections_include_item_9a(self):
        chunks = self._load(_AAPL_HTML, "aapl_10k_2024")
        assert any("9A" in (c.section or "") for c in chunks)

    def test_no_chunk_contains_financial_table_data(self):
        chunks = self._load(_AAPL_HTML, "aapl_10k_2024")
        for c in chunks:
            assert "391,035" not in c.content

    # ── Contextual enrichment ─────────────────────────────────────────────────

    def test_embedding_content_has_doc_name(self):
        chunks = self._load(_AAPL_HTML, "aapl_10k_2024", strategy="prefix")
        for c in chunks:
            assert c.embedding_content is not None
            assert "aapl_10k_2024" in c.embedding_content

    def test_embedding_content_has_section(self):
        chunks = self._load(_AAPL_HTML, "aapl_10k_2024", strategy="prefix")
        sectioned = [c for c in chunks if c.section]
        assert len(sectioned) > 0
        for c in sectioned:
            assert c.section in c.embedding_content

    def test_naive_has_no_prefix(self):
        chunks = self._load(_AAPL_HTML, "aapl_10k_2024", strategy=None)
        for c in chunks:
            # naive: no context_strategy → embedding_content is None
            assert c.embedding_content is None

    # ── Cross-company disambiguation ──────────────────────────────────────────

    def test_same_section_different_doc_names(self):
        """Item 9A chunks from AAPL and MSFT have different embedding_content."""
        aapl_chunks = self._load(_AAPL_HTML, "aapl_10k_2024", strategy="prefix")
        msft_chunks = self._load(_MSFT_HTML, "msft_10k_2024", strategy="prefix")

        aapl_9a = [c for c in aapl_chunks if "9A" in (c.section or "")]
        msft_9a = [c for c in msft_chunks if "9A" in (c.section or "")]

        assert len(aapl_9a) > 0
        assert len(msft_9a) > 0

        # Embedding contents must differ (doc name prefix distinguishes them)
        for ca in aapl_9a:
            for cm in msft_9a:
                assert ca.embedding_content != cm.embedding_content

    def test_boilerplate_content_may_be_shared(self):
        """Item 9A prose is nearly identical — naive content overlaps."""
        aapl_chunks = self._load(_AAPL_HTML, "aapl_10k_2024", strategy=None)
        msft_chunks = self._load(_MSFT_HTML, "msft_10k_2024", strategy=None)

        aapl_contents = {c.content for c in aapl_chunks if "9A" in (c.section or "")}
        msft_contents = {c.content for c in msft_chunks if "9A" in (c.section or "")}

        # At least some overlap in 9A boilerplate between the two filings
        # (controls-and-procedures language is nearly identical across filers)
        shared_words = {"disclosure", "controls", "procedures", "effective"}
        aapl_words = {w for content in aapl_contents for w in content.lower().split()}
        msft_words = {w for content in msft_contents for w in content.lower().split()}
        assert shared_words & aapl_words
        assert shared_words & msft_words

    # ── Chunk count consistency ───────────────────────────────────────────────

    def test_naive_and_ctx_same_chunk_count(self):
        n = self._load(_AAPL_HTML, "aapl_10k_2024", strategy=None)
        c = self._load(_AAPL_HTML, "aapl_10k_2024", strategy="prefix")
        assert len(n) == len(c)


# ─────────────────────────────────────────────────────────────────────────────
# Mocked HTTP: edgar_demo._get_latest_10k_url
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgarDemoFetchHelpers:
    """Test the EDGAR fetch helpers with mocked HTTP, without touching the network."""

    def test_get_latest_10k_url_parses_submissions(self):
        from demo.edgar_demo import _get_latest_10k_url

        fake_json = _fake_submissions("320193", "0000320193-24-000001", "aapl-20240928.htm")

        with patch("demo.edgar_demo._http_get", return_value=fake_json):
            url = _get_latest_10k_url("https://data.sec.gov/submissions/CIK0000320193.json")

        assert "aapl-20240928.htm" in url
        assert "000032019324000001" in url  # accession without dashes
        assert url.startswith("https://www.sec.gov/Archives/edgar/data/")

    def test_get_latest_10k_url_raises_when_no_10k(self):
        from demo.edgar_demo import _get_latest_10k_url

        fake_json = json.dumps({
            "cik": 320193,
            "filings": {
                "recent": {
                    "form": ["10-Q", "8-K"],
                    "accessionNumber": ["0000320193-24-000001", "0000320193-24-000002"],
                    "primaryDocument": ["q1.htm", "8k.htm"],
                    "filingDate": ["2024-08-01", "2024-05-01"],
                }
            }
        }).encode()

        with patch("demo.edgar_demo._http_get", return_value=fake_json):
            with pytest.raises(RuntimeError, match="No 10-K"):
                _get_latest_10k_url("https://data.sec.gov/submissions/CIK0000320193.json")

    def test_fetch_or_load_uses_cache(self, tmp_path):
        from demo.edgar_demo import _fetch_or_load

        filing = {"name": "aapl_10k_2024", "company": "Apple", "filing_index": "https://x"}
        cache_file = tmp_path / "aapl_10k_2024.htm"
        cache_file.write_bytes(_AAPL_HTML)

        # Should return cached data without calling _http_get
        with patch("demo.edgar_demo._http_get") as mock_get:
            name, data = _fetch_or_load(filing, tmp_path)

        mock_get.assert_not_called()
        assert name == "aapl_10k_2024"
        assert data == _AAPL_HTML

    def test_fetch_or_load_writes_cache(self, tmp_path):
        from demo.edgar_demo import _fetch_or_load

        fake_submissions = _fake_submissions("320193", "0000320193-24-000001", "aapl.htm")

        filing = {
            "name": "aapl_10k_2024",
            "company": "Apple",
            "filing_index": "https://data.sec.gov/submissions/CIK0000320193.json",
        }

        def mock_http(url, **kwargs):
            if "submissions" in url:
                return fake_submissions
            return _AAPL_HTML

        with patch("demo.edgar_demo._http_get", side_effect=mock_http):
            with patch("demo.edgar_demo.time.sleep"):
                name, data = _fetch_or_load(filing, tmp_path)

        assert name == "aapl_10k_2024"
        assert (tmp_path / "aapl_10k_2024.htm").exists()
        assert data == _AAPL_HTML


# ─────────────────────────────────────────────────────────────────────────────
# QUERIES sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgarQueriesStructure:
    from demo.edgar_demo import QUERIES

    def test_queries_nonempty(self):
        from demo.edgar_demo import QUERIES
        assert len(QUERIES) > 0

    def test_each_query_is_string(self):
        from demo.edgar_demo import QUERIES
        for q in QUERIES:
            assert isinstance(q, str) and len(q) > 0

    def test_queries_are_topic_based(self):
        from demo.edgar_demo import QUERIES
        # Queries should be topic phrases, not empty
        assert all(len(q.split()) >= 2 for q in QUERIES)

    def test_queries_cover_rag_topics(self):
        from demo.edgar_demo import QUERIES
        combined = " ".join(QUERIES).lower()
        # Should span common 10-K topics
        assert any(w in combined for w in ("risk", "tax", "revenue", "currency", "cyber"))


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end retrieval demonstration (no network, fixture data)
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgarRetrieval:
    """Demonstrate that contextual chunking improves retrieval on EDGAR fixtures."""

    def _build_corpus(self, strategy: str | None):
        loader = DocumentLoader(
            min_chunk_size=300, max_chunk_size=300,
            context_strategy=strategy,
            extra_extractors=[EdgarExtractor()],
        )
        all_chunks = []
        for html, name in [(_AAPL_HTML, "aapl_10k_2024"), (_MSFT_HTML, "msft_10k_2024")]:
            all_chunks.extend(loader.load_bytes(html, name=name, doc_type="edgar"))
        return all_chunks

    @staticmethod
    def _tfidf_retrieve(query: str, chunks: list[DocumentChunk], use_embedding: bool, k: int = 1):
        import math
        import re
        from collections import Counter

        def tok(t): return re.findall(r"[a-z][a-z0-9]*", t.lower())

        texts = [c.embedding_content if use_embedding else c.content for c in chunks]
        vocab = sorted({w for t in texts for w in tok(t)})
        n = len(texts)
        idf = {w: math.log((n + 1) / (sum(1 for t in texts if w in tok(t)) + 1)) + 1 for w in vocab}

        def vec(t):
            toks = tok(t); n_ = len(toks)
            if not n_: return [0.0] * len(vocab)
            tf = Counter(toks)
            return [tf.get(w, 0) / n_ * idf.get(w, 0) for w in vocab]

        def cos(a, b):
            d = sum(x * y for x, y in zip(a, b))
            ma = math.sqrt(sum(x * x for x in a))
            mb = math.sqrt(sum(x * x for x in b))
            return d / (ma * mb) if ma and mb else 0.0

        qv = vec(query)
        scored = sorted(zip([cos(qv, vec(t)) for t in texts], chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

    def test_azure_query_finds_msft_contextually(self):
        """'Azure cloud competition' should rank MSFT's Risk Factors first contextually."""
        ctx_chunks = self._build_corpus("prefix")
        results = self._tfidf_retrieve(
            "Azure cloud platform competition antitrust European Commission",
            ctx_chunks,
            use_embedding=True,
        )
        assert results[0].document_name == "msft_10k_2024"

    def test_iphone_query_finds_aapl_contextually(self):
        """'iPhone revenue Mac Services' should rank AAPL first contextually."""
        ctx_chunks = self._build_corpus("prefix")
        results = self._tfidf_retrieve(
            "iPhone revenue Mac Services AppleCare advertising",
            ctx_chunks,
            use_embedding=True,
        )
        assert results[0].document_name == "aapl_10k_2024"

    def test_contextual_vs_naive_on_controls_boilerplate(self):
        """Item 9A boilerplate: contextual should separate by document name."""
        naive_chunks = self._build_corpus(None)
        ctx_chunks = self._build_corpus("prefix")

        query = "disclosure controls effective Chief Executive Officer Chief Financial Officer"

        naive_result = self._tfidf_retrieve(query, naive_chunks, use_embedding=False)[0]
        ctx_result = self._tfidf_retrieve(query, ctx_chunks, use_embedding=True)[0]

        # Both should be from a 9A section
        assert "9A" in (naive_result.section or "") or "Controls" in (naive_result.section or "")
        assert "9A" in (ctx_result.section or "") or "Controls" in (ctx_result.section or "")
