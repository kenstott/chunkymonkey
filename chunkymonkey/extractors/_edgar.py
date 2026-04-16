# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 6867f7d9-7f36-462c-b2a9-6c425a1470cb
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""EDGAR inline XBRL extractor — extracts prose sections from SEC 10-K filings.

SEC EDGAR 10-K filings are served as inline XBRL HTML.  They do NOT use
standard <h1>–<h6> heading tags.  Instead section boundaries are marked
by anchor IDs referenced from a table-of-contents at the top of the document.

This extractor:
  1. Parses the TOC to discover section anchors in document order.
  2. Extracts raw text between consecutive anchor positions.
  3. Maps SEC Item numbers to descriptive names (mandated by SEC regulation).
  4. Skips financial-statement items (Item 8) — pure numerical tables that
     add noise to RAG corpora without contributing retrievable prose.
  5. Returns plain markdown with '# Item X.Y  Section Name' headings so
     that chunkymonkey's section-aware chunker creates proper breadcrumbs.

Usage (via DocumentLoader)::

    from chunkymonkey import DocumentLoader
    from chunkymonkey.extractors import EdgarExtractor

    loader = DocumentLoader(extra_extractors=[EdgarExtractor()])
    chunks = loader.load_bytes(html_bytes, name="aapl_10k_2025", doc_type="edgar")

Or register globally::

    from chunkymonkey.extractors import register_extractor, EdgarExtractor
    register_extractor(EdgarExtractor())
"""
from __future__ import annotations

import re
from html.parser import HTMLParser


# ── SEC-mandated Item → section name mapping ─────────────────────────────────
# These are fixed by SEC regulation S-K and do not change across filers.

_ITEM_NAMES: dict[str, str] = {
    "1":   "Business",
    "1A":  "Risk Factors",
    "1B":  "Unresolved Staff Comments",
    "1C":  "Cybersecurity",
    "2":   "Properties",
    "3":   "Legal Proceedings",
    "4":   "Mine Safety Disclosures",
    "5":   "Market Information",
    "6":   "Reserved",
    "7":   "Management Discussion and Analysis",
    "7A":  "Quantitative Qualitative Disclosures About Market Risk",
    "8":   "Financial Statements",
    "9":   "Disagreements with Accountants",
    "9A":  "Controls and Procedures",
    "9B":  "Other Information",
    "9C":  "Foreign Jurisdictions Disclosures",
    "10":  "Directors Officers Corporate Governance",
    "11":  "Executive Compensation",
    "12":  "Security Ownership",
    "13":  "Certain Relationships Related Transactions",
    "14":  "Principal Accountant Fees",
    "15":  "Exhibits Financial Statement Schedules",
}

# Items that contain primarily prose.  Items 8 and 10-15 are financial
# statements and governance tables — skip them.
_PROSE_ITEMS: frozenset[str] = frozenset({"1", "1A", "1B", "1C", "2", "3", "7", "7A", "9", "9A"})

# Regex to identify an Item number inside a TOC link label
_ITEM_RE = re.compile(
    r"Item\s+(1[ABC]?|[2-9][AB]?|1[0-5])\s*\.",
    re.IGNORECASE,
)


# ── HTML → plain text helpers ─────────────────────────────────────────────────

class _PlainTextParser(HTMLParser):
    """Strip all HTML tags, decode entities, insert paragraph breaks."""

    _BLOCK = frozenset(
        "p div section article li tr td th h1 h2 h3 h4 h5 h6 br hr".split()
    )
    _SKIP = frozenset("script style ix:header ix:nonfraction ix:nonnumeric".split())

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        t = tag.lower()
        if t in self._SKIP or t.startswith("ix:"):
            self._skip_depth += 1
        if self._skip_depth:
            return
        if t in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if t in self._SKIP or t.startswith("ix:"):
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str):
        if not self._skip_depth:
            self._parts.append(data)

    def result(self) -> str:
        text = "".join(self._parts)
        # Collapse runs of whitespace except newlines
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse runs of blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class _MarkdownParser(HTMLParser):
    """Like _PlainTextParser but emits # markers for h1–h6 tags."""

    _BLOCK = frozenset("p div section article li tr td th br hr".split())
    _HEADING_MARKERS: dict[str, str] = {
        "h1": "#", "h2": "##", "h3": "###",
        "h4": "####", "h5": "#####", "h6": "######",
    }
    _SKIP = frozenset("script style ix:header ix:nonfraction ix:nonnumeric".split())

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._heading_marker: str | None = None   # "##" etc. when inside a heading tag
        self._heading_prefixed = False             # True once we've emitted the marker

    def handle_starttag(self, tag: str, attrs):
        t = tag.lower()
        if t in self._SKIP or t.startswith("ix:"):
            self._skip_depth += 1
        if self._skip_depth:
            return
        if t in self._HEADING_MARKERS:
            self._heading_marker = self._HEADING_MARKERS[t]
            self._heading_prefixed = False
            self._parts.append("\n")
        elif t in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if t in self._SKIP or t.startswith("ix:"):
            self._skip_depth = max(0, self._skip_depth - 1)
        if t in self._HEADING_MARKERS:
            self._heading_marker = None
            self._heading_prefixed = False
            self._parts.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth:
            return
        if self._heading_marker is not None and not self._heading_prefixed:
            if data.strip():
                self._parts.append(f"{self._heading_marker} ")
                self._heading_prefixed = True
        self._parts.append(data)

    def result(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _strip_tags(html: str) -> str:
    p = _PlainTextParser()
    try:
        p.feed(html)
    except Exception:
        pass
    return p.result()


def _to_markdown(html: str) -> str:
    p = _MarkdownParser()
    try:
        p.feed(html)
    except Exception:
        pass
    return p.result()


# ── Bold heading inference ────────────────────────────────────────────────────
#
# Many HTML documents (especially SEC EDGAR filings) use <b> or <strong> inside
# a bare <p> to create visual headings rather than semantic <h2>/<h3> tags.
# Pattern: <p><b>Short phrase without line breaks</b></p>
#
# Detection criteria (all must hold):
#   1. The <b>/<strong> is the only non-whitespace content in its <p>.
#   2. The bold text contains no HTML tags (so no <br>, <hr>, nested elements).
#   3. The bold text contains no newlines (single-line phrase only).
#   4. After stripping, the text length is within max_len.
#
# Qualifying elements are promoted to <h2> so that _to_markdown() emits "## "
# and the chunker builds a breadcrumb one level below the Item heading.
#
# Patterns detected (sole bold content inside a block-level element):
#   1. <p|div|li|td|th [...]><b|strong [...]>text</b|strong></p|div|...>
#      — traditional bold tags, common in older EDGAR filings and HTML converters
#   2. <p|div|li|td|th [...]><span style="...font-weight:600-900|bold|bolder...">text</span></...>
#      — CSS inline bold (≥600 = semi-bold through black), modern inline XBRL filings
#
# Both patterns require that the bold inline element is the SOLE content of the block
# (only surrounding whitespace allowed) so that running-text bold emphasis in prose
# paragraphs is never mistaken for a heading.

_BLOCK_TAGS = r"(?:p|div|li|td|th)"

# Pattern 1 — <b> / <strong> as sole block content
_BOLD_B_RE = re.compile(
    rf"<{_BLOCK_TAGS}[^>]*>\s*<(?:b|strong)[^>]*>([^\n<]+)</(?:b|strong)>\s*\.?\s*</{_BLOCK_TAGS}>",
    re.IGNORECASE,
)

# Pattern 2 — <span style="...font-weight:≥600|bold..."> as sole block content.
# CSS font-weight 600–900 all render as visually bold (semi-bold through black).
# The regex matches the numeric range 6xx-9xx or the keyword "bold"/"bolder".
_BOLD_SPAN_RE = re.compile(
    rf"<{_BLOCK_TAGS}[^>]*>\s*<span[^>]*font-weight\s*:\s*(?:[6-9]\d\d|bold(?:er)?)[^>]*>([^<]+)</span>\s*</{_BLOCK_TAGS}>",
    re.IGNORECASE,
)


def _promote_bold_headings(html: str, max_len: int = 120) -> str:
    """Replace sole-bold block elements that look like headings with ``<h2>`` tags.

    Covers all common EDGAR bold-heading conventions:

    * ``<p><b>text</b></p>`` — old-style ``<b>``/``<strong>`` in paragraphs
    * ``<div><b>text</b></div>`` — bold in div blocks
    * ``<div><span style="font-weight:700">text</span></div>`` — inline XBRL CSS bold
    * Same patterns with ``<li>``, ``<td>``, ``<th>`` as the outer block

    Args:
        html:    Raw HTML of a single section (between two TOC anchors).
        max_len: Maximum character length of the bold text (after stripping
                 whitespace and a trailing period) to qualify as a heading.

    Returns:
        HTML with qualifying bold blocks replaced by ``<h2>...</h2>``.
    """
    def _replace(m: re.Match) -> str:
        text = m.group(1).strip().rstrip(".")
        if not text or len(text) > max_len:
            return m.group(0)
        return f"<h2>{text}</h2>"

    html = _BOLD_B_RE.sub(_replace, html)
    html = _BOLD_SPAN_RE.sub(_replace, html)
    return html


# ── TOC parsing ───────────────────────────────────────────────────────────────

_HREF_ITEM_RE = re.compile(r"\bitem[_\-](\d+[abc]?)[_\-]", re.IGNORECASE)


def _parse_toc(html: str) -> list[tuple[str, str]]:
    """Return ordered [(anchor_id, item_number), ...] from the filing TOC.

    Handles the three anchor/label layouts found across EDGAR filers:

    1. **Link-text style** (Apple): ``<a href="#opaque_hash">Item 1A.</a>``
       The item label is directly inside the ``<a>`` tag.

    2. **Descriptive-href style** (Microsoft):
       ``<a href="#item_1a_risk_factors">Risk Factors</a>``
       The href slug encodes the item number.

    3. **Adjacent-cell style** (Amazon, Salesforce):
       ``<td><span>Item 1A.</span></td><td>...<a href="#opaque_hash">Risk Factors</a>``
       The item label sits in a preceding ``<td>`` of the same ``<tr>``.
    """
    results: list[tuple[str, str]] = []
    seen_anchors: set[str] = set()

    link_re = re.compile(
        r'<a\b[^>]+href="#([^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )

    for m in link_re.finditer(html):
        anchor_id = m.group(1).strip()
        if anchor_id in seen_anchors:
            continue

        # ── Strategy 1: item text directly in link (Apple) ───────────────────
        link_text = _strip_tags(m.group(2))
        item_m = _ITEM_RE.search(link_text)
        if item_m:
            raw = item_m.group(1).upper().rstrip(".")
            seen_anchors.add(anchor_id)
            results.append((anchor_id, raw))
            continue

        # ── Strategy 2: item number encoded in href (Microsoft) ──────────────
        href_m = _HREF_ITEM_RE.search(anchor_id)
        if href_m:
            raw = href_m.group(1).upper()
            if raw in _ITEM_NAMES:
                seen_anchors.add(anchor_id)
                results.append((anchor_id, raw))
                continue

        # ── Strategy 3: item label in adjacent <td>, same <tr> (Amazon/CRM) ──
        # Walk back to the enclosing <tr>; limit to 2 000 chars to avoid
        # matching body cross-references that happen to be near a heading.
        row_start = html.rfind("<tr", 0, m.start())
        if row_start != -1 and (m.start() - row_start) < 2000:
            row_text = _strip_tags(html[row_start : m.start()])
            ctx_m = _ITEM_RE.search(row_text)
            if ctx_m:
                raw = ctx_m.group(1).upper().rstrip(".")
                seen_anchors.add(anchor_id)
                results.append((anchor_id, raw))

    return results


# ── Main extractor ────────────────────────────────────────────────────────────

def _extract_edgar_prose(
    html: str,
    prose_only: bool = True,
    infer_bold_headings: bool = False,
    bold_heading_max_len: int = 120,
) -> str:
    """Extract prose sections from EDGAR inline XBRL HTML as Markdown.

    Args:
        html:                  Full HTML string of the 10-K filing.
        prose_only:            If True (default), skip financial-statement and
                               governance items (Item 8, 10-15).
        infer_bold_headings:   If True, promote sole-bold block elements
                               (``<p><b>``, ``<div><span style="font-weight:700">``,
                               ``<li><b>``, etc.) to ``<h2>`` so the chunker builds
                               a two-level breadcrumb:
                               ``Item 1A. Risk Factors > Foreign Exchange Risk``.
        bold_heading_max_len:  Maximum length of bold text (chars, after
                               stripping whitespace/trailing period) to qualify
                               as an inferred heading.  Default 120.

    Returns:
        Markdown string with ``# Item X.Y  Section Name`` headings, and
        optionally ``## Inferred Sub-Heading`` entries when
        *infer_bold_headings* is True.
    """
    toc = _parse_toc(html)
    if not toc:
        # Fallback: can't find EDGAR TOC structure — just strip all tags
        return _strip_tags(html)

    # Filter to prose items if requested
    if prose_only:
        toc_filtered = [(aid, item) for aid, item in toc if item in _PROSE_ITEMS]
    else:
        toc_filtered = toc

    # All anchor IDs in document order (for slicing)
    all_ids = [aid for aid, _ in toc]

    sections: list[str] = []

    for anchor_id, item_num in toc_filtered:
        # Position of this anchor attribute in the raw HTML
        attr_pos = html.find(f'id="{anchor_id}"')
        if attr_pos == -1:
            continue

        # Walk back to the opening < of the enclosing element so we don't
        # slice mid-attribute and leak 'id="...">' as text content.
        tag_start = html.rfind("<", 0, attr_pos)
        pos_start = tag_start if tag_start != -1 else attr_pos

        # Position of the NEXT anchor after this one (any anchor, not just prose)
        try:
            idx = all_ids.index(anchor_id)
        except ValueError:
            continue

        if idx + 1 < len(all_ids):
            next_id = all_ids[idx + 1]
            next_attr = html.find(f'id="{next_id}"')
            pos_end = html.rfind("<", 0, next_attr) if next_attr != -1 else -1
            if pos_end <= pos_start:
                pos_end = len(html)
        else:
            pos_end = len(html)

        section_html = html[pos_start:pos_end]
        if infer_bold_headings:
            section_html = _promote_bold_headings(section_html, bold_heading_max_len)
            text = _to_markdown(section_html)
        else:
            text = _strip_tags(section_html)
        if not text.strip():
            continue

        name = _ITEM_NAMES.get(item_num, f"Item {item_num}")
        heading = f"# Item {item_num}.  {name}"
        sections.append(f"{heading}\n\n{text}")

    return "\n\n---\n\n".join(sections)


# ── Extractor class (satisfies chunkymonkey Extractor protocol) ──────────────

class EdgarExtractor:
    """Extract prose from SEC EDGAR 10-K inline XBRL HTML.

    Args:
        infer_bold_headings:   Promote sole-bold block elements (``<p><b>``,
                               ``<div><span style="font-weight:700">``, etc.)
                               to sub-headings so the chunker builds two-level
                               breadcrumbs inside each Item section, e.g.
                               ``Item 1A. Risk Factors > Foreign Exchange Risk``.
                               Default False (plain text, Item-level paths only).
        bold_heading_max_len:  Maximum character length of bold text to qualify
                               as an inferred heading.  Default 120.

    Register with DocumentLoader::

        loader = DocumentLoader(extra_extractors=[EdgarExtractor(infer_bold_headings=True)])
        chunks = loader.load_bytes(data, "aapl_10k", doc_type="edgar")

    Or auto-detect (adds to default pipeline)::

        from chunkymonkey.extractors import register_extractor
        register_extractor(EdgarExtractor(infer_bold_headings=True))
    """

    def __init__(
        self,
        infer_bold_headings: bool = False,
        bold_heading_max_len: int = 120,
    ) -> None:
        self._infer_bold_headings = infer_bold_headings
        self._bold_heading_max_len = bold_heading_max_len

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in ("edgar", "edgar-html", "edgar-xbrl")

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        html = data.decode("utf-8", errors="replace")
        return _extract_edgar_prose(
            html,
            prose_only=True,
            infer_bold_headings=self._infer_bold_headings,
            bold_heading_max_len=self._bold_heading_max_len,
        )

    @staticmethod
    def is_edgar_html(data: bytes) -> bool:
        """Heuristic: returns True if the bytes look like EDGAR inline XBRL."""
        head = data[:4096].decode("utf-8", errors="replace").lower()
        return "xmlns:ix=" in head or "<ix:header" in head or "sec.gov" in head
