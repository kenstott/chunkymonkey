# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 4ad05fa2-5f30-41e3-b3ed-9ca4ec3e6a84
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""HTML extractor — converts HTML to Markdown using stdlib html.parser."""

from __future__ import annotations

import re
from html.parser import HTMLParser


def _strip_html_chrome(html: str) -> str:
    """Pre-strip navigation chrome tags from raw HTML.

    Removes <nav>, <aside>, <noscript>, <script>, <style> and elements
    with navigation-related CSS classes/IDs. Uses regex on raw HTML
    since html.parser can't handle unclosed tags reliably.
    """
    for tag in ("nav", "aside", "header", "footer", "noscript", "script", "style"):
        html = re.sub(
            rf"<{tag}[\s>].*?</{tag}>",
            "", html, flags=re.DOTALL | re.IGNORECASE,
        )
    _NAV_ATTR_RE = re.compile(
        r"<(div|section|ul|table)[^>]*?"
        r"(?:class|id|role)\s*=\s*[\"'][^\"']*?"
        r"(?:sidebar|navbox|navbar|navigation|toc\b|catlinks"
        r"|mw-panel|mw-head|mw-editsection"
        r"|menu|breadcrumb|noprint"
        r"|portal|sister-?project|interlanguage|authority-control"
        r"|reflist|references|footnotes|mw-references-wrap|citation)"
        r"[^\"']*?[\"'][^>]*>.*?</\1>",
        re.DOTALL | re.IGNORECASE,
    )
    html = _NAV_ATTR_RE.sub("", html)
    html = re.sub(
        r"<sup[^>]*class=[\"'][^\"']*reference[^\"']*[\"'][^>]*>.*?</sup>",
        "", html, flags=re.DOTALL | re.IGNORECASE,
    )
    return html


class _MarkdownConverter(HTMLParser):
    def __init__(self):
        super().__init__()
        self._output: list[str] = []
        self._tag_stack: list[str] = []
        self._list_stack: list[str] = []  # "ul" or "ol"
        self._ol_counters: list[int] = []
        self._in_pre = False
        self._href: str | None = None
        self._link_text: list[str] = []
        self._in_link = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._output.append("\n\n")
        elif tag == "p":
            self._output.append("\n\n")
        elif tag == "br":
            self._output.append("\n")
        elif tag == "pre":
            self._in_pre = True
            self._output.append("\n\n```\n")
        elif tag == "ul":
            self._list_stack.append("ul")
        elif tag == "ol":
            self._list_stack.append("ol")
            self._ol_counters.append(0)
        elif tag == "li":
            indent = "  " * (len(self._list_stack) - 1)
            if self._list_stack and self._list_stack[-1] == "ol":
                self._ol_counters[-1] += 1
                self._output.append(f"\n{indent}{self._ol_counters[-1]}. ")
            else:
                self._output.append(f"\n{indent}- ")
        elif tag in ("strong", "b"):
            self._output.append("**")
        elif tag in ("em", "i"):
            self._output.append("*")
        elif tag == "a":
            attr_dict = dict(attrs)
            self._href = attr_dict.get("href")
            self._in_link = True
            self._link_text = []
        elif tag == "tr":
            self._output.append("\n|")
        elif tag in ("td", "th"):
            self._output.append(" ")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            prefix = "#" * level + " "
            text_parts: list[str] = []
            while self._output and self._output[-1] != "\n\n":
                text_parts.append(self._output.pop())
            text = "".join(reversed(text_parts)).strip()
            self._output.append(f"{prefix}{text}\n\n")
        elif tag == "p":
            self._output.append("\n")
        elif tag == "pre":
            self._in_pre = False
            self._output.append("\n```\n\n")
        elif tag == "ul":
            if self._list_stack:
                self._list_stack.pop()
            self._output.append("\n")
        elif tag == "ol":
            if self._list_stack:
                self._list_stack.pop()
            if self._ol_counters:
                self._ol_counters.pop()
            self._output.append("\n")
        elif tag in ("strong", "b"):
            self._output.append("**")
        elif tag in ("em", "i"):
            self._output.append("*")
        elif tag == "a":
            link_text = "".join(self._link_text).strip()
            if self._href and link_text:
                self._output.append(f"[{link_text}]({self._href})")
            else:
                self._output.append(link_text)
            self._in_link = False
            self._href = None
            self._link_text = []
        elif tag in ("td", "th"):
            self._output.append(" |")
        elif tag == "thead":
            self._output.append("\n|---|")

    def handle_data(self, data: str):
        if self._in_link:
            self._link_text.append(data)
            return
        self._output.append(data)

    def get_markdown(self) -> str:
        text = "".join(self._output)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _convert_html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown, preserving heading structure.

    Uses stdlib html.parser — no external dependencies.
    Handles: headings, paragraphs, lists (ul/ol/li), <br>, <pre>/<code>,
    bold, italic, links, and tables.
    Strips navigation chrome (nav, aside, sidebar, navbox, etc.).
    """
    html = _strip_html_chrome(html)
    converter = _MarkdownConverter()
    converter.feed(html)
    return converter.get_markdown()


class HtmlExtractor:
    """Extract plain text (as Markdown) from HTML documents."""

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in ("html", "htm")

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        text = data.decode("utf-8", errors="replace")
        return _convert_html_to_markdown(text)
