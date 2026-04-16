# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 7c4b1e9f-d23a-4f7e-8b05-c1d2e3f4a5b6
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""XML extractor — emits markdown with element-path breadcrumb headings."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def _tag(element: ET.Element) -> str:
    """Strip namespace URI from a tag name."""
    tag = element.tag
    if tag.startswith("{"):
        tag = tag[tag.index("}") + 1:]
    return tag


def _walk(element: ET.Element, lines: list[str], depth: int, path: str) -> None:
    tag = _tag(element)
    current_path = f"{path} > {tag}" if path else tag
    heading = "#" * min(depth, 6)
    lines.append(f"{heading} {current_path}")

    text = (element.text or "").strip()
    if text:
        lines.append(text)

    for child in element:
        _walk(child, lines, depth + 1, current_path)

    tail = (element.tail or "").strip()
    if tail:
        lines.append(tail)


class XmlExtractor:
    """Extract XML files into markdown with element-path headings."""

    HANDLED = {"xml"}

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1")

        lines: list[str] = []
        _walk(root, lines, depth=1, path="")
        return "\n\n".join(lines)
