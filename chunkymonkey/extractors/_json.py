# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 3a7e2f1d-b849-4c8e-9d01-e5f6a2b3c4d7
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""JSON / JSONL extractor — emits markdown with key-path breadcrumb headings."""

from __future__ import annotations

import json
from typing import Any


def _walk(obj: Any, lines: list[str], depth: int, path: str) -> None:
    heading = "#" * min(depth, 6)
    if isinstance(obj, dict):
        for key, val in obj.items():
            child_path = f"{path} > {key}" if path else key
            if isinstance(val, (dict, list)):
                lines.append(f"{heading} {child_path}")
                _walk(val, lines, depth + 1, child_path)
            else:
                lines.append(f"{heading} {child_path}")
                if val is not None and str(val).strip():
                    lines.append(str(val))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            child_path = f"{path}[{i}]" if path else f"[{i}]"
            if isinstance(item, (dict, list)):
                lines.append(f"{heading} {child_path}")
                _walk(item, lines, depth + 1, child_path)
            else:
                if item is not None and str(item).strip():
                    lines.append(str(item))
    else:
        if obj is not None and str(obj).strip():
            lines.append(str(obj))


class JsonExtractor:
    """Extract JSON / JSONL files into markdown with hierarchical headings."""

    HANDLED = {"json", "jsonl"}

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")

        lines: list[str] = []
        # JSONL: one JSON object per line
        if source_path and source_path.lower().endswith(".jsonl"):
            for raw_line in text.splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    lines.append(raw_line)
                    continue
                _walk(obj, lines, depth=1, path="")
        else:
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return text
            _walk(obj, lines, depth=1, path="")

        return "\n\n".join(lines)
