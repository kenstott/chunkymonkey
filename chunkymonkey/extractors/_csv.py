# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CSV extractor — renders tabular data as a markdown table."""

from __future__ import annotations

import csv
import io


class CsvExtractor:
    """Extract CSV / TSV files into a markdown table.

    Each CSV becomes one markdown table.  Column headers are used as the
    header row; if the file has no header the columns are numbered (col_1, …).
    """

    HANDLED = {"csv", "tsv"}

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")

        # Sniff dialect; fall back to excel (standard CSV)
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t|;")
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(io.StringIO(text), dialect)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            return ""

        headers = rows[0]
        body = rows[1:]

        def _cell(value: str) -> str:
            return value.replace("|", "\\|").replace("\n", " ").strip()

        header_row = "| " + " | ".join(_cell(h) for h in headers) + " |"
        sep_row = "| " + " | ".join("---" for _ in headers) + " |"
        data_rows = [
            "| " + " | ".join(_cell(c) for c in row) + " |"
            for row in body
        ]

        return "\n".join([header_row, sep_row] + data_rows)