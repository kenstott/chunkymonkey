# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 139c6f70-ad89-481c-9112-5ba9fd8a1e7e
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Custom extractor example — plug in your own format.

Shows how to implement the Extractor protocol and register it with DocumentLoader.
This example implements a simple CSV summary extractor as a demonstration.
(For audio transcription, JIRA exports, SharePoint pages, etc. — same pattern.)
"""
import csv
import io
from chunkymonkey import DocumentLoader


class CsvSummaryExtractor:
    """Extracts a human-readable summary from a CSV file."""

    def can_handle(self, doc_type: str) -> bool:
        return doc_type == "csv-summary"

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        text = data.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return "Empty CSV file."
        columns = list(rows[0].keys())
        lines = [
            "# CSV Data Summary",
            f"Columns: {', '.join(columns)}",
            f"Row count: {len(rows)}",
            "",
            "## Sample rows",
        ]
        for row in rows[:5]:
            lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(lines)


if __name__ == "__main__":
    import tempfile
    import os

    sample_csv = b"name,age,city\nAlice,30,NYC\nBob,25,LA\nCarol,35,Chicago\n"

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(sample_csv)
        tmp_path = f.name

    try:
        loader = DocumentLoader(extra_extractors=[CsvSummaryExtractor()])
        chunks = loader.load_bytes(sample_csv, "people.csv", doc_type="csv-summary")
        for chunk in chunks:
            print(f"Section: {chunk.section}")
            print(f"Content:\n{chunk.content}\n")
    finally:
        os.unlink(tmp_path)
