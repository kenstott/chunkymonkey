# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 3b668873-ceda-4c0e-8dc1-814a2da8d5b8
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Contextual chunking — the thesis in action.

Shows how enrich_chunks() adds section breadcrumbs to embedding_content
while leaving the original content untouched.
"""
from chunkymonkey import chunk_document, enrich_chunks

SAMPLE = """
# Annual Report 2024

## Financial Highlights

Revenue grew 23% year-over-year to $4.2B.

### Revenue by Region

| Region        | 2024     | 2023     | Growth |
|---------------|----------|----------|--------|
| North America | $2.1B    | $1.8B    | +17%   |
| Europe        | $1.2B    | $0.9B    | +33%   |
| Asia Pacific  | $0.9B    | $0.6B    | +50%   |

## Risk Factors

The following risks could materially affect our business.

### Market Risk

Foreign exchange fluctuations pose significant risk.
""".strip()

if __name__ == "__main__":
    chunks = chunk_document("report.md", SAMPLE, chunk_size=400, table_chunk_limit=600)
    enriched = enrich_chunks(chunks, strategy="prefix")

    print("=" * 70)
    print("CONTEXTUAL CHUNKING DEMO")
    print("=" * 70)
    for chunk in enriched:
        print(f"\nChunk {chunk.chunk_index} — section: {chunk.section!r}")
        print(f"  Original content ({len(chunk.content)} chars):")
        print(f"    {chunk.content[:80].replace(chr(10), ' ')!r}...")
        print(f"  Embedding content ({len(chunk.embedding_content)} chars):")
        print(f"    {chunk.embedding_content[:100].replace(chr(10), ' ')!r}...")
    print()
