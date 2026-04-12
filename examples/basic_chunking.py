"""Basic chunking example — no optional dependencies required."""
from chunkeymonkey import chunk_document

SAMPLE = """
# Introduction

Machine learning models require high-quality training data. The preprocessing
pipeline determines the quality of downstream model performance.

## Data Collection

We collected data from three sources:
- Public datasets (60%)
- Proprietary datasets (30%)
- Synthetic generation (10%)

## Feature Engineering

| Feature        | Type    | Importance |
|----------------|---------|------------|
| Token count    | integer | High       |
| Section depth  | integer | Medium     |
| Table density  | float   | Low        |

# Methods

## Preprocessing

Text was normalized using standard NLP techniques.
""".strip()

if __name__ == "__main__":
    chunks = chunk_document("example.md", SAMPLE, chunk_size=300, table_chunk_limit=500)
    for chunk in chunks:
        print(f"[{chunk.chunk_index:02d}] section={chunk.section!r:40s} len={len(chunk.content)}")
