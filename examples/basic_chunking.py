# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 707a49f0-60de-4ce9-8efe-7da3d45b9038
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Basic chunking example — no optional dependencies required."""
from chunkymonkey import chunk_document

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
