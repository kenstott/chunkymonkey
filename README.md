# 🐵 Chunkeymonkey

**A dairy-free RAG pipeline for delicious semantic similarity, clustering and NER.**

*Made with chunks. No fillers, no artificial embeddings, no harmful factory-farm vector stores.*

---

Chunkeymonkey is a zero-dependency Python library for splitting documents into semantically coherent chunks. It respects paragraph boundaries, keeps tables and lists atomic, tracks section breadcrumbs, and handles oversized table blocks with continuation markers for clean reassembly downstream.

Extracted from [constat](https://github.com/kennethstott/constat) and packaged for standalone use.

---

## Features

- **Paragraph-aware splitting** — chunks never break mid-paragraph
- **Table detection** — pipe-delimited rows (Markdown, DOCX, XLSX, PPTX, HTML) are merged into atomic blocks before chunking
- **List detection** — consecutive unordered and ordered list items stay together
- **Section breadcrumbs** — Markdown headings (`#`, `##`, `###`) and spreadsheet `[Sheet: ...]` markers populate `DocumentChunk.section`
- **Large table splitting** — tables that exceed `table_chunk_limit` are split at row boundaries with `[TABLE:start]` / `[TABLE:cont]` / `[TABLE:end]` markers for reassembly
- **Zero dependencies** — pure Python 3.11+, no third-party packages required

---

## Installation

```bash
pip install chunkeymonkey
```

Or from source:

```bash
git clone https://github.com/kennethstott/chunkeymonkey
cd chunkeymonkey
pip install -e .
```

---

## Quick start

```python
from chunkeymonkey import chunk_document

chunks = chunk_document(
    name="annual_report.md",
    content=open("annual_report.md").read(),
    chunk_size=1500,       # target max chars per chunk
    table_chunk_limit=800, # split tables larger than this at row boundaries
)

for chunk in chunks:
    print(chunk.chunk_index, chunk.section, len(chunk.content))
```

### `DocumentChunk` fields

| Field | Type | Description |
|---|---|---|
| `document_name` | `str` | Source document name |
| `content` | `str` | Chunk text |
| `section` | `str \| None` | Breadcrumb path of enclosing headings (`"Intro > Background"`) |
| `chunk_index` | `int` | Zero-based position within the document |
| `source_offset` | `int \| None` | Byte offset of chunk start in original content |
| `source_length` | `int \| None` | Byte length of chunk content |

---

## Lower-level API

```python
from chunkeymonkey import is_table_line, is_list_line, merge_blocks

# Detect structural line types
is_table_line("| col1 | col2 | col3 |")  # True
is_list_line("- bullet item")             # True

# Merge runs of table or list paragraphs into atomic blocks
merged = merge_blocks(paragraphs, separator="\n\n")
```

---

## Ethics & sourcing

- **Sustainably harvested tokens** — no embeddings are computed, stored, or billable
- **Free-range paragraphs** — chunks are never split mid-sentence against their will
- **Non-GMO** — zero monkey-patching of built-ins
- **Fair trade** — MIT licensed, attribution appreciated

---

## License

MIT
