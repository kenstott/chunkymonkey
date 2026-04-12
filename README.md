# chunkeymonkey

> A dairy-free RAG pipeline for delicious semantic similarity, clustering and NER.
> No artificial embeddings. No factory-farm vector stores. Sustainably harvested tokens.

---

## The thesis

Naive RAG pipelines embed raw chunk content, losing the document structure that gives the
content meaning. A table of numbers is ambiguous on its own — it only makes sense in the
context of its heading hierarchy. chunkeymonkey enriches each chunk with a section
breadcrumb (`"Section: Methods > Table 1"`) prepended to the text that gets embedded,
while leaving the original `content` field untouched. This small change consistently
improves retrieval precision, clustering coherence, and NER quality because the embedding
encodes both *what* the text says and *where in the document it lives*.

---

## Features

- **Zero mandatory dependencies** — stdlib only for core chunking and context enrichment
- **Transports** — local filesystem, HTTP/HTTPS, S3/S3A, FTP, SFTP — plug in more (SharePoint, Confluence, Notion)
- **Extractors** — PDF, DOCX, XLSX, PPTX, HTML, Markdown, plain text — plug in more (audio, images, JIRA exports)
- **Storage** — DuckDB VSS (HNSW) vector search + SQLAlchemy relational metadata store
- **Hybrid search** — cosine similarity + BM25 full-text search with Reciprocal Rank Fusion merge
- **Paragraph-aware splitting** — chunks never break mid-paragraph
- **Table and list detection** — pipe-delimited rows and list items stay atomic
- **Large table continuation markers** — oversized tables split at row boundaries with `[TABLE:start]` / `[TABLE:cont]` / `[TABLE:end]` for clean reassembly

---

## Installation

Core (no optional dependencies):
```bash
pip install chunkeymonkey
```

With specific extras:
```bash
pip install "chunkeymonkey[http]"       # HTTP/HTTPS transport
pip install "chunkeymonkey[s3]"         # Amazon S3 transport
pip install "chunkeymonkey[sftp]"       # SFTP transport
pip install "chunkeymonkey[pdf]"        # PDF extraction
pip install "chunkeymonkey[docx]"       # DOCX extraction
pip install "chunkeymonkey[xlsx]"       # XLSX extraction
pip install "chunkeymonkey[pptx]"       # PPTX extraction
pip install "chunkeymonkey[storage]"    # DuckDB vector store
pip install "chunkeymonkey[full]"       # Everything
```

For development:
```bash
git clone https://github.com/kennethstott/chunkeymonkey
cd chunkeymonkey
pip install -e ".[dev]"
```

---

## Quick start

```python
from chunkeymonkey import chunk_document, enrich_chunks

# 1. Chunk a document
chunks = chunk_document(
    name="annual_report.md",
    content=open("annual_report.md").read(),
    chunk_size=1500,
    table_chunk_limit=800,
)

# 2. Enrich chunks with section breadcrumbs for embedding
enriched = enrich_chunks(chunks, strategy="prefix")

for chunk in enriched:
    # chunk.content        — original text (for display, storage)
    # chunk.embedding_content — "Section: ...\n\noriginal text" (for embedding)
    print(chunk.chunk_index, chunk.section, len(chunk.embedding_content))
```

Using `DocumentLoader` for the full pipeline:

```python
from chunkeymonkey import DocumentLoader

loader = DocumentLoader(chunk_size=1500, context_strategy="prefix")

# From a local file
chunks = loader.load("/path/to/document.pdf")

# From a URL (requires pip install chunkeymonkey[http])
chunks = loader.load("https://example.com/report.html")

# From raw bytes
chunks = loader.load_bytes(pdf_bytes, "report.pdf", doc_type="pdf")

# From pre-extracted text
chunks = loader.load_text("Paragraph one.\n\nParagraph two.", "notes.txt")
```

---

## Pipeline

```
URI
 |
 v
Transport (LocalTransport / HttpTransport / S3Transport / ...)
 |  fetch(uri) -> FetchResult(data: bytes, detected_mime, source_path)
 v
Extractor (PdfExtractor / HtmlExtractor / TextExtractor / ...)
 |  extract(data) -> str
 v
chunk_document(name, text, chunk_size, table_chunk_limit)
 |  -> List[DocumentChunk]  (content, section, chunk_index, ...)
 v
enrich_chunks(chunks, strategy="prefix")
 |  -> List[DocumentChunk]  (+ embedding_content set on each)
 v
Your embedding model / vector store
```

---

## API reference

### `DocumentChunk` fields

| Field | Type | Description |
|---|---|---|
| `document_name` | `str` | Source document name |
| `content` | `str` | Chunk text (original, unchanged) |
| `section` | `str | None` | Breadcrumb path of enclosing headings (`"Intro > Background"`) |
| `chunk_index` | `int` | Zero-based position within the document |
| `source_offset` | `int | None` | Byte offset of chunk start in original content |
| `source_length` | `int | None` | Byte length of chunk content |
| `embedding_content` | `str | None` | Set by `enrich_chunks()`; what actually gets embedded |
| `chunk_type` | `str` | `"document"` | `"schema"` | `"api"` |

### `chunk_document`

```python
chunk_document(
    name: str,
    content: str,
    chunk_size: int = 1500,
    table_chunk_limit: int = 800,
) -> list[DocumentChunk]
```

Split a text document into semantically coherent chunks. Respects paragraph boundaries,
keeps tables and lists atomic, tracks heading hierarchy in `section`, and splits large
tables with continuation markers.

### `enrich_chunk` / `enrich_chunks`

```python
enrich_chunk(chunk: DocumentChunk, strategy: str = "prefix") -> DocumentChunk
enrich_chunks(chunks: list[DocumentChunk], strategy: str = "prefix") -> list[DocumentChunk]
```

Return new chunk(s) with `embedding_content` populated. Never mutates input.

- `strategy="prefix"` — `"Section: Methods > Table 1\n\n<original content>"`
- `strategy="inline"` — `"[Methods > Table 1] <original content>"`

Raises `ValueError` for unknown strategy values.

### `DocumentLoader`

```python
DocumentLoader(
    chunk_size: int = 1500,
    table_chunk_limit: int = 800,
    context_strategy: str | None = "prefix",
    extra_transports: list | None = None,
    extra_extractors: list | None = None,
)
```

Full pipeline: fetch → extract → chunk → enrich. Methods:

- `loader.load(uri, name=None)` — fetch from URI and return chunks
- `loader.load_bytes(data, name, doc_type="auto", source_path=None)` — extract from raw bytes
- `loader.load_text(text, name)` — chunk and enrich pre-extracted text

---

## Extending chunkeymonkey

### Custom extractor

Implement `can_handle(doc_type) -> bool` and `extract(data, source_path) -> str`, then
pass to `DocumentLoader(extra_extractors=[...])`. See [`examples/custom_extractor.py`](examples/custom_extractor.py).

```python
class CsvSummaryExtractor:
    def can_handle(self, doc_type): return doc_type == "csv-summary"
    def extract(self, data, source_path=None):
        ...  # return str

loader = DocumentLoader(extra_extractors=[CsvSummaryExtractor()])
chunks = loader.load_bytes(csv_bytes, "data.csv", doc_type="csv-summary")
```

### Custom transport

Implement `can_handle(uri) -> bool` and `fetch(uri, **kwargs) -> FetchResult`, then pass
to `DocumentLoader(extra_transports=[...])`. See [`examples/custom_transport.py`](examples/custom_transport.py).

```python
from chunkeymonkey.transports._protocol import FetchResult

class SharePointTransport:
    def can_handle(self, uri): return uri.startswith("sharepoint://")
    def fetch(self, uri, **kwargs):
        data = ...  # your fetch logic
        return FetchResult(data=data, detected_mime="text/html", source_path=uri)

loader = DocumentLoader(extra_transports=[SharePointTransport()])
chunks = loader.load("sharepoint://site/document")
```

---

## Storage

Requires `pip install "chunkeymonkey[storage]"`.

```python
import numpy as np
from chunkeymonkey import DocumentLoader
from chunkeymonkey.storage import Store

loader = DocumentLoader(context_strategy="prefix")
chunks = loader.load("report.pdf")

# Generate embeddings with your model
embeddings = your_model.encode([c.embedding_content for c in chunks])
embeddings = np.array(embeddings, dtype=np.float32)

with Store("index.duckdb", embedding_dim=1024) as store:
    store.add_document(chunks, embeddings)

    query_vec = your_model.encode(["What were the primary outcomes?"])
    results = store.search(query_vec[0], limit=5)
    for chunk_id, score, chunk in results:
        print(f"{score:.3f} [{chunk.section}] {chunk.content[:80]}")
```

---

## Running the demo

```bash
cd /path/to/chunkeymonkey
python demo/contextual_vs_naive.py
```

The demo loads `demo/sample_doc.md` — a realistic multi-section scientific document —
twice: once with naive chunking and once with contextual enrichment. It then runs five
queries against both approaches using a pure-Python TF-IDF retriever and shows a
side-by-side comparison of which approach retrieved the relevant section.

---

## Ethics & sourcing

- **Sustainably harvested tokens** — no embeddings computed, stored, or billed without your consent
- **Free-range paragraphs** — chunks never split mid-sentence against their will
- **Cage-free section breadcrumbs** — every chunk knows where it came from
- **Conflict-free text extraction** — no third-party cloud APIs consulted without consent
- **Non-GMO transport layer** — no monkey-patching of built-ins
- **Fair trade** — MIT licensed, attribution appreciated

---

## License

MIT
