# Chunky Monkey

> A dairy-free RAG pipeline for delicious semantic similarity, clustering and NER.
> No artificial embeddings. No factory-farm vector stores. Sustainably harvested tokens.

---

## The problem with naive chunking

Most RAG pipelines embed raw chunk content and nothing else. This works when every
chunk contains enough distinctive vocabulary to describe itself. That is a narrow
special case.

In practice, almost every document type you want to retrieve from has repeating
structure:

- **Technical documentation** — every function reference has `Parameters`, `Returns`,
  `Raises` sections with the same words across every function in every library
- **Code** — every `__init__`, test setup, error handler, and config block shares
  vocabulary across the entire codebase
- **Contracts** — indemnification, limitation of liability, and governing law clauses
  are assembled from a shared clause library; the boilerplate is identical across
  every agreement
- **Regulatory filings** — every 10-K has the same Items in the same order; every
  company's Controls and Procedures section (Item 9A) is near-verbatim identical
- **Clinical protocols** — ECOG performance criteria, RECIST endpoints, and organ
  function thresholds appear word-for-word across hundreds of trials
- **Academic papers** — Abstract, Introduction, Methods, Results, Discussion; the
  heading hierarchy is fixed by convention

When sections share vocabulary, the embedding vectors for chunks from different
documents — or different sections of the same document — are indistinguishable.
Retrieval returns the wrong chunk, from the wrong document, for the wrong reason.

The fix is simple: the document name and section path are known at chunk time.
Put them in the text that gets embedded.

```
Document: techcorp_msa_2024
Section: Limitation of Liability

IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY…
```

This is strictly better than embedding the chunk alone. If the content is already
distinctive, the prefix adds a few redundant tokens and costs nothing. If the
content is ambiguous — which it usually is — the prefix is the only thing that
makes the embedding retrievable. There is no downside.

---

## What Chunky Monkey does

Chunky Monkey is a document chunking and contextual enrichment pipeline. It:

1. **Fetches** documents from local disk, HTTP/HTTPS, S3, FTP, SFTP, or any custom
   source (SharePoint, Confluence, Google Drive, Notion)
2. **Extracts** text from PDF, DOCX, XLSX, PPTX, HTML, Markdown, plain text, SEC
   EDGAR inline XBRL, or any custom format
3. **Chunks** into semantically coherent pieces — never breaking mid-paragraph,
   keeping tables and lists atomic, tracking the full heading hierarchy
4. **Enriches** each chunk: sets `embedding_content` to
   `"Document: {name}\nSection: {path}\n\n{content}"` before it reaches your
   embedding model

The original `content` field is never modified. `embedding_content` is what you
embed. Everything downstream — your embedding model, vector store, retrieval
logic — is unchanged.

---

## Installation

Core (no optional dependencies):
```bash
pip install chunkymonkey
```

With specific extras:
```bash
pip install "chunkymonkey[http]"       # HTTP/HTTPS transport
pip install "chunkymonkey[s3]"         # Amazon S3 transport
pip install "chunkymonkey[sftp]"       # SFTP transport
pip install "chunkymonkey[pdf]"        # PDF extraction
pip install "chunkymonkey[docx]"       # DOCX extraction
pip install "chunkymonkey[xlsx]"       # XLSX extraction
pip install "chunkymonkey[pptx]"       # PPTX extraction
pip install "chunkymonkey[storage]"    # DuckDB vector store
pip install "chunkymonkey[full]"       # Everything
```

---

## Quick start

```python
from chunkymonkey import DocumentLoader

loader = DocumentLoader()   # context_strategy="prefix" is the default

# Local file, URL, or raw bytes — same interface
chunks = loader.load("/path/to/report.pdf")
chunks = loader.load("https://example.com/docs/api.html")
chunks = loader.load_bytes(pdf_bytes, name="report", doc_type="pdf")
chunks = loader.load_text("Paragraph one.\n\nParagraph two.", name="notes")

for chunk in chunks:
    # chunk.content           — original text, unchanged (for display, storage)
    # chunk.embedding_content — "Document: ...\nSection: ...\n\n..." (embed this)
    # chunk.section           — "Item 1A.  Risk Factors" (metadata, not in content)
    # chunk.document_name     — "aapl_10k_2025" (metadata, not in content)
    embed(chunk.embedding_content)
```

The section path and document name appear in both `chunk.section` / `chunk.document_name`
(as metadata, for filtering and display) **and** in `embedding_content` (as text, for
disambiguation during vector search). These are separate concerns. The metadata is
always present; `embedding_content` is what makes retrieval accurate.

---

## Pipeline

```
URI
 │
 ▼
Transport  (Local / HTTP / S3 / FTP / SFTP / custom)
 │  fetch(uri) → FetchResult(data: bytes, detected_mime, source_path)
 ▼
Extractor  (PDF / DOCX / XLSX / PPTX / HTML / Markdown / EDGAR / custom)
 │  extract(data) → str
 ▼
chunk_document(name, text, chunk_size, table_chunk_limit)
 │  → list[DocumentChunk]  (content, section, document_name, …)
 ▼
enrich_chunks(chunks, strategy="prefix")
 │  → list[DocumentChunk]  (+ embedding_content on each chunk)
 ▼
Your embedding model / vector store
```

---

## API reference

### `DocumentChunk` fields

| Field | Type | Description |
|---|---|---|
| `document_name` | `str` | Source document name |
| `content` | `str` | Chunk text — original, never modified |
| `section` | `str \| None` | Breadcrumb of enclosing headings (`"Methods > Table 1"`) |
| `chunk_index` | `int` | Zero-based position within the document |
| `source_offset` | `int \| None` | Character offset of chunk start in source text |
| `source_length` | `int \| None` | Character length of chunk content |
| `embedding_content` | `str \| None` | Set by `enrich_chunks()` — embed this, not `content` |
| `chunk_type` | `str` | `"document"`, `"schema"`, or `"api"` |

### `chunk_document`

```python
chunk_document(
    name: str,
    content: str,
    chunk_size: int = 1500,
    table_chunk_limit: int = 800,
) -> list[DocumentChunk]
```

Splits a document into semantically coherent chunks. Respects paragraph boundaries,
keeps tables and lists atomic, tracks heading hierarchy in `section`, and splits
large tables with continuation markers (`[TABLE:start]` / `[TABLE:cont]` / `[TABLE:end]`).

### `enrich_chunk` / `enrich_chunks`

```python
enrich_chunk(chunk: DocumentChunk, strategy: str = "prefix") -> DocumentChunk
enrich_chunks(chunks: list[DocumentChunk], strategy: str = "prefix") -> list[DocumentChunk]
```

Returns new chunk(s) with `embedding_content` set. Never mutates input.

| Strategy | `embedding_content` format |
|---|---|
| `"prefix"` (default) | `"Document: aapl_10k_2025\nSection: Item 1A.  Risk Factors\n\n<content>"` |
| `"inline"` | `"[aapl_10k_2025 > Item 1A.  Risk Factors] <content>"` |

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

Full pipeline: fetch → extract → chunk → enrich. `context_strategy=None` disables
enrichment and is only useful as a baseline for benchmarking.

Methods:
- `loader.load(uri, name=None)` — fetch from any supported URI
- `loader.load_bytes(data, name, doc_type="auto")` — extract from raw bytes
- `loader.load_text(text, name)` — chunk and enrich pre-extracted text
- `loader.load_site(url, max_pages=50, max_depth=3)` — crawl and chunk a website
- `loader.load_directory(path, extensions=None, recursive=True)` — chunk a local directory

---

## Extending Chunky Monkey

### Custom extractor

```python
class CsvExtractor:
    def can_handle(self, doc_type): return doc_type == "csv"
    def extract(self, data, source_path=None):
        return data.decode()  # return plain text

loader = DocumentLoader(extra_extractors=[CsvExtractor()])
chunks = loader.load_bytes(csv_bytes, name="data.csv", doc_type="csv")
```

### Custom transport

```python
from chunkymonkey.transports._protocol import FetchResult

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

Requires `pip install "chunkymonkey[storage]"`.

```python
import numpy as np
from chunkymonkey import DocumentLoader
from chunkymonkey.storage import Store

loader = DocumentLoader()
chunks = loader.load("report.pdf")

embeddings = your_model.encode([c.embedding_content for c in chunks])

with Store("index.duckdb", embedding_dim=1024) as store:
    store.add_document(chunks, np.array(embeddings, dtype=np.float32))

    results = store.search(your_model.encode(["primary outcomes"])[0], limit=5)
    for chunk_id, score, chunk in results:
        print(f"{score:.3f}  [{chunk.document_name} > {chunk.section}]")
        print(f"       {chunk.content[:80]}")
```

---

## Demos

```bash
# Synthetic multi-section docs (ops reports, product catalogs, incident logs)
python demo/contextual_vs_naive.py

# Real SEC EDGAR 10-K filings (AAPL, MSFT, AMZN, CRM) — requires internet
python demo/edgar_demo.py

# Real ClinicalTrials.gov Phase 2/3 oncology protocols — requires internet
python demo/clinicaltrials_demo.py

# Python standard library documentation — requires internet
python demo/python_docs_demo.py
```

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
