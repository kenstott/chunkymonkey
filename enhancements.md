# ChunkyMonkey Enhancements

Status key: `[complete]` = fully implemented in code | `[pending]` = not yet implemented

---

## Enhancement 3: Cross-Encoder Reranking `[pending]`

BM25 (`_bm25_search`), RRF merge (`_rrf_merge`), and hybrid search are already fully implemented in `chunkymonkey/storage/_vector.py`. The one missing piece from Constat's pipeline is the optional cross-encoder reranking pass.

An optional second-pass reranker (e.g. `ms-marco-MiniLM-L-6-v2`) scores each candidate directly against the query string. Applied after RRF to the top-N pool (N = 3×k), then truncated to k. Loaded lazily on first use, singleton, background thread — same pattern as the embedding model loader.

Constat reference: `constat/embedding_loader.py` (reranker loader), `constat/discovery/doc_tools/_access.py` (application after RRF).

---

## Enhancement 4: Federated Libraries — Multi-Store with Use-Case Aggregation `[pending]`

### Problem

Document collections naturally partition into independent libraries (e.g. "SEC filings", "internal policies", "contracts"). Each library has its own embedding store — independently maintained, independently versioned. But NER, clustering, and entity relationships only become meaningful when libraries are combined for a specific use case (e.g. "compliance analysis" draws from filings + policies + contracts).

`Store` already supports isolation today — each library is a separate DuckDB file. The missing piece is the **use-case layer**: combining multiple stores for federated search and cross-library NER/clustering.

### Two-Level Architecture

```
Library level (per store):
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │  sec.duckdb │  │policy.duckdb│  │contract.duckdb│
  │  embeddings │  │  embeddings │  │  embeddings │
  └─────────────┘  └─────────────┘  └─────────────┘

Use-case level (federated):
  ┌──────────────────────────────────────────────┐
  │  UseCase("compliance", [sec, policy, contract])│
  │  - federated search (fan-out + RRF merge)    │
  │  - cross-library NER index                   │
  │  - cross-library cluster map                 │
  │  - cross-library entity relationships        │
  └──────────────────────────────────────────────┘
```

### Library Level

Each library is an independent `Store`. Loaded, searched, and maintained in isolation. NER and clustering can optionally be run per-library for local search quality, but are not required.

### Use-Case Level

A `UseCase` holds references to N libraries and builds aggregated indexes across them:

```python
uc = UseCase("compliance", stores=[sec_store, policy_store, contract_store])
uc.build()          # runs cross-library NER + clustering
results = uc.search(query="reporting threshold", k=5)
```

**Federated search:** fan-out query to each library's HNSW index in parallel, collect top-k per library, RRF merge across libraries, return global top-k. Each result carries a `library` provenance tag.

**Cross-library NER:** entity-chunk associations span all libraries. An entity appearing in both SEC filings and internal policy chunks surfaces connections that per-library NER would miss.

**Cross-library clustering:** co-occurrence matrix built from all libraries. Cluster adjacency expansion in enhanced search traverses across library boundaries.

### What Doesn't Change

- Library stores are unmodified — no schema changes
- A library can participate in multiple use cases simultaneously
- Use-case indexes are derived/disposable — can be rebuilt from library stores at any time

---

## Enhancement 5: Parallel Document Loading `[pending]`

Constat loads documents with up to 8 concurrent threads, significant for I/O-bound sources (S3, HTTP, IMAP). The loader currently processes documents sequentially.

Add a `max_workers: int = 1` parameter to the loader. When `> 1`, use `ThreadPoolExecutor` to fetch and extract concurrently. Chunk and embed remain sequential per document (embedding model is not thread-safe without locking).

IMAP specifically benefits from per-message streaming: chunk and embed each message inline rather than accumulating all messages in memory before embedding.

Constat reference: `constat/discovery/doc_tools/_loaders.py` lines 101–119, 324–376.

---

## Enhancement 6: Content Hash Freshness `[pending]`

Refines Enhancement 1. Constat tracks both `file_mtime` and a SHA-256 `content_hash` of the extracted text. The hash catches cases where mtime changes but content does not (e.g. file copy, touch, metadata-only update) and cases where content changes without mtime update (e.g. S3 overwrites that preserve timestamp).

`freshness_meta` for local and S3 transports should always include `content_hash`. The `check_freshness` logic: if hash unchanged, skip re-ingest regardless of timestamp.

Constat reference: `constat/discovery/doc_tools/_core.py` lines 119–124.

---

## Enhancement 7: Image Classification for OCR `[pending]`

Refines Enhancement 4 (Image extractor). Constat classifies images before deciding extraction strategy:

- Run local OCR (Tesseract); score by word count and confidence
- If `word_count >= 50` AND `confidence >= 60%`: classify as **text-primary** → use OCR text
- Otherwise: classify as **image-primary** → send to LLM vision for description + tag extraction

The image tags (labels) are fed into NER as pseudo-entities, making image content searchable via entity adjacency.

chunkymonkey's image extractor (Enhancement 4) should adopt this classification step. LLM-only path (no Tesseract) remains valid — the classification just influences the prompt: ask for verbatim transcription vs. descriptive summary.

Constat reference: `constat/discovery/doc_tools/_image.py`.

---

## Enhancement 8: Entity Relationships and SVO Triples `[pending]`

Constat extracts Subject-Verb-Object triples from text and stores them as typed relationship records with confidence scores, `user_edited` flags, and verb categories. This is the foundation of a knowledge graph built from document ingestion without an LLM graph-extraction step.

Potential addition to chunkymonkey's NER layer: after entity extraction, run a lightweight SVO extractor (spaCy dependency parse) over chunks and store triples in a new `entity_relationships` table. Enables graph traversal as a fifth cohort-assembly dimension beyond the current four.

Constat reference: `constat/models.py` lines 310–327.

---

## Enhancement 9: Migration Compatibility with Constat `[pending]`

When Constat deprecates its internal embeddings and switches to chunkymonkey, the following must be compatible:

| Concern | Constat | chunkymonkey action needed |
|---------|---------|---------------------------|
| Chunk ID algorithm | SHA-256 of `document_name:chunk_index:content[:100]` | Verify hash matches; document in both codebases |
| Table continuation markers | `[TABLE:start]`, `[TABLE:cont]`, `[TABLE:end]` | chunkymonkey must preserve these if re-chunking Constat docs, or strip them cleanly |
| Domain/session scoping | `domain_id`, `session_id` columns | Enhancement 4 above |
| `chunk_type` enum | `DB_TABLE`, `API_ENDPOINT`, `DOCUMENT`, etc. | chunkymonkey's `chunk_type: str` is compatible; ensure same string values |
| Entity junction table | `chunk_entities` with `entity_class` | Already implemented in chunkymonkey; verify column names match |
| Reranking pipeline | Cross-encoder in Constat | Enhancement 3 above |
| Vector dimensionality | 1024-dim BGE-large | chunkymonkey must use same model or support configurable dim |

---

## Enhancement 10: Additional Transports `[pending]`

### Azure Blob Storage `[pending]`

Enterprise S3-equivalent for Microsoft shops. URI scheme: `az://container/blob-path`. Freshness via `Last-Modified` + `ETag` from blob properties. Pure Python via `azure-storage-blob`. Structurally symmetric with the existing S3 transport — minimal new design required.

### Google Cloud Storage `[pending]`

GCP equivalent. URI scheme: `gs://bucket/object-path`. Freshness via `updated` timestamp + `etag`. Pure Python via `google-cloud-storage`. Same structural symmetry with S3.

### Google Drive `[pending]`

URI scheme: `gdrive://file-id` (file IDs are stable even when files are renamed or moved). Freshness via `modifiedTime` from Drive metadata API. Handles Google Workspace native formats (Docs, Sheets, Slides) by exporting to PDF or plain text before extraction. Pure Python via `google-api-python-client`.

Note: shared drives and folder traversal require a crawler variant analogous to the existing directory crawler.

### SharePoint `[pending]`

Two sub-variants:

**SharePoint Online (Microsoft 365):** REST via Microsoft Graph API. URI scheme: `sharepoint://tenant/site/drive/item-id`. Freshness via `lastModifiedDateTime` from Graph item metadata. OAuth 2.0 client credentials flow for service account access. `msal` for token acquisition.

**SharePoint On-Premises (legacy):** REST via SharePoint 2013+ REST API or SOAP (`_vti_bin`). URI scheme: `sharepoint-onprem://host/site/item-path`. NTLM or Kerberos auth via `requests-ntlm`. Freshness via `Modified` field from list item metadata.

Both variants surface the SharePoint site/library/folder hierarchy as breadcrumbs, which maps directly to the chunk section structure.

### Git Repository `[pending]`

Index files from a Git repository at a specific ref (branch, tag, or commit SHA). URI scheme: `git://repo-path@ref/file-path`. Freshness via commit SHA of the last commit touching the file — if the SHA hasn't changed, the file hasn't changed.

Two modes:
- **Working tree:** read files from disk at the repo path (delegates to local transport after resolving ref)
- **Bare / remote:** read blob objects directly from Git object store via `gitpython`, without checkout

Useful for indexing documentation, source code, and configuration files with stable, auditable provenance.

### IMAP — Configurable Folder Selection `[pending]`

The existing IMAP transport indexes a single mailbox folder (defaulting to INBOX). Sent mail, drafts, and archive folders use provider-specific names (`Sent`, `Sent Items`, `[Gmail]/Sent Mail`). Extend the transport with:

- `folders: list[str]` parameter — explicit folder list to index
- `LIST` command support to enumerate available folders
- URI scheme extended to include folder: `imap://user@host/FOLDER-NAME/<message-id>`

This covers sent mail without a separate SMTP transport. SMTP is a delivery protocol, not a retrieval protocol, and cannot be used to read already-sent messages.

---

## Enhancement 4: Additional Extractors `[pending]`

### RTF `[pending]`

Rich Text Format — pervasive in legal and healthcare legacy systems. Pure Python via `striprtf`. Strips RTF control words and returns plain text; table detection via the existing `is_table_line` / `merge_blocks` pipeline applies downstream.

### MSG `[pending]`

Outlook `.msg` files — complement to the IMAP transport for locally-exported or forwarded email. Pure Python via `extract-msg`. Extracts body, subject, sender, recipients, and attachments. Attachments produce child `source_uri` entries under the Enhancement 1 hierarchy (e.g. `file:///path/to/message.msg/attachment.pdf`).

### ZIP and TAR `[pending]`

Container extractors — unpack the archive and recursively dispatch each member to the appropriate extractor based on its filename/MIME type. Not text extractors themselves; they are multiplexers.

Key behaviours:
- Each member produces a child `source_uri`: `zip:///path/to/archive.zip/member.pdf`, `tar:///path/to/archive.tar.gz/dir/member.docx`
- The archive itself is the parent document in the Enhancement 1 registry; members are children
- Nested archives (zip-in-zip) are recursed up to a configurable depth limit (default 2)
- Password-protected archives: fail loudly, no silent skip

### Image + OCR via LLM `[pending]`

Pass image bytes (PNG, JPG, TIFF, BMP) directly to a multimodal LLM (Claude) for text extraction. Advantages over Tesseract:
- Pure Python — no binary dependency
- Handles handwriting, rotated text, complex layouts, mixed language
- Tables in images can be extracted as markdown tables, feeding directly into the existing table-merge pipeline

The extractor takes an optional `llm_client` at construction. If not supplied, the extractor raises `NotImplementedError` rather than silently degrading.

Prompt to LLM: extract all text from the image verbatim; render any tables as pipe-delimited markdown; preserve heading hierarchy with `#` markers where visually apparent.

---

## Enhancement 1: Document Registry with Source URI and Freshness `[pending]`

### Problem

`DocumentChunk.document_name` is a caller-supplied display label — not unique, not a locator. There is no way to:
- Reliably identify the original source of a chunk
- Detect whether a source document has changed (staleness check)
- Avoid re-ingesting unchanged documents
- Represent the parent-child relationship between a document and its attachments
- Re-fetch and re-ingest a document given only a returned chunk
- Reconstruct a simplified approximation of a document when the source is unavailable

### Design

#### Document Registry Table

A new `documents` table with a surrogate PK, referenced by FK from `embeddings`:

```sql
CREATE TABLE documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_uri      TEXT UNIQUE,          -- canonical locator; NULL if no URI (rare)
    display_name    TEXT NOT NULL,        -- human label (formerly document_name)
    freshness_meta  TEXT,                 -- JSON; transport-specific staleness tokens
    created_at      TEXT NOT NULL,        -- ISO-8601
    updated_at      TEXT NOT NULL         -- ISO-8601; set on re-ingest
);
```

`embeddings.document_name` is replaced by `embeddings.document_id INTEGER REFERENCES documents(id)`.

#### Source URI

Every chunk has a canonical `source_uri`. Rules:

- Filesystem: `file:///absolute/path/to/doc.pdf`
- HTTP/HTTPS: full URL including query string if relevant
- S3: `s3://bucket/key`
- SFTP/FTP: `sftp://host/path`, `ftp://host/path`
- IMAP body: `imap://user@host/MAILBOX/<message-id>`
- IMAP attachment: `imap://user@host/MAILBOX/<message-id>/<part-filename>`
- SQL query result: `sqlalchemy://<connection-slug>/<query-hash>`
- Raw bytes (no transport): caller must supply a URI; no fallback

`source_uri` is always required. The loader enforces this. Callers of `load_bytes` must pass a meaningful URI — they always know why they have those bytes.

#### Synthetic URI Hierarchy for Attachments

Attachments are child documents of their parent. The URI encodes the relationship:

```
imap://user@host/INBOX/<message-id>              ← email body
imap://user@host/INBOX/<message-id>/report.pdf   ← PDF attachment
imap://user@host/INBOX/<message-id>/data.xlsx    ← spreadsheet attachment
```

Querying all chunks from one email: `WHERE source_uri LIKE 'imap://.../<message-id>/%'`

Parent document freshness applies to all children unless the child has independent staleness signals.

#### Freshness Metadata

`freshness_meta` is a transport-owned JSON blob. Each transport defines what it stores and how to interpret it for staleness detection. Examples:

| Transport | Freshness signals |
|-----------|------------------|
| Local filesystem | `{"mtime": 1713200000, "size": 204800}` |
| HTTP | `{"etag": "\"abc123\"", "last_modified": "Wed, 16 Apr 2026 10:00:00 GMT"}` |
| S3 | `{"etag": "abc123", "last_modified": "2026-04-16T10:00:00Z"}` |
| IMAP | `{"uid": 12345, "internaldate": "16-Apr-2026 10:00:00 +0000"}` |
| SQL | `{"query_hash": "sha256:...", "row_count": 512}` |

#### Transport Protocol Changes

The transport owns URI construction and freshness evaluation — not the extractor (which only sees bytes).

`FetchResult` gains two fields:

```python
@dataclass
class FetchResult:
    data: bytes
    source_uri: str                  # canonical, always present, transport-constructed
    source_path: str                 # local path for type detection (may equal source_uri)
    detected_mime: str | None
    freshness_meta: dict             # transport-specific; stored in documents table
```

The transport protocol gains a freshness check method:

```python
class Transport(Protocol):
    def fetch(self, uri: str) -> FetchResult: ...
    def check_freshness(self, source_uri: str, stored_meta: dict) -> bool:
        """Return True if the document at source_uri has changed since stored_meta was recorded."""
        ...
```

The loader calls `check_freshness` before re-fetching. If unchanged, skip re-ingest.

#### Extractor Protocol

`source_path` becomes required (not optional):

```python
class Extractor(Protocol):
    def extract(self, data: bytes, source_path: str) -> str: ...
    def can_handle(self, doc_type: str) -> bool: ...
```

#### Deduplication on Ingest

```
1. Resolve source_uri (transport constructs it)
2. Look up source_uri in documents table
3. If found:
   a. Call transport.check_freshness(source_uri, stored_meta)
   b. If unchanged: skip re-ingest, return existing chunks
   c. If changed: delete existing chunks, re-ingest, update documents row
4. If not found: ingest, insert documents row
```

#### Reload from Chunk

Given any `DocumentChunk` (e.g. returned from search), the caller can trigger a reload of its source document:

```python
store.reload_document(chunk)
# 1. Look up documents row via chunk.document_id
# 2. Call transport.check_freshness(source_uri, freshness_meta)
# 3. If stale (or force=True):
#    a. delete_by_document(document_id)
#    b. Re-fetch via transport.fetch(source_uri)
#    c. Re-extract, re-chunk, re-embed
#    d. Update documents row (freshness_meta, updated_at)
# 4. Return new chunks
```

This is the primary entry point for freshness-driven refresh triggered by a user noticing a stale result.

#### Document Reconstruction from Chunks

When `source_uri` is unavailable (source deleted, access revoked, bytes-only ingest with no transport), a simplified approximation of the document can be reconstructed from its stored chunks:

```python
store.reconstruct_document(document_id) -> str
# 1. Fetch all chunks for document_id, ordered by chunk_index
# 2. Reassemble content in order
# 3. Prepend section headers at boundaries (from chunk.section)
# 4. Return as plain text or markdown
```

This is explicitly a **lossy approximation** — chunk boundaries, merged tables, and list blocks mean the reconstruction is not identical to the original. It is useful for:
- Displaying context when the source is gone
- Feeding a reconstructed document to an LLM when re-fetch is impossible
- Debugging chunk coverage

The reconstructed document carries a warning header indicating it was assembled from chunks, not retrieved from source.

---

## Enhancement 2: Enhanced Semantic Similarity Search `[complete]`
## NER, Cluster, and Cohort Assembly Design

### Overview

An enhanced semantic similarity search that accepts a query and returns a top-k cohort of chunks, identical in interface to standard semantic similarity search, but assembles the cohort using four pre-computed dimensions: semantic similarity, structural adjacency, entity adjacency, and cluster adjacency.

From the caller's perspective, nothing changes. Query in, ranked chunks out. The method of deriving the cohort is different.

### Architecture

#### Three Layers, All Pre-Computed at Index Time

```
Documents
    │
    ▼
┌─────────────────────────────┐
│  Layer 1: Contextual Chunks │  (existing)
│  - Structure-preserving     │
│  - Breadcrumbs              │
│  - Continuation markers     │
│  - Next/prev/parent links   │
│  - Embedded with BGE        │
│  - FAISS indexed            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Layer 2: Entity Index      │  (new)
│  - NER against vocabulary   │
│  - Entity-chunk associations│
│  - Association scores       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Layer 3: Cluster Map       │  (new)
│  - Entity co-occurrence     │
│  - Cluster assignments      │
│  - Cluster neighbor index   │
└─────────────────────────────┘
```

All three layers are built at index time. All are deterministic. All are incrementally updatable. No LLM is involved.

### Layer 2: Entity Index

#### Vocabulary Sources

**Production (enterprise):**
- Customer/counterparty registries from CRM or risk systems
- Employee directories from HR systems
- Vendor/supplier catalogs from procurement systems
- RDB schema metadata: table names, column names, data types, foreign keys, business glossary descriptions
- Regulatory entity lists: agency names, regulation identifiers, CFR citations

**Benchmark (public corpus):**
- SEC EDGAR registrant index: company names, CIK numbers, ticker symbols, SIC codes
- ClinicalTrials.gov structured fields: drug names, conditions, sponsors, investigators
- FDA National Drug Code directory: drug names, active ingredients, manufacturers
- Federal Register API metadata: agency names, CFR parts, docket identifiers
- NIST publication metadata: control identifiers, family names
- Standard NER (spaCy): person names, organization names, locations, dates

#### NER Pass

Runs at index time over every chunk. For each chunk:

1. Match against the vocabulary using exact match, case-insensitive match, and configurable alias matching (e.g., "HCA" matches "HCA Healthcare" matches "Hospital Corporation of America").
2. Record each match as an entity-chunk association.

**Output per association:**

```json
{
    "entity_id": "ent_hca_healthcare",
    "chunk_id": "chunk_4327",
    "frequency": 3,
    "positions": [124, 456, 892],
    "score": 0.85
}
```

**Association score** is a function of:
- Frequency of the entity in the chunk (more mentions = stronger association)
- Position in the chunk (entity in first sentence scores higher than entity in last sentence, indicating topical centrality)
- Entity specificity (a rare entity appearing in a chunk is more informative than a common entity)

Score formula (tunable):

```
score = frequency_weight * log(1 + frequency)
      + position_weight * (1 - first_position / chunk_length)
      + specificity_weight * log(total_chunks / chunks_containing_entity)
```

Default weights: frequency 0.4, position 0.3, specificity 0.3.

#### Entity-Chunk Association Index

A bidirectional index:

**Entity to chunks:** For each entity, a ranked list of chunks where it appears, sorted by association score.

```
ent_hca_healthcare → [
    (chunk_4327, 0.92),
    (chunk_4331, 0.87),
    (chunk_12044, 0.71),
    ...
]
```

**Chunk to entities:** For each chunk, the list of entities it contains.

```
chunk_4327 → [
    (ent_hca_healthcare, 0.92),
    (ent_revenue_recognition, 0.78),
    (ent_reporting_threshold, 0.65)
]
```

Both directions are pre-computed and stored. Query-time lookup is O(1) per chunk or entity.

#### Incremental Update

When a new document enters the corpus:
1. Chunk the document.
2. Run NER on each new chunk.
3. Add new entity-chunk associations to the index.
4. If new entities appear (not in vocabulary), flag for vocabulary review. Do not auto-add.

When a document is removed:
1. Remove its chunks from the vector index.
2. Remove all entity-chunk associations referencing those chunks.
3. Cluster statistics update in the next cluster refresh cycle.

### Layer 3: Cluster Map

#### Construction

Build an entity co-occurrence matrix. Two entities co-occur when they appear in the same chunk. The co-occurrence count is the number of chunks where both entities appear.

```
Co-occurrence matrix (sparse):

                    ent_wire_transfer  ent_reporting_threshold  ent_ofac_screening
ent_hca_healthcare         12                    8                      3
ent_wire_transfer           -                   15                      9
ent_reporting_threshold     -                    -                      6
```

Normalize by entity frequency to produce a co-occurrence score (Jaccard, PMI, or simple conditional probability). This prevents high-frequency entities from dominating.

#### Clustering Algorithm

Run a standard clustering algorithm over the normalized co-occurrence matrix. Options:

- **Agglomerative clustering** with a distance threshold (simple, deterministic, no k parameter)
- **DBSCAN** (density-based, finds natural clusters without specifying count)
- **Leiden** (community detection, same algorithm GraphRAG uses for graph communities, but applied to co-occurrence rather than LLM-inferred relationships)

The choice is a tuning decision. All are deterministic given the same input.

**Output:**

```json
{
    "cluster_id": "cluster_017",
    "entities": [
        "ent_wire_transfer",
        "ent_reporting_threshold",
        "ent_ofac_screening",
        "ent_beneficiary_country_code",
        "ent_category_a_transactions"
    ],
    "cohesion_score": 0.72
}
```

#### Cluster Neighbor Index

For each entity, its cluster assignment and the other entities in the same cluster.

```
ent_wire_transfer → cluster_017 → [
    ent_reporting_threshold,
    ent_ofac_screening,
    ent_beneficiary_country_code,
    ent_category_a_transactions
]
```

For each cluster neighbor entity, the entity-chunk association index provides the ranked list of chunks. This is the two-hop path: query chunk → entity → cluster → neighbor entity → neighbor chunks.

#### Incremental Update

When new chunks are added, their entity associations update the co-occurrence matrix. Clusters can be refreshed periodically (nightly, weekly) rather than on every document addition, since cluster structure is statistical and small perturbations don't materially change the topology. Full cluster recomputation is cheap (it's a matrix operation over the entity vocabulary, not over the chunk corpus).

### Enhanced Semantic Similarity Search

#### Interface

Identical to standard semantic similarity search:

```python
def search(query: str, k: int = 5) -> list[ScoredChunk]:
    """
    Input:  query string, target cohort size
    Output: ranked list of (chunk, score) tuples, len <= k
    """
```

The caller does not know or care that the cohort was assembled from four dimensions. The output is a ranked list of chunks with scores.

#### Cohort Assembly Algorithm

```
┌──────────────────────────────────────────────────────────────┐
│  1. SEED: Semantic similarity (vector search)                │
│     - Embed query                                            │
│     - Retrieve top-m candidates from FAISS (m = 3 * k)      │
│     - These are the seed chunks                              │
├──────────────────────────────────────────────────────────────┤
│  2. STRUCTURAL: Next/prev/parent expansion                   │
│     - For each seed chunk:                                   │
│       - If continuation marker present: pull prev chunk      │
│       - If chunk ends with cross-reference: pull next chunk  │
│       - Optionally: pull parent for hierarchical context     │
│     - Add to candidate pool                                  │
├──────────────────────────────────────────────────────────────┤
│  3. ENTITY: Entity adjacency expansion                       │
│     - Extract entities from seed + structural chunks         │
│       (pre-computed, index lookup, not runtime NER)          │
│     - For each entity, pull top-n associated chunks          │
│       from entity-chunk index                                │
│     - Add to candidate pool                                  │
├──────────────────────────────────────────────────────────────┤
│  4. CLUSTER: Cluster adjacency expansion (budget-limited)    │
│     - For entities from Steps 1-3, look up cluster neighbors │
│     - For each neighbor entity, pull top-1 associated chunk  │
│     - Add to candidate pool                                  │
│     - Budget: max 2 * k cluster-adjacent candidates          │
├──────────────────────────────────────────────────────────────┤
│  5. SELECT: Score and assemble final cohort                  │
│     - Deduplicate candidate pool                             │
│     - Score each candidate (see scoring below)               │
│     - Select top-k by composite score                        │
│     - Return ranked list                                     │
└──────────────────────────────────────────────────────────────┘
```

#### Candidate Pool Sizing

For a target cohort of k=5:

- Step 1 (seed): 15 candidates (3 * k)
- Step 2 (structural): 0-15 additional (depends on continuation markers and cross-references, typically 3-8)
- Step 3 (entity): up to 15 additional (top-3 per entity, capped)
- Step 4 (cluster): up to 10 additional (2 * k, budget-limited)

Total candidate pool: typically 25-50 candidates, from which 5 are selected. The pool is small enough to score in microseconds.

#### Scoring Function

Each candidate receives a composite score based on how it entered the pool and what it contributes to the cohort.

**Relevance component (0-1):** Cosine similarity between the candidate's embedding and the query embedding. All candidates get this score, regardless of which dimension produced them. This ensures that entity-adjacent and cluster-adjacent chunks that happen to also be semantically relevant score highest.

**Source priority component (0-1):**
- Seed chunk: 1.0
- Structural neighbor of seed: 0.9
- Entity-adjacent to seed: 0.7
- Cluster-adjacent: 0.5

This ensures that when relevance scores are close, direct matches win over indirect expansions.

**Marginal coverage component (0-1):** Computed iteratively during selection. After each chunk is selected, the remaining candidates are penalized for redundancy with already-selected chunks. This is the MMR principle: semantic similarity to the query minus semantic similarity to the already-selected cohort.

```
coverage_score = query_similarity - lambda * max_similarity_to_selected
```

Lambda is tunable (default 0.3). Higher lambda penalizes redundancy more aggressively, producing more diverse cohorts.

**Composite score:**

```
composite = relevance_weight * relevance
          + priority_weight * source_priority
          + coverage_weight * coverage_score
```

Default weights: relevance 0.5, priority 0.2, coverage 0.3.

#### Selection Algorithm (Greedy Sequential)

```python
def select_cohort(candidates, query_embedding, k=5, lambda_diversity=0.3):
    selected = []
    remaining = list(candidates)

    for i in range(k):
        best_score = -1
        best_candidate = None

        for candidate in remaining:
            relevance = cosine_similarity(candidate.embedding, query_embedding)
            priority = candidate.source_priority

            if selected:
                max_sim_to_selected = max(
                    cosine_similarity(candidate.embedding, s.embedding)
                    for s in selected
                )
                coverage = relevance - lambda_diversity * max_sim_to_selected
            else:
                coverage = relevance

            composite = (0.5 * relevance
                       + 0.2 * priority
                       + 0.3 * coverage)

            if composite > best_score:
                best_score = composite
                best_candidate = candidate

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected
```

This is O(k * pool_size), which for k=5 and pool_size=50 is 250 comparisons. Negligible.

#### Provenance Tagging

Each chunk in the returned cohort carries a provenance tag indicating how it was selected:

```json
{
    "chunk_id": "chunk_4327",
    "text": "...",
    "score": 0.87,
    "provenance": "seed",
    "breadcrumb": "Compliance Framework > Wire Transfers > Reporting Requirements"
}
```

```json
{
    "chunk_id": "chunk_4331",
    "text": "...",
    "score": 0.72,
    "provenance": "entity_adjacent",
    "linked_by": "ent_reporting_threshold",
    "breadcrumb": "Compliance Framework > Wire Transfers > Exceptions"
}
```

```json
{
    "chunk_id": "chunk_8901",
    "text": "...",
    "score": 0.61,
    "provenance": "cluster_adjacent",
    "linked_by": "ent_ofac_screening",
    "cluster": "cluster_017",
    "breadcrumb": "AML Policy > Transaction Screening > OFAC Requirements"
}
```

The provenance tag is metadata, not part of the chunk text the LLM sees. But it's available for audit, for the UI to display, and for the LLM's system prompt to distinguish primary results from expansions.

#### What the LLM Receives

The LLM receives the cohort as a flat list of chunks, ordered by composite score, each with its breadcrumb prefix. The LLM does not see provenance tags or scoring details. It sees source text with hierarchical context labels, identical in format to what the chunking-only pipeline would produce.

The difference is in what's there, not how it's presented. The seed chunks answer the question as asked. The structural neighbors complete the local reading context. The entity-adjacent chunks fill gaps the seed chunks reference. The cluster-adjacent chunks surface cross-domain connections.

The LLM reads a briefing package, not a search result.

#### Optionally: Structured Prompt with Dimensions

For advanced use, the system can present the cohort to the LLM with dimension labels:

```
PRIMARY RESULTS (directly matching your query):
[seed chunk 1 with breadcrumb]
[seed chunk 2 with breadcrumb]

RELATED CONTEXT (sharing key entities with primary results):
[entity-adjacent chunk with breadcrumb]

BROADER CONTEXT (topically connected through related concepts):
[cluster-adjacent chunk with breadcrumb]
```

This gives the LLM explicit signal about which chunks are direct answers and which are supporting context, enabling more structured generation. Whether this improves answer quality versus a flat list is an empirical question testable in the CRQI benchmark.

### Configuration Parameters

#### NER Layer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocabulary_path` | required | Path to entity vocabulary file |
| `match_mode` | "case_insensitive" | Matching mode: exact, case_insensitive, alias |
| `alias_file` | null | Optional alias mapping (e.g., "HCA" → "HCA Healthcare") |
| `min_entity_length` | 2 | Minimum character length for entity matches |
| `score_weights` | [0.4, 0.3, 0.3] | Frequency, position, specificity weights |

#### Cluster Layer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm` | "agglomerative" | Clustering algorithm |
| `distance_threshold` | 0.5 | Cluster merge threshold (agglomerative) |
| `min_cooccurrence` | 2 | Minimum co-occurrence count to form an edge |
| `normalization` | "pmi" | Co-occurrence normalization: raw, jaccard, pmi |
| `refresh_interval` | "on_demand" | When to recompute clusters |

#### Cohort Assembly

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 5 | Target cohort size |
| `seed_pool_multiplier` | 3 | Candidate pool = k * multiplier |
| `entity_expansion_top_n` | 3 | Chunks per entity in expansion |
| `cluster_budget` | 10 | Max cluster-adjacent candidates (2 * k) |
| `lambda_diversity` | 0.3 | Redundancy penalty in coverage scoring |
| `relevance_weight` | 0.5 | Composite score weight for relevance |
| `priority_weight` | 0.2 | Composite score weight for source priority |
| `coverage_weight` | 0.3 | Composite score weight for marginal coverage |
| `structural_expansion` | true | Enable next/prev/parent expansion |
| `entity_expansion` | true | Enable entity adjacency expansion |
| `cluster_expansion` | true | Enable cluster adjacency expansion |

Each expansion dimension can be independently enabled or disabled, supporting incremental benchmarking: chunking-only (all expansions off), chunking+structural, chunking+structural+entity, full (all on).

### Incremental Pipeline

#### On Document Add

```
1. Chunk new document                         (deterministic)
2. Embed new chunks                           (deterministic)
3. Add to FAISS index                         (incremental)
4. Run NER on new chunks                      (deterministic)
5. Add entity-chunk associations              (incremental)
6. Update co-occurrence matrix                (incremental)
7. Flag cluster refresh needed                (deferred)
8. Update structural links (next/prev/parent) (deterministic)
```

#### On Document Remove

```
1. Remove chunks from FAISS index             (incremental)
2. Remove entity-chunk associations           (incremental)
3. Update co-occurrence matrix                (incremental)
4. Flag cluster refresh needed                (deferred)
5. Remove structural links                    (deterministic)
```

#### Cluster Refresh

Runs on demand or on a schedule. Recomputes clusters from the current co-occurrence matrix. Cheap (matrix operation over entity vocabulary, not over chunk corpus). Existing query results are unaffected until the refresh completes. New cluster assignments take effect atomically.

### Benchmark Integration

#### CRQI Benchmark Variants

Run the CRQI benchmark with four configurations to measure per-layer contribution:

| Variant | Seed | Structural | Entity | Cluster | Measures |
|---------|------|-----------|--------|---------|----------|
| A: Flat top-k | yes | no | no | no | Chunking-only baseline |
| B: Structural | yes | yes | no | no | Next/prev/parent contribution |
| C: Entity | yes | yes | yes | no | NER layer contribution |
| D: Full | yes | yes | yes | yes | Complete pipeline |

The delta between each successive variant measures the incremental contribution of each layer. If a layer doesn't move the CRQI, it doesn't ship.

#### GraphRAG-Bench Variants

Run GraphRAG-Bench with variants A and D. Variant A is the comparison against published GraphRAG scores. Variant D shows whether the NER/cluster layer closes any gap on Complex Reasoning (GraphRAG's strongest category) while maintaining the cost and determinism advantages.