# Horizontal Scale Plan

## Motivation

The current DuckDB backend is single-writer. It supports local development and single-process production deployments well, but cannot support concurrent ingestion workers. Switching to a PostgreSQL backend (pgvector + tsvector) enables horizontal scale as a natural consequence of PG's concurrent-write semantics — no application-level coordination required for chunk writes.

---

## Backend Swap

A `VectorBackend` protocol is extracted from `DuckDBVectorBackend`. `PGVectorBackend` implements the same interface against pgvector (vector search) and `tsvector` (BM25). The `Store` facade selects the backend from config.

**Protocol surface:**

```python
class VectorBackend(Protocol):
    def add_chunks(self, chunks, embeddings, namespace, source_id, domain_id, session_fingerprint) -> None: ...
    def register_document(self, document_name, content_hash, source_uri, chunk_count) -> None: ...
    def delete_by_document(self, document_name, *, gc_entities=True) -> int: ...
    def gc_orphaned_entities(self) -> int: ...
    def chunk_ids_for_document(self, document_name) -> list[str]: ...
    def clear(self) -> None: ...
    def search(self, query_embedding, limit, query_text, include_breadcrumbs, namespaces, chunk_types, domain_ids, session_fingerprint) -> list[tuple[str, float, object]]: ...
    def get_all_chunks(self) -> list: ...
    def get_document_hash(self, document_name) -> str | None: ...
    def list_documents(self) -> list[dict]: ...
    def count(self) -> int: ...
    def rebuild_fts_index(self) -> None: ...  # no-op for PG (live tsvector index)
    def preload_embeddings(self) -> None: ...  # no-op for PG (index-backed ANN)
```

`chunk_id` is the idempotency key. `add_chunks` uses `ON CONFLICT DO NOTHING` — concurrent workers writing the same chunk are safe.

---

## PG as Infrastructure Substrate

A single PG instance serves three roles:

| Role | Mechanism |
|------|-----------|
| Vector store | pgvector embeddings + `tsvector` BM25 |
| Ingest queue | `ingest_queue` table, `SELECT FOR UPDATE SKIP LOCKED` |
| Coordinator control | `control` table, pause/resume flag |

No additional infrastructure required. One connection string in config covers the full stack.

### Ingest Queue Schema

```sql
CREATE TABLE ingest_queue (
    id            BIGSERIAL PRIMARY KEY,
    source_uri    TEXT NOT NULL,
    namespace     TEXT NOT NULL,
    content_hash  TEXT,
    status        TEXT DEFAULT 'pending',  -- pending | processing | done | failed
    worker_id     TEXT,
    leased_at     TIMESTAMPTZ,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE control (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- coordinator sets key='workers_paused', value='1' during graph build
```

### Worker Dispatch

```sql
UPDATE ingest_queue
SET status = 'processing', worker_id = ?, leased_at = now()
WHERE id = (
    SELECT id FROM ingest_queue
    WHERE status = 'pending'
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
RETURNING *;
```

`SKIP LOCKED` prevents double-assignment without application-level locking. Stale leases (worker crash) are requeued by the coordinator: rows where `status = 'processing'` and `leased_at < now() - interval '10 minutes'`.

---

## Coordinator / Worker Pattern

Both roles are modes of a single script (`ingest.py`):

```bash
# Worker — as many processes as needed, local or remote
python ingest.py --worker --queue pg://... --backend pg://...

# Coordinator — one instance, owns graph build
python ingest.py --coordinator --queue pg://... --backend pg://... --graph-interval 300
```

Local development runs the coordinator mode with no queue — sequential, single-process, no PG required (falls back to DuckDB).

### Coordinator State Machine

```
DISPATCHING → (interval elapsed or queue drained)
    → DRAINING   (stop dispatching, wait for in-flight workers to ack)
    → BUILDING   (run graph build: co-occurrence + Louvain per namespace)
    → DISPATCHING
```

Workers check `control.workers_paused` at the top of each loop before pulling the next message. One extra PG read per document.

### Graph Build Frequency

| Corpus type | Recommended interval |
|-------------|---------------------|
| Live feed (CVE, regulatory) | 5–15 minutes |
| Periodic bulk (10-K filings) | After ingest completes |
| Static corpus | Once, on initial build |

Graph build is per-namespace and runs serially across namespaces. Namespaces with no new documents since the last build are skipped.

---

## Topology

```
                    ┌─────────────────────────────┐
                    │         PG Instance          │
                    │  • embeddings (pgvector)     │
                    │  • documents                 │
                    │  • ingest_queue              │
                    │  • control                   │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         worker_1         worker_2         worker_N
     chunk→NER→embed   chunk→NER→embed  chunk→NER→embed
       →write PG          →write PG       →write PG
              │                │                │
              └────────────────┴────────────────┘
                               │
                          coordinator
                    (graph build, lease requeue,
                     pause/resume signal)
```

Scale workers by running more processes — same machine or across machines. The coordinator runs once, anywhere with PG access.

---

## Namespace Partitioning

Each data source maps to a namespace. Namespaces are independent at index time — graph build for `sec` does not block `cve`. Workers are not namespace-specific; the queue row carries the namespace and workers write to the correct partition automatically.

User namespaces (`user:{user_id}`) are low-volume by design. The global namespace and high-volume data source namespaces are the scaling target.

---

## What DuckDB Retains

DuckDB remains the default backend for:
- Local development
- Single-process production (small corpora)
- Benchmarking and research (no external dependencies)

PG backend is opt-in via config. The `VectorBackend` protocol makes both backends interchangeable at the `Store` facade level.

---

## Implementation Sequence

1. Extract `VectorBackend` protocol from `DuckDBVectorBackend`
2. Implement `PGVectorBackend` (pgvector + tsvector + `ON CONFLICT DO NOTHING`)
3. Add backend factory to `Store` (config-driven)
4. Add `ingest_queue` and `control` schema to PG migration
5. Implement `--worker` and `--coordinator` modes in `ingest.py`
6. Add lease requeue and pause/resume to coordinator loop
