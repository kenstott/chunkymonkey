# Incremental Update — Remediation Plan

Fixes the six defects found in the `sync_document` / `delete_by_document` path.

| # | Defect | Location | Phase |
|---|--------|----------|-------|
| 1 | Deleted source documents never removed | no prune API exists | 3 |
| 2 | Derived rows orphaned on delete/update | `chonk/storage/_vector.py:628` | 2 |
| 3 | `chunk_id` hashes only `content[:100]` | `chonk/storage/_vector.py:175` | 1 |
| 4 | NER incremental gate skips updated chunks | `chonk/ner/_build.py:88` | 4 |
| 5 | Community cache invalidation is count-only | `chonk/storage/_store.py:220` | 4 |
| 6 | Non-DuckDB backends leave stale registry row → silent data loss | `_pg.py:690`, `_qdrant.py:650`, `_pinecone.py:683`, `_weaviate.py:634` | 5 |

Ordering rationale: #3 must land first — every downstream invalidation check keys off
`chunk_id`, so a collision-prone id makes phases 2/4 unverifiable. Phase 2 must land
before 4 because the NER gate reads `chunk_entities`, which phase 2 makes truthful.

---

## Phase 1 — Content-complete chunk IDs

**Change:** `DuckDBVectorBackend._generate_chunk_id` (`_vector.py:175`) hashes the full
`content`, not `content[:100]`.

```python
content_hash = hashlib.sha256(
    f"{document_name}:{chunk_index}:{content}".encode()
).hexdigest()[:16]
```

**Breaking:** existing databases produce different ids for identical content. Add a
schema-version row and require `--force` reindex when the stored version predates this
change; do not attempt in-place id rewriting (embeddings, `chunk_entities`,
`chunk_clusters`, and `context_graph_edges` would all need coordinated rekeying).

**Add:** `SCHEMA_VERSION` constant + `schema_meta` table in `chonk/storage/_schema.py`;
`DuckDBVectorBackend._init_schema` raises on a version mismatch. No silent migration.

### Tests — `tests/unit/test_chunk_id.py`
- `test_id_changes_when_content_differs_past_100_chars` — two chunks sharing a 100-char
  prefix, same `document_name`/`chunk_index`, produce different ids. **This is the
  regression test for #3.**
- `test_id_stable_for_identical_content` — same inputs → same id across calls.
- `test_id_varies_by_document_and_index`
- `test_open_stale_schema_version_raises` — DB written with an older `SCHEMA_VERSION`
  raises on open; message names the required action.

---

## Phase 2 — Cascade deletes

**Change:** `DuckDBVectorBackend.delete_by_document` (`_vector.py:628`) captures
`chunk_id`s before deleting, then deletes from every chunk-keyed table:

```
SELECT chunk_id FROM embeddings WHERE document_name = ?   -- capture first
DELETE FROM chunk_entities  WHERE chunk_id IN (...)
DELETE FROM chunk_clusters  WHERE chunk_id IN (...)
DELETE FROM embeddings      WHERE document_name = ?
DELETE FROM documents       WHERE document_name = ?
```

Batch the `IN` lists at 1000 ids to stay under DuckDB's parameter ceiling.

**Entity-level GC:** after the chunk-entity deletes, remove entities that no longer have
any `chunk_entities` row, then the `entity_aliases` and `context_graph_edges` that
reference them. Extract as `_gc_orphaned_entities()` so phase 3's prune reuses it.

**Do not** add SQL `ON DELETE CASCADE` — DuckDB does not enforce it across these tables
and the chunk-id capture is needed regardless.

**Wire up `RelationalBackend`:** `delete_entities_by_document` (`_relational.py:131`)
already takes `chunk_ids`; call it from the `Store` layer when a relational backend is
attached, using the ids captured above.

### Tests — `tests/unit/test_incremental_cascade.py`
- `test_delete_removes_chunk_entities` — index doc, link entities, delete, assert
  `chunk_entities` count is 0 for those ids.
- `test_delete_removes_chunk_clusters`
- `test_update_leaves_no_orphaned_chunk_entities` — index v1, build entity links, sync v2
  (different content), reindex; assert no `chunk_entities` row references a `chunk_id`
  absent from `embeddings`. **Regression test for #2.**
- `test_update_same_chunk_count_does_not_rebind_stale_entities` — v1 and v2 chunk to the
  same count at the same indices; assert entities linked to v2 chunks are those extracted
  from v2, not v1.
- `test_orphaned_entity_removed` — entity referenced by exactly one document is gone after
  that document is deleted.
- `test_shared_entity_survives` — entity referenced by two documents survives deletion of
  one.
- `test_entity_aliases_and_graph_edges_follow_entity_gc`
- `test_delete_unknown_document_returns_zero`

**Invariant helper** (`tests/conftest.py`), asserted at the end of every incremental test:

```python
def assert_no_orphans(conn):
    """No chunk-keyed row may reference a chunk_id absent from embeddings."""
```

---

## Phase 3 — Prune removed documents

**Add:** `chonk/storage/_vector.py`

```python
def prune_documents(
    backend: DuckDBVectorBackend,
    present: Iterable[str],
    *,
    dry_run: bool = False,
) -> list[SyncResult]:
    """Delete every registered document absent from *present*.

    Returns one SyncResult per removed document with action="deleted".
    """
```

- `present` is the full set of document names currently in the source. Passing a partial
  set deletes real data, so the docstring states this and the function raises `ValueError`
  on an empty `present` when the registry is non-empty — that combination is far more
  often a bug than an intent to wipe the index. `backend.clear()` is the explicit way to
  empty it.
- Extend `SyncResult.action` to include `"deleted"`; update the class docstring
  (`_vector.py:46`).
- Export `prune_documents` from `chonk/storage/__init__.py` and `chonk/__init__.py`
  alongside `sync_document`.
- Update `demo/graphrag_bench.py:cmd_index` to collect `{doc_id for doc_id, _ in corpus}`
  and call `prune_documents` after the indexing loop; report `n_deleted` in the summary
  line at `demo/graphrag_bench.py:618`.

### Tests — `tests/unit/test_prune_documents.py`
- `test_prune_removes_absent_document` — two docs registered, one present → the other's
  chunks and registry row are gone. **Regression test for #1.**
- `test_prune_keeps_present_documents`
- `test_prune_returns_deleted_results` — action, name, `previous_chunk_count` correct.
- `test_prune_dry_run_deletes_nothing` — returns the same list, DB unchanged.
- `test_prune_empty_present_raises` — non-empty registry + empty `present` → `ValueError`.
- `test_prune_empty_present_on_empty_registry_is_noop`
- `test_prune_cascades` — deleted docs leave no orphans (`assert_no_orphans`).
- `test_prune_is_idempotent` — second call returns `[]`.

---

## Phase 4 — Cache invalidation keyed on content

**4a. NER gate** (`chonk/ner/_build.py:88`). With phase 2 cascading and phase 1's
content-complete ids, `processed_ids` becomes truthful: a changed chunk gets a new id, is
absent from `chunk_entities`, and lands in `new_chunk_ids`. The remaining change is to
stop trusting a bare fingerprint match — recompute `new_chunk_ids` before the
`config_match and not new_chunk_ids` early return and assert `processed_ids ⊆
all_chunk_ids` (phase 2 guarantees it; a violation means a delete path was missed).

**4b. Community cache** (`chonk/storage/_store.py:220`). Replace the `chunk_count`
comparison with a chunk-id-set fingerprint, reusing `_chunk_fingerprint` from
`chonk/graph/_context_graph.py:38`. Store it in a new `community_cache.chunk_fingerprint`
column; `write_community_cache` (`_store.py:239`) writes it, `community_cache_valid`
compares it. Bump `SCHEMA_VERSION`.

### Tests — `tests/unit/test_incremental_invalidation.py`
- `test_ner_reruns_after_document_update` — index v1, build NER, sync+reindex v2, assert
  the gate reports the changed chunks as new. **Regression test for #4.**
- `test_ner_skips_when_nothing_changed`
- `test_ner_processes_only_changed_chunks` — a two-document corpus where one changes;
  the unchanged document's chunks are not reprocessed.
- `test_community_cache_invalidated_when_content_changes_but_count_does_not` — v1 and v2
  chunk to identical counts; cache must report invalid. **Regression test for #5.**
- `test_community_cache_valid_when_unchanged`
- `test_context_graph_cache_invalidated_on_update`

---

## Phase 5 — Backend parity

**Change:** `delete_by_document` deletes the `documents` registry row in every backend:
`_pg.py:690`, `_qdrant.py:650`, `_pinecone.py:683`, `_weaviate.py:634`. PG uses
`self._docs_table`; the other three use their DuckDB catalog's `documents` table.

**Change:** `sync_document`'s signature widens from `DuckDBVectorBackend` to the
`VectorBackend` protocol (`chonk/storage/_protocol.py`), which already declares
`register_document`, `get_document_hash`, and `delete_by_document`. Add
`get_document_hash` and `list_documents` to the protocol — they are implemented by all
five backends but only the first three are declared.

`prune_documents` likewise takes the protocol type.

### Tests
Shared contract suite, `tests/unit/test_backend_registry_contract.py`, parametrized over
every available backend (each skipped when its client library or service is absent):

- `test_delete_clears_registry_row` — after `delete_by_document`, `get_document_hash`
  returns `None`. **Regression test for #6.**
- `test_delete_then_sync_reindexes` — delete, then `sync_document` with the *same* hash
  returns `"added"`, not `"skipped"`. This is the silent-data-loss path.
- `test_register_then_get_hash_roundtrip`
- `test_register_updates_existing_hash`
- `test_list_documents_reflects_deletes`
- `test_sync_add_update_skip_cycle` — the full lifecycle against each backend.
- `test_prune_documents_across_backends`

DuckDB runs in-memory. PG/Qdrant/Pinecone/Weaviate cases are marked `integration` and skip
without credentials — they must not silently pass as no-ops, so the suite asserts at least
the DuckDB parametrization ran.

---

## Verification

```
pytest tests/unit/test_chunk_id.py \
       tests/unit/test_incremental_cascade.py \
       tests/unit/test_prune_documents.py \
       tests/unit/test_incremental_invalidation.py \
       tests/unit/test_backend_registry_contract.py \
       tests/unit/test_sync_document.py -v
pytest tests/unit -q
ruff check chonk/ && mypy chonk/storage chonk/ner chonk/graph
```

Each phase lands green before the next begins. Phase 1 and 4b bump `SCHEMA_VERSION`;
existing indexes require a `--force` rebuild, which the release notes must state.
