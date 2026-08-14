# Defects #7–#8 — `clear()` parity and the missing re-derive path on service backends

Two defects on the full-reset path, found by running the phase 5 contract suite against a
live Postgres. Both make Postgres the wrong backend to re-ingest on.

| # | Defect | Location | Severity |
|---|--------|----------|----------|
| 7 | `clear()` deletes only the chunk rows on the four service backends, leaving the `documents` registry and all derived rows behind | `_pg.py:819`, `_qdrant.py:775`, `_pinecone.py:795`, `_weaviate.py:755` | silent data loss |
| 8 | `Ingestor.rebuild()` issues a DuckDB `PRAGMA` against `vector._conn`, an attribute no service backend has | `chonk/ingest.py:530` | re-derive unavailable off DuckDB |

Together these remove both reset lanes on Postgres: #8 removes re-derive-in-place, and #7
makes the remaining lane — wipe and re-ingest — quietly do nothing.

---

## Reproduction

```
docker run -d --name chonk-test-pg -e POSTGRES_PASSWORD=chonk -e POSTGRES_DB=chonk \
  -p 55432:5432 pgvector/pgvector:pg16
CHONK_TEST_PG_DSN="postgresql://postgres:chonk@localhost:55432/chonk" \
  python -m pytest tests/unit -q -p no:randomly --maxfail=50
```

```
1 failed, 1388 passed, 37 skipped in 17.74s
FAILED tests/unit/test_backend_registry_contract.py::TestSyncLifecycle::test_prune_documents_across_backends[pg]

tests/unit/test_backend_registry_contract.py:168: in test_prune_documents_across_backends
    assert [r.document_name for r in removed] == ["gone"]
E   AssertionError: assert ['doc', 'gone'] == ['gone']
```

`"doc"` belongs to the preceding test. The fixture calls `store.vector.clear()` on both
sides of its `yield` (`test_backend_registry_contract.py:60,63`), so the leak localises to
`clear()` and not to the assertion under test. The remaining 37 skips are `reportlab`,
`qdrant_client`, `weaviate`, and `pinecone` — client libraries absent from the
interpreter, and Pinecone has no local mode.

---

## Defect #7 — `clear()` does not cascade

DuckDB establishes the contract (`_vector.py:812-828`):

```python
def clear(self) -> None:
    """Delete all chunks and everything derived from them.

    Leaves the namespace/domain/source registries intact — those describe
    where content comes from, not the content itself.
    """
    self._conn.execute("DELETE FROM embeddings").fetchall()
    self._conn.execute("DELETE FROM documents").fetchall()
    for table in (
        *_cascade.CHUNK_KEYED_TABLES,
        "entities",
        "entity_aliases",
        "context_graph_edges",
    ):
        if self._table_exists(table):
            self._conn.execute(f"DELETE FROM {table}").fetchall()
    self._fts_dirty = True
```

Postgres deletes one table (`_pg.py:819-824`):

```python
def clear(self) -> None:
    """Delete all chunks from the table."""
    self._ensure_connection()
    with self._pgconn.cursor() as cur:
        cur.execute(f"DELETE FROM {self._table}")
    self._pgconn.commit()
```

Qdrant, Pinecone, and Weaviate drop and recreate their remote collection, then clear only
`embeddings` (plus their own id map) from the DuckDB catalog: `_qdrant.py:785`,
`_pinecone.py:807-808`, `_weaviate.py:758-759`. None of the four touches `documents`,
`chunk_entities`, `chunk_clusters`, `svo_triples`, `entities`, `entity_aliases`, or
`context_graph_edges`.

This is defect #6's failure mode reached through a second door. Phase 5 fixed
`delete_by_document`, which does delete the registry row on every backend
(`_pg.py:807-810`), but `clear()` was never brought along.

### Why the registry row is the wrong thing to keep

`clear()` has exactly one stated contract in the codebase, at `_vector.py:993-998`:

```python
raise ValueError(
    f"prune_documents() received an empty 'present' set while {len(registered)} "
    f"documents are registered — this would delete the entire index. If that is "
    f"intended, call backend.clear() instead."
)
```

"Delete the entire index" deliberately — the escape hatch for a raw-to-stage re-ingest. A
surviving `documents` row defeats precisely that: `sync_document` compares the incoming
content hash against the stored one, finds it unchanged, and returns `"skipped"`. The
re-ingest reports success against an empty index. That is the wording phase 5 already used
for #6, and it holds here unchanged.

Re-vectorizing is a separate lane and does not route through `clear()` —
`Ingestor.rebuild()` (`ingest.py:508-553`) calls `build_namespace_async(..., force=True)`,
documented at `lifecycle.py:45,58` as `Phases: crawl → chunk → embed → NER → community →
FTS` / `force: Rebuild even if namespace_cache_valid() is True`. It re-derives in place and
keeps both the chunks and the registry, which is what a changed embedding model or NER
vocabulary calls for. So the two behaviours are not two defensible readings of `clear()`;
one of them already has its own entry point.

`grep -rn "\.clear()" chonk/` finds no caller expecting the registry to survive — every
other hit is a local `dict`, `set`, or `Event`.

### Change

`clear()` cascades on all four service backends, matching DuckDB: delete the chunk rows,
the `documents` row, `CHUNK_KEYED_TABLES`, `entities`, `entity_aliases`, and
`context_graph_edges`, guarded by the existing `_table_exists` check. Postgres addresses
`self._docs_table`; the other three address their catalog's `documents` table via
`self._catalog` (`_qdrant.py:277`, `_weaviate.py:330`, `_pinecone.py:315`).

Route it through `_cascade` rather than open-coding the table list a fifth time — a shared
helper taking the same `(run, table_exists)` callables that
`_cascade.delete_chunk_dependents` already accepts. Five copies of the table tuple is how
`clear()` came to disagree with `delete_by_document` in the first place.

Postgres wraps the deletes in the single existing transaction, committing once, so a
failure mid-cascade cannot leave the registry emptied while entities survive.

### Tests — `tests/unit/test_backend_registry_contract.py`

- `test_clear_empties_registry` — after `clear()`, `list_documents()` is empty and
  `get_document_hash(name)` is `None` for every previously indexed document.
  **Regression test for #7.**
- `test_clear_then_sync_reindexes` — index, `clear()`, then `sync_document` with the *same*
  content hash returns `"added"`, not `"skipped"`. This is the silent-data-loss path, and
  the assertion that would have caught #7 directly.
- `test_clear_drops_derived_rows` — with NER and graph build run, `clear()` leaves
  `chunk_entities`, `entities`, `entity_aliases`, and `context_graph_edges` empty. Guards
  the retrieval consequence: orphaned rows keep satisfying `entity_types && …` and keep
  their graph edges, so a cleared index still returns entity matches for deleted content.
- `test_clear_preserves_source_registries` — namespace, domain, and source rows survive, per
  the DuckDB docstring. Pins the boundary so the fix does not overshoot.

Each is parametrized over the existing `backend` fixture, so DuckDB always runs and the
service backends run wherever their env var and client library are present.

---

## Defect #8 — `rebuild()` is DuckDB-only

`chonk/ingest.py:530`:

```python
db_path = self._store.vector._conn.execute("PRAGMA database_list").fetchone()[2]
```

Two DuckDB assumptions in one line: the attribute name `_conn`, and `PRAGMA
database_list`. `PgVectorBackend` holds `self._pgconn` (`_pg.py:98,137,178`) and has no
`_conn` at all, so `rebuild()` raises `AttributeError` on Postgres before reaching any
query. The three catalog-backed backends do own a DuckDB handle, but it is `self._catalog`
and its path is the catalog file, not the vector store.

`rebuild()` is the only public re-derive entry point, so on Postgres there is currently no
way to rebuild NER, community, or FTS output over chunks already indexed — which is what
pushes a caller toward `clear()` and into #7.

### Change

`build_namespace_async` takes the store, not a `db_path` recovered by introspecting a
private connection. Where a path is genuinely needed for the background thread's own
connection, the backend supplies it through a protocol method rather than the caller
guessing at internals.

`VectorBackend` (`chonk/storage/_protocol.py`) is the right home for that declaration, the
same widening phase 5 applied to `sync_document` and `prune_documents`. No fallback: a
backend that cannot hand out a connection target raises with a message naming the backend
and the operation.

### Tests

- `test_rebuild_runs_on_pg` — `rebuild(namespace_id=...)` with `async_=False` completes
  against `CHONK_TEST_PG_DSN` and repopulates `chunk_entities`. **Regression test for #8.**
- `test_rebuild_preserves_chunks_and_registry` — chunk count and `get_document_hash` are
  unchanged across a rebuild, which is the property distinguishing this lane from `clear()`.
- `test_rebuild_unsupported_backend_raises` — the error names the backend and the
  operation; it does not return `None` or no-op.

---

## Verification

```
docker run -d --name chonk-test-pg -e POSTGRES_PASSWORD=chonk -e POSTGRES_DB=chonk \
  -p 55432:5432 pgvector/pgvector:pg16
export CHONK_TEST_PG_DSN="postgresql://postgres:chonk@localhost:55432/chonk"
pytest tests/unit/test_backend_registry_contract.py -v -p no:randomly
pytest tests/unit -q -p no:randomly --maxfail=50
ruff check chonk/ && mypy chonk/storage chonk/ner chonk/graph
```

The suite must be green with the Postgres lane enabled — the contract points run per
backend, none of them skipped when the DSN is set. A green run with the DSN unset does not
verify either defect.

Current: `1606 passed, 47 skipped` for the full suite with `CHONK_TEST_PG_DSN` set (the
count moves with every added test, so treat it as a floor, not a target). Remaining skips
are Qdrant, Pinecone, and Weaviate, which still have no live service here.
