# Copyright (c) 2025 Kenneth Stott. MIT License.
"""clear() cascades to derived tables; rebuild() is backend-aware."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from chonk.models import DocumentChunk
from chonk.storage._cascade import DERIVED_TABLES
from chonk.storage._store import Store

DIM = 4


def _chunk(name: str, content: str) -> DocumentChunk:
    return DocumentChunk(document_name=name, content=content, chunk_index=0, section=[])


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "idx.duckdb", embedding_dim=DIM)
    yield store
    store.close()


def _seed(store) -> str:
    store.add_document([_chunk("doc_a", "acme corp filed")], np.ones((1, DIM), dtype="float32"))
    (chunk_id,) = store.vector._conn.execute("SELECT chunk_id FROM embeddings").fetchone()
    store.vector.register_document("doc_a", "hash_v1", source_uri="file:///a.txt", chunk_count=1)
    store.vector._conn.execute(
        "INSERT INTO entities(id, name, display_name, entity_type) "
        "VALUES ('customer:acme_corp', 'acme corp', 'Acme Corp', 'customer')"
    )
    store.vector._conn.execute(
        "INSERT INTO chunk_entities(chunk_id, entity_id, frequency, positions_json, score, "
        "namespace) VALUES (?, 'customer:acme_corp', 1, '[]', 1.0, 'global')",
        [chunk_id],
    )
    store.add_entity_alias("acme", "customer:acme_corp")
    return chunk_id


class TestClearCascades:
    def test_clear_empties_every_derived_table(self, store):
        _seed(store)
        store.vector.clear()
        for table in DERIVED_TABLES:
            if not store.vector._table_exists(table):
                continue
            (count,) = store.vector._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
            assert count == 0, f"{table} survived clear()"

    def test_clear_then_sync_reindexes(self, store):
        """The assertion that catches it: a surviving documents row makes re-ingest a no-op."""
        _seed(store)
        assert store.vector.get_document_hash("doc_a") == "hash_v1"

        store.vector.clear()

        # sync() skips a document whose stored hash is unchanged. After a wipe the
        # registry must not claim the document is still indexed.
        assert store.vector.get_document_hash("doc_a") is None

        store.add_document([_chunk("doc_a", "acme corp filed")], np.ones((1, DIM), dtype="float32"))
        (count,) = store.vector._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        assert count == 1

    def test_clear_preserves_source_registries(self, store):
        """Boundary: registries describe where content comes from, not the content."""
        store.register_namespace("retail", description="retail division")
        store.register_domain("retail:support", "retail", "support")
        store.register_source("src_1", "retail:support", "directory", "/tmp/docs")
        _seed(store)

        store.vector.clear()

        assert store.list_namespaces() == ["retail"]
        assert store.list_domains("retail") == ["support"]
        (sources,) = store.vector._conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        assert sources == 1

    def test_clear_is_idempotent(self, store):
        _seed(store)
        store.vector.clear()
        store.vector.clear()
        (count,) = store.vector._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        assert count == 0


class TestRebuildTarget:
    """rebuild() must identify the backend before deriving a path from it."""

    def _ingestor(self, store):
        from chonk.ingest import Index

        ing = Index.__new__(Index)
        ing._store = store
        return ing

    def test_duckdb_returns_the_file_path(self, store, tmp_path):
        db_path, dsn = self._ingestor(store)._rebuild_target()
        assert db_path == str(tmp_path / "idx.duckdb")
        assert dsn is None

    def test_pg_returns_the_dsn_not_a_path(self, store):
        """PgVectorBackend has a _conn adapter, so the old PRAGMA reached Postgres
        as invalid syntax rather than failing fast."""
        fake = MagicMock()
        fake._dsn = "postgresql://user:pass@host/db"
        ing = self._ingestor(store)
        ing._store = MagicMock(vector=fake)

        db_path, dsn = ing._rebuild_target()
        assert dsn == "postgresql://user:pass@host/db"
        assert db_path == ""

    def test_external_backend_raises_clearly(self, store):
        fake = MagicMock(spec=[])  # no _dsn
        ing = self._ingestor(store)
        ing._store = MagicMock(vector=fake, _db=None)

        with pytest.raises(NotImplementedError, match="Re-ingest the sources instead"):
            ing._rebuild_target()
