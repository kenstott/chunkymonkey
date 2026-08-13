# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 0e3b6a41-9c2f-4d18-8a75-2b41d0f6e9c7

"""Tests for chunk ID derivation and index schema versioning (fixed.md phase 1)."""

from __future__ import annotations

import pytest

from chonk.storage import SCHEMA_VERSION, SchemaVersionError, Store
from chonk.storage._vector import DuckDBVectorBackend

_gen = DuckDBVectorBackend._generate_chunk_id


# ---------------------------------------------------------------------------
# Chunk ID derivation
# ---------------------------------------------------------------------------


class TestChunkId:
    def test_id_changes_when_content_differs_past_100_chars(self):
        """Regression: defect #3 — the hash truncated content to 100 characters, so
        two chunks sharing a prefix collided and stale entity rows rebound to new
        content on an incremental update."""
        prefix = "A" * 100
        a = _gen("doc", 0, prefix + "original tail")
        b = _gen("doc", 0, prefix + "revised tail")
        assert a != b

    def test_id_changes_on_single_char_edit_deep_in_content(self):
        base = "B" * 5000
        assert _gen("doc", 0, base) != _gen("doc", 0, base[:-1] + "C")

    def test_id_stable_for_identical_content(self):
        assert _gen("doc", 3, "same content") == _gen("doc", 3, "same content")

    def test_id_varies_by_document(self):
        assert _gen("doc-a", 0, "text") != _gen("doc-b", 0, "text")

    def test_id_varies_by_chunk_index(self):
        assert _gen("doc", 0, "text") != _gen("doc", 1, "text")

    def test_id_is_prefixed_with_document_and_index(self):
        assert _gen("doc", 7, "text").startswith("doc_7_")


# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "index.duckdb"


def _stamped_version(path):
    import duckdb

    conn = duckdb.connect(str(path))
    try:
        row = conn.execute("SELECT version FROM schema_meta WHERE id = 1").fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _chunk_count(path):
    import duckdb

    conn = duckdb.connect(str(path))
    try:
        return conn.execute("SELECT COUNT(*) FROM embeddings").fetchall()[0][0]
    finally:
        conn.close()


def _write_raw(path, *statements):
    import duckdb

    conn = duckdb.connect(str(path))
    try:
        for sql in statements:
            conn.execute(sql)
    finally:
        conn.close()


class TestSchemaVersion:
    def test_new_index_is_stamped(self, db_path):
        with Store(db_path, embedding_dim=4):
            pass
        assert _stamped_version(db_path) == SCHEMA_VERSION

    def test_reopen_current_version_succeeds(self, db_path):
        with Store(db_path, embedding_dim=4):
            pass
        with Store(db_path, embedding_dim=4) as store:
            assert store.count() == 0

    def test_open_stale_schema_version_raises(self, db_path):
        """An index stamped with an older version must be rejected, not migrated."""
        with Store(db_path, embedding_dim=4) as store:
            store.add_document(*_one_chunk())
        _write_raw(db_path, "UPDATE schema_meta SET version = 1 WHERE id = 1")

        with pytest.raises(SchemaVersionError) as exc:
            with Store(db_path, embedding_dim=4):
                pass
        assert "version 1" in str(exc.value)
        assert "Rebuild" in str(exc.value)

    def test_open_unstamped_populated_index_raises(self, db_path):
        """A pre-versioning index that already holds chunks is incompatible."""
        with Store(db_path, embedding_dim=4) as store:
            store.add_document(*_one_chunk())
        _write_raw(db_path, "DROP TABLE schema_meta")

        with pytest.raises(SchemaVersionError) as exc:
            with Store(db_path, embedding_dim=4):
                pass
        assert "pre-2" in str(exc.value)

    def test_open_unstamped_empty_index_is_adopted(self, db_path):
        """An empty database has nothing stale in it — stamp and continue."""
        with Store(db_path, embedding_dim=4):
            pass
        _write_raw(db_path, "DROP TABLE schema_meta")

        with Store(db_path, embedding_dim=4) as store:
            assert store.count() == 0
        assert _stamped_version(db_path) == SCHEMA_VERSION

    def test_incompatible_index_is_left_untouched(self, db_path):
        """Verification runs before any DDL, so a rejected index keeps its rows."""
        with Store(db_path, embedding_dim=4) as store:
            store.add_document(*_one_chunk())
        _write_raw(db_path, "UPDATE schema_meta SET version = 1 WHERE id = 1")

        with pytest.raises(SchemaVersionError):
            with Store(db_path, embedding_dim=4):
                pass

        assert _chunk_count(db_path) == 1
        assert _stamped_version(db_path) == 1

    def test_memory_store_is_stamped(self):
        with Store(":memory:", embedding_dim=4) as store:
            backend = store.vector
            assert isinstance(backend, DuckDBVectorBackend)
            assert backend._stored_schema_version() == SCHEMA_VERSION


def _one_chunk():
    import numpy as np

    from chonk.models import DocumentChunk

    chunk = DocumentChunk(document_name="doc", content="hello world", chunk_index=0)
    return [chunk], np.zeros((1, 4), dtype="float32")
