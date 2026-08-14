# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 21c77bc2-66c8-4969-9da8-ef7d6c93b4a8
"""Unit tests for PgVectorBackend — fully mocked, no real PostgreSQL required."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chonk.models import DocumentChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4


def _make_chunk(name: str = "doc", idx: int = 0, content: str = "hello") -> DocumentChunk:
    return DocumentChunk(
        document_name=name,
        content=content,
        chunk_index=idx,
        section=[],
        source_detail={"row_start": 1},
    )


def _make_embeddings(n: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# Mock psycopg2 + pgvector so the import succeeds without the real packages
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend():
    """PgVectorBackend with a fully mocked psycopg2 connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.autocommit = False
    mock_conn.closed = False

    mock_psycopg2 = MagicMock()
    mock_psycopg2.connect.return_value = mock_conn
    mock_psycopg2.OperationalError = Exception
    mock_psycopg2.InterfaceError = Exception

    mock_pgvec_psycopg2 = MagicMock()

    with (
        patch.dict(
            "sys.modules",
            {
                "psycopg2": mock_psycopg2,
                "pgvector": MagicMock(),
                "pgvector.psycopg2": mock_pgvec_psycopg2,
            },
        ),
    ):
        from importlib import reload

        import chonk.storage._pg as _pg_mod

        reload(_pg_mod)
        backend_obj = _pg_mod.PgVectorBackend.__new__(_pg_mod.PgVectorBackend)
        backend_obj._dsn = "postgresql://user:pass@localhost/test"
        backend_obj._embedding_dim = DIM
        backend_obj._table = "chonk_embeddings"
        backend_obj._docs_table = "chonk_documents"
        backend_obj._global_attached = False
        backend_obj._pgconn = mock_conn
        yield backend_obj, mock_conn, mock_cursor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPgVectorBackendInit:
    def test_missing_deps_raises(self):
        with patch.dict("sys.modules", {"psycopg2": None, "pgvector": None}):
            import importlib

            import chonk.storage._pg as _pg_mod

            importlib.reload(_pg_mod)
            with pytest.raises(ImportError, match="psycopg2 and pgvector"):
                _pg_mod._require_deps()


class TestAddChunks:
    def test_add_chunks_inserts_records(self, backend):
        backend_obj, mock_conn, mock_cursor = backend
        chunks = [_make_chunk("doc_a", 0), _make_chunk("doc_a", 1)]
        embeddings = _make_embeddings(2)

        backend_obj.add_chunks(chunks, embeddings, namespace="ns1")

        insert_calls = [c for c in mock_cursor.execute.call_args_list if "INSERT" in c[0][0]]
        assert len(insert_calls) == 2
        sql, params = insert_calls[0][0]
        assert "INSERT INTO chonk_embeddings" in sql
        assert "ON CONFLICT" in sql
        assert params[0].startswith("doc_a_0_")  # chunk_id
        assert params[1] == "doc_a"  # document_name
        assert params[9] == "ns1"  # namespace
        mock_conn.commit.assert_called_once()

    def test_add_chunks_empty_is_noop(self, backend):
        backend_obj, mock_conn, mock_cursor = backend
        mock_cursor.reset_mock()
        mock_conn.reset_mock()

        backend_obj.add_chunks([], _make_embeddings(0))

        mock_cursor.execute.assert_not_called()
        mock_conn.commit.assert_not_called()

    def test_source_detail_serialised_as_json(self, backend):
        backend_obj, _conn, mock_cursor = backend
        chunk = _make_chunk()
        chunk.source_detail = {"row_start": 3, "row_end": 5}
        embeddings = _make_embeddings(1)

        backend_obj.add_chunks([chunk], embeddings)

        insert_calls = [c for c in mock_cursor.execute.call_args_list if "INSERT" in c[0][0]]
        _, params = insert_calls[0][0]
        source_detail_arg = params[10]
        parsed = json.loads(source_detail_arg)
        assert parsed == {"row_start": 3, "row_end": 5}

    def test_no_namespace_stored_as_none(self, backend):
        backend_obj, _conn, mock_cursor = backend
        chunk = _make_chunk()
        backend_obj.add_chunks([chunk], _make_embeddings(1))

        insert_calls = [c for c in mock_cursor.execute.call_args_list if "INSERT" in c[0][0]]
        _, params = insert_calls[0][0]
        assert params[9] is None


class TestSearch:
    def _row(
        self,
        chunk_id="c1",
        doc="doc_a",
        section="[]",
        idx=0,
        content="hello",
        breadcrumb=None,
        chunk_type="document",
        src_off=None,
        src_len=None,
        src_det=None,
        sim=0.9,
    ):
        return (
            chunk_id,
            doc,
            section,
            idx,
            content,
            breadcrumb,
            chunk_type,
            src_off,
            src_len,
            src_det,
            sim,
        )

    def test_search_returns_scored_chunks(self, backend):
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchall.return_value = [
            self._row("c1", sim=0.9),
            self._row("c2", sim=0.7),
        ]

        results = backend_obj.search(_make_embeddings(1)[0], limit=2)

        assert len(results) == 2
        cid, score, chunk = results[0]
        assert cid == "c1"
        assert score == pytest.approx(0.9)
        assert isinstance(chunk, DocumentChunk)

    def test_search_includes_breadcrumb_in_content(self, backend):
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchall.return_value = [
            self._row("c1", content="body text", breadcrumb="[doc > sec]", sim=0.8),
        ]

        _, _, chunk = backend_obj.search(_make_embeddings(1)[0], limit=1)[0]
        assert "[doc > sec]" in chunk.content
        assert "body text" in chunk.content

    def test_search_without_breadcrumb(self, backend):
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchall.return_value = [
            self._row("c1", content="raw", breadcrumb=None, sim=0.8),
        ]

        _, _, chunk = backend_obj.search(
            _make_embeddings(1)[0], limit=1, include_breadcrumbs=False
        )[0]
        assert chunk.content == "raw"

    def test_search_with_namespace_filter(self, backend):
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchall.return_value = []

        backend_obj.search(_make_embeddings(1)[0], limit=5, namespaces=["ns1"])

        sql, params = mock_cursor.execute.call_args[0]
        assert "namespace = ANY(%s)" in sql
        assert any(p == ["ns1"] for p in params if isinstance(p, list))

    def test_search_with_chunk_type_filter(self, backend):
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchall.return_value = []

        backend_obj.search(_make_embeddings(1)[0], limit=5, chunk_types=["summary"])

        sql, params = mock_cursor.execute.call_args[0]
        assert "chunk_type = ANY(%s)" in sql
        assert any(p == ["summary"] for p in params if isinstance(p, list))

    def test_search_source_detail_deserialised(self, backend):
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchall.return_value = [
            self._row("c1", src_det='{"row_start": 2}', sim=0.9),
        ]

        _, _, chunk = backend_obj.search(_make_embeddings(1)[0], limit=1)[0]
        assert chunk.source_detail == {"row_start": 2}

    def test_hybrid_binds_query_text_as_parameter(self, backend):
        """query_text must never be interpolated into the SQL text."""
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchall.return_value = []
        payload = "'); DROP TABLE chonk_embeddings; --"

        backend_obj.search(_make_embeddings(1)[0], limit=2, query_text=payload)

        bm25_calls = [c for c in mock_cursor.execute.call_args_list if "plainto_tsquery" in c[0][0]]
        assert bm25_calls, "no BM25 statement executed"
        for call in bm25_calls:
            sql, params = call[0]
            assert payload not in sql
            assert sql.count("plainto_tsquery('english', %s)") == 2
            assert params.count(payload) == 2

    def test_hybrid_param_order_matches_placeholders(self, backend):
        """Filter params must sit between the ts_rank and WHERE query_text binds."""
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchall.return_value = []

        backend_obj.search(_make_embeddings(1)[0], limit=2, query_text="cats", namespaces=["ns1"])

        bm25_calls = [c for c in mock_cursor.execute.call_args_list if "plainto_tsquery" in c[0][0]]
        sql, params = bm25_calls[0][0]
        # Placeholder order in the statement text: ts_rank, namespace filter, WHERE, LIMIT
        assert sql.index("ts_rank") < sql.index("namespace = ANY(%s)")
        assert params == ["cats", ["ns1"], "cats", 8]  # candidate_limit = limit * 4


class TestDeleteAndClear:
    def test_delete_by_document_returns_count(self, backend):
        backend_obj, mock_conn, mock_cursor = backend
        # First fetchall: the chunk_ids being deleted. Later fetchall calls are
        # cascade lookups; fetchone backs the orphan-entity COUNT(*).
        mock_cursor.fetchall.return_value = [(f"c{i}",) for i in range(5)]
        mock_cursor.fetchone.return_value = (0,)

        count = backend_obj.delete_by_document("doc_a")

        assert count == 5
        mock_conn.commit.assert_called_once()

    def test_delete_by_document_removes_registry_row(self, backend):
        """Regression: defect #6 — a surviving documents row makes the next
        sync_document report 'skipped' for a document that is no longer indexed."""
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchall.return_value = [("c0",)]
        mock_cursor.fetchone.return_value = (0,)

        backend_obj.delete_by_document("doc_a")

        deletes = [c[0][0] for c in mock_cursor.execute.call_args_list if "DELETE" in c[0][0]]
        assert any("chonk_embeddings" in sql for sql in deletes)
        assert any("chonk_documents" in sql for sql in deletes)

    def test_delete_by_document_cascades_to_chunk_keyed_tables(self, backend):
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchall.return_value = [("c0",)]
        # Non-zero backs both the information_schema table_exists probes and the
        # orphan-entity COUNT(*), so every cascade branch is exercised.
        mock_cursor.fetchone.return_value = (1,)

        backend_obj.delete_by_document("doc_a")

        deletes = [c[0][0] for c in mock_cursor.execute.call_args_list if "DELETE" in c[0][0]]
        for table in ("chunk_entities", "chunk_clusters", "svo_triples"):
            assert any(f"DELETE FROM {table}" in sql for sql in deletes), table

    def test_delete_by_document_no_rows(self, backend):
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = (0,)

        count = backend_obj.delete_by_document("missing")
        assert count == 0

    def test_clear_deletes_all(self, backend):
        backend_obj, mock_conn, mock_cursor = backend

        backend_obj.clear()

        statements = [c[0][0] for c in mock_cursor.execute.call_args_list]
        assert any("DELETE FROM chonk_embeddings" in s for s in statements)
        mock_conn.commit.assert_called_once()

    def test_clear_cascades_to_derived_tables(self, backend):
        """A surviving documents row makes the next sync a no-op against an empty index."""
        from chonk.storage._cascade import DERIVED_TABLES

        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchone.return_value = (1,)  # _table_exists -> True

        backend_obj.clear()

        statements = [c[0][0] for c in mock_cursor.execute.call_args_list]
        for table in DERIVED_TABLES:
            assert any(f"DELETE FROM {table}" in s for s in statements), table

    def test_clear_commits_once_after_the_cascade(self, backend):
        """Commit last, so the wipe is atomic rather than half-applied."""
        backend_obj, mock_conn, mock_cursor = backend
        mock_cursor.fetchone.return_value = (1,)

        backend_obj.clear()

        mock_conn.commit.assert_called_once()
        statements = [c[0][0] for c in mock_cursor.execute.call_args_list]
        assert "DELETE FROM context_graph_edges" in statements[-1]


class TestCount:
    def test_count_returns_integer(self, backend):
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchone.return_value = (42,)

        assert backend_obj.count() == 42

    def test_count_empty_table(self, backend):
        backend_obj, _conn, mock_cursor = backend
        mock_cursor.fetchone.return_value = None

        assert backend_obj.count() == 0


class TestChunkIdStability:
    def test_same_inputs_produce_same_id(self, backend):
        backend_obj, *_ = backend
        from chonk.storage._pg import PgVectorBackend

        id1 = PgVectorBackend._generate_chunk_id("doc", 0, "hello world")
        id2 = PgVectorBackend._generate_chunk_id("doc", 0, "hello world")
        assert id1 == id2

    def test_different_docs_produce_different_ids(self, backend):
        from chonk.storage._pg import PgVectorBackend

        id1 = PgVectorBackend._generate_chunk_id("doc_a", 0, "hello")
        id2 = PgVectorBackend._generate_chunk_id("doc_b", 0, "hello")
        assert id1 != id2
