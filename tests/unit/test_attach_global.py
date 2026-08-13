# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: a6f73e38-04cb-446c-b7e0-86087e1b029c
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for Store.attach_global() / detach_global() union-view feature."""

from __future__ import annotations

import pytest

try:
    import duckdb  # noqa: F401 — availability probe only
    import numpy as np

    from chonk.storage import Store

    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not STORAGE_AVAILABLE,
    reason="chonk-rag[storage] not installed — pip install chonk-rag[storage]",
)

from chonk.models import DocumentChunk  # noqa: E402 — after the skip guard above

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(doc: str, content: str, idx: int = 0) -> DocumentChunk:
    return DocumentChunk(document_name=doc, content=content, chunk_index=idx)


def _rand_emb(n: int, dim: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, dim)).astype(np.float32)


def _store(tmp_path, name: str, dim: int = 4) -> Store:
    return Store(db_path=str(tmp_path / name), embedding_dim=dim)


# ---------------------------------------------------------------------------
# Test 1: search() spans global store after attach
# ---------------------------------------------------------------------------


class TestAttachGlobalSearch:
    def test_global_chunks_visible_after_attach(self, tmp_path):
        # Global store — add one chunk with a specific embedding direction
        with _store(tmp_path, "global.duckdb") as g:
            emb_g = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            g.add_document([_chunk("global_doc", "hello from global")], emb_g)

        # User store — empty
        user = _store(tmp_path, "user.duckdb")
        assert user.count() == 0

        user.attach_global(str(tmp_path / "global.duckdb"))
        results = user.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), limit=5)
        user.close()

        contents = [r[2].content for r in results]
        assert any("hello from global" in c for c in contents), (
            f"global chunk not found; contents={contents}"
        )

    def test_user_chunks_visible_after_attach(self, tmp_path):
        # Global store with one chunk
        with _store(tmp_path, "global.duckdb") as g:
            g.add_document(
                [_chunk("global_doc", "global content")],
                np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
            )

        # User store with a different chunk
        user = _store(tmp_path, "user.duckdb")
        emb_u = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        user.add_document([_chunk("user_doc", "user content")], emb_u)

        user.attach_global(str(tmp_path / "global.duckdb"))
        results = user.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), limit=10)
        user.close()

        contents = [r[2].content for r in results]
        assert any("user content" in c for c in contents)
        assert any("global content" in c for c in contents)


# ---------------------------------------------------------------------------
# Test 2: User writes don't appear in global store
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_user_write_not_in_global(self, tmp_path):
        with _store(tmp_path, "global.duckdb") as g:
            count_before = g.count()

        user = _store(tmp_path, "user.duckdb")
        user.attach_global(str(tmp_path / "global.duckdb"))
        user.add_document(
            [_chunk("user_doc", "user-only chunk")],
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        user.close()

        # Open global read-only and verify no user chunks present
        with _store(tmp_path, "global.duckdb") as g:
            assert g.count() == count_before


# ---------------------------------------------------------------------------
# Test 3: Write to user store doesn't error when global is attached read-only
# ---------------------------------------------------------------------------


class TestWriteWithGlobalAttached:
    def test_add_document_succeeds_when_global_attached(self, tmp_path):
        with _store(tmp_path, "global.duckdb"):
            pass  # create empty global

        user = _store(tmp_path, "user.duckdb")
        user.attach_global(str(tmp_path / "global.duckdb"))

        chunks = [_chunk("doc", f"chunk {i}", i) for i in range(3)]
        embs = _rand_emb(3, dim=4, seed=42)
        user.add_document(chunks, embs)
        assert user.count() == 3
        user.close()


# ---------------------------------------------------------------------------
# Test 4: detach_global() restores single-store behaviour
# ---------------------------------------------------------------------------


class TestDetachGlobal:
    def test_detach_restores_single_store(self, tmp_path):
        with _store(tmp_path, "global.duckdb") as g:
            g.add_document(
                [_chunk("gdoc", "global chunk")],
                np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            )

        user = _store(tmp_path, "user.duckdb")
        user.attach_global(str(tmp_path / "global.duckdb"))

        # Verify global chunk visible
        results_before = user.search(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), limit=10)
        assert any("global chunk" in r[2].content for r in results_before)

        user.detach_global()

        # After detach, global chunk should no longer be visible
        results_after = user.search(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), limit=10)
        assert not any("global chunk" in r[2].content for r in results_after)
        user.close()

    def test_global_attached_flag_cleared_after_detach(self, tmp_path):
        with _store(tmp_path, "global.duckdb"):
            pass

        user = _store(tmp_path, "user.duckdb")
        user.attach_global(str(tmp_path / "global.duckdb"))
        assert user.vector._global_attached is True

        user.detach_global()
        assert user.vector._global_attached is False
        user.close()


# ---------------------------------------------------------------------------
# Test 5: resolve_domain_ids spans both stores after attach
# ---------------------------------------------------------------------------


class TestResolveDomainIdsGlobal:
    def test_resolve_spans_global_domains(self, tmp_path):
        # Global store: register a namespace + domain
        with _store(tmp_path, "global.duckdb") as g:
            g.register_namespace("global", description="global ns")
            g.register_domain("dom-global-1", "global", "shared_domain")

        # User store: register its own namespace + domain
        user = _store(tmp_path, "user.duckdb")
        user.register_namespace("user-ns")
        user.register_domain("dom-user-1", "user-ns", "user_domain")

        user.attach_global(str(tmp_path / "global.duckdb"))

        # resolve user domain by name
        user_ids = user.resolve_domain_ids([("user-ns", "user_domain")], include_global=False)
        assert "dom-user-1" in user_ids
        assert "dom-global-1" not in user_ids

        # resolve with include_global=True should fold in global domains
        all_ids = user.resolve_domain_ids([("user-ns", "user_domain")], include_global=True)
        assert "dom-user-1" in all_ids
        assert "dom-global-1" in all_ids

        user.close()
