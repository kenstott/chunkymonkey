# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 9e2f7a13-58cb-4d06-a3e1-70b4d9c25f8e

"""Backend parity contract for the document registry (fixed.md phase 5).

Every backend must delete its ``documents`` registry row along with the chunks.
A surviving row makes the next :func:`sync_document` report "skipped" for a
document that is no longer indexed — silent, permanent data loss.

The DuckDB parametrization always runs. The service-backed backends (PG,
Qdrant, Pinecone, Weaviate) skip unless their client library and connection
details are available; :func:`test_at_least_duckdb_ran` guards against the whole
suite silently degrading to a no-op.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

from chonk.storage import prune_documents, sync_document

_RAN: set[str] = set()

# name -> (env var gating the live service, client module that must import)
_SERVICE_BACKENDS = {
    "pg": ("CHONK_TEST_PG_DSN", "psycopg2"),
    "qdrant": ("CHONK_TEST_QDRANT_URL", "qdrant_client"),
    "pinecone": ("CHONK_TEST_PINECONE_API_KEY", "pinecone"),
    "weaviate": ("CHONK_TEST_WEAVIATE_URL", "weaviate"),
}


@pytest.fixture(params=["duckdb", *_SERVICE_BACKENDS])
def backend(request):
    from chonk.storage import Store

    name = request.param
    if name == "duckdb":
        with Store(":memory:", embedding_dim=4) as store:
            _RAN.add(name)
            yield store.vector
        return

    env_var, module = _SERVICE_BACKENDS[name]
    value = os.environ.get(env_var)
    if not value:
        pytest.skip(f"{name}: set {env_var} to run against a live service")
    pytest.importorskip(module)

    kwargs = {
        "pg": {"dsn": value},
        "qdrant": {"qdrant_url": value},
        "pinecone": {"pinecone_api_key": value},
        "weaviate": {"weaviate_url": value},
    }[name]
    with Store(embedding_dim=4, **kwargs) as store:
        store.vector.clear()
        _RAN.add(name)
        yield store.vector
        store.vector.clear()


def _index(backend, document_name: str, contents: list[str], content_hash: str | None = None):
    from chonk.models import DocumentChunk

    chunks = [
        DocumentChunk(document_name=document_name, content=text, chunk_index=i)
        for i, text in enumerate(contents)
    ]
    backend.add_chunks(chunks, np.zeros((len(chunks), 4), dtype="float32"))
    backend.register_document(document_name, content_hash or f"hash-{document_name}")


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


class TestRegistryContract:
    def test_register_then_get_hash_roundtrip(self, backend):
        backend.register_document("doc", "hash-1", source_uri="s3://b/doc", chunk_count=3)
        assert backend.get_document_hash("doc") == "hash-1"

    def test_register_updates_existing_hash(self, backend):
        backend.register_document("doc", "hash-1")
        backend.register_document("doc", "hash-2")
        assert backend.get_document_hash("doc") == "hash-2"

    def test_unknown_document_has_no_hash(self, backend):
        assert backend.get_document_hash("never-indexed") is None

    def test_delete_clears_registry_row(self, backend):
        """Regression: defect #6 — every backend except DuckDB left the row behind."""
        _index(backend, "doc", ["alpha content"])
        assert backend.get_document_hash("doc") is not None

        backend.delete_by_document("doc")

        assert backend.get_document_hash("doc") is None

    def test_delete_then_sync_reindexes(self, backend):
        """The silent-data-loss path: delete, then sync with the SAME hash.

        A stale registry row makes this return "skipped", so the caller never
        re-embeds and the document is gone from the index for good.
        """
        _index(backend, "doc", ["alpha content"], content_hash="stable-hash")
        backend.delete_by_document("doc")

        result = sync_document(backend, "doc", content_hash="stable-hash")

        assert result.action == "added"

    def test_list_documents_reflects_deletes(self, backend):
        _index(backend, "keep", ["kept content"])
        _index(backend, "drop", ["dropped content"])

        backend.delete_by_document("drop")

        assert [d["document_name"] for d in backend.list_documents()] == ["keep"]

    def test_delete_returns_chunk_count(self, backend):
        _index(backend, "doc", ["one", "two", "three"])
        assert backend.delete_by_document("doc") == 3

    def test_delete_unknown_document_is_a_noop(self, backend):
        assert backend.delete_by_document("never-indexed") == 0


# ---------------------------------------------------------------------------
# Sync / prune lifecycle
# ---------------------------------------------------------------------------


class TestSyncLifecycle:
    def test_sync_add_update_skip_cycle(self, backend):
        first = sync_document(backend, "doc", b"version one")
        assert first.action == "added"
        _index(backend, "doc", ["version one"], content_hash=first.content_hash)

        assert sync_document(backend, "doc", b"version one").action == "skipped"

        second = sync_document(backend, "doc", b"version two")
        assert second.action == "updated"
        assert second.previous_chunk_count == 1
        _index(backend, "doc", ["version two"], content_hash=second.content_hash)

        assert backend.get_document_hash("doc") == second.content_hash
        assert backend.count() == 1

    def test_skip_leaves_chunks_untouched(self, backend):
        result = sync_document(backend, "doc", b"stable")
        _index(backend, "doc", ["chunk a", "chunk b"], content_hash=result.content_hash)

        assert sync_document(backend, "doc", b"stable").action == "skipped"
        assert backend.count() == 2

    def test_prune_documents_across_backends(self, backend):
        _index(backend, "keep", ["kept content"])
        _index(backend, "gone", ["removed content"])

        removed = prune_documents(backend, {"keep"})

        assert [r.document_name for r in removed] == ["gone"]
        assert [d["document_name"] for d in backend.list_documents()] == ["keep"]
        assert backend.get_document_hash("gone") is None

    def test_prune_empty_present_raises(self, backend):
        _index(backend, "doc", ["alpha content"])

        with pytest.raises(ValueError):
            prune_documents(backend, set())


# ---------------------------------------------------------------------------
# Static parity — runs without any service
# ---------------------------------------------------------------------------

_BACKEND_CLASSES = [
    "DuckDBVectorBackend",
    "PgVectorBackend",
    "QdrantVectorBackend",
    "PineconeVectorBackend",
    "WeaviateVectorBackend",
]


def _backend_class(name: str):
    import chonk.storage as storage

    return getattr(storage, name)


class TestStaticParity:
    @pytest.mark.parametrize("class_name", _BACKEND_CLASSES)
    def test_implements_registry_and_cascade_surface(self, class_name):
        cls = _backend_class(class_name)
        for method in (
            "register_document",
            "get_document_hash",
            "list_documents",
            "delete_by_document",
            "chunk_ids_for_document",
            "gc_orphaned_entities",
            "clear",
        ):
            assert callable(getattr(cls, method, None)), f"{class_name} missing {method}"

    @pytest.mark.parametrize("class_name", _BACKEND_CLASSES)
    def test_delete_accepts_gc_entities_keyword(self, class_name):
        sig = inspect.signature(_backend_class(class_name).delete_by_document)
        param = sig.parameters.get("gc_entities")
        assert param is not None, f"{class_name}.delete_by_document lacks gc_entities"
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is True

    @pytest.mark.parametrize("class_name", _BACKEND_CLASSES)
    def test_delete_removes_documents_registry_row(self, class_name):
        """Static guard for defect #6 on backends with no live service here."""
        source = inspect.getsource(_backend_class(class_name).delete_by_document)
        assert "DELETE FROM documents" in source or "DELETE FROM {dt}" in source, class_name


def test_at_least_duckdb_ran():
    """Fail loudly if the parametrized suite degraded to all-skips."""
    assert "duckdb" in _RAN
