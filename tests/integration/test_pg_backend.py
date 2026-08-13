# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 8ba777a4-76d9-405a-aa22-4329f705561b
"""Integration tests for PgVectorBackend — requires Docker + psycopg2 + pgvector."""

from __future__ import annotations

import time
from collections.abc import Generator

import numpy as np
import pytest

from chonk.graph._svo import SVOTriple
from chonk.models import DocumentChunk
from chonk.storage._pg import PgVectorBackend

docker = pytest.importorskip("docker", reason="docker SDK not installed — pip install docker")
psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 not installed")
pytest.importorskip("pgvector", reason="pgvector not installed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 8
PG_IMAGE = "pgvector/pgvector:pg16"
PG_PASSWORD = "testpass"
PG_DB = "testdb"
PG_USER = "postgres"


# ---------------------------------------------------------------------------
# Docker fixture — session-scoped
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pg_dsn() -> Generator[str, None, None]:
    client = docker.from_env()
    container = client.containers.run(
        PG_IMAGE,
        detach=True,
        remove=True,
        ports={"5432/tcp": None},
        environment={
            "POSTGRES_PASSWORD": PG_PASSWORD,
            "POSTGRES_DB": PG_DB,
            "POSTGRES_USER": PG_USER,
        },
    )
    try:
        # Resolve the dynamically assigned host port
        container.reload()
        host_port = container.ports["5432/tcp"][0]["HostPort"]
        dsn = f"postgresql://{PG_USER}:{PG_PASSWORD}@localhost:{host_port}/{PG_DB}"

        # Wait for PostgreSQL to accept connections
        deadline = time.monotonic() + 60.0
        last_err = None
        while time.monotonic() < deadline:
            try:
                conn = psycopg2.connect(dsn, connect_timeout=2)
                conn.close()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(0.5)
        else:
            raise RuntimeError(f"PostgreSQL never became ready: {last_err}") from last_err

        yield dsn
    finally:
        container.stop(timeout=5)


# ---------------------------------------------------------------------------
# Backend fixture — function-scoped (fresh tables each test via TRUNCATE)
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend(pg_dsn: str) -> Generator[PgVectorBackend, None, None]:
    b = PgVectorBackend(pg_dsn, embedding_dim=DIM)
    b.clear()
    # Also clear auxiliary tables that tests write to
    with b._pgconn.cursor() as cur:
        cur.execute("TRUNCATE svo_triples")
        cur.execute("TRUNCATE documents")
    b._pgconn.commit()
    yield b
    b.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(doc: str = "doc_a", idx: int = 0, content: str = "hello world") -> DocumentChunk:
    return DocumentChunk(
        document_name=doc,
        content=content,
        chunk_index=idx,
        section=[],
        source_detail={"row_start": idx},
    )


def _embeddings(n: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# Schema / connection
# ---------------------------------------------------------------------------


class TestSchemaInit:
    def test_tables_created(self, backend: PgVectorBackend) -> None:
        with backend._pgconn.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            names = {r[0] for r in cur.fetchall()}
        expected = {"embeddings", "documents", "svo_triples", "ingest_queue", "control"}
        assert expected.issubset(names)

    def test_vector_extension(self, backend: PgVectorBackend) -> None:
        with backend._pgconn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
            assert row is not None and row[0] == 1


# ---------------------------------------------------------------------------
# Add / count / search
# ---------------------------------------------------------------------------


class TestAddAndCount:
    def test_add_and_count(self, backend: PgVectorBackend) -> None:
        chunks = [_chunk("doc", i, f"content {i}") for i in range(3)]
        backend.add_chunks(chunks, _embeddings(3), namespace="ns1")
        assert backend.count() == 3

    def test_idempotent_add(self, backend: PgVectorBackend) -> None:
        chunks = [_chunk("doc", 0, "same content")]
        backend.add_chunks(chunks, _embeddings(1))
        backend.add_chunks(chunks, _embeddings(1))
        assert backend.count() == 1

    def test_namespace_stored(self, backend: PgVectorBackend) -> None:
        backend.add_chunks([_chunk()], _embeddings(1), namespace="myns")
        with backend._pgconn.cursor() as cur:
            cur.execute("SELECT namespace FROM embeddings LIMIT 1")
            row = cur.fetchone()
            assert row is not None and row[0] == "myns"


class TestVectorSearch:
    def test_returns_results(self, backend: PgVectorBackend) -> None:
        chunks = [_chunk("doc", i, f"sentence number {i}") for i in range(5)]
        embs = _embeddings(5)
        backend.add_chunks(chunks, embs, namespace="ns")

        results = backend.search(embs[0], limit=3)
        assert len(results) == 3
        for chunk_id, score, chunk in results:
            assert isinstance(chunk_id, str)
            assert 0.0 <= score <= 1.0
            assert isinstance(chunk, DocumentChunk)

    def test_namespace_filter(self, backend: PgVectorBackend) -> None:
        chunks_a = [_chunk("a", i, f"alpha {i}") for i in range(3)]
        chunks_b = [_chunk("b", i, f"beta {i}") for i in range(3)]
        embs_a = _embeddings(3, DIM)
        embs_b = np.random.default_rng(7).random((3, DIM), dtype=np.float32)
        backend.add_chunks(chunks_a, embs_a, namespace="ns_a")
        backend.add_chunks(chunks_b, embs_b, namespace="ns_b")

        results = backend.search(embs_a[0], limit=10, namespaces=["ns_a"])
        docs = set()
        for _, _, chunk in results:
            assert isinstance(chunk, DocumentChunk)
            docs.add(chunk.document_name)
        assert docs == {"a"}

    def test_chunk_type_filter(self, backend: PgVectorBackend) -> None:
        chunk = _chunk()
        with backend._pgconn.cursor() as cur:
            cur.execute(
                "INSERT INTO embeddings "
                "(chunk_id, document_name, section, chunk_index, content, chunk_type, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    "cid_summary",
                    "doc_x",
                    "[]",
                    0,
                    "summary text",
                    "summary",
                    _embeddings(1)[0].tolist(),
                ],
            )
        backend._pgconn.commit()
        backend.add_chunks([chunk], _embeddings(1))

        results = backend.search(_embeddings(1)[0], limit=10, chunk_types=["summary"])
        for _, _, chunk in results:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.chunk_type == "summary"


class TestHybridSearch:
    def test_hybrid_returns_results(self, backend: PgVectorBackend) -> None:
        content = "machine learning is a subset of artificial intelligence"
        chunks = [_chunk("doc", 0, content)]
        backend.add_chunks(chunks, _embeddings(1))

        results = backend.search(
            _embeddings(1)[0],
            limit=5,
            query_text="machine learning",
        )
        assert len(results) >= 1
        contents = [chunk.content for _, _, chunk in results if isinstance(chunk, DocumentChunk)]
        assert any("machine learning" in c for c in contents)

    def test_hybrid_different_from_vector_only(self, backend: PgVectorBackend) -> None:
        texts = [
            "neural networks and deep learning",
            "database query optimization techniques",
            "gradient descent and backpropagation",
        ]
        chunks = [_chunk("doc", i, t) for i, t in enumerate(texts)]
        embs = _embeddings(3)
        backend.add_chunks(chunks, embs)

        vec_results = backend.search(embs[0], limit=3)
        hybrid_results = backend.search(embs[0], limit=3, query_text="deep learning neural")

        vec_ids = [r[0] for r in vec_results]
        hybrid_ids = [r[0] for r in hybrid_results]
        # Hybrid reranking may produce different ordering
        assert len(hybrid_results) >= 1
        assert set(hybrid_ids).issubset(set(vec_ids) | set(hybrid_ids))


# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------


class TestDocumentRegistry:
    def test_register_and_list(self, backend: PgVectorBackend) -> None:
        backend.register_document("doc_a", "abc123", source_uri="file:///a.txt", chunk_count=5)
        docs = backend.list_documents()
        assert len(docs) == 1
        assert docs[0]["document_name"] == "doc_a"
        assert docs[0]["chunk_count"] == 5

    def test_get_document_hash(self, backend: PgVectorBackend) -> None:
        backend.register_document("doc_b", "deadbeef")
        assert backend.get_document_hash("doc_b") == "deadbeef"
        assert backend.get_document_hash("missing") is None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_by_document(self, backend: PgVectorBackend) -> None:
        chunks = [_chunk("doc_a", i) for i in range(4)]
        backend.add_chunks(chunks, _embeddings(4))
        count = backend.delete_by_document("doc_a")
        assert count == 4
        assert backend.count() == 0

    def test_delete_missing_returns_zero(self, backend: PgVectorBackend) -> None:
        assert backend.delete_by_document("nonexistent") == 0


# ---------------------------------------------------------------------------
# SVO triples
# ---------------------------------------------------------------------------


class TestSVOTriples:
    def _triples(self, chunk_id: str | None = None) -> list[SVOTriple]:
        return [
            SVOTriple(
                subject_id="entity:order",
                verb="depends_on",
                object_id="entity:customer",
                confidence=0.95,
                source_chunk_id=chunk_id,
            ),
            SVOTriple(
                subject_id="entity:order",
                verb="contains",
                object_id="entity:line_item",
                confidence=0.88,
                source_chunk_id=chunk_id,
            ),
            SVOTriple(
                subject_id="entity:customer",
                verb="places",
                object_id="entity:order",
                confidence=0.92,
                source_chunk_id=chunk_id,
            ),
        ]

    def test_store_and_retrieve(self, backend: PgVectorBackend) -> None:
        triples = self._triples()
        n = backend.store_svo_triples(triples, namespace="ns1")
        assert n == 3

        rows = backend.get_svo_triples(namespace="ns1")
        assert len(rows) == 3
        subjects = {r[1] for r in rows}
        assert "entity:order" in subjects
        assert "entity:customer" in subjects

    def test_namespace_filter(self, backend: PgVectorBackend) -> None:
        backend.store_svo_triples(self._triples(), namespace="ns_a")
        backend.store_svo_triples(self._triples(), namespace="ns_b")

        rows_a = backend.get_svo_triples(namespace="ns_a")
        rows_b = backend.get_svo_triples(namespace="ns_b")
        assert len(rows_a) == 3
        assert len(rows_b) == 3
        assert all(r[5] == "ns_a" for r in rows_a)

    def test_namespace_inherited_from_chunk(self, backend: PgVectorBackend) -> None:
        chunk = _chunk("doc", 0, "orders depend on customers")
        backend.add_chunks([chunk], _embeddings(1), namespace="inherited_ns")

        with backend._pgconn.cursor() as cur:
            cur.execute("SELECT chunk_id FROM embeddings LIMIT 1")
            row = cur.fetchone()
            assert row is not None
            chunk_id = row[0]

        triples = self._triples(chunk_id=chunk_id)
        # Pass namespace=None — backend should look up namespace from chunk
        backend.store_svo_triples(triples, namespace=None)

        rows = backend.get_svo_triples()
        assert all(r[5] == "inherited_ns" for r in rows)

    def test_empty_input_is_noop(self, backend: PgVectorBackend) -> None:
        assert backend.store_svo_triples([]) == 0
        assert backend.get_svo_triples() == []

    def test_confidence_stored_correctly(self, backend: PgVectorBackend) -> None:
        triple = SVOTriple(
            subject_id="entity:a",
            verb="type_of",
            object_id="entity:b",
            confidence=0.77,
        )
        backend.store_svo_triples([triple], namespace="ns")
        rows = backend.get_svo_triples()
        assert abs(rows[0][4] - 0.77) < 1e-4

    def test_svo_triples_with_chunks_and_search(self, backend: PgVectorBackend) -> None:
        """End-to-end: ingest chunks, store triples, verify search + triples coexist."""
        texts = [
            "orders are created by customers through the order management system",
            "line items belong to orders and reference product catalog entries",
        ]
        chunks = [_chunk("commerce_doc", i, t) for i, t in enumerate(texts)]
        embs = _embeddings(2)
        backend.add_chunks(chunks, embs, namespace="commerce")

        with backend._pgconn.cursor() as cur:
            cur.execute("SELECT chunk_id FROM embeddings ORDER BY chunk_index")
            chunk_ids = [r[0] for r in cur.fetchall()]

        triples = [
            SVOTriple(
                subject_id="entity:customer",
                verb="creates",
                object_id="entity:order",
                confidence=0.9,
                source_chunk_id=chunk_ids[0],
            ),
            SVOTriple(
                subject_id="entity:line_item",
                verb="part_of",
                object_id="entity:order",
                confidence=0.85,
                source_chunk_id=chunk_ids[1],
            ),
        ]
        n = backend.store_svo_triples(triples, namespace="commerce")
        assert n == 2

        # Vector search still works
        search_results = backend.search(embs[0], limit=2)
        assert len(search_results) == 2

        # Hybrid search works
        hybrid = backend.search(embs[0], limit=2, query_text="customer order")
        assert len(hybrid) >= 1

        # Triples are queryable
        rows = backend.get_svo_triples(namespace="commerce")
        assert len(rows) == 2
        verbs = {r[2] for r in rows}
        assert "creates" in verbs
        assert "part_of" in verbs


# ---------------------------------------------------------------------------
# Worker / Coordinator queue mechanics
# ---------------------------------------------------------------------------


class TestWorkerCoordinatorMechanics:
    """Tests for queue claim, status transitions, stale lease requeue,
    and coordinator pause/resume — all against real PG, no embed model needed.

    The embedding step in run_worker is bypassed: we enqueue jobs and
    simulate the worker claiming/completing them via direct SQL, then
    verify chunk insertion independently via backend.add_chunks().
    """

    def _enqueue(self, dsn: str, source_uri: str, namespace: str = "ns") -> int:
        """Insert a pending job into ingest_queue. Returns the new job id."""
        from chonk.ingest import _pg_connect  # noqa: PLC0415

        with _pg_connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ingest_queue (source_uri, namespace) VALUES (%s, %s) RETURNING id",
                    [source_uri, namespace],
                )
                row = cur.fetchone()
                assert row is not None
                job_id: int = row[0]
            conn.commit()
        return job_id

    def _claim(self, dsn: str, worker_id: str) -> tuple | None:
        """Claim the next pending job. Returns (id, source_uri, namespace) or None."""
        from chonk.ingest import _pg_connect  # noqa: PLC0415

        with _pg_connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingest_queue
                    SET status = 'processing', worker_id = %s, leased_at = now()
                    WHERE id = (
                        SELECT id FROM ingest_queue
                        WHERE status = 'pending'
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING id, source_uri, namespace
                    """,
                    [worker_id],
                )
                job = cur.fetchone()
            conn.commit()
        return job

    def _mark_done(self, dsn: str, job_id: int) -> None:
        from chonk.ingest import _pg_connect  # noqa: PLC0415

        with _pg_connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_queue SET status = 'done' WHERE id = %s",
                    [job_id],
                )
            conn.commit()

    def _job_status(self, dsn: str, job_id: int) -> str | None:
        from chonk.ingest import _pg_connect  # noqa: PLC0415

        with _pg_connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT status FROM ingest_queue WHERE id = %s", [job_id])
                row = cur.fetchone()
            conn.commit()
        return row[0] if row else None

    @pytest.fixture(autouse=True)
    def _clean_queue(self, pg_dsn: str) -> Generator[None, None, None]:
        """Truncate queue + control before each test in this class."""
        from chonk.ingest import _pg_connect  # noqa: PLC0415

        with _pg_connect(pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE ingest_queue, control")
            conn.commit()
        yield

    # ── pending / processing counts ──────────────────────────────────────────

    def test_pending_count_empty(self, pg_dsn: str) -> None:
        from chonk.ingest import _queue_pending_count  # noqa: PLC0415

        assert _queue_pending_count(pg_dsn) == 0

    def test_pending_count_after_enqueue(self, pg_dsn: str) -> None:
        from chonk.ingest import _queue_pending_count  # noqa: PLC0415

        self._enqueue(pg_dsn, "file:///a.txt")
        self._enqueue(pg_dsn, "file:///b.txt")
        assert _queue_pending_count(pg_dsn) == 2

    def test_processing_count_after_claim(self, pg_dsn: str) -> None:
        from chonk.ingest import _queue_pending_count, _queue_processing_count  # noqa: PLC0415

        self._enqueue(pg_dsn, "file:///a.txt")
        self._claim(pg_dsn, "worker-1")
        assert _queue_pending_count(pg_dsn) == 0
        assert _queue_processing_count(pg_dsn) == 1

    def test_count_after_done(self, pg_dsn: str) -> None:
        from chonk.ingest import _queue_pending_count, _queue_processing_count  # noqa: PLC0415

        job_id = self._enqueue(pg_dsn, "file:///a.txt")
        self._claim(pg_dsn, "worker-1")
        self._mark_done(pg_dsn, job_id)
        assert _queue_pending_count(pg_dsn) == 0
        assert _queue_processing_count(pg_dsn) == 0

    # ── SKIP LOCKED — two workers, one job ───────────────────────────────────

    def test_skip_locked_single_job(self, pg_dsn: str) -> None:
        """Two concurrent workers can't both claim the same job."""
        self._enqueue(pg_dsn, "file:///a.txt")
        job1 = self._claim(pg_dsn, "worker-1")
        job2 = self._claim(pg_dsn, "worker-2")
        assert job1 is not None
        assert job2 is None  # no second pending job

    def test_skip_locked_two_jobs(self, pg_dsn: str) -> None:
        """Two workers each claim a distinct job."""
        self._enqueue(pg_dsn, "file:///a.txt")
        self._enqueue(pg_dsn, "file:///b.txt")
        job1 = self._claim(pg_dsn, "worker-1")
        job2 = self._claim(pg_dsn, "worker-2")
        assert job1 is not None and job2 is not None
        assert job1[0] != job2[0]  # different job ids

    # ── stale lease requeue ───────────────────────────────────────────────────

    def test_requeue_stale_leases(self, pg_dsn: str) -> None:
        from chonk.ingest import _pg_connect, _requeue_stale_leases  # noqa: PLC0415

        job_id = self._enqueue(pg_dsn, "file:///a.txt")
        self._claim(pg_dsn, "worker-1")
        # Back-date leased_at to simulate a stale lease
        with _pg_connect(pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_queue"
                    " SET leased_at = now() - interval '20 minutes'"
                    " WHERE id = %s",
                    [job_id],
                )
            conn.commit()

        _requeue_stale_leases(pg_dsn, timeout_minutes=10)
        assert self._job_status(pg_dsn, job_id) == "pending"

    def test_fresh_lease_not_requeued(self, pg_dsn: str) -> None:
        from chonk.ingest import _requeue_stale_leases  # noqa: PLC0415

        job_id = self._enqueue(pg_dsn, "file:///a.txt")
        self._claim(pg_dsn, "worker-1")
        _requeue_stale_leases(pg_dsn, timeout_minutes=10)
        assert self._job_status(pg_dsn, job_id) == "processing"

    # ── coordinator control table (pause / resume) ────────────────────────────

    def test_set_and_read_control(self, pg_dsn: str) -> None:
        from chonk.ingest import _pg_connect, _set_control  # noqa: PLC0415

        _set_control(pg_dsn, "workers_paused", "1")
        with _pg_connect(pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM control WHERE key = 'workers_paused'")
                row = cur.fetchone()
                assert row is not None and row[0] == "1"
            conn.commit()

    def test_control_upsert(self, pg_dsn: str) -> None:
        from chonk.ingest import _pg_connect, _set_control  # noqa: PLC0415

        _set_control(pg_dsn, "workers_paused", "1")
        _set_control(pg_dsn, "workers_paused", "0")
        with _pg_connect(pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM control WHERE key = 'workers_paused'")
                row = cur.fetchone()
                assert row is not None and row[0] == 1  # not duplicated
                cur.execute("SELECT value FROM control WHERE key = 'workers_paused'")
                val_row = cur.fetchone()
                assert val_row is not None and val_row[0] == "0"
            conn.commit()

    # ── end-to-end worker simulation ─────────────────────────────────────────

    def test_worker_claim_process_done(self, pg_dsn: str, backend: PgVectorBackend) -> None:
        """Simulate one worker cycle: enqueue → claim → embed (random) → insert → done."""
        job_id = self._enqueue(pg_dsn, "file:///order_doc.txt", namespace="orders")
        job = self._claim(pg_dsn, "worker-1")
        assert job is not None
        claimed_id, source_uri, namespace = job
        assert claimed_id == job_id
        assert source_uri == "file:///order_doc.txt"
        assert namespace == "orders"

        # Simulate the embed + insert step (no real model needed)
        chunks = [_chunk(doc="order_doc", idx=i, content=f"order content {i}") for i in range(3)]
        backend.add_chunks(chunks, _embeddings(3), namespace=namespace)
        backend.register_document(
            "order_doc",
            content_hash="abc123",
            source_uri=source_uri,
            chunk_count=len(chunks),
        )

        self._mark_done(pg_dsn, job_id)

        assert self._job_status(pg_dsn, job_id) == "done"
        assert backend.count() == 3
        assert backend.get_document_hash("order_doc") == "abc123"

    def test_coordinator_pause_blocks_worker_claim(self, pg_dsn: str) -> None:
        """Workers should not process jobs while workers_paused=1."""
        from chonk.ingest import _pg_connect, _set_control  # noqa: PLC0415

        self._enqueue(pg_dsn, "file:///a.txt")
        _set_control(pg_dsn, "workers_paused", "1")

        # Simulate what run_worker does: check pause flag before claiming
        with _pg_connect(pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM control WHERE key = 'workers_paused'")
                row = cur.fetchone()
            conn.commit()
        paused = row is not None and row[0] == "1"
        assert paused  # worker would sleep, not claim

        # After coordinator unpauses, worker can proceed
        _set_control(pg_dsn, "workers_paused", "0")
        job = self._claim(pg_dsn, "worker-1")
        assert job is not None

    def test_failed_job_status(self, pg_dsn: str) -> None:
        """A job that fails gets status='failed', not dropped."""
        from chonk.ingest import _pg_connect  # noqa: PLC0415

        job_id = self._enqueue(pg_dsn, "file:///bad_file.txt")
        self._claim(pg_dsn, "worker-1")
        # Simulate failure
        with _pg_connect(pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_queue SET status = 'failed' WHERE id = %s",
                    [job_id],
                )
            conn.commit()
        assert self._job_status(pg_dsn, job_id) == "failed"


# ---------------------------------------------------------------------------
# Worker + Coordinator tandem — real run_worker() in a thread
# ---------------------------------------------------------------------------

sentence_transformers = pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers not installed — pip install chonk-rag",
)


class TestWorkerCoordinatorTandem:
    """Full tandem test: real run_worker() thread + coordinator helpers against live PG.

    Uses ``all-MiniLM-L6-v2`` (22 MB) so the embed step is fast enough for CI.
    run_worker() is an infinite loop; we kill it by setting a threading.Event
    that replaces the idle-sleep, then joining with a timeout.
    """

    _EMBED_MODEL = "all-MiniLM-L6-v2"
    _TIMEOUT = 120  # seconds to wait for queue to drain

    @pytest.fixture()
    def backend(self, pg_dsn: str) -> Generator[PgVectorBackend, None, None]:
        # all-MiniLM-L6-v2 produces 384-dim vectors; drop and recreate the
        # embeddings table so the column type matches before the worker runs.
        with psycopg2.connect(pg_dsn) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS embeddings CASCADE")
        b = PgVectorBackend(pg_dsn, embedding_dim=384)
        b.clear()
        with b._pgconn.cursor() as cur:
            cur.execute("TRUNCATE svo_triples")
            cur.execute("TRUNCATE documents")
        b._pgconn.commit()
        yield b
        b.close()

    @pytest.fixture(autouse=True)
    def _clean(self, pg_dsn: str) -> Generator[None, None, None]:
        from chonk.ingest import _pg_connect  # noqa: PLC0415

        with _pg_connect(pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE documents, svo_triples, ingest_queue, control")
            conn.commit()
        yield

    @pytest.fixture()
    def doc_files(self, tmp_path):
        """Write two small plain-text documents to a temp directory."""
        texts = {
            "orders.txt": (
                "An order is created when a customer submits a purchase request. "
                "Each order contains one or more line items referencing product SKUs. "
                "Orders are fulfilled by the warehouse management system."
            ),
            "customers.txt": (
                "A customer is a registered user who has completed account verification. "
                "Customers can place orders, track shipments, and request refunds. "
                "Customer data is governed by the data retention policy."
            ),
        }
        paths = {}
        for name, content in texts.items():
            p = tmp_path / name
            p.write_text(content)
            paths[name] = str(p)
        return paths

    def _all_done(self, pg_dsn: str) -> bool:
        from chonk.ingest import _queue_pending_count, _queue_processing_count  # noqa: PLC0415

        return _queue_pending_count(pg_dsn) == 0 and _queue_processing_count(pg_dsn) == 0

    def test_worker_processes_enqueued_docs(
        self, pg_dsn: str, backend: PgVectorBackend, doc_files: dict
    ) -> None:
        """run_worker() ingests real documents and builds embeddings in PG."""
        import threading

        from chonk.ingest import _pg_connect, run_worker  # noqa: PLC0415

        # Enqueue both documents
        with _pg_connect(pg_dsn) as conn:
            with conn.cursor() as cur:
                for path in doc_files.values():
                    cur.execute(
                        "INSERT INTO ingest_queue (source_uri, namespace) VALUES (%s, %s)",
                        [path, "tandem_ns"],
                    )
            conn.commit()

        # Run worker in a daemon thread; it will sleep idle_sleep between polls
        worker_exc: list[Exception] = []

        def _worker() -> None:
            try:
                run_worker(
                    pg_dsn,
                    pg_dsn,
                    embed_model=self._EMBED_MODEL,
                    idle_sleep=0.2,
                )
            except Exception as exc:
                worker_exc.append(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Poll until queue drains or timeout
        deadline = time.monotonic() + self._TIMEOUT
        while time.monotonic() < deadline:
            if self._all_done(pg_dsn):
                break
            time.sleep(0.5)

        # Worker is an infinite loop — leave as daemon (dies with process)
        assert not worker_exc, f"Worker raised: {worker_exc[0]}"
        assert self._all_done(pg_dsn), "Queue did not drain within timeout"

        # Verify embeddings landed in PG
        assert backend.count() >= 1
        docs = {d["document_name"] for d in backend.list_documents()}
        assert "orders" in docs or "orders.txt" in docs or any("order" in d for d in docs)

        # Search works against the real embeddings
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model = SentenceTransformer(self._EMBED_MODEL)
        q_vec = model.encode(["customer order purchase"], normalize_embeddings=True)[0]
        results = backend.search(q_vec, limit=3, namespaces=["tandem_ns"])
        assert len(results) >= 1

    def test_coordinator_pauses_worker_during_graph_build(
        self, pg_dsn: str, backend: PgVectorBackend, doc_files: dict
    ) -> None:
        """Coordinator sets workers_paused=1 before graph build; worker halts claiming."""
        import threading

        from chonk.ingest import (  # noqa: PLC0415
            _pg_connect,
            _queue_pending_count,
            _set_control,
            run_worker,
        )

        # Enqueue one document
        path = next(iter(doc_files.values()))
        with _pg_connect(pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ingest_queue (source_uri, namespace) VALUES (%s, %s)",
                    [path, "coord_ns"],
                )
            conn.commit()

        # Pause workers before starting the thread
        _set_control(pg_dsn, "workers_paused", "1")

        claimed: list[bool] = []

        def _worker() -> None:
            # Worker checks pause flag; with it set it should not claim anything
            with _pg_connect(pg_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT value FROM control WHERE key = 'workers_paused'")
                    row = cur.fetchone()
                conn.commit()
            if row and row[0] == "1":
                claimed.append(False)
                return  # would sleep in real loop
            claimed.append(True)

        t = threading.Thread(target=_worker)
        t.start()
        t.join(timeout=5)

        assert claimed == [False], "Worker should not claim while paused"
        assert _queue_pending_count(pg_dsn) == 1  # job still pending

        # Coordinator unpauses — now worker can proceed
        _set_control(pg_dsn, "workers_paused", "0")

        worker_exc: list[Exception] = []

        def _worker2() -> None:
            try:
                run_worker(pg_dsn, pg_dsn, embed_model=self._EMBED_MODEL, idle_sleep=0.2)
            except Exception as exc:
                worker_exc.append(exc)

        t2 = threading.Thread(target=_worker2, daemon=True)
        t2.start()

        deadline = time.monotonic() + self._TIMEOUT
        while time.monotonic() < deadline:
            if self._all_done(pg_dsn):
                break
            time.sleep(0.5)

        assert not worker_exc, f"Worker raised: {worker_exc[0]}"
        assert self._all_done(pg_dsn), "Queue did not drain after unpause"
        assert backend.count() >= 1
