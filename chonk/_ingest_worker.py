# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 7c1e2a90-4d3b-4f2e-9a6c-2b8f1d5e0a47
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Horizontal-scale worker / coordinator and PostgreSQL queue helpers.

Split out of :mod:`chonk.ingest` to keep that module under the 1000-line limit.
Public symbols (``run_worker``, ``run_coordinator``) are re-exported from
``chonk.ingest`` for API stability.
"""

from __future__ import annotations

import hashlib
import threading
from typing import TYPE_CHECKING, Any

from .loader import DocumentLoader

if TYPE_CHECKING:
    from .storage._protocol import VectorBackend

# ---------------------------------------------------------------------------
# Horizontal scale: worker / coordinator
# ---------------------------------------------------------------------------


def _pg_connect(dsn: str) -> Any:  # noqa: ANN401
    try:
        import psycopg2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("psycopg2 required: pip install chonk-rag[pgvector]") from exc
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


def _process_queue_job(
    backend: VectorBackend,
    source_uri: str,
    namespace: str,
    embed_model: str,
    batch_size: int,
    run_ner: bool,
    spacy_model: str,
) -> None:
    """Load source_uri, chunk, embed, optionally NER, write to backend."""
    import logging

    from .ingest import _embed_texts

    _log = logging.getLogger(__name__)

    loader = DocumentLoader()
    try:
        chunks = loader.load(source_uri)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {source_uri!r}: {exc}") from exc

    if not chunks:
        _log.warning("No chunks produced from %s", source_uri)
        return

    content_hash = hashlib.sha256("\n".join(c.content for c in chunks).encode()).hexdigest()[:32]

    emb = _embed_texts([c.content for c in chunks], embed_model, batch_size)
    backend.add_chunks(chunks, emb, namespace=namespace)
    backend.register_document(
        chunks[0].document_name,
        content_hash,
        source_uri=source_uri,
        chunk_count=len(chunks),
    )

    if run_ner:
        from .ner import build_ner

        # build_ner expects store.vector; wrap backend in a simple adapter
        class _StoreAdapter:
            def __init__(self, vector: VectorBackend) -> None:
                self.vector = vector

        build_ner(_StoreAdapter(backend), spacy_model=spacy_model)


def run_worker(
    queue_dsn: str,
    backend_dsn: str,
    *,
    embed_model: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 256,
    run_ner: bool = False,
    spacy_model: str = "en_core_web_sm",
    idle_sleep: float = 2.0,
    stop_event: threading.Event | None = None,
) -> None:
    """Pull items from ``ingest_queue`` and process them.

    Runs until interrupted, or until *stop_event* is set. Workers check the
    ``control`` table's ``workers_paused`` flag before each item — coordinator
    sets this during graph builds to drain in-flight workers cleanly.

    Args:
        queue_dsn: PostgreSQL DSN for the queue/control tables.
        backend_dsn: PostgreSQL DSN for the vector store.
        embed_model: SentenceTransformer model name for embedding.
        batch_size: Embedding batch size.
        run_ner: Whether to run NER on each document.
        spacy_model: spaCy model for NER.
        idle_sleep: Seconds to sleep when the queue is empty.
        stop_event: Set it to end the loop. Without one the worker never returns,
            so an in-process caller — a test, or any embedded worker — has no way
            to shut it down and it keeps claiming work after the caller is done.
            Also replaces the idle sleep, so a stop is noticed immediately.
    """
    import logging
    import socket
    import time
    import uuid

    log = logging.getLogger(__name__)
    worker_id = f"{socket.gethostname()}:{uuid.uuid4().hex[:8]}"
    log.info("Worker %s starting", worker_id)

    from .storage import PgVectorBackend

    backend = PgVectorBackend(backend_dsn)

    def _idle() -> bool:
        """Sleep between polls; return True when the worker should stop."""
        if stop_event is None:
            time.sleep(idle_sleep)
            return False
        return stop_event.wait(idle_sleep)

    while not (stop_event is not None and stop_event.is_set()):
        # Check pause flag
        with _pg_connect(queue_dsn) as qconn:
            with qconn.cursor() as cur:
                cur.execute("SELECT value FROM control WHERE key = 'workers_paused'")
                row = cur.fetchone()
            qconn.commit()
        if row and row[0] == "1":
            if _idle():
                break
            continue

        # Claim next pending item (SKIP LOCKED prevents double-assignment)
        with _pg_connect(queue_dsn) as qconn:
            with qconn.cursor() as cur:
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
            qconn.commit()

        if job is None:
            if _idle():
                break
            continue

        job_id, source_uri, namespace = job
        log.info("Worker %s processing job %s: %s", worker_id, job_id, source_uri)

        try:
            _process_queue_job(
                backend,
                source_uri,
                namespace,
                embed_model=embed_model,
                batch_size=batch_size,
                run_ner=run_ner,
                spacy_model=spacy_model,
            )
            with _pg_connect(queue_dsn) as qconn:
                with qconn.cursor() as cur:
                    cur.execute(
                        "UPDATE ingest_queue SET status = 'done' WHERE id = %s",
                        [job_id],
                    )
                qconn.commit()
            log.info("Job %s done", job_id)
        except Exception as exc:
            log.error("Job %s failed: %s", job_id, exc, exc_info=True)
            with _pg_connect(queue_dsn) as qconn:
                with qconn.cursor() as cur:
                    cur.execute(
                        "UPDATE ingest_queue SET status = 'failed' WHERE id = %s",
                        [job_id],
                    )
                qconn.commit()


def run_coordinator(
    queue_dsn: str,
    backend_dsn: str,
    *,
    graph_interval: int = 300,
    embed_model: str = "BAAI/bge-large-en-v1.5",
    spacy_model: str = "en_core_web_sm",
    community_alpha: float = 0.2,
    community_sim_threshold: float = 0.6,
    lease_timeout_minutes: int = 10,
    poll_sleep: float = 5.0,
) -> None:
    """Coordinator: graph build, stale-lease requeue, and pause/resume signal.

    State machine::

        DISPATCHING → (interval elapsed or queue drained)
            → DRAINING   (stop dispatching, wait for in-flight workers)
            → BUILDING   (pause workers, run graph build, unpause)
            → DISPATCHING

    Args:
        queue_dsn: PostgreSQL DSN for the queue/control tables.
        backend_dsn: PostgreSQL DSN for the vector store.
        graph_interval: Seconds between graph builds (default 300).
        embed_model: SentenceTransformer model used for community embeddings.
        spacy_model: spaCy model used by NER (for community build context).
        community_alpha: Alpha for community detection weighting.
        community_sim_threshold: Minimum cosine sim for community edges.
        lease_timeout_minutes: Minutes before a 'processing' job is re-queued.
        poll_sleep: Seconds between coordinator loop ticks.
    """
    import logging
    import time

    log = logging.getLogger(__name__)
    log.info("Coordinator starting (graph_interval=%ds)", graph_interval)

    state = "DISPATCHING"
    last_graph_build = 0.0

    while True:
        now = time.time()

        if state == "DISPATCHING":
            _requeue_stale_leases(queue_dsn, lease_timeout_minutes)

            queue_empty = _queue_pending_count(queue_dsn) == 0
            interval_elapsed = (now - last_graph_build) >= graph_interval
            if queue_empty or interval_elapsed:
                log.info(
                    "Transitioning DISPATCHING → DRAINING (queue_empty=%s, interval_elapsed=%s)",
                    queue_empty,
                    interval_elapsed,
                )
                state = "DRAINING"

        elif state == "DRAINING":
            if _queue_processing_count(queue_dsn) == 0:
                log.info("All workers drained. Transitioning → BUILDING")
                state = "BUILDING"

        elif state == "BUILDING":
            _set_control(queue_dsn, "workers_paused", "1")
            log.info("Workers paused. Starting graph build.")
            try:
                _run_graph_build(
                    backend_dsn,
                    embed_model=embed_model,
                    alpha=community_alpha,
                    sim_threshold=community_sim_threshold,
                )
                log.info("Graph build complete.")
            except Exception as exc:
                log.error("Graph build failed: %s", exc, exc_info=True)
            finally:
                _set_control(queue_dsn, "workers_paused", "0")
                log.info("Workers unpaused.")
            last_graph_build = time.time()
            state = "DISPATCHING"

        time.sleep(poll_sleep)


# ---------------------------------------------------------------------------
# Coordinator helpers
# ---------------------------------------------------------------------------


def _set_control(dsn: str, key: str, value: str) -> None:
    with _pg_connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO control (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                [key, value],
            )
        conn.commit()


def _queue_pending_count(dsn: str) -> int:
    with _pg_connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ingest_queue WHERE status = 'pending'")
            row = cur.fetchone()
        conn.commit()
    return row[0] if row else 0


def _queue_processing_count(dsn: str) -> int:
    with _pg_connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ingest_queue WHERE status = 'processing'")
            row = cur.fetchone()
        conn.commit()
    return row[0] if row else 0


def _requeue_stale_leases(dsn: str, timeout_minutes: int = 10) -> None:
    """Reset 'processing' jobs whose lease has expired back to 'pending'."""
    with _pg_connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingest_queue
                SET status = 'pending', worker_id = NULL, leased_at = NULL
                WHERE status = 'processing'
                  AND leased_at < now() - interval '%s minutes'
                """,
                [timeout_minutes],
            )
            count = cur.rowcount
        conn.commit()
    if count:
        import logging

        logging.getLogger(__name__).warning("Requeued %d stale lease(s)", count)


def _run_graph_build(
    backend_dsn: str,
    *,
    embed_model: str,
    alpha: float,
    sim_threshold: float,
) -> None:
    """Fetch all namespaces from PG and run community detection per namespace.

    Reads chunks from the PG backend via get_all_chunks(), builds a temporary
    in-memory DuckDB store for graph computation, then writes results back.
    """
    import logging
    import tempfile

    import numpy as np

    log = logging.getLogger(__name__)

    from .storage import PgVectorBackend, Store

    pg = PgVectorBackend(backend_dsn)

    # Enumerate namespaces with chunks
    namespaces_row = pg._conn.execute(
        "SELECT DISTINCT namespace FROM embeddings WHERE namespace IS NOT NULL"
    ).fetchall()
    namespaces = [r[0] for r in namespaces_row]

    if not namespaces:
        log.info("No namespaces found; skipping graph build.")
        pg.close()
        return

    for ns in namespaces:
        log.info("Building community graph for namespace %s", ns)
        # Pull chunks for this namespace into a temp DuckDB for graph building
        chunks_rows = pg._conn.execute(
            "SELECT chunk_id, document_name, content, section, chunk_index, "
            "       breadcrumb, chunk_type, source_offset, source_length, "
            "       source_detail, source_id, domain_id, embedding "
            "FROM embeddings WHERE namespace = %s",
            [ns],
        ).fetchall()

        if not chunks_rows:
            continue

        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
            tmp_path = f.name

        with Store(tmp_path, embedding_dim=pg._embedding_dim) as tmp_store:
            from .models import DocumentChunk

            tmp_chunks = []
            tmp_embeddings = []
            for row in chunks_rows:
                (
                    _chunk_id,
                    doc_name,
                    content,
                    section_str,
                    chunk_idx,
                    breadcrumb,
                    chunk_type,
                    src_off,
                    src_len,
                    src_det_str,
                    src_id,
                    dom_id,
                    embedding_vec,
                ) = row
                import json as _json

                sec = _json.loads(section_str) if section_str else []
                src_det = _json.loads(src_det_str) if src_det_str else None
                tmp_chunks.append(
                    DocumentChunk(
                        document_name=doc_name,
                        content=content,
                        section=sec,
                        chunk_index=chunk_idx,
                        breadcrumb=breadcrumb,
                        chunk_type=chunk_type or "document",
                        source_offset=src_off,
                        source_length=src_len,
                        source_detail=src_det,
                    )
                )
                tmp_embeddings.append(np.array(embedding_vec, dtype="float32"))

            emb_array = np.stack(tmp_embeddings)
            tmp_store.add_document(tmp_chunks, emb_array, namespace=ns)

            from .community import build_community

            n = build_community(
                tmp_path,
                embed_model,
                alpha=alpha,
                sim_threshold=sim_threshold,
                force=True,
            )
            log.info("Namespace %s: %d communities built", ns, n)

        import os

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    pg.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="chonk ingest — worker / coordinator modes for horizontal scale",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--worker",
        action="store_true",
        help="Run as a worker: pull jobs from queue and process them.",
    )
    mode.add_argument(
        "--coordinator",
        action="store_true",
        help="Run as coordinator: dispatch leases, build graphs, pause/resume.",
    )
    parser.add_argument(
        "--queue",
        required=True,
        metavar="DSN",
        help="PostgreSQL DSN for the ingest_queue / control tables.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        metavar="DSN",
        help="PostgreSQL DSN for the vector store.",
    )
    parser.add_argument(
        "--graph-interval",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Coordinator: seconds between graph builds (default 300).",
    )
    parser.add_argument(
        "--embed-model",
        default="BAAI/bge-large-en-v1.5",
        metavar="MODEL",
        help="SentenceTransformer model name for embedding (default BAAI/bge-large-en-v1.5).",
    )
    parser.add_argument(
        "--ner",
        action="store_true",
        help="Worker: run NER on each ingested document.",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        metavar="MODEL",
        help="spaCy model for NER (default en_core_web_sm).",
    )
    args = parser.parse_args()

    if args.worker:
        run_worker(
            args.queue,
            args.backend,
            embed_model=args.embed_model,
            run_ner=args.ner,
            spacy_model=args.spacy_model,
        )
    else:
        run_coordinator(
            args.queue,
            args.backend,
            graph_interval=args.graph_interval,
            embed_model=args.embed_model,
            spacy_model=args.spacy_model,
        )
        sys.exit(0)
