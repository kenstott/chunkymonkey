# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 7d268018-663f-42b4-ae26-dab0555ce04d
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Store: composed facade over DuckDBVectorBackend and RelationalStore."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._pool import ThreadLocalDuckDB
from ._protocol import VectorBackend  # noqa: F401 — runtime isinstance check
from ._relational import RelationalStore
from ._vector import DuckDBVectorBackend

if TYPE_CHECKING:
    from ..graph._context_graph import ContextEdge, ContextGraphStats
    from ..models import DocumentChunk

GLOBAL_NAMESPACE = "global"


@dataclass(frozen=True)
class NamespaceEvidence:
    """How much of a namespace's corpus mentions a given entity.

    Attributes:
        namespace: The namespace id.
        chunk_count: Chunks in this namespace associated with the entity.
        score: Summed association score (``chunk_entities.score``).
        share: ``chunk_count`` over the namespace's total chunk count — the
            size-normalised measure. Compare namespaces on this; compare
            absolute volume on ``chunk_count``.
    """

    namespace: str
    chunk_count: int
    score: float
    share: float


@dataclass(frozen=True)
class EntityLookup:
    """Why an entity lookup returned what it did.

    An empty ``ids`` has two very different causes — the name is not in the
    index, or it is and the caller's filter excluded it. This separates them.

    Attributes:
        ids: What :meth:`Store.resolve_entity_ids` returns for the same query.
        name_exists: The name is present under *some* entity type. With an empty
            ``ids`` this means the filter excluded it, not that the name is unknown.
        available_types: Every entity type the name exists under, ignoring the
            requested filter. Sorted.
        available_namespaces: Every namespace the name has evidence in, ignoring
            the requested filter. With an empty ``ids`` and a namespace filter
            set, this is what the caller should have asked for. Sorted.
        near_matches: Entity ids whose name slug contains the queried one, or is
            contained by it — so both an under- and over-specified query gets
            help ("mercury" finds mercury_systems; "mercury systems corp" finds
            both). Compared on the slug only, never the type prefix. Excludes
            ``ids``. Ordered by closeness in length, then id, and capped at
            ``NEAR_MATCH_LIMIT``; ``near_matches_truncated`` says when more existed.
        near_matches_truncated: More near matches existed than were returned.
    """

    ids: list[str]
    name_exists: bool
    available_types: list[str]
    available_namespaces: list[str]
    near_matches: list[str]
    near_matches_truncated: bool


NEAR_MATCH_LIMIT = 10

# A stored slug shorter than this is not used for contained-by matching — a
# two-character slug appears inside almost any query and would drown the result.
_MIN_CONTAINED_SLUG = 3


def _like_escape(value: str) -> str:
    """Escape LIKE wildcards in *value*.

    Name slugs collapse every non-alphanumeric character to ``_``, which is a
    single-character wildcard in LIKE — without escaping, ``acme_corp`` also
    matches ``acmeXcorp``. Used with ``ESCAPE '\\'``.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Store:
    """Composed storage facade backed by DuckDB (default) or PostgreSQL.

    DuckDB is the default for local development and single-process deployments.
    Pass ``dsn`` to use the PostgreSQL + pgvector backend for horizontal scale.

    Usage::

        # DuckDB (local dev)
        with Store("index.duckdb") as store:
            store.add_document(chunks, embeddings)
            results = store.search(query_vec, limit=5)

        # PostgreSQL (horizontal scale)
        with Store(dsn="postgresql://user:pass@host/db") as store:
            store.add_document(chunks, embeddings)
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedding_dim: int = 1024,
        read_only: bool = False,
        dsn: str | None = None,
        qdrant_url: str | None = None,
        qdrant_collection: str = "chonk",
        qdrant_catalog_path: str = ":memory:",
        qdrant_api_key: str | None = None,
        pinecone_api_key: str | None = None,
        pinecone_index: str = "chonk",
        pinecone_catalog_path: str = ":memory:",
        pinecone_cloud: str = "aws",
        pinecone_region: str = "us-east-1",
        weaviate_url: str | None = None,
        weaviate_api_key: str | None = None,
        weaviate_collection: str = "Chonk",
        weaviate_catalog_path: str = ":memory:",
    ) -> None:
        """Create a Store.

        Args:
            db_path: Path to DuckDB file, or ":memory:" for an in-memory store.
                     Ignored when ``dsn``, ``qdrant_url``, ``pinecone_api_key``,
                     or ``weaviate_url`` is provided.
            embedding_dim: Embedding vector dimension. Must match your model.
            read_only: Open in read-only mode (DuckDB only).
            dsn: PostgreSQL DSN (e.g. ``"postgresql://user:pass@host/db"``).
                 When set, uses PgVectorBackend instead of DuckDB.
            qdrant_url: Qdrant server URL (e.g. ``"http://localhost:6333"``).
                 When set, uses QdrantVectorBackend. Takes precedence over ``dsn``.
            qdrant_collection: Qdrant collection name (default: ``"chonk"``).
            qdrant_catalog_path: DuckDB catalog file for Qdrant backend metadata.
            qdrant_api_key: Optional Qdrant Cloud API key.
            pinecone_api_key: Pinecone API key. When set, uses PineconeVectorBackend.
                 Takes precedence over ``qdrant_url`` and ``dsn``.
            pinecone_index: Pinecone index name (default: ``"chonk"``).
            pinecone_catalog_path: DuckDB catalog file for Pinecone backend metadata.
            pinecone_cloud: Serverless cloud provider (default: ``"aws"``).
            pinecone_region: Serverless region (default: ``"us-east-1"``).
            weaviate_url: Weaviate Cloud cluster URL
                 (e.g. ``"https://abc.c0.us-east-1.aws.weaviate.cloud"``).
                 When set, uses WeaviateVectorBackend. Takes precedence over
                 ``pinecone_api_key``, ``qdrant_url``, and ``dsn``.
            weaviate_api_key: Weaviate Cloud API key.
            weaviate_collection: Weaviate collection name (default: ``"Chonk"``).
            weaviate_catalog_path: DuckDB catalog file for Weaviate backend metadata.
        """
        if weaviate_url is not None:
            from ._weaviate import WeaviateVectorBackend

            self._db: ThreadLocalDuckDB | None = None
            self.vector: VectorBackend = WeaviateVectorBackend(  # type: ignore[assignment]
                cluster_url=weaviate_url,
                api_key=weaviate_api_key or "",
                collection=weaviate_collection,
                embedding_dim=embedding_dim,
                catalog_path=weaviate_catalog_path,
            )
            assert isinstance(self.vector, VectorBackend)
            self.relational = None  # type: ignore
            return

        if pinecone_api_key is not None:
            from ._pinecone import PineconeVectorBackend

            self._db: ThreadLocalDuckDB | None = None
            self.vector: VectorBackend = PineconeVectorBackend(  # type: ignore[assignment]
                api_key=pinecone_api_key,
                index_name=pinecone_index,
                embedding_dim=embedding_dim,
                catalog_path=pinecone_catalog_path,
                cloud=pinecone_cloud,
                region=pinecone_region,
            )
            assert isinstance(self.vector, VectorBackend)
            self.relational = None  # type: ignore
            return

        if qdrant_url is not None:
            from ._qdrant import QdrantVectorBackend

            self._db: ThreadLocalDuckDB | None = None
            self.vector: VectorBackend = QdrantVectorBackend(  # type: ignore[assignment]
                url=qdrant_url,
                collection=qdrant_collection,
                embedding_dim=embedding_dim,
                catalog_path=qdrant_catalog_path,
                api_key=qdrant_api_key,
            )
            assert isinstance(self.vector, VectorBackend)
            self.relational = None  # type: ignore
            return

        if dsn is not None:
            from ._pg import PgVectorBackend

            self._db: ThreadLocalDuckDB | None = None  # type: ignore[no-redef]
            self.vector: VectorBackend = PgVectorBackend(dsn, embedding_dim=embedding_dim)  # type: ignore[assignment]  # property satisfies bool protocol at runtime
            assert isinstance(self.vector, VectorBackend)
            self.relational = None  # type: ignore
            return

        db_path = str(db_path)
        self._db = ThreadLocalDuckDB(db_path, read_only=read_only)
        self.vector = DuckDBVectorBackend(self._db, embedding_dim=embedding_dim)
        assert isinstance(self.vector, VectorBackend)
        if read_only:
            self.relational = None  # type: ignore
            return
        relational_url = f"duckdb:///{db_path}" if db_path != ":memory:" else "duckdb://"
        try:
            self.relational = RelationalStore(relational_url)
            self.relational.init_schema()
        except ImportError as e:
            # duckdb-engine or sqlalchemy is not installed; relational features are optional.
            # Install with: pip install duckdb-engine
            import logging

            logging.getLogger(__name__).warning(
                f"RelationalStore unavailable (duckdb-engine not installed): {e}. "
                "Install duckdb-engine for full SQLAlchemy+DuckDB support."
            )
            self.relational = None  # type: ignore
        except Exception as e:
            # DuckDB allows only one connection per file with the same config; duckdb_engine
            # opens a second connection which may fail with ConnectionException.  This is an
            # expected RelationalStore limitation, not a data-loss failure.  Any other error
            # is unexpected and must surface.
            if "different configuration" not in str(e):
                raise
            import logging

            logging.getLogger(__name__).warning(
                f"RelationalStore unavailable (DuckDB connection conflict — relational "
                f"features disabled): {e}"
            )
            self.relational = None  # type: ignore

    def add_document(
        self,
        chunks: list[DocumentChunk],
        embeddings: Any,  # noqa: ANN401
        namespace: str | None = None,
        source_id: str | None = None,
        domain_id: str | None = None,
        session_fingerprint: str | None = None,
    ) -> None:
        """Add chunks with embeddings. embeddings is np.ndarray shape (n, dim).

        Args:
            namespace: Optional partition key (e.g. "__base__" or a project ID).
                       None means no namespace — backwards-compatible default.
            source_id: Optional source registry ID for the originating source.
            domain_id: Optional domain registry ID (denormalization of source_id → domain).
            session_fingerprint: Optional fingerprint tagging community summary chunks.
        """
        self.vector.add_chunks(
            chunks,
            embeddings,
            namespace=namespace,
            source_id=source_id,
            domain_id=domain_id,
            session_fingerprint=session_fingerprint,
        )

    @staticmethod
    def session_fingerprint(domain_ids: list[str]) -> str:
        """Stable hex fingerprint of a sorted domain_ids set."""
        import hashlib

        key = ",".join(sorted(domain_ids))
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _domain_chunk_fingerprint(self, domain_ids: list[str]) -> str:
        """Stable fingerprint of the chunk_ids currently in *domain_ids*.

        Keyed on chunk identity rather than chunk count: an updated document
        that happens to produce the same number of chunks still changes the
        fingerprint, because chunk_id derives from content.
        """
        from ..graph._context_graph import _chunk_fingerprint

        rows = self.vector._conn.execute(
            "SELECT chunk_id FROM embeddings WHERE domain_id IN ({})".format(
                ", ".join("?" * len(domain_ids))
            ),
            domain_ids,
        ).fetchall()
        return _chunk_fingerprint([r[0] for r in rows])

    def community_cache_valid(self, fingerprint: str, domain_ids: list[str]) -> bool:
        """True if *fingerprint* is cached AND its chunk set is unchanged."""
        row = self.vector._conn.execute(
            "SELECT chunk_fingerprint FROM community_cache WHERE fingerprint = ?",
            [fingerprint],
        ).fetchone()
        if row is None or row[0] is None:
            return False
        return bool(row[0] == self._domain_chunk_fingerprint(domain_ids))

    def write_community_cache(self, fingerprint: str, domain_ids: list[str]) -> None:
        """Record a community cache entry after building communities for domain_ids."""
        import json as _json

        _count_row = self.vector._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE domain_id IN ({})".format(
                ", ".join("?" * len(domain_ids))
            ),
            domain_ids,
        ).fetchone()
        if _count_row is None:
            raise RuntimeError("COUNT(*) returned no rows")
        chunk_count = _count_row[0]
        self.vector._conn.execute(
            """
            INSERT INTO community_cache
                (fingerprint, domain_ids, chunk_count, chunk_fingerprint)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (fingerprint) DO UPDATE SET
                domain_ids        = excluded.domain_ids,
                chunk_count       = excluded.chunk_count,
                chunk_fingerprint = excluded.chunk_fingerprint,
                created_at        = now()
            """,
            [
                fingerprint,
                _json.dumps(sorted(domain_ids)),
                chunk_count,
                self._domain_chunk_fingerprint(domain_ids),
            ],
        ).fetchall()

    def invalidate_community_cache(self, domain_id: str) -> int:
        """Delete cache entries that include domain_id. Returns count deleted."""
        rows = self.vector._conn.execute(
            "SELECT fingerprint, domain_ids FROM community_cache"
        ).fetchall()
        import json as _json

        to_delete = [r[0] for r in rows if domain_id in _json.loads(r[1])]
        if not to_delete:
            return 0
        placeholders = ", ".join("?" * len(to_delete))
        # Delete stale community summary chunks
        self.vector._conn.execute(
            f"DELETE FROM embeddings WHERE session_fingerprint IN ({placeholders})",
            to_delete,
        ).fetchall()
        self.vector._conn.execute(
            f"DELETE FROM community_cache WHERE fingerprint IN ({placeholders})",
            to_delete,
        ).fetchall()
        self.vector._fts_dirty = True
        return len(to_delete)

    def search(
        self,
        query_embedding: Any,  # noqa: ANN401
        limit: int = 5,
        query_text: str | None = None,
        namespaces: list[str] | None = None,
        chunk_types: list[str] | None = None,
        domain_ids: list[str] | None = None,
        session_fingerprint: str | None = None,
        exclude_chunk_types: list[str] | None = None,
        entity_types: list[str] | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Hybrid or pure vector search.

        Args:
            namespaces: If provided, restrict results to rows in these namespaces.
                        None searches all namespaces — backwards-compatible default.
            chunk_types: If provided, restrict results to rows with these chunk_types.
                         None searches all chunk types — backwards-compatible default.
            domain_ids: If provided, restrict results to rows in these domain_ids.
                        Independent of namespaces; when both set both apply (AND).
            session_fingerprint: If provided, restrict results to rows with this
                        session_fingerprint. Used to filter community summary chunks.
            exclude_chunk_types: If provided, drop rows with these chunk_types. A bare
                        search returns every type; pass SYNTHETIC_CHUNK_TYPES when
                        retrieving document evidence so generated entity and community
                        rows are not cited as source text.
            entity_types: If provided, restrict results to chunks that mention at least
                        one entity of these types (e.g. ``["customer"]``). Reads the
                        denormalised ``entity_types`` column, which build_ner populates
                        after NER — chunks indexed but never NER'd match nothing.

        Returns:
            List of (chunk_id, score, DocumentChunk).
        """
        return self.vector.search(
            query_embedding,
            limit=limit,
            query_text=query_text,
            namespaces=namespaces,
            chunk_types=chunk_types,
            exclude_chunk_types=exclude_chunk_types,
            entity_types=entity_types,
            domain_ids=domain_ids,
            session_fingerprint=session_fingerprint,
        )

    # ------------------------------------------------------------------
    # Namespace / domain / source registry
    # ------------------------------------------------------------------

    def register_namespace(
        self,
        namespace_id: str,
        owner: str | None = None,
        description: str | None = None,
    ) -> None:
        """Upsert a namespace record."""
        self.vector._conn.execute(
            """
            INSERT INTO namespaces (namespace_id, owner, description)
            VALUES (?, ?, ?)
            ON CONFLICT (namespace_id) DO UPDATE SET
                owner       = excluded.owner,
                description = excluded.description,
                updated_at  = now()
            """,
            [namespace_id, owner, description],
        ).fetchall()

    def register_domain(
        self,
        domain_id: str,
        namespace_id: str,
        name: str,
        description: str | None = None,
        parent_id: str | None = None,
    ) -> None:
        """Upsert a domain record."""
        self.vector._conn.execute(
            """
            INSERT INTO domains (domain_id, namespace_id, name, description, parent_id)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (domain_id) DO UPDATE SET
                namespace_id = excluded.namespace_id,
                name         = excluded.name,
                description  = excluded.description,
                parent_id    = excluded.parent_id,
                updated_at   = now()
            """,
            [domain_id, namespace_id, name, description, parent_id],
        ).fetchall()

    def register_source(
        self,
        source_id: str,
        domain_id: str,
        type: str,
        uri: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a source record."""
        import json as _json

        config_json = _json.dumps(config) if config is not None else None
        self.vector._conn.execute(
            """
            INSERT INTO sources (source_id, domain_id, type, uri, config)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (source_id) DO UPDATE SET
                domain_id = excluded.domain_id,
                type      = excluded.type,
                uri       = excluded.uri,
                config    = excluded.config
            """,
            [source_id, domain_id, type, uri, config_json],
        ).fetchall()

    def resolve_domain_ids(
        self,
        namespace_domain_pairs: list[tuple[str, str]],
        include_global: bool = True,
    ) -> list[str]:
        """Resolve (namespace_id, domain_name) pairs to domain_ids, including all descendants.

        When include_global=True, always folds in all domains from namespace_id='global'.
        """
        domains_table = (
            "all_domains" if getattr(self.vector, "_global_attached", False) else "domains"
        )
        pairs = list(namespace_domain_pairs)
        if include_global:
            global_rows = self.vector._conn.execute(
                f"SELECT namespace_id, name FROM {domains_table} WHERE namespace_id = ?",
                [GLOBAL_NAMESPACE],
            ).fetchall()
            for row in global_rows:
                if (row[0], row[1]) not in pairs:
                    pairs.append((row[0], row[1]))

        if not pairs:
            return []

        values_placeholders = ", ".join("(?, ?)" for _ in pairs)
        params: list[str] = []
        for ns_id, dom_name in pairs:
            params.extend([ns_id, dom_name])

        sql = f"""
            WITH RECURSIVE domain_tree AS (
                SELECT domain_id FROM {domains_table}
                WHERE (namespace_id, name) IN (VALUES {values_placeholders})
                UNION ALL
                SELECT d.domain_id FROM {domains_table} d
                JOIN domain_tree dt ON d.parent_id = dt.domain_id
            )
            SELECT DISTINCT domain_id FROM domain_tree
        """
        rows = self.vector._conn.execute(sql, params).fetchall()
        return [r[0] for r in rows]

    def resolve_session(
        self,
        namespace_id: str,
        active_domains: list[str],
        include_global: bool = True,
    ) -> list[str]:
        """Resolve a session to domain_ids.

        A session belongs to one namespace. active_domains are the domain names
        within that namespace the user has activated. Global namespace domains
        are always folded in when include_global=True.

        Search sessions should open the store read-only so multiple concurrent
        sessions never conflict with each other or with a background Indexer::

            # Any number of these can run concurrently — read-only, no conflict.
            store = Store("user_alice.duckdb", read_only=True)
            store.attach_global("global.duckdb")
            domain_ids = store.resolve_session("user:alice", ["my_notes", "finance"])
            results = store.search(query_vec, domain_ids=domain_ids)
        """
        pairs = [(namespace_id, domain) for domain in active_domains]
        return self.resolve_domain_ids(pairs, include_global=include_global)

    def delete_domain(self, domain_id: str) -> int:
        """Delete all chunks for a domain_id.

        Also deletes associated chunk_entities and svo_triples rows.

        Returns:
            Number of chunks deleted from embeddings.
        """
        conn = self.vector._conn
        _count_row = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE domain_id = ?",
            [domain_id],
        ).fetchone()
        if _count_row is None:
            raise RuntimeError("COUNT(*) returned no rows")
        count_before = _count_row[0]

        # Collect chunk_ids for cascade deletes
        chunk_ids = [
            r[0]
            for r in conn.execute(
                "SELECT chunk_id FROM embeddings WHERE domain_id = ?",
                [domain_id],
            ).fetchall()
        ]

        if chunk_ids:
            import logging as _logging

            placeholders = ", ".join("?" * len(chunk_ids))
            try:
                conn.execute(
                    f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
            except Exception as e:
                _logging.getLogger(__name__).warning(f"DELETE chunk_entities failed: {e}")
                raise
            try:
                conn.execute(
                    f"DELETE FROM svo_triples WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
            except Exception as e:
                _logging.getLogger(__name__).warning(f"DELETE svo_triples failed: {e}")
                raise

        conn.execute(
            "DELETE FROM embeddings WHERE domain_id = ?",
            [domain_id],
        ).fetchall()
        self.vector._fts_dirty = True
        return count_before

    def list_namespaces(self) -> list[str]:
        """Return sorted namespace ids from the catalog.

        Backend-agnostic: reads through the vector backend's connection adapter,
        so it works against DuckDB and PostgreSQL alike.
        """
        rows = self.vector._conn.execute(
            "SELECT namespace_id FROM namespaces ORDER BY namespace_id"
        ).fetchall()
        return [r[0] for r in rows]

    def list_domains(self, namespace_id: str) -> list[str]:
        """Return sorted domain names for *namespace_id*."""
        rows = self.vector._conn.execute(
            "SELECT name FROM domains WHERE namespace_id = ? ORDER BY name",
            [namespace_id],
        ).fetchall()
        return [r[0] for r in rows]

    def namespace_cache_valid(self, namespace_id: str) -> bool:
        """True if namespace exists, has crawled sources, and has chunks."""
        import os

        conn = self.vector._conn

        ns_row = conn.execute(
            "SELECT 1 FROM namespaces WHERE namespace_id = ?",
            [namespace_id],
        ).fetchone()
        if ns_row is None:
            return False

        sources = conn.execute(
            """
            SELECT s.uri, s.last_crawled
            FROM sources s
            JOIN domains d ON s.domain_id = d.domain_id
            WHERE d.namespace_id = ?
            """,
            [namespace_id],
        ).fetchall()

        if not sources:
            return False

        for uri, last_crawled in sources:
            if last_crawled is None:
                return False
            if not any(
                uri.startswith(p)
                for p in ("http://", "https://", "github://", "s3://", "ftp://", "sftp://")
            ):
                try:
                    mtime = os.path.getmtime(uri)
                    if mtime > last_crawled.timestamp():
                        return False
                except OSError:
                    return False

        _count_row = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE namespace = ?",
            [namespace_id],
        ).fetchone()
        if _count_row is None:
            raise RuntimeError("COUNT(*) returned no rows")
        chunk_count = _count_row[0]
        return chunk_count > 0

    def _mark_namespace_built(self, namespace_id: str, phase: str) -> None:
        """Upsert namespace_build_log setting {phase}_built_at = now()."""
        valid_phases = {"chunks", "ner", "svo", "community"}
        if phase not in valid_phases:
            raise ValueError(f"Unknown phase {phase!r}; must be one of {sorted(valid_phases)}")
        col = f"{phase}_built_at"
        self.vector._conn.execute(
            f"""
            INSERT INTO namespace_build_log (namespace_id, {col})
            VALUES (?, now())
            ON CONFLICT (namespace_id) DO UPDATE SET {col} = now()
            """,
            [namespace_id],
        ).fetchall()

    def promote_domain(
        self,
        domain_name: str,
        from_namespace: str,
        to_namespace: str,
        target_db_path: str | Path,
    ) -> None:
        """Copy domain data cross-DB, then delete from this store."""
        from pathlib import Path as _Path

        target_db_path = str(_Path(target_db_path))

        conn = self.vector._conn

        row = conn.execute(
            "SELECT domain_id FROM domains WHERE namespace_id = ? AND name = ?",
            [from_namespace, domain_name],
        ).fetchone()
        if row is None:
            raise ValueError(f"Domain {domain_name!r} not found in namespace {from_namespace!r}")
        domain_id = row[0]

        try:
            conn.execute(f"ATTACH '{target_db_path}' AS _promote_target")

            conn.execute(
                """
                INSERT INTO _promote_target.domains
                    (domain_id, namespace_id, name, description, parent_id, created_at, updated_at)
                SELECT domain_id, ?, name, description, parent_id, created_at, now()
                FROM domains WHERE domain_id = ?
                ON CONFLICT (domain_id) DO UPDATE
                    SET namespace_id = excluded.namespace_id, updated_at = now()
                """,
                [to_namespace, domain_id],
            ).fetchall()

            conn.execute(
                """
                INSERT INTO _promote_target.embeddings
                    (chunk_id, document_name, section, chunk_index, content, breadcrumb,
                     chunk_type, source_offset, source_length, namespace, embedding,
                     source_detail, source_id, domain_id, session_fingerprint)
                SELECT chunk_id, document_name, section, chunk_index, content, breadcrumb,
                       chunk_type, source_offset, source_length, ?, embedding,
                       source_detail, source_id, domain_id, session_fingerprint
                FROM embeddings WHERE domain_id = ?
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                [to_namespace, domain_id],
            ).fetchall()

            conn.execute(
                """
                INSERT INTO _promote_target.chunk_entities
                    (chunk_id, entity_id, frequency, positions_json, score, namespace)
                SELECT chunk_id, entity_id, frequency, positions_json, score, ?
                FROM chunk_entities
                WHERE chunk_id IN (SELECT chunk_id FROM embeddings WHERE domain_id = ?)
                ON CONFLICT (chunk_id, entity_id) DO NOTHING
                """,
                [to_namespace, domain_id],
            ).fetchall()

            conn.execute(
                """
                INSERT INTO _promote_target.svo_triples
                    (chunk_id, subject_id, verb, object_id, confidence, namespace)
                SELECT chunk_id, subject_id, verb, object_id, confidence, ?
                FROM svo_triples
                WHERE chunk_id IN (SELECT chunk_id FROM embeddings WHERE domain_id = ?)
                ON CONFLICT DO NOTHING
                """,
                [to_namespace, domain_id],
            ).fetchall()

            self.delete_domain(domain_id)
            conn.execute("DELETE FROM domains WHERE domain_id = ?", [domain_id]).fetchall()

        finally:
            conn.execute("DETACH _promote_target")

    # ------------------------------------------------------------------
    # Global attach / detach
    # ------------------------------------------------------------------

    def attach_global(self, global_db_path: str | Path) -> None:
        """Attach a global read-only DuckDB and create union views.

        After calling this, all read queries transparently span both this
        store's tables and the global store's tables. Write queries always
        target only this store's base tables.

        Creates views: all_embeddings, all_chunk_entities, all_svo_triples,
        all_domains, all_sources, all_namespaces.

        Not supported for the PG backend; raises ``NotImplementedError``.
        """
        if self._db is None:
            raise NotImplementedError("attach_global is not supported for the PG backend.")
        from ._schema import CHUNK_ENTITIES_DDL, CHUNK_ENTITIES_MIGRATE_NAMESPACE

        conn = self.vector._conn

        # Ensure lazily-created local tables exist so views can reference them.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS svo_triples ("
            "  chunk_id   VARCHAR,"
            "  subject_id VARCHAR NOT NULL,"
            "  verb       VARCHAR NOT NULL,"
            "  object_id  VARCHAR NOT NULL,"
            "  confidence FLOAT   NOT NULL DEFAULT 1.0,"
            "  namespace  VARCHAR"
            ")"
        ).fetchall()
        conn.execute(CHUNK_ENTITIES_DDL).fetchall()
        conn.execute(CHUNK_ENTITIES_MIGRATE_NAMESPACE).fetchall()

        conn.execute(f"ATTACH '{global_db_path}' AS global_db (READ_ONLY)")

        def _global_has(table: str) -> bool:
            _count_row = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_catalog = 'global_db' AND table_name = ?",
                [table],
            ).fetchone()
            if _count_row is None:
                raise RuntimeError("COUNT(*) returned no rows")
            return _count_row[0] > 0

        # ── all_embeddings ──────────────────────────────────────────────────
        if _global_has("embeddings"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_embeddings AS
                SELECT chunk_id, document_name, section, chunk_index, content,
                       breadcrumb, chunk_type, source_offset, source_length,
                       namespace, source_detail, source_id, domain_id,
                       session_fingerprint, embedding
                FROM embeddings
                UNION ALL
                SELECT chunk_id, document_name, section, chunk_index, content,
                       breadcrumb, chunk_type, source_offset, source_length,
                       namespace, source_detail, source_id, domain_id,
                       session_fingerprint, embedding
                FROM global_db.embeddings
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_embeddings AS
                SELECT chunk_id, document_name, section, chunk_index, content,
                       breadcrumb, chunk_type, source_offset, source_length,
                       namespace, source_detail, source_id, domain_id,
                       session_fingerprint, embedding
                FROM embeddings
            """)

        # ── all_chunk_entities ──────────────────────────────────────────────
        if _global_has("chunk_entities"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_chunk_entities AS
                SELECT chunk_id, entity_id, frequency, positions_json, score, namespace
                FROM chunk_entities
                UNION ALL
                SELECT chunk_id, entity_id, frequency, positions_json, score, namespace
                FROM global_db.chunk_entities
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_chunk_entities AS
                SELECT chunk_id, entity_id, frequency, positions_json, score, namespace
                FROM chunk_entities
            """)

        # ── all_svo_triples ─────────────────────────────────────────────────
        if _global_has("svo_triples"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_svo_triples AS
                SELECT chunk_id, subject_id, verb, object_id, confidence, namespace
                FROM svo_triples
                UNION ALL
                SELECT chunk_id, subject_id, verb, object_id, confidence, namespace
                FROM global_db.svo_triples
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_svo_triples AS
                SELECT chunk_id, subject_id, verb, object_id, confidence, namespace
                FROM svo_triples
            """)

        # ── all_domains ─────────────────────────────────────────────────────
        if _global_has("domains"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_domains AS
                SELECT domain_id, namespace_id, name, description, parent_id,
                       created_at, updated_at
                FROM domains
                UNION ALL
                SELECT domain_id, namespace_id, name, description, parent_id,
                       created_at, updated_at
                FROM global_db.domains
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_domains AS
                SELECT domain_id, namespace_id, name, description, parent_id,
                       created_at, updated_at
                FROM domains
            """)

        # ── all_sources ─────────────────────────────────────────────────────
        if _global_has("sources"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_sources AS
                SELECT source_id, domain_id, type, uri, config, last_crawled
                FROM sources
                UNION ALL
                SELECT source_id, domain_id, type, uri, config, last_crawled
                FROM global_db.sources
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_sources AS
                SELECT source_id, domain_id, type, uri, config, last_crawled
                FROM sources
            """)

        # ── all_namespaces ──────────────────────────────────────────────────
        if _global_has("namespaces"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_namespaces AS
                SELECT namespace_id, owner, description, created_at, updated_at
                FROM namespaces
                UNION ALL
                SELECT namespace_id, owner, description, created_at, updated_at
                FROM global_db.namespaces
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_namespaces AS
                SELECT namespace_id, owner, description, created_at, updated_at
                FROM namespaces
            """)

        assert isinstance(
            self.vector, DuckDBVectorBackend
        )  # guaranteed by _db is not None guard above
        self.vector._global_attached = True

    def detach_global(self) -> None:
        """Drop the union views and detach the global DB.

        Not supported for the PG backend; raises ``NotImplementedError``.
        """
        if self._db is None:
            raise NotImplementedError("detach_global is not supported for the PG backend.")
        assert isinstance(
            self.vector, DuckDBVectorBackend
        )  # guaranteed by _db is not None guard above
        conn = self.vector._conn
        for view in (
            "all_embeddings",
            "all_chunk_entities",
            "all_svo_triples",
            "all_domains",
            "all_sources",
            "all_namespaces",
        ):
            conn.execute(f"DROP VIEW IF EXISTS {view}")
        conn.execute("DETACH global_db")
        self.vector._global_attached = False

    def delete_document(self, document_name: str) -> int:
        """Delete all chunks for a document, cascading to derived rows.

        Removes the document's ``chunk_entities``, ``chunk_clusters``, and
        ``svo_triples`` rows, its ``documents`` registry entry, and any entity
        left with no remaining reference. When a relational backend is attached,
        its ``chunk_entities`` rows are cleaned too.

        Returns:
            Number of chunks deleted.
        """
        chunk_ids: list[str] = []
        if self.relational is not None and isinstance(self.vector, DuckDBVectorBackend):
            chunk_ids = self.vector.chunk_ids_for_document(document_name)

        deleted = self.vector.delete_by_document(document_name)

        if chunk_ids:
            assert self.relational is not None  # guarded above
            self.relational.delete_entities_by_document(chunk_ids)

        return deleted

    # ── Entity descriptions ───────────────────────────────────────────────────

    def set_entity_description(self, entity_id: str, description: str) -> None:
        """Set description on an entity by ID."""
        self.vector._conn.execute(
            "UPDATE entities SET description = ? WHERE id = ?",
            [description, entity_id],
        )

    def set_entity_descriptions_batch(self, descriptions: dict[str, str]) -> int:
        """Set descriptions for multiple entities keyed by entity_id. Returns count."""
        conn = self.vector._conn
        rows = [(desc, eid) for eid, desc in descriptions.items()]
        if rows:
            conn.executemany("UPDATE entities SET description = ? WHERE id = ?", rows)
        return len(rows)

    def get_entity_descriptions(
        self,
        entity_ids: list[str],
        namespaces: list[str] | None = None,
    ) -> dict[str, str]:
        """Return ``{entity_id: description}`` for the given IDs.

        Args:
            entity_ids: Entity IDs to look up.
            namespaces: Restrict to entities associated with a chunk in one of
                these namespaces. ``None`` applies no restriction; an empty list
                matches nothing, mirroring ``search(namespaces=[])``.
        """
        if not entity_ids:
            return {}
        if namespaces is not None and not namespaces:
            return {}
        conn = self.vector._conn
        placeholders = ", ".join("?" * len(entity_ids))
        params: list[Any] = list(entity_ids)
        ns_clause = ""
        if namespaces is not None:
            ns_placeholders = ", ".join("?" * len(namespaces))
            ns_clause = (
                f" AND EXISTS (SELECT 1 FROM chunk_entities ce WHERE ce.entity_id = entities.id "
                f"AND COALESCE(ce.namespace, 'global') IN ({ns_placeholders}))"
            )
            params.extend(namespaces)
        rows = conn.execute(
            f"SELECT id, COALESCE(description, '') FROM entities "
            f"WHERE id IN ({placeholders}){ns_clause}",
            params,
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_chunk_entity_ids(self, chunk_id: str) -> list[str]:
        """Return all entity_ids associated with a chunk."""
        conn = self.vector._conn
        rows = conn.execute(
            "SELECT entity_id FROM chunk_entities WHERE chunk_id = ?",
            [chunk_id],
        ).fetchall()
        return [r[0] for r in rows]

    # ── Entity aliases ────────────────────────────────────────────────────────

    def add_entity_alias(
        self,
        alias: str,
        entity_id: str,
        source: str = "llm",
        namespace: str = GLOBAL_NAMESPACE,
    ) -> None:
        """Register *alias* as an alternate name for *entity_id*.

        One alias may name several entities in the same namespace — "John Doe"
        can be both ``customer:john_doe`` and ``employee:john_doe``. Each
        (alias, namespace, entity_id) mapping is its own row; adding one never
        displaces another. Re-registering an existing mapping is a no-op unless
        source is ``'user'``, which refreshes that row's source.
        """
        conn = self.vector._conn
        if source == "user":
            conn.execute(
                """
                INSERT INTO entity_aliases (alias, entity_id, namespace, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (alias, namespace, entity_id) DO UPDATE SET
                    source = excluded.source
                """,
                [alias, entity_id, namespace, source],
            )
        else:
            conn.execute(
                """
                INSERT INTO entity_aliases (alias, entity_id, namespace, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (alias, namespace, entity_id) DO NOTHING
                """,
                [alias, entity_id, namespace, source],
            )

    def add_entity_aliases_batch(
        self,
        aliases: dict[str, str],
        source: str = "llm",
        namespace: str = GLOBAL_NAMESPACE,
    ) -> int:
        """Register multiple aliases. ``aliases`` maps alias → entity_id.

        A dict holds one entity_id per alias; call repeatedly to map one alias to
        several entities. Returns the count inserted — an existing mapping for the
        same (alias, namespace, entity_id) is skipped for non-user sources.
        """
        written = 0
        for alias, entity_id in aliases.items():
            before = self.vector._conn.execute(
                "SELECT source FROM entity_aliases "
                "WHERE alias = ? AND namespace = ? AND entity_id = ?",
                [alias, namespace, entity_id],
            ).fetchone()
            if before and source != "user":
                continue
            self.add_entity_alias(alias, entity_id, source=source, namespace=namespace)
            written += 1
        return written

    def resolve_entity_alias(
        self,
        alias: str,
        namespace: str = GLOBAL_NAMESPACE,
    ) -> str | None:
        """Return one entity_id for *alias*, or None if unknown.

        An alias may name several entities of different types in the same
        namespace; this returns the lowest-sorting ID of them. Use
        :meth:`resolve_entity_aliases` when the caller must see all of them.
        """
        row = self.vector._conn.execute(
            "SELECT entity_id FROM entity_aliases WHERE alias = ? AND namespace = ? "
            "ORDER BY entity_id",
            [alias, namespace],
        ).fetchone()
        return row[0] if row else None

    def resolve_entity_aliases(
        self,
        alias: str,
        namespace: str = GLOBAL_NAMESPACE,
    ) -> list[str]:
        """Return every entity_id *alias* names in *namespace*, sorted.

        ``["customer:john_doe", "employee:john_doe"]`` when one person is both.
        """
        rows = self.vector._conn.execute(
            "SELECT entity_id FROM entity_aliases WHERE alias = ? AND namespace = ? "
            "ORDER BY entity_id",
            [alias, namespace],
        ).fetchall()
        return [r[0] for r in rows]

    def get_entity_namespaces(self, entity_id: str) -> list[str]:
        """Return every namespace *entity_id* was **declared** in, sorted.

        This is declaration provenance, not evidence: it reads ``entity_aliases``,
        whose rows record which namespace's vocabulary contributed the entity. A
        shared customer list declared once at ``global`` returns ``["global"]``
        even when documents in several namespaces mention it. Use
        :meth:`get_entity_namespace_evidence` for where the entity actually appears.
        """
        rows = self.vector._conn.execute(
            "SELECT DISTINCT namespace FROM entity_aliases WHERE entity_id = ? ORDER BY namespace",
            [entity_id],
        ).fetchall()
        return [r[0] for r in rows]

    def resolve_entity_ids(
        self,
        name: str,
        entity_type: str | None = None,
        namespaces: list[str] | None = None,
    ) -> list[str]:
        """Resolve an unqualified entity name to every matching entity id.

        Callers should not have to know the type prefix. ``"mercury"`` returns
        every entity of that name — ``["customer:mercury", "element:mercury"]`` —
        and the caller disambiguates, or uses all of them. Pass *entity_type* to
        narrow to one. An already-qualified id (``"customer:mercury"``) is
        returned as-is when it exists, so callers can pass either form.

        Matching is on the name slug, so ``"Acme Corp"``, ``"acme corp"``, and
        ``"acme_corp"`` are the same query.

        Args:
            name: Entity name, qualified or not.
            entity_type: Restrict to this type.
            namespaces: Restrict to entities with evidence in these namespaces —
                that is, associated with a chunk there, the same notion
                :meth:`get_entity_namespace_evidence` reports. ``None`` applies no
                restriction; ``[]`` matches nothing, mirroring ``search``.

        Returns:
            Matching entity ids, sorted. Empty when nothing matched — use
            :meth:`explain_entity_lookup` to find out why.
        """
        from ..ner._vocabulary import _auto_id, split_typed_id

        if namespaces is not None and not namespaces:
            return []

        conn = self.vector._conn
        given_type, slug = split_typed_id(name)
        if given_type:
            # Already qualified — confirm it exists rather than guessing.
            rows = conn.execute("SELECT id FROM entities WHERE id = ?", [name]).fetchall()
            ids = [r[0] for r in rows]
        else:
            slug = _auto_id(slug)
            rows = conn.execute(
                "SELECT id FROM entities WHERE id LIKE ? ESCAPE '\\' ORDER BY id",
                [f"%:{_like_escape(slug)}"],
            ).fetchall()
            ids = [r[0] for r in rows]

        if entity_type is not None:
            prefix = f"{_auto_id(entity_type)}:"
            ids = [i for i in ids if i.startswith(prefix)]
        if namespaces is not None and ids:
            with_evidence = self._entities_with_evidence_in(ids, namespaces)
            ids = [i for i in ids if i in with_evidence]
        return ids

    def _entities_with_evidence_in(self, entity_ids: list[str], namespaces: list[str]) -> set[str]:
        """Return the subset of *entity_ids* associated with a chunk in *namespaces*."""
        if not entity_ids or not namespaces:
            return set()
        id_ph = ", ".join("?" * len(entity_ids))
        ns_ph = ", ".join("?" * len(namespaces))
        rows = self.vector._conn.execute(
            f"SELECT DISTINCT entity_id FROM chunk_entities "  # noqa: S608
            f"WHERE entity_id IN ({id_ph}) AND COALESCE(namespace, ?) IN ({ns_ph})",
            [*entity_ids, GLOBAL_NAMESPACE, *namespaces],
        ).fetchall()
        return {r[0] for r in rows}

    def explain_entity_lookup(
        self,
        name: str,
        entity_type: str | None = None,
        namespaces: list[str] | None = None,
    ) -> EntityLookup:
        """Explain what :meth:`resolve_entity_ids` found, and what it did not.

        A miss is a normal outcome, so neither method raises. This one answers
        the question an empty list cannot: whether the name is absent from the
        index, or present under a type the caller did not ask for.

        ``resolve_entity_ids("Mercury", entity_type="customer")`` returning ``[]``
        while ``element:mercury`` exists is a caller mistake with an obvious fix;
        the same empty list for an unknown name is not. ``name_exists``,
        ``available_types``, and ``available_namespaces`` separate them — the last
        one covering the same mistake made with a namespace filter.

        Takes the same filters as :meth:`resolve_entity_ids` so the two answer the
        same question; the reported alternatives always ignore those filters.
        """
        from ..ner._vocabulary import _auto_id, split_typed_id

        conn = self.vector._conn
        ids = self.resolve_entity_ids(name, entity_type=entity_type, namespaces=namespaces)

        _given_type, raw_slug = split_typed_id(name)
        slug = _auto_id(raw_slug)

        escaped = _like_escape(slug)
        all_typed = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM entities WHERE id LIKE ? ESCAPE '\\' ORDER BY id",
                [f"%:{escaped}"],
            ).fetchall()
        ]
        available_types = sorted({split_typed_id(i)[0] for i in all_typed})
        available_namespaces: list[str] = []
        if all_typed:
            id_ph = ", ".join("?" * len(all_typed))
            available_namespaces = sorted(
                r[0]
                for r in conn.execute(
                    f"SELECT DISTINCT COALESCE(namespace, ?) FROM chunk_entities "  # noqa: S608
                    f"WHERE entity_id IN ({id_ph})",
                    [GLOBAL_NAMESPACE, *all_typed],
                ).fetchall()
            )

        exact = set(all_typed) | set(ids)
        # Both directions, on the slug only:
        #   contains     — stored slug holds the query ("mercury" -> mercury_systems)
        #   contained by — query holds the stored slug ("mercury systems corp" -> both)
        # strpos avoids treating a stored slug's "_" as a LIKE wildcard. Very short
        # slugs are excluded from the contained-by arm or they match nearly anything.
        near = [
            r[0]
            for r in conn.execute(
                """
                SELECT id FROM entities
                WHERE split_part(id, ':', 2) LIKE ? ESCAPE '\\'
                   OR (
                        length(split_part(id, ':', 2)) >= ?
                        AND strpos(?, split_part(id, ':', 2)) > 0
                   )
                ORDER BY abs(length(split_part(id, ':', 2)) - ?), id
                """,
                [f"%{escaped}%", _MIN_CONTAINED_SLUG, slug, len(slug)],
            ).fetchall()
            if r[0] not in exact
        ]
        return EntityLookup(
            ids=ids,
            name_exists=bool(all_typed),
            available_types=available_types,
            available_namespaces=available_namespaces,
            near_matches=near[:NEAR_MATCH_LIMIT],
            near_matches_truncated=len(near) > NEAR_MATCH_LIMIT,
        )

    def get_entity_namespace_evidence(self, entity_id: str) -> list[NamespaceEvidence]:
        """Return the namespaces whose documents actually mention *entity_id*.

        Evidence, not declaration — reads ``chunk_entities``, so the same customer
        seen across several divisions ranks by how much each division talks about
        them. Ordered by score, then chunk count, then namespace.

        ``share`` is ``chunk_count / (chunks in that namespace)``. Without it a
        large namespace outranks a small one on volume alone; a division with 2 of
        its 3 chunks mentioning the customer is more about them than one with 5 of
        5000. Both measures are returned so the caller can choose.

        A chunk row exists for every association — ``_check_cache`` raises on an
        orphaned ``chunk_entities`` row — so the denominator is never zero here.
        """
        conn = self.vector._conn
        rows = conn.execute(
            # GROUP BY 1 (ordinal): DuckDB will not match a parameterised
            # COALESCE() in the GROUP BY against the same expression in SELECT.
            "SELECT COALESCE(namespace, ?) AS ns, COUNT(*), SUM(score) "
            "FROM chunk_entities WHERE entity_id = ? "
            "GROUP BY 1",
            [GLOBAL_NAMESPACE, entity_id],
        ).fetchall()
        if not rows:
            return []

        totals = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT COALESCE(namespace, ?) AS ns, COUNT(*) FROM embeddings GROUP BY 1",
                [GLOBAL_NAMESPACE],
            ).fetchall()
        }

        evidence = [
            NamespaceEvidence(
                namespace=ns,
                chunk_count=count,
                score=float(score_sum or 0.0),
                share=count / totals[ns],
            )
            for ns, count, score_sum in rows
        ]
        evidence.sort(key=lambda e: (-e.score, -e.chunk_count, e.namespace))
        return evidence

    def get_entity_aliases(
        self,
        entity_id: str,
        namespace: str = GLOBAL_NAMESPACE,
    ) -> list[str]:
        """Return all aliases registered for *entity_id*."""
        rows = self.vector._conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ? AND namespace = ?",
            [entity_id, namespace],
        ).fetchall()
        return [r[0] for r in rows]

    def get_entity_aliases_by_names(
        self,
        entity_names: list[str],
        namespace: str = GLOBAL_NAMESPACE,
    ) -> dict[str, list[str]]:
        """Return ``{entity_name: [alias, ...]}`` for the given names.

        Names with no aliases are omitted from the result.
        """
        if not entity_names:
            return {}
        conn = self.vector._conn
        placeholders = ", ".join("?" * len(entity_names))
        rows = conn.execute(
            f"SELECT e.name, ea.alias "
            f"FROM entity_aliases ea "
            f"JOIN entities e ON ea.entity_id = e.id "
            f"WHERE e.name IN ({placeholders}) AND ea.namespace = ?",
            [*entity_names, namespace],
        ).fetchall()
        result: dict[str, list[str]] = {}
        for name, alias in rows:
            result.setdefault(name, []).append(alias)
        return result

    _FORWARD_HIERARCHY_VERBS = frozenset(
        {
            "type_of",
            "instance_of",
            "classified_as",
            "part_of",
            "member_of",
            "extends",
        }
    )
    _REVERSE_HIERARCHY_VERBS = frozenset({"contains", "composed_of"})

    def get_entity_parents(
        self,
        entity_names: list[str],
        namespace: str = GLOBAL_NAMESPACE,
    ) -> dict[str, tuple[str, str]]:
        """Return ``{entity_name: (parent_name, svo_verb)}`` for the given names.

        Only entries with a hierarchy SVO triple are returned.
        For forward verbs (type_of, part_of, etc.) the object entity is the parent.
        For reverse verbs (contains, composed_of) the subject entity is the parent.
        When multiple triples exist for a name, the first result is returned.
        """
        if not entity_names:
            return {}
        conn = self.vector._conn
        placeholders = ", ".join("?" * len(entity_names))
        fwd_verbs = ", ".join(f"'{v}'" for v in sorted(self._FORWARD_HIERARCHY_VERBS))
        rev_verbs = ", ".join(f"'{v}'" for v in sorted(self._REVERSE_HIERARCHY_VERBS))
        rows = conn.execute(
            f"SELECT child_e.name, parent_e.name, st.verb "
            f"FROM svo_triples st "
            f"JOIN entities child_e  ON st.subject_id = child_e.id "
            f"JOIN entities parent_e ON st.object_id  = parent_e.id "
            f"WHERE child_e.name IN ({placeholders}) "
            f"  AND st.verb IN ({fwd_verbs}) "
            f"  AND (st.namespace = ? OR st.namespace IS NULL) "
            f"UNION ALL "
            f"SELECT child_e.name, parent_e.name, st.verb "
            f"FROM svo_triples st "
            f"JOIN entities child_e  ON st.object_id  = child_e.id "
            f"JOIN entities parent_e ON st.subject_id = parent_e.id "
            f"WHERE child_e.name IN ({placeholders}) "
            f"  AND st.verb IN ({rev_verbs}) "
            f"  AND (st.namespace = ? OR st.namespace IS NULL)",
            [*entity_names, namespace, *entity_names, namespace],
        ).fetchall()
        result: dict[str, tuple[str, str]] = {}
        for child_name, parent_name, verb in rows:
            if child_name not in result:
                result[child_name] = (parent_name, verb)
        return result

    def build_context_graph(
        self,
        namespace: str | None = "global",
        min_weight: float = 0.1,
        force: bool = False,
        algorithm: str = "agglomerative",
        min_chunks: int = 10,
    ) -> ContextGraphStats | dict[str, ContextGraphStats]:
        """Build context graph edges for one or all namespaces.

        If *namespace* is ``None``, builds for every namespace present in
        ``chunk_entities`` and returns a ``{namespace: ContextGraphStats}`` dict.
        Otherwise builds for the specified namespace and returns a single
        :class:`ContextGraphStats`.

        Not supported for the PG backend; raises ``NotImplementedError``.
        """
        if self._db is None:
            raise NotImplementedError(
                "build_context_graph is not supported for the PG backend. "
                "Use --coordinator mode to run graph builds."
            )
        if namespace is None:
            from ..graph._context_graph import build_context_graph_all_namespaces

            return build_context_graph_all_namespaces(
                self._db.conn,
                min_weight=min_weight,
                force=force,
                algorithm=algorithm,
                min_chunks=min_chunks,
            )
        from ..graph._context_graph import build_context_graph_edges

        return build_context_graph_edges(
            self._db.conn,
            namespace=namespace,
            min_weight=min_weight,
            force=force,
            algorithm=algorithm,
            min_chunks=min_chunks,
        )

    def get_context_graph(
        self,
        entity_id: str,
        namespace: str = "global",
        min_weight: float = 0.1,
    ) -> list[ContextEdge]:
        if self._db is None:
            raise NotImplementedError("get_context_graph is not supported for the PG backend.")
        from ..graph._context_graph import get_context_graph_edges

        return get_context_graph_edges(
            self._db.conn, entity_id, namespace=namespace, min_weight=min_weight
        )

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self.vector.count()

    def close(self) -> None:
        """Close the underlying storage connection."""
        if self._db is not None:
            self._db.close()
        else:
            self.vector.close()  # type: ignore[union-attr]

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
