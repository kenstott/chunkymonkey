# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 122f06b0-3685-465e-9ca2-de49c1829517
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Config-driven build and search: build(config) -> Index."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._config import ChonkConfig, EmbedConfig, IndexConfig, LoaderConfig
from .extractors import Extractor
from .indexer import IndexHandle
from .loader import DocumentLoader
from .models import DocumentChunk, ScoredChunk
from .storage import Store

if TYPE_CHECKING:
    import numpy as np

    from .search._enhanced import EnhancedSearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_id(*parts: str) -> str:
    """Stable 16-char hex ID from string parts."""
    return hashlib.sha1(":".join(parts).encode(), usedforsecurity=False).hexdigest()[:16]


def _make_extractor(name: str) -> Extractor:
    if name == "edgar":
        from .extractors._edgar import EdgarExtractor

        return EdgarExtractor(infer_bold_headings=True)
    raise ValueError(f"Unknown extractor: {name!r}")


def _make_loader(lc: LoaderConfig, enrich_override: bool | None = None) -> DocumentLoader:
    extractors = [_make_extractor(e) for e in lc.extra_extractors]
    enrich = enrich_override if enrich_override is not None else lc.enrich_context
    return DocumentLoader(
        min_chunk_size=lc.min_chunk_size,
        max_chunk_size=lc.max_chunk_size,
        enrich_context=enrich,
        extra_extractors=extractors or None,
    )


def _ingest_glob(loader: DocumentLoader, src: dict[str, Any]) -> list[DocumentChunk]:
    base = Path(src["path"])
    pattern = src.get("pattern", "*")
    prefix = src.get("name_prefix", "")
    doc_type = src.get("doc_type")
    chunks = []
    for p in sorted(f for f in base.glob(pattern) if not f.name.startswith(".")):
        name = prefix + p.stem
        if doc_type:
            chunks.extend(loader.load_bytes(p.read_bytes(), name=name, doc_type=doc_type))
        else:
            chunks.extend(loader.load(str(p), name=name))
    return chunks


def _ingest_json_array(loader: DocumentLoader, src: dict[str, Any]) -> list[DocumentChunk]:
    path = Path(src["path"])
    array_field = src["array_field"]
    id_path: list[str] = src.get("id_path", "").split(".") if src.get("id_path") else []
    obj = json.loads(path.read_text())
    items = obj.get(array_field, []) if isinstance(obj, dict) else obj
    chunks = []
    for item in items:
        name = item
        for key in id_path:
            name = name.get(key, {}) if isinstance(name, dict) else "unknown"
        name = str(name) if name else path.stem
        payload = json.dumps({array_field: [item]}).encode()
        chunks.extend(loader.load_bytes(payload, name=name, doc_type="json"))
    return chunks


def _ingest_sql(loader: DocumentLoader, src: dict[str, Any]) -> list[DocumentChunk]:
    connection = src["connection"]
    query = src["query"].strip()
    name = src.get("name", "sql_source")
    name_col = src.get("name_col")
    content_col = src.get("content_col")
    if name_col and content_col:
        import duckdb

        db_path = connection.replace("duckdb:///", "")
        conn = duckdb.connect(db_path, read_only=True)
        rows = conn.execute(query).fetchall()
        col_names = [d[0] for d in conn.description]
        conn.close()
        name_idx = col_names.index(name_col)
        content_idx = col_names.index(content_col)
        chunks = []
        for row in rows:
            chunks.extend(
                loader.load_bytes(
                    str(row[content_idx]).encode(), name=str(row[name_idx]), doc_type="text"
                )
            )
        return chunks
    return loader.load_from_db(connection, queries={name: query})


_INGEST_FNS = {
    "glob": _ingest_glob,
    "json_array": _ingest_json_array,
    "sql": _ingest_sql,
}


def _embed_chunks(chunks: list[DocumentChunk], ec: EmbedConfig) -> np.ndarray:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model_name = ec.model
    batch_size = ec.batch_size
    model = SentenceTransformer(model_name)
    texts = [c.content for c in chunks]
    vecs = []
    for i in range(0, len(texts), batch_size):
        v = model.encode(
            texts[i : i + batch_size],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        vecs.append(np.array(v, dtype="float32"))
        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 10 == 0:
            print(f"  embedded {done:,}/{len(texts):,}", flush=True)
    return np.vstack(vecs)


def _embed_texts(texts: list[str], model_name: str, batch_size: int = 256) -> np.ndarray:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vecs = []
    for i in range(0, len(texts), batch_size):
        v = model.encode(
            texts[i : i + batch_size], show_progress_bar=False, normalize_embeddings=True
        )
        vecs.append(np.array(v, dtype="float32"))
    return np.vstack(vecs)


def _ingest_source(src: dict[str, Any], loader: DocumentLoader) -> list[DocumentChunk]:
    stype = src.get("type") or ""
    fn = _INGEST_FNS.get(stype)
    if fn is None:
        raise ValueError(f"Unknown source type: {stype!r}")
    return fn(loader, src)


# ---------------------------------------------------------------------------
# Index — returned by build()
# ---------------------------------------------------------------------------


class Index:
    """Returned by :func:`build`. Wraps a fully-built Store with search and RT mutation.

    Domain names are fully-qualified strings (e.g. ``"sales/north-america/q1"``).
    The client owns the hierarchy and passes explicit FQ name lists when filtering.

    Args:
        store: Open Store instance.
        embed_model: SentenceTransformer model name — used for query embedding and rebuilds.
        search_defaults: Constructor kwargs forwarded to EnhancedSearch.
        index_cfg: ``index:`` config section — drives rebuild phases.
        loader_cfg: ``loader:`` config section — drives source ingestion.
        embed_cfg: ``embed:`` config section.
    """

    def __init__(
        self,
        store: Store,
        embed_model: str,
        search_defaults: dict[str, Any],
        index_cfg: IndexConfig,
        loader_cfg: LoaderConfig,
        embed_cfg: EmbedConfig,
    ) -> None:
        self._store = store
        self._embed_model = embed_model
        self._search_defaults = search_defaults
        self._index_cfg = index_cfg
        self._loader_cfg = loader_cfg
        self._embed_cfg = embed_cfg
        self._enhanced: EnhancedSearch | None = None
        self._domain_map: dict[str, dict[str, str]] = self._load_domain_map()
        _routing_fn = self._make_routing_fn()
        self.namespace_filter_llm_fn: Callable[[str], str] | None = _routing_fn
        self.domain_filter_llm_fn: Callable[[str], str] | None = _routing_fn

    # -- internal --------------------------------------------------------------

    def _load_domain_map(self) -> dict[str, dict[str, str]]:
        """Reload {namespace_id: {fq_domain_name: domain_id}} from DB."""
        rows = self._store.vector._conn.execute(
            "SELECT namespace_id, name, domain_id FROM domains"
        ).fetchall()
        result: dict[str, dict[str, str]] = {}
        for ns, name, did in rows:
            result.setdefault(ns, {})[name] = did
        return result

    def _refresh_domain_map(self) -> None:
        self._domain_map = self._load_domain_map()

    def _resolve_domains(
        self,
        namespaces: list[str] | None,
        domains: list[str] | None,
    ) -> list[str] | None:
        if namespaces is None and domains is None:
            return None
        target_ns = namespaces or list(self._domain_map.keys())
        ids: list[str] = []
        if domains is not None:
            for ns in target_ns:
                ns_map = self._domain_map.get(ns, {})
                for name in domains:
                    if name in ns_map:
                        ids.append(ns_map[name])
        else:
            for ns in target_ns:
                ids.extend(self._domain_map.get(ns, {}).values())
        return ids or None

    def _get_or_register_domain(
        self, namespace_id: str, domain_name: str, description: str | None = None
    ) -> str:
        ns_map = self._domain_map.get(namespace_id, {})
        if domain_name in ns_map:
            return ns_map[domain_name]
        did = _stable_id(namespace_id, domain_name)
        self._store.register_domain(did, namespace_id, domain_name, description=description)
        self._domain_map.setdefault(namespace_id, {})[domain_name] = did
        return did

    def _make_routing_fn(self) -> Callable[[str], str] | None:
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            return None
        model = self._index_cfg.svo_model
        try:
            from openai import OpenAI as _OpenAI

            _client = _OpenAI()

            def _fn(prompt: str) -> str:
                resp = _client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return resp.choices[0].message.content or ""

            return _fn
        except ImportError:
            return None

    def _make_embed_fn(self) -> Callable[[list[str]], np.ndarray]:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self._embed_model)

        def _embed_fn(texts: list[str]) -> np.ndarray:
            return np.array(
                model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                dtype="float32",
            )

        return _embed_fn

    def _get_search(self) -> EnhancedSearch:
        if self._enhanced is None:
            from .search._enhanced import EnhancedSearch

            self._enhanced = EnhancedSearch(
                self._store,
                embed_fn=self._make_embed_fn(),
                **self._search_defaults,
            )
        return self._enhanced

    def _invalidate_search(self) -> None:
        """Drop cached EnhancedSearch so it reloads indexes on next call."""
        self._enhanced = None

    # -- context manager -------------------------------------------------------

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._store.close()

    # -- properties ------------------------------------------------------------

    @property
    def store(self) -> Store:
        return self._store

    # -- domain / namespace inspection -----------------------------------------

    def domains(self, namespace: str | None = None) -> dict[str, list[str]]:
        """Return registered fully-qualified domain names.

        Args:
            namespace: If set, return only domains for this namespace.

        Returns:
            ``{namespace_id: [fq_domain_name, ...]}`` sorted dict.
        """
        if namespace is not None:
            return {namespace: sorted(self._domain_map.get(namespace, {}).keys())}
        return {ns: sorted(names.keys()) for ns, names in self._domain_map.items()}

    # -- RT mutations ----------------------------------------------------------

    def add_namespace(self, namespace_id: str, description: str | None = None) -> None:
        """Register a namespace. No-op if already registered."""
        self._store.register_namespace(namespace_id, description=description)
        self._domain_map.setdefault(namespace_id, {})

    def remove_namespace(self, namespace_id: str) -> None:
        """Delete a namespace and all its domains, chunks, and secondary index rows."""
        conn = self._store.vector._conn
        domain_ids = list(self._domain_map.get(namespace_id, {}).values())
        if domain_ids:
            # placeholders is a string of '?' param markers (one per domain_id), not
            # user data — every value is bound via parameters. Safe interpolation.
            placeholders = ", ".join("?" * len(domain_ids))
            conn.execute(
                f"DELETE FROM chunk_entities WHERE chunk_id IN (SELECT chunk_id FROM embeddings WHERE domain_id IN ({placeholders}))",  # noqa: E501
                domain_ids,
            )  # nosec B608  # placeholders are param markers, not user data
            conn.execute(
                f"DELETE FROM svo_triples WHERE chunk_id IN (SELECT chunk_id FROM embeddings WHERE domain_id IN ({placeholders}))",  # noqa: E501
                domain_ids,
            )  # nosec B608  # placeholders are param markers, not user data
            conn.execute(f"DELETE FROM embeddings WHERE domain_id IN ({placeholders})", domain_ids)  # nosec B608  # placeholders are param markers, not user data
            conn.execute(f"DELETE FROM domains WHERE domain_id IN ({placeholders})", domain_ids)  # nosec B608  # placeholders are param markers, not user data
        conn.execute("DELETE FROM namespaces WHERE namespace_id = ?", [namespace_id])
        self._domain_map.pop(namespace_id, None)
        self._invalidate_search()

    def add_domain(
        self, namespace_id: str, domain_name: str, description: str | None = None
    ) -> str:
        """Register a domain. Returns the domain_id. No-op if already registered.

        Args:
            namespace_id: Parent namespace (must be registered first).
            domain_name: Fully-qualified domain name, e.g. ``"sales/north-america/q1"``.
            description: Optional human-readable description.

        Returns:
            Stable domain_id string.
        """
        return self._get_or_register_domain(namespace_id, domain_name, description)

    def remove_domain(self, namespace_id: str, domain_name: str) -> int:
        """Delete a domain and all its chunks. Returns chunk count deleted.

        Args:
            namespace_id: Namespace containing the domain.
            domain_name: Fully-qualified domain name.
        """
        ns_map = self._domain_map.get(namespace_id, {})
        domain_id = ns_map.get(domain_name)
        if domain_id is None:
            return 0
        n = self._store.delete_domain(domain_id)
        self._domain_map.get(namespace_id, {}).pop(domain_name, None)
        self._invalidate_search()
        return n

    def add_source(
        self,
        src: dict[str, Any],
        *,
        rebuild: bool = True,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_complete: Callable[[int], None] | None = None,
        on_error: Callable[[str, Exception], None] | None = None,
    ) -> IndexHandle | None:
        """Ingest a new source, register its domain, and optionally trigger async rebuild.

        Args:
            src: Source dict — same schema as a ``sources:`` entry in the config.
                 Required keys: ``type``, and type-specific path/connection.
                 Optional keys: ``name``, ``namespace``, ``domain``, ``enrich_context``.
            rebuild: If True (default), trigger async secondary-index rebuild for the
                     namespace after ingest. Returns an :class:`IndexHandle` you can
                     ``.join()`` to wait for completion.
            on_progress: ``(phase, done, total) -> None`` callback for rebuild phases.
            on_complete: ``(total_chunks) -> None`` callback on rebuild success.
            on_error: ``(phase, exc) -> None`` callback on rebuild error.

        Returns:
            :class:`IndexHandle` if rebuild=True, else None.
        """
        namespace_id = src.get("namespace") or "global"
        domain_name = src.get("domain") or src.get("name", src.get("type", "unnamed"))
        enrich_override = src.get("enrich_context")
        loader = _make_loader(self._loader_cfg, enrich_override=enrich_override)

        # Ensure namespace and domain exist
        if namespace_id not in self._domain_map:
            self._store.register_namespace(namespace_id)
            self._domain_map[namespace_id] = {}
        domain_id = self._get_or_register_domain(namespace_id, domain_name)

        # Ingest
        chunks = _ingest_source(src, loader)
        if not chunks:
            return None

        texts = [c.content for c in chunks]
        emb = _embed_texts(texts, self._embed_model, self._embed_cfg.batch_size)
        self._store.add_document(chunks, emb, namespace=namespace_id, domain_id=domain_id)
        self._store.vector.rebuild_fts_index()
        self._invalidate_search()

        if rebuild:
            from .lifecycle import build_namespace_async

            return build_namespace_async(
                namespace_id,
                self._store.vector._conn.execute("PRAGMA database_list").fetchone()[2],
                self._embed_model,
                on_progress=on_progress,
                on_complete=on_complete,
                on_error=on_error,
                force=True,
                run_ner=self._index_cfg.ner,
                run_community=self._index_cfg.community,
                spacy_model=self._index_cfg.spacy_model,
                community_alpha=self._index_cfg.community_alpha,
                community_sim_threshold=self._index_cfg.community_sim_threshold,
            )
        return None

    def remove_source(
        self,
        namespace_id: str,
        domain_name: str,
        *,
        rebuild: bool = True,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_complete: Callable[[int], None] | None = None,
        on_error: Callable[[str, Exception], None] | None = None,
    ) -> IndexHandle | None:
        """Remove all chunks for a domain and optionally rebuild secondary indexes.

        Args:
            namespace_id: Namespace containing the domain.
            domain_name: Fully-qualified domain name to remove.
            rebuild: If True, trigger async secondary-index rebuild.

        Returns:
            :class:`IndexHandle` if rebuild=True and namespace has remaining chunks, else None.
        """
        self.remove_domain(namespace_id, domain_name)

        if rebuild and self._domain_map.get(namespace_id):
            from .lifecycle import build_namespace_async

            return build_namespace_async(
                namespace_id,
                self._store.vector._conn.execute("PRAGMA database_list").fetchone()[2],
                self._embed_model,
                on_progress=on_progress,
                on_complete=on_complete,
                on_error=on_error,
                force=True,
                run_ner=self._index_cfg.ner,
                run_community=self._index_cfg.community,
                spacy_model=self._index_cfg.spacy_model,
                community_alpha=self._index_cfg.community_alpha,
                community_sim_threshold=self._index_cfg.community_sim_threshold,
            )
        return None

    def rebuild(
        self,
        namespace_id: str | None = None,
        *,
        async_: bool = True,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_complete: Callable[[int], None] | None = None,
        on_error: Callable[[str, Exception], None] | None = None,
    ) -> list[IndexHandle]:
        """Trigger secondary-index rebuild for one or all namespaces.

        Args:
            namespace_id: If set, rebuild only this namespace. Otherwise rebuild all.
            async_: If True (default), run in background threads. If False, block.
            on_progress: ``(phase, done, total) -> None`` per-phase callback.
            on_complete: ``(total_chunks) -> None`` on success.
            on_error: ``(phase, exc) -> None`` on failure.

        Returns:
            List of :class:`IndexHandle` objects (one per namespace rebuilt).

        Raises:
            NotImplementedError: the store is backed by Qdrant, Pinecone, or
                Weaviate, whose chunk content lives outside a rebuildable local
                index.
        """
        from .lifecycle import build_namespace_async

        db_path, dsn = self._rebuild_target()
        if dsn is not None and self._index_cfg.community:
            # build_community() reads a DuckDB file directly, so it cannot run
            # against Postgres. Raise here rather than inside a worker thread,
            # where the default no-op on_error would swallow it.
            raise NotImplementedError(
                "rebuild() cannot build the community index on the PostgreSQL "
                "backend: build_community() reads a DuckDB file directly. Set "
                "community=False in the index config to rebuild the rest."
            )
        namespaces = [namespace_id] if namespace_id else list(self._domain_map.keys())
        handles = []
        for ns in namespaces:
            h = build_namespace_async(
                ns,
                db_path,
                self._embed_model,
                dsn=dsn,
                on_progress=on_progress,
                on_complete=on_complete,
                on_error=on_error,
                force=True,
                run_ner=self._index_cfg.ner,
                run_community=self._index_cfg.community,
                spacy_model=self._index_cfg.spacy_model,
                community_alpha=self._index_cfg.community_alpha,
                community_sim_threshold=self._index_cfg.community_sim_threshold,
            )
            handles.append(h)
        if not async_:
            for h in handles:
                h.join()
                # A synchronous rebuild that swallowed its failure would report
                # success against an unbuilt index.
                h.raise_if_failed()
            self._invalidate_search()
        return handles

    def _rebuild_target(self) -> tuple[str, str | None]:
        """Return ``(db_path, dsn)`` identifying what build_namespace_async rebuilds.

        Exactly one is meaningful: DuckDB rebuilds from a file, PostgreSQL from a
        DSN. Reading the path with ``PRAGMA database_list`` would send DuckDB-only
        syntax to whatever backend is configured, so the backend is identified
        first.
        """
        vector = self._store.vector
        dsn = getattr(vector, "_dsn", None)
        if dsn is not None:
            return "", dsn
        db = getattr(self._store, "_db", None)
        if db is None:
            raise NotImplementedError(
                f"rebuild() is not supported for {type(vector).__name__}: chunk "
                f"content lives in an external service, not a local index this "
                f"process can re-derive. Re-ingest the sources instead."
            )
        return db._db_path, None

    # -- search ----------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        mode: str | None = None,
        namespaces: list[str] | None = None,
        domains: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> list[ScoredChunk]:
        """Search the index.

        Args:
            query: Natural-language query string.
            k: Number of results (overrides config default).
            mode: ``"vector_first"`` | ``"graph_first"`` | ``"global"``.
            namespaces: Restrict to these namespace IDs. None = all.
            domains: Restrict to these fully-qualified domain names. None = all.
                     Combined with ``namespaces``: matches names within those namespaces only.
            **kwargs: Forwarded to :meth:`EnhancedSearch.search`.
        """
        es = self._get_search()
        call_k = k if k is not None else self._search_defaults.get("k", 10)
        call_mode = mode if mode is not None else self._search_defaults.get("mode", "vector_first")
        domain_ids = self._resolve_domains(namespaces, domains)
        kwargs.setdefault("namespace_filter_llm_fn", self.namespace_filter_llm_fn)
        kwargs.setdefault("domain_filter_llm_fn", self.domain_filter_llm_fn)
        return es.search(
            query_text=query,
            k=call_k,
            mode=call_mode,
            domain_ids=domain_ids,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# build() helpers
# ---------------------------------------------------------------------------


def _build_ingest_phase(
    chonk_cfg: ChonkConfig,
    db_path: Path,
    embedding_dim: int,
    force: bool,
) -> Store:
    """Phase: ingest — load sources, embed, write FTS. Returns the open Store."""
    if db_path.exists() and force:
        db_path.unlink()
        print(f"Removed {db_path}")

    store = Store(db_path, embedding_dim=embedding_dim)

    if store.count() == 0 or force:
        default_loader = _make_loader(chonk_cfg.loader)

        # Register namespaces and their domain dictionaries
        ns_domains: dict[str, dict[str, str]] = {}  # ns → {fq_name: description}
        for ns_id, ns_cfg_raw in chonk_cfg.namespaces.items():
            ns_cfg: dict[str, Any] = ns_cfg_raw if isinstance(ns_cfg_raw, dict) else {}
            store.register_namespace(ns_id, description=ns_cfg.get("description"))
            ns_domains[ns_id] = ns_cfg.get("domains") or {}

        seen_domain: dict[str, str] = {}  # "{ns}:{fq_name}" → domain_id

        def _get_domain_id(namespace_id: str, domain_name: str) -> str:
            key = f"{namespace_id}:{domain_name}"
            if key not in seen_domain:
                desc = ns_domains.get(namespace_id, {}).get(domain_name)
                did = _stable_id(namespace_id, domain_name)
                store.register_domain(did, namespace_id, domain_name, description=desc)
                seen_domain[key] = did
            return seen_domain[key]

        # (chunks, namespace_id, domain_id)
        source_chunks: list[tuple[list[DocumentChunk], str | None, str | None]] = []

        for src in chonk_cfg.sources:
            src_d: dict[str, Any] = dict(src)
            stype: str = str(src_d.get("type") or "")
            fn = _INGEST_FNS.get(stype)
            if fn is None:
                raise ValueError(f"Unknown source type: {stype!r}")
            name: str = str(src_d.get("name") or stype)
            namespace_id = src_d.get("namespace") or None
            domain_name: str = str(src_d.get("domain") or name)
            domain_id: str | None = None

            if namespace_id:
                ns_id_str: str = str(namespace_id)
                if ns_id_str not in ns_domains:
                    store.register_namespace(ns_id_str)
                    ns_domains[ns_id_str] = {}
                domain_id = _get_domain_id(ns_id_str, domain_name)

            enrich_override = src_d.get("enrich_context")
            loader = (
                _make_loader(chonk_cfg.loader, enrich_override=enrich_override)
                if enrich_override is not None
                else default_loader
            )
            label = f"{name!r}" + (f" → {namespace_id}/{domain_name}" if namespace_id else "")
            print(f"Ingesting {label}...")
            chunks = fn(loader, src_d)
            source_chunks.append((chunks, namespace_id, domain_id))
            print(f"  {len(chunks):,} chunks")

        all_chunks = [c for chunks, _, _ in source_chunks for c in chunks]
        print(f"Total: {len(all_chunks):,} chunks — embedding...")
        emb = _embed_chunks(all_chunks, chonk_cfg.embed)

        offset = 0
        for chunks, namespace_id, domain_id in source_chunks:
            n = len(chunks)
            store.add_document(
                chunks, emb[offset : offset + n], namespace=namespace_id, domain_id=domain_id
            )
            offset += n

        print("Building FTS index...")
        store.vector.rebuild_fts_index()
        print(f"Ingest complete: {store.count():,} chunks → {db_path}")
    else:
        print(f"Existing store: {store.count():,} chunks at {db_path}")

    return store


def _build_svo_phase(
    store: Store,
    ic: IndexConfig,
    force: bool,
) -> None:
    """Phase: SVO — build subject-verb-object triple graph via LLM."""
    _row = store.vector._conn.execute("SELECT COUNT(*) FROM svo_triples").fetchone()
    if _row is None:
        raise RuntimeError("COUNT(*) returned no rows")
    existing = _row[0]
    if existing == 0 or force:
        print(f"Building SVO graph via {ic.svo_model!r} (calls LLM API)...")
        from .graph import EntityGraphPipeline, SVOExtractor

        llm_client = ic.svo_llm_client
        if llm_client is None:
            try:
                from openai import OpenAI as _OpenAI

                class _OpenAILLMClient:
                    def __init__(self, model: str) -> None:
                        self._client = _OpenAI()
                        self._model = model

                    def complete(self, prompt: str) -> str:
                        resp = self._client.chat.completions.create(
                            model=self._model,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        return resp.choices[0].message.content or ""

                llm_client = _OpenAILLMClient(ic.svo_model)
            except ImportError as exc:
                raise RuntimeError(
                    "index.svo=true requires openai installed or an LLMClient "
                    "passed as index.svo_llm_client in the config dict."
                ) from exc

        extractor = SVOExtractor(llm_client)
        pipeline = EntityGraphPipeline(extractor)
        stats = pipeline.build(store, force=force)
        print(f"  {stats.triples_written} triples.")
    else:
        print(f"SVO index: {existing:,} triples (skipped)")


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


def build(config: str | Path | dict[str, Any], *, force: bool = False) -> Index:
    """Build a fully-indexed store from a YAML config and return an :class:`Index`.

    Phases: ingest → embed → FTS → NER → community → SVO (opt-in).
    Each phase is skipped if already built (idempotent). Use ``force=True`` to rebuild.

    Config schema:

    .. code-block:: yaml

        store:
          path: my.duckdb
          embedding_dim: 1024          # default 1024

        loader:
          min_chunk_size: 1100
          max_chunk_size: 2200
          enrich_context: true

        embed:
          model: BAAI/bge-large-en-v1.5
          batch_size: 256

        index:
          ner: true
          community: true
          svo: false                   # requires LLM API access
          spacy_model: en_core_web_sm
          svo_model: gpt-4o-mini
          community_alpha: 0.2
          community_sim_threshold: 0.6

        search:
          k: 10
          mode: vector_first
          entity_ref_expansion: true
          lane_entity_min_sim: 0.60

        namespaces:
          global:
            description: "Shared company knowledge"
            domains:
              sales: "Sales data"
              sales/north-america: "North America sales"
              legal: "Legal and compliance"
          user:alice:
            domains:
              my-notes: "Alice's notes"

        sources:
          - name: na-sales
            type: glob
            path: ./data/na
            pattern: "*.pdf"
            namespace: global
            domain: sales/north-america  # fully-qualified; defaults to source name
    """
    if not isinstance(config, dict):
        import yaml

        raw: dict[str, Any] = yaml.safe_load(Path(config).read_text())
    else:
        raw = config

    chonk_cfg = ChonkConfig.from_dict(raw)

    sc: dict[str, Any] = dict(chonk_cfg.store)
    db_path = Path(str(sc["path"]))
    embedding_dim = int(sc.get("embedding_dim", 1024))

    search_defaults: dict[str, Any] = {
        "entity_ref_expansion": True,
        "lane_entity_min_sim": 0.60,
    }
    search_defaults.update({k: v for k, v in chonk_cfg.search.items()})

    # ── Phase: ingest ────────────────────────────────────────────────────────
    store = _build_ingest_phase(chonk_cfg, db_path, embedding_dim, force)

    # ── Phase: NER ───────────────────────────────────────────────────────────
    if chonk_cfg.index.ner:
        _row = store.vector._conn.execute("SELECT COUNT(*) FROM chunk_entities").fetchone()
        if _row is None:
            raise RuntimeError("COUNT(*) returned no rows")
        existing = _row[0]
        if existing == 0 or force:
            print("Building NER index...")
            from .ner import build_ner

            build_ner(store, spacy_model=chonk_cfg.index.spacy_model)
            print("  NER done.")
        else:
            print(f"NER index: {existing:,} associations (skipped)")

    # ── Phase: community ─────────────────────────────────────────────────────
    if chonk_cfg.index.community:
        _comm_row = store.vector._conn.execute("SELECT COUNT(*) FROM chunk_communities").fetchone()
        if _comm_row is None:
            raise RuntimeError("COUNT(*) returned no rows")
        existing = _comm_row[0]
        if existing == 0 or force:
            print("Building community index...")
            from .community import build_community

            n = build_community(
                db_path,
                chonk_cfg.embed.model,
                alpha=chonk_cfg.index.community_alpha,
                sim_threshold=chonk_cfg.index.community_sim_threshold,
                force=force,
            )
            print(f"  {n} communities.")
        else:
            print(f"Community index: {existing:,} assignments (skipped)")

    # ── Phase: SVO ───────────────────────────────────────────────────────────
    if chonk_cfg.index.svo:
        _build_svo_phase(store, chonk_cfg.index, force)

    return Index(
        store,
        chonk_cfg.embed.model,
        search_defaults,
        chonk_cfg.index,
        chonk_cfg.loader,
        chonk_cfg.embed,
    )


# ---------------------------------------------------------------------------
# Horizontal scale: worker / coordinator (implemented in chonk._ingest_worker)
# ---------------------------------------------------------------------------
# Re-exported here for API stability: `from chonk.ingest import run_worker` etc.
# still resolve. The implementation lives in a sibling module to keep this file
# under the 1000-line limit. `X as X` marks these as intentional re-exports.
from ._ingest_worker import (  # noqa: E402  # re-export after Index/build defs
    _pg_connect as _pg_connect,
)
from ._ingest_worker import (  # noqa: E402
    _process_queue_job as _process_queue_job,
)
from ._ingest_worker import (  # noqa: E402
    _queue_pending_count as _queue_pending_count,
)
from ._ingest_worker import (  # noqa: E402
    _queue_processing_count as _queue_processing_count,
)
from ._ingest_worker import (  # noqa: E402
    _requeue_stale_leases as _requeue_stale_leases,
)
from ._ingest_worker import (  # noqa: E402
    _run_graph_build as _run_graph_build,
)
from ._ingest_worker import (  # noqa: E402
    _set_control as _set_control,
)
from ._ingest_worker import (  # noqa: E402
    run_coordinator as run_coordinator,
)
from ._ingest_worker import (  # noqa: E402
    run_worker as run_worker,
)
