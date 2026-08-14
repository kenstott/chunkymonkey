# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 213e7f8a-5158-4869-8346-d12e39dc5fb1
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Enhanced semantic similarity search with 4-dimensional cohort assembly.

Interface is identical to standard search::

    results: list[ScoredChunk] = search.search(query_embedding, k=5)

Internally assembles the cohort across four dimensions:
  1. Seed     — vector similarity (FAISS / DuckDB VSS)
  2. Structural — next/prev/parent chunk expansion (via chunk_index adjacency)
  3. Entity   — entity-adjacent chunks from EntityIndex
  4. Cluster  — cluster-neighbour chunks (budget-limited)

Each dimension can be independently enabled or disabled via constructor args,
supporting incremental ablation benchmarking.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from ..models import DocumentChunk, ScoredChunk
from ..storage._schema import SYNTHETIC_CHUNK_TYPES
from ._enhanced_graph import _GraphMixin
from ._enhanced_scoring import _ScoringMixin
from ._enhanced_support import (
    RetrievalTrace,
    _load_community_index_from_store,
    _load_entity_index_from_store,
    _load_relationship_index_from_store,
    _tally_provenance,
)

if TYPE_CHECKING:
    from ..cluster._map import ClusterMap
    from ..community._index import CommunityIndex
    from ..generation._answer import Answer
    from ..graph._index import RelationshipIndex
    from ..ner._index import EntityIndex
    from ..storage._store import Store

_log = logging.getLogger(__name__)


class EnhancedSearch(_GraphMixin, _ScoringMixin):
    """4-dimensional cohort assembler.

    Args:
        store: A chonk Store (DuckDBVectorBackend under the hood).
        entity_index: Optional populated EntityIndex for entity expansion.
        cluster_map: Optional populated ClusterMap for cluster expansion.
        seed_pool_multiplier: Seed pool = k * multiplier (default 3).
        entity_expansion_top_n: Max chunks per entity in entity expansion (default 3).
        cluster_budget: Max cluster-adjacent candidates (default 2 * k).
        lambda_diversity: MMR redundancy penalty weight (default 0.3).
        relevance_weight: Composite score weight for relevance (default 0.5).
        priority_weight: Composite score weight for source priority (default 0.2).
        coverage_weight: Composite score weight for marginal coverage (default 0.3).
        structural_expansion: Enable next/prev/parent expansion (default True).
        entity_expansion: Enable entity adjacency expansion (default True).
        cluster_expansion: Enable cluster adjacency expansion (default True).
        redaction_filter: Optional ``(Answer) -> Answer`` applied after generation in
            ``ask()``. Use to sanitize generated answer text before it leaves the
            perimeter. Not applied when ``search()`` is called directly.
        chunk_filter: Optional ``(list[ScoredChunk]) -> list[ScoredChunk]`` applied at
            the end of every ``search()`` call. Use to redact or drop sensitive chunks
            before they are returned to the caller. **Not applied** when ``search()``
            is called internally by ``ask()`` — use ``redaction_filter`` for that path.

            Example — drop chunks from a restricted source::

                def drop_restricted(chunks):
                    return [c for c in chunks if "restricted" not in c.chunk.source]

                search = EnhancedSearch(store, chunk_filter=drop_restricted)
    """

    # Source priority constants (from spec)
    _PRIORITY = {
        "seed": 1.0,
        "structural": 0.9,
        "entity_adjacent": 0.7,
        "cluster_adjacent": 0.5,
        "context_graph_expansion": 0.6,
    }

    def __init__(
        self,
        store: Store,
        entity_index: EntityIndex | None = None,
        cluster_map: ClusterMap | None = None,
        relationship_index: RelationshipIndex | None = None,
        seed_pool_multiplier: int = 3,
        entity_expansion_top_n: int = 3,
        cluster_budget: int | None = None,
        lambda_diversity: float = 0.3,
        relevance_weight: float = 0.5,
        priority_weight: float = 0.2,
        coverage_weight: float = 0.3,
        structural_expansion: bool = True,
        entity_expansion: bool = True,
        cluster_expansion: bool = True,
        entity_embedding_expansion: bool = False,
        entity_embeddings: np.ndarray | None = None,
        entity_embedding_ids: list[str] | None = None,
        ner_fn: Callable[[str], list[str]] | None = None,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        entity_embedding_top_k: int = 10,
        entity_ref_expansion: bool = False,
        entity_ref_expansion_k: int = 20,
        entity_ref_expansion_per_k: int | None = None,
        entity_ref_expansion_min_sim: float | None = None,
        query_ner_fn: Callable[[str], list[str]] | None = None,
        query_entity_id_fn: Callable[[str], list[str]] | None = None,
        lane_entity_min_sim: float | None = None,
        session_fingerprint: str | None = None,
        community_index: CommunityIndex | None = None,
        context_graph_expansion: bool = False,
        context_graph_min_weight: float = 0.1,
        context_graph_top_k: int = 5,
        redaction_filter: Callable[[Answer], Answer] | None = None,
        chunk_filter: Callable[[list[ScoredChunk]], list[ScoredChunk]] | None = None,
    ) -> None:
        self._store = store
        conn = store.vector._conn
        if entity_index is None:
            entity_index = _load_entity_index_from_store(conn)
        self._entity_index = entity_index
        self._cluster_map = cluster_map
        if relationship_index is None:
            relationship_index = _load_relationship_index_from_store(conn)
        self._relationship_index = relationship_index
        if community_index is None:
            community_index = _load_community_index_from_store(store)
        self._community_index = community_index
        self._chunk_cache: dict[str, DocumentChunk] | None = None
        self._embedding_cache: dict[str, list[float]] | None = None
        self._seed_multiplier = seed_pool_multiplier
        self._entity_top_n = entity_expansion_top_n
        self._cluster_budget = cluster_budget  # resolved to 2*k at call time if None
        self._lambda = lambda_diversity
        self._rw = relevance_weight
        self._pw = priority_weight
        self._cw = coverage_weight
        self._structural = structural_expansion
        self._entity = entity_expansion
        self._cluster = cluster_expansion
        self._entity_embed = entity_embedding_expansion
        self._entity_embeddings = entity_embeddings
        self._entity_embedding_ids = entity_embedding_ids
        self._ner_fn = ner_fn
        self._embed_fn = embed_fn
        self._entity_embed_top_k = entity_embedding_top_k
        self._entity_ref_expansion = entity_ref_expansion
        self._entity_ref_expansion_k = entity_ref_expansion_k
        self._entity_ref_expansion_per_k = entity_ref_expansion_per_k
        self._entity_ref_expansion_min_sim = entity_ref_expansion_min_sim
        self._query_ner_fn = query_ner_fn
        self._query_entity_id_fn = query_entity_id_fn
        self._lane_entity_min_sim = lane_entity_min_sim
        self._session_fingerprint = session_fingerprint
        self._context_graph_expansion = context_graph_expansion
        self._context_graph_min_weight = context_graph_min_weight
        self._context_graph_top_k = context_graph_top_k
        self._redaction_filter = redaction_filter
        self._chunk_filter = chunk_filter
        self.last_expansion_stats: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Namespace / domain pre-filter
    # ------------------------------------------------------------------

    _NAMESPACE_FILTER_PROMPT = (
        "You are a retrieval routing assistant. Below are the available knowledge namespaces "
        "and their descriptions. Select the namespaces most likely to contain evidence for "
        'the query. Return ONLY a JSON array of namespace IDs, e.g. ["cyber", "financial"]. '
        "Include all namespaces that may be relevant; omit those that are clearly unrelated.\n\n"
        "Namespaces:\n{namespace_list}\n\n"
        "Query: {query}\n\n"
        "Return ONLY a JSON array of namespace IDs."
    )

    _DOMAIN_FILTER_PROMPT = (
        "You are a retrieval routing assistant. Below are the available knowledge domains "
        "and their descriptions. Select the domains most likely to contain evidence for "
        "the query. Return ONLY a JSON array of domain names exactly as listed, "
        'e.g. ["sales/north-america", "finance/q1"]. '
        "Include all domains that may be relevant; omit those that are clearly unrelated.\n\n"
        "Domains:\n{domain_list}\n\n"
        "Query: {query}\n\n"
        "Return ONLY a JSON array of domain names exactly as listed."
    )

    def _select_namespaces(
        self,
        query: str,
        llm_fn: Callable[[str], str],
    ) -> list[str] | None:
        """Call llm_fn to select relevant namespaces for query.

        Fetches (namespace_id, description) rows from the store. If fewer than
        two namespaces have descriptions, returns None (no filtering). Otherwise
        calls llm_fn with a routing prompt and parses the JSON array response.
        Falls back to None on any parse error so search degrades gracefully.
        """
        import json as _json

        rows = self._store.vector._conn.execute(
            "SELECT namespace_id, description FROM all_namespaces "
            "WHERE description IS NOT NULL AND description != '' "
            "ORDER BY namespace_id"
        ).fetchall()

        if len(rows) < 2:
            return None

        ns_lines = "\n".join(f"- {ns_id}: {desc}" for ns_id, desc in rows)
        prompt = self._NAMESPACE_FILTER_PROMPT.format(
            namespace_list=ns_lines,
            query=query,
        )
        try:
            raw = llm_fn(prompt)
            # Extract JSON array — strip markdown fences if present
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not match:
                return None
            selected: list[str] = _json.loads(match.group())
            known = {ns_id for ns_id, _ in rows}
            return [ns for ns in selected if ns in known] or None
        except _json.JSONDecodeError:
            return None  # LLM returned unparseable JSON — no namespace filtering (expected)

    def _select_domains(
        self,
        query: str,
        llm_fn: Callable[[str], str],
        namespaces: list[str] | None = None,
    ) -> list[str] | None:
        """Call llm_fn to select relevant domain_ids for query.

        Fetches domains with descriptions from the store (restricted to
        *namespaces* when supplied). Returns None when fewer than two domains
        have descriptions, or on any parse/LLM error.
        """
        import json as _json

        where = "WHERE d.description IS NOT NULL AND d.description != ''"
        params: list[str] = []
        if namespaces:
            placeholders = ", ".join("?" * len(namespaces))
            where += f" AND d.namespace_id IN ({placeholders})"
            params.extend(namespaces)

        rows = self._store.vector._conn.execute(
            f"SELECT d.domain_id, d.namespace_id, d.name, d.description "
            f"FROM domains d {where} ORDER BY d.namespace_id, d.name",
            params,
        ).fetchall()

        if len(rows) < 2:
            return None

        # name_to_id for resolving LLM output back to domain_ids
        name_to_id = {name: domain_id for domain_id, _ns, name, _desc in rows}
        domain_lines = "\n".join(f"- {name} ({ns}): {desc}" for _did, ns, name, desc in rows)
        prompt = self._DOMAIN_FILTER_PROMPT.format(
            domain_list=domain_lines,
            query=query,
        )
        try:
            raw = llm_fn(prompt)
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not match:
                return None
            selected_names: list[str] = _json.loads(match.group())
            ids = [name_to_id[n] for n in selected_names if n in name_to_id]
            return ids or None
        except _json.JSONDecodeError:
            return None  # LLM returned unparseable JSON — no domain filtering (expected)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @overload
    def search(
        self,
        query_embedding: np.ndarray | None = ...,
        k: int = ...,
        query_text: str | None = ...,
        query_entities: list[str] | None = ...,
        precomputed_entity_vecs: dict[str, np.ndarray] | None = ...,
        mode: str = ...,
        namespaces: list[str] | None = ...,
        domain_ids: list[str] | None = ...,
        namespace_filter_llm_fn: Callable[[str], str] | None = ...,
        domain_filter_llm_fn: Callable[[str], str] | None = ...,
        return_trace: Literal[False] = ...,
    ) -> list[ScoredChunk]: ...

    @overload
    def search(
        self,
        query_embedding: np.ndarray | None = ...,
        k: int = ...,
        query_text: str | None = ...,
        query_entities: list[str] | None = ...,
        precomputed_entity_vecs: dict[str, np.ndarray] | None = ...,
        mode: str = ...,
        namespaces: list[str] | None = ...,
        domain_ids: list[str] | None = ...,
        namespace_filter_llm_fn: Callable[[str], str] | None = ...,
        domain_filter_llm_fn: Callable[[str], str] | None = ...,
        *,
        return_trace: Literal[True],
    ) -> tuple[list[ScoredChunk], RetrievalTrace]: ...

    def search(
        self,
        query_embedding: np.ndarray | None = None,
        k: int = 30,
        query_text: str | None = None,
        query_entities: list[str] | None = None,
        precomputed_entity_vecs: dict[str, np.ndarray] | None = None,
        mode: str = "vector_first",
        namespaces: list[str] | None = None,
        domain_ids: list[str] | None = None,
        namespace_filter_llm_fn: Callable[[str], str] | None = None,
        domain_filter_llm_fn: Callable[[str], str] | None = None,
        return_trace: bool = False,
        _bypass_chunk_filter: bool = False,
    ) -> list[ScoredChunk] | tuple[list[ScoredChunk], RetrievalTrace]:
        """Assemble a top-k cohort using all enabled expansion dimensions.

        Args:
            query_embedding: np.ndarray shape (dim,) or (1, dim). If None,
                auto-generated from *query_text* using the ``embed_fn`` passed
                to the constructor (raises ValueError if neither is available).
            k: Target cohort size.
            query_text: Optional query text for BM25 hybrid seed search.
            mode: Retrieval mode — "vector_first" (default), "graph_first", or "global".
                  "graph_first": drives on RelationshipIndex traversal, vector reranks.
                  "global": searches community_summary chunks only.
            namespace_filter_llm_fn: Optional ``(prompt: str) -> str`` callable. When
                supplied and *namespaces* is None, fetches namespace descriptions from
                the store, calls the LLM to select relevant namespaces for *query_text*,
                and restricts the search to those namespaces.
            return_trace: If True, return ``(results, RetrievalTrace)`` instead of just results.

        Returns:
            Ranked list of up to k ScoredChunk objects.

        ## Agent guidance — framing queries for best retrieval

        **Issue one atomic sub-query per call.** Compound questions ("what is X and
        how does it relate to Y?") split the embedding across two intents and dilute
        both. Decompose before calling; recombine in reasoning.

        **Name the entity explicitly.** "What was Amazon's Total Net Sales for FY2025?"
        retrieves far better than "What were the e-commerce company's sales?" Named
        entities anchor the BM25 lane and entity-graph expansion. If the entity name
        is unknown, use search() to resolve it first, then re-query with the resolved
        name.

        **State the answer type in the query.** "What *date* did...", "What *dollar
        amount*...", "Which *CVE ID*..." biases the embedding toward passages that
        contain that answer type, not just passages that discuss the topic.

        **Prefer declarative over interrogative form for lookup queries.** Embedding
        "Amazon Total Net Sales FY2025 revenue" often outperforms "What was Amazon's
        revenue?" for precise fact retrieval because it matches the register of the
        source document.

        **Scope the namespace when you know the source type.** Pass
        ``namespaces=["financial"]`` for a 10-K question, ``namespaces=["cyber"]``
        for a CVE question. This eliminates off-domain noise and is the single
        highest-leverage accuracy lever for cross-domain corpora. Use
        ``namespace_filter_llm_fn`` to automate scoping when the source type is
        ambiguous.

        **Use ``return_trace=True`` to diagnose poor results.** If top chunks are
        off-topic, inspect ``RetrievalTrace.final_provenance`` to see which expansion
        dimension produced them. A high "entity_adjacent" share with low scores
        indicates the entity lane is over-expanding; tighten ``lane_entity_min_sim``
        or narrow the namespace.

        **Widen k before concluding evidence is absent.** A fact may rank outside
        the default k=5 window. Retry with k=20 before deciding the corpus does not
        contain the answer.
        """
        query_embedding = self._resolve_query_embedding(query_embedding, query_text)
        assert query_embedding is not None
        namespaces, domain_ids = self._apply_llm_filters(
            query_text, namespaces, domain_ids, namespace_filter_llm_fn, domain_filter_llm_fn
        )
        trace = RetrievalTrace(mode=mode) if return_trace else None

        if mode == "global":
            results = self._global_search(
                query_embedding, k, query_text, namespaces, domain_ids, _trace=trace
            )
            if trace is not None:
                trace.pool_size = len(results)
                trace.final_provenance = _tally_provenance(results)
            if not _bypass_chunk_filter and self._chunk_filter is not None:
                results = self._chunk_filter(results)
            if return_trace:
                assert trace is not None
                return results, trace
            return results
        if mode == "graph_first":
            results = self._graph_first_search(
                query_embedding,
                k,
                query_text,
                query_entities,
                precomputed_entity_vecs,
                namespaces,
                domain_ids,
                _trace=trace,
            )
            if trace is not None:
                trace.pool_size = len(results)
                trace.final_provenance = _tally_provenance(results)
            if not _bypass_chunk_filter and self._chunk_filter is not None:
                results = self._chunk_filter(results)
            if return_trace:
                assert trace is not None
                return results, trace
            return results
        if mode != "vector_first":
            raise ValueError(
                f"Unknown search mode {mode!r}. Use 'vector_first', 'graph_first', or 'global'."
            )

        ns_chunk_ids = self._resolve_ns_chunk_ids(namespaces, domain_ids)
        seed_limit = k * self._seed_multiplier
        cluster_budget = self._cluster_budget if self._cluster_budget is not None else 2 * k

        if trace is not None:
            trace.query_entities = list(query_entities) if query_entities else []

        # ------ Step 1: Seed -----------------------------------------------
        pool, seed_chunk_ids = self._build_seed_pool(
            query_embedding, k, query_text, namespaces, domain_ids, seed_limit
        )
        if trace is not None:
            trace.seed_chunk_ids = list(seed_chunk_ids)

        # ------ Step 2: Structural expansion --------------------------------
        _structural_added = self._expand_structural(pool, seed_chunk_ids)
        if trace is not None:
            trace.structural_chunk_ids = _structural_added

        # ------ Step 3: Entity expansion ------------------------------------
        expanded_entity_ids, _entity_added = self._expand_entity(
            pool, ns_chunk_ids, query_embedding
        )
        if trace is not None:
            trace.entity_expanded_chunk_ids = _entity_added

        # ------ Step 4: Cluster expansion (budget-limited) ------------------
        _cluster_added = self._expand_cluster(
            pool, expanded_entity_ids, ns_chunk_ids, cluster_budget
        )
        if trace is not None:
            trace.cluster_expanded_chunk_ids = _cluster_added

        # ------ Step 5: Entity embedding ANN expansion ----------------------
        _entity_embed_added = self._expand_entity_embedding(
            pool, ns_chunk_ids, query_text, query_entities, precomputed_entity_vecs, trace
        )
        if trace is not None:
            trace.entity_embed_expanded_chunk_ids = _entity_embed_added

        # ------ Step 6: Score and select top-k ------------------------------
        candidates = list(pool.values())
        results = self._select_cohort(candidates, query_embedding, k)

        # ------ Step 7: Entity-ref expansion (adaptive post-selection) ------
        self.last_expansion_stats = None
        if self._entity_ref_expansion:
            ents = query_entities
            if ents is None and self._query_ner_fn is not None and query_text:
                ents = self._query_ner_fn(query_text)
            if ents:
                results = self._entity_ref_expand(
                    results,
                    pool,
                    query_embedding,
                    ents,
                    k,
                    query_text,
                    precomputed_entity_vecs,
                    namespaces,
                    domain_ids,
                    _trace=trace,
                )

        if trace is not None:
            trace.pool_size = len(pool)
            trace.final_provenance = _tally_provenance(results)

        if not _bypass_chunk_filter and self._chunk_filter is not None:
            results = self._chunk_filter(results)

        if return_trace:
            assert trace is not None
            return results, trace
        return results

    # ------------------------------------------------------------------
    # search() step helpers (vector_first path)
    # ------------------------------------------------------------------

    def _resolve_query_embedding(
        self,
        query_embedding: np.ndarray | None,
        query_text: str | None,
    ) -> np.ndarray:
        """Return a valid embedding, auto-generating from text when embedding is None."""
        if query_embedding is not None:
            return query_embedding
        if query_text is None:
            raise ValueError("Either query_embedding or query_text must be supplied.")
        if self._embed_fn is None:
            raise ValueError(
                "query_embedding is None and no embed_fn was set on EnhancedSearch. "
                "Pass embed_fn to the constructor or supply query_embedding directly."
            )
        return self._embed_fn([query_text])[0]

    def _apply_llm_filters(
        self,
        query_text: str | None,
        namespaces: list[str] | None,
        domain_ids: list[str] | None,
        namespace_filter_llm_fn: Callable[[str], str] | None,
        domain_filter_llm_fn: Callable[[str], str] | None,
    ) -> tuple[list[str] | None, list[str] | None]:
        """Apply namespace/domain LLM pre-filters when callables are supplied."""
        if namespace_filter_llm_fn is not None and namespaces is None and query_text:
            namespaces = self._select_namespaces(query_text, namespace_filter_llm_fn)
        if domain_filter_llm_fn is not None and domain_ids is None and query_text:
            domain_ids = self._select_domains(query_text, domain_filter_llm_fn, namespaces)
        return namespaces, domain_ids

    def _resolve_ns_chunk_ids(
        self,
        namespaces: list[str] | None,
        domain_ids: list[str] | None,
    ) -> set[str] | None:
        """Build the allowlist of chunk IDs from namespace and domain filters."""
        ns_chunk_ids: set[str] | None = None
        if namespaces:
            placeholders = ", ".join("?" * len(namespaces))
            rows = self._store.vector._conn.execute(
                f"SELECT chunk_id FROM embeddings WHERE namespace IN ({placeholders})",
                list(namespaces),
            ).fetchall()
            ns_chunk_ids = {r[0] for r in rows}
        if domain_ids:
            placeholders = ", ".join("?" * len(domain_ids))
            rows = self._store.vector._conn.execute(
                f"SELECT chunk_id FROM embeddings WHERE domain_id IN ({placeholders})",
                list(domain_ids),
            ).fetchall()
            di_chunk_ids = {r[0] for r in rows}
            ns_chunk_ids = di_chunk_ids if ns_chunk_ids is None else ns_chunk_ids & di_chunk_ids
        return ns_chunk_ids

    def _build_seed_pool(
        self,
        query_embedding: np.ndarray,
        k: int,
        query_text: str | None,
        namespaces: list[str] | None,
        domain_ids: list[str] | None,
        seed_limit: int,
    ) -> tuple[dict[str, ScoredChunk], list[str]]:
        """Step 1: vector similarity seeds. Returns (pool, seed_chunk_ids)."""
        raw_seeds = self._store.search(
            query_embedding,
            limit=seed_limit,
            query_text=query_text,
            namespaces=namespaces,
            domain_ids=domain_ids,
            exclude_chunk_types=SYNTHETIC_CHUNK_TYPES,
        )
        pool: dict[str, ScoredChunk] = {}
        seed_chunk_ids: list[str] = []
        for chunk_id, score, chunk in raw_seeds:
            if chunk_id not in pool:
                pool[chunk_id] = ScoredChunk(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=score,
                    provenance="seed",
                    embedding=None,
                )
            seed_chunk_ids.append(chunk_id)
        return pool, seed_chunk_ids

    def _expand_structural(
        self,
        pool: dict[str, ScoredChunk],
        seed_chunk_ids: list[str],
    ) -> list[str]:
        """Step 2: add next/prev neighbors to pool. Returns list of newly added IDs."""
        added: list[str] = []
        if not self._structural:
            return added
        for chunk_id, chunk in self._structural_neighbors(seed_chunk_ids):
            if chunk_id not in pool:
                pool[chunk_id] = ScoredChunk(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=0.0,
                    provenance="structural",
                )
                added.append(chunk_id)
        return added

    def _expand_entity(
        self,
        pool: dict[str, ScoredChunk],
        ns_chunk_ids: set[str] | None,
        query_embedding: np.ndarray,
    ) -> tuple[set[str], list[str]]:
        """Step 3: entity-adjacent expansion. Returns (expanded_entity_ids, added_chunk_ids)."""
        expanded_entity_ids: set[str] = set()
        added: list[str] = []
        if not self._entity or self._entity_index is None:
            return expanded_entity_ids, added
        query_vec_flat = (
            query_embedding.flatten().astype("float32")
            if self._lane_entity_min_sim is not None
            else None
        )
        for cid in list(pool.keys()):
            for entity_id, _ in self._entity_index.get_entities_for_chunk(cid):
                expanded_entity_ids.add(entity_id)
                for linked_chunk_id, _ in self._entity_index.get_chunks_for_entity(
                    entity_id, top_n=self._entity_top_n
                ):
                    if ns_chunk_ids is not None and linked_chunk_id not in ns_chunk_ids:
                        continue
                    if linked_chunk_id not in pool:
                        if query_vec_flat is not None:
                            emb = self._get_embedding(linked_chunk_id)
                            lane_min_sim = self._lane_entity_min_sim  # non-None: guard at line 618
                            assert lane_min_sim is not None
                            if (
                                emb is not None
                                and self._cosine_similarity(query_vec_flat, emb) < lane_min_sim
                            ):
                                continue
                        chunk = self._fetch_chunk(linked_chunk_id)
                        if chunk is not None:
                            pool[linked_chunk_id] = ScoredChunk(
                                chunk_id=linked_chunk_id,
                                chunk=chunk,
                                score=0.0,
                                provenance="entity_adjacent",
                                linked_by=entity_id,
                            )
                            added.append(linked_chunk_id)
        return expanded_entity_ids, added

    def _expand_cluster(
        self,
        pool: dict[str, ScoredChunk],
        expanded_entity_ids: set[str],
        ns_chunk_ids: set[str] | None,
        cluster_budget: int,
    ) -> list[str]:
        """Step 4: cluster-adjacent expansion (budget-limited). Returns added chunk IDs."""
        added: list[str] = []
        if not self._cluster or self._cluster_map is None or cluster_budget <= 0:
            return added
        cluster_count = 0
        for entity_id in expanded_entity_ids:
            if cluster_count >= cluster_budget:
                break
            for neighbor_entity_id in self._cluster_map.get_neighbors(entity_id):
                if cluster_count >= cluster_budget:
                    break
                cluster_chunks = (
                    self._entity_index.get_chunks_for_entity(neighbor_entity_id, top_n=1)
                    if self._entity_index
                    else []
                )
                for linked_chunk_id, _ in cluster_chunks:
                    if ns_chunk_ids is not None and linked_chunk_id not in ns_chunk_ids:
                        continue
                    if linked_chunk_id not in pool:
                        chunk = self._fetch_chunk(linked_chunk_id)
                        if chunk is not None:
                            cluster_id = self._cluster_map.get_cluster(neighbor_entity_id)
                            pool[linked_chunk_id] = ScoredChunk(
                                chunk_id=linked_chunk_id,
                                chunk=chunk,
                                score=0.0,
                                provenance="cluster_adjacent",
                                linked_by=neighbor_entity_id,
                                cluster=cluster_id,
                            )
                            cluster_count += 1
                            added.append(linked_chunk_id)
        return added

    def _expand_entity_embedding(
        self,
        pool: dict[str, ScoredChunk],
        ns_chunk_ids: set[str] | None,
        query_text: str | None,
        query_entities: list[str] | None,
        precomputed_entity_vecs: dict[str, np.ndarray] | None,
        trace: RetrievalTrace | None,
    ) -> list[str]:
        """Step 5: entity-embedding ANN expansion. Returns added chunk IDs."""
        added: list[str] = []
        if not (
            self._entity_embed
            and self._entity_embeddings is not None
            and self._entity_embedding_ids is not None
            and self._entity_index is not None
            and query_text
            and self._ner_fn is not None
            and (self._embed_fn is not None or precomputed_entity_vecs is not None)
        ):
            return added
        query_ents = query_entities if query_entities is not None else self._ner_fn(query_text)
        if trace is not None and query_ents and not trace.query_entities:
            trace.query_entities = list(query_ents)
        if not query_ents:
            return added
        q_ent_vecs, query_ents = self._resolve_entity_vecs(query_ents, precomputed_entity_vecs)
        if q_ent_vecs is None or not query_ents:
            return added
        scores = q_ent_vecs @ self._entity_embeddings.T
        max_scores = scores.max(axis=0)
        top_indices = np.argsort(-max_scores)[: self._entity_embed_top_k]
        for idx in top_indices:
            eid = self._entity_embedding_ids[int(idx)]
            for linked_chunk_id, _ in self._entity_index.get_chunks_for_entity(eid, top_n=2):
                if ns_chunk_ids is not None and linked_chunk_id not in ns_chunk_ids:
                    continue
                if linked_chunk_id not in pool:
                    chunk = self._fetch_chunk(linked_chunk_id)
                    if chunk is not None:
                        pool[linked_chunk_id] = ScoredChunk(
                            chunk_id=linked_chunk_id,
                            chunk=chunk,
                            score=0.0,
                            provenance="entity_adjacent",
                            linked_by=eid,
                        )
                        added.append(linked_chunk_id)
        return added

    def _resolve_entity_vecs(
        self,
        query_ents: list[str],
        precomputed_entity_vecs: dict[str, np.ndarray] | None,
    ) -> tuple[np.ndarray | None, list[str]]:
        """Return (stacked_vecs, filtered_ents) for entity embedding ANN."""
        if precomputed_entity_vecs is not None:
            available = [e for e in query_ents if e in precomputed_entity_vecs]
            vecs = [precomputed_entity_vecs[e] for e in available]
            return (np.stack(vecs) if vecs else None), available
        # Caller guarantees embed_fn is non-None when precomputed_entity_vecs is None
        assert self._embed_fn is not None
        return self._embed_fn(query_ents), query_ents

    def ask(
        self,
        query: str,
        embed_fn: Callable[[list[str]], np.ndarray],
        llm_fn: Callable[[str], str],
        k: int = 30,
        mode: str = "vector_first",
        namespaces: list[str] | None = None,
        domain_ids: list[str] | None = None,
        namespace_filter_llm_fn: Callable[[str], str] | None = None,
        domain_filter_llm_fn: Callable[[str], str] | None = None,
        token_budget: int = 4096,
        redaction_filter: Callable[[Answer], Answer] | None = None,
    ) -> Answer:
        """Embed query, search, generate and return an Answer.

        Args:
            query: Natural-language question.
            embed_fn: ``(texts: list[str]) -> np.ndarray`` — produces query embedding.
            llm_fn: ``(prompt: str) -> str`` — generates the answer.
            k: Retrieval cohort size.
            mode: Search mode passed to search().
            namespaces: Explicit namespace filter; overrides namespace_filter_llm_fn.
            domain_ids: Domain filter passed to search().
            namespace_filter_llm_fn: LLM callable for automatic namespace pre-filtering.
            token_budget: Max prompt tokens passed to AnswerGenerator.
            redaction_filter: Optional ``(Answer) -> Answer`` callable applied to the
                generated answer before it is returned. Overrides the instance-level
                filter set at construction time. Use this to sanitize sensitive data
                before the answer leaves a sovereign deployment perimeter.

                **Sovereign RAG pattern** — frontier model as planner, sovereign model
                as retriever/generator, redaction filter as the trust boundary:

                .. code-block:: python

                    import re
                    from chonk.generation import Answer

                    _SSN = re.compile(r"\\b\\d{3}-\\d{2}-\\d{4}\\b")
                    _EIN = re.compile(r"\\b\\d{2}-\\d{7}\\b")

                    def redact_pii(answer: Answer) -> Answer:
                        clean = _SSN.sub("[SSN REDACTED]", answer.text)
                        clean = _EIN.sub("[EIN REDACTED]", clean)
                        return Answer(text=clean, citations=answer.citations)

                    # Set once at construction — applies to every ask() call:
                    search = EnhancedSearch(store, redaction_filter=redact_pii)

                    # Or pass per-call to override:
                    result = search.ask(query, embed_fn, llm_fn, redaction_filter=redact_pii)

                The filter receives the fully-generated ``Answer`` (text + citations)
                and must return an ``Answer``. It may modify, replace, or raise. The
                frontier planner only ever sees the returned text — raw chunks and
                the original generated text never leave the perimeter.

        ## Agent guidance — when and how to use ask()

        **Use ask() only when the sub-query is fully formed.** ask() is a one-shot
        call: embed → retrieve → generate. There is no opportunity to inspect chunks,
        adjust scope, or retry between steps. If the agent needs to evaluate evidence
        quality or branch on what was found, use search() instead.

        **Frame the query as a complete, self-contained question.** The query string
        drives both the embedding (retrieval) and the prompt (generation). Ambiguous
        or truncated queries produce off-topic retrievals and underspecified answers.
        Include the entity name, the attribute sought, and any scoping constraint
        (fiscal year, version number, jurisdiction) in a single sentence.

        **Scope the namespace before calling.** Pass ``namespaces`` explicitly when
        the source type is known, or pass ``namespace_filter_llm_fn`` to let the
        LLM route automatically. An unscoped ask() on a heterogeneous corpus
        retrieves cross-domain noise that the generator may hallucinate from.

        **Treat Answer.citations as ground truth, not Answer.text.** The generator
        can hallucinate even with good retrieval. When the answer will be used as
        input to a subsequent reasoning step, verify the claim against
        ``Answer.citations`` before propagating it.

        **Use ask() for terminal steps, search() for intermediate steps.** A
        well-structured multi-step plan ends each branch with ask(); intermediate
        steps that feed into further reasoning use search() so the agent retains
        control over how retrieved evidence is interpreted and combined.
        """
        from ..generation import AnswerContext, AnswerGenerator

        query_embedding = embed_fn([query])[0]
        # Bypass chunk_filter — ask() applies redaction_filter on the generated Answer instead.
        chunks: list[ScoredChunk] = self.search(  # type: ignore[call-overload]
            query_embedding,
            k=k,
            query_text=query,
            mode=mode,
            namespaces=namespaces,
            domain_ids=domain_ids,
            namespace_filter_llm_fn=namespace_filter_llm_fn,
            domain_filter_llm_fn=domain_filter_llm_fn,
            _bypass_chunk_filter=True,
        )
        context = AnswerContext(chunks=chunks, query=query)
        answer = AnswerGenerator(llm_fn, token_budget=token_budget).generate(context)
        _filter = redaction_filter if redaction_filter is not None else self._redaction_filter
        if _filter is not None:
            answer = _filter(answer)
        return answer
