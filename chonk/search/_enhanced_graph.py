# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 213e7f8a-5158-4869-8346-d12e39dc5fb1
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Graph-first search, global search, graph context assembly, map-reduce,
and entity-ref expansion mixin for EnhancedSearch."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..models import DocumentChunk, ScoredChunk
from ..storage._schema import SYNTHETIC_CHUNK_TYPES
from ._enhanced_support import RetrievalTrace

if TYPE_CHECKING:
    from ..community._index import CommunityIndex
    from ..graph._index import RelationshipIndex
    from ..ner._index import EntityIndex
    from ..storage._store import Store


class _GraphMixin:
    """Mixin providing graph_first, global, graph context assembly, map-reduce,
    and entity-ref expansion methods.

    Depends on ``self._store``, ``self._relationship_index``, ``self._entity_index``,
    ``self._community_index``, ``self._entity_top_n``, ``self._seed_multiplier``,
    ``self._context_graph_expansion``, ``self._context_graph_min_weight``,
    ``self._context_graph_top_k``, ``self._entity_ref_expansion_k``,
    ``self._entity_ref_expansion_per_k``, ``self._entity_ref_expansion_min_sim``,
    ``self._session_fingerprint``, ``self._query_entity_id_fn``,
    ``self._embed_fn``, ``self._MAP_PROMPT``,
    and the ``_select_cohort``, ``_fetch_chunk``, ``_resolve_ns_chunk_ids``
    methods supplied by other mixins / the base class.
    """

    # Attributes/methods provided by the host EnhancedSearch and sibling mixins
    # (declared for the type checker; bare annotations create no runtime attribute).
    _store: Store
    _entity_index: EntityIndex | None
    _relationship_index: RelationshipIndex | None
    _community_index: CommunityIndex | None
    _embed_fn: Callable[[list[str]], np.ndarray] | None
    _query_entity_id_fn: Callable[[str], list[str]] | None
    _seed_multiplier: int
    _entity_top_n: int
    _session_fingerprint: str | None
    _context_graph_expansion: bool
    _context_graph_min_weight: float
    _context_graph_top_k: int
    _entity_ref_expansion_k: int
    _entity_ref_expansion_per_k: int | None
    _entity_ref_expansion_min_sim: float | None
    # Methods supplied by sibling mixins / the host EnhancedSearch. Declared as
    # TYPE_CHECKING method stubs (with self) so override checks align and the
    # concrete signatures are visible to the type checker. search() is overloaded
    # on return_trace; graph callers always use the default (return_trace=False)
    # form, hence the list[ScoredChunk] return.
    if TYPE_CHECKING:

        def search(self, *args: Any, **kwargs: Any) -> list[ScoredChunk]: ...  # noqa: ANN401
        def _resolve_ns_chunk_ids(
            self, namespaces: list[str] | None, domain_ids: list[str] | None
        ) -> set[str] | None: ...
        def _select_cohort(
            self, candidates: list[ScoredChunk], query_embedding: np.ndarray, k: int
        ) -> list[ScoredChunk]: ...
        def _fetch_chunk(self, chunk_id: str) -> DocumentChunk | None: ...

    # ------------------------------------------------------------------
    # graph_first mode
    # ------------------------------------------------------------------

    def _graph_first_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        query_text: str | None,
        query_entities: list[str] | None,
        precomputed_entity_vecs: dict[str, np.ndarray] | None,
        namespaces: list[str] | None = None,
        domain_ids: list[str] | None = None,
        _trace: RetrievalTrace | None = None,
    ) -> list[ScoredChunk]:
        """Driver: RelationshipIndex traversal. Assist: vector rerank via _select_cohort.

        Falls back to vector_first when prerequisites are absent:
          - no relationship_index
          - no query_text and no query_entities provided
          - NER produces no entity hits
        """
        fallback = (
            self._relationship_index is None
            or self._entity_index is None
            or (query_text is None and not query_entities)
        )
        if fallback:
            return self.search(
                query_embedding,
                k=k,
                query_text=query_text,
                query_entities=query_entities,
                precomputed_entity_vecs=precomputed_entity_vecs,
                mode="vector_first",
                namespaces=namespaces,
                domain_ids=domain_ids,
            )

        # Resolve query entity IDs (slugs) for graph traversal
        ents = None
        if self._query_entity_id_fn is not None and query_text:
            ents = self._query_entity_id_fn(query_text)
        if not ents:
            return self.search(
                query_embedding,
                k=k,
                query_text=query_text,
                query_entities=query_entities,
                precomputed_entity_vecs=precomputed_entity_vecs,
                mode="vector_first",
                namespaces=namespaces,
                domain_ids=domain_ids,
            )

        if _trace is not None:
            _trace.query_entities = list(ents)

        related, svo_count = self._graph_traverse_hops(ents)

        if _trace is not None:
            _trace.graph_traversal_entities = list(related)
            _trace.svo_triples_count = svo_count

        ns_chunk_ids = self._resolve_ns_chunk_ids(namespaces, domain_ids)

        # _entity_index non-None: fallback guard at top of this function checked it
        assert self._entity_index is not None

        # Build pool from related-entity chunks
        pool: dict[str, ScoredChunk] = {}
        for related_entity_id in related:
            for linked_chunk_id, _ in self._entity_index.get_chunks_for_entity(
                related_entity_id, top_n=self._entity_top_n
            ):
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
                            linked_by=related_entity_id,
                        )

        # Augment with vector seeds so reranker has enough candidates
        seed_limit = max(k * self._seed_multiplier, k - len(pool))
        for chunk_id, score, chunk in self._store.search(
            query_embedding,
            limit=seed_limit,
            query_text=query_text,
            namespaces=namespaces,
            domain_ids=domain_ids,
            exclude_chunk_types=SYNTHETIC_CHUNK_TYPES,
        ):
            if chunk_id not in pool:
                pool[chunk_id] = ScoredChunk(
                    chunk_id=chunk_id, chunk=chunk, score=score, provenance="seed"
                )

        if not pool:
            return []

        return self._select_cohort(list(pool.values()), query_embedding, k)

    def _graph_traverse_hops(self, ents: list[str]) -> tuple[set[str], int]:
        """2-hop RelationshipIndex traversal. Returns (related_entity_ids, svo_count)."""
        # Caller (_graph_first_search) guarantees non-None via fallback guard
        assert self._relationship_index is not None
        ents_set = set(ents)
        hop1: set[str] = set()
        svo_count = 0
        for entity_id in ents:
            for triple in self._relationship_index.get_objects(entity_id):
                hop1.add(triple.object_id)
                svo_count += 1
            for triple in self._relationship_index.get_subjects(entity_id):
                hop1.add(triple.subject_id)
                svo_count += 1
        hop1 -= ents_set

        hop2: set[str] = set()
        for entity_id in hop1:
            for triple in self._relationship_index.get_objects(entity_id):
                hop2.add(triple.object_id)
            for triple in self._relationship_index.get_subjects(entity_id):
                hop2.add(triple.subject_id)
        hop2 -= ents_set | hop1

        return hop1 | hop2, svo_count

    # ------------------------------------------------------------------
    # global mode
    # ------------------------------------------------------------------

    def _global_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        query_text: str | None,
        namespaces: list[str] | None = None,
        domain_ids: list[str] | None = None,
        _trace: RetrievalTrace | None = None,
    ) -> list[ScoredChunk]:
        """Driver: vector search over community_summary chunks only."""
        raw = self._store.search(
            query_embedding,
            limit=k,
            query_text=query_text,
            chunk_types=["community_summary"],
            namespaces=namespaces,
            domain_ids=domain_ids,
            session_fingerprint=self._session_fingerprint,
        )
        results: list[ScoredChunk] = []
        for chunk_id, score, chunk in raw:
            results.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=score,
                    provenance="seed",
                )
            )
        if _trace is not None:
            _trace.community_chunk_ids = [sc.chunk_id for sc in results]
        return results

    # ------------------------------------------------------------------
    # MS-GraphRAG context assembly
    # ------------------------------------------------------------------

    def assemble_graph_context(
        self,
        hits: list[Any],
        query_text: str | None = None,
        query_entities: list[str] | None = None,
        context_token_budget: int = 8000,
        namespaces: list[str] | None = None,
    ) -> str:
        """Assemble MS-GraphRAG-style structured context from retrieved chunks.

        Sections: Entities | Relationships | Community Reports | Source Text

        Each section is budget-trimmed so the total approximate token count
        (chars / 4) stays within *context_token_budget*.

        Args:
            hits: List of ``(chunk_id, score, DocumentChunk)`` tuples or ScoredChunk objects.
            query_text: Optional query text for NER-based entity resolution.
            query_entities: Pre-resolved entity IDs; used if query_text NER is absent.
            context_token_budget: Approximate token ceiling for the assembled context.
            namespaces: Restrict entity records to entities associated with a chunk
                in one of these namespaces. ``None`` (the default) applies no
                namespace restriction. Chunks themselves are already namespace-filtered
                upstream; this scopes the entity metadata joined into the context, which
                also reaches entity IDs supplied by ``query_entity_id_fn``.
        """
        chunk_pairs = self._normalise_hits(hits)
        chunk_ids = [cid for cid, _ in chunk_pairs]

        entity_ids = self._collect_entity_ids(chunk_ids, query_text)
        entity_records = self._fetch_entity_records(entity_ids, namespaces)
        rel_rows = self._collect_rel_rows(entity_ids, entity_records)
        community_texts = self._collect_community_texts(chunk_ids)

        return self._assemble_sections(
            entity_records, rel_rows, community_texts, chunk_pairs, context_token_budget
        )

    @staticmethod
    def _normalise_hits(hits: list[Any]) -> list[tuple[str, DocumentChunk]]:
        """Normalise mixed hit types to (chunk_id, DocumentChunk) pairs."""
        chunk_pairs: list[tuple[str, DocumentChunk]] = []
        for h in hits:
            if isinstance(h, ScoredChunk):
                chunk_pairs.append((h.chunk_id, h.chunk))
            else:
                chunk_pairs.append((h[0], h[2]))
        return chunk_pairs

    def _collect_entity_ids(
        self,
        chunk_ids: list[str],
        query_text: str | None,
    ) -> set[str]:
        """Collect entity IDs from retrieved chunks and query entity resolver."""
        entity_ids: set[str] = set()
        if self._entity_index is not None:
            for cid in chunk_ids:
                for eid, _ in self._entity_index.get_entities_for_chunk(cid):
                    entity_ids.add(eid)
        if self._query_entity_id_fn is not None and query_text:
            entity_ids.update(self._query_entity_id_fn(query_text))
        return entity_ids

    def _fetch_entity_records(
        self,
        entity_ids: set[str],
        namespaces: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Fetch (id, name, type, description) rows for the given entity IDs.

        When *namespaces* is given, an entity is kept only if it is associated
        with a chunk in one of those namespaces. Matches the
        ``COALESCE(namespace, 'global')`` convention used by the context graph.
        """
        if not entity_ids:
            return []
        if namespaces is not None and not namespaces:
            # Restricting to an empty namespace set matches nothing, mirroring
            # Store.search(namespaces=[]).
            return []
        conn = self._store.vector._conn
        eid_list = list(entity_ids)
        placeholders = ", ".join("?" * len(eid_list))
        ns_clause = ""
        params: list[Any] = list(eid_list)
        if namespaces is not None:
            ns_placeholders = ", ".join("?" * len(namespaces))
            ns_clause = (
                f" AND EXISTS (SELECT 1 FROM chunk_entities ce WHERE ce.entity_id = e.id "
                f"AND COALESCE(ce.namespace, 'global') IN ({ns_placeholders}))"
            )
            params.extend(namespaces)
        rows = conn.execute(
            f"SELECT e.id, e.name, e.entity_type, COALESCE(e.description, '') "
            f"FROM entities e "
            f"WHERE e.id IN ({placeholders}){ns_clause}",
            params,
        ).fetchall()
        return [{"id": r[0], "name": r[1], "type": r[2], "description": r[3]} for r in rows]

    def _collect_rel_rows(
        self,
        entity_ids: set[str],
        entity_records: list[dict[str, str]],
    ) -> list[tuple[str, str, str, str]]:
        """Collect 1-hop SVO triples for matched entities as (subj_name, verb, obj_name, desc)."""
        rel_rows: list[tuple[str, str, str, str]] = []
        if self._relationship_index is None or not entity_ids:
            return rel_rows
        id_to_name = {r["id"]: r["name"] for r in entity_records}
        seen: set[tuple[str, str, str]] = set()
        for eid in entity_ids:
            for t in self._relationship_index.get_objects(eid):
                key = (t.subject_id, t.verb, t.object_id)
                if key not in seen:
                    seen.add(key)
                    rel_rows.append(
                        (
                            id_to_name.get(t.subject_id, t.subject_id),
                            t.verb,
                            id_to_name.get(t.object_id, t.object_id),
                            t.description or "",
                        )
                    )
        return rel_rows

    def _collect_community_texts(self, chunk_ids: list[str]) -> list[str]:
        """Fetch community summary texts for the communities containing chunk_ids."""
        if self._community_index is None or not chunk_ids:
            return []
        comm_ids: set[int] = set()
        for cid in chunk_ids:
            c = self._community_index.community_id(cid)
            if c is not None:
                comm_ids.add(c)
        if not comm_ids:
            return []
        conn = self._store.vector._conn
        names = [f"community:{cid}" for cid in comm_ids]
        placeholders = ", ".join("?" * len(names))
        rows = conn.execute(
            f"SELECT content FROM embeddings "
            f"WHERE document_name IN ({placeholders}) "
            f"AND chunk_type = 'community_summary'",
            names,
        ).fetchall()
        return [r[0] for r in rows if r[0]]

    def _assemble_sections(
        self,
        entity_records: list[dict[str, str]],
        rel_rows: list[tuple[str, str, str, str]],
        community_texts: list[str],
        chunk_pairs: list[tuple[str, DocumentChunk]],
        context_token_budget: int,
    ) -> str:
        """Budget-aware assembly of Entities / Relationships / Community / Source sections."""
        budget_chars = context_token_budget * 4
        used = 0
        sections: list[str] = []

        def _add(block: str) -> bool:
            nonlocal used
            c = len(block)
            if used + c > budget_chars:
                return False
            sections.append(block)
            used += c
            return True

        if entity_records:
            header = "## Entities\n\n| Name | Type | Description |\n|------|------|-------------|"
            rows_str = "\n".join(
                f"| {r['name']} | {r['type']} | {r['description'] or '—'} |" for r in entity_records
            )
            _add(f"{header}\n{rows_str}")

        if rel_rows:
            header = "## Relationships\n\n| Subject | Relationship | Object | Description |\n|---------|-------------|--------|-------------|"  # noqa: E501
            rows_str = "\n".join(
                f"| {subj} | {verb} | {obj} | {desc or '—'} |" for subj, verb, obj, desc in rel_rows
            )
            _add(f"{header}\n{rows_str}")

        for text in community_texts:
            block = (
                f"## Community Reports\n\n{text}"
                if not any(s.startswith("## Community Reports") for s in sections)
                else text
            )
            if not _add(block):
                break

        for _, chunk in chunk_pairs:
            block = (
                f"## Source Text\n\n{chunk.content or ''}"
                if not any(s.startswith("## Source Text") for s in sections)
                else (chunk.content or "")
            )
            if not _add(block):
                break

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # MS-GraphRAG map-reduce global context
    # ------------------------------------------------------------------

    _MAP_PROMPT = (
        "You are an expert analyst. Using ONLY the community report below, "
        "provide a concise answer to the question and rate how relevant this "
        "community report is (0 = not relevant, 100 = fully answers the question).\n\n"
        'Return ONLY valid JSON: {{"answer": "<answer>", "score": <0-100>}}\n\n'
        "Question: {query}\n\n"
        "Community Report:\n{community_text}"
    )

    def map_reduce_global_context(
        self,
        hits: list[Any],
        query_text: str,
        llm_fn: Callable[[str], str],
        concurrency: int = 4,
    ) -> str:
        """MS-GraphRAG map-reduce global context assembly.

        Map: call *llm_fn* once per community summary hit with a scoring prompt.
        Reduce: filter score > 0, sort by score desc, format top answers as context.

        Args:
            hits: Community summary chunks as ``(chunk_id, score, DocumentChunk)``
                tuples or ``ScoredChunk`` objects.
            query_text: The user's question.
            llm_fn: ``(prompt: str) -> str`` — LLM text completion callable.
            concurrency: Thread-pool size for parallel map calls.

        Returns:
            Formatted context string ready for final generation.
        """
        import concurrent.futures
        import json as _json

        chunk_pairs: list[tuple[str, DocumentChunk]] = []
        for h in hits:
            if isinstance(h, ScoredChunk):
                chunk_pairs.append((h.chunk_id, h.chunk))
            else:
                chunk_pairs.append((h[0], h[2]))

        def _map_one(pair: tuple[str, DocumentChunk]) -> tuple[str, int] | None:
            _, chunk = pair
            text = (chunk.content or "").strip()
            if not text:
                return None
            prompt = self._MAP_PROMPT.format(query=query_text, community_text=text)
            raw = llm_fn(prompt)
            # Extract JSON from response (may be wrapped in ```json ... ```)
            m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            try:
                parsed = _json.loads(m.group() if m else raw)
                answer = str(parsed.get("answer", "")).strip()
                score = int(parsed.get("score", 0))
            except (_json.JSONDecodeError, ValueError):
                return (
                    None  # LLM returned unparseable/invalid JSON — skip this community (expected)
                )
            if not answer or score <= 0:
                return None
            return answer, score

        intermediate: list[tuple[str, int]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_map_one, p) for p in chunk_pairs]
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                if result is not None:
                    intermediate.append(result)

        if not intermediate:
            return ""

        intermediate.sort(key=lambda x: x[1], reverse=True)

        lines = ["## Intermediate Answers\n"]
        for answer, score in intermediate:
            lines.append(f"### Community Report (relevance: {score})\n{answer}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Entity-ref expansion
    # ------------------------------------------------------------------

    def _entity_ref_expand(
        self,
        results: list[ScoredChunk],
        existing_pool: dict[str, ScoredChunk],
        query_embedding: np.ndarray,
        query_entities: list[str],
        k: int,
        query_text: str | None,
        precomputed_entity_vecs: dict[str, np.ndarray] | None = None,
        namespaces: list[str] | None = None,
        domain_ids: list[str] | None = None,
        _trace: RetrievalTrace | None = None,
    ) -> list[ScoredChunk]:
        """Post-selection: if query entities are absent from top-k, expand via semantic search."""
        result_text = " ".join((sc.chunk.content or "") for sc in results).lower()
        missing = [e for e in query_entities if e.lower() not in result_text]

        if not missing:
            self.last_expansion_stats = {"invoked": False, "missing_entities": []}
            if _trace is not None:
                _trace.ref_expansion_missing = []
                _trace.ref_expansion_found = []
                _trace.ref_expansion_chunks_added = 0
            return results

        new_pool = dict(existing_pool)
        found_entities: list[str] = []

        if self._context_graph_expansion and self._entity_index is not None:
            self._context_graph_expand_missing(missing, new_pool, found_entities, namespaces)

        if self._embed_fn is not None or precomputed_entity_vecs is not None:
            missing = self._semantic_expand_missing(
                missing, new_pool, found_entities, precomputed_entity_vecs, namespaces, domain_ids
            )
        else:
            self._literal_expand_missing(
                missing,
                new_pool,
                found_entities,
                query_embedding,
                query_text,
                namespaces,
                domain_ids,
            )

        chunks_added = len(new_pool) - len(existing_pool)
        cg_chunks = sum(1 for sc in new_pool.values() if sc.provenance == "context_graph_expansion")
        self.last_expansion_stats = {
            "invoked": True,
            "missing_entities": missing,
            "found_entities": found_entities,
            "unresolved_entities": [e for e in missing if e not in found_entities],
            "new_chunks_added": chunks_added,
            "context_graph_chunks_added": cg_chunks,
        }
        if _trace is not None:
            _trace.ref_expansion_missing = list(missing)
            _trace.ref_expansion_found = list(found_entities)
            _trace.ref_expansion_chunks_added = chunks_added
            _trace.context_graph_expansion_chunks_added = cg_chunks

        if not found_entities:
            return results

        return self._select_cohort(list(new_pool.values()), query_embedding, k)

    def _context_graph_expand_missing(
        self,
        missing: list[str],
        new_pool: dict[str, ScoredChunk],
        found_entities: list[str],
        namespaces: list[str] | None,
    ) -> None:
        """Expand missing entities via context graph edges into new_pool (mutates in place)."""  # noqa: E501
        # Caller (_entity_ref_expand) guards: self._context_graph_expansion and self._entity_index is not None  # noqa: E501
        assert self._entity_index is not None
        namespace = namespaces[0] if namespaces else "global"
        for entity in missing:
            edges = self._store.get_context_graph(
                entity,
                namespace=namespace,
                min_weight=self._context_graph_min_weight,
            )
            for edge in edges[: self._context_graph_top_k]:
                for linked_chunk_id, _ in self._entity_index.get_chunks_for_entity(
                    edge.target_entity_id, top_n=self._entity_top_n
                ):
                    if linked_chunk_id not in new_pool:
                        chunk = self._fetch_chunk(linked_chunk_id)
                        if chunk is not None:
                            new_pool[linked_chunk_id] = ScoredChunk(
                                chunk_id=linked_chunk_id,
                                chunk=chunk,
                                score=edge.weight,
                                provenance="context_graph_expansion",
                                linked_by=edge.target_entity_id,
                            )
            if entity not in found_entities:
                found_entities.append(entity)

    def _semantic_expand_missing(
        self,
        missing: list[str],
        new_pool: dict[str, ScoredChunk],
        found_entities: list[str],
        precomputed_entity_vecs: dict[str, np.ndarray] | None,
        namespaces: list[str] | None,
        domain_ids: list[str] | None,
    ) -> list[str]:
        """Per-entity vector search for missing entities. Returns (possibly filtered) missing list."""  # noqa: E501
        if self._entity_ref_expansion_per_k is not None:
            per_k = self._entity_ref_expansion_per_k
        else:
            per_k = max(3, self._entity_ref_expansion_k // max(len(missing), 1))
        min_sim = self._entity_ref_expansion_min_sim
        if precomputed_entity_vecs is not None:
            available = [e for e in missing if e in precomputed_entity_vecs]
            entity_vecs = (
                np.stack([precomputed_entity_vecs[e] for e in available]) if available else None
            )
            missing = available
        else:
            # Caller (_entity_ref_expand) guards: self._embed_fn is not None
            assert self._embed_fn is not None
            entity_vecs = self._embed_fn(missing)
        _missing_iter = missing if (entity_vecs is not None and len(missing) > 0) else []
        for i, entity in enumerate(_missing_iter):
            # entity_vecs non-None: loop condition above guards entry
            assert entity_vecs is not None
            hits = self._store.search(
                entity_vecs[i],
                limit=per_k,
                query_text=None,
                namespaces=namespaces,
                domain_ids=domain_ids,
                exclude_chunk_types=SYNTHETIC_CHUNK_TYPES,
            )
            for chunk_id, score, chunk in hits:
                if min_sim is not None and score < min_sim:
                    continue
                if chunk_id not in new_pool:
                    new_pool[chunk_id] = ScoredChunk(
                        chunk_id=chunk_id,
                        chunk=chunk,
                        score=score,
                        provenance="entity_ref_expansion",
                    )
                if entity not in found_entities:
                    found_entities.append(entity)
        return missing

    def _literal_expand_missing(
        self,
        missing: list[str],
        new_pool: dict[str, ScoredChunk],
        found_entities: list[str],
        query_embedding: np.ndarray,
        query_text: str | None,
        namespaces: list[str] | None,
        domain_ids: list[str] | None,
    ) -> None:
        """Literal substring fallback: expand pool with chunks containing missing entity text."""
        expanded_seeds = self._store.search(
            query_embedding,
            limit=self._entity_ref_expansion_k,
            query_text=query_text,
            namespaces=namespaces,
            domain_ids=domain_ids,
            exclude_chunk_types=SYNTHETIC_CHUNK_TYPES,
        )
        for chunk_id, score, chunk in expanded_seeds:
            if chunk_id in new_pool:
                continue
            chunk_text = (chunk.content or "").lower()
            matched = [e for e in missing if e.lower() in chunk_text]
            if matched:
                new_pool[chunk_id] = ScoredChunk(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=score,
                    provenance="entity_ref_expansion",
                )
                for e in matched:
                    if e not in found_entities:
                        found_entities.append(e)
