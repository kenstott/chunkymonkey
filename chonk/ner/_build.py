# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: c8346bd4-3a05-4e2b-b9bb-03772c391075
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""build_ner — run NER on a Store and persist results to chunk_entities."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from ._index import EntityIndex
from ._spacy import SpacyMatcher
from ._spacy_labels import SpacyLabel
from ._vocabulary import EntityMatch

if TYPE_CHECKING:
    import duckdb

    from ..models import DocumentChunk
    from ._schema import SchemaMatcher
    from ._vocabulary import VocabularyMatcher

_SPACY_MODEL = "en_core_web_sm"
_NUMERIC_TYPES = {
    SpacyLabel.ORDINAL,
    SpacyLabel.MONEY,
    SpacyLabel.PERCENT,
    SpacyLabel.QUANTITY,
}
_ID_SUFFIXES = (
    "_identifier",
    "_reference",
    "_number",
    "_num",
    "_code",
    "_key",
    "_id",
    "_ref",
    "_no",
)


def _strip_id_alias(entity_id: str) -> str | None:
    """Return the suffix-stripped alias for an entity ID, or None if no suffix matches."""
    for suffix in _ID_SUFFIXES:
        if entity_id.endswith(suffix) and len(entity_id) > len(suffix):
            return entity_id[: -len(suffix)]
    return None


_NER_CACHE_DDL = (
    "CREATE TABLE IF NOT EXISTS ner_cache "
    "(config_fingerprint VARCHAR PRIMARY KEY, chunk_count INTEGER NOT NULL, "
    "created_at TIMESTAMP DEFAULT current_timestamp)"
)


def _ner_config_fingerprint(
    spacy_model: str,
    use_schema_vocab: bool,
    vocab_entities: list[dict[str, Any]] | None,
) -> str:
    payload = json.dumps(
        {
            "spacy_model": spacy_model,
            "use_schema_vocab": use_schema_vocab,
            "vocab_entities": sorted(
                [json.dumps(e, sort_keys=True) for e in (vocab_entities or [])]
            ),
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode(), usedforsecurity=False).hexdigest()


def _check_cache(
    con: duckdb.DuckDBPyConnection, fingerprint: str, force: bool
) -> tuple[bool, bool, set[str]]:
    """Determine incremental mode and skip set.

    Returns (should_return_early, incremental, skip_ids).
    should_return_early=True means caller should return the current chunk_entities count.
    """
    all_chunk_ids: set[str] = {
        row[0] for row in con.execute("SELECT chunk_id FROM embeddings").fetchall()
    }
    processed_ids: set[str] = {
        row[0] for row in con.execute("SELECT DISTINCT chunk_id FROM chunk_entities").fetchall()
    }
    cached = con.execute(
        "SELECT chunk_count FROM ner_cache WHERE config_fingerprint = ?", [fingerprint]
    ).fetchone()

    # Phase 2 cascades every chunk-keyed delete, so a chunk_entities row for a
    # chunk that no longer exists means a delete path was missed. Left unchecked
    # it inflates processed_ids forever and, when a chunk_id is reused, marks
    # changed content as already processed.
    orphaned = processed_ids - all_chunk_ids
    if orphaned:
        raise RuntimeError(
            f"chunk_entities references {len(orphaned)} chunk_id(s) absent from embeddings "
            f"(e.g. {sorted(orphaned)[:3]}). The NER cache cannot be trusted — a delete "
            f"path failed to cascade. Rebuild the index."
        )

    config_match = cached is not None
    new_chunk_ids = all_chunk_ids - processed_ids

    if force:
        return False, False, set()
    if config_match and not new_chunk_ids:
        return True, False, set()
    if config_match and new_chunk_ids:
        return False, True, processed_ids
    return False, False, set()


def _all_chunk_id_count(con: duckdb.DuckDBPyConnection) -> int:
    row = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()
    if row is None:
        raise RuntimeError("COUNT(*) returned no rows")
    return row[0]


def _build_vocab_matchers(
    all_chunks: list[DocumentChunk],
    use_schema_vocab: bool,
    vocab_entities: list[dict[str, Any]] | None,
) -> tuple[SchemaMatcher | None, VocabularyMatcher | None]:
    """Build schema and data matchers when vocabulary sources are configured.

    Returns (schema_matcher, data_matcher) — either may be None.
    """
    if not use_schema_vocab and not vocab_entities:
        return None, None

    from ._schema_vocab import SchemaVocabBuilder

    builder = SchemaVocabBuilder()
    if use_schema_vocab:
        builder.add_chunks(all_chunks)
    for entry in vocab_entities or []:
        etype = entry.get("entity_type", "term")
        if entry.get("type") == "static":
            builder.add_entities(entry.get("names", []), entity_type=etype)
        elif entry.get("type") == "db_query":
            builder.add_from_db(entry["connection"], {etype: entry["sql"]})
    schema_matcher = builder.build()
    data_matcher = builder.build_data_matcher() if builder.data_term_count() > 0 else None
    return schema_matcher, data_matcher


def _collect_chunks_to_process(
    all_chunks: list[DocumentChunk], skip_ids: set[str]
) -> list[tuple[str, DocumentChunk]]:
    """Filter all_chunks to those not in skip_ids, returning (chunk_id, chunk) pairs."""
    from chonk.storage._vector import DuckDBVectorBackend

    result: list[tuple[str, DocumentChunk]] = []
    for c in all_chunks:
        embed_content = c.embedding_content if c.embedding_content else c.content
        cid = DuckDBVectorBackend._generate_chunk_id(c.document_name, c.chunk_index, embed_content)
        if cid not in skip_ids:
            result.append((cid, c))
    return result


def _run_ner_on_chunks(
    chunks_to_process: list[tuple[str, DocumentChunk]],
    matcher: SpacyMatcher,
    schema_matcher: SchemaMatcher | None,
    data_matcher: VocabularyMatcher | None,
) -> tuple[EntityIndex, dict[str, tuple[str, str, str]]]:
    """Run NER over chunks_to_process, return (entity_index, entity_meta)."""
    from ._merge import merge_matches

    entity_index = EntityIndex()
    entity_meta: dict[str, tuple[str, str, str]] = {}

    for chunk_id, chunk in chunks_to_process:
        if schema_matcher is not None or data_matcher is not None:
            vocab_hits: list[EntityMatch] = []
            if schema_matcher is not None:
                vocab_hits = merge_matches(
                    schema_matcher.match(chunk.content),
                    vocab_hits,
                    source_text=chunk.content,
                )
            if data_matcher is not None:
                vocab_hits = merge_matches(
                    data_matcher.match(chunk.content),
                    vocab_hits,
                    source_text=chunk.content,
                )
            combined = merge_matches(
                vocab_hits, matcher.match(chunk.content), source_text=chunk.content
            )
            for m in combined:
                if m.entity_id not in entity_meta:
                    entity_meta[m.entity_id] = (m.name, m.display_name, m.entity_type or "concept")
            entity_index.index_chunk(chunk_id, chunk.content, combined)
        else:
            matches = matcher.match(chunk.content)
            for m in matches:
                if m.entity_id not in entity_meta:
                    entity_meta[m.entity_id] = (m.name, m.display_name, m.entity_type or "concept")
            entity_index.index_chunk(chunk_id, chunk.content, matches)

    entity_index.recompute_scores()
    return entity_index, entity_meta


def _persist_associations(
    con: duckdb.DuckDBPyConnection,
    data: dict[str, Any],
    entity_meta: dict[str, tuple[str, str, str]],
    incremental: bool,
) -> None:
    """Write associations and entities to the DB, clearing first unless incremental."""
    if not incremental:
        con.execute("DELETE FROM chunk_entities")
        con.execute("DELETE FROM entities")
    for a in data["associations"]:
        ns_row = con.execute(
            "SELECT namespace FROM embeddings WHERE chunk_id = ?", [a["chunk_id"]]
        ).fetchone()
        chunk_namespace = ns_row[0] if ns_row else None
        con.execute(
            "INSERT OR REPLACE INTO chunk_entities"
            "(chunk_id, entity_id, frequency, positions_json, score, namespace)"
            " VALUES (?,?,?,?,?,?)",
            [
                a["chunk_id"],
                a["entity_id"],
                a["frequency"],
                json.dumps(a["positions"]),
                a["score"],
                chunk_namespace,
            ],
        )
        name, display_name, entity_type = entity_meta.get(
            a["entity_id"], (a["entity_id"], a["entity_id"], "concept")
        )
        con.execute(
            "INSERT OR IGNORE INTO entities(id, name, display_name, entity_type) VALUES (?,?,?,?)",
            [a["entity_id"], name, display_name, entity_type],
        )
        alias = _strip_id_alias(a["entity_id"])
        if alias:
            con.execute(
                "INSERT OR IGNORE INTO entity_aliases(alias, entity_id, source) VALUES (?,?,?)",
                [alias, a["entity_id"], "strip_suffix"],
            )


def build_ner(
    store: Any,  # noqa: ANN401
    *,
    spacy_model: str = _SPACY_MODEL,
    use_schema_vocab: bool = False,
    vocab_entities: list[dict[str, Any]] | None = None,
    force: bool = False,
    build_context_graph: bool = False,
    namespace: str = "global",
) -> int:
    """Run NER on all chunks in *store* and persist results to ``chunk_entities``.

    Incremental by default: skips chunks already processed under the same config
    fingerprint.  Pass ``force=True`` to rebuild from scratch regardless.

    Args:
        store: A ``Store`` instance (open context manager).
        spacy_model: spaCy model name.
        use_schema_vocab: Build schema vocabulary from db_schema/api chunks.
        vocab_entities: Extra entity vocab entries — each dict has keys
            ``type`` ("static" or "db_query"), ``entity_type``, and either
            ``names`` (static) or ``connection`` + ``sql`` (db_query).
        force: If True, ignore cache and rebuild all chunks.

    Returns:
        Number of ``chunk_entities`` associations written.
    """
    con = store.vector._conn
    con.execute(_NER_CACHE_DDL)

    fingerprint = _ner_config_fingerprint(spacy_model, use_schema_vocab, vocab_entities)

    should_return_early, incremental, skip_ids = _check_cache(con, fingerprint, force)
    if should_return_early:
        _row = con.execute("SELECT COUNT(*) FROM chunk_entities").fetchone()
        if _row is None:
            raise RuntimeError("COUNT(*) returned no rows")
        return _row[0]

    label_types = [t for t in SpacyLabel if t not in _NUMERIC_TYPES]
    matcher = SpacyMatcher(model=spacy_model, strip_numeric=True, entity_types=label_types)

    all_chunks = store.vector.get_all_chunks()
    schema_matcher, data_matcher = _build_vocab_matchers(
        all_chunks, use_schema_vocab, vocab_entities
    )
    chunks_to_process = _collect_chunks_to_process(all_chunks, skip_ids)

    entity_index, entity_meta = _run_ner_on_chunks(
        chunks_to_process, matcher, schema_matcher, data_matcher
    )

    data = entity_index.to_dict()
    _persist_associations(con, data, entity_meta, incremental)

    all_chunk_count = _all_chunk_id_count(con)
    con.execute(
        "INSERT OR REPLACE INTO ner_cache(config_fingerprint, chunk_count) VALUES (?, ?)",
        [fingerprint, all_chunk_count],
    )

    if build_context_graph:
        from ..graph._context_graph import build_context_graph_edges

        build_context_graph_edges(con, namespace=namespace, force=True)

    return len(data["associations"])
