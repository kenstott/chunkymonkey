# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 75483088-32a9-4c97-bbaa-288624d47278
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DDL for chonk's minimal storage schema."""

# Bump whenever a change invalidates the contents of an existing index — a new
# chunk_id derivation, a cache key change, or any column whose meaning shifts.
# Indexes stamped with a different version are rejected on open; there is no
# in-place migration because chunk_id is a foreign key in chunk_entities,
# chunk_clusters, and context_graph_edges.
#
# 2 — chunk_id hashes the full chunk content (was: first 100 characters only).
# 3 — community_cache validity keys off a chunk-id fingerprint (was: chunk count).
# 4 — entity IDs are "{entity_type}:{name_slug}" (was: name slug only), so
#     customer:mercury and element:mercury are distinct entities. Every table
#     keyed by entity_id — chunk_entities, entity_aliases, svo_triples,
#     context_graph_edges — holds the old scheme and must be rebuilt.
SCHEMA_VERSION = 4

# Chunk types chonk generates itself rather than reading from a source document.
# A bare search() returns them like any other row — callers retrieving document
# evidence pass exclude_chunk_types=SYNTHETIC_CHUNK_TYPES so a generated summary
# or entity card is never cited as source text.
SYNTHETIC_CHUNK_TYPES = ["entity", "community_summary"]


class SchemaVersionError(RuntimeError):
    """Raised when an index was written by an incompatible chonk schema version."""


SCHEMA_META_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    id         INTEGER PRIMARY KEY,
    version    INTEGER NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
""".strip()

EMBEDDINGS_DDL = """
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id     TEXT PRIMARY KEY,
    document_name TEXT NOT NULL,
    section      TEXT,
    chunk_index  INTEGER NOT NULL DEFAULT 0,
    content      TEXT NOT NULL,
    breadcrumb   TEXT,
    chunk_type   TEXT NOT NULL DEFAULT 'document',
    source_offset INTEGER,
    source_length INTEGER,
    namespace    TEXT,
    -- Denormalised from chunk_entities so every backend can filter chunks by
    -- the type of entity they mention without a join. Written by build_ner
    -- after NER; NULL until then.
    entity_types VARCHAR[],
    embedding    FLOAT[{dim}]
)
""".strip()

EMBEDDINGS_MIGRATE_BREADCRUMB = """
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS breadcrumb TEXT
""".strip()

EMBEDDINGS_MIGRATE_NAMESPACE = """
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS namespace TEXT
""".strip()

EMBEDDINGS_MIGRATE_SOURCE_DETAIL = """
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS source_detail TEXT
""".strip()

EMBEDDINGS_MIGRATE_SOURCE_ID = """
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS source_id VARCHAR
""".strip()

EMBEDDINGS_MIGRATE_DOMAIN_ID = """
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS domain_id VARCHAR
""".strip()

EMBEDDINGS_MIGRATE_SESSION_FINGERPRINT = """
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS session_fingerprint VARCHAR
""".strip()

COMMUNITY_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS community_cache (
    fingerprint   VARCHAR PRIMARY KEY,
    domain_ids    JSON,
    chunk_count   INTEGER,
    created_at    TIMESTAMP DEFAULT current_timestamp
)
""".strip()

COMMUNITY_CACHE_MIGRATE_CHUNK_FINGERPRINT = """
ALTER TABLE community_cache ADD COLUMN IF NOT EXISTS chunk_fingerprint VARCHAR
""".strip()

NAMESPACES_DDL = """
CREATE TABLE IF NOT EXISTS namespaces (
    namespace_id  VARCHAR PRIMARY KEY,
    owner         VARCHAR,
    description   VARCHAR,
    created_at    TIMESTAMP DEFAULT current_timestamp,
    updated_at    TIMESTAMP DEFAULT current_timestamp
)
""".strip()

DOMAINS_DDL = """
CREATE TABLE IF NOT EXISTS domains (
    domain_id     VARCHAR PRIMARY KEY,
    namespace_id  VARCHAR,
    name          VARCHAR,
    description   VARCHAR,
    parent_id     VARCHAR,
    created_at    TIMESTAMP DEFAULT current_timestamp,
    updated_at    TIMESTAMP DEFAULT current_timestamp
)
""".strip()

DOMAINS_MIGRATE_PARENT_ID = """
ALTER TABLE domains ADD COLUMN IF NOT EXISTS parent_id VARCHAR
""".strip()

SOURCES_DDL = """
CREATE TABLE IF NOT EXISTS sources (
    source_id     VARCHAR PRIMARY KEY,
    domain_id     VARCHAR,
    type          VARCHAR,
    uri           VARCHAR,
    config        JSON,
    last_crawled  TIMESTAMP
)
""".strip()

ENTITIES_DDL = """
CREATE TABLE IF NOT EXISTS entities (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    display_name TEXT NOT NULL,
    entity_type  TEXT NOT NULL DEFAULT 'concept',
    description  TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""".strip()

ENTITIES_MIGRATE_DESCRIPTION = """
ALTER TABLE entities ADD COLUMN IF NOT EXISTS description TEXT
""".strip()

CHUNK_ENTITIES_DDL = """
CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id        TEXT NOT NULL,
    entity_id       TEXT NOT NULL,
    frequency       INTEGER NOT NULL DEFAULT 1,
    positions_json  TEXT NOT NULL DEFAULT '[]',
    score           REAL NOT NULL DEFAULT 0.0,
    namespace       TEXT,
    PRIMARY KEY (chunk_id, entity_id)
)
""".strip()

CHUNK_ENTITIES_MIGRATE_NAMESPACE = """
ALTER TABLE chunk_entities ADD COLUMN IF NOT EXISTS namespace TEXT
""".strip()

ENTITY_ALIASES_DDL = """
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias       TEXT    NOT NULL,
    entity_id   TEXT    NOT NULL,
    namespace   TEXT    NOT NULL DEFAULT 'global',
    source      TEXT    NOT NULL DEFAULT 'llm',
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (alias, namespace, entity_id)
)
""".strip()

DOCUMENTS_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    document_name TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    source_uri    TEXT NOT NULL DEFAULT '',
    indexed_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    chunk_count   INTEGER NOT NULL DEFAULT 0
)
""".strip()

NER_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS ner_cache (
    config_fingerprint  VARCHAR PRIMARY KEY,
    chunk_count         INTEGER NOT NULL,
    created_at          TIMESTAMP DEFAULT current_timestamp
)
""".strip()

CHUNK_CLUSTERS_DDL = """
CREATE TABLE IF NOT EXISTS chunk_clusters (
    chunk_id    TEXT NOT NULL,
    cluster_id  INTEGER NOT NULL,
    namespace   TEXT NOT NULL DEFAULT 'global',
    PRIMARY KEY (chunk_id, namespace, cluster_id)
)
""".strip()

CONTEXT_GRAPH_EDGES_DDL = """
CREATE TABLE IF NOT EXISTS context_graph_edges (
    source_entity_id  TEXT NOT NULL,
    target_entity_id  TEXT NOT NULL,
    namespace         TEXT NOT NULL DEFAULT 'global',
    weight            REAL NOT NULL,
    svo_signal        REAL NOT NULL DEFAULT 0.0,
    cooccur_signal    REAL NOT NULL DEFAULT 0.0,
    cluster_signal    REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (source_entity_id, target_entity_id, namespace)
)
""".strip()

CONTEXT_GRAPH_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS context_graph_cache (
    namespace         TEXT PRIMARY KEY,
    chunk_fingerprint TEXT NOT NULL,
    entity_count      INTEGER NOT NULL,
    edge_count        INTEGER NOT NULL,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""".strip()

NAMESPACE_BUILD_LOG_DDL = """
CREATE TABLE IF NOT EXISTS namespace_build_log (
    namespace_id       VARCHAR PRIMARY KEY,
    chunks_built_at    TIMESTAMP,
    ner_built_at       TIMESTAMP,
    svo_built_at       TIMESTAMP,
    community_built_at TIMESTAMP
)
""".strip()

VSS_INDEX_DDL = "CREATE INDEX IF NOT EXISTS embeddings_vss ON embeddings USING HNSW (embedding) WITH (metric = 'cosine')"  # noqa: E501
VSS_DROP_INDEX_DDL = "DROP INDEX IF EXISTS embeddings_vss"

FTS_DDL = "PRAGMA create_fts_index('embeddings', 'chunk_id', 'content', overwrite=1)"


def get_ddl(embedding_dim: int = 1024) -> list[str]:
    return [
        SCHEMA_META_DDL,
        EMBEDDINGS_DDL.format(dim=embedding_dim),
        EMBEDDINGS_MIGRATE_BREADCRUMB,
        EMBEDDINGS_MIGRATE_NAMESPACE,
        EMBEDDINGS_MIGRATE_SOURCE_DETAIL,
        EMBEDDINGS_MIGRATE_SOURCE_ID,
        EMBEDDINGS_MIGRATE_DOMAIN_ID,
        EMBEDDINGS_MIGRATE_SESSION_FINGERPRINT,
        COMMUNITY_CACHE_DDL,
        COMMUNITY_CACHE_MIGRATE_CHUNK_FINGERPRINT,
        NAMESPACES_DDL,
        DOMAINS_DDL,
        DOMAINS_MIGRATE_PARENT_ID,
        SOURCES_DDL,
        ENTITIES_DDL,
        ENTITIES_MIGRATE_DESCRIPTION,
        CHUNK_ENTITIES_DDL,
        CHUNK_ENTITIES_MIGRATE_NAMESPACE,
        ENTITY_ALIASES_DDL,
        DOCUMENTS_DDL,
        NER_CACHE_DDL,
        CHUNK_CLUSTERS_DDL,
        CONTEXT_GRAPH_EDGES_DDL,
        CONTEXT_GRAPH_CACHE_DDL,
        NAMESPACE_BUILD_LOG_DDL,
    ]
