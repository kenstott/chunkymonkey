"""DDL for chunkeymonkey's minimal storage schema."""

EMBEDDINGS_DDL = """
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id     TEXT PRIMARY KEY,
    document_name TEXT NOT NULL,
    section      TEXT,
    chunk_index  INTEGER NOT NULL DEFAULT 0,
    content      TEXT NOT NULL,
    chunk_type   TEXT NOT NULL DEFAULT 'document',
    source_offset INTEGER,
    source_length INTEGER,
    embedding    FLOAT[{dim}]
)
""".strip()

ENTITIES_DDL = """
CREATE TABLE IF NOT EXISTS entities (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    display_name TEXT NOT NULL,
    entity_type  TEXT NOT NULL DEFAULT 'concept',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""".strip()

CHUNK_ENTITIES_DDL = """
CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id   TEXT NOT NULL,
    entity_id  TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    PRIMARY KEY (chunk_id, entity_id)
)
""".strip()

VSS_INDEX_DDL = "CREATE INDEX IF NOT EXISTS embeddings_vss ON embeddings USING HNSW (embedding)"

FTS_DDL = "PRAGMA create_fts_index('embeddings', 'chunk_id', 'content', overwrite=1)"


def get_ddl(embedding_dim: int = 1024) -> list[str]:
    return [
        EMBEDDINGS_DDL.format(dim=embedding_dim),
        ENTITIES_DDL,
        CHUNK_ENTITIES_DDL,
    ]
