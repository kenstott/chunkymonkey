# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d433c31c-035d-4fc5-a7da-9e6596502656
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Chunky Monkey — a dairy-free RAG pipeline for delicious semantic similarity, clustering and NER."""  # noqa: E501

from ._config import ChonkConfig, EmbedConfig, IndexConfig, LoaderConfig
from ._struct_inference import infer_csv, infer_json, infer_jsonl, infer_parquet
from ._versioning import VersionedRef
from .chunking import (
    NOVEL_STRUCTURAL_LEVELS,
    chunk_document,
    extract_markdown_sections,
    is_list_line,
    is_table_line,
    merge_blocks,
    promote_plain_text_headers,
)
from .cluster import ClusterMap, CooccurrenceMatrix, cluster_entities
from .community import CommunityIndex, CommunityIndexBuilder, CommunitySummarizer, build_community
from .context import enrich_chunk, enrich_chunks
from .generation import Answer, AnswerContext, AnswerGenerator, PromptBuilder
from .graph import (
    PHASE_EMBED_ENTITIES,
    PHASE_EXTRACT,
    PHASE_LOAD,
    PHASE_PERSIST_ALIASES,
    PHASE_PERSIST_DESCRIPTIONS,
    PHASE_PERSIST_TRIPLES,
    VERB_SET,
    EntityGraphPipeline,
    EntityGraphStats,
    LLMClient,
    RelationshipIndex,
    RelationshipIndexBuilder,
    SVOExtractor,
    SVOTriple,
)
from .indexer import Indexer, IndexHandle, get_indexer, release_indexer
from .ingest import Index, build
from .lifecycle import NamespaceRefresher, build_namespace_async
from .loader import DocumentLoader
from .models import (
    ClusterRecord,
    DocumentChunk,
    Entity,
    EntityAssociation,
    LoadedDocument,
    ScoredChunk,
)
from .ner import (
    ALL_SPACY_LABELS,
    EntityIndex,
    EntityMatch,
    NerPipeline,
    SchemaMatcher,
    SchemaVocabBuilder,
    SpacyLabel,
    SpacyMatcher,
    VocabularyMatcher,
    merge_matches,
    normalize_schema_term,
)
from .schema import ColumnMeta, EndpointMeta, FieldMeta, TableMeta
from .search import EnhancedSearch, RetrievalTrace
from .storage import (
    DuckDBVectorBackend,
    EntityLookup,
    NamespaceEvidence,
    PgVectorBackend,
    Store,
    SyncResult,
    VectorBackend,
    prune_documents,
    sync_document,
)
from .transports import (
    CosmosCrawler,
    Crawler,
    DirectoryCrawler,
    DynamoDBCrawler,
    ElasticsearchCrawler,
    FetchResult,
    FirestoreCrawler,
    FtpTransport,
    HttpTransport,
    ImapTransport,
    ImportCrawler,
    LocalTransport,
    MongoCrawler,
    S3Transport,
    SftpTransport,
    SolrCrawler,
    SqlAlchemyTransport,
    SqlQueryTransport,
    Transport,
    WebCrawler,
)

__all__ = [
    "DocumentChunk",
    "LoadedDocument",
    "EntityAssociation",
    "Entity",
    "ClusterRecord",
    "ScoredChunk",
    "chunk_document",
    "extract_markdown_sections",
    "is_list_line",
    "is_table_line",
    "merge_blocks",
    "promote_plain_text_headers",
    "NOVEL_STRUCTURAL_LEVELS",
    "enrich_chunk",
    "enrich_chunks",
    "DocumentLoader",
    "Indexer",
    "IndexHandle",
    "get_indexer",
    "release_indexer",
    "ChonkConfig",
    "EmbedConfig",
    "LoaderConfig",
    "IndexConfig",
    "build",
    "Index",
    "build_namespace_async",
    "NamespaceRefresher",
    # Transports & Crawlers
    "Transport",
    "FetchResult",
    "Crawler",
    "WebCrawler",
    "DirectoryCrawler",
    "ImportCrawler",
    "LocalTransport",
    "HttpTransport",
    "S3Transport",
    "FtpTransport",
    "SftpTransport",
    "SqlAlchemyTransport",
    "SqlQueryTransport",
    "ImapTransport",
    "MongoCrawler",
    "ElasticsearchCrawler",
    "SolrCrawler",
    "DynamoDBCrawler",
    "FirestoreCrawler",
    "CosmosCrawler",
    # NER
    "VocabularyMatcher",
    "EntityMatch",
    "EntityIndex",
    "SpacyMatcher",
    "SpacyLabel",
    "ALL_SPACY_LABELS",
    "merge_matches",
    "SchemaMatcher",
    "normalize_schema_term",
    "SchemaVocabBuilder",
    "NerPipeline",
    # Cluster
    "CooccurrenceMatrix",
    "cluster_entities",
    "ClusterMap",
    # Search
    "EnhancedSearch",
    "RetrievalTrace",
    # Schema metadata
    "ColumnMeta",
    "TableMeta",
    "FieldMeta",
    "EndpointMeta",
    # Schema inference
    "infer_csv",
    "infer_json",
    "infer_jsonl",
    "infer_parquet",
    # Generation
    "AnswerContext",
    "PromptBuilder",
    "Answer",
    "AnswerGenerator",
    # Graph
    "SVOTriple",
    "VERB_SET",
    "RelationshipIndex",
    "LLMClient",
    "SVOExtractor",
    "RelationshipIndexBuilder",
    "EntityGraphPipeline",
    "EntityGraphStats",
    "PHASE_LOAD",
    "PHASE_EXTRACT",
    "PHASE_PERSIST_TRIPLES",
    "PHASE_PERSIST_DESCRIPTIONS",
    "PHASE_PERSIST_ALIASES",
    "PHASE_EMBED_ENTITIES",
    # Community
    "CommunityIndex",
    "CommunitySummarizer",
    "CommunityIndexBuilder",
    "build_community",
    # Versioning
    "VersionedRef",
    # Storage backends
    "Store",
    "EntityLookup",
    "NamespaceEvidence",
    "VectorBackend",
    "DuckDBVectorBackend",
    "PgVectorBackend",
    "SyncResult",
    "sync_document",
    "prune_documents",
]
