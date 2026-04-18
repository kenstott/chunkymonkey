# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d433c31c-035d-4fc5-a7da-9e6596502656
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Chunky Monkey — a dairy-free RAG pipeline for delicious semantic similarity, clustering and NER."""
from .chunking import chunk_document, extract_markdown_sections, is_list_line, is_table_line, merge_blocks, promote_plain_text_headers
from .models import DocumentChunk, LoadedDocument, EntityAssociation, Entity, ClusterRecord, ScoredChunk
from .ner import VocabularyMatcher, EntityMatch, EntityIndex, SpacyMatcher, SpacyLabel, ALL_SPACY_LABELS, merge_matches
from .cluster import CooccurrenceMatrix, cluster_entities, ClusterMap
from .search import EnhancedSearch
from .context import enrich_chunk, enrich_chunks
from .loader import DocumentLoader
from .transports import (
    Transport,
    FetchResult,
    Crawler,
    WebCrawler,
    DirectoryCrawler,
    LocalTransport,
    HttpTransport,
    S3Transport,
    FtpTransport,
    SftpTransport,
    SqlAlchemyTransport,
    ImapTransport,
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
    "enrich_chunk",
    "enrich_chunks",
    "DocumentLoader",
    # Transports & Crawlers
    "Transport",
    "FetchResult",
    "Crawler",
    "WebCrawler",
    "DirectoryCrawler",
    "LocalTransport",
    "HttpTransport",
    "S3Transport",
    "FtpTransport",
    "SftpTransport",
    "SqlAlchemyTransport",
    "ImapTransport",
    # NER
    "VocabularyMatcher",
    "EntityMatch",
    "EntityIndex",
    "SpacyMatcher",
    "SpacyLabel",
    "ALL_SPACY_LABELS",
    "merge_matches",
    # Cluster
    "CooccurrenceMatrix",
    "cluster_entities",
    "ClusterMap",
    # Search
    "EnhancedSearch",
]
