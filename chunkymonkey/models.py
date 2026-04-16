# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 19005421-3091-4ff1-9dc5-cf99f1f3ef96
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Core data models for chunkymonkey."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EntityAssociation:
    """Association between an entity and a chunk, with scoring metadata."""
    entity_id: str
    chunk_id: str
    frequency: int
    positions: list[int]
    score: float
    chunk_length: int = 1  # character length of the chunk, for score recomputation


@dataclass
class Entity:
    """A named entity from the vocabulary."""
    entity_id: str
    name: str
    display_name: str
    entity_type: str = "concept"
    aliases: list[str] = field(default_factory=list)


@dataclass
class ClusterRecord:
    """A cluster of entities with cohesion score."""
    cluster_id: str
    entities: list[str]
    cohesion_score: float = 0.0


@dataclass
class ScoredChunk:
    """A chunk returned from enhanced search, with composite score and provenance."""
    chunk_id: str
    chunk: "DocumentChunk"
    score: float
    provenance: str  # "seed" | "structural" | "entity_adjacent" | "cluster_adjacent"
    linked_by: Optional[str] = None   # entity_id that linked this chunk
    cluster: Optional[str] = None     # cluster_id for cluster-adjacent chunks
    embedding: Optional[list] = None  # cached embedding for scoring


@dataclass
class DocumentChunk:
    """A chunk of a document for embedding and search.

    Attributes:
        document_name: Name of the document this chunk belongs to
        content: The text content of this chunk
        section: Optional section header this chunk is under
        chunk_index: Index of this chunk within the document
        source_offset: Byte offset in the original source document
        source_length: Length in bytes of the original source span
        embedding_content: Set by enrich_chunks(); what actually gets embedded
        chunk_type: "document" | "schema" | "api"
    """
    document_name: str
    content: str
    section: Optional[str] = None
    chunk_index: int = 0
    source_offset: Optional[int] = None
    source_length: Optional[int] = None
    embedding_content: Optional[str] = None
    chunk_type: str = "document"


@dataclass
class LoadedDocument:
    """A document loaded from a source, ready for chunking.

    Attributes:
        name: Display name of the document
        content: Full extracted text content
        doc_format: "pdf", "docx", "markdown", "text", etc.
        source_uri: Original URI or path the document was fetched from
        sections: Ordered list of section headings found in the document
    """
    name: str
    content: str
    doc_format: str
    source_uri: str = ""
    sections: list[str] = field(default_factory=list)
