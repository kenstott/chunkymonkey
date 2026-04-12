"""Core data models for chunkeymonkey."""

from dataclasses import dataclass, field
from typing import Optional


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
