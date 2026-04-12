"""Core data models for chunkeymonkey."""

from dataclasses import dataclass
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
    """
    document_name: str
    content: str
    section: Optional[str] = None
    chunk_index: int = 0
    source_offset: Optional[int] = None
    source_length: Optional[int] = None
