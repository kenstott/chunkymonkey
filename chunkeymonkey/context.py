"""Contextual enrichment — prepend section path into embedding_content.

The thesis: encoding section breadcrumbs into embedding_content before
embedding improves retrieval relevance, clustering, and NER vs embedding
raw content alone.
"""

from __future__ import annotations

import dataclasses
from .models import DocumentChunk

_STRATEGIES = ("prefix", "inline")


def enrich_chunk(chunk: DocumentChunk, strategy: str = "prefix") -> DocumentChunk:
    """Return a new DocumentChunk with embedding_content set.

    Never mutates the input chunk.

    Args:
        chunk: The source DocumentChunk to enrich.
        strategy: "prefix" (default) or "inline".

    Returns:
        A new DocumentChunk with embedding_content populated.

    Raises:
        ValueError: If strategy is not a known value.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown enrichment strategy: {strategy!r}. "
            f"Choose one of: {_STRATEGIES}"
        )

    if chunk.section is None:
        embedding_content = chunk.content
    elif strategy == "prefix":
        embedding_content = f"Section: {chunk.section}\n\n{chunk.content}"
    else:  # inline
        embedding_content = f"[{chunk.section}] {chunk.content}"

    return dataclasses.replace(chunk, embedding_content=embedding_content)


def enrich_chunks(
    chunks: list[DocumentChunk],
    strategy: str = "prefix",
) -> list[DocumentChunk]:
    """Return a new list of DocumentChunks with embedding_content set on each.

    Args:
        chunks: Source chunks to enrich.
        strategy: "prefix" (default) or "inline".

    Returns:
        A new list of enriched DocumentChunks.
    """
    return [enrich_chunk(chunk, strategy=strategy) for chunk in chunks]
