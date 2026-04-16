# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 79a01dcf-aad6-4086-a01a-4b3465b18e71
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Contextual enrichment — prepend document name and section path into embedding_content.

The thesis: encoding document name and section breadcrumbs into embedding_content
before embedding improves retrieval relevance, clustering, and NER vs embedding
raw content alone.

WHY DOCUMENT NAME:
  Section headings are often generic even in well-structured documents:
  "1. Definitions", "8. Limitation of Liability", "9. Indemnification" — these
  appear identically across every SaaS contract. The filename ("techcorp_msa",
  "cloudsolutions_agreement") is the primary disambiguator.

  Similarly, spreadsheet sheets named "Data" or "Q1", slides titled "Overview",
  and wiki pages with repeated section structures all depend on the document
  name for context that the heading alone cannot provide.

WHY SECTION PATH:
  Within a single document, repeated leaf headings ("Parameters", "Returns",
  "Notes", "Headcount") need the parent path to be meaningful. A chunk from
  "APAC > Engineering > Headcount" is indistinguishable from "EMEA > Engineering
  > Headcount" on content alone if the region name only appears in the ancestor
  heading.

TOGETHER:
  embedding_content = "Document: {name}\nSection: {path}\n\n{content}"
  gives every chunk a unique, human-readable address that survives any split
  boundary.
"""

from __future__ import annotations

import dataclasses
from .models import DocumentChunk

_STRATEGIES = ("prefix", "inline")


def enrich_chunk(chunk: DocumentChunk, strategy: str = "prefix") -> DocumentChunk:
    """Return a new DocumentChunk with embedding_content set.

    Never mutates the input chunk.

    The generated embedding_content includes both the document name and the
    section path so that even chunks with generic headings can be distinguished
    across multiple documents.

    Args:
        chunk: The source DocumentChunk to enrich.
        strategy: One of:
            "prefix" (default) — multi-line header block before content::

                Document: techcorp_msa
                Section: Limitation of Liability

                IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY…

            "inline" — compact single-line prefix::

                [techcorp_msa > Limitation of Liability] IN NO EVENT…

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

    doc_name = chunk.document_name or ""
    section = chunk.section

    if strategy == "prefix":
        if doc_name and section:
            embedding_content = (
                f"Document: {doc_name}\nSection: {section}\n\n{chunk.content}"
            )
        elif doc_name:
            embedding_content = f"Document: {doc_name}\n\n{chunk.content}"
        elif section:
            embedding_content = f"Section: {section}\n\n{chunk.content}"
        else:
            embedding_content = chunk.content

    else:  # inline
        if doc_name and section:
            embedding_content = f"[{doc_name} > {section}] {chunk.content}"
        elif doc_name:
            embedding_content = f"[{doc_name}] {chunk.content}"
        elif section:
            embedding_content = f"[{section}] {chunk.content}"
        else:
            embedding_content = chunk.content

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
