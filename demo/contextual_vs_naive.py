"""
Contextual vs Naive RAG — demonstration of the chunkeymonkey thesis.

Shows that prepending section breadcrumbs to chunk content before embedding
improves retrieval relevance. Runs with zero external dependencies.

Usage: python demo/contextual_vs_naive.py
"""

from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path

from chunkeymonkey import DocumentLoader
from chunkeymonkey.models import DocumentChunk

# ---------------------------------------------------------------------------
# Locate sample document relative to this script
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
SAMPLE_DOC = _HERE / "sample_doc.md"


# ---------------------------------------------------------------------------
# Minimal TF-IDF bag-of-words vectoriser (no sklearn, no numpy required)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    import re
    return re.findall(r"[a-z]+", text.lower())


def _build_vocab(docs: list[str]) -> list[str]:
    vocab: set[str] = set()
    for doc in docs:
        vocab.update(_tokenise(doc))
    return sorted(vocab)


def _tf_idf_vector(text: str, vocab: list[str], idf: dict[str, float]) -> list[float]:
    tokens = _tokenise(text)
    if not tokens:
        return [0.0] * len(vocab)
    tf = Counter(tokens)
    total = len(tokens)
    return [tf.get(w, 0) / total * idf.get(w, 0.0) for w in vocab]


def _compute_idf(docs: list[str], vocab: list[str]) -> dict[str, float]:
    n = len(docs)
    idf: dict[str, float] = {}
    for word in vocab:
        df = sum(1 for doc in docs if word in _tokenise(doc))
        idf[word] = math.log((n + 1) / (df + 1)) + 1.0
    return idf


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _retrieve(
    query: str,
    chunks: list[DocumentChunk],
    vectors: list[list[float]],
    vocab: list[str],
    idf: dict[str, float],
    limit: int = 3,
) -> list[tuple[float, DocumentChunk]]:
    q_vec = _tf_idf_vector(query, vocab, idf)
    scored = [((_cosine(q_vec, v), c)) for v, c in zip(vectors, chunks)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


# ---------------------------------------------------------------------------
# Queries and expected sections
# ---------------------------------------------------------------------------

QUERIES = [
    ("participant demographics and sample size", ["2.1", "Participants", "Demographics"]),
    ("working memory capacity moderation effect", ["3.2", "Subgroup", "Moderation"]),
    ("statistical results p-value effect size", ["3.1", "Primary", "Statistical"]),
    ("limitations of the study generalisability", ["4.2", "Limitations"]),
    ("spaced retrieval procedure study schedule", ["2.2", "Procedure"]),
]


def _top3_sections(results: list[tuple[float, DocumentChunk]]) -> list[str | None]:
    return [c.section for _, c in results]


def _query_hits_expected(
    sections: list[str | None],
    expected_keywords: list[str],
) -> bool:
    """Return True if any top-3 section contains at least one expected keyword."""
    for sec in sections:
        if sec is None:
            continue
        for kw in expected_keywords:
            if kw.lower() in sec.lower():
                return True
    return False


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    if not SAMPLE_DOC.exists():
        raise FileNotFoundError(
            f"Sample document not found: {SAMPLE_DOC}\n"
            "Run this script from the project root or demo/ directory."
        )

    print("=" * 72)
    print("chunkeymonkey — Contextual vs Naive RAG Demo")
    print("=" * 72)
    print(f"Document: {SAMPLE_DOC.name}")
    print()

    # Load naive chunks (no section enrichment in embedding_content)
    naive_chunks = DocumentLoader(context_strategy=None).load(str(SAMPLE_DOC))

    # Load contextual chunks (section breadcrumb prepended to embedding_content)
    ctx_chunks = DocumentLoader(context_strategy="prefix").load(str(SAMPLE_DOC))

    print(f"Total chunks: {len(naive_chunks)} (naive) / {len(ctx_chunks)} (contextual)")
    print()

    # Build vectors for naive approach using raw content
    naive_texts = [c.content for c in naive_chunks]
    naive_vocab = _build_vocab(naive_texts)
    naive_idf = _compute_idf(naive_texts, naive_vocab)
    naive_vecs = [_tf_idf_vector(t, naive_vocab, naive_idf) for t in naive_texts]

    # Build vectors for contextual approach using embedding_content (has breadcrumbs)
    ctx_texts = [c.embedding_content for c in ctx_chunks]
    ctx_vocab = _build_vocab(ctx_texts)
    ctx_idf = _compute_idf(ctx_texts, ctx_vocab)
    ctx_vecs = [_tf_idf_vector(t, ctx_vocab, ctx_idf) for t in ctx_texts]

    naive_wins = 0
    ctx_wins = 0
    both_hit = 0
    neither_hit = 0

    for query, expected_keywords in QUERIES:
        print(f"Query: {query!r}")
        print(f"Expected section keywords: {expected_keywords}")

        naive_results = _retrieve(query, naive_chunks, naive_vecs, naive_vocab, naive_idf)
        ctx_results = _retrieve(query, ctx_chunks, ctx_vecs, ctx_vocab, ctx_idf)

        naive_sections = _top3_sections(naive_results)
        ctx_sections = _top3_sections(ctx_results)

        naive_hit = _query_hits_expected(naive_sections, expected_keywords)
        ctx_hit = _query_hits_expected(ctx_sections, expected_keywords)

        print(f"  Naive top-3 sections:      {naive_sections}")
        print(f"  Contextual top-3 sections: {ctx_sections}")
        print(f"  Naive hit: {'YES' if naive_hit else 'NO':3s}  |  Contextual hit: {'YES' if ctx_hit else 'NO'}")

        if ctx_hit and not naive_hit:
            ctx_wins += 1
            print("  --> Contextual WINS")
        elif naive_hit and not ctx_hit:
            naive_wins += 1
            print("  --> Naive wins")
        elif ctx_hit and naive_hit:
            both_hit += 1
            print("  --> Both hit")
        else:
            neither_hit += 1
            print("  --> Neither hit")
        print()

    total = len(QUERIES)
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Queries:             {total}")
    print(f"Contextual only win: {ctx_wins}")
    print(f"Naive only win:      {naive_wins}")
    print(f"Both correct:        {both_hit}")
    print(f"Neither correct:     {neither_hit}")
    print()
    if ctx_wins > naive_wins:
        print("Thesis supported: contextual enrichment improved retrieval relevance.")
    elif ctx_wins == naive_wins and both_hit == total:
        print("Both approaches performed equally well on this document.")
    else:
        print("Results mixed — consider tuning chunk_size or the sample document.")
    print()


if __name__ == "__main__":
    main()
