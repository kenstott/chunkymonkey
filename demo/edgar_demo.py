# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 04abe60c-11b9-40f5-bbb8-d7954bd9b922
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""
EDGAR 10-K RAG Demo — Contextual vs Naive retrieval on real SEC filings.

Downloads four 10-K filings from SEC EDGAR (no login required), extracts
prose sections using EdgarExtractor, and demonstrates two effects:

  1. RETRIEVAL  — contextual embedding_content surfaces the right company
                  and section even when chunk content alone is ambiguous.
  2. CLUSTERING — contextual chunks from the same section cluster more
                  tightly than naive chunks.

Why EDGAR is a challenging corpus:

  SEC 10-K filings are a nearly perfect stress-test for contextual chunking:

  a) IDENTICAL STRUCTURE across all filers: every company writes the same
     Items (1A Risk Factors, 7 MD&A, 9A Controls and Procedures) in the same
     order, with overlapping boilerplate phrasing.

  b) LONG SECTIONS that split mid-paragraph: Risk Factors runs 20-50 pages;
     a 600-char chunk cut from the middle of a factor contains no company
     name, no item number — just generic risk language that appears across
     every filing.

  c) NUMERIC DENSITY: Item 7 (MD&A) mixes short boilerplate paragraphs with
     multi-sentence analytical prose. Continuation chunks after the opening
     paragraph contain only numbers and hedged forward-looking statements.

  The document name ("aapl_10k_2025") and section path ("Item 1A. Risk Factors")
  are the only reliable disambiguators when the chunk content alone matches
  across all four companies.

Usage::

    python demo/edgar_demo.py
    python demo/edgar_demo.py --cache-dir /tmp/edgar_cache
    python demo/edgar_demo.py --chunk-size 800 --top-k 5
    python demo/edgar_demo.py --model all-mpnet-base-v2

Requires: pip install sentence-transformers

Caching:
    First run fetches ~4 HTML files (each 1-3 MB) from EDGAR and writes them
    to --cache-dir (default: /tmp/edgar_cache).  Subsequent runs use the local
    cache to avoid network I/O.
"""
from __future__ import annotations

import argparse
import gzip
import math
import ssl
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── optional certifi for macOS SSL ───────────────────────────────────────────
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

# ── chunkymonkey ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunkymonkey import DocumentLoader
from chunkymonkey.extractors import EdgarExtractor
from chunkymonkey.models import DocumentChunk

# ─────────────────────────────────────────────────────────────────────────────
# EDGAR filing index — four large-cap companies, fiscal year 2024/2025
# ─────────────────────────────────────────────────────────────────────────────

_FILINGS: list[dict] = [
    {
        "name": "aapl_10k_2025",
        "company": "Apple Inc.",
        "cik": "320193",
        "filing_index": "https://data.sec.gov/submissions/CIK0000320193.json",
    },
    {
        "name": "msft_10k_2024",
        "company": "Microsoft Corp.",
        "cik": "789019",
        "filing_index": "https://data.sec.gov/submissions/CIK0000789019.json",
    },
    {
        "name": "amzn_10k_2024",
        "company": "Amazon.com Inc.",
        "cik": "1018724",
        "filing_index": "https://data.sec.gov/submissions/CIK0001018724.json",
    },
    {
        "name": "crm_10k_2025",
        "company": "Salesforce Inc.",
        "cik": "1108524",
        "filing_index": "https://data.sec.gov/submissions/CIK0001108524.json",
    },
]

_HEADERS = {
    "User-Agent": "chunkymonkey-demo/1.0 (research; contact@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

# ─────────────────────────────────────────────────────────────────────────────
# Queries
# ─────────────────────────────────────────────────────────────────────────────

# Queries are company-agnostic and topic-based.
# Evaluation does NOT check for a "correct" section — it measures cohort
# coherence: are the top-k results semantically similar to each other?
# A focused, coherent cohort makes for better LLM synthesis than a scattered one.

QUERIES: list[str] = [
    "foreign exchange rate currency exposure hedging",
    "goodwill impairment testing reporting unit fair value",
    "remaining performance obligations deferred revenue subscription",
    "cybersecurity risk management incident response governance board oversight",
    "stock repurchase share buyback program capital return dividends",
    "income tax effective rate uncertain tax positions valuation allowance",
    "disclosure controls procedures evaluation effectiveness management",
    "competition market position pricing pressure customer retention",
]


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR fetch utilities
# ─────────────────────────────────────────────────────────────────────────────

def _http_get(url: str, *, retries: int = 3, delay: float = 1.0) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=30) as resp:
                data = resp.read()
                if data[:2] == b"\x1f\x8b":
                    data = gzip.decompress(data)
                return data
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
    raise RuntimeError("unreachable")


def _get_latest_10k_url(submissions_url: str) -> str:
    import json
    data = json.loads(_http_get(submissions_url))
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    acc_nos = filings.get("accessionNumber", [])
    doc_lists = filings.get("primaryDocument", [])
    cik = str(data.get("cik", "")).zfill(10)
    for form, acc_no, primary_doc in zip(forms, acc_nos, doc_lists):
        if form == "10-K":
            acc_clean = acc_no.replace("-", "")
            return (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{acc_clean}/{primary_doc}"
            )
    raise RuntimeError(f"No 10-K found in submissions at {submissions_url}")


def _fetch_or_load(filing: dict, cache_dir: Path) -> tuple[str, bytes]:
    cache_file = cache_dir / f"{filing['name']}.htm"
    if cache_file.exists():
        print(f"  [cache] {filing['name']}")
        return filing["name"], cache_file.read_bytes()
    print(f"  [fetch] {filing['company']} — looking up latest 10-K...")
    url = _get_latest_10k_url(filing["filing_index"])
    print(f"          {url}")
    html_bytes = _http_get(url)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(html_bytes)
    print(f"          cached → {cache_file} ({len(html_bytes):,} bytes)")
    time.sleep(0.5)
    return filing["name"], html_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Dense embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode(model, texts: list[str]) -> np.ndarray:
    """Return (N, D) float32 array of L2-normalised embeddings."""
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def _retrieve(
    query: str,
    chunks: list[DocumentChunk],
    vecs: np.ndarray,
    model,
    k: int = 3,
) -> list[tuple[float, DocumentChunk]]:
    qv = model.encode([query], normalize_embeddings=True)[0]
    scores = vecs @ qv  # cosine (vecs are L2-normalised)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), chunks[i]) for i in top_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Clustering quality
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_quality(
    chunks: list[DocumentChunk],
    vecs: np.ndarray,
) -> tuple[float, float]:
    """Return (mean_intra, mean_inter) cosine similarity."""
    def _key(c: DocumentChunk) -> str:
        return (c.section or "").split(".")[0].strip() or "root"

    sims = vecs @ vecs.T  # (N, N) pairwise cosines
    intra_sims: list[float] = []
    inter_sims: list[float] = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            sim = float(sims[i, j])
            if _key(chunks[i]) == _key(chunks[j]):
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)
    intra = sum(intra_sims) / len(intra_sims) if intra_sims else 0.0
    inter = sum(inter_sims) / len(inter_sims) if inter_sims else 0.0
    return intra, inter


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDGAR 10-K contextual vs naive RAG demo (dense embeddings)"
    )
    parser.add_argument(
        "--cache-dir", default="/tmp/edgar_cache",
        help="Directory to cache downloaded .htm files (default: /tmp/edgar_cache)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=600,
        help="Target chars per chunk (default: 600)",
    )
    parser.add_argument(
        "--top-k", type=int, default=8,
        help="Cohort size per query (default: 8)",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    print("=" * 72)
    print("Chunky Monkey — EDGAR 10-K Contextual vs Naive RAG Demo")
    print("=" * 72)
    print()
    print("Filing corpus (four large-cap 10-Ks from SEC EDGAR):")
    for f in _FILINGS:
        print(f"  {f['name']:25s}  {f['company']}")
    print()

    # ── Fetch / load filings ─────────────────────────────────────────────────
    print("Loading filings...")
    extractor = EdgarExtractor(infer_bold_headings=True)

    naive_loader = DocumentLoader(
        chunk_size=args.chunk_size,
        table_chunk_limit=args.chunk_size,
        context_strategy=None,
        extra_extractors=[extractor],
    )
    ctx_loader = DocumentLoader(
        chunk_size=args.chunk_size,
        table_chunk_limit=args.chunk_size,
        context_strategy="prefix",
        extra_extractors=[extractor],
    )

    naive_all: list[DocumentChunk] = []
    ctx_all:   list[DocumentChunk] = []

    for filing in _FILINGS:
        doc_name, html_bytes = _fetch_or_load(filing, cache_dir)
        naive_chunks = naive_loader.load_bytes(html_bytes, name=doc_name, doc_type="edgar")
        ctx_chunks   = ctx_loader.load_bytes(html_bytes, name=doc_name, doc_type="edgar")
        naive_all.extend(naive_chunks)
        ctx_all.extend(ctx_chunks)
        print(
            f"  {doc_name}: {len(naive_chunks)} chunks  "
            f"({sum(len(c.content) for c in naive_chunks):,} chars extracted)"
        )

    print()
    print(f"Total: {len(naive_all)} chunks naive / {len(ctx_all)} chunks contextual")
    print()

    # ── Section distribution ──────────────────────────────────────────────────
    by_doc: dict[str, Counter] = defaultdict(Counter)
    for c in ctx_all:
        sec = (c.section or "").split(".")[0].strip() or "(none)"
        by_doc[c.document_name][sec] += 1

    print("Section distribution (contextual chunks):")
    for doc_name in sorted(by_doc):
        total = sum(by_doc[doc_name].values())
        print(f"  {doc_name}  ({total} chunks)")
        for sec, cnt in sorted(by_doc[doc_name].items()):
            print(f"    {cnt:4d}×  {sec}")
    print()

    # ── Embed ─────────────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    print("Encoding naive chunks...")
    naive_vecs = _encode(model, [c.content for c in naive_all])
    print("Encoding contextual chunks...")
    ctx_vecs = _encode(model, [c.embedding_content for c in ctx_all])
    print(f"  Embedding dim: {naive_vecs.shape[1]}")
    print()

    # ── Part 1: Retrieval — cohort coherence ─────────────────────────────────
    print("━" * 72)
    print(f"PART 1 — RETRIEVAL COHORT COHERENCE  [{args.model}]")
    print("━" * 72)
    print()
    print("Queries are company-agnostic. No 'correct' chunk is designated.")
    print()
    print("Metric: Mean Marginal Relevance (MMR)")
    print("  MMR = mean( sim(chunk, query) − mean_pairwise_sim_within_cohort )")
    print("  High MMR = cohort is relevant AND non-redundant — better for LLM synthesis.")
    print("  Naive cohort retrieved in naive space; contextual cohort in ctx space.")
    print("  Both evaluated in shared naive space for a fair apples-to-apples comparison.")
    print("  Also shown: section spread across the cohort.")
    print()

    k = args.top_k
    naive_coherence: list[float] = []
    ctx_coherence:   list[float] = []

    def _mean_marginal_relevance(idx: list[int], qv: np.ndarray, vecs: np.ndarray) -> float:
        """MMR: mean( sim(chunk, query) − mean_pairwise_sim_within_cohort ).
        High = cohort is relevant AND non-redundant — optimal for LLM synthesis.
        Pass naive_vecs for both cohorts for a fair shared-space comparison.
        """
        if len(idx) < 2:
            return float(vecs[idx[0]] @ qv) if idx else 0.0
        sub = vecs[idx]                              # (k, D), L2-normalised
        q_sim = sub @ qv                             # (k,) relevance scores
        pair = sub @ sub.T                           # (k, k) pairwise cosines
        k = len(idx)
        redundancy = (pair.sum(axis=1) - 1.0) / (k - 1)   # mean pairwise (excl. self)
        return float((q_sim - redundancy).mean())

    def _top_indices(query: str, vecs: np.ndarray, k: int) -> list[int]:
        qv = model.encode([query], normalize_embeddings=True)[0]
        scores = vecs @ qv
        return list(np.argsort(scores)[::-1][:k])

    def _sec_label(c: DocumentChunk) -> str:
        return (c.section or "").split(".")[0].strip() or "(none)"

    for query in QUERIES:
        naive_idx = _top_indices(query, naive_vecs, k)
        ctx_idx   = _top_indices(query, ctx_vecs,   k)

        # MMR evaluated in shared naive space — only chunk selection differs
        qv = model.encode([query], normalize_embeddings=True)[0]
        naive_mmr = _mean_marginal_relevance(naive_idx, qv, naive_vecs)
        ctx_mmr   = _mean_marginal_relevance(ctx_idx,   qv, naive_vecs)

        naive_coherence.append(naive_mmr)
        ctx_coherence.append(ctx_mmr)

        naive_chunks = [naive_all[i] for i in naive_idx]
        ctx_chunks   = [ctx_all[i]   for i in ctx_idx]

        print(f"Query: {query!r}")
        print()

        for label, chunks in [("Naive     ", naive_chunks), ("Contextual", ctx_chunks)]:
            sec_counts: dict[str, int] = {}
            for c in chunks:
                sec_counts[_sec_label(c)] = sec_counts.get(_sec_label(c), 0) + 1
            spread = "  ".join(f"{s}×{n}" for s, n in sorted(sec_counts.items(), key=lambda x: -x[1]))
            for rank, c in enumerate(chunks, 1):
                trunc = c.content[:60].replace("\n", " ")
                print(f"  {label} #{rank}  doc={c.document_name!r}  sec={c.section!r}")
                print(f"             content: {trunc!r}…")
            print(f"  {label} sections: {spread}")
            print()

        delta = ctx_mmr - naive_mmr
        if delta > 0.002:
            verdict = f"CONTEXTUAL higher MMR  Δ={delta:+.4f}"
        elif delta < -0.002:
            verdict = f"naive higher MMR       Δ={delta:+.4f}"
        else:
            verdict = f"tied                   Δ={delta:+.4f}"
        print(f"  MMR:  naive={naive_mmr:.4f}  contextual={ctx_mmr:.4f}  → {verdict}")
        print()

    n = len(QUERIES)
    naive_avg = sum(naive_coherence) / n
    ctx_avg   = sum(ctx_coherence)   / n
    print("─" * 72)
    print(f"  Mean MMR @{k}:")
    print(f"    naive={naive_avg:.4f}   contextual={ctx_avg:.4f}   Δ={ctx_avg - naive_avg:+.4f}")
    print(f"  Contextual higher MMR: {sum(c > v + 0.002 for c, v in zip(ctx_coherence, naive_coherence))}/{n} queries")
    print(f"  Naive higher MMR:      {sum(v > c + 0.002 for c, v in zip(ctx_coherence, naive_coherence))}/{n} queries")
    print(f"  Tied:                  {sum(abs(c-v) <= 0.002 for c, v in zip(ctx_coherence, naive_coherence))}/{n} queries")
    print()

    # ── Part 2: Clustering quality ────────────────────────────────────────────
    print("━" * 72)
    print("PART 2 — CLUSTERING QUALITY")
    print("━" * 72)
    print()
    print("Metric: mean cosine similarity between chunk pairs.")
    print("  Intra = same Item (e.g. all 'Item 1A' chunks)  (want HIGH)")
    print("  Inter = different Items                         (want LOW)")
    print("  Separation = intra − inter                     (want HIGH)")
    print()

    print("Computing pairwise similarities (may take a moment)...")
    MAX_PAIRS = 5000
    sample_n = min(len(naive_all), int(math.sqrt(MAX_PAIRS * 2)))
    if sample_n < len(naive_all):
        import random
        random.seed(42)
        idx = random.sample(range(len(naive_all)), sample_n)
        naive_sample = [naive_all[i] for i in idx]
        naive_vecs_s = naive_vecs[idx]
        ctx_sample   = [ctx_all[i]   for i in idx]
        ctx_vecs_s   = ctx_vecs[idx]
        print(f"  (Sampling {sample_n} of {len(naive_all)} chunks for O(n²) computation)")
    else:
        naive_sample, naive_vecs_s = naive_all, naive_vecs
        ctx_sample,   ctx_vecs_s   = ctx_all,   ctx_vecs

    naive_intra, naive_inter = _cluster_quality(naive_sample, naive_vecs_s)
    ctx_intra,   ctx_inter   = _cluster_quality(ctx_sample,   ctx_vecs_s)

    naive_sep = naive_intra - naive_inter
    ctx_sep   = ctx_intra   - ctx_inter

    col = 14
    print()
    print(f"  {'':20s}  {'Naive':>{col}}  {'Contextual':>{col}}  {'Δ (ctx−naive)':>{col}}")
    print(f"  {'─'*20}  {'─'*col}  {'─'*col}  {'─'*col}")
    print(f"  {'Intra-section':20s}  {naive_intra:>{col}.4f}  {ctx_intra:>{col}.4f}  {ctx_intra-naive_intra:>+{col}.4f}")
    print(f"  {'Inter-section':20s}  {naive_inter:>{col}.4f}  {ctx_inter:>{col}.4f}  {ctx_inter-naive_inter:>+{col}.4f}")
    print(f"  {'Separation':20s}  {naive_sep:>{col}.4f}  {ctx_sep:>{col}.4f}  {ctx_sep-naive_sep:>+{col}.4f}")
    print()

    if ctx_sep > naive_sep:
        improvement = (ctx_sep - naive_sep) / abs(naive_sep) * 100 if naive_sep else float("inf")
        print(f"  Contextual separation is {improvement:.1f}% better than naive.")
        print("  Thesis supported: document name + section path in embedding_content")
        print("  produces tighter within-section clusters and better cross-section")
        print("  separation on real-world EDGAR filings.")
    else:
        print("  Separation did not improve — inspect per-item breakdown below.")

    print()
    print("─" * 72)
    print("  Per-Item chunk distribution across all four companies:")
    print()
    all_secs: Counter = Counter()
    for c in ctx_all:
        sec = (c.section or "").split(".")[0].strip() or "(none)"
        all_secs[sec] += 1
    for sec, cnt in sorted(all_secs.items()):
        print(f"    {cnt:4d}×  {sec}")
    print()


if __name__ == "__main__":
    main()