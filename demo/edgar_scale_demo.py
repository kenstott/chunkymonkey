# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: fb7bd5bb-73d0-4da4-b758-8ee2afc3b3c6
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""
EDGAR Scale Demo — clustering separation at 4, 10, 25, 50 companies.

Validates the claim: contextual chunking's clustering advantage grows with
corpus size because boilerplate collision between companies increases.

At 4 companies naive retrieval can still distinguish filings by content.
At 50 companies the boilerplate is dense enough that the document/section
prefix in embedding_content becomes essential for cluster separation.

Usage::

    python demo/edgar_scale_demo.py
    python demo/edgar_scale_demo.py --cache-dir /tmp/edgar_cache --n 50

Requires: pip install sentence-transformers numpy
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

sys.path.insert(0, str(Path(__file__).parent.parent))
from chunkymonkey import DocumentLoader
from chunkymonkey.extractors import EdgarExtractor
from chunkymonkey.models import DocumentChunk

# ─────────────────────────────────────────────────────────────────────────────
# 50 S&P 500 tickers — diverse mix of sectors so boilerplate collision is
# the dominant source of similarity, not sector-specific vocabulary.
# ─────────────────────────────────────────────────────────────────────────────

TARGET_TICKERS = [
    # Tech / Software
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "CRM", "ADBE", "ORCL", "IBM",
    "CSCO", "INTC", "AMD", "QCOM", "TXN",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "V", "MA",
    # Healthcare / Pharma
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "ABT", "TMO", "UNH", "CVS", "HUM",
    # Consumer / Retail
    "WMT", "COST", "HD", "MCD", "KO", "PEP", "PG", "NKE", "SBUX", "TGT",
    # Industrials / Energy
    "CAT", "BA", "GE", "UPS", "CVX", "XOM", "NEE", "DIS", "NFLX", "TSLA",
]

_HEADERS = {
    "User-Agent": "chunkymonkey-demo/1.0 (research; contact@example.com)",
    "Accept-Encoding": "gzip, deflate",
}


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR fetch helpers
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


def _load_cik_map(cache_dir: Path) -> dict[str, int]:
    cache_file = cache_dir / "_tickers.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    print("  Fetching EDGAR company tickers list...")
    raw = _http_get("https://www.sec.gov/files/company_tickers.json")
    data = json.loads(raw)
    mapping = {v["ticker"]: v["cik_str"] for v in data.values()}
    cache_file.write_text(json.dumps(mapping))
    return mapping


def _get_latest_10k_url(cik: int) -> str:
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    data = json.loads(_http_get(url))
    filings = data.get("filings", {}).get("recent", {})
    forms    = filings.get("form", [])
    acc_nos  = filings.get("accessionNumber", [])
    docs     = filings.get("primaryDocument", [])
    for form, acc, doc in zip(forms, acc_nos, docs):
        if form == "10-K":
            acc_clean = acc.replace("-", "")
            return (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                f"{acc_clean}/{doc}"
            )
    raise RuntimeError(f"No 10-K for CIK {cik}")


def _fetch_filing(ticker: str, cik: int, cache_dir: Path) -> bytes | None:
    cache_file = cache_dir / f"{ticker}.htm"
    if cache_file.exists():
        return cache_file.read_bytes()
    try:
        url = _get_latest_10k_url(cik)
        data = _http_get(url)
        cache_file.write_bytes(data)
        time.sleep(0.5)   # EDGAR rate-limit courtesy
        return data
    except Exception as exc:
        print(f"    SKIP {ticker}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode(model, texts: list[str], desc: str) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def _clustering_separation(
    chunks: list[DocumentChunk],
    vecs: np.ndarray,
    sample: int = 300,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (intra, inter, separation) from a random sample of chunks."""
    rng = random.Random(seed)
    idx = rng.sample(range(len(chunks)), min(sample, len(chunks)))
    sub_chunks = [chunks[i] for i in idx]
    sub_vecs   = vecs[idx]

    def _key(c: DocumentChunk) -> str:
        return (c.section or "").split(".")[0].strip() or "root"

    sims = sub_vecs @ sub_vecs.T
    intra, inter = [], []
    for i in range(len(sub_chunks)):
        for j in range(i + 1, len(sub_chunks)):
            s = float(sims[i, j])
            if _key(sub_chunks[i]) == _key(sub_chunks[j]):
                intra.append(s)
            else:
                inter.append(s)
    i_mean = sum(intra) / len(intra) if intra else 0.0
    x_mean = sum(inter) / len(inter) if inter else 0.0
    return i_mean, x_mean, i_mean - x_mean


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDGAR clustering separation vs corpus size"
    )
    parser.add_argument("--cache-dir", default="/tmp/edgar_cache")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--n", type=int, default=50,
                        help="Max companies to load (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Chunky Monkey — EDGAR Clustering Separation vs Corpus Scale")
    print("=" * 72)
    print()

    # ── Resolve CIKs ─────────────────────────────────────────────────────────
    print("Resolving CIKs from EDGAR...")
    cik_map = _load_cik_map(cache_dir)
    tickers = [t for t in TARGET_TICKERS if t in cik_map][: args.n]
    missing = [t for t in TARGET_TICKERS[: args.n] if t not in cik_map]
    if missing:
        print(f"  Not found in EDGAR tickers: {missing}")
    print(f"  {len(tickers)} tickers resolved")
    print()

    # ── Fetch / load filings ─────────────────────────────────────────────────
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

    loaded: list[str] = []
    naive_all: list[DocumentChunk] = []
    ctx_all:   list[DocumentChunk] = []

    print(f"Loading up to {len(tickers)} 10-K filings...")
    for ticker in tickers:
        cik = cik_map[ticker]
        html = _fetch_filing(ticker, cik, cache_dir)
        if html is None:
            continue
        try:
            naive_chunks = naive_loader.load_bytes(html, name=ticker, doc_type="edgar")
            ctx_chunks   = ctx_loader.load_bytes(html,  name=ticker, doc_type="edgar")
        except Exception as exc:
            print(f"    SKIP {ticker} (parse error): {exc}")
            continue
        if not naive_chunks:
            print(f"    SKIP {ticker} (0 chunks)")
            continue
        naive_all.extend(naive_chunks)
        ctx_all.extend(ctx_chunks)
        loaded.append(ticker)
        status = "cache" if (cache_dir / f"{ticker}.htm").exists() else "fetch"
        print(f"  [{status}] {ticker:6s}  {len(naive_chunks):4d} chunks  "
              f"(total {len(naive_all):,})")

    print()
    print(f"Loaded {len(loaded)} companies / {len(naive_all):,} chunks")
    print()

    # ── Embed ─────────────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    vec_cache_naive = cache_dir / f"_vecs_naive_{args.model.replace('/', '_')}_{len(loaded)}.npy"
    vec_cache_ctx   = cache_dir / f"_vecs_ctx_{args.model.replace('/', '_')}_{len(loaded)}.npy"

    if vec_cache_naive.exists() and vec_cache_ctx.exists():
        print(f"Loading cached embeddings from {cache_dir}...")
        naive_vecs = np.load(str(vec_cache_naive))
        ctx_vecs   = np.load(str(vec_cache_ctx))
        print(f"  naive={naive_vecs.shape}  contextual={ctx_vecs.shape}")
    else:
        print(f"Loading model: {args.model}")
        model = SentenceTransformer(args.model)
        print("Encoding naive chunks...")
        naive_vecs = _encode(model, [c.content for c in naive_all], "naive")
        print("Encoding contextual chunks...")
        ctx_vecs   = _encode(model, [c.embedding_content for c in ctx_all], "contextual")
        np.save(str(vec_cache_naive), naive_vecs)
        np.save(str(vec_cache_ctx),   ctx_vecs)
        print(f"  Cached → {vec_cache_naive.name}, {vec_cache_ctx.name}")
    print()

    # ── Clustering at increasing corpus sizes ─────────────────────────────────
    print("━" * 72)
    print("CLUSTERING SEPARATION vs CORPUS SIZE")
    print("━" * 72)
    print()
    print("Metric: separation = intra-section similarity − inter-section similarity")
    print("  Higher separation → tighter, better-differentiated clusters.")
    print("  Claim: contextual advantage grows as corpus scales.")
    print()

    # Build per-company index for subsetting
    company_chunks: dict[str, list[int]] = {}
    for i, c in enumerate(naive_all):
        company_chunks.setdefault(c.document_name, []).append(i)

    scale_points = sorted({4, 10, 25, min(50, len(loaded)), len(loaded)} & set(range(1, len(loaded) + 1)))

    col = 10
    print(f"  {'N':>4}  {'naive intra':>{col}}  {'naive inter':>{col}}  "
          f"{'naive sep':>{col}}  {'ctx sep':>{col}}  {'Δ sep':>{col}}  {'ctx/naive':>{col}}")
    print(f"  {'─'*4}  {'─'*col}  {'─'*col}  {'─'*col}  {'─'*col}  {'─'*col}  {'─'*col}")

    rng = random.Random(args.seed)
    companies = list(company_chunks.keys())

    for n in scale_points:
        sample_companies = companies[:n]   # deterministic prefix, not random
        idx = [i for co in sample_companies for i in company_chunks[co]]

        sub_naive  = [naive_all[i] for i in idx]
        sub_ctx    = [ctx_all[i]   for i in idx]
        sub_nvecs  = naive_vecs[idx]
        sub_cvecs  = ctx_vecs[idx]

        ni, nx, ns = _clustering_separation(sub_naive, sub_nvecs)
        ci, cx, cs = _clustering_separation(sub_ctx,   sub_cvecs)
        delta = cs - ns
        ratio = cs / ns if ns else float("inf")

        print(f"  {n:>4}  {ni:>{col}.4f}  {nx:>{col}.4f}  "
              f"{ns:>{col}.4f}  {cs:>{col}.4f}  {delta:>+{col}.4f}  {ratio:>{col}.2f}×")

    print()
    print("  Interpretation:")
    print("  • If ctx/naive ratio grows with N → contextual advantage scales with corpus size.")
    print("  • If ratio is flat → advantage is constant regardless of scale.")
    print()

    # ── Full-corpus clustering summary ───────────────────────────────────────
    print("━" * 72)
    print(f"FULL CORPUS CLUSTERING ({len(loaded)} companies, {len(naive_all):,} chunks)")
    print("━" * 72)
    print()
    ni, nx, ns = _clustering_separation(naive_all, naive_vecs, sample=500)
    ci, cx, cs = _clustering_separation(ctx_all,   ctx_vecs,   sample=500)

    col = 14
    print(f"  {'':20s}  {'Naive':>{col}}  {'Contextual':>{col}}  {'Δ':>{col}}")
    print(f"  {'─'*20}  {'─'*col}  {'─'*col}  {'─'*col}")
    print(f"  {'Intra-section':20s}  {ni:>{col}.4f}  {ci:>{col}.4f}  {ci-ni:>+{col}.4f}")
    print(f"  {'Inter-section':20s}  {nx:>{col}.4f}  {cx:>{col}.4f}  {cx-nx:>+{col}.4f}")
    print(f"  {'Separation':20s}  {ns:>{col}.4f}  {cs:>{col}.4f}  {cs-ns:>+{col}.4f}")
    print()
    if cs > ns:
        pct = (cs - ns) / abs(ns) * 100
        print(f"  Contextual separation is {pct:.1f}% better than naive at {len(loaded)} companies.")
    else:
        print("  Naive separation >= contextual at this scale.")
    print()

    # ── Cohort quality: mean marginal relevance ───────────────────────────────
    from sentence_transformers import SentenceTransformer
    if "model" not in dir():
        model = SentenceTransformer(args.model)

    K = 12   # cohort size — enough to span multiple companies per topic

    print("━" * 72)
    print(f"COHORT QUALITY — mean marginal relevance @{K}, {len(loaded)} companies")
    print("━" * 72)
    print()
    print("For each topic query, retrieve top-k chunks and compute mean marginal")
    print("relevance across the cohort:")
    print()
    print("  MMR(chunk) = sim(chunk, query) − mean(sim(chunk, other_chunks))")
    print()
    print("High MMR = every chunk is query-relevant AND adds new information.")
    print("Low MMR  = near-duplicate results (high relevance, high redundancy).")
    print("Both cohorts evaluated in the shared naive space — only the chunk")
    print("selection differs between approaches.")
    print()

    TOPIC_QUERIES = [
        "foreign exchange rate currency exposure hedging",
        "goodwill impairment testing reporting unit fair value",
        "remaining performance obligations deferred revenue subscription",
        "cybersecurity risk management incident response board oversight",
        "stock repurchase share buyback capital return dividends",
        "income tax effective rate uncertain tax positions valuation allowance",
        "disclosure controls procedures evaluation effectiveness management",
        "competition market position pricing pressure customer retention",
        "operating lease right-of-use asset liability",
        "revenue recognition performance obligation contract customer",
    ]

    def _top_k_idx(query: str, vecs: np.ndarray, k: int) -> list[int]:
        qv = model.encode([query], normalize_embeddings=True)[0]
        return list(np.argsort(vecs @ qv)[::-1][:k])

    def _mean_marginal_relevance(
        idx: list[int],
        query_vec: np.ndarray,
        ref_vecs: np.ndarray,
    ) -> float:
        """Mean per-chunk marginal relevance, evaluated in a shared reference space.

        marginal_value(chunk_i) = sim(chunk_i, query)
                                  − mean(sim(chunk_i, other_chunks))

        Both similarities computed against ref_vecs (the naive space) so
        naive and contextual cohorts are compared on identical ground.
        """
        if len(idx) < 2:
            return float(ref_vecs[idx[0]] @ query_vec) if idx else 0.0
        sub   = ref_vecs[idx]                   # (k, D) — always naive space
        q_sim = sub @ query_vec                 # (k,)
        pair  = sub @ sub.T                     # (k, k)
        k     = len(idx)
        redundancy = (pair.sum(axis=1) - 1.0) / (k - 1)
        return float((q_sim - redundancy).mean())

    naive_mmr_all: list[float] = []
    ctx_mmr_all:   list[float] = []

    col = 7
    print(f"  {'Query':<48}  {'naive':>{col}}  {'ctx':>{col}}  {'Δ':>{col}}  winner")
    print(f"  {'─'*48}  {'─'*col}  {'─'*col}  {'─'*col}  {'─'*6}")

    for query in TOPIC_QUERIES:
        qv = model.encode([query], normalize_embeddings=True)[0]

        # Retrieve using each approach's own space
        n_idx = _top_k_idx(query, naive_vecs, K)
        c_idx = _top_k_idx(query, ctx_vecs,   K)

        # Evaluate both cohorts in the shared naive space
        n_mmr = _mean_marginal_relevance(n_idx, qv, naive_vecs)
        c_mmr = _mean_marginal_relevance(c_idx, qv, naive_vecs)
        naive_mmr_all.append(n_mmr)
        ctx_mmr_all.append(c_mmr)

        delta = c_mmr - n_mmr
        winner = "CTX  " if delta > 0.005 else ("naive" if delta < -0.005 else "tied ")
        print(f"  {query[:48]:<48}  {n_mmr:>{col}.3f}  {c_mmr:>{col}.3f}  {delta:>+{col}.3f}  {winner}")

    print()
    n_avg = sum(naive_mmr_all) / len(naive_mmr_all)
    c_avg = sum(ctx_mmr_all)   / len(ctx_mmr_all)
    ctx_wins   = sum(c > v + 0.005 for c, v in zip(ctx_mmr_all, naive_mmr_all))
    naive_wins = sum(v > c + 0.005 for c, v in zip(ctx_mmr_all, naive_mmr_all))
    tied       = len(TOPIC_QUERIES) - ctx_wins - naive_wins
    print(f"  Mean marginal relevance @{K}:")
    print(f"    naive={n_avg:.3f}   contextual={c_avg:.3f}   Δ={c_avg - n_avg:+.3f}")
    print(f"  Contextual higher MMR: {ctx_wins}/{len(TOPIC_QUERIES)}")
    print(f"  Naive higher MMR:      {naive_wins}/{len(TOPIC_QUERIES)}")
    print(f"  Tied:                  {tied}/{len(TOPIC_QUERIES)}")
    print()


if __name__ == "__main__":
    main()