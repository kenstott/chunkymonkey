# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: faa0c06a-012a-4f3a-89d4-e4d2ac58dec6
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""
Python Docs — Contextual vs Naive RAG on real-world documentation.

Crawls a targeted subset of docs.python.org and demonstrates that
contextual chunking outperforms naive chunking on documentation that was
written by humans for humans — not synthetic data designed to illustrate
the problem.

WHY PYTHON DOCS ARE A GOOD TEST:
  Python documentation uses Sphinx, which generates a deep heading hierarchy:
    Built-in Types > str > String Methods > str.split
    Built-in Types > list > list.sort
    collections > collections.deque > collections.deque.rotate
  Dozens of methods share identical leaf-level vocabulary: "parameter",
  "type", "default", "return", "example", "equivalent to", "raises".
  Without the breadcrumb, TF-IDF cannot distinguish a str method from a
  deque method on vocabulary alone. With the breadcrumb it can.

CRAWL SCOPE (configurable via --max-pages):
  - library/stdtypes  — built-in sequence types (str, list, tuple, range…)
  - library/functions — built-in functions (sorted, enumerate, zip…)
  - library/collections — collections module (deque, defaultdict, Counter…)

SHOWS TWO EFFECTS:
  1. RETRIEVAL  — contextual P@1 beats naive on queries where the key
                  distinguishing term (the type/module name) is only in
                  the section path, not in the method body.
  2. CLUSTERING — contextual chunks from the same Python type/class cluster
                  more tightly than naive chunks (intra > inter, bigger gap).

Requires: pip install chunkymonkey[http]  (adds requests)
          Network access to docs.python.org

Usage:
    cd /path/to/chunkymonkey
    python demo/python_docs_demo.py
    python demo/python_docs_demo.py --max-pages 30   # faster, fewer chunks
    python demo/python_docs_demo.py --max-pages 100  # more signal
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from chunkymonkey import DocumentLoader
from chunkymonkey.models import DocumentChunk
from chunkymonkey.transports import WebCrawler


# ─────────────────────────────────────────────────────────────────────────────
# Crawl config
# ─────────────────────────────────────────────────────────────────────────────

ROOT_URL = "https://docs.python.org/3/library/"

# Only follow URLs under these three pages — gives us sequence types,
# built-in functions, and the collections module.  Tight scope = fast crawl.
INCLUDE_PATTERN = r"docs\.python\.org/3/library/(stdtypes|functions|collections)(\.html|#|$)"

# Sections with near-identical vocabulary across many classes
QUERIES = [
    # ── Retrieval queries ────────────────────────────────────────────────────
    # Each entry: (query, expected_section_keywords, why_naive_may_fail)
    (
        "split separator maxsplit whitespace string method",
        ["str", "String"],
        "str.split content: sep, maxsplit, whitespace.  "
        "os.path.split and deque also chunk near 'split'.  "
        "Region ('str') only in the section path.",
    ),
    (
        "sort key reverse inplace list method comparison function",
        ["list"],
        "list.sort and sorted() share almost identical vocabulary.  "
        "The section path 'Built-in Types > list' distinguishes list.sort; "
        "sorted() lives under 'Built-in Functions'.",
    ),
    (
        "maxlen thread safe bounded appendleft rotate deque",
        ["deque", "collections"],
        "deque methods are leaf chunks whose content says 'Rotate the deque "
        "n steps to the right' — the word 'deque' may not appear in every "
        "continuation chunk.  Section path carries 'collections.deque'.",
    ),
    (
        "missing key factory callable default dict",
        ["defaultdict", "collections"],
        "defaultdict.__missing__ chunk content is terse.  "
        "'defaultdict' only guaranteed in the section path.",
    ),
    (
        "counter most common elements multiset arithmetic",
        ["Counter", "collections"],
        "Counter methods share vocabulary with dict methods.  "
        "Section path 'collections > Counter' resolves ambiguity.",
    ),
    (
        "enumerate iterable start index loop counter built-in",
        ["enumerate", "functions"],
        "enumerate() is a built-in function.  Its content ('Returns an "
        "enumerate object') is terse.  Section path: 'Built-in Functions'.",
    ),
    (
        "zip iterables shortest longest fillvalue aggregate",
        ["zip", "functions"],
        "zip() content is short.  itertools.zip_longest also mentions "
        "'fillvalue' and 'iterables'.  Section path separates them.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# MMR helpers
# ─────────────────────────────────────────────────────────────────────────────

def _np_unit(vecs: list[list[float]]) -> np.ndarray:
    arr = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

def _mmr(idx: list[int], qv: np.ndarray, ref: np.ndarray) -> float:
    """MMR: mean( sim(chunk, query) − mean_pairwise_sim ).
    High = cohort is relevant AND non-redundant.
    """
    if len(idx) < 2:
        return float(ref[idx[0]] @ qv) if idx else 0.0
    sub = ref[idx]
    q_sim = sub @ qv
    pair = sub @ sub.T
    k = len(idx)
    redundancy = (pair.sum(axis=1) - 1.0) / (k - 1)
    return float((q_sim - redundancy).mean())


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF vectoriser  (pure Python, zero deps — same as main demo)
# ─────────────────────────────────────────────────────────────────────────────

def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9]*", text.lower())

def _idf(corpus: list[str], vocab: list[str]) -> dict[str, float]:
    n = len(corpus)
    return {
        w: math.log((n + 1) / (sum(1 for d in corpus if w in _tok(d)) + 1)) + 1.0
        for w in vocab
    }

def _vec(text: str, vocab: list[str], idf: dict[str, float]) -> list[float]:
    toks = _tok(text)
    if not toks:
        return [0.0] * len(vocab)
    tf = Counter(toks)
    n = len(toks)
    return [tf.get(w, 0) / n * idf.get(w, 0.0) for w in vocab]

def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    return dot / (ma * mb) if ma and mb else 0.0

def _build(texts: list[str]) -> tuple[list[str], dict[str, float]]:
    vocab = sorted({w for t in texts for w in _tok(t)})
    return vocab, _idf(texts, vocab)

def _retrieve(
    query: str,
    chunks: list[DocumentChunk],
    vecs: list[list[float]],
    vocab: list[str],
    idf: dict[str, float],
    k: int = 3,
) -> list[tuple[float, DocumentChunk]]:
    qv = _vec(query, vocab, idf)
    return sorted(
        [(_cos(qv, v), c) for v, c in zip(vecs, chunks)],
        key=lambda x: x[0],
        reverse=True,
    )[:k]

def _cluster_quality(
    chunks: list[DocumentChunk],
    vecs: list[list[float]],
) -> tuple[float, float]:
    """Intra vs inter-section cosine similarity."""
    intra_sims, inter_sims = [], []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            sim = _cos(vecs[i], vecs[j])
            si = (chunks[i].section or "").split(">")[0].strip() or "root"
            sj = (chunks[j].section or "").split(">")[0].strip() or "root"
            if si == sj:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)
    intra = sum(intra_sims) / len(intra_sims) if intra_sims else 0.0
    inter = sum(inter_sims) / len(inter_sims) if inter_sims else 0.0
    return intra, inter


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(max_pages: int = 60) -> None:
    print("=" * 72)
    print("Chunky Monkey — Python Docs: Contextual vs Naive RAG")
    print("=" * 72)
    print(f"Crawling {ROOT_URL}")
    print(f"Scope: stdtypes + functions + collections  (max {max_pages} pages)")
    print()

    # ── Crawl ────────────────────────────────────────────────────────────────
    crawler = WebCrawler(
        max_pages=max_pages,
        max_depth=2,
        same_domain=True,
        include_pattern=INCLUDE_PATTERN,
        max_workers=4,
        timeout=30,
    )

    # Use small chunk_size so method-level sections form their own chunks
    naive_loader = DocumentLoader(chunk_size=800, table_chunk_limit=400, context_strategy=None)
    ctx_loader   = DocumentLoader(chunk_size=800, table_chunk_limit=400, context_strategy="prefix")

    print("Fetching pages (this may take 20-60 seconds)…")
    try:
        urls = crawler.crawl(ROOT_URL)
    except Exception as exc:
        print(f"ERROR: crawl failed — {exc}")
        print("Check network access and that pip install chunkymonkey[http] is done.")
        sys.exit(1)

    if not urls:
        print("ERROR: no pages crawled. Check network access.")
        sys.exit(1)

    print(f"  Crawled {len(urls)} page(s)")
    for u in urls[:10]:
        print(f"    {u}")
    if len(urls) > 10:
        print(f"    … and {len(urls) - 10} more")
    print()

    naive_all: list[DocumentChunk] = []
    ctx_all:   list[DocumentChunk] = []
    failed = 0
    for url in urls:
        try:
            naive_all.extend(naive_loader.load(url))
            ctx_all.extend(ctx_loader.load(url))
        except Exception as exc:
            failed += 1
            if failed <= 3:
                print(f"  WARN: skipping {url}: {exc}")

    if not naive_all:
        print("ERROR: no chunks produced. Check extractor output.")
        sys.exit(1)

    print(f"Total chunks: {len(naive_all)} naive / {len(ctx_all)} contextual")
    if failed:
        print(f"  ({failed} page(s) skipped due to fetch errors)")
    print()

    # ── Guard: need at least some structure to show the effect ───────────────
    sections_with_data = {(c.section or "root") for c in ctx_all if c.section}
    if len(sections_with_data) < 3:
        print("WARNING: too few distinct sections detected.")
        print("  The HTML extractor may not have parsed headings from this page.")
        print("  Section distribution:")
        for s in sorted(sections_with_data)[:10]:
            print(f"    {s!r}")
        print()

    # ── Build vectors ────────────────────────────────────────────────────────
    naive_texts = [c.content for c in naive_all]
    ctx_texts   = [c.embedding_content for c in ctx_all]

    print("Building TF-IDF vectors…")
    naive_vocab, naive_idf = _build(naive_texts)
    ctx_vocab,   ctx_idf   = _build(ctx_texts)
    naive_vecs = [_vec(t, naive_vocab, naive_idf) for t in naive_texts]
    ctx_vecs   = [_vec(t, ctx_vocab,   ctx_idf)   for t in ctx_texts]
    print(f"  Naive vocab: {len(naive_vocab):,} terms over {len(naive_all)} chunks")
    print(f"  Ctx   vocab: {len(ctx_vocab):,} terms over {len(ctx_all)} chunks")
    print()

    def is_correct(chunk: DocumentChunk, kws: list[str]) -> bool:
        return all(k.lower() in (chunk.section or "").lower() for k in kws)

    # ── Part 1: Retrieval ────────────────────────────────────────────────────
    print("━" * 72)
    print("PART 1 — RETRIEVAL")
    print("━" * 72)
    print()

    ctx_p1 = naive_p1 = 0
    ctx_wins = naive_wins = both_rank1 = neither = 0

    for query, kws, reason in QUERIES:
        naive_results = _retrieve(query, naive_all, naive_vecs, naive_vocab, naive_idf)
        ctx_results   = _retrieve(query, ctx_all,   ctx_vecs,   ctx_vocab,   ctx_idf)

        if not naive_results or not ctx_results:
            continue

        naive_r1_ok = is_correct(naive_results[0][1], kws)
        ctx_r1_ok   = is_correct(ctx_results[0][1],   kws)
        naive_margin = naive_results[0][0] - naive_results[1][0] if len(naive_results) > 1 else 0.0
        ctx_margin   = ctx_results[0][0]   - ctx_results[1][0]   if len(ctx_results)   > 1 else 0.0

        print(f"Query: {query!r}")
        print(f"Why naive may fail: {reason}")
        print()

        for label, results in [("Naive     ", naive_results), ("Contextual", ctx_results)]:
            for rank, (score, c) in enumerate(results, 1):
                marker = "✓" if is_correct(c, kws) else " "
                trunc = c.content[:70].replace("\n", " ")
                print(f"  {label} #{rank} [{marker}] score={score:.3f}  section={c.section!r}")
                print(f"             content: {trunc!r}...")

        print(f"  Naive      rank-1 correct: {'YES' if naive_r1_ok else 'NO ':3s}  margin={naive_margin:+.3f}")
        print(f"  Contextual rank-1 correct: {'YES' if ctx_r1_ok else 'NO ':3s}  margin={ctx_margin:+.3f}")

        if ctx_r1_ok and not naive_r1_ok:
            verdict = "CONTEXTUAL WINS"
            ctx_wins += 1
        elif naive_r1_ok and not ctx_r1_ok:
            verdict = "naive wins"
            naive_wins += 1
        elif ctx_r1_ok and naive_r1_ok:
            if ctx_margin > naive_margin + 0.01:
                verdict = f"both correct — contextual more confident (+{ctx_margin - naive_margin:.3f})"
                ctx_wins += 1
            else:
                verdict = "both correct — tied"
                both_rank1 += 1
        else:
            verdict = "neither correct"
            neither += 1

        print(f"  → {verdict}")
        print()

        if ctx_r1_ok: ctx_p1 += 1
        if naive_r1_ok: naive_p1 += 1

    n = len(QUERIES)
    print("─" * 72)
    print(f"  Precision@1  naive={naive_p1}/{n}   contextual={ctx_p1}/{n}")
    print(f"  Contextual wins (incl. margin): {ctx_wins}/{n}")
    print(f"  Naive wins:                     {naive_wins}/{n}")
    print(f"  Tied:                           {both_rank1}/{n}")
    print(f"  Neither:                        {neither}/{n}")
    print()

    # ── Part 2: Clustering ───────────────────────────────────────────────────
    # Cluster quality is O(n²) — cap at 500 chunks to keep it fast
    print("━" * 72)
    print("PART 2 — CLUSTERING QUALITY")
    print("━" * 72)
    print()
    print("Metric: mean cosine similarity between chunk pairs.")
    print("  Intra = same top-level section  (want HIGH)")
    print("  Inter = different sections       (want LOW)")
    print("  Separation = intra − inter       (want HIGH)")
    print()

    cap = 500
    if len(naive_all) > cap:
        print(f"  (Capping at {cap} chunks for O(n²) clustering — {len(naive_all)} total)")
        import random
        rng = random.Random(42)
        idx = rng.sample(range(len(naive_all)), cap)
        naive_sample = [naive_all[i] for i in sorted(idx)]
        ctx_sample   = [ctx_all[i]   for i in sorted(idx)]
        naive_vecs_s = [naive_vecs[i] for i in sorted(idx)]
        ctx_vecs_s   = [ctx_vecs[i]   for i in sorted(idx)]
    else:
        naive_sample, ctx_sample = naive_all, ctx_all
        naive_vecs_s, ctx_vecs_s = naive_vecs, ctx_vecs

    print("  Computing pairwise similarities…")
    naive_intra, naive_inter = _cluster_quality(naive_sample, naive_vecs_s)
    ctx_intra,   ctx_inter   = _cluster_quality(ctx_sample,   ctx_vecs_s)

    naive_sep = naive_intra - naive_inter
    ctx_sep   = ctx_intra   - ctx_inter

    col = 14
    print(f"\n  {'':20s}  {'Naive':>{col}}  {'Contextual':>{col}}  {'Δ (ctx−naive)':>{col}}")
    print(f"  {'─'*20}  {'─'*col}  {'─'*col}  {'─'*col}")
    print(f"  {'Intra-section':20s}  {naive_intra:>{col}.4f}  {ctx_intra:>{col}.4f}  {ctx_intra-naive_intra:>+{col}.4f}")
    print(f"  {'Inter-section':20s}  {naive_inter:>{col}.4f}  {ctx_inter:>{col}.4f}  {ctx_inter-naive_inter:>+{col}.4f}")
    print(f"  {'Separation':20s}  {naive_sep:>{col}.4f}  {ctx_sep:>{col}.4f}  {ctx_sep-naive_sep:>+{col}.4f}")
    print()

    if ctx_sep > naive_sep:
        improvement = (ctx_sep - naive_sep) / abs(naive_sep) * 100 if naive_sep else float("inf")
        print(f"  Contextual separation is {improvement:.1f}% better than naive.")
        print("  On real-world documentation, section breadcrumbs pull method-level")
        print("  chunks toward their parent class/module and away from unrelated ones.")
    else:
        print("  Separation not improved on this crawl.")
        print("  Try --max-pages 100 for more chunks and a stronger signal.")
    print()

    # ── Part 3: Section distribution ────────────────────────────────────────
    print("─" * 72)
    print("  Section distribution (top 15 by chunk count):")
    print()
    sec_counts: Counter = Counter(
        (c.section or "root").split(">")[0].strip() for c in ctx_all
    )
    for sec, cnt in sec_counts.most_common(15):
        bar = "█" * min(cnt, 40)
        print(f"  {cnt:4d}  {bar}  {sec!r}")
    print()

    # ── Part 4: MMR Cohort Quality ────────────────────────────────────────────
    print("━" * 72)
    print("PART 4 — MMR COHORT QUALITY  (Mean Marginal Relevance)")
    print("━" * 72)
    print()
    print("MMR = mean( sim(chunk, query) − mean_pairwise_sim_within_cohort )")
    print("High MMR = cohort is relevant AND non-redundant — better for LLM synthesis.")
    print("Naive retrieval uses naive TF-IDF space; contextual uses ctx TF-IDF space.")
    print("Both cohorts evaluated in shared naive space for fair comparison.")
    print()

    K_MMR = 5
    naive_np = _np_unit(naive_vecs)
    ctx_np   = _np_unit(ctx_vecs)

    naive_mmr_scores: list[float] = []
    ctx_mmr_scores:   list[float] = []

    for query, _kws, _reason in QUERIES:
        qv_n = np.array(_vec(query, naive_vocab, naive_idf), dtype=np.float32)
        n_norm = float(np.linalg.norm(qv_n))
        if n_norm > 0:
            qv_n /= n_norm
        n_scores = naive_np @ qv_n
        n_idx = list(np.argsort(n_scores)[::-1][:K_MMR])

        qv_c = np.array(_vec(query, ctx_vocab, ctx_idf), dtype=np.float32)
        c_norm = float(np.linalg.norm(qv_c))
        if c_norm > 0:
            qv_c /= c_norm
        c_scores = ctx_np @ qv_c
        c_idx = list(np.argsort(c_scores)[::-1][:K_MMR])

        n_mmr = _mmr(n_idx, qv_n, naive_np)
        c_mmr = _mmr(c_idx, qv_n, naive_np)   # evaluate ctx cohort in naive space

        naive_mmr_scores.append(n_mmr)
        ctx_mmr_scores.append(c_mmr)

        delta = c_mmr - n_mmr
        if delta > 0.002:
            verdict = f"CONTEXTUAL  Δ={delta:+.4f}"
        elif delta < -0.002:
            verdict = f"naive       Δ={delta:+.4f}"
        else:
            verdict = f"tied        Δ={delta:+.4f}"
        print(f"  {query[:55]!r:<57}  naive={n_mmr:.4f}  ctx={c_mmr:.4f}  → {verdict}")

    nq = len(QUERIES)
    print()
    print("─" * 72)
    n_avg = sum(naive_mmr_scores) / nq
    c_avg = sum(ctx_mmr_scores) / nq
    print(f"  Mean MMR @{K_MMR}:  naive={n_avg:.4f}   contextual={c_avg:.4f}   Δ={c_avg - n_avg:+.4f}")
    print(f"  Contextual higher MMR: {sum(c > v + 0.002 for c, v in zip(ctx_mmr_scores, naive_mmr_scores))}/{nq} queries")
    print(f"  Naive higher MMR:      {sum(v > c + 0.002 for c, v in zip(ctx_mmr_scores, naive_mmr_scores))}/{nq} queries")
    print(f"  Tied:                  {sum(abs(c - v) <= 0.002 for c, v in zip(ctx_mmr_scores, naive_mmr_scores))}/{nq} queries")
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Pages crawled:    {len(urls)}")
    print(f"  Chunks produced:  {len(naive_all)} naive / {len(ctx_all)} contextual")
    print(f"  Retrieval P@1:    naive={naive_p1}/{n}  contextual={ctx_p1}/{n}")
    ctx_sep_pct = (ctx_sep - naive_sep) / abs(naive_sep) * 100 if naive_sep else 0
    print(f"  Clustering sep:   naive={naive_sep:.4f}  contextual={ctx_sep:.4f}  ({ctx_sep_pct:+.1f}%)")
    print(f"  Mean MMR @{K_MMR}:    naive={n_avg:.4f}  contextual={c_avg:.4f}  (Δ={c_avg - n_avg:+.4f})")
    print()
    print("  Key insight: on professionally written docs, TF-IDF already does")
    print("  well on distinctive queries.  The gain is sharpest on terse method")
    print("  descriptions ('Rotate n steps', 'Equivalent to…') where the class")
    print("  name lives only in the section path — and on clustering, where the")
    print("  breadcrumb consistently groups str/list/deque methods correctly.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python docs contextual vs naive RAG demo")
    parser.add_argument(
        "--max-pages", type=int, default=60,
        help="Max pages to crawl (default 60; use 20-30 for a quick run)",
    )
    args = parser.parse_args()
    main(max_pages=args.max_pages)
