"""
Contextual vs Naive RAG — demonstration of the chunkeymonkey thesis.

Loads three documents of different structures:
  - ops_report.md:      repeated section names across geographic regions
  - product_catalog.md: long product tables split across multiple chunks
  - incident_log.md:    long incident tables losing regional context at split points

Shows two effects:
  1. RETRIEVAL  — contextual embedding_content returns the right section;
                  naive content loses regional context in continuation chunks.
  2. CLUSTERING — contextual chunks from the same section are more similar
                  to each other than to chunks from other sections.

Requires only chunkeymonkey core (zero external dependencies).

Usage:
    cd /path/to/chunkeymonkey
    python demo/contextual_vs_naive.py
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from chunkeymonkey import DocumentLoader
from chunkeymonkey.models import DocumentChunk

_HERE = Path(__file__).parent
DOCS_DIR = _HERE / "docs"

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF vectoriser (pure Python, zero deps)
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


# ─────────────────────────────────────────────────────────────────────────────
# Clustering quality metric (no sklearn required)
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_quality(
    chunks: list[DocumentChunk],
    vecs: list[list[float]],
) -> tuple[float, float]:
    """Return (mean_intra_section_similarity, mean_inter_section_similarity).

    Intra = average cosine similarity between pairs of chunks in the SAME section.
    Inter = average cosine similarity between pairs of chunks in DIFFERENT sections.
    A good embedding has high intra and low inter (tight clusters, good separation).
    """
    by_section: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        label = (c.section or "").split(">")[0].strip() or "root"
        by_section[label].append(i)

    intra_sims, inter_sims = [], []

    indices = list(range(len(chunks)))
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
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
# Retrieval queries
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (query, expected_section_keywords, description_of_why_naive_fails)
QUERIES = [
    # ops_report.md — repeated section names; region only in parent heading
    (
        "EMEA sales headcount AEs SDRs CSMs",
        ["EMEA", "Sales", "Headcount"],
        "Headcount chunk content is just a table of role counts. 'EMEA' is not in the "
        "content — it was in the parent '# EMEA' heading which chunked separately.",
    ),
    (
        "APAC engineering P0 incidents MTTR DORA",
        ["APAC", "Engineering", "Performance"],
        "All Engineering Performance chunks contain 'Deployments', 'P0 incidents', 'MTTR'. "
        "None of the non-first Engineering chunks contain their region name.",
    ),
    (
        "Americas engineering headcount QA DevOps",
        ["Americas", "Engineering", "Headcount"],
        "Engineering Headcount chunk for Americas starts with '### Headcount' and a table. "
        "'Americas' only lives in the section breadcrumb, not the chunk text.",
    ),
    # product_catalog.md — long table split; continuation chunks lose heading context
    (
        "endpoint protection XDR products pricing",
        ["Endpoint", "Protection", "Product"],
        "Table continuation chunks contain only product rows (SKUs, prices). The heading "
        "'Endpoint Protection' is only in the first chunk and the section breadcrumb.",
    ),
    (
        "identity management PAM zero trust products",
        ["Identity", "Management", "Product"],
        "Identity Management product list is a separate table. SKU rows have no "
        "'Identity' or 'Management' text — only the section path does.",
    ),
    # incident_log.md — long incident table split across chunks
    (
        "EMEA P0 CDN auth service incidents 2024",
        ["EMEA", "P0"],
        "Incident table continuation chunks contain only ID/date/service rows. "
        "The region 'EMEA' is only in the '## EMEA' parent heading chunk and the section path.",
    ),
    (
        "APAC database auth search incidents outage",
        ["APAC", "P0"],
        "APAC incident continuation chunks have the same column structure as Americas and EMEA. "
        "Only the section breadcrumb 'APAC > P0 Incidents' distinguishes them.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    docs = sorted(DOCS_DIR.glob("*.md"))
    if not docs:
        raise FileNotFoundError(f"No .md files found in {DOCS_DIR}")

    print("=" * 72)
    print("chunkeymonkey — Contextual vs Naive RAG Demo")
    print("=" * 72)
    print(f"Loading {len(docs)} document(s) from {DOCS_DIR.name}/")
    for d in docs:
        print(f"  {d.name}")
    print()

    # Load with a small chunk_size so long tables definitely split
    naive_loader = DocumentLoader(chunk_size=600, table_chunk_limit=400, context_strategy=None)
    ctx_loader   = DocumentLoader(chunk_size=600, table_chunk_limit=400, context_strategy="prefix")

    naive_all: list[DocumentChunk] = []
    ctx_all:   list[DocumentChunk] = []
    for doc in docs:
        naive_all.extend(naive_loader.load(str(doc)))
        ctx_all.extend(ctx_loader.load(str(doc)))

    print(f"Total chunks: {len(naive_all)} naive / {len(ctx_all)} contextual")
    print()

    # ── Build vectors ────────────────────────────────────────────────────────
    naive_texts = [c.content for c in naive_all]
    ctx_texts   = [c.embedding_content for c in ctx_all]  # section-prefixed

    naive_vocab, naive_idf = _build(naive_texts)
    ctx_vocab,   ctx_idf   = _build(ctx_texts)

    naive_vecs = [_vec(t, naive_vocab, naive_idf) for t in naive_texts]
    ctx_vecs   = [_vec(t, ctx_vocab,   ctx_idf)   for t in ctx_texts]

    # ── Part 1: Retrieval ────────────────────────────────────────────────────
    print("━" * 72)
    print("PART 1 — RETRIEVAL")
    print("━" * 72)
    print()

    ctx_p1 = naive_p1 = 0  # precision at rank 1
    ctx_wins = naive_wins = both_rank1 = neither = 0

    def is_correct(chunk: DocumentChunk, kws: list[str]) -> bool:
        return all(k.lower() in (chunk.section or "").lower() for k in kws)

    for query, kws, reason in QUERIES:
        naive_results = _retrieve(query, naive_all, naive_vecs, naive_vocab, naive_idf)
        ctx_results   = _retrieve(query, ctx_all,   ctx_vecs,   ctx_vocab,   ctx_idf)

        naive_r1_ok = is_correct(naive_results[0][1], kws)
        ctx_r1_ok   = is_correct(ctx_results[0][1],   kws)

        # Score margin: gap between #1 and #2 scores (higher = more confident)
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

        naive_margin_str = f"rank-1 margin={naive_margin:+.3f}"
        ctx_margin_str   = f"rank-1 margin={ctx_margin:+.3f}"
        print(f"  Naive      rank-1 correct: {'YES' if naive_r1_ok else 'NO ':3s}  {naive_margin_str}")
        print(f"  Contextual rank-1 correct: {'YES' if ctx_r1_ok else 'NO ':3s}  {ctx_margin_str}")

        if ctx_r1_ok and not naive_r1_ok:
            verdict = "CONTEXTUAL WINS (rank-1)"
            ctx_wins += 1
        elif naive_r1_ok and not ctx_r1_ok:
            verdict = "naive wins (rank-1)"
            naive_wins += 1
        elif ctx_r1_ok and naive_r1_ok:
            if ctx_margin > naive_margin + 0.01:
                verdict = f"both rank-1 correct — contextual more confident (+{ctx_margin - naive_margin:.3f} margin)"
                ctx_wins += 1
            else:
                verdict = "both rank-1 correct — tied"
                both_rank1 += 1
        else:
            verdict = "neither rank-1 correct"
            neither += 1

        print(f"  → {verdict}")
        print()

        if ctx_r1_ok: ctx_p1 += 1
        if naive_r1_ok: naive_p1 += 1

    n = len(QUERIES)
    print("─" * 72)
    print(f"  Precision@1  naive={naive_p1}/{n}   contextual={ctx_p1}/{n}")
    print(f"  Contextual rank-1 wins (incl. margin): {ctx_wins}/{n}")
    print(f"  Naive rank-1 wins:                     {naive_wins}/{n}")
    print(f"  Tied at rank-1:                        {both_rank1}/{n}")
    print()

    # ── Part 2: Clustering quality ───────────────────────────────────────────
    print("━" * 72)
    print("PART 2 — CLUSTERING QUALITY")
    print("━" * 72)
    print()
    print("Metric: mean cosine similarity between chunk pairs.")
    print("  Intra = same top-level section  (want HIGH — tight clusters)")
    print("  Inter = different sections       (want LOW  — good separation)")
    print("  Separation = intra − inter       (want HIGH)")
    print()

    naive_intra, naive_inter = _cluster_quality(naive_all, naive_vecs)
    ctx_intra,   ctx_inter   = _cluster_quality(ctx_all,   ctx_vecs)

    naive_sep = naive_intra - naive_inter
    ctx_sep   = ctx_intra   - ctx_inter

    col = 14
    print(f"  {'':20s}  {'Naive':>{col}}  {'Contextual':>{col}}  {'Δ (ctx−naive)':>{col}}")
    print(f"  {'─'*20}  {'─'*col}  {'─'*col}  {'─'*col}")
    print(f"  {'Intra-section':20s}  {naive_intra:>{col}.4f}  {ctx_intra:>{col}.4f}  {ctx_intra-naive_intra:>+{col}.4f}")
    print(f"  {'Inter-section':20s}  {naive_inter:>{col}.4f}  {ctx_inter:>{col}.4f}  {ctx_inter-naive_inter:>+{col}.4f}")
    print(f"  {'Separation':20s}  {naive_sep:>{col}.4f}  {ctx_sep:>{col}.4f}  {ctx_sep-naive_sep:>+{col}.4f}")
    print()

    if ctx_sep > naive_sep:
        improvement = (ctx_sep - naive_sep) / abs(naive_sep) * 100 if naive_sep else float("inf")
        print(f"  Contextual separation is {improvement:.1f}% better than naive.")
        print("  Thesis supported: section breadcrumbs in embedding_content produce")
        print("  tighter within-section clusters and better cross-section separation.")
    else:
        print("  Results inconclusive on this corpus — try a larger document set.")
    print()

    # ── Per-document breakdown ───────────────────────────────────────────────
    print("─" * 72)
    print("  Per-document chunk count and section distribution:")
    print()
    by_doc: dict[str, list[DocumentChunk]] = defaultdict(list)
    for c in ctx_all:
        by_doc[c.document_name].append(c)
    for doc_name, chunks in sorted(by_doc.items()):
        top_sections = Counter(
            (c.section or "").split(">")[0].strip() for c in chunks
        )
        print(f"  {doc_name}  ({len(chunks)} chunks)")
        for sec, cnt in top_sections.most_common():
            print(f"    {cnt:3d}×  {sec!r}")
    print()


if __name__ == "__main__":
    main()
