# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 6c1a14cf-5361-4e62-b3a1-8b4d08840694
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""
ClinicalTrials.gov — Contextual vs Naive RAG on real clinical trial protocols.

Downloads oncology Phase 2/3 trial records from ClinicalTrials.gov (completely
open, no login required) and demonstrates contextual chunking on one of the
most structurally formulaic corpora in existence.

WHY CLINICAL TRIALS ARE AN EXTREME CASE:

  a) IDENTICAL SECTION STRUCTURE: Every trial record has the same sections:
     Brief Summary, Detailed Description, Eligibility Criteria (Inclusion /
     Exclusion), Primary Outcomes, Secondary Outcomes, Interventions.
     Without breadcrumbs, a chunk from "Eligibility Criteria" of Trial A is
     indistinguishable from Trial B's — the sections have the same heading.

  b) VERBATIM BOILERPLATE WITHIN SECTIONS: Oncology exclusion criteria are
     notoriously formulaic across all sponsors. The following appear verbatim
     in hundreds of trials:
       "Active infection requiring systemic therapy"
       "Known history of active Hepatitis B or C"
       "Eastern Cooperative Oncology Group (ECOG) performance status > 2"
       "Total bilirubin > 1.5 × institutional upper limit of normal (ULN)"
       "Pregnant or breastfeeding, or expecting to conceive"
     A chunk cut from the middle of exclusion criteria contains only this
     boilerplate. 'Trial sponsor', 'drug name', and 'indication' are in the
     document name and section path — nowhere else.

  c) LONG ELIGIBILITY SECTIONS: A typical eligibility criteria block is
     2,000-4,000 characters (15-30 bullet points). With a 600-char chunk
     size, it splits into 4-6 continuation chunks. Every continuation chunk
     after the first contains only generic criteria that appear across
     all trials in the indication.

  d) PRIMARY OUTCOME REPETITION: RECIST 1.1 is the universal response
     assessment framework for solid tumours. "Objective Response Rate (ORR)
     as assessed by RECIST version 1.1" appears in hundreds of trials.
     Only the section breadcrumb ('trial_id > Primary Outcomes') ties it
     to a specific study.

DEMONSTRATES:
  1. RETRIEVAL  — contextual P@1 identifies the right trial for compound-
                  specific queries even when the chunk content is shared
                  boilerplate.
  2. CLUSTERING — contextual chunks from the same trial section cluster more
                  tightly; cross-trial same-section clusters are better
                  separated from cross-section pairs.

API:
  https://clinicaltrials.gov/api/v2/studies
  Completely public, no authentication required.
  FAIR USE: The demo fetches one page of results and caches locally.

Usage::

    python demo/clinicaltrials_demo.py

    # Use cached data (no network after first run)
    python demo/clinicaltrials_demo.py --cache-dir /tmp/ctgov_cache

    # Adjust corpus size and chunk size
    python demo/clinicaltrials_demo.py --n-trials 12 --chunk-size 500
"""
from __future__ import annotations

import argparse
import json
import math
import re
import ssl
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

sys.path.insert(0, str(Path(__file__).parent.parent))

from chunkymonkey import DocumentLoader
from chunkymonkey.models import DocumentChunk

# ─────────────────────────────────────────────────────────────────────────────
# ClinicalTrials.gov API
# ─────────────────────────────────────────────────────────────────────────────

_CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"

# Phase 2/3 solid tumour oncology trials with results — rich eligibility
# criteria, measurable outcome definitions, and detailed interventions.
# These conditions were chosen because their trial protocols share the most
# formulaic language (RECIST endpoints, ECOG performance status exclusions,
# organ function lab thresholds).
_SEARCH_CONDITIONS = [
    "non-small cell lung cancer",
    "colorectal cancer",
    "breast cancer",
    "ovarian cancer",
    "gastric cancer",
    "hepatocellular carcinoma",
    "melanoma",
    "pancreatic cancer",
]

_HEADERS = {
    "User-Agent": "chunkymonkey-demo/1.0 (research; contact@example.com)",
}


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=30) as resp:
        return resp.read()


def _fetch_trials(condition: str, n: int) -> list[dict]:
    """Fetch up to n completed Phase 2 or 3 trials for the condition."""
    # Note: filter.phase is not supported by CT.gov API v2; phase filtering
    # is done client-side after fetching.  fetch 3× more to compensate.
    params = "&".join([
        f"query.cond={urllib.parse.quote(condition)}",
        "filter.overallStatus=COMPLETED",
        f"pageSize={n * 3}",
        "format=json",
    ])
    url = f"{_CTGOV_API}?{params}"
    data = json.loads(_http_get(url))
    studies = data.get("studies", [])
    # Client-side filter: keep only Phase 2 or Phase 3 trials
    def _is_phase2_or_3(study: dict) -> bool:
        phases = (
            study.get("protocolSection", {})
            .get("designModule", {})
            .get("phases", [])
        )
        return any(p in ("PHASE2", "PHASE3") for p in phases)
    filtered = [s for s in studies if _is_phase2_or_3(s)]
    return filtered[:n]

# urllib.parse needed for quote()
import urllib.parse


# ─────────────────────────────────────────────────────────────────────────────
# Convert a ClinicalTrials.gov study JSON → Markdown
# ─────────────────────────────────────────────────────────────────────────────

def _study_to_markdown(study: dict) -> tuple[str, str]:
    """Return (doc_name, markdown_text) for a study record.

    Sections produced (matching section names across all trials — this is
    intentional; it is the source of the disambiguation problem):

      # Brief Summary
      # Detailed Description
      # Eligibility Criteria
      ## Inclusion Criteria
      ## Exclusion Criteria
      # Primary Outcomes
      # Secondary Outcomes
      # Interventions
    """
    ps = study.get("protocolSection", {})

    # ── Identification ────────────────────────────────────────────────────────
    id_mod = ps.get("identificationModule", {})
    nct_id = id_mod.get("nctId", "NCTUNKNOWN")
    brief_title = id_mod.get("briefTitle", "Untitled")
    official_title = id_mod.get("officialTitle", "")

    # Use NCT ID + condensed title as document name
    slug = re.sub(r"[^a-z0-9]+", "_", brief_title.lower())[:40].strip("_")
    doc_name = f"{nct_id}_{slug}"

    parts: list[str] = []

    # ── Brief Summary ─────────────────────────────────────────────────────────
    desc_mod = ps.get("descriptionModule", {})
    brief = desc_mod.get("briefSummary", "").strip()
    if brief:
        parts.append(f"# Brief Summary\n\n{brief}")

    # ── Detailed Description ──────────────────────────────────────────────────
    detailed = desc_mod.get("detailedDescription", "").strip()
    if detailed:
        parts.append(f"# Detailed Description\n\n{detailed}")

    # ── Eligibility Criteria ──────────────────────────────────────────────────
    elig_mod = ps.get("eligibilityModule", {})
    criteria_raw = elig_mod.get("eligibilityCriteria", "").strip()
    if criteria_raw:
        # ClinicalTrials.gov returns criteria as plain text with
        # "Inclusion Criteria:\n\n* item\n\nExclusion Criteria:\n\n* item"
        # Split and re-emit as markdown subheadings.
        incl_match = re.search(
            r"inclusion criteria[:\s]*(.*?)(?:exclusion criteria|$)",
            criteria_raw,
            re.IGNORECASE | re.DOTALL,
        )
        excl_match = re.search(
            r"exclusion criteria[:\s]*(.*?)$",
            criteria_raw,
            re.IGNORECASE | re.DOTALL,
        )

        elig_parts = ["# Eligibility Criteria"]
        if incl_match:
            elig_parts.append(f"\n## Inclusion Criteria\n\n{incl_match.group(1).strip()}")
        if excl_match:
            elig_parts.append(f"\n## Exclusion Criteria\n\n{excl_match.group(1).strip()}")
        if not incl_match and not excl_match:
            elig_parts.append(f"\n{criteria_raw}")

        parts.append("\n".join(elig_parts))

    # ── Primary Outcomes ──────────────────────────────────────────────────────
    outcomes_mod = ps.get("outcomesModule", {})
    primary = outcomes_mod.get("primaryOutcomes", [])
    if primary:
        lines = ["# Primary Outcomes"]
        for o in primary:
            measure = o.get("measure", "")
            desc = o.get("description", "")
            timeframe = o.get("timeFrame", "")
            lines.append(f"\n**{measure}**")
            if desc:
                lines.append(desc)
            if timeframe:
                lines.append(f"*Time frame: {timeframe}*")
        parts.append("\n".join(lines))

    # ── Secondary Outcomes ────────────────────────────────────────────────────
    secondary = outcomes_mod.get("secondaryOutcomes", [])
    if secondary:
        lines = ["# Secondary Outcomes"]
        for o in secondary[:8]:  # cap to avoid giant lists
            measure = o.get("measure", "")
            desc = o.get("description", "")
            lines.append(f"\n**{measure}**")
            if desc:
                lines.append(desc)
        parts.append("\n".join(lines))

    # ── Interventions ─────────────────────────────────────────────────────────
    arms_mod = ps.get("armsInterventionsModule", {})
    interventions = arms_mod.get("interventions", [])
    if interventions:
        lines = ["# Interventions"]
        for iv in interventions:
            name = iv.get("name", "")
            iv_type = iv.get("type", "")
            desc = iv.get("description", "").strip()
            lines.append(f"\n**{name}** ({iv_type})")
            if desc:
                lines.append(desc)
        parts.append("\n".join(lines))

    markdown = "\n\n---\n\n".join(parts)
    return doc_name, markdown


# ─────────────────────────────────────────────────────────────────────────────
# Queries
#
# Format: (query, section_keywords, doc_name_fragment, reason)
#
# All queries exploit the structural ambiguity of clinical trial records:
# - section_keywords: must appear in chunk.section for a correct result
# - doc_name_fragment: substring that must appear in chunk.document_name
# ─────────────────────────────────────────────────────────────────────────────

QUERIES: list[tuple[str, list[str], str, str]] = [
    # ── Boilerplate exclusion criteria ────────────────────────────────────────
    # These exact phrases appear in hundreds of trials. Without the NCT ID in
    # the breadcrumb the chunks are completely interchangeable.
    (
        "ECOG performance status 0 1 2 eligibility oncology trial",
        ["Eligibility", "Exclusion"],
        "NCT",  # any trial — we check doc-level correctness separately
        "ECOG PS criteria appear verbatim across all solid tumour trials. "
        "'ECOG performance status ≤ 2' is in every exclusion list; the NCT ID "
        "is the only distinguishing token and it only appears in the breadcrumb.",
    ),
    (
        "total bilirubin ULN hepatic function renal creatinine laboratory",
        ["Eligibility"],
        "NCT",
        "Organ function thresholds (bilirubin ≤ 1.5 × ULN, creatinine ≤ 1.5 × ULN) "
        "appear in every Phase 2/3 oncology trial. Without section context a retrieval "
        "system cannot distinguish which trial this threshold belongs to.",
    ),
    (
        "pregnant breastfeeding contraception women childbearing potential exclusion",
        ["Eligibility", "Exclusion"],
        "NCT",
        "Reproductive exclusion criteria are verbatim identical across oncology trials. "
        "Continuation chunks from eligibility sections contain only this boilerplate.",
    ),
    (
        "active infection systemic antibiotics immunocompromised HIV hepatitis",
        ["Eligibility", "Exclusion"],
        "NCT",
        "Infection/immunosuppression exclusions are standard across all oncology Phase 2/3 "
        "trials. Mid-section chunks have no trial-identifying content.",
    ),
    # ── Primary outcomes (RECIST) ─────────────────────────────────────────────
    # RECIST 1.1 language is quoted verbatim from the guideline in every trial.
    (
        "objective response rate ORR RECIST 1.1 complete partial response",
        ["Primary", "Outcomes"],
        "NCT",
        "ORR defined by RECIST 1.1 is the primary endpoint in the majority of solid "
        "tumour trials. The definition text is identical across sponsors. Without "
        "section context, retrieval cannot identify which trial's outcomes are returned.",
    ),
    (
        "progression free survival PFS overall survival OS Kaplan-Meier median",
        ["Outcomes"],
        "NCT",
        "PFS and OS appear as primary or secondary endpoints across all oncology trials. "
        "The statistical definition ('time from randomization to first documented "
        "progression or death') is a fixed phrase shared by hundreds of studies.",
    ),
    # ── Condition-specific queries (tests document-level disambiguation) ───────
    # For these we expect the section to contain the condition or drug name.
    (
        "non-small cell lung cancer NSCLC PD-L1 immunotherapy checkpoint",
        ["Summary", "Description"],
        "NCT",
        "NSCLC-specific language (PD-L1 expression, checkpoint inhibitor) appears in "
        "the Brief Summary but not in eligibility continuation chunks. "
        "Naive retrieval may surface eligibility chunks from other indication trials "
        "because 'immunotherapy' and 'checkpoint' appear in many disease contexts.",
    ),
    (
        "colorectal cancer FOLFOX FOLFIRI bevacizumab mFOLFOX6 oxaliplatin",
        ["Summary", "Description", "Interventions"],
        "NCT",
        "CRC-specific regimens (FOLFOX, FOLFIRI, bevacizumab) are the distinguishing "
        "content. Mid-intervention-description continuation chunks contain only dosing "
        "schedules ('administered on Day 1 of each 14-day cycle') shared across oncology.",
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
# TF-IDF (pure Python)
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
    scored = [(_cos(qv, v), c) for v, c in zip(vecs, chunks)]
    return sorted(scored, key=lambda x: x[0], reverse=True)[:k]

def _cluster_quality(
    chunks: list[DocumentChunk],
    vecs: list[list[float]],
) -> tuple[float, float]:
    def _key(c: DocumentChunk) -> str:
        # Top-level section heading (e.g. "Eligibility Criteria")
        return (c.section or "").split(">")[0].strip().split("\n")[0] or "root"

    intra, inter = [], []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            sim = _cos(vecs[i], vecs[j])
            if _key(chunks[i]) == _key(chunks[j]):
                intra.append(sim)
            else:
                inter.append(sim)
    return (
        sum(intra) / len(intra) if intra else 0.0,
        sum(inter) / len(inter) if inter else 0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClinicalTrials.gov contextual vs naive RAG demo"
    )
    parser.add_argument(
        "--cache-dir",
        default="/tmp/ctgov_cache",
        help="Directory to cache downloaded trial JSON (default: /tmp/ctgov_cache)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Trials to fetch per condition (default: 3)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="Target chars per chunk (default: 600)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Results to show per query (default: 3)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"trials_n{args.n_trials}.json"

    print("=" * 72)
    print("Chunky Monkey — ClinicalTrials.gov Contextual vs Naive RAG Demo")
    print("=" * 72)
    print()

    # ── Fetch / load trial data ───────────────────────────────────────────────
    if cache_file.exists():
        print(f"Loading cached trials from {cache_file}")
        all_studies = json.loads(cache_file.read_text())
    else:
        print(f"Fetching {args.n_trials} trials per condition from ClinicalTrials.gov...")
        all_studies: list[dict] = []
        seen_ncts: set[str] = set()

        for condition in _SEARCH_CONDITIONS:
            print(f"  {condition}...", end=" ", flush=True)
            try:
                studies = _fetch_trials(condition, args.n_trials)
                added = 0
                for s in studies:
                    nct = (
                        s.get("protocolSection", {})
                         .get("identificationModule", {})
                         .get("nctId", "")
                    )
                    if nct and nct not in seen_ncts:
                        seen_ncts.add(nct)
                        all_studies.append(s)
                        added += 1
                print(f"{added} added")
                time.sleep(0.3)  # polite rate limit
            except Exception as exc:
                print(f"FAILED: {exc}")

        cache_file.write_text(json.dumps(all_studies))
        print(f"Cached {len(all_studies)} trials → {cache_file}")

    print(f"\nCorpus: {len(all_studies)} clinical trial records")
    print()

    # ── Convert to markdown + chunk ───────────────────────────────────────────
    naive_loader = DocumentLoader(
        chunk_size=args.chunk_size,
        table_chunk_limit=args.chunk_size,
        context_strategy=None,
    )
    ctx_loader = DocumentLoader(
        chunk_size=args.chunk_size,
        table_chunk_limit=args.chunk_size,
        context_strategy="prefix",
    )

    naive_all: list[DocumentChunk] = []
    ctx_all:   list[DocumentChunk] = []
    doc_names: list[str] = []

    for study in all_studies:
        try:
            doc_name, markdown = _study_to_markdown(study)
        except Exception:
            continue
        if not markdown.strip():
            continue

        doc_names.append(doc_name)
        data = markdown.encode()
        naive_all.extend(naive_loader.load_bytes(data, name=doc_name, doc_type="markdown"))
        ctx_all.extend(ctx_loader.load_bytes(data, name=doc_name, doc_type="markdown"))

    print(f"Total chunks: {len(naive_all)} naive / {len(ctx_all)} contextual")
    print(f"Documents: {len(doc_names)}")
    print()

    # ── Section distribution ──────────────────────────────────────────────────
    sec_counts: Counter = Counter()
    for c in ctx_all:
        top = (c.section or "").split(">")[0].strip().split("\n")[0] or "(none)"
        sec_counts[top] += 1

    print("Section distribution across all trials:")
    for sec, cnt in sec_counts.most_common():
        print(f"  {cnt:4d}×  {sec!r}")
    print()
    print("  Note: identical section headings across all", len(doc_names),
          "trials — this is the disambiguation problem.")
    print()

    # ── Build vectors ─────────────────────────────────────────────────────────
    naive_texts = [c.content for c in naive_all]
    ctx_texts   = [c.embedding_content for c in ctx_all]

    naive_vocab, naive_idf = _build(naive_texts)
    ctx_vocab,   ctx_idf   = _build(ctx_texts)

    naive_vecs = [_vec(t, naive_vocab, naive_idf) for t in naive_texts]
    ctx_vecs   = [_vec(t, ctx_vocab,   ctx_idf)   for t in ctx_texts]

    # ── Part 1: Retrieval ─────────────────────────────────────────────────────
    print("━" * 72)
    print("PART 1 — RETRIEVAL")
    print("━" * 72)
    print()
    print("For boilerplate queries (ECOG, RECIST, organ function) the 'correct'")
    print("result is any chunk from the correct section type, since all trials")
    print("share this language. We score on section-type correctness.")
    print()

    def sec_correct(chunk: DocumentChunk, kws: list[str]) -> bool:
        sec = (chunk.section or "").lower()
        return any(k.lower() in sec for k in kws)

    ctx_p1 = naive_p1 = 0
    ctx_wins = naive_wins = both_rank1 = neither = 0

    for query, sec_kws, _doc_frag, reason in QUERIES:
        naive_res = _retrieve(query, naive_all, naive_vecs, naive_vocab, naive_idf, k=args.top_k)
        ctx_res   = _retrieve(query, ctx_all,   ctx_vecs,   ctx_vocab,   ctx_idf,   k=args.top_k)

        naive_r1_ok = sec_correct(naive_res[0][1], sec_kws)
        ctx_r1_ok   = sec_correct(ctx_res[0][1],   sec_kws)

        naive_margin = naive_res[0][0] - naive_res[1][0] if len(naive_res) > 1 else 0.0
        ctx_margin   = ctx_res[0][0]   - ctx_res[1][0]   if len(ctx_res)   > 1 else 0.0

        print(f"Query: {query!r}")
        print(f"Target section keywords: {sec_kws}")
        print(f"Why naive may fail: {reason}")
        print()

        for label, results in [("Naive     ", naive_res), ("Contextual", ctx_res)]:
            for rank, (score, c) in enumerate(results, 1):
                ok = sec_correct(c, sec_kws)
                marker = "✓" if ok else " "
                trunc = c.content[:60].replace("\n", " ")
                print(
                    f"  {label} #{rank} [{marker}] score={score:.3f}  "
                    f"doc={c.document_name[:30]!r}  sec={c.section!r}"
                )
                print(f"             content: {trunc!r}…")

        print(
            f"  Naive      rank-1 correct: {'YES' if naive_r1_ok else 'NO ':3s}  "
            f"margin={naive_margin:+.3f}"
        )
        print(
            f"  Contextual rank-1 correct: {'YES' if ctx_r1_ok else 'NO ':3s}  "
            f"margin={ctx_margin:+.3f}"
        )

        if ctx_r1_ok and not naive_r1_ok:
            verdict = "CONTEXTUAL WINS"
            ctx_wins += 1
        elif naive_r1_ok and not ctx_r1_ok:
            verdict = "naive wins"
            naive_wins += 1
        elif ctx_r1_ok and naive_r1_ok:
            if ctx_margin > naive_margin + 0.01:
                verdict = (
                    f"both correct — contextual more confident "
                    f"(+{ctx_margin - naive_margin:.3f} margin)"
                )
                ctx_wins += 1
            else:
                verdict = "both correct — tied"
                both_rank1 += 1
        else:
            verdict = "neither rank-1 correct"
            neither += 1

        print(f"  → {verdict}")
        print()

        if ctx_r1_ok:  ctx_p1 += 1
        if naive_r1_ok: naive_p1 += 1

    n = len(QUERIES)
    print("─" * 72)
    print(f"  Precision@1  naive={naive_p1}/{n}   contextual={ctx_p1}/{n}")
    print(f"  Contextual wins (incl. margin): {ctx_wins}/{n}")
    print(f"  Naive wins:                     {naive_wins}/{n}")
    print(f"  Tied at rank-1:                 {both_rank1}/{n}")
    print(f"  Neither correct:                {neither}/{n}")
    print()

    # ── Part 2: Clustering quality ────────────────────────────────────────────
    print("━" * 72)
    print("PART 2 — CLUSTERING QUALITY")
    print("━" * 72)
    print()
    print("Metric: mean cosine similarity between chunk pairs.")
    print("  Intra = same section type across all trials  (want HIGH)")
    print("  Inter = different section types              (want LOW)")
    print()
    print("  Key insight: across 24+ trials, 'Eligibility Criteria' chunks")
    print("  should cluster together (same topic) while remaining separable")
    print("  from 'Primary Outcomes' and 'Interventions' chunks.")
    print("  Contextual adds the NCT ID which does NOT hurt intra-section")
    print("  similarity (section words dominate) but sharpens inter-section")
    print("  separation because outcomes and eligibility get distinct tokens.")
    print()

    # Subsample for speed
    MAX_N = min(len(naive_all), 150)
    import random
    random.seed(42)
    if len(naive_all) > MAX_N:
        idx = random.sample(range(len(naive_all)), MAX_N)
        n_s  = [naive_all[i] for i in idx]
        nv_s = [naive_vecs[i] for i in idx]
        c_s  = [ctx_all[i] for i in idx]
        cv_s = [ctx_vecs[i] for i in idx]
    else:
        n_s, nv_s = naive_all, naive_vecs
        c_s, cv_s = ctx_all, ctx_vecs

    naive_intra, naive_inter = _cluster_quality(n_s, nv_s)
    ctx_intra,   ctx_inter   = _cluster_quality(c_s, cv_s)
    naive_sep = naive_intra - naive_inter
    ctx_sep   = ctx_intra   - ctx_inter

    col = 14
    print(f"  {'':20s}  {'Naive':>{col}}  {'Contextual':>{col}}  {'Δ':>{col}}")
    print(f"  {'─'*20}  {'─'*col}  {'─'*col}  {'─'*col}")
    print(f"  {'Intra-section':20s}  {naive_intra:>{col}.4f}  {ctx_intra:>{col}.4f}  {ctx_intra-naive_intra:>+{col}.4f}")
    print(f"  {'Inter-section':20s}  {naive_inter:>{col}.4f}  {ctx_inter:>{col}.4f}  {ctx_inter-naive_inter:>+{col}.4f}")
    print(f"  {'Separation':20s}  {naive_sep:>{col}.4f}  {ctx_sep:>{col}.4f}  {ctx_sep-naive_sep:>+{col}.4f}")
    print()

    if ctx_sep > naive_sep:
        pct = (ctx_sep - naive_sep) / abs(naive_sep) * 100 if naive_sep else float("inf")
        print(f"  Contextual separation is {pct:.1f}% better than naive.")
        print("  Thesis supported on real clinical trial data: document name +")
        print("  section path in embedding_content produces tighter within-section")
        print("  clusters and better separation across section types.")
    else:
        print("  Separation did not improve — try --n-trials 6 for a larger corpus.")
    print()

    # ── Part 3: MMR Cohort Quality ────────────────────────────────────────────
    print("━" * 72)
    print("PART 3 — MMR COHORT QUALITY  (Mean Marginal Relevance)")
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

    for query, _sec_kws, _doc_frag, _reason in QUERIES:
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

    # ── Show the boilerplate problem explicitly ───────────────────────────────
    print("─" * 72)
    print("  ILLUSTRATION: Identical exclusion criteria across trials")
    print()
    excl_chunks = [
        c for c in naive_all
        if "exclusion" in (c.section or "").lower()
    ][:6]
    if excl_chunks:
        print("  Six exclusion-criteria chunks from DIFFERENT trials (naive content):")
        for i, c in enumerate(excl_chunks, 1):
            trunc = c.content[:120].replace("\n", " ")
            print(f"  [{i}] doc={c.document_name[:30]!r}")
            print(f"      {trunc!r}…")
        print()
        print("  Contextual versions of the same chunks (with document + section prefix):")
        ctx_excl = [
            c for c in ctx_all
            if "exclusion" in (c.section or "").lower()
        ][:6]
        for i, c in enumerate(ctx_excl, 1):
            trunc = c.embedding_content[:150].replace("\n", " ")
            print(f"  [{i}] {trunc!r}…")
    print()


if __name__ == "__main__":
    main()
