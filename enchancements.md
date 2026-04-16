# Enhanced Semantic Similarity Search
## NER, Cluster, and Cohort Assembly Design

## Overview

An enhanced semantic similarity search that accepts a query and returns a top-k cohort of chunks, identical in interface to standard semantic similarity search, but assembles the cohort using four pre-computed dimensions: semantic similarity, structural adjacency, entity adjacency, and cluster adjacency.

From the caller's perspective, nothing changes. Query in, ranked chunks out. The method of deriving the cohort is different.

## Architecture

### Three Layers, All Pre-Computed at Index Time

```
Documents
    │
    ▼
┌─────────────────────────────┐
│  Layer 1: Contextual Chunks │  (existing)
│  - Structure-preserving     │
│  - Breadcrumbs              │
│  - Continuation markers     │
│  - Next/prev/parent links   │
│  - Embedded with BGE        │
│  - FAISS indexed            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Layer 2: Entity Index      │  (new)
│  - NER against vocabulary   │
│  - Entity-chunk associations│
│  - Association scores       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Layer 3: Cluster Map       │  (new)
│  - Entity co-occurrence     │
│  - Cluster assignments      │
│  - Cluster neighbor index   │
└─────────────────────────────┘
```

All three layers are built at index time. All are deterministic. All are incrementally updatable. No LLM is involved.

## Layer 2: Entity Index

### Vocabulary Sources

**Production (enterprise):**
- Customer/counterparty registries from CRM or risk systems
- Employee directories from HR systems
- Vendor/supplier catalogs from procurement systems
- RDB schema metadata: table names, column names, data types, foreign keys, business glossary descriptions
- Regulatory entity lists: agency names, regulation identifiers, CFR citations

**Benchmark (public corpus):**
- SEC EDGAR registrant index: company names, CIK numbers, ticker symbols, SIC codes
- ClinicalTrials.gov structured fields: drug names, conditions, sponsors, investigators
- FDA National Drug Code directory: drug names, active ingredients, manufacturers
- Federal Register API metadata: agency names, CFR parts, docket identifiers
- NIST publication metadata: control identifiers, family names
- Standard NER (spaCy): person names, organization names, locations, dates

### NER Pass

Runs at index time over every chunk. For each chunk:

1. Match against the vocabulary using exact match, case-insensitive match, and configurable alias matching (e.g., "HCA" matches "HCA Healthcare" matches "Hospital Corporation of America").
2. Record each match as an entity-chunk association.

**Output per association:**

```json
{
    "entity_id": "ent_hca_healthcare",
    "chunk_id": "chunk_4327",
    "frequency": 3,
    "positions": [124, 456, 892],
    "score": 0.85
}
```

**Association score** is a function of:
- Frequency of the entity in the chunk (more mentions = stronger association)
- Position in the chunk (entity in first sentence scores higher than entity in last sentence, indicating topical centrality)
- Entity specificity (a rare entity appearing in a chunk is more informative than a common entity)

Score formula (tunable):

```
score = frequency_weight * log(1 + frequency)
      + position_weight * (1 - first_position / chunk_length)
      + specificity_weight * log(total_chunks / chunks_containing_entity)
```

Default weights: frequency 0.4, position 0.3, specificity 0.3.

### Entity-Chunk Association Index

A bidirectional index:

**Entity to chunks:** For each entity, a ranked list of chunks where it appears, sorted by association score.

```
ent_hca_healthcare → [
    (chunk_4327, 0.92),
    (chunk_4331, 0.87),
    (chunk_12044, 0.71),
    ...
]
```

**Chunk to entities:** For each chunk, the list of entities it contains.

```
chunk_4327 → [
    (ent_hca_healthcare, 0.92),
    (ent_revenue_recognition, 0.78),
    (ent_reporting_threshold, 0.65)
]
```

Both directions are pre-computed and stored. Query-time lookup is O(1) per chunk or entity.

### Incremental Update

When a new document enters the corpus:
1. Chunk the document.
2. Run NER on each new chunk.
3. Add new entity-chunk associations to the index.
4. If new entities appear (not in vocabulary), flag for vocabulary review. Do not auto-add.

When a document is removed:
1. Remove its chunks from the vector index.
2. Remove all entity-chunk associations referencing those chunks.
3. Cluster statistics update in the next cluster refresh cycle.

## Layer 3: Cluster Map

### Construction

Build an entity co-occurrence matrix. Two entities co-occur when they appear in the same chunk. The co-occurrence count is the number of chunks where both entities appear.

```
Co-occurrence matrix (sparse):

                    ent_wire_transfer  ent_reporting_threshold  ent_ofac_screening
ent_hca_healthcare         12                    8                      3
ent_wire_transfer           -                   15                      9
ent_reporting_threshold     -                    -                      6
```

Normalize by entity frequency to produce a co-occurrence score (Jaccard, PMI, or simple conditional probability). This prevents high-frequency entities from dominating.

### Clustering Algorithm

Run a standard clustering algorithm over the normalized co-occurrence matrix. Options:

- **Agglomerative clustering** with a distance threshold (simple, deterministic, no k parameter)
- **DBSCAN** (density-based, finds natural clusters without specifying count)
- **Leiden** (community detection, same algorithm GraphRAG uses for graph communities, but applied to co-occurrence rather than LLM-inferred relationships)

The choice is a tuning decision. All are deterministic given the same input.

**Output:**

```json
{
    "cluster_id": "cluster_017",
    "entities": [
        "ent_wire_transfer",
        "ent_reporting_threshold",
        "ent_ofac_screening",
        "ent_beneficiary_country_code",
        "ent_category_a_transactions"
    ],
    "cohesion_score": 0.72
}
```

### Cluster Neighbor Index

For each entity, its cluster assignment and the other entities in the same cluster.

```
ent_wire_transfer → cluster_017 → [
    ent_reporting_threshold,
    ent_ofac_screening,
    ent_beneficiary_country_code,
    ent_category_a_transactions
]
```

For each cluster neighbor entity, the entity-chunk association index provides the ranked list of chunks. This is the two-hop path: query chunk → entity → cluster → neighbor entity → neighbor chunks.

### Incremental Update

When new chunks are added, their entity associations update the co-occurrence matrix. Clusters can be refreshed periodically (nightly, weekly) rather than on every document addition, since cluster structure is statistical and small perturbations don't materially change the topology. Full cluster recomputation is cheap (it's a matrix operation over the entity vocabulary, not over the chunk corpus).

## Enhanced Semantic Similarity Search

### Interface

Identical to standard semantic similarity search:

```python
def search(query: str, k: int = 5) -> list[ScoredChunk]:
    """
    Input:  query string, target cohort size
    Output: ranked list of (chunk, score) tuples, len <= k
    """
```

The caller does not know or care that the cohort was assembled from four dimensions. The output is a ranked list of chunks with scores.

### Cohort Assembly Algorithm

```
┌──────────────────────────────────────────────────────────────┐
│  1. SEED: Semantic similarity (vector search)                │
│     - Embed query                                            │
│     - Retrieve top-m candidates from FAISS (m = 3 * k)      │
│     - These are the seed chunks                              │
├──────────────────────────────────────────────────────────────┤
│  2. STRUCTURAL: Next/prev/parent expansion                   │
│     - For each seed chunk:                                   │
│       - If continuation marker present: pull prev chunk      │
│       - If chunk ends with cross-reference: pull next chunk  │
│       - Optionally: pull parent for hierarchical context     │
│     - Add to candidate pool                                  │
├──────────────────────────────────────────────────────────────┤
│  3. ENTITY: Entity adjacency expansion                       │
│     - Extract entities from seed + structural chunks         │
│       (pre-computed, index lookup, not runtime NER)          │
│     - For each entity, pull top-n associated chunks          │
│       from entity-chunk index                                │
│     - Add to candidate pool                                  │
├──────────────────────────────────────────────────────────────┤
│  4. CLUSTER: Cluster adjacency expansion (budget-limited)    │
│     - For entities from Steps 1-3, look up cluster neighbors │
│     - For each neighbor entity, pull top-1 associated chunk  │
│     - Add to candidate pool                                  │
│     - Budget: max 2 * k cluster-adjacent candidates          │
├──────────────────────────────────────────────────────────────┤
│  5. SELECT: Score and assemble final cohort                  │
│     - Deduplicate candidate pool                             │
│     - Score each candidate (see scoring below)               │
│     - Select top-k by composite score                        │
│     - Return ranked list                                     │
└──────────────────────────────────────────────────────────────┘
```

### Candidate Pool Sizing

For a target cohort of k=5:

- Step 1 (seed): 15 candidates (3 * k)
- Step 2 (structural): 0-15 additional (depends on continuation markers and cross-references, typically 3-8)
- Step 3 (entity): up to 15 additional (top-3 per entity, capped)
- Step 4 (cluster): up to 10 additional (2 * k, budget-limited)

Total candidate pool: typically 25-50 candidates, from which 5 are selected. The pool is small enough to score in microseconds.

### Scoring Function

Each candidate receives a composite score based on how it entered the pool and what it contributes to the cohort.

**Relevance component (0-1):** Cosine similarity between the candidate's embedding and the query embedding. All candidates get this score, regardless of which dimension produced them. This ensures that entity-adjacent and cluster-adjacent chunks that happen to also be semantically relevant score highest.

**Source priority component (0-1):**
- Seed chunk: 1.0
- Structural neighbor of seed: 0.9
- Entity-adjacent to seed: 0.7
- Cluster-adjacent: 0.5

This ensures that when relevance scores are close, direct matches win over indirect expansions.

**Marginal coverage component (0-1):** Computed iteratively during selection. After each chunk is selected, the remaining candidates are penalized for redundancy with already-selected chunks. This is the MMR principle: semantic similarity to the query minus semantic similarity to the already-selected cohort.

```
coverage_score = query_similarity - lambda * max_similarity_to_selected
```

Lambda is tunable (default 0.3). Higher lambda penalizes redundancy more aggressively, producing more diverse cohorts.

**Composite score:**

```
composite = relevance_weight * relevance
          + priority_weight * source_priority
          + coverage_weight * coverage_score
```

Default weights: relevance 0.5, priority 0.2, coverage 0.3.

### Selection Algorithm (Greedy Sequential)

```python
def select_cohort(candidates, query_embedding, k=5, lambda_diversity=0.3):
    selected = []
    remaining = list(candidates)

    for i in range(k):
        best_score = -1
        best_candidate = None

        for candidate in remaining:
            relevance = cosine_similarity(candidate.embedding, query_embedding)
            priority = candidate.source_priority

            if selected:
                max_sim_to_selected = max(
                    cosine_similarity(candidate.embedding, s.embedding)
                    for s in selected
                )
                coverage = relevance - lambda_diversity * max_sim_to_selected
            else:
                coverage = relevance

            composite = (0.5 * relevance
                       + 0.2 * priority
                       + 0.3 * coverage)

            if composite > best_score:
                best_score = composite
                best_candidate = candidate

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected
```

This is O(k * pool_size), which for k=5 and pool_size=50 is 250 comparisons. Negligible.

### Provenance Tagging

Each chunk in the returned cohort carries a provenance tag indicating how it was selected:

```json
{
    "chunk_id": "chunk_4327",
    "text": "...",
    "score": 0.87,
    "provenance": "seed",
    "breadcrumb": "Compliance Framework > Wire Transfers > Reporting Requirements"
}
```

```json
{
    "chunk_id": "chunk_4331",
    "text": "...",
    "score": 0.72,
    "provenance": "entity_adjacent",
    "linked_by": "ent_reporting_threshold",
    "breadcrumb": "Compliance Framework > Wire Transfers > Exceptions"
}
```

```json
{
    "chunk_id": "chunk_8901",
    "text": "...",
    "score": 0.61,
    "provenance": "cluster_adjacent",
    "linked_by": "ent_ofac_screening",
    "cluster": "cluster_017",
    "breadcrumb": "AML Policy > Transaction Screening > OFAC Requirements"
}
```

The provenance tag is metadata, not part of the chunk text the LLM sees. But it's available for audit, for the UI to display, and for the LLM's system prompt to distinguish primary results from expansions.

### What the LLM Receives

The LLM receives the cohort as a flat list of chunks, ordered by composite score, each with its breadcrumb prefix. The LLM does not see provenance tags or scoring details. It sees source text with hierarchical context labels, identical in format to what the chunking-only pipeline would produce.

The difference is in what's there, not how it's presented. The seed chunks answer the question as asked. The structural neighbors complete the local reading context. The entity-adjacent chunks fill gaps the seed chunks reference. The cluster-adjacent chunks surface cross-domain connections.

The LLM reads a briefing package, not a search result.

### Optionally: Structured Prompt with Dimensions

For advanced use, the system can present the cohort to the LLM with dimension labels:

```
PRIMARY RESULTS (directly matching your query):
[seed chunk 1 with breadcrumb]
[seed chunk 2 with breadcrumb]

RELATED CONTEXT (sharing key entities with primary results):
[entity-adjacent chunk with breadcrumb]

BROADER CONTEXT (topically connected through related concepts):
[cluster-adjacent chunk with breadcrumb]
```

This gives the LLM explicit signal about which chunks are direct answers and which are supporting context, enabling more structured generation. Whether this improves answer quality versus a flat list is an empirical question testable in the CRQI benchmark.

## Configuration Parameters

### NER Layer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocabulary_path` | required | Path to entity vocabulary file |
| `match_mode` | "case_insensitive" | Matching mode: exact, case_insensitive, alias |
| `alias_file` | null | Optional alias mapping (e.g., "HCA" → "HCA Healthcare") |
| `min_entity_length` | 2 | Minimum character length for entity matches |
| `score_weights` | [0.4, 0.3, 0.3] | Frequency, position, specificity weights |

### Cluster Layer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm` | "agglomerative" | Clustering algorithm |
| `distance_threshold` | 0.5 | Cluster merge threshold (agglomerative) |
| `min_cooccurrence` | 2 | Minimum co-occurrence count to form an edge |
| `normalization` | "pmi" | Co-occurrence normalization: raw, jaccard, pmi |
| `refresh_interval` | "on_demand" | When to recompute clusters |

### Cohort Assembly

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 5 | Target cohort size |
| `seed_pool_multiplier` | 3 | Candidate pool = k * multiplier |
| `entity_expansion_top_n` | 3 | Chunks per entity in expansion |
| `cluster_budget` | 10 | Max cluster-adjacent candidates (2 * k) |
| `lambda_diversity` | 0.3 | Redundancy penalty in coverage scoring |
| `relevance_weight` | 0.5 | Composite score weight for relevance |
| `priority_weight` | 0.2 | Composite score weight for source priority |
| `coverage_weight` | 0.3 | Composite score weight for marginal coverage |
| `structural_expansion` | true | Enable next/prev/parent expansion |
| `entity_expansion` | true | Enable entity adjacency expansion |
| `cluster_expansion` | true | Enable cluster adjacency expansion |

Each expansion dimension can be independently enabled or disabled, supporting incremental benchmarking: chunking-only (all expansions off), chunking+structural, chunking+structural+entity, full (all on).

## Incremental Pipeline

### On Document Add

```
1. Chunk new document                         (deterministic)
2. Embed new chunks                           (deterministic)
3. Add to FAISS index                         (incremental)
4. Run NER on new chunks                      (deterministic)
5. Add entity-chunk associations              (incremental)
6. Update co-occurrence matrix                (incremental)
7. Flag cluster refresh needed                (deferred)
8. Update structural links (next/prev/parent) (deterministic)
```

### On Document Remove

```
1. Remove chunks from FAISS index             (incremental)
2. Remove entity-chunk associations           (incremental)
3. Update co-occurrence matrix                (incremental)
4. Flag cluster refresh needed                (deferred)
5. Remove structural links                    (deterministic)
```

### Cluster Refresh

Runs on demand or on a schedule. Recomputes clusters from the current co-occurrence matrix. Cheap (matrix operation over entity vocabulary, not over chunk corpus). Existing query results are unaffected until the refresh completes. New cluster assignments take effect atomically.

## Benchmark Integration

### CRQI Benchmark Variants

Run the CRQI benchmark with four configurations to measure per-layer contribution:

| Variant | Seed | Structural | Entity | Cluster | Measures |
|---------|------|-----------|--------|---------|----------|
| A: Flat top-k | yes | no | no | no | Chunking-only baseline |
| B: Structural | yes | yes | no | no | Next/prev/parent contribution |
| C: Entity | yes | yes | yes | no | NER layer contribution |
| D: Full | yes | yes | yes | yes | Complete pipeline |

The delta between each successive variant measures the incremental contribution of each layer. If a layer doesn't move the CRQI, it doesn't ship.

### GraphRAG-Bench Variants

Run GraphRAG-Bench with variants A and D. Variant A is the comparison against published GraphRAG scores. Variant D shows whether the NER/cluster layer closes any gap on Complex Reasoning (GraphRAG's strongest category) while maintaining the cost and determinism advantages.
