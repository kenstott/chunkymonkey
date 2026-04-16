# Cohort Retrieval Quality Benchmark for Structure-Preserving Chunking

## Objective

Measure how well a chunking strategy produces a top-k cohort that enables an LLM to generate a correct, scoped, and faithful answer using semantic similarity as the sole retrieval mechanism. The unit of evaluation is the cohort, not the individual chunk. The consumer is an LLM with no access to source documents beyond what the cohort provides. The benchmark evaluates performance across five corpus scale points to establish a scaling trajectory rather than a single-point comparison.

## Assumptions

The corpus represents an enterprise document environment: heterogeneous document types spanning multiple organizational domains with different vocabulary, structure, and meaning conventions. The same surface terminology appears across domains with different semantics. Documents range from structurally dense (regulatory filings, policy manuals, clinical protocols) to structurally flat (narrative reports, technical guides, correspondence). The corpus is not curated for retrieval quality. It reflects the overlapping, contradictory reality of institutional knowledge.

## Corpus Design

Five document domains, balanced by chunk volume (not document count) at each scale point. Target equal chunk representation (plus or minus 15%) across domains. When document sizes differ substantially across domains (as with EDGAR filings versus clinical trial protocols), adjust document counts per domain to achieve chunk volume parity.

**Financial.** EDGAR 10-Ks, 10-Qs, proxy statements. Dense tables, nested footnotes, defined terms, cross-references to prior filings. Structural homogeneity within domain is high.

**Regulatory.** Federal Register proposed and final rules from financial, environmental, and health agencies. Deep conditional logic, internal cross-references, preamble discussion with rationale and comment responses.

**Clinical.** ClinicalTrials.gov protocols and results, PubMed abstracts. Structured methodology sections, eligibility criteria, endpoint definitions, adverse event tables.

**Product/Technical.** FDA drug labels, technical documentation. Prescriptive content, dosage tables, contraindication lists, usage specifications.

**Operational.** Government agency guidance documents, procedures, standards (NIST publications, agency handbooks). Numbered requirements, informative annexes, defined terms.

All sources are public, free, API-accessible, and government works with no copyright restrictions, enabling full reproducibility.

## Scale Points

Five runs producing a trajectory across three orders of magnitude.

| Scale Point | Approximate Documents | Purpose |
|---|---|---|
| S1 | ~30 | Baseline, establishing small-corpus behavior |
| S2 | ~2,500 | Moderate scale validation |
| S3 | ~12,500 | 5x trajectory extension |
| S4 | ~25,000 | 10x trajectory extension |
| S5 | ~50,000 | Practical ceiling where embedding and evaluation costs remain manageable |

At each scale point, maintain chunk volume balance across domains. Report actual document counts, chunk counts, and chunk volume distribution per domain for each scale point.

## Query Design

160 queries, constant across all scale points. Four categories, each testing a different cohort failure mode.

### Scoped Lookup (48 queries, 30% weight)

The answer exists in a narrow set of passages, but structurally similar passages from other contexts exist as distractors. These test the chunking strategy's ability to provide enough context for the retrieval to find the right passage and enough differentiation for the LLM to scope its answer correctly.

Examples: drug dosage for a specific indication and patient population, reporting threshold for a specific transaction category and jurisdiction, compliance requirement under a specific regulatory subsection.

### Cross-Domain Synthesis (40 queries, 25% weight)

The answer requires chunks from at least two domains. These test whether the cohort assembles complementary information across organizational boundaries without requiring domain routing.

Examples: regulatory obligations that span clinical trial protocols and FDA labeling requirements, compliance controls that span financial reporting rules and operational procedures.

### Disambiguation (32 queries, 20% weight)

The query uses terminology that carries different meanings across domains. These test whether the cohort provides sufficient context for the LLM to resolve semantic ambiguity without external help.

Examples: "exposure" (credit risk, chemical contact, cybersecurity vulnerability), "material" (legal threshold, physical substance, content), "control" (audit mechanism, engineering constraint, management authority), "adverse events" (clinical safety, financial loss events).

### Broad Topical (40 queries, 25% weight)

The query is open-ended and multiple passages are legitimately relevant. These test cohort diversity and complementarity when there is no single right answer.

Examples: overview of a regulatory framework, summary of risk factors across an industry, description of a therapeutic area's clinical development landscape.

### Reference Answer Construction

Each query has a human-authored reference answer decomposed into discrete factual claims (facets). Each facet is tagged with the source passage(s) that support it. The facet decomposition is performed once and held constant across all scale points and strategies.

## Experimental Design

The chunking strategy is the only variable. All other pipeline components are held constant across all runs.

### Fixed Controls

Embedding model, vector store, similarity function, k (primary evaluation at k=5, secondary at k=3 and k=10), query set, reference answers, evaluation methodology.

### Strategies Under Test

**Contextual.** Structure-preserving chunking with breadcrumbs embedded in chunk text.

**Contextual-no-breadcrumb.** Identical chunk boundaries to contextual, but breadcrumbs stored as metadata only, not embedded in chunk text. Breadcrumbs available to the LLM at generation time but not influencing retrieval. This isolates the breadcrumb contribution from the structural chunking contribution.

**Naive-1600.** Fixed 1600-token chunks.

**Naive-800.** Fixed 800-token chunks.

**Naive-400.** Fixed 400-token chunks.

### Breadcrumb Isolation

The contextual-no-breadcrumb variant is the critical comparison. The gap between contextual and contextual-no-breadcrumb measures what breadcrumbs contribute to retrieval. The gap between contextual-no-breadcrumb and naive measures what structural preservation contributes independently. If the breadcrumb contribution is negative on some query categories at some scale points, that finding is reported and analyzed rather than suppressed.

## Metrics

Six components, each evaluated at the cohort level.

### M1: Cohort Coverage (weight 0.25)

What fraction of the reference answer's facets are recoverable from the cohort? For each query, the reference answer is decomposed into N independent factual claims. An evaluator (LLM-as-judge with human validation on a random 20% sample) determines whether the cohort contains sufficient information to support each claim.

**Score** = (supported claims) / (total claims).

This measures the cohort's breadth across the answer space. A cohort covering 8 of 10 facets scores 0.80 regardless of how many chunks contributed or how they ranked individually. This replaces nDCG as the primary retrieval quality measure because it evaluates the cohort as an information unit rather than scoring chunks independently.

### M2: Cohort Coherence (weight 0.20)

Two sub-dimensions, equally weighted within the component.

**M2a: Structural Completeness (0.10).** Fraction of chunks in the cohort that are structurally whole: no mid-sentence truncation, no orphaned table rows, no split list items. Score = (complete chunks) / k. This is chunk-level but aggregated to the cohort. It measures the baseline readability of the material the LLM receives.

**M2b: Cohort Complementarity (0.10).** Inverse of intra-cohort redundancy. Compute mean pairwise semantic similarity among all chunks in the cohort. Score = 1 minus mean pairwise cosine similarity, normalized to 0-1. This penalizes retrieval strategies that fill the cohort with near-duplicate chunks from adjacent passages. A cohort of five diverse, complementary chunks scores higher than a cohort of five chunks that say the same thing in slightly different words.

### M3: Cohort Differentiation (weight 0.20)

When two or more chunks in the cohort address similar concepts but from different scopes or contexts, does the cohort contain enough information for the LLM to distinguish which applies where?

Identify all chunk pairs in the cohort where semantic similarity exceeds a threshold (cosine > 0.75) but the chunks originate from different source contexts (different documents, different sections, different domains). For each such pair, an LLM-as-judge evaluates whether the chunks as presented contain sufficient contextual signal to determine which chunk applies to which scope.

**Score** = (distinguishable pairs) / (total similar pairs). If no similar pairs exist in the cohort, score defaults to 1.0.

This is the component that directly tests the breadcrumb hypothesis. It only activates when the cohort contains potential confusion. On queries where all retrieved chunks are lexically distinct, it contributes nothing. On queries where the cohort contains relevant-looking chunks from different scopes, it measures whether the chunking strategy provided the differentiation signal the LLM needs.

Report the activation rate (what percentage of queries triggered M3 evaluation) alongside the score. The activation rate itself is informative: it tells you how often the retrieval produced potentially confusable cohorts.

### M4: Answer Fidelity (weight 0.25)

The end-to-end measure. Give an LLM the cohort (and only the cohort) plus the query. Evaluate the generated answer on two sub-dimensions, equally weighted.

**M4a: Correctness (0.125).** Does the answer contain accurate claims supported by the cohort? Score against the reference answer's facet list. Penalize both omission (missing facets the cohort could support) and fabrication (claims not supported by any chunk in the cohort).

**M4b: Faithfulness (0.125).** Does the answer stay grounded in the retrieved cohort? Score = (claims in the generated answer attributable to a specific chunk) / (total claims in the generated answer). An answer that introduces information not present in any chunk indicates the LLM drew on parametric knowledge, which is a governance problem in enterprise contexts because the answer can't be traced to an authoritative source.

M4 is the ground truth. Everything else in the metric is a proxy for whether the LLM produced a trustworthy answer from this cohort. M4 measures it directly.

### M5: Failure Rate (weight 0.05)

Percentage of queries where the cohort produces a materially wrong answer: not incomplete, but confidently incorrect because the cohort provided relevant-looking but wrong-scope chunks. An LLM-as-judge compares the generated answer against the reference answer and flags cases where specific factual claims are contradicted.

**Score** = 1 minus failure rate. Weight is low because this should be infrequent for any reasonable strategy. It is included because in regulated contexts a single confident wrong answer is more damaging than a hundred incomplete responses.

Report failure cases qualitatively with examples showing what went wrong. The failure mode taxonomy is as valuable as the aggregate score.

### M6: Cohort Efficiency (weight 0.05)

Total token count of the cohort relative to M4 answer fidelity. Score = M4 / (cohort tokens / normalization constant), where the normalization constant is the mean cohort token count across all strategies at that scale point. This rewards strategies that deliver equivalent answer quality with fewer tokens, which matters for context window utilization, latency, and cost in production.

## Composite Score

### Cohort Retrieval Quality Index (CRQI)

```
CRQI = (0.25 × M1) + (0.20 × M2) + (0.20 × M3) + (0.25 × M4) + (0.05 × M5) + (0.05 × M6)
```

Computed per query, aggregated as weighted average across query categories. Reported at each scale point for each strategy.

## Trajectory Analysis

The primary deliverable is not a single CRQI number but a set of scaling curves. For each metric component and each query category, plot the score across the five scale points for all strategies. The trajectory reveals:

**Durability.** Whether each advantage is stable (flat line across scale points), eroding (downward slope), or amplifying (upward slope).

**Attribution.** Whether the contextual-no-breadcrumb variant tracks with contextual (meaning structural preservation drives the advantage) or with naive (meaning breadcrumbs drive the advantage). If it falls between the two, the contributions are additive and separable.

**Disambiguation behavior.** Whether the disambiguation performance at scale confirms or reverses the finding from the S2 run where naive outperformed contextual.

**Differentiation pressure.** Whether M3 activation rate increases with scale (more confusable cohorts in a denser embedding space) and whether contextual chunking's M3 score holds under that increasing pressure.

The inflection points and plateaus in these curves are the findings. They tell adopters what to expect at their own corpus scale and they identify where additional retrieval mechanisms (NER, clustering, reranking) become necessary to maintain quality.

## Reporting

### Per Scale Point

For each scale point, report:

Corpus statistics: document count, chunk count, and chunk volume percentage per domain.

CRQI composite per strategy.

Per-component scores per strategy.

Per-category scores per strategy.

Gap between contextual and best naive configuration per component and per category.

Gap between contextual and contextual-no-breadcrumb per component and per category.

M3 activation rate.

M5 failure cases with qualitative analysis.

### Across Scale Points

Trajectory curves for each component, each category, and the composite.

Identification of stable, eroding, and amplifying advantages with proposed explanations.

Explicit statement of the scale boundary beyond which results have not been empirically validated.

## Reproducibility

Publish the complete benchmark package: corpus document identifiers and retrieval instructions (not the documents themselves, since they are all publicly available via API), query set with facet-decomposed reference answers, evaluation prompts for LLM-as-judge, scoring code, and raw results at every scale point. Any researcher can pull the same documents from the same public APIs, run the same queries, and verify the results.

## Scope and Limitations

**What this benchmark measures.** Whether structure-preserving chunking with hierarchical breadcrumbs produces higher-quality retrieval cohorts for LLM consumption than naive fixed-size chunking, how that advantage decomposes into structural, contextual, and efficiency contributions, whether the breadcrumb contribution is net positive or net negative at enterprise-representative scale, and how all of these behave as corpus size increases across three orders of magnitude.

**What this benchmark does not measure.** Performance at true enterprise scale (millions of documents, tens of millions of chunks). The contribution of NER vocabulary, cluster analysis, or schema metadata integration. The effect of domain-specific embedding models or hybrid retrieval strategies. These are explicitly out of scope and identified as directions for subsequent work.
