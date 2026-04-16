# Contextual Chunking Benchmark Plan
## Based on GraphRAG-Bench (ICLR 2026)

## Why This Benchmark

GraphRAG-Bench is the most authoritative benchmark for comparing retrieval-augmented generation approaches. It is peer-reviewed (accepted at ICLR 2026), provides standardized evaluation code, publishes a leaderboard with results from nine GraphRAG frameworks, and directly addresses the question "Is GraphRAG really effective, and in which scenarios do graph structures provide measurable benefits?"

The benchmark evaluates the full RAG pipeline (graph construction, knowledge retrieval, answer generation) using two domain-specific corpora with four task difficulty levels. Published results show GraphRAG frequently underperforms vanilla RAG, making this benchmark favorable terrain for demonstrating that a simpler, cheaper approach delivers equivalent or better results.

Alternative considered: BEIR (NeurIPS 2021) is the gold standard for retrieval evaluation (18 datasets, 9 task types, widely cited). However, BEIR's corpus is pre-chunked at the passage level, making it unsuitable for evaluating chunking strategies without significant adaptation. GraphRAG-Bench provides raw source text that can be chunked.

## What We're Measuring

Whether contextual chunking with breadcrumbs, using semantic similarity retrieval alone, matches or exceeds the published performance of GraphRAG implementations on their own benchmark, at a fraction of the cost and with deterministic, auditable execution.

## Benchmark Structure

### Corpora

Two domain-specific datasets derived from textbooks:

**Novel subset:** ~2,010 question-answer pairs from literary/fictional content. Tests narrative comprehension, character relationships, thematic analysis.

**Medical subset:** ~2,060 question-answer pairs from medical/healthcare textbooks. Tests factual recall, diagnostic reasoning, treatment knowledge.

Total: ~4,070 question-answer pairs, available on HuggingFace.

### Task Levels

**Level 1 (Fact Retrieval):** Single-hop lookups requiring precise passage identification. Example: "What is the most common type of skin cancer?"

**Level 2 (Complex Reasoning):** Multi-hop chains requiring synthesis across passages. Example: "How did Hinze's agreement with Felicia relate to the perception of England's rulers?"

**Level 3 (Contextual Summarization):** Synthesize across multiple sources. Example: "What role does John Curgenven play as a Cornish boatman for visitors exploring this region?"

**Level 4 (Creative Generation):** Produce novel content grounded in retrieved facts. Example: "Retell King Arthur's comparison to John Curgenven as a newspaper article." (Not an enterprise use case; report for completeness, analyze with and without.)

### Evaluation Metrics

**Generation evaluation:** Accuracy, ROUGE-L, Factual Coverage (scored by LLM judge).

**Retrieval evaluation:** Whether the retrieved context contains the evidence needed to answer (scored by LLM judge plus BGE embedding similarity).

### Published Baselines

The leaderboard includes results for:
- Vanilla RAG (the naive baseline)
- Microsoft GraphRAG (the reference implementation)
- LightRAG
- HippoRAG2
- fast-graphrag
- RAPTOR
- Additional frameworks

All evaluated with GPT-4-turbo as judge and BGE-large-en-v1.5 as the embedding model.

## Experimental Design

### Strategy Under Test

One strategy: contextual chunking (min=400, max=1200, breadcrumbs=true). This is the recommended configuration validated by the scale sweep results showing stable +7-10% advantage over naive chunking across 500 to 44,676 documents.

### Nested Corpus Scaling (Optional Extension)

If time permits, run at three nested scales (S1 contained within S2 contained within S3) to confirm the advantage holds at increasing density. Each larger scale is a superset of the smaller one, eliminating corpus composition artifacts.

### Fixed Controls

- Embedding model: BGE-large-en-v1.5 (matches benchmark default)
- Vector store: FAISS
- k = 5 (matches typical benchmark configuration)
- Generation model: GPT-4o-mini (for answer generation)
- Evaluation judge: GPT-4o-mini (initial pass), GPT-4.1 (final pass if results warrant)

## Step-by-Step Execution

### Phase 1: Setup (Day 1)

#### Step 1.1: Clone the benchmark repository

```bash
mkdir ~/graphrag-bench && cd ~/graphrag-bench
git clone https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git
cd GraphRAG-Benchmark
```

#### Step 1.2: Install dependencies

```bash
conda create -n chunky-bench python=3.10 -y
conda activate chunky-bench
pip install -r requirements.txt
pip install beir sentence-transformers faiss-cpu openai datasets
```

#### Step 1.3: Download the dataset

```python
# download_data.py
from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

medical = load_dataset("GraphRAG-Bench/GraphRAG-Bench", "medical", split="train")
novel = load_dataset("GraphRAG-Bench/GraphRAG-Bench", "novel", split="train")

medical.to_json("data/medical_questions.json")
novel.to_json("data/novel_questions.json")

print(f"Medical: {len(medical)} questions")
print(f"Novel: {len(novel)} questions")
print(f"Question types: {set(medical['question_type'])}")
```

#### Step 1.4: Inspect the dataset structure

Verify each record contains:
```
- id: unique identifier (e.g., "Medical-73586ddc")
- source: "Medical" or "Novel" (identifies the subset)
- question: the query text
- answer: gold reference answer
- question_type: "Fact Retrieval" | "Complex Reasoning" |
                  "Contextual Summarization" | "Creative Generation"
- evidence: list of ground truth evidence passages
- evidence_relations: relationship descriptions connecting evidence
```

#### Step 1.5: Locate and verify the source corpus

```bash
ls Datasets/
```

The source textbooks must be available as raw text for chunking. Check:
- Are full text files present in the Datasets directory?
- If structured data only, check the paper (arxiv:2506.05690, Appendix C) for source references.
- If source text is not directly available, contact benchmark authors: GraphRAG@hotmail.com

**GATE: Do not proceed past this step without confirmed access to source corpus as chunkable text files.**

If the corpus is available, inventory it:

```python
# inventory_corpus.py
import os

corpus_dir = "Datasets"  # adjust path as needed
for root, dirs, files in os.walk(corpus_dir):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path)
        print(f"{path}: {size:,} bytes")
```

Record: number of source documents, total corpus size in bytes, file formats present.

### Phase 2: Chunking (Day 2)

#### Step 2.1: Chunk the corpus with contextual chunking

Run the contextual chunking library with the validated configuration:

```python
# chunk_corpus.py
from contextual_chunker import ContextualChunker  # your library
import json

chunker = ContextualChunker(
    min_chunk_size=400,
    max_chunk_size=1200,
    overflow_margin=0.15,
    breadcrumbs=True
)

chunks = []
for doc_path in corpus_files:
    with open(doc_path) as f:
        text = f.read()
    doc_chunks = chunker.chunk(text, document_id=doc_path)
    chunks.extend(doc_chunks)

# Save chunks
with open("data/contextual_chunks.json", "w") as f:
    json.dump(chunks, f)

print(f"Total chunks: {len(chunks)}")
print(f"Avg chunk size: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} chars")
```

Each chunk record should contain:
```json
{
    "chunk_id": "unique_id",
    "document_id": "source_document_path",
    "text": "breadcrumb prefix + chunk content",
    "breadcrumb": "the LCA breadcrumb string",
    "continuation_type": null
}
```

#### Step 2.2: Verify chunk quality

Spot-check 20 random chunks:
- Does the breadcrumb accurately reflect the document hierarchy?
- Are paragraph boundaries preserved?
- Are tables and lists intact?
- Do continuation markers appear where expected?

```python
# verify_chunks.py
import random, json

with open("data/contextual_chunks.json") as f:
    chunks = json.load(f)

sample = random.sample(chunks, 20)
for i, chunk in enumerate(sample):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Breadcrumb: {chunk['breadcrumb']}")
    print(f"Continuation: {chunk['continuation_type']}")
    print(f"Text ({len(chunk['text'])} chars):")
    print(chunk['text'][:500])
    print("...")
```

### Phase 3: Embedding (Day 2, continued)

#### Step 3.1: Embed all chunks

```python
# embed_chunks.py
from sentence_transformers import SentenceTransformer
import numpy as np
import json

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

with open("data/contextual_chunks.json") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

# Embed in batches to manage memory
batch_size = 256
embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    batch_embeddings = model.encode(batch, show_progress_bar=True)
    embeddings.append(batch_embeddings)
    print(f"Embedded {min(i+batch_size, len(texts))}/{len(texts)}")

embeddings = np.vstack(embeddings)
np.save("data/contextual_embeddings.npy", embeddings)

print(f"Embedding matrix shape: {embeddings.shape}")
```

#### Step 3.2: Build the FAISS index

```python
# build_index.py
import faiss
import numpy as np

embeddings = np.load("data/contextual_embeddings.npy").astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# Build index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # inner product on normalized = cosine
index.add(embeddings)

faiss.write_index(index, "data/contextual.faiss")

print(f"Index built: {index.ntotal} vectors, {dimension} dimensions")
```

### Phase 4: Retrieval and Generation (Day 3)

#### Step 4.1: Set up the retrieval and generation pipeline

```python
# retrieve_and_generate.py
import faiss
import numpy as np
import json
import openai
from sentence_transformers import SentenceTransformer

# Load components
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
index = faiss.read_index("data/contextual.faiss")

with open("data/contextual_chunks.json") as f:
    chunks = json.load(f)

client = openai.OpenAI()  # uses OPENAI_API_KEY env var

K = 5  # top-k retrieval

def retrieve(query, k=K):
    query_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, k)
    return [(chunks[idx], float(scores[0][i])) for i, idx in enumerate(indices[0])]

def generate_answer(question, context):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer the question based only on the provided context. "
                           "If the context does not contain enough information, "
                           "say so rather than making up an answer."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.0,
        max_tokens=500
    )
    return response.choices[0].message.content
```

#### Step 4.2: Run retrieval and generation for all questions

```python
# run_benchmark.py
import json
import time

# Load questions
with open("data/medical_questions.json") as f:
    medical_qs = [json.loads(line) for line in f]

with open("data/novel_questions.json") as f:
    novel_qs = [json.loads(line) for line in f]

all_questions = medical_qs + novel_qs

results = []
for i, q in enumerate(all_questions):
    # Retrieve
    retrieved = retrieve(q["question"])
    context = "\n\n".join([chunk["text"] for chunk, score in retrieved])

    # Generate
    generated_answer = generate_answer(q["question"], context)

    # Format per benchmark spec
    result = {
        "id": q["id"],
        "question": q["question"],
        "source": q["source"],
        "context": context,
        "evidence": q["evidence"],
        "question_type": q["question_type"],
        "generated_answer": generated_answer,
        "gold_answer": q["answer"]
    }
    results.append(result)

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(all_questions)}")
        # Save checkpoint
        with open("results/contextual_checkpoint.json", "w") as f:
            json.dump(results, f, indent=2)

# Save final results
with open("results/contextual.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Complete: {len(results)} results saved")
```

**Estimated cost:** ~$2-4 (GPT-4o-mini, 4,070 queries, ~2K input tokens each)

**Estimated time:** 2-3 hours depending on API rate limits

#### Step 4.3: Verify results format

```python
# verify_results.py
import json

with open("results/contextual.json") as f:
    results = json.load(f)

# Check required fields
required = ["id", "question", "source", "context", "evidence",
            "question_type", "generated_answer", "gold_answer"]

for r in results[:5]:
    missing = [k for k in required if k not in r]
    if missing:
        print(f"MISSING FIELDS in {r['id']}: {missing}")
    else:
        print(f"{r['id']}: OK ({r['question_type']}, {len(r['context'])} chars context)")

# Count by type
from collections import Counter
types = Counter(r["question_type"] for r in results)
sources = Counter(r["source"] for r in results)
print(f"\nBy type: {dict(types)}")
print(f"By source: {dict(sources)}")
```

### Phase 5: Initial Evaluation with GPT-4o-mini (Day 4)

#### Step 5.1: Run generation evaluation

```bash
export OPENAI_API_KEY=your_key_here

cd Evaluation

python -m Evaluation.generation_eval \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --bge_model BAAI/bge-large-en-v1.5 \
  --data_file ../results/contextual.json \
  --output_file ../results/contextual_gen_eval_mini.json
```

If the evaluation script has different argument names or requires adaptation, inspect it first:

```bash
cat Evaluation/generation_eval.py | head -50
python -m Evaluation.generation_eval --help
```

Adapt arguments as needed. The key inputs are: the results file, the judge model, and the embedding model for similarity scoring.

**Estimated cost:** ~$3-5 (GPT-4o-mini judging 4,070 answers)

#### Step 5.2: Run retrieval evaluation

```bash
python -m Evaluation.retrieval_eval \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --bge_model BAAI/bge-large-en-v1.5 \
  --data_file ../results/contextual.json \
  --output_file ../results/contextual_ret_eval_mini.json
```

**Estimated cost:** ~$3-5 (GPT-4o-mini judging 4,070 retrieval contexts)

#### Step 5.3: Parse and summarize initial results

```python
# summarize_eval.py
import json

with open("results/contextual_gen_eval_mini.json") as f:
    gen_eval = json.load(f)

with open("results/contextual_ret_eval_mini.json") as f:
    ret_eval = json.load(f)

# Aggregate by question_type and source
# (exact aggregation logic depends on evaluation output format)
# Inspect the output structure first:
print("Gen eval keys:", list(gen_eval[0].keys()) if isinstance(gen_eval, list) else list(gen_eval.keys()))
print("Ret eval keys:", list(ret_eval[0].keys()) if isinstance(ret_eval, list) else list(ret_eval.keys()))
```

Produce a summary table:

```
| Metric         | Medical Fact | Medical Reason | Medical Summ | Medical Creative |
|----------------|-------------|----------------|--------------|------------------|
| Accuracy       |             |                |              |                  |
| ROUGE-L        |             |                |              |                  |
| Coverage       |             |                |              |                  |
```

Repeat for Novel subset.

#### Step 5.4: Decision point

Pull published leaderboard scores from graphrag-bench.github.io.

Compare your GPT-4o-mini-judged results against published scores (noting judge model difference).

**If results are clearly competitive or better:** Proceed to Phase 6 (final evaluation with GPT-4.1).

**If results are clearly worse than vanilla RAG:** Stop. Investigate. Check whether Creative Generation is dragging scores down (report with and without). Check whether the corpus structure is suitable for contextual chunking. Check whether the evaluation script expects a different context format.

**If results are ambiguous:** Proceed to Phase 6 anyway. The additional $50-60 resolves the ambiguity with a judge model closer to the published benchmark's GPT-4-turbo.

### Phase 6: Final Evaluation with GPT-4.1 (Day 5)

#### Step 6.1: Rerun evaluation with GPT-4.1

Same results file, stronger judge. No answer regeneration needed.

```bash
python -m Evaluation.generation_eval \
  --model gpt-4.1 \
  --base_url https://api.openai.com/v1 \
  --bge_model BAAI/bge-large-en-v1.5 \
  --data_file ../results/contextual.json \
  --output_file ../results/contextual_gen_eval_final.json

python -m Evaluation.retrieval_eval \
  --model gpt-4.1 \
  --base_url https://api.openai.com/v1 \
  --bge_model BAAI/bge-large-en-v1.5 \
  --data_file ../results/contextual.json \
  --output_file ../results/contextual_ret_eval_final.json
```

**Estimated cost:** $50-60

### Phase 6: Analysis and Reporting (Day 5-6)

#### Step 6.1: Build the comparison table

```
| Strategy                | Fact Ret ACC | Complex Reason ACC | Ctx Summarize ACC | Creative Gen ACC | Average |
|-------------------------|-------------|-------------------|-------------------|------------------|---------|
| Contextual chunking     |             |                   |                   |                  |         |
| RAPTOR (pub.)           | —           | —                 | —                 | —                | 73.58   |
| HippoRAG (pub.)        | —           | —                 | —                 | —                | 72.64   |
| GFM-RAG (pub.)         | —           | —                 | —                 | —                | ~70+    |
| MS GraphRAG (pub.)     | —           | —                 | —                 | —                | ~70+    |
| LightRAG (pub.)        | —           | —                 | —                 | —                | marginal|
| DALK (pub.)            | —           | —                 | —                 | —                | DEGRADED|
| G-Retriever (pub.)     | —           | —                 | —                 | —                | DEGRADED|
| Vanilla RAG            | 83.2% CR    | lower             | lower             | 40.0% coverage   | —       |
```

Note: Published per-task-level breakdowns from the leaderboard should be pulled at execution time from graphrag-bench.github.io, as the leaderboard is dynamically updated. The scores above are from the paper; the leaderboard may contain additional or updated results.

#### Step 6.2: Retrieval comparison

```
| Strategy                | Level 1 CR | Level 2-3 ER | Level 4 ER | Overall |
|-------------------------|-----------|-------------|-----------|---------|
| Contextual chunking     |           |             |           |         |
| Vanilla RAG (pub.)      | 83.2%     | —           | —         | —       |
| HippoRAG (pub.)        | —         | 87.9-90.9%  | —         | —       |
| HippoRAG2 (pub.)       | —         | 85.8-87.8%  | —         | —       |
| Global-GraphRAG (pub.) | —         | —           | 83.1%     | —       |
```

#### Step 6.3: Cost comparison

```
| Approach              | LLM in Indexing | Index Cost per 1M tokens | Prompt per Query | Incremental Update |
|-----------------------|-----------------|--------------------------|------------------|--------------------|
| Contextual (ours)     | No              | $0 (deterministic)       | ~6K tokens       | Per-document        |
| RAPTOR                | Yes (summaries) | Moderate                 | Moderate         | Full rebuild        |
| HippoRAG2             | Yes (extraction)| Moderate                 | ~1K tokens       | Partial rebuild     |
| LightRAG              | Yes (extraction)| Moderate                 | ~10K tokens      | Partial rebuild     |
| MS GraphRAG (Global)  | Yes (full)      | High                     | ~40K tokens      | Full rebuild        |
| Vanilla RAG           | No              | $0 (deterministic)       | ~6K tokens       | Per-document        |
```

#### Step 6.4: Write findings

Structure:

1. **Accuracy:** Contextual chunking achieves [X]% on GraphRAG-Bench. Top GraphRAG (RAPTOR) achieves 73.58%. Several GraphRAG methods degrade baseline LLM performance. Vanilla RAG beats all GraphRAG on simple fact retrieval.

2. **Cost:** Zero LLM indexing cost. Total benchmark cost under $75. GraphRAG methods require LLM extraction passes during indexing, with Global-GraphRAG pushing prompt lengths to 40,000 tokens per query.

3. **Properties:** Deterministic, incrementally updatable, auditable provenance. GraphRAG produces non-reproducible graphs from non-deterministic LLM extraction.

4. **Limitations:** Single-domain textbook corpus. Enterprise heterogeneity not tested. Scale sweep results (44K docs, +7-10% over naive) provide supplementary evidence from a more diverse corpus.

#### Step 6.5: Report without Creative Generation

Recompute excluding Level 4 tasks. This is the enterprise-relevant view.

## Cost Summary

| Phase | Duration | Cost |
|-------|----------|------|
| Setup and data (Phase 1) | 1 day | Free |
| Chunking and embedding (Phase 2) | 1 day | Free (local compute) |
| Retrieval and generation (Phase 3) | 1 day | ~$2-4 (GPT-4o-mini) |
| Initial evaluation (Phase 4) | 1 day | ~$6-10 (GPT-4o-mini) |
| Final evaluation (Phase 5) | 0.5 day | ~$50-60 (GPT-4.1) |
| Analysis and reporting (Phase 6) | 1-2 days | Free |
| **Total** | **5-6 days** | **~$60-75** |

Initial pass through decision point: **~$10**

## Success Criteria

**Strong result:** Contextual chunking lands in the 72-74% accuracy range alongside RAPTOR and HippoRAG, the top GraphRAG methods. Publish with the argument: "Equivalent accuracy, zero LLM indexing cost, deterministic, incrementally updatable, auditable, reusable artifacts."

**Acceptable result:** Contextual chunking lands above vanilla RAG but below top GraphRAG (65-72% range). Publish with the argument: "Better than vanilla RAG, competitive with most GraphRAG implementations (which provide only marginal improvement), at a fraction of the cost. The gap to top GraphRAG methods motivates the NER/cluster layer as next phase."

**Weak result:** Contextual chunking at or below vanilla RAG. Investigate corpus structure. Note that multiple GraphRAG methods also degrade baseline performance on this benchmark. Report findings and pivot to CRQI benchmark where the approach has demonstrated +7-10% advantage on structurally diverse corpora.

## Risks

### Source corpus availability (Critical)

Gate at Step 1.4. If raw text unavailable, contact benchmark authors.

### Judge model comparability

Published scores use GPT-4-turbo. GPT-4.1 is successor at lower cost. Relative rankings are the meaningful comparison.

### Corpus structure

Medical/novel textbooks may lack deep heading hierarchies. Contextual chunking advantage may be understated. Note in report and reference scale sweep results from structurally diverse corpus.

### Creative Generation impact

Level 4 tests LLM creativity, not retrieval. Always report with and without. Lead with without-Creative numbers for enterprise audiences.

## Deliverables

1. Results JSON in GraphRAG-Bench format
2. Evaluation scores (GPT-4o-mini and GPT-4.1 passes)
3. Comparison table against published leaderboard with per-task-level breakdown
4. Cost analysis comparing indexing and per-query costs
5. Written findings suitable for OSS library README, blog post, or paper
