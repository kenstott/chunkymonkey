# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 7a4e2b91-3c88-4f02-b5d1-e920c7f84a3d
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""
Chunky Monkey — GraphRAG-Bench evaluation.

Evaluates contextual chunking against published GraphRAG-Bench leaderboard.
Uses BGE-large-en-v1.5 embeddings + benchmark's native answer_correctness metric.

Usage:
    python demo/graphrag_bench.py download      --out-dir /tmp/grb
    python demo/graphrag_bench.py inspect       --out-dir /tmp/grb
    python demo/graphrag_bench.py index         --out-dir /tmp/grb [--force]
    python demo/graphrag_bench.py index-vanilla --out-dir /tmp/grb [--force]
    python demo/graphrag_bench.py run           --out-dir /tmp/grb [--rerank] [--enhanced] [--vanilla] [--run-name NAME]
    python demo/graphrag_bench.py eval          --out-dir /tmp/grb --run-name <name>
    python demo/graphrag_bench.py report        --out-dir /tmp/grb
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Load .env from project root before anything else
_PROJECT_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(_PROJECT_ROOT))
from chunkymonkey import DocumentLoader, NOVEL_STRUCTURAL_LEVELS
from chunkymonkey import chunk_document, promote_plain_text_headers
from chunkymonkey.context import enrich_chunks
from chunkymonkey.storage._store import Store

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EMBED_MODEL        = "BAAI/bge-large-en-v1.5"
EMBED_DIM          = 1024
GEN_MODEL          = "gpt-4o-mini"
GEN_MODEL_TOGETHER = "Qwen/Qwen2.5-72B-Instruct-Turbo"   # closest serverless Qwen2.5 (14B not available serverless on Together)
TOGETHER_BASE_URL  = "https://api.together.xyz/v1"
K             = 5
K_FETCH       = 20    # candidates to retrieve before reranking (ignored when --rerank is off)
RERANK_MODEL         = "BAAI/bge-reranker-large"       # local cross-encoder
RERANK_MODEL_TOGETHER = "Salesforce/Llama-Rank-V1.1"   # Together AI reranker API
RERANK_MODEL_COHERE  = "rerank-english-v3.0"           # Cohere reranker API
SPACY_MODEL   = "en_core_web_sm"
MIN_CHUNK     = 400
MAX_CHUNK     = 1200
DATASET_NAME  = "GraphRAG-Bench/GraphRAG-Bench"
SUBSETS       = ["medical", "novel"]
DB_FILENAME         = "chunkymonkey.duckdb"
VANILLA_DB_FILENAME = "vanilla_rag.duckdb"
VANILLA_K             = 5     # paper Appendix H.2: retrieval_topk=5
VANILLA_CHUNK_TOKENS  = 256   # benchmark uses 256-token chunks
VANILLA_CHUNK_OVERLAP = 32
VANILLA_TEMPERATURE   = 0.7   # paper: "generation temperature of 0.7"

# Published leaderboard results scraped from graphrag-bench.github.io, April 2026.
# Avg = mean(Fact ACC, Reason ACC, Summ ACC, Creative ACC) for each subset.
# Original scale is 0–100%; stored here as 0–1.
# Published leaderboard uses gpt-4o-mini generator + gpt-4o-mini judge (answer_correctness metric).
PUBLISHED_BASELINES = {
    # ── Top methods ───────────────────────────────────────────────────────────
    "G-reasoner":               {"med_acc": 0.7330, "nov_acc": 0.5894, "overall": 0.6612},
    "AutoPrunedRetriever-llm":  {"med_acc": 0.6700, "nov_acc": 0.6372, "overall": 0.6536},
    "HippoRAG2":                {"med_acc": 0.6485, "nov_acc": 0.5648, "overall": 0.6067},
    "Fast-GraphRAG":            {"med_acc": 0.6412, "nov_acc": 0.5202, "overall": 0.5807},
    "LightRAG":                 {"med_acc": 0.6259, "nov_acc": 0.4509, "overall": 0.5384},
    # ── Vanilla RAG baselines ─────────────────────────────────────────────────
    "RAG (w/ rerank)":          {"med_acc": 0.6243, "nov_acc": 0.4835, "overall": 0.5539},
    "RAG (w/o rerank)":         {"med_acc": 0.6100, "nov_acc": 0.4793, "overall": 0.5447},
    # ── Other methods ─────────────────────────────────────────────────────────
    "RAPTOR":                   {"med_acc": 0.5710, "nov_acc": 0.4324, "overall": 0.5017},
    "MS-GraphRAG (local)":      {"med_acc": 0.4516, "nov_acc": 0.5093, "overall": 0.4805},
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Download
# ─────────────────────────────────────────────────────────────────────────────

def cmd_download(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)

    from datasets import load_dataset

    for subset in SUBSETS:
        out_file = data_dir / f"{subset}_questions.jsonl"
        if out_file.exists():
            print(f"  {subset}: already downloaded ({out_file})")
            continue
        print(f"  Downloading {subset} subset...")
        ds = load_dataset(DATASET_NAME, subset, split="train", trust_remote_code=True)
        with open(out_file, "w") as f:
            for record in ds:
                f.write(json.dumps(record) + "\n")
        print(f"  {subset}: {len(ds)} questions → {out_file}")

    repo_dir = out_dir / "GraphRAG-Benchmark"
    if not repo_dir.exists():
        print("\nCloning GraphRAG-Benchmark repo...")
        ret = os.system(f"git clone --depth=1 https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git {repo_dir} 2>&1")
        if ret != 0:
            print("  WARNING: git clone failed — evaluation scripts unavailable")
    else:
        print(f"\nRepo already cloned at {repo_dir}")

    print("\nDownload complete. Run 'inspect' to check corpus availability.")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.5: Inspect
# ─────────────────────────────────────────────────────────────────────────────

def _load_questions(data_dir: Path) -> list[dict]:
    questions = []
    for subset in SUBSETS:
        f = data_dir / f"{subset}_questions.jsonl"
        if f.exists():
            with open(f) as fh:
                for line in fh:
                    questions.append(json.loads(line))
    return questions


def cmd_inspect(args: argparse.Namespace) -> None:
    out_dir  = Path(args.out_dir)
    data_dir = out_dir / "data"
    repo_dir = out_dir / "GraphRAG-Benchmark"

    questions = _load_questions(data_dir)
    if not questions:
        print("No questions found. Run 'download' first.")
        return

    print(f"Total questions: {len(questions)}")
    by_subset = defaultdict(list)
    by_type   = defaultdict(list)
    for q in questions:
        by_subset[q.get("source", q.get("subset", "?"))].append(q)
        by_type[q.get("question_type", "?")].append(q)

    print("\nBy subset:")
    for k, v in sorted(by_subset.items()):
        print(f"  {k}: {len(v)}")
    print("\nBy question type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {len(v)}")

    q = questions[0]
    print(f"\nRecord keys: {list(q.keys())}")
    print(f"\nSample question ({q.get('source','?')} / {q.get('question_type','?')}):")
    print(f"  Q: {q['question'][:120]}")
    print(f"  A: {str(q['answer'])[:120]}")
    ev = q.get("evidence", [])
    print(f"  Evidence passages: {len(ev)}")
    if ev:
        print(f"  Evidence[0]: {str(ev[0])[:200]}")

    print("\n── Source corpus check ──")
    corpus_dir = repo_dir / "Datasets"
    corpus_files = []
    if corpus_dir.exists():
        for root, _, files in os.walk(corpus_dir):
            for fname in files:
                p = Path(root) / fname
                corpus_files.append(p)
        print(f"  Files in Datasets/: {len(corpus_files)}")
        for p in corpus_files[:20]:
            print(f"    {p.relative_to(repo_dir)}  ({p.stat().st_size:,} bytes)")
        if len(corpus_files) > 20:
            print(f"    ... and {len(corpus_files)-20} more")
    else:
        print("  Datasets/ directory not found in cloned repo.")

    ev_chars = sum(len(str(p)) for q in questions for p in q.get("evidence", []))
    print(f"\n  Evidence reconstruction fallback:")
    print(f"    Total evidence chars across all questions: {ev_chars:,}")
    seen: set[str] = set()
    for q in questions:
        for p in q.get("evidence", []):
            seen.add(str(p).strip())
    print(f"    {len(seen):,} unique passages, ~{sum(len(s) for s in seen)//1000:,}K chars total")

    if not corpus_files:
        print("\n  GATE: No raw corpus found. Will reconstruct from evidence passages.")
    else:
        print("\n  GATE: Source corpus available. Proceeding with full chunking.")

    info = {
        "n_questions":   len(questions),
        "by_subset":     {k: len(v) for k, v in by_subset.items()},
        "by_type":       {k: len(v) for k, v in by_type.items()},
        "corpus_files":  [str(p) for p in corpus_files],
        "n_evidence":    len(seen),
        "corpus_source": "repo" if corpus_files else "evidence_reconstruction",
    }
    (out_dir / "corpus_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nSaved corpus_info.json")


# ─────────────────────────────────────────────────────────────────────────────
# Corpus builder (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _build_corpus(out_dir: Path) -> list[tuple[str, str]]:
    """Returns list of (doc_id, text)."""
    info_file = out_dir / "corpus_info.json"
    if not info_file.exists():
        raise RuntimeError("Run 'inspect' first.")

    info     = json.loads(info_file.read_text())
    data_dir = out_dir / "data"

    if info["corpus_source"] == "repo" and info["corpus_files"]:
        corpus_files = [Path(p) for p in info["corpus_files"]
                        if "Corpus" in p and p.endswith(".json")]
        docs = []
        for path in corpus_files:
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            records = raw if isinstance(raw, list) else [raw]
            for rec in records:
                name = rec.get("corpus_name", path.stem)
                text = rec.get("context", "")
                if isinstance(text, str) and text.strip():
                    docs.append((name, text))
        if docs:
            return docs

    print("  Using evidence reconstruction (no raw corpus available)")
    questions = _load_questions(data_dir)
    seen: dict[str, str] = {}
    for q in questions:
        src = q.get("source", q.get("subset", "unknown"))
        for passage in q.get("evidence", []):
            text = str(passage).strip()
            if text and text not in seen:
                doc_id = f"{src}_ev_{len(seen)}"
                seen[text] = doc_id
    return [(doc_id, text) for text, doc_id in seen.items()]


# ─────────────────────────────────────────────────────────────────────────────
# Naive chunker (vanilla RAG baseline)
# ─────────────────────────────────────────────────────────────────────────────

def _naive_chunks(text: str,
                  chunk_tokens: int = VANILLA_CHUNK_TOKENS,
                  overlap_tokens: int = VANILLA_CHUNK_OVERLAP) -> list[str]:
    """Split text into fixed-size token chunks with overlap using BGE BERT tokenizer."""
    from transformers import AutoTokenizer
    enc = AutoTokenizer.from_pretrained(EMBED_MODEL)
    tokens = enc.encode(text, add_special_tokens=False)
    step = max(1, chunk_tokens - overlap_tokens)
    result = []
    for start in range(0, len(tokens), step):
        chunk = enc.decode(tokens[start:start + chunk_tokens])
        if chunk.strip():
            result.append(chunk)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Index (chunk + embed + store via chunkymonkey)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_index(args: argparse.Namespace) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    out_dir  = Path(args.out_dir)
    data_dir = out_dir / "data"
    db_path  = data_dir / DB_FILENAME

    if db_path.exists() and not args.force:
        with Store(db_path, embedding_dim=EMBED_DIM) as store:
            n = store.count()
        print(f"Index already exists: {n:,} chunks at {db_path}")
        print("Use --force to reindex.")
        return

    if db_path.exists() and args.force:
        db_path.unlink()
        print(f"Removed existing index: {db_path}")

    corpus = _build_corpus(out_dir)
    print(f"Corpus: {len(corpus)} documents")

    print(f"Chunking with header promotion (min={MIN_CHUNK}, max={MAX_CHUNK})...")
    all_chunks = []
    for doc_id, text in corpus:
        is_novel = doc_id.lower().startswith("novel")
        if is_novel:
            promoted = promote_plain_text_headers(
                text,
                promote_questions=False,
                promote_short_phrases=False,
                structural_levels=NOVEL_STRUCTURAL_LEVELS,
                toc_proximity=300,
            )
        else:
            promoted = promote_plain_text_headers(
                text,
                promote_questions=True,
                promote_short_phrases=True,
            )
        chunks = chunk_document(
            doc_id, promoted,
            min_chunk_size=MIN_CHUNK,
            max_chunk_size=MAX_CHUNK,
            include_breadcrumb=True,
            promote_headings=False,  # already promoted above
        )
        chunks = enrich_chunks(chunks, strategy="prefix")
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks):,}")
    avg = sum(len(c.content) for c in all_chunks) / max(1, len(all_chunks))
    print(f"Avg chunk size: {avg:.0f} chars")

    print(f"Embedding {len(all_chunks):,} chunks with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    # Use embedding_content (breadcrumb + content) for embedding quality
    texts = [
        c.embedding_content if c.embedding_content else c.content
        for c in all_chunks
    ]

    batch_size = 256
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs  = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(vecs)
        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 10 == 0:
            print(f"  {done:,}/{len(texts):,}")

    emb = np.vstack(embeddings).astype("float32")
    print(f"Embeddings: {emb.shape}")

    print(f"Storing in {db_path}...")
    with Store(db_path, embedding_dim=EMBED_DIM) as store:
        store.add_document(all_chunks, emb)
        n = store.count()

    print(f"Index complete: {n:,} chunks → {db_path}")


def cmd_index_vanilla(args: argparse.Namespace) -> None:
    """Build vanilla RAG index: naive 256-token fixed chunks, no breadcrumbs."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from chunkymonkey.models import DocumentChunk

    out_dir  = Path(args.out_dir)
    data_dir = out_dir / "data"
    db_path  = data_dir / VANILLA_DB_FILENAME

    if db_path.exists() and not args.force:
        with Store(db_path, embedding_dim=EMBED_DIM) as store:
            n = store.count()
        print(f"Vanilla index exists: {n:,} chunks at {db_path}")
        print("Use --force to reindex.")
        return

    if db_path.exists() and args.force:
        db_path.unlink()
        print(f"Removed existing index: {db_path}")

    corpus = _build_corpus(out_dir)
    print(f"Corpus: {len(corpus)} documents")
    print(f"Naive chunking: {VANILLA_CHUNK_TOKENS}-token chunks, {VANILLA_CHUNK_OVERLAP}-token overlap...")

    all_chunks: list[DocumentChunk] = []
    for doc_id, text in corpus:
        for i, chunk_text in enumerate(_naive_chunks(text)):
            all_chunks.append(DocumentChunk(
                document_name=doc_id,
                chunk_index=i,
                content=chunk_text,
                breadcrumb="",
                embedding_content=chunk_text,
            ))

    print(f"Total vanilla chunks: {len(all_chunks):,}")
    avg = sum(len(c.content) for c in all_chunks) / max(1, len(all_chunks))
    print(f"Avg chunk size: {avg:.0f} chars")

    print(f"Embedding {len(all_chunks):,} chunks with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c.content for c in all_chunks]

    batch_size = 256
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs  = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(vecs)
        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 10 == 0:
            print(f"  {done:,}/{len(texts):,}")

    emb = np.vstack(embeddings).astype("float32")
    print(f"Embeddings: {emb.shape}")

    print(f"Storing in {db_path}...")
    with Store(db_path, embedding_dim=EMBED_DIM) as store:
        store.add_document(all_chunks, emb)
        n = store.count()
    print(f"Vanilla index complete: {n:,} chunks → {db_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Retrieve + generate
# ─────────────────────────────────────────────────────────────────────────────

def _generate(question: str, context: str, client, model: str = GEN_MODEL,
              temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": ("Answer the question based only on the provided context. "
                         "If the context does not contain enough information, "
                         "say so rather than making up an answer.")},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=temperature,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


def _build_enhanced_search(store):
    """Build EnhancedSearch with SpacyMatcher NER + agglomerative ClusterMap."""
    from chunkymonkey.ner import SpacyMatcher, EntityIndex
    from chunkymonkey.cluster import ClusterMap
    from chunkymonkey.search import EnhancedSearch
    from chunkymonkey.storage._vector import DuckDBVectorBackend

    print(f"Building EntityIndex with SpacyMatcher({SPACY_MODEL})...")
    matcher = SpacyMatcher(model=SPACY_MODEL, strip_numeric=True)
    entity_index = EntityIndex()

    all_chunks = store.vector.get_all_chunks()
    print(f"  Running NER on {len(all_chunks):,} chunks...")
    for chunk in all_chunks:
        embed_content = chunk.embedding_content if chunk.embedding_content else chunk.content
        chunk_id = DuckDBVectorBackend._generate_chunk_id(
            chunk.document_name, chunk.chunk_index, embed_content
        )
        entity_index.run_ner(chunk_id, chunk.content, matcher)

    entity_index.recompute_scores()
    print(f"  {entity_index.total_chunks():,} chunks, {len(entity_index.entity_ids()):,} entities")

    print("  Building ClusterMap...")
    cluster_map = ClusterMap.build(entity_index)
    print(f"  {cluster_map.cluster_count():,} clusters across {cluster_map.entity_count():,} entities")

    # Disable structural_expansion: _structural_neighbors scans all chunks per query
    return EnhancedSearch(
        store,
        entity_index=entity_index,
        cluster_map=cluster_map,
        structural_expansion=False,
    )


def cmd_run(args: argparse.Namespace) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import openai

    out_dir     = Path(args.out_dir)
    data_dir    = out_dir / "data"
    results_dir = out_dir / "results"
    results_dir.mkdir(exist_ok=True)
    run_name     = getattr(args, "run_name", "contextual")

    # Kill any stale processes already running the same run-name
    import signal, subprocess as _sp
    _my_pid = os.getpid()
    try:
        _procs = _sp.check_output(
            ["pgrep", "-f", f"graphrag_bench.py.*--run-name {run_name}"],
            text=True,
        ).split()
        for _pid in _procs:
            _pid = int(_pid)
            if _pid != _my_pid:
                os.kill(_pid, signal.SIGKILL)
                print(f"[preflight] killed stale pid {_pid} ({run_name})", flush=True)
    except _sp.CalledProcessError:
        pass  # no matching processes
    results_f    = results_dir / f"{run_name}.jsonl"
    ckpt_f       = results_dir / f"{run_name}_checkpoint.jsonl"
    use_vanilla  = getattr(args, "vanilla", False)
    db_path      = data_dir / (VANILLA_DB_FILENAME if use_vanilla else DB_FILENAME)
    top_k        = VANILLA_K if use_vanilla else K
    gen_temperature = VANILLA_TEMPERATURE  # paper: 0.7 for all systems
    use_rerank        = getattr(args, "rerank", False)
    rerank_provider   = getattr(args, "rerank_provider", "local")
    use_enhanced      = getattr(args, "enhanced", False)

    if not db_path.exists():
        print("No index found. Run 'index' first.")
        return

    questions = _load_questions(data_dir)
    if args.limit:
        questions = questions[:args.limit]
    print(f"Questions: {len(questions)}")

    done_ids: set[str] = set()
    if ckpt_f.exists():
        with open(ckpt_f) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r["id"])
        print(f"Resuming from checkpoint: {len(done_ids)} already done")

    pending = [(i, q) for i, q in enumerate(questions)
               if q.get("id", f"q{i}") not in done_ids]
    print(f"Pending: {len(pending)}")

    # ── 1. Embed all questions (cached — questions never change across runs)
    import numpy as np
    q_vecs_cache = data_dir / "question_embeddings.npy"
    q_ids_cache  = data_dir / "question_ids.json"
    all_ids = [q.get("id", f"q{i}") for i, q in enumerate(questions)]

    if q_vecs_cache.exists() and q_ids_cache.exists():
        cached_ids = json.loads(q_ids_cache.read_text())
        if cached_ids == all_ids:
            print(f"Loading cached question embeddings from {q_vecs_cache}")
            all_vecs = np.load(str(q_vecs_cache))
        else:
            q_vecs_cache.unlink()  # stale cache
            cached_ids = None

    if not q_vecs_cache.exists():
        embed_model = SentenceTransformer(EMBED_MODEL)
        print(f"Embedding {len(questions)} questions (will cache)...")
        all_texts = [q["question"] for q in questions]
        all_vecs  = embed_model.encode(
            all_texts, normalize_embeddings=True,
            show_progress_bar=False, batch_size=256,
        ).astype("float32")
        np.save(str(q_vecs_cache), all_vecs)
        q_ids_cache.write_text(json.dumps(all_ids))
        print(f"Cached → {q_vecs_cache}")

    # slice to pending indices only
    pending_indices = [i for i, _ in pending]
    q_vecs = all_vecs[pending_indices]

    # ── 2. Retrieve context for each question (sequential; DuckDB conn is serialized)
    fetch_k = K_FETCH if use_rerank else top_k
    print(f"Retrieving context from index (fetch_k={fetch_k}, k={top_k}, rerank={use_rerank}, enhanced={use_enhanced}, vanilla={use_vanilla})...")

    reranker = None
    together_rerank_client = None
    cohere_rerank_client = None
    if use_rerank:
        if rerank_provider == "together":
            from together import Together
            together_rerank_client = Together(api_key=os.environ["TOGETHER_API_KEY"])
            print(f"Using Together reranker: {RERANK_MODEL_TOGETHER}")
        elif rerank_provider == "cohere":
            import cohere
            cohere_rerank_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
            print(f"Using Cohere reranker: {RERANK_MODEL_COHERE}")
        else:
            from sentence_transformers import CrossEncoder
            print(f"Loading reranker: {RERANK_MODEL}...")
            reranker = CrossEncoder(RERANK_MODEL)

    work_items: list[dict] = []
    with Store(db_path, embedding_dim=EMBED_DIM, read_only=True) as store:
        enhanced_search = _build_enhanced_search(store) if use_enhanced else None

        for j, (i, q) in enumerate(pending):
            qid  = q.get("id", f"q{i}")
            if use_enhanced and enhanced_search is not None:
                scored = enhanced_search.search(q_vecs[j], k=fetch_k, query_text=q["question"])
                hits   = [(sc.chunk_id, sc.score, sc.chunk) for sc in scored]
            else:
                hits = store.vector.search(
                    q_vecs[j], limit=fetch_k,
                    query_text=q["question"],
                    include_breadcrumbs=False,
                )
            if use_rerank and together_rerank_client is not None:
                docs   = [chunk.content for _, _, chunk in hits]
                resp   = together_rerank_client.rerank.create(
                    model=RERANK_MODEL_TOGETHER,
                    query=q["question"],
                    documents=docs,
                    top_n=top_k,
                )
                hits   = [hits[r.index] for r in resp.results]
            elif use_rerank and cohere_rerank_client is not None:
                docs   = [chunk.content for _, _, chunk in hits]
                resp   = cohere_rerank_client.rerank(
                    model=RERANK_MODEL_COHERE,
                    query=q["question"],
                    documents=docs,
                    top_n=top_k,
                )
                hits   = [hits[r.index] for r in resp.results]
            elif use_rerank and reranker is not None:
                pairs  = [(q["question"], chunk.content) for _, _, chunk in hits]
                scores = reranker.predict(pairs)
                ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)[:top_k]
                hits   = [h for _, h in ranked]
            elif not use_rerank:
                hits = hits[:top_k]
            work_items.append({
                "_slot":      len(work_items),   # for round-robin endpoint selection
                "qid":        qid,
                "question":   q["question"],
                "source":     q.get("source", q.get("subset", "?")),
                "qtype":      q.get("question_type", "?"),
                "context":    "\n\n".join(chunk.content for _, _, chunk in hits),
                "chunk_ids":  [cid for cid, _, _ in hits],
                "scores":     [float(sc) for _, sc, _ in hits],
                "evidence":   q.get("evidence", []),
                "gold":       str(q.get("answer", "")),
            })

    # Build one client per endpoint (round-robin for parallelism across multiple dedicated endpoints)
    endpoint_ids: list[str] = getattr(args, "endpoint_ids", None) or [args.gen_model]
    if args.gen_provider == "together":
        clients = [
            openai.OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url=TOGETHER_BASE_URL, timeout=120.0)
            for _ in endpoint_ids
        ]
    else:
        clients = [openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)]
        endpoint_ids = [args.gen_model]

    n_endpoints = len(clients)
    print(f"Generating answers with {args.concurrency} parallel workers across {n_endpoints} endpoint(s)...")

    new_results: list[dict] = []
    ckpt_lock   = threading.Lock()
    done_count  = [len(done_ids)]

    def _process(item: dict) -> dict:
        # round-robin client + model selection by item slot index
        slot   = item["_slot"] % n_endpoints
        client = clients[slot]
        model  = endpoint_ids[slot]
        for attempt in range(3):
            try:
                answer = _generate(item["question"], item["context"], client, model,
                                   temperature=gen_temperature)
                break
            except Exception as exc:
                if attempt == 2:
                    answer = f"[ERROR: {exc}]"
                else:
                    time.sleep(2 ** attempt)
        return {
            "id":               item["qid"],
            "question":         item["question"],
            "source":           item["source"],
            "question_type":    item["qtype"],
            "context":          item["context"],
            "evidence":         item["evidence"],
            "generated_answer": answer,
            "gold_answer":      item["gold"],
            "retrieved_chunks": item["chunk_ids"],
            "retrieved_scores": item["scores"],
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_process, item): item for item in work_items}
        for fut in as_completed(futures):
            result = fut.result()
            with ckpt_lock:
                new_results.append(result)
                done_count[0] += 1
                total = done_count[0]
                if total % 50 == 0:
                    print(f"  {total}/{len(questions)}", flush=True)
                # Checkpoint every 100 completions
                if len(new_results) % 100 == 0:
                    with open(ckpt_f, "a") as f:
                        for r in new_results[-100:]:
                            f.write(json.dumps(r) + "\n")
                    print(f"  {total}/{len(questions)}  (checkpoint saved)", flush=True)

    # Final checkpoint flush + merge
    with open(ckpt_f, "a") as f:
        checkpointed = set()
        if ckpt_f.exists():
            pass  # already flushed incrementally
        remainder = len(new_results) % 100
        if remainder:
            for r in new_results[-remainder:]:
                f.write(json.dumps(r) + "\n")

    all_results: list[dict] = []
    if ckpt_f.exists():
        with open(ckpt_f) as f:
            for line in f:
                all_results.append(json.loads(line))
    ckpt_ids = {r["id"] for r in all_results}
    for r in new_results:
        if r["id"] not in ckpt_ids:
            all_results.append(r)

    with open(results_f, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nComplete: {len(all_results)} results → {results_f}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Evaluate (benchmark's native metric)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Together dedicated endpoint lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def cmd_bench_eval(args: argparse.Namespace) -> None:
    """Run benchmark's native generation_eval.py on our output with checkpointing."""
    import asyncio
    import sys
    import numpy as np

    out_dir     = Path(args.out_dir)
    results_dir = out_dir / "results"
    repo_dir    = out_dir / "GraphRAG-Benchmark"
    run_name    = getattr(args, "run_name", "contextual")
    results_f   = results_dir / f"{run_name}.jsonl"
    ckpt_f      = results_dir / f"bench_eval_ckpt_{run_name}.jsonl"
    out_f       = results_dir / f"bench_eval_{run_name}.json"

    if not results_f.exists():
        print(f"No results found: {results_f}. Run 'run' first.")
        return
    if not repo_dir.exists():
        print(f"Benchmark repo not found at {repo_dir}. Run 'download' first.")
        return

    # Load our results
    records = [json.loads(line) for line in open(results_f)]
    if args.limit:
        records = records[:args.limit]

    # Convert to benchmark format
    bench_records = [{
        "id":            r["id"],
        "question":      r["question"],
        "question_type": r["question_type"],
        "generated_answer": r["generated_answer"],
        "ground_truth":  r.get("gold_answer") or r.get("ground_truth", ""),
        "context":       [r["context"]],
    } for r in records]

    # Load checkpoint
    done: dict[str, dict] = {}
    if ckpt_f.exists():
        for line in open(ckpt_f):
            item = json.loads(line)
            done[item["id"]] = item
        print(f"Resuming bench-eval: {len(done)} already done")

    pending = [r for r in bench_records if r["id"] not in done]
    print(f"Pending: {len(pending)} samples")

    if not pending:
        print("All samples already evaluated.")
    else:
        # Add benchmark repo to path
        sys.path.insert(0, str(repo_dir))
        from pydantic import SecretStr
        from langchain_openai import ChatOpenAI
        from Evaluation.metrics import (
            compute_answer_correctness, compute_coverage_score,
            compute_faithfulness_score, compute_rouge_score,
        )

        from tenacity import retry, wait_exponential, retry_if_exception_type, stop_after_attempt
        import httpx

        judge_provider = getattr(args, "judge_provider", "openai")
        _judge_kwargs = dict(
            model=args.judge,
            temperature=0.0,
            top_p=1,
            seed=42,
            presence_penalty=0,
            frequency_penalty=0,
            max_retries=3,
            timeout=30,
        )
        if judge_provider == "together":
            llm = ChatOpenAI(
                **_judge_kwargs,
                base_url="https://api.together.xyz/v1",
                api_key=SecretStr(os.environ["TOGETHER_API_KEY"]),
            )
        else:
            llm = ChatOpenAI(
                **_judge_kwargs,
                base_url="https://api.openai.com/v1",
                api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
            )

        # ── Pre-compute embeddings in one batched pass ──────────────────────
        # Ground truth embeddings are shared across runs; cache to disk.
        from sentence_transformers import SentenceTransformer
        from langchain_core.embeddings import Embeddings as LCEmbeddings

        gt_cache_f = out_dir / "data" / "gt_embeddings.npy"
        gt_id_f    = out_dir / "data" / "gt_embedding_ids.json"

        embed_model = SentenceTransformer(EMBED_MODEL)

        # Ground truth embeddings (cached globally)
        gt_texts = [r["ground_truth"] for r in bench_records]
        gt_ids   = [r["id"] for r in bench_records]
        if gt_cache_f.exists() and gt_id_f.exists() and json.loads(gt_id_f.read_text()) == gt_ids:
            print(f"Loading cached ground-truth embeddings...")
            gt_vecs = np.load(str(gt_cache_f))
        else:
            print(f"Encoding {len(gt_texts)} ground-truth texts (batched)...")
            gt_vecs = embed_model.encode(
                gt_texts, batch_size=32, normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            np.save(str(gt_cache_f), gt_vecs)
            gt_id_f.write_text(json.dumps(gt_ids))
            print(f"  Cached → {gt_cache_f}")

        # Answer embeddings (per run — only pending items)
        ans_texts  = [r["generated_answer"] for r in pending]
        print(f"Encoding {len(ans_texts)} generated answers (batched)...")
        ans_vecs = embed_model.encode(
            ans_texts, batch_size=32, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        del embed_model  # free memory

        # Build lookup: text → embedding vector
        emb_lookup: dict[str, np.ndarray] = {}
        for r, vec in zip(bench_records, gt_vecs):
            emb_lookup[r["ground_truth"]] = vec
        for r, vec in zip(pending, ans_vecs):
            emb_lookup[r["generated_answer"]] = vec

        class CachedEmbeddings(LCEmbeddings):
            """Returns pre-computed embeddings by text lookup; never calls the model."""
            def embed_documents(self, texts):
                return [emb_lookup[t].tolist() for t in texts]
            def embed_query(self, text):
                return emb_lookup[text].tolist()
            async def aembed_query(self, text):
                return emb_lookup[text].tolist()
            async def aembed_documents(self, texts):
                return [emb_lookup[t].tolist() for t in texts]

        embedding = CachedEmbeddings()

        METRIC_CONFIG = {
            "Fact Retrieval":       ["rouge_score", "answer_correctness"],
            "Complex Reasoning":    ["rouge_score", "answer_correctness"],
            "Contextual Summarize": ["answer_correctness", "coverage_score"],
            "Creative Generation":  ["answer_correctness", "coverage_score", "faithfulness"],
        }

        semaphore = asyncio.Semaphore(args.concurrency)

        async def _eval_one(r: dict) -> dict:
            import asyncio as _aio
            qtype   = r["question_type"]
            metrics = METRIC_CONFIG.get(qtype, ["answer_correctness"])
            result  = {"id": r["id"], "question_type": qtype}
            async with semaphore:
                for attempt in range(5):
                    try:
                        tasks = {}
                        if "rouge_score" in metrics:
                            tasks["rouge_score"] = compute_rouge_score(r["generated_answer"], r["ground_truth"])
                        if "answer_correctness" in metrics:
                            tasks["answer_correctness"] = compute_answer_correctness(
                                r["question"], r["generated_answer"], r["ground_truth"], llm, embedding
                            )
                        if "coverage_score" in metrics:
                            tasks["coverage_score"] = compute_coverage_score(
                                r["question"], r["ground_truth"], r["generated_answer"], llm
                            )
                        if "faithfulness" in metrics:
                            tasks["faithfulness"] = compute_faithfulness_score(
                                r["question"], r["generated_answer"], r["context"], llm
                            )
                        vals = await asyncio.gather(*tasks.values(), return_exceptions=True)
                        for key, val in zip(tasks.keys(), vals):
                            if isinstance(val, BaseException):
                                print(f"[eval] {r['id']} {key} exception: {type(val).__name__}: {val}", flush=True)
                            result[key] = float(val) if isinstance(val, (int, float)) else float("nan")
                        break
                    except Exception as e:
                        if "429" in str(e) or "rate_limit" in str(e).lower():
                            await _aio.sleep(10 * (2 ** attempt))
                        else:
                            for key in METRIC_CONFIG.get(qtype, ["answer_correctness"]):
                                result.setdefault(key, float("nan"))
                            break
            return result

        async def _run_all():
            import asyncio as _aio
            batch_size = 10
            for batch_start in range(0, len(pending), batch_size):
                batch = pending[batch_start:batch_start + batch_size]
                results_batch = await _aio.gather(*[_eval_one(r) for r in batch], return_exceptions=True)
                with open(ckpt_f, "a") as f:
                    for item in results_batch:
                        if isinstance(item, dict):
                            done[item["id"]] = item
                            f.write(json.dumps(item) + "\n")
                completed = min(batch_start + batch_size, len(pending))
                print(f"  {completed}/{len(pending)} evaluated", flush=True)
                await _aio.sleep(3)   # pace between batches to avoid TPM burst

        asyncio.run(_run_all())

    # Aggregate results by question type
    by_type: dict[str, list] = defaultdict(list)
    for item in done.values():
        by_type[item["question_type"]].append(item)

    aggregated: dict[str, dict] = {}
    for qtype, items in by_type.items():
        agg: dict[str, float] = {}
        for key in ["rouge_score", "answer_correctness", "coverage_score", "faithfulness"]:
            vals = [i[key] for i in items if key in i and not (isinstance(i[key], float) and i[key] != i[key])]
            if vals:
                agg[key] = float(np.nanmean(vals))
        aggregated[qtype] = agg

    with open(out_f, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nBench eval complete → {out_f}")
    for qtype, scores in aggregated.items():
        print(f"  {qtype}: " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))


def cmd_bench_report(args: argparse.Namespace) -> None:
    """Combined matrix: our runs + full leaderboard, scored with benchmark's native metric."""
    out_dir     = Path(args.out_dir)
    results_dir = out_dir / "results"

    # Discover all bench_eval_*.json files
    run_results: dict[str, dict] = {}
    for p in sorted(results_dir.glob("bench_eval_*.json")):
        if "ckpt" in p.stem:
            continue
        run_name = p.stem.replace("bench_eval_", "")
        data = json.loads(p.read_text())
        # Compute avg answer_correctness across all question types
        scores = []
        med_scores, nov_scores = [], []
        for _qtype, metrics in data.items():
            ac = metrics.get("answer_correctness", float("nan"))
            if ac == ac:  # not nan
                scores.append(ac)
        run_results[run_name] = {"scores_by_type": data, "avg_acc": float(sum(scores)/len(scores)) if scores else float("nan")}

    # Load source info to split med/nov
    # For now just print overall (splitting requires per-item source which bench_eval aggregates away)
    QUESTION_TYPES = ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]

    print("\n── Benchmark Eval Results (answer_correctness, benchmark native metric) ──\n")
    header = f"  {'Run':<28}  {'Fact':>7}  {'Reason':>7}  {'Summ':>7}  {'Creative':>9}  {'Avg':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for run_name, info in run_results.items():
        d = info["scores_by_type"]
        fact  = d.get("Fact Retrieval",       {}).get("answer_correctness", float("nan"))
        rsn   = d.get("Complex Reasoning",    {}).get("answer_correctness", float("nan"))
        summ  = d.get("Contextual Summarize", {}).get("answer_correctness", float("nan"))
        crea  = d.get("Creative Generation",  {}).get("answer_correctness", float("nan"))
        vals  = [v for v in [fact, rsn, summ, crea] if v == v]
        avg   = sum(vals) / len(vals) if vals else float("nan")
        fmt   = lambda v: f"{v:.3f}" if v == v else "  —  "  # noqa: E731
        print(f"  {run_name:<28}  {fmt(fact):>7}  {fmt(rsn):>7}  {fmt(summ):>7}  {fmt(crea):>9}  {fmt(avg):>7}")

    # Leaderboard (their metric, gpt-4o-mini generator + judge)
    # Qwen2.5-14B appendix numbers from paper (Table 7, RAG w/ rerank)
    # Combined med+nov avg per question type is not directly available — showing overall acc
    print("\n── Published Leaderboard (Qwen2.5-14B generator, gpt-4o-mini judge) ──\n")
    print(f"  {'Method':<28}  {'Med ACC':>8}  {'Nov ACC':>8}  {'Overall':>8}")
    print("  " + "-" * 58)
    for name, scores in PUBLISHED_BASELINES.items():
        med = f"{scores['med_acc']:.3f}" if scores["med_acc"] is not None else "   —"
        nov = f"{scores['nov_acc']:.3f}" if scores["nov_acc"] is not None else "   —"
        ov  = f"{scores['overall']:.3f}" if scores["overall"] is not None else "   —"
        print(f"  {name:<28}  {med:>8}  {nov:>8}  {ov:>8}")

    print(f"\n  * Leaderboard: gpt-4o-mini generator + gpt-4o-mini judge (main results).")
    print(f"  * Our bench-eval: gpt-4o-mini judge, same answer_correctness metric.")


# ─────────────────────────────────────────────────────────────────────────────
# Together dedicated endpoint lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def cmd_use_endpoints(args: argparse.Namespace) -> None:
    """Run benchmark using pre-existing Together dedicated endpoints."""
    import types
    run_name = getattr(args, "run_name", "contextual")
    rerank   = getattr(args, "rerank", False)
    enhanced = getattr(args, "enhanced", False)
    run_args = types.SimpleNamespace(
        out_dir=args.out_dir,
        gen_provider="together",
        gen_model=args.endpoint_ids[0],
        endpoint_ids=args.endpoint_ids,
        concurrency=args.concurrency * len(args.endpoint_ids),
        limit=None,
        run_name=run_name,
        rerank=rerank,
        enhanced=enhanced,
    )
    cmd_run(run_args)

    eval_args = types.SimpleNamespace(
        out_dir=args.out_dir,
        judge=args.judge,
        limit=None,
        concurrency=20,
        run_name=run_name,
    )
    cmd_bench_eval(eval_args)


def cmd_provision(args: argparse.Namespace) -> None:
    """Create N Together dedicated endpoints in parallel, run benchmark, stop all."""
    import time
    import types
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from together import Together
    from together.types.autoscaling_param import AutoscalingParam

    together_client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    n = args.num_endpoints

    # Store (id, name) tuples — API calls need id, generation needs name
    def _create_one(i: int) -> tuple[str, str]:
        ep = together_client.endpoints.create(
            model=args.model,
            display_name=f"{args.model.replace('/', '_')}_{i}",
            hardware=args.hardware,
            autoscaling=AutoscalingParam(min_replicas=1, max_replicas=1),
        )
        return (ep.id, ep.name)

    print(f"Creating {n} dedicated endpoint(s) for {args.model} on {args.hardware}...")
    with ThreadPoolExecutor(max_workers=n) as ex:
        ep_tuples = list(ex.map(_create_one, range(n)))
    endpoint_ids   = [t[0] for t in ep_tuples]   # internal IDs for lifecycle ops
    endpoint_names = [t[1] for t in ep_tuples]   # names for API model param
    print(f"Endpoints created (ids): {endpoint_ids}")
    print(f"Endpoint names: {endpoint_names}")

    def _wait_one(eid: str) -> bool:
        for _ in range(120):
            time.sleep(10)
            state = getattr(together_client.endpoints.retrieve(eid), "state", "unknown")
            print(f"  {eid[:20]}… [{state}]", flush=True)
            if state == "STARTED":
                return True
        return False

    print("Waiting for all endpoints to be ready...")
    with ThreadPoolExecutor(max_workers=n) as ex:
        ready = list(ex.map(_wait_one, endpoint_ids))

    failed = [eid for eid, ok in zip(endpoint_ids, ready) if not ok]
    for eid in failed:
        together_client.endpoints.update(eid, state="STOPPED")
        endpoint_ids.remove(eid)

    if not endpoint_ids:
        print("No endpoints became ready. Aborting.")
        return

    print(f"\n{len(endpoint_ids)} endpoint(s) ready. Starting benchmark...")

    try:
        args.gen_provider  = "together"
        args.gen_model     = endpoint_names[0]   # fallback for single-endpoint path
        args.endpoint_ids  = endpoint_names       # names are used as model param
        args.concurrency   = args.concurrency * len(endpoint_names)
        args.limit         = None
        cmd_run(args)

        eval_args = types.SimpleNamespace(
            out_dir=args.out_dir,
            judge="gpt-4.1",
            limit=None,
            concurrency=20,
        )
        cmd_bench_eval(eval_args)
    finally:
        print("\nStopping all endpoints...")
        for eid in endpoint_ids:
            together_client.endpoints.update(eid, state="STOPPED")
            print(f"  Stopped {eid}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Chunky Monkey GraphRAG-Bench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # download
    p = sub.add_parser("download", help="Download dataset and clone benchmark repo")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.set_defaults(func=cmd_download)

    # inspect
    p = sub.add_parser("inspect", help="Inspect dataset structure and check corpus availability")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.set_defaults(func=cmd_inspect)

    # index (replaces chunk + embed)
    p = sub.add_parser("index", help="Chunk + embed + store corpus via chunkymonkey")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--force", action="store_true", help="Delete existing index and reindex")
    p.set_defaults(func=cmd_index)

    # index-vanilla
    p = sub.add_parser("index-vanilla", help="Build vanilla RAG index: naive 256-token chunks")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--force", action="store_true", help="Delete existing index and reindex")
    p.set_defaults(func=cmd_index_vanilla)

    # run
    p = sub.add_parser("run", help="Retrieve and generate answers for all questions")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--limit",       type=int, default=None, metavar="N",
                   help="Limit to first N questions (for testing)")
    p.add_argument("--gen-model",     default=GEN_MODEL,
                   help=f"Generation model (default: {GEN_MODEL})")
    p.add_argument("--gen-provider",  default="openai", choices=["openai", "together"],
                   help="API provider for generation: openai or together (default: openai)")
    p.add_argument("--concurrency",   type=int, default=20,
                   help="Parallel workers (default: 20)")
    p.add_argument("--run-name",      default="contextual",
                   help="Output file prefix (default: contextual)")
    p.add_argument("--rerank",        action="store_true",
                   help=f"Rerank top-{K_FETCH} candidates to top-{K}")
    p.add_argument("--rerank-provider", default="local", choices=["local", "together", "cohere"],
                   help=f"Reranker: local={RERANK_MODEL}, together={RERANK_MODEL_TOGETHER} (default: local)")
    p.add_argument("--enhanced",      action="store_true",
                   help=f"Use NER+cluster EnhancedSearch (SpacyMatcher/{SPACY_MODEL} + agglomerative clustering)")
    p.add_argument("--vanilla",       action="store_true",
                   help=f"Use vanilla RAG index ({VANILLA_CHUNK_TOKENS}-token chunks, k={VANILLA_K})")
    p.set_defaults(func=cmd_run)

    # use-endpoints — run benchmark against existing Together dedicated endpoints
    p = sub.add_parser("use-endpoints", help="Run benchmark using existing Together dedicated endpoints")
    p.add_argument("endpoint_ids", nargs="+", metavar="ENDPOINT_ID",
                   help="One or more Together dedicated endpoint IDs")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--concurrency", type=int, default=20,
                   help="Workers per endpoint (default: 20, total = N * 20)")
    p.add_argument("--judge",       default="gpt-4.1")
    p.add_argument("--run-name",    default="contextual",
                   help="Output file prefix (default: contextual)")
    p.add_argument("--rerank",      action="store_true",
                   help=f"Rerank top-{K_FETCH} candidates to top-{K} with {RERANK_MODEL}")
    p.add_argument("--enhanced",    action="store_true",
                   help=f"Use NER+cluster EnhancedSearch ({SPACY_MODEL} + agglomerative clustering)")
    p.set_defaults(func=cmd_use_endpoints)

    # provision — create Together dedicated endpoint, run benchmark, stop endpoint
    p = sub.add_parser("provision", help="Create Together dedicated endpoint, run benchmark, stop endpoint")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--model",       default="Qwen/Qwen2.5-14B-Instruct",
                   help="Model to deploy (default: Qwen/Qwen2.5-14B-Instruct)")
    p.add_argument("--hardware",       default="2x_nvidia_h100_80gb_sxm",
                   help="Together hardware tier (default: 2x_nvidia_h100_80gb_sxm)")
    p.add_argument("--num-endpoints",  type=int, default=1,
                   help="Number of parallel dedicated endpoints (default: 1)")
    p.add_argument("--concurrency",    type=int, default=20,
                   help="Workers per endpoint (default: 20, total = N * 20)")
    p.set_defaults(func=cmd_provision)

    # eval — score a run with benchmark's native answer_correctness metric
    p = sub.add_parser("eval", help="Score a run with benchmark's native answer_correctness metric")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--run-name",    default="contextual",
                   help="Results file prefix to evaluate (default: contextual)")
    p.add_argument("--judge",          default="gpt-4o-mini",
                   help="Judge model (default: gpt-4o-mini)")
    p.add_argument("--judge-provider", default="openai", choices=["openai", "together"],
                   help="API provider for judge (default: openai)")
    p.add_argument("--limit",          type=int, default=None, metavar="N")
    p.add_argument("--concurrency",    type=int, default=5,
                   help="Parallel workers (default: 5)")
    p.set_defaults(func=cmd_bench_eval)

    # report — combined matrix of all eval runs + leaderboard
    p = sub.add_parser("report", help="Combined matrix: our eval runs + leaderboard")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.set_defaults(func=cmd_bench_report)

    return ap


def main() -> None:
    ap   = _make_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
