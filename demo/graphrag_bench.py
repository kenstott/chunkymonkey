# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 7a4e2b91-3c88-4f02-b5d1-e920c7f84a3d
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""
Chunky Monkey — GraphRAG-Bench evaluation.

Evaluates contextual chunking against published GraphRAG-Bench leaderboard.
Uses BGE-large-en-v1.5 embeddings via chunkymonkey Store + GPT-4o-mini generation/judge.

Usage:
    python demo/graphrag_bench.py download  --out-dir /tmp/grb
    python demo/graphrag_bench.py inspect   --out-dir /tmp/grb
    python demo/graphrag_bench.py index     --out-dir /tmp/grb [--force]
    python demo/graphrag_bench.py run       --out-dir /tmp/grb [--limit N]
    python demo/graphrag_bench.py evalrun   --out-dir /tmp/grb [--judge gpt-4o-mini]
    python demo/graphrag_bench.py report    --out-dir /tmp/grb
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
from chunkymonkey import DocumentLoader
from chunkymonkey.storage._store import Store

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EMBED_MODEL        = "BAAI/bge-large-en-v1.5"
EMBED_DIM          = 1024
GEN_MODEL          = "gpt-4.1"
GEN_MODEL_TOGETHER = "Qwen/Qwen2.5-72B-Instruct-Turbo"   # paper used 14B; 72B available on Together
TOGETHER_BASE_URL  = "https://api.together.xyz/v1"
K             = 5
MIN_CHUNK     = 400
MAX_CHUNK     = 1200
DATASET_NAME  = "GraphRAG-Bench/GraphRAG-Bench"
SUBSETS       = ["medical", "novel"]
DB_FILENAME   = "chunkymonkey.duckdb"

# Published leaderboard results (from GraphRAG-Bench paper / leaderboard)
# Scores from: "When to use Graphs in RAG" (ICLR'26), Table 2.
# Original scale is 0–100%; stored here as 0–1.  Averages are unweighted
# across the 4 question types (Fact/Complex/Summarize/Creative).
# Note: paper uses Qwen2.5-14B generator + GPT-4 judge; our run uses
# GPT-4o-mini for both generation and judging, so numbers are not
# directly comparable — treat as rough order-of-magnitude reference.
PUBLISHED_BASELINES = {
    # RAG w/ rerank — best vanilla baseline in the paper
    "Vanilla RAG (rerank)": {"med_acc": 0.624, "nov_acc": 0.484, "overall": 0.555},
    # MS GraphRAG local — medical not reported in paper
    "MS GraphRAG (local)":  {"med_acc": None,  "nov_acc": 0.509, "overall": None},
    "LightRAG":             {"med_acc": 0.626, "nov_acc": 0.451, "overall": 0.540},
    "HippoRAG2":            {"med_acc": 0.648, "nov_acc": 0.565, "overall": 0.607},
    "fast-graphrag":        {"med_acc": 0.641, "nov_acc": 0.520, "overall": 0.582},
    "RAPTOR":               {"med_acc": 0.553, "nov_acc": 0.432, "overall": 0.493},
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

    loader = DocumentLoader(
        min_chunk_size=MIN_CHUNK,
        max_chunk_size=MAX_CHUNK,
        context_strategy="prefix",
    )

    print(f"Chunking with DocumentLoader (min={MIN_CHUNK}, max={MAX_CHUNK})...")
    all_chunks = []
    for doc_id, text in corpus:
        chunks = loader.load_text(text, doc_id)
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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Retrieve + generate
# ─────────────────────────────────────────────────────────────────────────────

def _generate(question: str, context: str, client, model: str = GEN_MODEL) -> str:
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
        temperature=0.0,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


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
    results_f = results_dir / "contextual.jsonl"
    ckpt_f    = results_dir / "contextual_checkpoint.jsonl"
    db_path   = data_dir / DB_FILENAME

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
    print("Retrieving context from index...")
    work_items: list[dict] = []
    with Store(db_path, embedding_dim=EMBED_DIM) as store:
        for j, (i, q) in enumerate(pending):
            qid  = q.get("id", f"q{i}")
            hits = store.vector.search(
                q_vecs[j], limit=K,
                query_text=q["question"],
                include_breadcrumbs=False,
            )
            work_items.append({
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

    print(f"Generating answers with {args.concurrency} parallel workers ({args.gen_model} via {args.gen_provider})...")
    if args.gen_provider == "together":
        client = openai.OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url=TOGETHER_BASE_URL,
            timeout=120.0,
        )
    else:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)

    new_results: list[dict] = []
    ckpt_lock   = threading.Lock()
    done_count  = [len(done_ids)]

    def _process(item: dict) -> dict:
        for attempt in range(3):
            try:
                answer = _generate(item["question"], item["context"], client, args.gen_model)
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
# Phase 4/5: Evaluate
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_ACCURACY_PROMPT = """\
You are evaluating a RAG system answer against a gold reference answer.

Question: {question}
Gold answer: {gold}
Generated answer: {generated}

Score the generated answer:
2 = Correct and complete (captures the key information in the gold answer)
1 = Partially correct (captures some but not all key information)
0 = Incorrect or irrelevant

Respond with a single integer (0, 1, or 2) and nothing else."""

_JUDGE_COVERAGE_PROMPT = """\
You are evaluating whether a retrieved context contains enough information to answer a question.

Question: {question}
Gold answer: {gold}
Retrieved context: {context}

Does the retrieved context contain sufficient information to produce the gold answer?
2 = Yes, the context clearly contains the information needed
1 = Partially — context has some relevant information but is incomplete
0 = No — the context does not contain the necessary information

Respond with a single integer (0, 1, or 2) and nothing else."""


def _llm_judge(prompt: str, client, model: str) -> float:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            text = resp.choices[0].message.content.strip()
            m    = re.search(r"[012]", text)
            return int(m.group()) / 2.0 if m else 0.0
        except Exception:
            if attempt == 2:
                return 0.0
            time.sleep(2 ** attempt)
    return 0.0


def _rouge_l(reference: str, hypothesis: str) -> float:
    """Sentence-level ROUGE-L F1."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, hypothesis)["rougeL"].fmeasure
    except Exception:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if ref_tokens[i-1] == hyp_tokens[j-1] else max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        p = lcs / n if n else 0.0
        r = lcs / m if m else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def cmd_evalrun(args: argparse.Namespace) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import openai

    out_dir     = Path(args.out_dir)
    results_dir = out_dir / "results"
    results_f   = results_dir / "contextual.jsonl"
    eval_f      = results_dir / f"eval_{args.judge.replace('-','_').replace('.','_')}.jsonl"
    ckpt_f      = results_dir / f"eval_ckpt_{args.judge.replace('-','_').replace('.','_')}.jsonl"

    if not results_f.exists():
        print("No results found. Run 'run' first.")
        return

    results = [json.loads(line) for line in open(results_f)]
    if args.limit:
        results = results[:args.limit]
    print(f"Evaluating {len(results)} results with judge={args.judge}")

    done_ids: set[str] = set()
    if ckpt_f.exists():
        with open(ckpt_f) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} already evaluated")

    pending = [r for r in results if r["id"] not in done_ids]
    print(f"Pending: {len(pending)} — using {args.concurrency} parallel workers")

    client    = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)
    # Flat pool: submit acc + cov as separate tasks, collect by result id
    # avoids nested ThreadPoolExecutor thread exhaustion
    partials: dict[str, dict] = {}  # id → partial eval record
    ckpt_lock = threading.Lock()
    new_evals: list[dict] = []
    done_count = [len(done_ids)]

    def _judge_task(r: dict, kind: str) -> tuple[str, str, float]:
        """Returns (result_id, kind, score)."""
        if kind == "acc":
            prompt = _JUDGE_ACCURACY_PROMPT.format(
                question=r["question"],
                gold=r["gold_answer"][:800],
                generated=r["generated_answer"][:800],
            )
        else:
            prompt = _JUDGE_COVERAGE_PROMPT.format(
                question=r["question"],
                gold=r["gold_answer"][:800],
                context=r["context"][:1500],
            )
        return (r["id"], kind, _llm_judge(prompt, client, args.judge))

    # Pre-compute rouge (CPU-only, fast) and index pending by id
    pending_by_id = {r["id"]: r for r in pending}
    rouge_map = {r["id"]: _rouge_l(r["gold_answer"], r["generated_answer"]) for r in pending}

    # Submit all acc + cov tasks flat — 2 tasks per result
    # Interleave acc+cov per result so pairs complete quickly, not all-acc then all-cov
    tasks = [(r, k) for r in pending for k in ("acc", "cov")]

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_judge_task, r, kind): (r, kind) for r, kind in tasks}
        for fut in as_completed(futures):
            rid, kind, score = fut.result()
            with ckpt_lock:
                if rid not in partials:
                    partials[rid] = {}
                partials[rid][kind] = score
                if "acc" in partials[rid] and "cov" in partials[rid]:
                    src = pending_by_id[rid]
                    ev = {
                        "id":            rid,
                        "source":        src["source"],
                        "question_type": src["question_type"],
                        "accuracy":      partials[rid]["acc"],
                        "coverage":      partials[rid]["cov"],
                        "rouge_l":       rouge_map[rid],
                    }
                    new_evals.append(ev)
                    done_count[0] += 1
                    total = done_count[0]
                    if total % 50 == 0:
                        print(f"  {total}/{len(results)}", flush=True)
                    if len(new_evals) % 100 == 0:
                        with open(ckpt_f, "a") as f:
                            for e in new_evals[-100:]:
                                f.write(json.dumps(e) + "\n")
                        print(f"  {total}/{len(results)}  (checkpoint saved)", flush=True)

    # Final flush remainder
    with open(ckpt_f, "a") as f:
        remainder = len(new_evals) % 100
        if remainder:
            for e in new_evals[-remainder:]:
                f.write(json.dumps(e) + "\n")

    all_evals: list[dict] = []
    if ckpt_f.exists():
        with open(ckpt_f) as f:
            for line in f:
                all_evals.append(json.loads(line))
    ckpt_ids = {e["id"] for e in all_evals}
    for e in new_evals:
        if e["id"] not in ckpt_ids:
            all_evals.append(e)

    with open(eval_f, "w") as f:
        for e in all_evals:
            f.write(json.dumps(e) + "\n")

    print(f"\nEvaluation complete: {len(all_evals)} records → {eval_f}")
    _print_eval_summary(all_evals)


def _print_eval_summary(evals: list[dict]) -> None:
    by_source: dict[str, list] = defaultdict(list)
    by_type:   dict[str, list] = defaultdict(list)
    for e in evals:
        by_source[e["source"]].append(e)
        by_type[e["question_type"]].append(e)

    def _avg(records, key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n── Results by subset ──")
    print(f"  {'Subset':<16}  {'N':>5}  {'Accuracy':>9}  {'Coverage':>9}  {'ROUGE-L':>8}")
    print("  " + "─" * 54)
    all_combined = []
    for src, recs in sorted(by_source.items()):
        all_combined.extend(recs)
        print(f"  {src:<16}  {len(recs):>5}  {_avg(recs,'accuracy'):>9.3f}"
              f"  {_avg(recs,'coverage'):>9.3f}  {_avg(recs,'rouge_l'):>8.3f}")
    print(f"  {'OVERALL':<16}  {len(all_combined):>5}  {_avg(all_combined,'accuracy'):>9.3f}"
          f"  {_avg(all_combined,'coverage'):>9.3f}  {_avg(all_combined,'rouge_l'):>8.3f}")

    print("\n── Results by question type ──")
    print(f"  {'Type':<28}  {'N':>5}  {'Accuracy':>9}  {'Coverage':>9}  {'ROUGE-L':>8}")
    print("  " + "─" * 66)
    for qt, recs in sorted(by_type.items()):
        print(f"  {qt:<28}  {len(recs):>5}  {_avg(recs,'accuracy'):>9.3f}"
              f"  {_avg(recs,'coverage'):>9.3f}  {_avg(recs,'rouge_l'):>8.3f}")

    no_creative = [e for e in evals if e["question_type"] != "Creative Generation"]
    if len(no_creative) < len(evals):
        print(f"\n── Without Creative Generation (n={len(no_creative)}) ──")
        print(f"  Accuracy: {_avg(no_creative,'accuracy'):.3f}"
              f"  Coverage: {_avg(no_creative,'coverage'):.3f}"
              f"  ROUGE-L:  {_avg(no_creative,'rouge_l'):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Report
# ─────────────────────────────────────────────────────────────────────────────

def cmd_report(args: argparse.Namespace) -> None:
    out_dir     = Path(args.out_dir)
    results_dir = out_dir / "results"

    eval_files = sorted(results_dir.glob("eval_*.jsonl"))
    if not eval_files:
        print("No evaluation files found. Run 'evalrun' first.")
        return

    print("=" * 72)
    print("Chunky Monkey — GraphRAG-Bench Results")
    print("=" * 72)

    for eval_f in eval_files:
        judge = eval_f.stem.replace("eval_", "").replace("_", "-")
        evals = [json.loads(line) for line in open(eval_f)]
        print(f"\nJudge: {judge}  ({len(evals)} questions evaluated)")
        _print_eval_summary(evals)

    print("\n── Comparison vs Published Leaderboard ──")
    print("(Published scores from GraphRAG-Bench leaderboard; fill in when available)")
    print(f"\n  {'Strategy':<24}  {'Med Acc':>8}  {'Nov Acc':>8}  {'Overall':>8}")
    print("  " + "─" * 56)
    if eval_files:
        evals = [json.loads(line) for line in open(eval_files[-1])]
        med_evals = [e for e in evals if e["source"] == "Medical"]
        nov_evals = [e for e in evals if e["source"] != "Medical"]
        med_acc = sum(e["accuracy"] for e in med_evals) / max(1, len(med_evals))
        nov_acc = sum(e["accuracy"] for e in nov_evals) / max(1, len(nov_evals))
        overall = sum(e["accuracy"] for e in evals) / max(1, len(evals))
        print(f"  {'Contextual (ours)':<24}  {med_acc:>8.3f}  {nov_acc:>8.3f}  {overall:>8.3f}")
    for name, scores in PUBLISHED_BASELINES.items():
        med = f"{scores['med_acc']:.3f}" if scores["med_acc"] is not None else "   —"
        nov = f"{scores['nov_acc']:.3f}" if scores["nov_acc"] is not None else "   —"
        ov  = f"{scores['overall']:.3f}" if scores["overall"] is not None else "   —"
        print(f"  {name:<24}  {med:>8}  {nov:>8}  {ov:>8}")
    print("\n  * Paper uses Qwen2.5-14B generator + GPT-4 judge; ours uses GPT-4o-mini for both.")


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
    p.set_defaults(func=cmd_run)

    # evalrun (generic)
    p = sub.add_parser("evalrun", help="Evaluate generated answers with LLM judge")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--judge",       default="gpt-4o-mini",
                   help="Judge model (default: gpt-4o-mini)")
    p.add_argument("--limit",       type=int, default=None, metavar="N")
    p.add_argument("--concurrency", type=int, default=20,
                   help="Parallel judge workers (default: 20)")
    p.set_defaults(func=cmd_evalrun)

    # eval — shortcut: GPT-4o-mini judge
    p = sub.add_parser("eval", help="Evaluate with GPT-4o-mini judge")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--limit",       type=int, default=None, metavar="N")
    p.add_argument("--concurrency", type=int, default=20,
                   help="Parallel judge workers (default: 20)")
    p.set_defaults(func=cmd_evalrun, judge="gpt-4o-mini")

    # final — shortcut: GPT-4.1 judge
    p = sub.add_parser("final", help="Final evaluation with GPT-4.1 judge")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--limit",       type=int, default=None, metavar="N")
    p.add_argument("--concurrency", type=int, default=20,
                   help="Parallel judge workers (default: 20)")
    p.set_defaults(func=cmd_evalrun, judge="gpt-4.1")

    # report
    p = sub.add_parser("report", help="Print comparison table vs published leaderboard")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.set_defaults(func=cmd_report)

    return ap


def main() -> None:
    ap   = _make_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
