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

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

# Load .env from project root before anything else
_PROJECT_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(_PROJECT_ROOT))
from chonk import NOVEL_STRUCTURAL_LEVELS, chunk_document, promote_plain_text_headers
from chonk.context import enrich_chunks
from chonk.storage._store import GLOBAL_NAMESPACE, Store

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EMBED_MODEL        = "BAAI/bge-large-en-v1.5"
EMBED_DIM          = 1024
GEN_MODEL           = "gpt-4o-mini-2024-07-18"
GEN_MODEL_TOGETHER  = "Qwen/Qwen2.5-72B-Instruct-Turbo"   # closest serverless Qwen2.5 (14B not available serverless on Together)
GEN_MODEL_ANTHROPIC = "claude-sonnet-4-6"
TOGETHER_BASE_URL   = "https://api.together.xyz/v1"
ANTHROPIC_BASE_URL  = "https://api.anthropic.com/v1"
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
DB_FILENAME         = "chonk.duckdb"
VANILLA_DB_FILENAME = "vanilla_rag.duckdb"
VANILLA_K             = 5     # paper Appendix H.2: retrieval_topk=5
VANILLA_CHUNK_TOKENS  = 256   # benchmark uses 256-token chunks
VANILLA_CHUNK_OVERLAP = 0
VANILLA_TEMPERATURE   = 0.7   # paper: "generation temperature of 0.7"

def _model_rpm_limit(model: str) -> int:
    """Return 80%-of-tier-4 RPM limit for a given OpenAI model, loaded from openai_rpm_limits.json."""
    _limits_file = Path(__file__).parent / "openai_rpm_limits.json"
    try:
        _data = json.loads(_limits_file.read_text())
        return _data.get(model, _data.get("_default", 2400))
    except Exception:
        return 2400

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
# TOML config loading
# ─────────────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_config(config_path: str | None, _depth: int = 0) -> dict:
    if config_path is None:
        return {}
    if _depth > 5:
        raise RuntimeError(f"TOML extends chain exceeds max depth 5 at {config_path}")
    path = Path(config_path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    parent: dict = {}
    if "extends" in data:
        parent_path = (path.parent / data.pop("extends")).resolve()
        parent = _load_config(str(parent_path), _depth + 1)
    return _deep_merge(parent, data)


def _apply_config(cfg: dict, args: argparse.Namespace) -> None:
    if "run_name" in cfg and not getattr(args, "run_name", None):
        args.run_name = cfg["run_name"]

    idx = cfg.get("index", {})
    if idx.get("db_name") and not getattr(args, "db_name", None):
        args.db_name = idx["db_name"]

    features = idx.get("features", {})
    if features.get("svo") and not getattr(args, "with_svo", False):
        args.with_svo = True
    if features.get("community") and not getattr(args, "with_community", False):
        args.with_community = True
    if features.get("ner") and not getattr(args, "with_ner", False):
        args.with_ner = True

    rnk = cfg.get("rerank", {})
    if rnk.get("enabled") and not getattr(args, "rerank", False):
        args.rerank = True

    ret = cfg.get("retrieval", {})
    if ret.get("top_k") is not None and getattr(args, "top_k", None) is None:
        args.top_k = ret["top_k"]
    if ret.get("search_mode") and getattr(args, "search_mode", "vector_first") == "vector_first":
        args.search_mode = ret["search_mode"]
    if ret.get("enhanced") and not getattr(args, "enhanced", False):
        args.enhanced = True
    if ret.get("entity_ref_expansion") and not getattr(args, "entity_ref_expansion", False):
        args.entity_ref_expansion = True
    if ret.get("lane_entity_min_sim") is not None and getattr(args, "lane_entity_min_sim", None) is None:
        args.lane_entity_min_sim = ret["lane_entity_min_sim"]
    if ret.get("redundancy_threshold") is not None and getattr(args, "redundancy_threshold", None) is None:
        args.redundancy_threshold = ret["redundancy_threshold"]
    if ret.get("cluster") and not getattr(args, "cluster", False):
        args.cluster = True
    if ret.get("context_graph") and not getattr(args, "context_graph", False):
        args.context_graph = True
    if ret.get("context_graph_min_weight") is not None and getattr(args, "context_graph_min_weight", None) in (None, 0.1):
        args.context_graph_min_weight = ret["context_graph_min_weight"]
    if ret.get("context_graph_top_k") is not None and getattr(args, "context_graph_top_k", None) in (None, 5):
        args.context_graph_top_k = ret["context_graph_top_k"]
    if ret.get("vanilla") and not getattr(args, "vanilla", False):
        args.vanilla = True
    if ret.get("multi_step") and not getattr(args, "multi_step", False):
        args.multi_step = True
    if ret.get("bm25") and not getattr(args, "bm25", False):
        args.bm25 = True
    if ret.get("rerank_device") and not getattr(args, "rerank_device", None):
        args.rerank_device = ret["rerank_device"]

    comm = ret.get("community", {})
    if comm.get("enabled") and not getattr(args, "community_context", False):
        args.community_context = True
    if comm.get("min_coherence") is not None and getattr(args, "community_min_coherence", None) is None:
        args.community_min_coherence = comm["min_coherence"]

    g = cfg.get("gen", {})
    if g.get("model") and getattr(args, "gen_model", None) in (None, "gpt-4o-mini-2024-07-18"):
        args.gen_model = g["model"]
    if g.get("provider") and getattr(args, "gen_provider", "openai") == "openai":
        args.gen_provider = g["provider"]

    sr_cfg = cfg.get("sr", {})
    if sr_cfg.get("enabled") and not getattr(args, "sr", False):
        args.sr = True

    srr_cfg = cfg.get("srr", {})
    if srr_cfg.get("enabled") and not getattr(args, "srr", False):
        args.srr = True
    if srr_cfg.get("model") and not getattr(args, "srr_model", None):
        args.srr_model = srr_cfg["model"]
    if srr_cfg.get("provider") and not getattr(args, "srr_provider", None):
        args.srr_provider = srr_cfg["provider"]

    vocab_entries = cfg.get("vocab", {}).get("entities", [])
    if vocab_entries and not getattr(args, "vocab_entities", None):
        # namespace is optional on every vocab entry type; unset means global —
        # same rule [[source]] uses below.
        args.vocab_entities = [
            dict(entry, namespace=entry.get("namespace", GLOBAL_NAMESPACE))
            for entry in vocab_entries
        ]

    sources = cfg.get("source", [])
    if sources and not getattr(args, "sources", None):
        args.sources = [
            dict(entry, namespace=entry.get("namespace", GLOBAL_NAMESPACE))
            for entry in sources
        ]

    if ret.get("namespaces") is not None and getattr(args, "namespaces", None) is None:
        args.namespaces = ret["namespaces"]

    if ret.get("domain_ids") is not None and getattr(args, "domain_ids", None) is None:
        args.domain_ids = ret["domain_ids"]

    if ret.get("auto_domain_filter") and not getattr(args, "auto_domain_filter", False):
        args.auto_domain_filter = True


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
    seen_files: set[Path] = set()
    # Load canonical subsets first (stable order)
    for subset in SUBSETS:
        f = data_dir / f"{subset}_questions.jsonl"
        if f.exists():
            seen_files.add(f)
            with open(f) as fh:
                for line in fh:
                    questions.append(json.loads(line))
    # Pick up any additional *_questions.jsonl files in the same dir
    for f in sorted(data_dir.glob("*_questions.jsonl")):
        if f.name.startswith("."):
            continue
        if f not in seen_files:
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
    print("\n  Evidence reconstruction fallback:")
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
    (out_dir / "corpus_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nSaved corpus_info.json")


# ─────────────────────────────────────────────────────────────────────────────
# Corpus builder (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _table_exists(con, table_name: str) -> bool:
    """Return True if *table_name* exists in the connected DuckDB."""
    try:
        return con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0] > 0
    except Exception:
        return False


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
# Phase 2: Index (chunk + embed + store via chonk)
# ─────────────────────────────────────────────────────────────────────────────


def cmd_build_ner(args: argparse.Namespace) -> None:
    """Run NER on an existing index and persist chunk_entities to DB."""
    data_dir = Path(args.out_dir) / "data"
    db_path  = data_dir / args.db_name
    force    = getattr(args, "force", False)
    if not db_path.exists():
        raise FileNotFoundError(f"Index DB not found: {db_path}")

    use_schema_vocab = getattr(args, "with_schema_vocab", False)
    vocab_entities   = getattr(args, "vocab_entities", None)

    with_context_graph = getattr(args, "with_context_graph", False)
    from chonk.ner import build_ner
    with Store(db_path, embedding_dim=EMBED_DIM) as store:
        n_written = build_ner(
            store,
            spacy_model=SPACY_MODEL,
            use_schema_vocab=use_schema_vocab,
            vocab_entities=vocab_entities,
            force=force,
            build_context_graph=with_context_graph,
        )
    print(f"  Persisted {n_written:,} associations → {db_path}")
    if with_context_graph:
        print("  Context graph built.")

    # Build entity embeddings if requested or not yet present
    with_embeddings = getattr(args, "with_embeddings", False)
    if with_embeddings:
        import duckdb as _ddb
        _con = _ddb.connect(str(db_path), read_only=True)
        try:
            ne = _con.execute("SELECT COUNT(*) FROM entity_embeddings").fetchone()[0]
        except Exception:
            ne = 0
        _con.close()
        if ne > 0 and not force:
            print(f"entity_embeddings already populated ({ne:,} rows) — skipping.")
        else:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer(EMBED_MODEL)
            entity_index = _load_entity_index_from_db(db_path)
            _build_and_persist_entity_embeddings(entity_index, embed_model, db_path)


def cmd_index(args: argparse.Namespace) -> None:
    import hashlib as _hl

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from chonk.storage._vector import prune_documents, sync_document

    out_dir  = Path(args.out_dir)
    data_dir = out_dir / "data"
    embed_content_only = getattr(args, "embed_content_only", False)
    include_doc_name   = getattr(args, "include_doc_name", False)
    min_chunk = getattr(args, 'min_chunk', MIN_CHUNK) or MIN_CHUNK
    max_chunk = getattr(args, 'max_chunk', MAX_CHUNK) or MAX_CHUNK
    size_suffix = f"_{min_chunk}_{max_chunk}" if (min_chunk != MIN_CHUNK or max_chunk != MAX_CHUNK) else ""
    if embed_content_only:
        base = "chonk_nobc"
    elif include_doc_name:
        base = "chunkymonkey_bc"
    else:
        base = "chonk"
    db_path = data_dir / f"{base}{size_suffix}.duckdb"

    if db_path.exists() and args.force:
        db_path.unlink()
        print(f"Removed existing index: {db_path}")

    corpus = _build_corpus(out_dir)
    print(f"Corpus: {len(corpus)} documents")

    sources_cfg = getattr(args, "sources", None)
    # Build per-doc source maps upfront
    source_ns_map: dict[str, str | None] = {}
    source_sid_map: dict[str, str | None] = {}
    source_did_map: dict[str, str | None] = {}

    with Store(db_path, embedding_dim=EMBED_DIM) as store:
        if sources_cfg:
            store.register_namespace(GLOBAL_NAMESPACE)
            corpus_doc_names = {doc_id for doc_id, _ in corpus}
            for src in sources_cfg:
                ns = src.get("namespace", GLOBAL_NAMESPACE)
                domain = src.get("domain")
                src_type = src.get("type", "")
                uri = src.get("uri", "")
                namespace_id = ns if ns else None
                domain_id = f"{ns}:{domain}" if (ns and domain) else None
                source_id = f"{ns}:{domain}:{src_type}:{uri[:40]}" if (ns and domain) else None
                if namespace_id:
                    store.register_namespace(namespace_id)
                if domain_id:
                    store.register_domain(domain_id, namespace_id, domain)
                if source_id:
                    src_config = {k: v for k, v in src.items() if k not in ("namespace", "domain", "type", "uri")}
                    store.register_source(source_id, domain_id, src_type, uri, src_config or None)
                for doc_id in corpus_doc_names:
                    if uri and doc_id.startswith(uri):
                        source_ns_map[doc_id] = ns
                        source_sid_map[doc_id] = source_id
                        source_did_map[doc_id] = domain_id

        model = SentenceTransformer(EMBED_MODEL)
        n_skipped = n_added = n_updated = 0
        present_doc_names = {doc_id for doc_id, _ in corpus}

        print(f"Chunking and indexing with header promotion (min={min_chunk}, max={max_chunk})...")
        all_chunks_for_schema: list = []

        for doc_id, text in corpus:
            content_hash = _hl.sha256(text.encode()).hexdigest()
            result = sync_document(store.vector, doc_id, content_hash=content_hash)
            if result.action == "skipped":
                n_skipped += 1
                continue

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
                    strip_isolated_letters=True,
                )
            chunks = chunk_document(
                doc_id, promoted,
                min_chunk_size=min_chunk,
                max_chunk_size=max_chunk,
                include_breadcrumb=True,
                include_doc_name=include_doc_name,
                promote_headings=False,
            )
            chunks = enrich_chunks(chunks)

            texts = [
                c.content if embed_content_only else (c.embedding_content if c.embedding_content else c.content)
                for c in chunks
            ]
            batch_size = 256
            doc_vecs = []
            for i in range(0, len(texts), batch_size):
                vecs = model.encode(texts[i:i + batch_size], show_progress_bar=False, normalize_embeddings=True)
                doc_vecs.append(vecs)
            emb = np.vstack(doc_vecs).astype("float32") if doc_vecs else np.empty((0, EMBED_DIM), dtype="float32")

            ns  = source_ns_map.get(doc_id)
            sid = source_sid_map.get(doc_id)
            did = source_did_map.get(doc_id)
            store.add_document(chunks, emb, namespace=ns, source_id=sid, domain_id=did)
            store.vector.register_document(doc_id, content_hash, chunk_count=len(chunks))

            all_chunks_for_schema.extend(chunks)
            if result.action == "updated":
                n_updated += 1
            else:
                n_added += 1

        # Documents that vanished from the corpus since the last run.
        removed = prune_documents(store.vector, present_doc_names)
        n_deleted = len(removed)

        n = store.count()

    print(f"Index complete: {n:,} chunks → {db_path}")
    if n_skipped or n_updated or n_deleted:
        print(
            f"  skipped={n_skipped:,} (unchanged)  added={n_added:,}  "
            f"updated={n_updated:,}  deleted={n_deleted:,}"
        )

    # Populate entity descriptions from DB schema chunks (source='schema').
    # Table/column comments extracted at crawl time — free, no LLM needed.
    _SCHEMA_CHUNK_TYPES = {"db_table", "db_column", "api_endpoint",
                           "api_graphql_type", "api_field"}
    schema_chunks = [c for c in all_chunks_for_schema if c.chunk_type in _SCHEMA_CHUNK_TYPES]
    if schema_chunks:
        with Store(db_path, embedding_dim=EMBED_DIM) as _sd_store:
            schema_descs: dict[str, str] = {}
            for sc in schema_chunks:
                if sc.content and sc.document_name:
                    # Use the chunk's document_name as the entity_id (normalized)
                    eid = sc.document_name.split(":")[-1].strip()
                    if eid and sc.content.strip():
                        schema_descs[eid] = sc.content.strip()[:300]
            if schema_descs:
                n_schema = _sd_store.set_entity_descriptions_batch(schema_descs)
                print(f"Populated {n_schema:,} schema entity descriptions → {db_path}")

    if getattr(args, "with_ner", False):
        with Store(db_path, embedding_dim=EMBED_DIM) as store:
            entity_index = _build_entity_index_from_store(store)
        _persist_entity_index(entity_index, db_path)

    if getattr(args, "with_community", False):
        args.db_name = db_path.name
        cmd_build_community(args)

    if getattr(args, "with_svo", False):
        args.db_name = db_path.name
        cmd_build_svo(args)


def _corpus_from_store(store_db: str) -> list[tuple[str, str]]:
    """Read document texts from an existing Store DB, grouped by document_name."""
    import duckdb
    con = duckdb.connect(store_db, read_only=True)
    rows = con.execute(
        "SELECT document_name, content FROM embeddings ORDER BY document_name, chunk_index"
    ).fetchall()
    con.close()
    docs: dict[str, list[str]] = {}
    for doc_name, content in rows:
        docs.setdefault(doc_name, []).append(content)
    return [(name, " ".join(chunks)) for name, chunks in docs.items()]


def cmd_index_vanilla(args: argparse.Namespace) -> None:
    """Build vanilla RAG index: naive 256-token fixed chunks, no breadcrumbs."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from chonk.models import DocumentChunk

    out_dir  = Path(args.out_dir)
    data_dir = out_dir / "data"
    chunk_tokens = getattr(args, 'chunk_tokens', VANILLA_CHUNK_TOKENS) or VANILLA_CHUNK_TOKENS
    db_suffix = f"_{chunk_tokens}" if chunk_tokens != VANILLA_CHUNK_TOKENS else ""
    db_path = data_dir / f"vanilla_rag{db_suffix}.duckdb"

    if db_path.exists() and not args.force:
        with Store(db_path, embedding_dim=EMBED_DIM) as store:
            n = store.count()
        print(f"Vanilla index exists: {n:,} chunks at {db_path}")
        print("Use --force to reindex.")
        return

    if db_path.exists() and args.force:
        db_path.unlink()
        print(f"Removed existing index: {db_path}")

    from_store = getattr(args, "from_store", None)
    if from_store:
        corpus = _corpus_from_store(from_store)
    else:
        corpus = _build_corpus(out_dir)
    print(f"Corpus: {len(corpus)} documents")
    print(f"Naive chunking: {chunk_tokens}-token chunks, {VANILLA_CHUNK_OVERLAP}-token overlap...")

    all_chunks: list[DocumentChunk] = []
    for doc_id, text in corpus:
        for i, chunk_text in enumerate(_naive_chunks(text, chunk_tokens=chunk_tokens)):
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

_REFUSAL_PHRASES = (
    "does not contain", "does not provide", "does not mention",
    "cannot determine", "cannot answer", "not enough information",
    "insufficient information", "not mentioned", "no information",
    "context does not", "not specified", "not stated",
)

def _format_breadcrumb(crumb: str, style: str = "markdown") -> str:
    """Convert [doc > sec > subsec] to various heading formats.

    style='markdown': ## doc\\n### sec\\n#### subsec  (default)
    style='literal':  Document: doc. Section: sec. Subsection: subsec.
    style='symbol':   original [doc > sec > subsec] unchanged
    """
    if style == "symbol":
        return crumb
    inner = crumb.strip("[]")
    parts = [p.strip() for p in inner.split(">") if p.strip()]
    if not parts:
        return crumb
    if style == "literal":
        labels = ["Document", "Section", "Subsection", "Topic"]
        return ". ".join(
            f"{labels[min(i, len(labels)-1)]}: {p}" for i, p in enumerate(parts)
        ) + "."
    # markdown (default)
    levels = ["##", "###", "####", "#####"]
    return "\n".join(f"{levels[min(i, len(levels)-1)]} {p}" for i, p in enumerate(parts))


def _is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in _REFUSAL_PHRASES)


_STRUCTURED_GEN_SYSTEM = (
    "Answer the question based only on the provided context. "
    "If the context does not contain enough information, say so. "
    "You MUST format your response exactly as:\nANSWER: <your answer here>\n"
    "Do not include any text before or after this line."
)
_STRUCTURED_GEN_RETRY_HINT = (
    "Your response did not follow the required format. "
    "You MUST respond with exactly one line starting with 'ANSWER: ' followed by your answer. "
    "Example: ANSWER: The capital of France is Paris."
)

# ── SRR: Structured Response Retry ────────────────────────────────────────────
_SRR_GEN_SYSTEM = (
    "Answer the question using only information from the provided context.\n"
    "You MUST respond with valid JSON in exactly this format — no other text:\n"
    '{\n'
    '  "answer": "<prose answer suitable for the reader>",\n'
    '  "key_claims": ["<discrete claim 1>", "<discrete claim 2>"],\n'
    '  "evidence_used": ["<verbatim quote or close paraphrase from context>"]\n'
    "}\n"
    "Do not fabricate evidence for a claim."
)
_SRR_RETRY_HINT = (
    "Your previous response was not valid JSON. "
    'Respond ONLY with a JSON object containing keys "answer", "key_claims", and "evidence_used". '
    'Example: {"answer": "...", "key_claims": ["..."], "evidence_used": ["..."]}'
)
_SRR_EVIDENCE_HINT = (
    "Your response included no evidence_used. Every key claim must be supported by at least one "
    "verbatim quote or close paraphrase from the context. Your claims were:\n{claims}\n"
    "Respond again with the same JSON format, adding evidence_used entries that support each claim."
)
_SRR_COVERAGE_THRESHOLD = 0.35  # min cosine sim for entity→evidence coverage
_UNSTRUCTURED_GEN_SYSTEM = (
    "Answer the question based only on the provided context. "
    "If the context does not contain enough information, "
    "say so rather than making up an answer."
)
# Exact prompt template used by GraphRAG-Bench vanilla RAG baseline (Appendix H.2)
_VANILLA_GEN_PROMPT = (
    "You are a helpful assistant.\n"
    "Based on the following context, answer the question.\n"
    "Context:\n{context}\n"
    "Question: {question}\n"
    "Answer:"
)

def _extract_structured_answer(text: str) -> str | None:
    """Extract content after 'ANSWER:' marker. Returns None if marker absent."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("ANSWER:"):
            return stripped[len("ANSWER:"):].strip()
    return None

_DECOMPOSE_PROMPT = """\
You are a retrieval query planner. Break the following question into exactly 3 focused sub-queries \
that, together, would retrieve all evidence needed to answer it. Each sub-query should target a \
distinct aspect. Return ONLY a JSON array of 3 strings.

Question type: {question_type}
Question: {question}

Sub-queries (JSON array of 3 strings):"""


def _decompose_question(
    question: str, question_type: str, client, model: str
) -> list[str]:
    """Return 3 targeted sub-queries for multi-step retrieval. Falls back to [question] on failure."""
    import json as _json
    import re as _re

    prompt = _DECOMPOSE_PROMPT.format(question=question, question_type=question_type or "Unknown")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        text = resp.choices[0].message.content.strip()
        # Try strict parse, then regex extraction
        try:
            subs = _json.loads(text)
        except _json.JSONDecodeError:
            m = _re.search(r'\[.*?\]', text, _re.DOTALL)
            subs = _json.loads(m.group(0)) if m else []
        if isinstance(subs, list) and len(subs) >= 2:
            return [str(s).strip() for s in subs if str(s).strip()][:4]
    except Exception as e:
        print(f"[decompose] error: {type(e).__name__}: {str(e)[:120]}", flush=True)
    return [question]


_SRR_CLAUDE_SUFFIX = (
    "\nBe maximally direct and concise. "
    "The 'answer' field must be 1-2 declarative sentences with no hedging, "
    "no attribution preamble ('Based on...', 'According to...'), and no qualifiers. "
    "For absence or negation, state the fact plainly: e.g. 'Company X is not mentioned in the filings.'"
)


def _generate_srr(question: str, context: str, client, model: str,
                  temperature: float = 0.0) -> dict:
    """Generate a structured response for SRR. Returns dict with answer/key_claims/evidence_used."""
    import json as _json
    system = _SRR_GEN_SYSTEM + (_SRR_CLAUDE_SUFFIX if "claude" in model.lower() else "")
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    raw = ""
    for attempt in range(2):
        prompt = user_content if attempt == 0 else user_content + f"\n\n{_SRR_RETRY_HINT}"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=temperature,
            max_tokens=700,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:])
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        try:
            obj = _json.loads(raw)
            if isinstance(obj.get("answer"), str):
                return {
                    "answer":       obj["answer"],
                    "key_claims":   [x for x in obj.get("key_claims", []) if isinstance(x, str)],
                    "evidence_used": [x for x in obj.get("evidence_used", []) if isinstance(x, str)],
                }
        except Exception:
            pass
    # Both attempts failed — return raw text as answer with empty structure
    return {"answer": raw, "key_claims": [], "evidence_used": []}


def _srr_gap_fill(
    question: str,
    uncovered_entities: list[str],
    existing_context: str,
    chunk_pool: list[tuple],  # [(chunk_id, text, embedding_vec)]
    embed_model,
    top_k: int = 3,
) -> str:
    """Retrieve chunks for uncovered entities and return augmented context."""
    import numpy as _np
    if not uncovered_entities or not chunk_pool:
        return existing_context
    ent_vecs = embed_model.encode(uncovered_entities, normalize_embeddings=True,
                                  show_progress_bar=False)
    pool_texts = [t for _, t, _ in chunk_pool]
    pool_vecs  = _np.array([v for _, _, v in chunk_pool], dtype=_np.float32)
    # Score each chunk against all uncovered entities (max similarity)
    sims = ent_vecs @ pool_vecs.T          # (n_ents, n_chunks)
    chunk_scores = sims.max(axis=0)        # (n_chunks,)
    top_idx = _np.argsort(chunk_scores)[::-1][:top_k]
    new_chunks = [pool_texts[i] for i in top_idx
                  if pool_texts[i] not in existing_context]
    if not new_chunks:
        return existing_context
    addition = "\n\n---\n".join(new_chunks)
    return existing_context + f"\n\n[Additional context]\n{addition}"


def _generate(question: str, context: str, client, model: str = GEN_MODEL,
              temperature: float = 0.0, retry_hint: str | None = None,
              structured: bool = False, vanilla: bool = False) -> str:
    if vanilla:
        user_content = _VANILLA_GEN_PROMPT.format(context=context, question=question)
        if retry_hint:
            user_content += f"\n\n{retry_hint}"
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            temperature=temperature,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    system = _STRUCTURED_GEN_SYSTEM if structured else _UNSTRUCTURED_GEN_SYSTEM
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    if retry_hint:
        user_content += f"\n\n{retry_hint}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=500,
    )
    raw = resp.choices[0].message.content.strip()
    if not structured:
        return raw
    answer = _extract_structured_answer(raw)
    if answer is not None:
        return answer
    # Format non-compliant — retry once with explicit format hint
    retry_content = user_content + f"\n\n{_STRUCTURED_GEN_RETRY_HINT}"
    resp2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": retry_content},
        ],
        temperature=temperature,
        max_tokens=500,
    )
    raw2 = resp2.choices[0].message.content.strip()
    answer2 = _extract_structured_answer(raw2)
    return answer2 if answer2 is not None else raw2


def _build_entity_index_from_store(
    store,
    use_schema_vocab: bool = False,
    vocab_entities: list[dict] | None = None,
    skip_chunk_ids: set[str] | None = None,
) -> EntityIndex:
    """Run NER on all chunks in store and return a populated EntityIndex."""
    from chonk.ner import EntityIndex, SpacyLabel, SpacyMatcher
    from chonk.storage._vector import DuckDBVectorBackend
    _NUMERIC_TYPES = {SpacyLabel.CARDINAL, SpacyLabel.ORDINAL, SpacyLabel.MONEY,
                      SpacyLabel.PERCENT, SpacyLabel.QUANTITY}
    _label_types = [t for t in SpacyLabel if t not in _NUMERIC_TYPES]
    print(f"Building EntityIndex with SpacyMatcher({SPACY_MODEL})...")
    matcher = SpacyMatcher(model=SPACY_MODEL, strip_numeric=True, entity_types=_label_types)
    entity_index = EntityIndex()

    all_chunks = store.vector.get_all_chunks()

    schema_matcher = None
    if use_schema_vocab or vocab_entities:
        from chonk.ner import SchemaVocabBuilder
        builder = SchemaVocabBuilder()
        if use_schema_vocab:
            builder.add_chunks(all_chunks)
            print(f"  SchemaVocab: {builder.table_count():,} tables, {builder.column_count():,} columns, {builder.api_term_count():,} API terms")
        for entry in (vocab_entities or []):
            etype = entry.get("entity_type", "term")
            ns = entry.get("namespace")
            if entry.get("type") == "static":
                names = entry.get("names", [])
                builder.add_entities(names, entity_type=etype, namespace=ns)
                print(f"  VocabEntities static: {len(names):,} {etype!r} names [ns={ns or 'global'}]")
            elif entry.get("type") == "db_query":
                conn_url = entry["connection"]
                sql = entry["sql"]
                builder.add_from_db(conn_url, {etype: sql}, namespace=ns)
                print(f"  VocabEntities db_query: {etype!r} from {conn_url!r} [ns={ns or 'global'}]")
        schema_matcher = builder.build()
        data_matcher = builder.build_data_matcher() if builder.data_term_count() > 0 else None
    else:
        data_matcher = None

    chunks_to_process = [
        c for c in all_chunks
        if (lambda cid: cid not in (skip_chunk_ids or set()))(
            DuckDBVectorBackend._generate_chunk_id(
                c.document_name, c.chunk_index,
                c.embedding_content if c.embedding_content else c.content
            )
        )
    ]
    skipped = len(all_chunks) - len(chunks_to_process)
    if skipped:
        print(f"  Incremental: {skipped:,} chunks already processed, {len(chunks_to_process):,} new")
    print(f"  Running NER on {len(chunks_to_process):,} chunks...")
    for chunk in chunks_to_process:
        embed_content = chunk.embedding_content if chunk.embedding_content else chunk.content
        chunk_id = DuckDBVectorBackend._generate_chunk_id(
            chunk.document_name, chunk.chunk_index, embed_content
        )
        if schema_matcher is not None or data_matcher is not None:
            from chonk.ner import merge_matches
            vocab_hits: list = []
            if schema_matcher is not None:
                vocab_hits = merge_matches(
                    schema_matcher.match(chunk.content), vocab_hits,
                    source_text=chunk.content,
                )
            if data_matcher is not None:
                vocab_hits = merge_matches(
                    data_matcher.match(chunk.content), vocab_hits,
                    source_text=chunk.content,
                )
            spacy_hits = matcher.match(chunk.content)
            combined   = merge_matches(vocab_hits, spacy_hits, source_text=chunk.content)
            entity_index.index_chunk(chunk_id, chunk.content, combined)
        else:
            entity_index.run_ner(chunk_id, chunk.content, matcher)

    entity_index.recompute_scores()
    print(f"  {entity_index.total_chunks():,} chunks, {len(entity_index.entity_ids()):,} entities")
    return entity_index


def _persist_entity_index(entity_index, db_path: Path, *, incremental: bool = False) -> None:
    """Write entity_index associations to chunk_entities/entities tables."""

    print("  Persisting to chunk_entities table...")
    data = entity_index.to_dict()
    con = _connect_with_retry(db_path)
    if not incremental:
        con.execute("DELETE FROM chunk_entities")
        con.execute("DELETE FROM entities")
    for a in data["associations"]:
        ns_row = con.execute(
            "SELECT namespace FROM embeddings WHERE chunk_id = ?", [a["chunk_id"]]
        ).fetchone()
        namespace = ns_row[0] if ns_row else None
        con.execute(
            "INSERT OR REPLACE INTO chunk_entities(chunk_id, entity_id, frequency, positions_json, score, namespace) VALUES (?,?,?,?,?,?)",
            [a["chunk_id"], a["entity_id"], a["frequency"], json.dumps(a["positions"]), a["score"], namespace],
        )
        con.execute(
            "INSERT OR IGNORE INTO entities(id, name, display_name) VALUES (?,?,?)",
            [a["entity_id"], a["entity_id"], a["entity_id"]],
        )
    con.close()
    print(f"  Persisted {len(data['associations']):,} associations → {db_path}")


def _build_and_persist_entity_embeddings(entity_index, embed_model, db_path: Path) -> None:
    """Embed all unique entity name strings and store in entity_embeddings table."""

    entity_ids = list(entity_index.entity_ids())
    if not entity_ids:
        return
    print(f"  Embedding {len(entity_ids):,} unique entity strings...")
    vecs = embed_model.encode(
        entity_ids, normalize_embeddings=True, show_progress_bar=False, batch_size=512
    ).astype("float32")
    con = _connect_with_retry(db_path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS entity_embeddings "
        "(entity_id TEXT PRIMARY KEY, embedding FLOAT[])"
    )
    con.execute("DELETE FROM entity_embeddings")
    for eid, vec in zip(entity_ids, vecs):
        con.execute(
            "INSERT INTO entity_embeddings(entity_id, embedding) VALUES (?, ?)",
            [eid, vec.tolist()],
        )
    con.close()
    print(f"  Persisted {len(entity_ids):,} entity embeddings → {db_path}")


def cmd_build_community(args: argparse.Namespace) -> None:
    """Build community index: heading vectors + weighted average + Louvain detection."""
    import duckdb
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from chonk.community import CommunityIndex
    from chonk.storage._store import Store

    data_dir = Path(args.out_dir) / "data"
    db_path  = data_dir / args.db_name
    alpha    = getattr(args, "alpha", 0.2)
    sim_threshold = getattr(args, "sim_threshold", 0.6)
    force    = getattr(args, "force", False)
    label_strategy = getattr(args, "community_label_strategy", "ner_embedding")
    domain_ids = getattr(args, "domain_ids", None)

    if not db_path.exists():
        raise FileNotFoundError(f"Index DB not found: {db_path}")

    # Fingerprint-based cache check when domain_ids are scoped
    if domain_ids and not force:
        fingerprint = Store.session_fingerprint(domain_ids)
        with Store(db_path, embedding_dim=EMBED_DIM) as _cache_store:
            if _cache_store.community_cache_valid(fingerprint, domain_ids):
                print(f"Community cache hit for fingerprint {fingerprint}")
                return

    # Check if already built
    if not force:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            n = con.execute("SELECT COUNT(*) FROM chunk_communities").fetchone()[0]
            con.close()
            if n > 0:
                print(f"chunk_communities already populated ({n:,} rows) — skipping. Use --force to rebuild.")
                return
        except Exception:
            con.close()

    print(f"Loading chunks and embeddings from {db_path}...")
    con_ro = duckdb.connect(str(db_path), read_only=True)
    rows = con_ro.execute(
        "SELECT chunk_id, content, breadcrumb, embedding FROM embeddings WHERE embedding IS NOT NULL"
    ).fetchall()
    con_ro.close()

    chunk_ids: list[str] = [r[0] for r in rows]
    chunk_texts: list[str] = [r[1] or "" for r in rows]
    breadcrumbs: list[str] = [r[2] or "" for r in rows]
    content_vecs = np.array([r[3] for r in rows], dtype="float32")
    print(f"  {len(chunk_ids):,} chunks with embeddings")

    # Embed breadcrumbs (heading vectors)
    heading_vecs = None
    non_empty_bc = [bc for bc in breadcrumbs if bc.strip()]
    if non_empty_bc:
        print(f"  Embedding {len(non_empty_bc):,} non-empty breadcrumbs (α={alpha})...")
        model = SentenceTransformer(EMBED_MODEL)
        bc_texts = [bc if bc.strip() else "" for bc in breadcrumbs]
        all_bc_vecs = model.encode(
            bc_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=256
        ).astype("float32")
        # Zero out empty breadcrumbs so they don't contribute
        for i, bc in enumerate(breadcrumbs):
            if not bc.strip():
                all_bc_vecs[i] = 0.0
        heading_vecs = all_bc_vecs
        del model
    else:
        print("  No breadcrumbs found — using content vectors only.")

    # ── Entity bridge edges from chunk_entities (NER) ─────────────────────
    # For each entity appearing in chunks from multiple documents, inject a
    # cross-document edge so community detection bridges domain boundaries.
    extra_edges: list[tuple[int, int, float]] = []
    try:
        import duckdb as _duckdb
        _con_er = _duckdb.connect(str(db_path), read_only=True)
        _ce_rows = _con_er.execute(
            "SELECT entity_id, chunk_id FROM chunk_entities WHERE frequency > 0"
        ).fetchall()
        _con_er.close()
        if _ce_rows:
            from collections import defaultdict as _dd
            id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
            entity_to_chunks: dict[str, list[int]] = _dd(list)
            for eid, cid in _ce_rows:
                if cid in id_to_idx:
                    entity_to_chunks[eid].append(id_to_idx[cid])
            bridge_count = 0
            seen: set[tuple[int, int]] = set()
            for eid, idxs in entity_to_chunks.items():
                if len(idxs) < 2:
                    continue
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        ia, ib = idxs[a], idxs[b]
                        doc_a = chunk_ids[ia].rsplit("_", 2)[0]
                        doc_b = chunk_ids[ib].rsplit("_", 2)[0]
                        if doc_a != doc_b:
                            key = (min(ia, ib), max(ia, ib))
                            if key not in seen:
                                seen.add(key)
                                extra_edges.append((ia, ib, 1.0))
                                bridge_count += 1
            print(f"  Entity bridge: {bridge_count:,} cross-document edges from {len(entity_to_chunks)} entities (chunk_entities)")
    except Exception as _e:
        print(f"  Entity bridge skipped: {_e}")

    print(f"  Building community index (sim_threshold={sim_threshold}, label_strategy={label_strategy})...")
    idx = CommunityIndex.build(
        chunk_ids=chunk_ids,
        content_vecs=content_vecs,
        chunk_texts=chunk_texts,
        heading_vecs=heading_vecs,
        alpha=alpha,
        sim_threshold=sim_threshold,
        label_strategy=label_strategy,
        db_path=db_path if label_strategy == "ner_embedding" else None,
        extra_edges=extra_edges or None,
    )
    print(f"  {idx.community_count():,} communities across {idx.chunk_count():,} chunks")

    # Persist — retry with backoff if DB is locked by another process
    import time as _time
    for _attempt in range(60):
        try:
            idx.persist(db_path)
            print(f"  Persisted → {db_path}")
            break
        except Exception as _e:
            if "lock" in str(_e).lower() or "conflict" in str(_e).lower():
                print(f"  DB locked, waiting 10s... (attempt {_attempt+1}/60)", flush=True)
                _time.sleep(10)
            else:
                raise
    else:
        raise RuntimeError("Could not acquire write lock on DB after 10 minutes.")

    # Generate LLM community summaries and embed + store as community_summary chunks
    gen_provider = getattr(args, "gen_provider", None)
    gen_model    = getattr(args, "gen_model", GEN_MODEL)
    if gen_provider:
        import numpy as np
        import openai as _oai
        from sentence_transformers import SentenceTransformer

        from chonk.community import CommunitySummarizer

        if gen_provider == "together":
            _c_client = _oai.OpenAI(
                api_key=os.environ["TOGETHER_API_KEY"], base_url=TOGETHER_BASE_URL, timeout=120.0
            )
        elif gen_provider == "anthropic":
            _c_client = _oai.OpenAI(
                api_key=os.environ["ANTHROPIC_API_KEY"], base_url=ANTHROPIC_BASE_URL,
                default_headers={"anthropic-version": "2023-06-01"}, timeout=120.0,
            )
        else:
            _c_client = _oai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)

        class _CommunityLLM:
            def complete(self, prompt: str) -> str:
                resp = _c_client.chat.completions.create(
                    model=gen_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return resp.choices[0].message.content or ""

        chunk_text_map: dict[str, str] = {r[0]: r[1] for r in rows}
        summarizer = CommunitySummarizer(_CommunityLLM())
        print(f"  Generating community summaries with {gen_model}...")
        summary_chunks = summarizer.summarize_all(
            idx,
            get_chunk_text=lambda cid: chunk_text_map.get(cid),
            min_chunks=2,
        )
        print(f"  {len(summary_chunks):,} community summaries generated")

        if summary_chunks:
            embed_model = SentenceTransformer(EMBED_MODEL)
            texts = [c.content for c in summary_chunks]
            vecs = embed_model.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False, batch_size=256).astype("float32")
            with Store(db_path, embedding_dim=EMBED_DIM) as _sum_store:
                # Remove stale summaries before reinserting
                _sum_store.vector._conn.execute(
                    "DELETE FROM embeddings WHERE chunk_type = 'community_summary'"
                )
                _sum_store.add_document(summary_chunks, vecs,
                                        session_fingerprint=Store.session_fingerprint(
                                            domain_ids or ["__all__"]
                                        ))
            print(f"  Community summaries embedded and stored → {db_path}")

    # Write community cache entry when domain-scoped
    if domain_ids:
        fingerprint = Store.session_fingerprint(domain_ids)
        with Store(db_path, embedding_dim=EMBED_DIM) as _cache_store:
            _cache_store.write_community_cache(fingerprint, domain_ids)
        print(f"  Community cache written for fingerprint {fingerprint}")


def cmd_build_svo(args: argparse.Namespace) -> None:
    """Extract SVO triples from all chunks and persist to svo_triples table."""
    import concurrent.futures

    import duckdb

    from chonk.graph import RelationshipIndex, SVOExtractor

    data_dir = Path(args.out_dir) / "data"
    db_name  = getattr(args, "db_name", None) or DB_FILENAME
    db_path  = data_dir / db_name
    force       = getattr(args, "force", False)
    gen_model   = getattr(args, "gen_model", GEN_MODEL)
    gen_provider = getattr(args, "gen_provider", "openai")
    concurrency = getattr(args, "concurrency", 4)
    max_chunks  = getattr(args, "max_chunks", None)

    if not db_path.exists():
        raise FileNotFoundError(f"Index DB not found: {db_path}")

    _init_con = duckdb.connect(str(db_path))
    _init_con.execute("""
        CREATE TABLE IF NOT EXISTS svo_triples (
            chunk_id VARCHAR, subject_id VARCHAR NOT NULL, verb VARCHAR NOT NULL,
            object_id VARCHAR NOT NULL, confidence FLOAT NOT NULL DEFAULT 1.0,
            namespace VARCHAR, description TEXT NOT NULL DEFAULT ''
        )
    """)
    _init_con.execute("ALTER TABLE svo_triples ADD COLUMN IF NOT EXISTS namespace VARCHAR")
    _init_con.execute("ALTER TABLE svo_triples ADD COLUMN IF NOT EXISTS description TEXT")
    if force:
        _init_con.execute("DELETE FROM svo_triples")
        print("--force: cleared existing svo_triples", flush=True)
    else:
        n = _init_con.execute("SELECT COUNT(*) FROM svo_triples").fetchone()[0]
        if n > 0:
            print(f"svo_triples has {n:,} rows — will resume from checkpoint. Use --force to rebuild.")
    _init_con.close()

    import httpx as _httpx
    import openai as _oai

    _svo_timeout = _httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)

    if gen_provider == "together":
        _oai_client = _oai.OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"], base_url=TOGETHER_BASE_URL,
            timeout=_svo_timeout,
        )
    elif gen_provider == "anthropic":
        _oai_client = _oai.OpenAI(
            api_key=os.environ["ANTHROPIC_API_KEY"], base_url=ANTHROPIC_BASE_URL,
            default_headers={"anthropic-version": "2023-06-01"}, timeout=_svo_timeout,
        )
    else:
        _oai_client = _oai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)

    import threading as _threading
    import time as _time
    _svo_log_lock = _threading.Lock()

    def _svo_log(msg: str) -> None:
        ts = _time.strftime("%H:%M:%S")
        with _svo_log_lock:
            import sys as _sys
            _sys.stderr.write(f"[svo {ts}] {msg}\n")
            _sys.stderr.flush()

    class _OpenAILLMClient:
        def complete(self, prompt: str) -> str:
            prompt_chars = len(prompt)
            _svo_log(f"API call → model={gen_model} prompt={prompt_chars:,}chars")
            t0 = _time.monotonic()
            try:
                resp = _oai_client.chat.completions.create(
                    model=gen_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                elapsed = _time.monotonic() - t0
                content = resp.choices[0].message.content or ""
                usage = resp.usage
                _svo_log(
                    f"API ok  ← {elapsed:.2f}s  "
                    f"in={usage.prompt_tokens}tok out={usage.completion_tokens}tok  "
                    f"response={len(content):,}chars"
                )
                return content
            except Exception as exc:
                elapsed = _time.monotonic() - t0
                _svo_log(f"API err ← {elapsed:.2f}s  {type(exc).__name__}: {str(exc)[:120]}")
                raise

    llm = _OpenAILLMClient()
    extractor = SVOExtractor(llm)

    con_ro = duckdb.connect(str(db_path), read_only=True)
    rows = con_ro.execute("SELECT chunk_id, content FROM embeddings").fetchall()

    # Build chunk → entity list from chunk_entities join table.
    # Each entry: (entity_id, entity_type) — type from entities table where available.
    chunk_entity_rows = con_ro.execute("""
        SELECT ce.chunk_id, ce.entity_id,
               COALESCE(e.entity_type, 'concept') AS entity_type
        FROM chunk_entities ce
        LEFT JOIN entities e ON e.id = ce.entity_id
    """).fetchall()

    try:
        desc_rows = con_ro.execute(
            "SELECT id, COALESCE(description, '') FROM entities"
        ).fetchall()
        existing_descriptions: dict[str, str] = {r[0]: r[1] for r in desc_rows if r[1]}
    except Exception:
        existing_descriptions = {}
    con_ro.close()

    # Index: chunk_id → [{id, type, description}]
    from collections import defaultdict as _defaultdict
    chunk_entities_map: dict[str, list[dict]] = _defaultdict(list)
    for chunk_id, entity_id, entity_type in chunk_entity_rows:
        chunk_entities_map[chunk_id].append({
            "id": entity_id,
            "type": entity_type,
            "description": existing_descriptions.get(entity_id, ""),
        })

    # Load already-checkpointed chunk_ids for resume support
    _checkpointed_chunk_ids: set[str] = set()
    if not force:
        try:
            _ckpt_con = duckdb.connect(str(db_path), read_only=True)
            _ckpt_con.execute("CREATE TABLE IF NOT EXISTS svo_triples (chunk_id VARCHAR, subject_id VARCHAR NOT NULL, verb VARCHAR NOT NULL, object_id VARCHAR NOT NULL, confidence FLOAT NOT NULL DEFAULT 1.0, namespace VARCHAR, description TEXT NOT NULL DEFAULT '')")
            _checkpointed_chunk_ids = {
                r[0] for r in _ckpt_con.execute(
                    "SELECT DISTINCT chunk_id FROM svo_triples WHERE chunk_id IS NOT NULL"
                ).fetchall()
            }
            _ckpt_con.close()
        except Exception:
            pass

    chunks_with_entities = [(cid, content) for cid, content in rows
                            if len(chunk_entities_map.get(cid, [])) >= 2
                            and cid not in _checkpointed_chunk_ids]
    chunks_without_entities = [(cid, content) for cid, content in rows
                               if len(chunk_entities_map.get(cid, [])) < 2]

    if max_chunks is not None:
        chunks_with_entities = chunks_with_entities[:max_chunks]

    print(f"Loaded {len(rows):,} chunks from {db_path}")
    if _checkpointed_chunk_ids:
        print(f"  Resuming: {len(_checkpointed_chunk_ids):,} chunks already checkpointed")
    print(f"  {len(chunks_with_entities):,} chunks have ≥2 entities → entity-anchored extraction"
          + (f" (capped at {max_chunks})" if max_chunks is not None else ""))
    print(f"  {len(chunks_without_entities):,} chunks have <2 entities → skipped")

    _CHECKPOINT_INTERVAL = 100
    _checkpoint_lock = _threading.Lock()
    _pending_index = RelationshipIndex()      # triples since last checkpoint
    _pending_descs: dict[str, str] = {}
    _pending_aliases: dict[str, list[str]] = {}

    def _flush_checkpoint():
        nonlocal _pending_index, _pending_descs, _pending_aliases
        if len(_pending_index) == 0:
            return
        for _attempt in range(10):
            try:
                _con_rw = duckdb.connect(str(db_path))
                _pending_index.save_to_db(_con_rw, incremental=True)
                _con_rw.close()
                break
            except Exception as _e:
                if "lock" in str(_e).lower() or "conflict" in str(_e).lower():
                    _time.sleep(5)
                else:
                    raise
        _pending_index = RelationshipIndex()
        _pending_descs = {}
        _pending_aliases = {}

    relationship_index = RelationshipIndex()
    new_descriptions: dict[str, str] = {}
    progress_out = getattr(args, "progress_out", None)
    _progress_fh = None
    if progress_out:
        import sys as _sys
        _progress_fh = _sys.stdout if progress_out == "-" else open(progress_out, "w", buffering=1)

    def _extract_one(row):
        chunk_id, content = row
        entities = chunk_entities_map.get(chunk_id, [])
        n_ent = len(entities)
        _svo_log(f"chunk start  chunk_id={chunk_id!r} entities={n_ent} content={len(content or ''):,}chars")
        if n_ent >= 2:
            t0 = _time.monotonic()
            triples, descs, aliases, rel_descs = extractor.extract_entity_anchored(
                content or "", chunk_id, entities
            )
            elapsed = _time.monotonic() - t0
            _svo_log(
                f"chunk done   chunk_id={chunk_id!r} "
                f"triples={len(triples)} descs={len(descs)} aliases={len(aliases)} "
                f"elapsed={elapsed:.2f}s"
            )
            return triples, descs, aliases, rel_descs
        _svo_log(f"chunk skip   chunk_id={chunk_id!r} entities={n_ent} (<2)")
        return [], {}, {}, {}

    new_aliases: dict[str, list[str]] = {}
    total_chunks = len(chunks_with_entities)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_extract_one, row): row for row in chunks_with_entities}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            chunk_id = futures[fut][0]
            triples, descs, aliases, rel_descs = fut.result()
            for t in triples:
                relationship_index.add(t)
                _pending_index.add(t)
            new_descriptions.update(descs)
            _pending_descs.update(descs)
            for eid, alias_list in aliases.items():
                new_aliases.setdefault(eid, []).extend(alias_list)
                _pending_aliases.setdefault(eid, []).extend(alias_list)
            done += 1
            if done % _CHECKPOINT_INTERVAL == 0:
                with _checkpoint_lock:
                    _flush_checkpoint()
            if _progress_fh is not None:
                import json as _json
                event = {
                    "done": done,
                    "total": total_chunks,
                    "chunk_id": chunk_id,
                    "triples": [
                        {"subject_id": t.subject_id, "verb": t.verb,
                         "object_id": t.object_id, "confidence": t.confidence,
                         "description": t.description}
                        for t in triples
                    ],
                    "descriptions": descs,
                    "aliases": aliases,
                    "rel_descriptions": rel_descs,
                }
                _progress_fh.write(_json.dumps(event) + "\n")
            _svo_log(f"progress     {done:,}/{total_chunks:,} total_triples={len(relationship_index):,}")

    if _progress_fh is not None and progress_out != "-":
        _progress_fh.close()


    from chonk.storage._store import Store as _Store
    # Flush any remaining pending triples
    with _checkpoint_lock:
        _flush_checkpoint()
    for _attempt in range(60):
        try:
            con_rw = duckdb.connect(str(db_path))
            n_written = con_rw.execute("SELECT COUNT(*) FROM svo_triples").fetchone()[0]
            con_rw.close()
            print(f"Persisted {n_written:,} triples → {db_path}")
            break
        except Exception as _e:
            if "lock" in str(_e).lower() or "conflict" in str(_e).lower():
                print(f"  DB locked, waiting 10s... (attempt {_attempt+1}/60)", flush=True)
                _time.sleep(10)
            else:
                raise
    else:
        raise RuntimeError("Could not acquire write lock on DB after 10 minutes.")

    if new_descriptions:
        with _Store(db_path, embedding_dim=EMBED_DIM) as _desc_store:
            n_desc = _desc_store.set_entity_descriptions_batch(new_descriptions)
        print(f"Persisted {n_desc:,} new entity descriptions → {db_path}")

    # Persist LLM-generated aliases (first-registration wins per alias/namespace)
    if new_aliases:
        flat_aliases = {
            alias: eid
            for eid, alias_list in new_aliases.items()
            for alias in alias_list
        }
        with _Store(db_path, embedding_dim=EMBED_DIM) as _alias_store:
            n_alias = _alias_store.add_entity_aliases_batch(flat_aliases, source="llm")
        print(f"Persisted {n_alias:,} entity aliases → {db_path}")

    # Embed entities as chunk_type='entity' rows in the embeddings table
    _upsert_entity_chunk_embeddings(db_path)

    if getattr(args, "with_context_graph", False):
        import duckdb as _ddb

        from chonk.graph._context_graph import build_context_graph_edges
        _con = _ddb.connect(str(db_path))
        stats = build_context_graph_edges(_con, namespace="global", force=True)
        _con.close()
        print(f"  Context graph built: {stats.entity_count:,} entities, {stats.edge_count:,} edges")


def _upsert_entity_chunk_embeddings(db_path: Path) -> None:
    """Embed all entities using name + aliases + description and upsert as chunk_type='entity'."""
    import duckdb
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from chonk.models import DocumentChunk
    from chonk.storage._store import Store as _Store

    con_ro = duckdb.connect(str(db_path), read_only=True)
    try:
        entity_rows = con_ro.execute("""
            SELECT e.id, e.name,
                   COALESCE(e.description, '') AS description,
                   e.entity_type
            FROM entities e
        """).fetchall()

        if not entity_rows:
            con_ro.close()
            return

        # Gather aliases per entity
        alias_rows = con_ro.execute(
            "SELECT entity_id, alias FROM entity_aliases WHERE namespace = 'global'"
        ).fetchall() if _table_exists(con_ro, "entity_aliases") else []
    finally:
        con_ro.close()

    from collections import defaultdict as _dd
    aliases_map: dict[str, list[str]] = _dd(list)
    for eid, alias in alias_rows:
        aliases_map[eid].append(alias)

    texts: list[str] = []
    chunks: list[DocumentChunk] = []
    for entity_id, name, description, entity_type in entity_rows:
        alias_str = ", ".join(aliases_map.get(entity_id, []))
        parts = [name]
        if alias_str:
            parts.append(alias_str)
        if description:
            parts.append(description)
        text = ". ".join(parts)
        texts.append(text)
        chunks.append(DocumentChunk(
            document_name=f"__entity__{entity_id}",
            content=text,
            chunk_index=0,
            chunk_type="entity",
        ))

    print(f"  Embedding {len(chunks):,} entities for semantic search...")
    embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    vecs = embed_model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
    ).astype("float32")
    del embed_model

    # chunk_id for entity rows = entity_id; use document_name as the lookup key
    with _Store(db_path, embedding_dim=EMBED_DIM) as store:
        # Remove stale entity rows before re-inserting
        store._db.conn.execute("DELETE FROM embeddings WHERE chunk_type = 'entity'")
        store.add_document(chunks, np.array(vecs))
    print(f"  Persisted {len(chunks):,} entity embeddings → {db_path}")


def cmd_build_context_graph(args: argparse.Namespace) -> None:
    """Build context graph edges from chunk_entities (and optionally svo_triples)."""
    from chonk.storage._store import Store

    data_dir = Path(args.out_dir) / "data"
    db_name  = getattr(args, "db_name", None) or DB_FILENAME
    db_path  = data_dir / db_name
    force    = getattr(args, "force", False)
    namespace_arg = getattr(args, "namespace", "global")
    namespace = None if namespace_arg == "all" else namespace_arg
    min_weight = getattr(args, "min_weight", 0.1)
    algorithm  = getattr(args, "algorithm", "agglomerative")
    min_chunks = getattr(args, "min_chunks", 10)

    if not db_path.exists():
        raise FileNotFoundError(f"Index DB not found: {db_path}")

    print(f"Building context graph: {db_path}")
    ns_label = "all namespaces" if namespace is None else repr(namespace)
    print(f"  namespace={ns_label}, min_weight={min_weight}, algorithm={algorithm!r}, min_chunks={min_chunks}")

    with Store(str(db_path), embedding_dim=EMBED_DIM) as store:
        result = store.build_context_graph(
            namespace=namespace,
            min_weight=min_weight,
            force=force,
            algorithm=algorithm,
            min_chunks=min_chunks,
        )

    if isinstance(result, dict):
        for ns, stats in result.items():
            print(f"  [{ns}] {stats.entity_count:,} entities, {stats.edge_count:,} edges, {stats.chunk_count:,} chunks")
    else:
        stats = result
        print(f"  {stats.entity_count:,} entities, {stats.edge_count:,} edges, {stats.chunk_count:,} chunks")


# ─────────────────────────────────────────────────────────────────────────────
# Per-run DuckDB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _connect_with_retry(db_path, max_attempts: int = 30, delay: float = 2.0):
    """Open a DuckDB write connection, retrying on lock conflicts."""
    import time as _time

    import duckdb
    for attempt in range(max_attempts):
        try:
            return duckdb.connect(str(db_path))
        except Exception as e:
            if ("lock" in str(e).lower() or "conflict" in str(e).lower()) and attempt < max_attempts - 1:
                _time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Could not acquire DB write lock after {max_attempts * delay:.0f}s")


def _run_db_path(data_dir: Path, run_name: str) -> Path:
    runs_dir = data_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    return runs_dir / f"{run_name}.duckdb"


def _init_run_db(db_path: Path) -> None:
    con = _connect_with_retry(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            question TEXT,
            source TEXT,
            question_type TEXT,
            context TEXT,
            evidence TEXT,
            generated_answer TEXT,
            gold_answer TEXT,
            retrieved_chunks TEXT,
            retrieved_scores TEXT,
            expansion_stats TEXT,
            entity_ref_retry TEXT,
            retrieval_trace TEXT,
            srr TEXT
        )
    """)
    for _col in ("retrieval_trace TEXT", "srr TEXT"):
        try:
            con.execute(f"ALTER TABLE results ADD COLUMN {_col}")
        except Exception:
            pass  # column already exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS eval_scores (
            id TEXT PRIMARY KEY,
            question_type TEXT,
            answer_correctness REAL,
            rouge_score REAL,
            coverage_score REAL,
            faithfulness REAL,
            nan_reason TEXT,
            decomposed_score REAL,
            decomposed_detail TEXT
        )
    """)
    for _col, _type in [("nan_reason", "TEXT"), ("decomposed_score", "REAL"), ("decomposed_detail", "TEXT")]:
        try:
            con.execute(f"ALTER TABLE eval_scores ADD COLUMN {_col} {_type}")
        except Exception:
            pass  # column already exists
    con.close()


def _upsert_results_to_db(db_path: Path, results: list[dict]) -> None:
    """Insert or replace a batch of results without clearing the table."""
    import json as _json
    con = _connect_with_retry(db_path)
    for r in results:
        con.execute(
            "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                r.get("id"), r.get("question"), r.get("source"),
                r.get("question_type"), r.get("context"),
                _json.dumps(r.get("evidence")),
                r.get("generated_answer"), r.get("gold_answer"),
                _json.dumps(r.get("retrieved_chunks")),
                _json.dumps(r.get("retrieved_scores")),
                _json.dumps(r.get("entity_ref_expansion")) if r.get("entity_ref_expansion") else None,
                _json.dumps(r.get("entity_ref_retry")) if r.get("entity_ref_retry") else None,
                _json.dumps(r.get("retrieval_trace")) if r.get("retrieval_trace") else None,
                _json.dumps(r["srr"]) if r.get("srr") else None,
            ],
        )
    con.close()


def _write_results_to_db(db_path: Path, results: list[dict]) -> None:
    import json as _json
    # Deduplicate by id (last write wins — handles checkpoint resume duplicates)
    seen: dict[str, dict] = {}
    for r in results:
        seen[r.get("id")] = r
    results = list(seen.values())
    con = _connect_with_retry(db_path)
    con.execute("DELETE FROM results")
    for r in results:
        con.execute(
            "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                r.get("id"), r.get("question"), r.get("source"),
                r.get("question_type"), r.get("context"),
                _json.dumps(r.get("evidence")),
                r.get("generated_answer"), r.get("gold_answer"),
                _json.dumps(r.get("retrieved_chunks")),
                _json.dumps(r.get("retrieved_scores")),
                _json.dumps(r.get("entity_ref_expansion")) if r.get("entity_ref_expansion") else None,
                _json.dumps(r.get("entity_ref_retry")) if r.get("entity_ref_retry") else None,
                _json.dumps(r.get("retrieval_trace")) if r.get("retrieval_trace") else None,
                _json.dumps(r["srr"]) if r.get("srr") else None,
            ],
        )
    con.close()


def _read_results_from_db(db_path: Path) -> list[dict]:
    import json as _json

    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute("SELECT * FROM results").fetchall()
    cols = ["id","question","source","question_type","context","evidence",
            "generated_answer","gold_answer","retrieved_chunks","retrieved_scores",
            "expansion_stats","entity_ref_retry","retrieval_trace","srr"]
    con.close()
    records = []
    for row in rows:
        r = dict(zip(cols[:len(row)], row))
        for key in ("evidence","retrieved_chunks","retrieved_scores","expansion_stats","entity_ref_retry","retrieval_trace","srr"):
            if r.get(key):
                try: r[key] = _json.loads(r[key])
                except Exception: pass
        records.append(r)
    return records


def _load_community_index(db_path: Path):
    """Load CommunityIndex from DB, return None if not built."""
    from chonk.community import CommunityIndex
    try:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        n = con.execute("SELECT COUNT(*) FROM chunk_communities").fetchone()[0]
        con.close()
        if n == 0:
            return None
        print(f"Loading CommunityIndex from DB ({n:,} assignments)...")
        return CommunityIndex.from_db(db_path)
    except Exception:
        return None


def _load_entity_embeddings(db_path: Path):
    """Load entity embeddings matrix and ID list from DB. Returns (matrix, ids) or (None, None)."""
    import duckdb
    import numpy as np

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("SELECT entity_id, embedding FROM entity_embeddings").fetchall()
    except Exception:
        con.close()
        return None, None
    con.close()
    if not rows:
        return None, None
    ids = [r[0] for r in rows]
    mat = np.array([r[1] for r in rows], dtype="float32")
    return mat, ids


def _load_entity_index_from_db(db_path: Path, namespaces: list[str] | None = None) -> EntityIndex:
    """Reconstruct EntityIndex from persisted chunk_entities table."""
    import duckdb

    from chonk.ner import EntityIndex

    con = duckdb.connect(str(db_path), read_only=True)
    _view_exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = 'all_chunk_entities'"
    ).fetchone()[0] > 0
    _ce_table = "all_chunk_entities" if _view_exists else "chunk_entities"
    if namespaces is not None:
        placeholders = ", ".join(["?" for _ in namespaces])
        rows = con.execute(
            f"SELECT chunk_id, entity_id, frequency, positions_json, score FROM {_ce_table} WHERE namespace IN ({placeholders})",
            namespaces,
        ).fetchall()
        total_chunks = con.execute(
            f"SELECT COUNT(DISTINCT chunk_id) FROM {_ce_table} WHERE namespace IN ({placeholders})",
            namespaces,
        ).fetchone()[0]
    else:
        rows = con.execute(
            f"SELECT chunk_id, entity_id, frequency, positions_json, score FROM {_ce_table}"
        ).fetchall()
        total_chunks = con.execute(f"SELECT COUNT(DISTINCT chunk_id) FROM {_ce_table}").fetchone()[0]
    con.close()

    associations = [
        {
            "entity_id": r[1],
            "chunk_id": r[0],
            "frequency": r[2],
            "positions": json.loads(r[3]) if r[3] else [],
            "score": r[4],
            "chunk_length": 1,
        }
        for r in rows
    ]
    return EntityIndex.from_dict({
        "total_chunks": total_chunks,
        "score_weights": [0.4, 0.3, 0.3],
        "associations": associations,
    })


def _prune_redundant(hits, db_conn, threshold):
    """Remove near-duplicate chunks from reranked hits (greedy, post-rerank).

    Fetches embeddings in one batch query. Keeps the first occurrence when
    two chunks have cosine similarity >= threshold.
    """
    import math
    if threshold is None or len(hits) <= 1:
        return hits
    chunk_ids = [cid for cid, _, _ in hits]
    placeholders = ", ".join(["?" for _ in chunk_ids])
    rows = db_conn.execute(
        f"SELECT chunk_id, embedding FROM embeddings WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    embs = {row[0]: list(row[1]) for row in rows if row[1] is not None}

    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    selected, selected_embs = [], []
    for hit in hits:
        cid = hit[0]
        emb = embs.get(cid, [])
        if emb and selected_embs:
            if max(_cosine(emb, se) for se in selected_embs if se) >= threshold:
                continue
        selected.append(hit)
        selected_embs.append(emb)
    return selected


_FANG_DOMAINS = [
    ("sec_10k",  "global", "SEC 10-K Filings",
     "Annual financial reports and business disclosures filed by public companies with the SEC"),
    ("cve",      "global", "CVE Security Records",
     "Common Vulnerabilities and Exposures records describing software security vulnerabilities"),
    ("fed_reg",  "global", "Federal Register",
     "US government regulatory documents, proposed rules, final rules, and agency notices"),
    ("patents",  "global", "US Patents",
     "United States utility and design patents describing inventions and innovations"),
]


def _register_fang_domains(store) -> None:
    """Register FANG corpus domains and tag embeddings with domain_id by document_name pattern."""
    for domain_id, ns, name, desc in _FANG_DOMAINS:
        store.register_domain(domain_id, ns, name, desc)
    conn = store.vector._conn
    conn.execute("UPDATE embeddings SET domain_id = 'sec_10k' WHERE document_name LIKE '%_10k_%'")
    conn.execute("UPDATE embeddings SET domain_id = 'cve'     WHERE document_name LIKE 'CVE-%'")
    conn.execute("UPDATE embeddings SET domain_id = 'fed_reg' WHERE document_name LIKE 'fr_%'")
    conn.execute(
        "UPDATE embeddings SET domain_id = 'patents' "
        "WHERE regexp_matches(document_name, '^US[0-9]') AND domain_id IS NULL"
    )
    tagged = conn.execute(
        "SELECT domain_id, COUNT(*) FROM embeddings WHERE domain_id IS NOT NULL GROUP BY domain_id"
    ).fetchall()
    print(f"[ADF] Domain registration: {dict(tagged)}")


def _build_domain_filter_fn(openai_client, model: str):
    """Return a callable suitable for EnhancedSearch.search(domain_filter_llm_fn=...)."""
    def _fn(prompt: str) -> str:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""
    return _fn


def _build_enhanced_search(store, db_path: Path | None = None, use_ner_x: bool = False, embed_model=None, entity_ref_expansion: bool = False, entity_ref_expansion_k: int = 20, entity_ref_expansion_per_k: int | None = None, entity_ref_expansion_min_sim: float | None = None, use_cluster: bool = False, lane_entity_min_sim: float | None = None, namespaces: list[str] | None = None, domain_ids: list[str] | None = None, community_index=None, context_graph_expansion: bool = False, context_graph_min_weight: float = 0.1, context_graph_top_k: int = 5):
    """Load EnhancedSearch: from pre-built DB tables if available, else rebuild in memory."""
    import duckdb

    from chonk.ner import EntityIndex, SpacyMatcher
    from chonk.search import EnhancedSearch
    from chonk.storage._store import Store as _Store
    from chonk.storage._vector import DuckDBVectorBackend

    _session_fingerprint: str | None = (
        _Store.session_fingerprint(domain_ids) if domain_ids else None
    )

    if db_path is not None and db_path.exists():
        con = duckdb.connect(str(db_path), read_only=True)
        n = con.execute("SELECT COUNT(*) FROM chunk_entities").fetchone()[0]
        con.close()
        if n > 0:
            print(f"Loading EntityIndex from DB ({n:,} associations)...")
            entity_index = _load_entity_index_from_db(db_path, namespaces=namespaces)
            print(f"  {entity_index.total_chunks():,} chunks, {len(entity_index.entity_ids()):,} entities")

            cluster_map = None
            if use_cluster:
                from chonk.cluster import ClusterMap
                print("  Building ClusterMap...")
                cluster_map = ClusterMap.build(entity_index)
                print(f"  {cluster_map.cluster_count():,} clusters across {cluster_map.entity_count():,} entities")

            matcher = SpacyMatcher(model=SPACY_MODEL, strip_numeric=True)
            query_ner_fn = lambda text: [m.display_name for m in matcher.match(text)]
            query_entity_id_fn = lambda text: [m.entity_id for m in matcher.match(text)]

            ner_x_kwargs: dict = {}
            if use_ner_x and embed_model is not None:
                mat, ids = _load_entity_embeddings(db_path)
                if mat is not None:
                    print(f"  Loaded {len(ids):,} entity embeddings for ner-x expansion.")
                    ner_x_kwargs = dict(
                        entity_embedding_expansion=True,
                        entity_embeddings=mat,
                        entity_embedding_ids=ids,
                        ner_fn=lambda text: [m.entity_id for m in matcher.match(text)],
                        entity_embedding_top_k=10,
                    )
                else:
                    print("  entity_embeddings table empty — ner-x disabled. Run build-ner --with-embeddings first.")

            embed_fn_kwargs: dict = {}
            if embed_model is not None:
                embed_fn_kwargs["embed_fn"] = lambda texts: embed_model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )

            import duckdb as _ddb

            from chonk.graph import RelationshipIndex

            _con = _ddb.connect(str(db_path), read_only=True)
            relationship_index = RelationshipIndex.load_from_db(_con, namespaces=namespaces)
            _con.close()
            if len(relationship_index) == 0:
                relationship_index = None
            else:
                print(f"  Loaded {len(relationship_index):,} SVO triples from DB.")

            return EnhancedSearch(
                store,
                entity_index=entity_index,
                cluster_map=cluster_map,
                structural_expansion=False,
                cluster_expansion=use_cluster,
                query_ner_fn=query_ner_fn,
                entity_ref_expansion=entity_ref_expansion,
                entity_ref_expansion_k=entity_ref_expansion_k,
                entity_ref_expansion_per_k=entity_ref_expansion_per_k,
                entity_ref_expansion_min_sim=entity_ref_expansion_min_sim,
                lane_entity_min_sim=lane_entity_min_sim,
                relationship_index=relationship_index,
                session_fingerprint=_session_fingerprint,
                community_index=community_index,
                query_entity_id_fn=query_entity_id_fn,
                context_graph_expansion=context_graph_expansion,
                context_graph_min_weight=context_graph_min_weight,
                context_graph_top_k=context_graph_top_k,
                **ner_x_kwargs,
                **embed_fn_kwargs,
            )

    print(f"Building EntityIndex with SpacyMatcher({SPACY_MODEL}) (not pre-built, rebuild in memory)...")
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
    cluster_map = None
    if use_cluster:
        from chonk.cluster import ClusterMap
        print("  Building ClusterMap...")
        cluster_map = ClusterMap.build(entity_index)
        print(f"  {cluster_map.cluster_count():,} clusters across {cluster_map.entity_count():,} entities")
    query_ner_fn = lambda text: [m.display_name for m in matcher.match(text)]
    query_entity_id_fn = lambda text: [m.entity_id for m in matcher.match(text)]
    return EnhancedSearch(
        store,
        entity_index=entity_index,
        cluster_map=cluster_map,
        structural_expansion=False,
        cluster_expansion=use_cluster,
        query_ner_fn=query_ner_fn,
        query_entity_id_fn=query_entity_id_fn,
        entity_ref_expansion=entity_ref_expansion,
        entity_ref_expansion_k=entity_ref_expansion_k,
        entity_ref_expansion_per_k=entity_ref_expansion_per_k,
        entity_ref_expansion_min_sim=entity_ref_expansion_min_sim,
        lane_entity_min_sim=lane_entity_min_sim,
        session_fingerprint=_session_fingerprint,
        community_index=community_index,
        context_graph_expansion=context_graph_expansion,
        context_graph_min_weight=context_graph_min_weight,
        context_graph_top_k=context_graph_top_k,
    )


def cmd_run(args: argparse.Namespace) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import numpy as np
    import openai
    from sentence_transformers import SentenceTransformer

    _cfg = _load_config(getattr(args, "config", None))
    _apply_config(_cfg, args)

    out_dir     = Path(args.out_dir)
    data_dir    = out_dir / "data"
    results_dir = out_dir / "results"
    results_dir.mkdir(exist_ok=True)
    run_name     = getattr(args, "run_name", "contextual")

    # Kill any stale processes already running the same run-name
    import signal
    import subprocess as _sp
    _my_pid = os.getpid()
    try:
        _procs = _sp.check_output(
            ["pgrep", "-f", f"graphrag_bench.py.*--run-name {run_name}( |$)"],
            text=True,
        ).split()
        for _pid in _procs:
            _pid = int(_pid)
            if _pid != _my_pid:
                try:
                    os.kill(_pid, signal.SIGKILL)
                    print(f"[preflight] killed stale pid {_pid} ({run_name})", flush=True)
                except ProcessLookupError:
                    pass
    except _sp.CalledProcessError:
        pass  # no matching processes
    results_f    = results_dir / f"{run_name}.jsonl"
    ckpt_f       = results_dir / f"{run_name}_checkpoint.jsonl"
    use_vanilla  = getattr(args, "vanilla", False)
    use_rerank        = getattr(args, "rerank", False)
    rerank_provider   = getattr(args, "rerank_provider", "local")
    use_enhanced      = getattr(args, "enhanced", False)
    use_ner_x         = getattr(args, "ner_x", False)
    use_entity_ref_expansion = getattr(args, "entity_ref_expansion", False)
    entity_ref_expansion_per_k  = getattr(args, "entity_ref_expansion_per_k", None)
    entity_ref_expansion_min_sim = getattr(args, "entity_ref_expansion_min_sim", None)
    use_cluster = getattr(args, "cluster", False)
    use_context_graph = getattr(args, "context_graph", False)
    context_graph_min_weight = getattr(args, "context_graph_min_weight", 0.1)
    context_graph_top_k = getattr(args, "context_graph_top_k", 5)
    search_mode = getattr(args, "search_mode", "vector_first")
    use_entity_ref_retry     = getattr(args, "entity_ref_retry", False)
    use_structured_gen       = getattr(args, "structured_gen", False)
    use_breadcrumb_context = getattr(args, "breadcrumb_context", False)
    breadcrumb_style       = getattr(args, "breadcrumb_style", "markdown")
    use_community_context  = getattr(args, "community_context", False)
    community_min_coherence = getattr(args, "community_min_coherence", 0.0)
    breadcrumb_embed       = getattr(args, "breadcrumb_embed", False)
    no_breadcrumb_embed    = not breadcrumb_embed  # legacy alias for internal logic
    redundancy_threshold   = getattr(args, "redundancy_threshold", None)
    lane_entity_min_sim    = getattr(args, "lane_entity_min_sim", None)
    concentration_threshold = getattr(args, "concentration_threshold", None)
    query_complexity_threshold = getattr(args, "query_complexity_threshold", 2)
    namespaces = getattr(args, "namespaces", None)
    domain_ids = getattr(args, "domain_ids", None)
    auto_domain_filter = getattr(args, "auto_domain_filter", False)
    db_name_override = getattr(args, 'db_name', None)
    question_ids_file = getattr(args, 'question_ids', None)
    if db_name_override:
        db_path = data_dir / db_name_override
    else:
        _ctx_db = DB_FILENAME.replace(".duckdb", "_nobc.duckdb") if no_breadcrumb_embed else DB_FILENAME
        db_path = data_dir / (VANILLA_DB_FILENAME if use_vanilla else _ctx_db)
    top_k        = VANILLA_K if use_vanilla else (getattr(args, "top_k", None) or K)
    gen_temperature = VANILLA_TEMPERATURE  # paper: 0.7 for all systems

    _gen_model_check = getattr(args, "gen_model", GEN_MODEL)
    if _gen_model_check == "gpt-4o" and top_k > 10:
        raise SystemExit(
            f"BLOCKED: gpt-4o with top_k={top_k} > 10 is too expensive (~${top_k * 4071 * 14000 // 4 * 250 // 1_000_000_000} estimated). "
            f"Use top_k <= 10 or switch to gpt-4o-mini."
        )

    if not db_path.exists():
        print("No index found. Run 'index' first.")
        return

    # Write sidecar flags file for reproducibility
    _flags = {
        "run_name": run_name,
        "db": str(db_path.name),
        "gen_model": getattr(args, "gen_model", GEN_MODEL),
        "vanilla": use_vanilla,
        "rerank": use_rerank,
        "rerank_provider": rerank_provider,
        "enhanced": use_enhanced,
        "entity_ref_expansion": use_entity_ref_expansion,
        "entity_ref_expansion_min_sim": entity_ref_expansion_min_sim,
        "entity_ref_expansion_per_k": entity_ref_expansion_per_k,
        "lane_entity_min_sim": lane_entity_min_sim,
        "community_context": use_community_context,
        "community_min_coherence": community_min_coherence,
        "redundancy_threshold": redundancy_threshold,
        "breadcrumb_embed": breadcrumb_embed,
        "breadcrumb_context": use_breadcrumb_context,
        "breadcrumb_style": breadcrumb_style,
        "cluster": use_cluster,
        "context_graph": use_context_graph,
        "context_graph_min_weight": context_graph_min_weight,
        "context_graph_top_k": context_graph_top_k,
        "ner_x": use_ner_x,
        "srr": getattr(args, "srr", False),
        "search_mode": search_mode,
        "concentration_threshold": concentration_threshold,
        "query_complexity_threshold": query_complexity_threshold,
        "auto_domain_filter": auto_domain_filter,
        "top_k": top_k,
        "question_ids": question_ids_file,
        "corpus": "full" if (question_ids_file and "full_corpus" in str(question_ids_file)) else "grid" if question_ids_file else "all",
    }
    _flags_path = results_dir / f"{run_name}_flags.json"
    _flags_path.write_text(json.dumps(_flags, indent=2), encoding="utf-8")
    # Append to consolidated manifest
    _manifest_path = out_dir / "run_manifest.jsonl"
    import datetime as _dt
    _manifest_entry = {"timestamp": _dt.datetime.utcnow().isoformat() + "Z", **_flags}
    with open(_manifest_path, "a", encoding="utf-8") as _mf:
        _mf.write(json.dumps(_manifest_entry) + "\n")

    import atexit as _atexit
    import signal as _signal
    _run_completed = [False]
    def _cleanup_flags():
        if not _run_completed[0] and _flags_path.exists():
            _flags_path.unlink(missing_ok=True)
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
    _atexit.register(_cleanup_flags)
    def _sigterm_handler(signum, frame):
        raise SystemExit(0)
    _signal.signal(_signal.SIGTERM, _sigterm_handler)

    questions = _load_questions(data_dir)
    if question_ids_file:
        with open(question_ids_file) as _f:
            _order = json.load(_f)
        _id_to_q = {q.get("id"): q for q in questions}
        questions = [_id_to_q[qid] for qid in _order if qid in _id_to_q]
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

    run_db = _run_db_path(data_dir, run_name)
    _init_run_db(run_db)

    # Resume from DuckDB: load fully-generated results AND pre-built work_items.
    _db_pending_items: list[dict] = []
    if run_db.exists():
        import duckdb as _duckdb
        try:
            _con = _duckdb.connect(str(run_db))
            _db_done = set(
                row[0] for row in _con.execute(
                    "SELECT id FROM results WHERE generated_answer IS NOT NULL"
                ).fetchall()
            )
            if _db_done - done_ids:
                print(f"Resuming from DB: {len(_db_done)} already generated")
                done_ids.update(_db_done)
            # Load rows with context already built but gen not yet done — skip retrieval for these
            _db_rows = _con.execute(
                "SELECT id, question, source, question_type, context, evidence, "
                "gold_answer, retrieved_chunks, retrieved_scores, expansion_stats "
                "FROM results WHERE generated_answer IS NULL AND context IS NOT NULL"
            ).fetchall()
            _con.close()
            for _row in _db_rows:
                _qid, _q, _src, _qtype, _ctx, _ev, _gold, _cids, _scores, _exp = _row
                if _qid not in done_ids:
                    _db_pending_items.append({
                        "qid": _qid, "question": _q, "source": _src or "?",
                        "qtype": _qtype or "?", "context": _ctx or "",
                        "chunk_ids": _cids or [], "scores": _scores or [],
                        "chunk_texts": [], "expansion_stats": _exp,
                        "evidence": _ev or [], "gold": _gold or "",
                        "sub_queries": None,
                    })
            if _db_pending_items:
                _db_pending_ids = {it["qid"] for it in _db_pending_items}
                done_ids.update(_db_pending_ids)
                print(f"Resuming {len(_db_pending_items)} pre-built work_items from DB (skipping retrieval)")
        except Exception:
            pass

    pending = [(i, q) for i, q in enumerate(questions)
               if q.get("id", f"q{i}") not in done_ids]
    print(f"Pending: {len(pending)}")

    # ── 1. Embed all questions (full-corpus cache — order-independent across runs)
    # Cache key is the full unfiltered corpus so any --question-ids subset or
    # reordering reuses the same cache file via ID lookup.
    q_vecs_cache = data_dir / "question_embeddings.npy"
    q_ids_cache  = data_dir / "question_ids.json"

    _corpus_qs  = _load_questions(data_dir)
    _corpus_ids = [q.get("id", f"q{i}") for i, q in enumerate(_corpus_qs)]

    _corpus_vecs = None
    if q_vecs_cache.exists() and q_ids_cache.exists():
        if json.loads(q_ids_cache.read_text()) == _corpus_ids:
            print(f"Loading cached question embeddings from {q_vecs_cache}")
            _corpus_vecs = np.load(str(q_vecs_cache))
        else:
            q_vecs_cache.unlink()  # stale — corpus changed

    import hashlib as _hl_pre
    _ent_cache_key_pre = _hl_pre.md5(
        (EMBED_MODEL + SPACY_MODEL + json.dumps(_corpus_ids)).encode()
    ).hexdigest()[:16]
    _ent_vecs_cache_path = data_dir / f"entity_vecs_{_ent_cache_key_pre}.npz"
    _ent_ents_cache_path = data_dir / f"entity_ents_{_ent_cache_key_pre}.json"
    _ent_cache_exists    = _ent_vecs_cache_path.exists() and _ent_ents_cache_path.exists()

    _needs_entity_embed_at_run = use_ner_x or use_entity_ref_expansion or concentration_threshold is not None
    if _needs_entity_embed_at_run and not _ent_cache_exists:
        raise RuntimeError(
            f"Entity embedding cache missing for model '{EMBED_MODEL}' / spaCy '{SPACY_MODEL}'.\n"
            f"Run: python demo/graphrag_bench.py prime-cache --out-dir {args.out_dir}"
        )
    if _corpus_vecs is None:
        raise RuntimeError(
            f"Question embedding cache missing.\n"
            f"Run: python demo/graphrag_bench.py prime-cache --out-dir {args.out_dir}"
        )

    use_sr  = getattr(args, "sr", False)
    use_srr = getattr(args, "srr", False)
    use_multi_step = getattr(args, "multi_step", False)
    embed_model = None
    if use_entity_ref_retry:
        _embed_device = os.environ.get("EMBED_DEVICE") or None
        if not _embed_device:
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not torch.cuda.is_available():
                    _embed_device = "cpu"
            except ImportError:
                pass
        embed_model = SentenceTransformer(EMBED_MODEL, device=_embed_device) if _embed_device else SentenceTransformer(EMBED_MODEL)

    # Build lookup and align to the (possibly filtered/reordered) questions list
    _id_to_vec = {qid: _corpus_vecs[i] for i, qid in enumerate(_corpus_ids)}
    all_vecs = np.stack([_id_to_vec[q.get("id", f"q{i}")] for i, q in enumerate(questions)])

    # slice to pending indices only
    pending_indices = [i for i, _ in pending]
    q_vecs = all_vecs[pending_indices]

    # ── 2. Retrieve context for each question (sequential; DuckDB conn is serialized)
    fetch_k = max(K_FETCH, top_k) if use_rerank else top_k
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
            import os as _os

            from sentence_transformers import CrossEncoder
            _rerank_device = getattr(args, "rerank_device", None) or _os.environ.get("RERANKER_DEVICE") or None
            if not _rerank_device:
                try:
                    import torch
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not torch.cuda.is_available():
                        _rerank_device = "cpu"
                except ImportError:
                    pass
            print(f"Loading reranker: {RERANK_MODEL}{'  [device='+_rerank_device+']' if _rerank_device else ''}...")
            reranker = CrossEncoder(RERANK_MODEL, max_length=512, device=_rerank_device) if _rerank_device else CrossEncoder(RERANK_MODEL, max_length=512)

    _need_community = use_community_context or search_mode in ("graph_first", "map_reduce_global")
    community_index = _load_community_index(db_path) if _need_community else None
    if use_community_context and community_index is None:
        print("WARNING: --community-context set but no community index found. Run 'build-community' first.")

    # Build map-reduce LLM fn when needed (before entering the store block)
    _map_reduce_llm_fn = None
    if search_mode == "map_reduce_global":
        import openai as _mr_oai
        _mr_gen_model = getattr(args, "gen_model", GEN_MODEL)
        _mr_provider  = getattr(args, "gen_provider", "openai")
        if _mr_provider == "together":
            _mr_client = _mr_oai.OpenAI(
                api_key=os.environ["TOGETHER_API_KEY"], base_url=TOGETHER_BASE_URL, timeout=120.0
            )
        elif _mr_provider == "anthropic":
            _mr_client = _mr_oai.OpenAI(
                api_key=os.environ["ANTHROPIC_API_KEY"], base_url=ANTHROPIC_BASE_URL,
                default_headers={"anthropic-version": "2023-06-01"}, timeout=120.0,
            )
        else:
            _mr_client = _mr_oai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)

        def _map_reduce_llm_fn(prompt: str) -> str:
            resp = _mr_client.chat.completions.create(
                model=_mr_gen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content or ""

    # ── Complexity / concentration helpers ────────────────────────────────────
    _complexity_matcher = None
    _CONJUNCTION_WORDS = frozenset(["and", "or", "vs", "versus", "between"])

    def _query_complexity(question: str) -> int:
        """Return complexity score = entity_count + clause_signal_count."""
        entity_count = 0
        if _complexity_matcher is not None:
            entity_count = len(_complexity_matcher.match(question))
        tokens = re.sub(r"[^\w\s?]", " ", question.lower()).split()
        clause_signals = (
            question.count("?")
            + question.count(",")
            + sum(1 for t in tokens if t in _CONJUNCTION_WORDS)
        )
        return entity_count + clause_signals

    needs_ner_for_complexity = use_community_context and query_complexity_threshold > 0
    needs_ner_for_concentration = concentration_threshold is not None and not use_enhanced
    if needs_ner_for_complexity or needs_ner_for_concentration:
        from chonk.ner import SpacyMatcher
        _complexity_matcher = SpacyMatcher(model=SPACY_MODEL, strip_numeric=True)
        print("Loaded SpacyMatcher for query complexity / concentration gating.")

    # Run schema migrations in write mode before opening read-only.
    # ALTER TABLE is idempotent; this is a no-op when columns already exist.
    with Store(db_path, embedding_dim=EMBED_DIM) as _mig:
        if auto_domain_filter:
            _register_fang_domains(_mig)

    _adf_fn = None
    if auto_domain_filter:
        import openai as _adf_oai
        _adf_client = _adf_oai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=30.0)
        _adf_model = getattr(args, "gen_model", GEN_MODEL)
        _adf_fn = _build_domain_filter_fn(_adf_client, _adf_model)
        print(f"[ADF] Automated domain filtering enabled (model={_adf_model})")

    work_items: list[dict] = []
    with Store(db_path, embedding_dim=EMBED_DIM, read_only=True) as store:
        enhanced_search = _build_enhanced_search(
            store, db_path, use_ner_x=use_ner_x, embed_model=embed_model,
            entity_ref_expansion=use_entity_ref_expansion,
            entity_ref_expansion_per_k=entity_ref_expansion_per_k,
            entity_ref_expansion_min_sim=entity_ref_expansion_min_sim,
            use_cluster=use_cluster,
            lane_entity_min_sim=lane_entity_min_sim,
            namespaces=namespaces,
            domain_ids=domain_ids,
            community_index=community_index,
            context_graph_expansion=use_context_graph,
            context_graph_min_weight=context_graph_min_weight,
            context_graph_top_k=context_graph_top_k,
        ) if use_enhanced else None

        # Build concentration fallback search (entity ref expansion) if gating is enabled
        # and entity_ref_expansion is not already always-on.
        _conc_search = None
        if concentration_threshold is not None and not use_entity_ref_expansion:
            _conc_search = _build_enhanced_search(
                store, db_path, use_ner_x=False, embed_model=embed_model,
                entity_ref_expansion=True,
                entity_ref_expansion_per_k=entity_ref_expansion_per_k,
                entity_ref_expansion_min_sim=entity_ref_expansion_min_sim,
                use_cluster=False,
                lane_entity_min_sim=lane_entity_min_sim,
                namespaces=namespaces,
                domain_ids=domain_ids,
            )

        # Preload embeddings + chunk metadata into RAM (eliminates per-query DuckDB round-trips)
        store.vector.preload_embeddings()
        print(f"  Embeddings preloaded ({store.vector.count()} chunks).", flush=True)
        if enhanced_search is not None:
            enhanced_search.preload_chunk_cache()
            print("  Chunk cache preloaded.", flush=True)
        if _conc_search is not None:
            _conc_search.preload_chunk_cache()

        # Load pre-computed NER entity embeddings from cache (built by prime-cache)
        _precomputed_entity_vecs: dict[str, np.ndarray] | None = None
        _precomputed_question_entities: list[list[str]] | None = None
        _needs_entity_embed = (
            use_enhanced
            and enhanced_search is not None
            and (use_ner_x or use_entity_ref_expansion)
        )
        if _needs_entity_embed:
            print(f"  Loading cached entity embeddings from {_ent_vecs_cache_path.name}", flush=True)
            _npz = np.load(str(_ent_vecs_cache_path))
            _ent_strings = list(_npz["strings"])
            _ent_matrix  = _npz["vecs"]
            _precomputed_entity_vecs = {s: _ent_matrix[i] for i, s in enumerate(_ent_strings)}
            _q_entities_by_id: dict[str, list[str]] = json.loads(_ent_ents_cache_path.read_text())
            _precomputed_question_entities = [
                _q_entities_by_id.get(q.get("id", f"q{i}"), []) for i, q in pending
            ]

        # Multi-step: decompose each question into sub-queries and pre-embed them
        _sub_queries_by_idx: dict[int, list[str]] = {}  # j -> [sub_q1, sub_q2, ...]
        _sub_vecs_by_idx:    dict[int, list] = {}        # j -> [vec1, vec2, ...]
        if use_multi_step:
            import openai as _oai
            _decomp_model = args.gen_model
            if args.gen_provider == "together":
                _decomp_client = _oai.OpenAI(
                    api_key=os.environ["TOGETHER_API_KEY"], base_url=TOGETHER_BASE_URL, timeout=60.0
                )
            else:
                _decomp_client = _oai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)
            _ms_embed = SentenceTransformer(EMBED_MODEL)
            print(f"[multi-step] Decomposing {len(pending)} questions with {_decomp_model}...", flush=True)
            for j, (i, q) in enumerate(pending):
                subs = _decompose_question(
                    q["question"], q.get("question_type", ""), _decomp_client, _decomp_model
                )
                _sub_queries_by_idx[j] = subs
                vecs = _ms_embed.encode(subs, show_progress_bar=False, normalize_embeddings=True)
                _sub_vecs_by_idx[j] = list(vecs)
                if (j + 1) % 50 == 0 or (j + 1) == len(pending):
                    print(f"  Decomposed {j+1}/{len(pending)}", flush=True)
            del _ms_embed

        # Phase 1: vector search for all questions (sequential, DuckDB serialized)
        # map_reduce_global uses global retrieval (community summary cosine search);
        # the map-reduce step happens later in context assembly.
        _retrieval_mode = "global" if search_mode == "map_reduce_global" else search_mode
        _capture_trace = search_mode == "graph_first" and use_enhanced

        _all_hits: list[tuple] = []
        _all_expansion_stats: list = []
        _all_traces: list = []
        for j, (i, q) in enumerate(pending):
            _q_entities = _precomputed_question_entities[j] if _precomputed_question_entities is not None else None
            _bm25_query_text = q["question"] if getattr(args, "bm25", False) else None
            # ADF needs query_text to trigger domain routing even when BM25 is off
            _eff_query_text = q["question"] if _adf_fn else _bm25_query_text
            # When ADF is active, domain_ids must be None so EnhancedSearch resolves per-query
            _static_domain_ids = None if _adf_fn else domain_ids
            _trace = None
            if use_multi_step and j in _sub_vecs_by_idx:
                # Retrieve for each sub-query, merge by best score per chunk_id
                _merged: dict[str, tuple] = {}
                for sub_vec in _sub_vecs_by_idx[j]:
                    if use_enhanced and enhanced_search is not None:
                        _sub_scored = enhanced_search.search(
                            sub_vec, k=fetch_k, query_text=_eff_query_text,
                            query_entities=_q_entities,
                            precomputed_entity_vecs=_precomputed_entity_vecs,
                            mode=_retrieval_mode,
                            namespaces=namespaces,
                            domain_ids=_static_domain_ids,
                            domain_filter_llm_fn=_adf_fn,
                        )
                        _sub_hits = [(sc.chunk_id, sc.score, sc.chunk) for sc in _sub_scored]
                    else:
                        _q_domain_ids = (
                            enhanced_search._select_domains(q["question"], _adf_fn)
                            if _adf_fn and enhanced_search else _static_domain_ids
                        )
                        _sub_hits = store.vector.search(
                            sub_vec, limit=fetch_k,
                            query_text=_bm25_query_text,
                            include_breadcrumbs=False,
                            namespaces=namespaces,
                            domain_ids=_q_domain_ids,
                        )
                    for cid, sc, chunk in _sub_hits:
                        if cid not in _merged or sc > _merged[cid][1]:
                            _merged[cid] = (cid, sc, chunk)
                hits = sorted(_merged.values(), key=lambda x: -x[1])[:fetch_k]
                expansion_stats = enhanced_search.last_expansion_stats if use_enhanced and enhanced_search else None
            elif use_enhanced and enhanced_search is not None:
                if _capture_trace:
                    scored, _trace = enhanced_search.search(
                        q_vecs[j], k=fetch_k, query_text=_eff_query_text,
                        query_entities=_q_entities,
                        precomputed_entity_vecs=_precomputed_entity_vecs,
                        mode=_retrieval_mode,
                        namespaces=namespaces,
                        domain_ids=_static_domain_ids,
                        domain_filter_llm_fn=_adf_fn,
                        return_trace=True,
                    )
                else:
                    scored = enhanced_search.search(
                        q_vecs[j], k=fetch_k, query_text=_eff_query_text,
                        query_entities=_q_entities,
                        precomputed_entity_vecs=_precomputed_entity_vecs,
                        mode=_retrieval_mode,
                        namespaces=namespaces,
                        domain_ids=_static_domain_ids,
                        domain_filter_llm_fn=_adf_fn,
                    )
                    _trace = None
                hits   = [(sc.chunk_id, sc.score, sc.chunk) for sc in scored]
                expansion_stats = enhanced_search.last_expansion_stats
            else:
                _q_domain_ids = (
                    enhanced_search._select_domains(q["question"], _adf_fn)
                    if _adf_fn and enhanced_search else _static_domain_ids
                )
                hits = store.vector.search(
                    q_vecs[j], limit=fetch_k,
                    query_text=_bm25_query_text,
                    include_breadcrumbs=False,
                    namespaces=namespaces,
                    domain_ids=_q_domain_ids,
                )
                expansion_stats = None

            if _conc_search is not None:
                src_counts: dict[str, int] = defaultdict(int)
                for _, _, _chunk in hits:
                    src_counts[_chunk.document_name] += 1
                _k_hits = len(hits)
                if _k_hits > 0:
                    _max_frac = max(src_counts.values()) / _k_hits
                    if _max_frac >= concentration_threshold:
                        scored_conc = _conc_search.search(
                            q_vecs[j], k=fetch_k, query_text=_eff_query_text,
                            query_entities=_q_entities,
                            precomputed_entity_vecs=_precomputed_entity_vecs,
                            namespaces=namespaces,
                            domain_ids=_static_domain_ids,
                            domain_filter_llm_fn=_adf_fn,
                        )
                        hits = [(sc.chunk_id, sc.score, sc.chunk) for sc in scored_conc]
                        expansion_stats = _conc_search.last_expansion_stats

            _all_hits.append(hits)
            _all_expansion_stats.append(expansion_stats)
            _all_traces.append(_trace)
            if (j + 1) % 100 == 0 or (j + 1) == len(pending):
                print(f"  Vector search {j+1}/{len(pending)}", flush=True)

        # Phase 2: batch reranking across all questions (GPU fully utilized)
        _rerank_ckpt_path = None
        if use_rerank and reranker is not None:
            print(f"  Batch reranking {len(_all_hits)} questions...", flush=True)
            # Move embedder off GPU/MPS so reranker has full VRAM for large batches
            try:
                import gc

                import torch
                if embed_model is not None:
                    del embed_model
                    embed_model = None
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # FFN intermediate dominates peak activation: B × seq_len × 4096 × 4 bytes
                    # Use 75% of free VRAM, accounting for other in-flight tensors
                    _free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
                    # Empirical: ~20 MiB/sample peak activation (model weights + FFN + attention)
                    # at seq_len=512 on XLM-RoBERTa-large. Scale by actual seq_len.
                    _seq_len = getattr(reranker, "max_length", 512)
                    _mib_per_sample = 20.0 * _seq_len / 512
                    _rerank_batch_size = max(16, min(512, int(_free_mib * 0.55 / _mib_per_sample) // 16 * 16))
                    print(f"  CUDA: {_free_mib} MiB free, computed batch_size={_rerank_batch_size}.", flush=True)
                else:
                    _rerank_batch_size = 64
                    print("  CPU reranking, batch_size=64.", flush=True)
            except Exception:
                _rerank_batch_size = 64
            _RERANK_CHUNK = args.rerank_chunk
            import hashlib as _hl_rr
            _rerank_key_str = json.dumps({
                "db": str(db_path),
                "fetch_k": fetch_k,
                "top_k": top_k,
                "enhanced": use_enhanced,
                "vanilla": use_vanilla,
                "lane_sim": lane_entity_min_sim,
                "cluster": use_cluster,
                "entity_ref": use_entity_ref_expansion,
                "search_mode": search_mode,
                "ner_x": use_ner_x,
                "rerank_provider": rerank_provider,
                "reranker": (RERANK_MODEL_TOGETHER if rerank_provider == "together"
                             else RERANK_MODEL_COHERE if rerank_provider == "cohere"
                             else RERANK_MODEL),
            }, sort_keys=True)
            _rerank_cache_key = _hl_rr.md5(_rerank_key_str.encode()).hexdigest()[:16]
            _rerank_ckpt_path = results_dir / f"_rerank_ckpt_{_rerank_cache_key}.json"
            # Load existing checkpoint: qid -> [chunk_id, ...]
            _rerank_ckpt: dict[str, list[str]] = {}
            if _rerank_ckpt_path.exists():
                try:
                    _rerank_ckpt = json.loads(_rerank_ckpt_path.read_text())
                    print(f"  Rerank checkpoint: {len(_rerank_ckpt)} questions already done.", flush=True)
                except Exception:
                    _rerank_ckpt = {}
            _reranked_hits: list = [None] * len(_all_hits)
            # Build per-question chunk_id->hit lookup for checkpoint restoration
            _hit_by_cid: list[dict] = [
                {cid: (cid, sc, chunk) for cid, sc, chunk in _all_hits[j]}
                for j in range(len(_all_hits))
            ]
            # Restore already-checkpointed questions
            _pending_rerank_indices = []
            for j, (i, q) in enumerate(pending):
                qid = q.get("id", f"q{i}")
                if qid in _rerank_ckpt:
                    ordered = [_hit_by_cid[j][cid] for cid in _rerank_ckpt[qid] if cid in _hit_by_cid[j]]
                    _reranked_hits[j] = ordered
                else:
                    _pending_rerank_indices.append(j)
            if len(_pending_rerank_indices) < len(_all_hits):
                print(f"  Restored {len(_all_hits) - len(_pending_rerank_indices)} from rerank checkpoint.", flush=True)
            for _ci, _chunk_start in enumerate(range(0, len(_pending_rerank_indices), _RERANK_CHUNK)):
                _chunk_idx = _pending_rerank_indices[_chunk_start:_chunk_start + _RERANK_CHUNK]
                _chunk_end_display = _chunk_start + len(_chunk_idx)
                _pair_offsets = [0]
                _chunk_pairs = []
                for j in _chunk_idx:
                    q = pending[j][1]
                    pairs = [(q["question"], chunk.content) for _, _, chunk in _all_hits[j]]
                    _chunk_pairs.extend(pairs)
                    _pair_offsets.append(len(_chunk_pairs))
                _chunk_scores = reranker.predict(_chunk_pairs, batch_size=_rerank_batch_size, show_progress_bar=False)
                for _ji, j in enumerate(_chunk_idx):
                    scores = _chunk_scores[_pair_offsets[_ji]:_pair_offsets[_ji + 1]]
                    ranked = sorted(zip(scores, _all_hits[j]), key=lambda x: x[0], reverse=True)[:top_k]
                    _reranked_hits[j] = [h for _, h in ranked]
                    qid = pending[j][1].get("id", f"q{pending[j][0]}")
                    _rerank_ckpt[qid] = [h[0] for h in _reranked_hits[j]]
                _rerank_ckpt_path.write_text(json.dumps(_rerank_ckpt))
                print(f"  Reranked {_chunk_end_display}/{len(_pending_rerank_indices)} pending", flush=True)
            _all_hits = _reranked_hits
            print("  Reranking complete.", flush=True)

        for j, (i, q) in enumerate(pending):
            import dataclasses as _dc
            qid  = q.get("id", f"q{i}")
            hits = _all_hits[j]
            expansion_stats = _all_expansion_stats[j]
            _trace_obj = _all_traces[j] if j < len(_all_traces) else None
            _trace_dict = _dc.asdict(_trace_obj) if _trace_obj is not None else None

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
            elif not use_rerank:
                hits = hits[:top_k]
            if redundancy_threshold is not None:
                hits = _prune_redundant(hits, store.vector._conn, redundancy_threshold)
            chunk_texts = [chunk.content or "" for _, _, chunk in hits]

            # ── Enhancement #3: Query Complexity Routing ──────────────────────
            _inject_community = community_index is not None
            if _inject_community and query_complexity_threshold > 0:
                _score = _query_complexity(q["question"])
                if _score < query_complexity_threshold:
                    _inject_community = False

            # Collect unique community topic labels for retrieved chunks
            community_header = ""
            if _inject_community:
                labels = []
                seen_cids: set = set()
                for cid, _, _ in hits:
                    comm_id = community_index.community_id(cid)
                    if comm_id is not None and comm_id not in seen_cids:
                        seen_cids.add(comm_id)
                        lbl = community_index.topic_label(cid, min_coherence=community_min_coherence)
                        if lbl:
                            labels.append(lbl)
                if labels:
                    community_header = "Topic context: " + "; ".join(dict.fromkeys(labels)) + "\n\n"

            def _fmt_chunk(cid, chunk):
                text = chunk.content or ""
                if use_breadcrumb_context and chunk.breadcrumb:
                    text = f"{_format_breadcrumb(chunk.breadcrumb, style=breadcrumb_style)}\n\n{text}"
                return text

            if search_mode == "graph_first" and enhanced_search is not None:
                _ctx = enhanced_search.assemble_graph_context(
                    hits, query_text=_bm25_query_text
                )
            elif search_mode == "map_reduce_global" and enhanced_search is not None and _map_reduce_llm_fn is not None:
                _ctx = enhanced_search.map_reduce_global_context(
                    hits, q["question"], llm_fn=_map_reduce_llm_fn,
                    concurrency=getattr(args, "concurrency", 4),
                )
            else:
                _ctx = community_header + "\n\n".join(
                    _fmt_chunk(cid, chunk) for cid, _, chunk in hits
                )

            wi = {
                "_slot":          len(work_items),
                "qid":            qid,
                "question":       q["question"],
                "source":         q.get("source", q.get("subset", "?")),
                "qtype":          q.get("question_type", "?"),
                "context":        _ctx,
                "chunk_ids":      [cid for cid, _, _ in hits],
                "scores":         [float(sc) for _, sc, _ in hits],
                "chunk_texts":    chunk_texts,
                "expansion_stats": expansion_stats,
                "evidence":       q.get("evidence", []),
                "gold":           str(q.get("answer", "")),
                "sub_queries":    _sub_queries_by_idx.get(j),
                "retrieval_trace": _trace_dict,
            }
            work_items.append(wi)

            _upsert_results_to_db(run_db, [{
                "id": wi["qid"], "question": wi["question"], "source": wi["source"],
                "question_type": wi["qtype"], "context": wi["context"],
                "evidence": wi["evidence"], "generated_answer": None,
                "gold_answer": wi["gold"], "retrieved_chunks": wi["chunk_ids"],
                "retrieved_scores": wi["scores"],
                "entity_ref_expansion": wi.get("expansion_stats"),
                "entity_ref_retry": None,
                "retrieval_trace": _trace_dict,
            }])
            if (j + 1) % 100 == 0 or (j + 1) == len(pending):
                pct = 100 * (j + 1) // len(pending)
                print(f"  Generated {j+1}/{len(pending)} ({pct}%)", flush=True)

    # Prepend pre-built work_items recovered from DB (retrieval already done on prior run)
    if _db_pending_items:
        for _it in _db_pending_items:
            _it["_slot"] = len(work_items)
            work_items.append(_it)
        print(f"  Added {len(_db_pending_items)} DB-recovered items; total work_items={len(work_items)}", flush=True)

    # Build one client per endpoint (round-robin for parallelism across multiple dedicated endpoints)
    endpoint_ids: list[str] = getattr(args, "endpoint_ids", None) or [args.gen_model]
    if args.gen_provider == "together":
        clients = [
            openai.OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url=TOGETHER_BASE_URL, timeout=120.0)
            for _ in endpoint_ids
        ]
    elif args.gen_provider == "anthropic":
        clients = [
            openai.OpenAI(api_key=os.environ["ANTHROPIC_API_KEY"], base_url=ANTHROPIC_BASE_URL,
                          default_headers={"anthropic-version": "2023-06-01"}, timeout=120.0)
            for _ in endpoint_ids
        ]
    else:
        clients = [openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)]
        endpoint_ids = [args.gen_model]

    n_endpoints = len(clients)
    print(f"Generating answers with {args.concurrency} parallel workers across {n_endpoints} endpoint(s)...")

    # Build dedicated SRR client (may differ from gen client)
    _srr_provider = getattr(args, "srr_provider", None) or args.gen_provider
    _srr_model    = getattr(args, "srr_model", None) or args.gen_model
    if (use_srr or use_sr) and (_srr_provider != args.gen_provider or _srr_model != args.gen_model):
        if _srr_provider == "together":
            srr_client = openai.OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url=TOGETHER_BASE_URL, timeout=120.0)
        elif _srr_provider == "anthropic":
            srr_client = openai.OpenAI(api_key=os.environ["ANTHROPIC_API_KEY"], base_url=ANTHROPIC_BASE_URL,
                                       default_headers={"anthropic-version": "2023-06-01"}, timeout=120.0)
        else:
            srr_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)
        print(f"SRR client: {_srr_provider}/{_srr_model}")
    else:
        srr_client = None  # use per-slot gen client

    retry_ner_fn = None
    if use_entity_ref_retry:
        from chonk.ner import SpacyMatcher
        _retry_matcher = SpacyMatcher(model=SPACY_MODEL, strip_numeric=True)
        retry_ner_fn = lambda text: [m.display_name for m in _retry_matcher.match(text)]

    new_results: list[dict] = []
    ckpt_lock   = threading.Lock()
    done_count  = [len(done_ids)]
    ckpt_counter = [0]

    def _process(item: dict) -> dict:
        slot   = item["_slot"] % n_endpoints
        client = clients[slot]
        model  = endpoint_ids[slot]

        answer: str = ""
        srr_stats: dict | None = None
        if use_srr or use_sr:
            _sc = srr_client if srr_client is not None else client
            _sm = _srr_model
            context = item["context"]
            srr_out = {"answer": "", "key_claims": [], "evidence_used": []}
            for attempt in range(3):
                try:
                    srr_out = _generate_srr(item["question"], context, _sc, _sm,
                                            temperature=gen_temperature)
                    break
                except Exception as exc:
                    if "insufficient_quota" in str(exc):
                        raise
                    if "429" in str(exc) or "rate_limit" in str(exc).lower():
                        import re as _re
                        _m = _re.search(r"try again in (\d+(?:\.\d+)?)s", str(exc), _re.IGNORECASE)
                        time.sleep(float(_m.group(1)) + 1 if _m else 60)
                        continue
                    if attempt == 2:
                        srr_out = {"answer": f"[ERROR: {exc}]", "key_claims": [], "evidence_used": []}
                    else:
                        time.sleep(2 ** attempt)

            # Evidence-compliance check (skipped for --sr): if no evidence cited, reprompt once
            _evidence_reprompt_done = False
            if use_srr and not use_sr and not srr_out["evidence_used"] and srr_out["key_claims"]:
                claims_text = "\n".join(
                    f"{i + 1}. {c}" for i, c in enumerate(srr_out["key_claims"])
                )
                hint = _SRR_EVIDENCE_HINT.format(claims=claims_text)
                user_content = f"Context:\n{item['context']}\n\nQuestion: {item['question']}\n\n{hint}"
                try:
                    import json as _json2
                    resp = _sc.chat.completions.create(
                        model=_sm,
                        messages=[
                            {"role": "system", "content": _SRR_GEN_SYSTEM + (_SRR_CLAUDE_SUFFIX if "claude" in _sm.lower() else "")},
                            {"role": "user",   "content": user_content},
                        ],
                        temperature=gen_temperature,
                        max_tokens=700,
                    )
                    raw = resp.choices[0].message.content.strip()
                    if raw.startswith("```"):
                        raw = "\n".join(raw.splitlines()[1:])
                        if raw.endswith("```"):
                            raw = raw[:-3].strip()
                    obj = _json2.loads(raw)
                    if isinstance(obj.get("answer"), str):
                        srr_out = {
                            "answer":       obj["answer"],
                            "key_claims":   [x for x in obj.get("key_claims", []) if isinstance(x, str)],
                            "evidence_used": [x for x in obj.get("evidence_used", []) if isinstance(x, str)],
                        }
                        _evidence_reprompt_done = True
                except Exception:
                    pass

            answer = srr_out["answer"]
            srr_stats = {
                "key_claims":        srr_out["key_claims"],
                "evidence_used":     srr_out["evidence_used"],
                "evidence_reprompt": _evidence_reprompt_done,
            }
        else:
            for attempt in range(3):
                try:
                    answer = _generate(item["question"], item["context"], client, model,
                                       temperature=gen_temperature, structured=use_structured_gen,
                                       vanilla=use_vanilla)
                    break
                except Exception as exc:
                    if "insufficient_quota" in str(exc):
                        raise
                    if "429" in str(exc) or "rate_limit" in str(exc).lower():
                        import re as _re
                        _m = _re.search(r"try again in (\d+(?:\.\d+)?)s", str(exc), _re.IGNORECASE)
                        time.sleep(float(_m.group(1)) + 1 if _m else 60)
                        continue
                    if attempt == 2:
                        answer = f"[ERROR: {exc}]"
                    else:
                        time.sleep(2 ** attempt)

        retry_stats: dict | None = None
        if use_entity_ref_retry and retry_ner_fn is not None:
            q_entities = retry_ner_fn(item["question"])
            uncovered: list[str] = []
            if q_entities:
                ent_vecs = embed_model.encode(q_entities, normalize_embeddings=True, show_progress_bar=False)
                ans_vec  = embed_model.encode([answer], normalize_embeddings=True, show_progress_bar=False)[0]
                sims     = ent_vecs @ ans_vec  # (n_entities,)
                uncovered = [e for e, s in zip(q_entities, sims) if s < 0.3]
            if uncovered:
                hint = (
                    f"Your answer does not appear to address the following key concepts from the question: "
                    f"{', '.join(uncovered)}. "
                    f"Please re-read the context and provide an answer that directly relates to these concepts."
                )
                retry_answer = answer
                for attempt in range(3):
                    try:
                        retry_answer = _generate(item["question"], item["context"], client, model,
                                                 temperature=gen_temperature, retry_hint=hint,
                                                 structured=use_structured_gen, vanilla=use_vanilla)
                        break
                    except Exception:
                        if attempt == 2:
                            retry_answer = answer
                        else:
                            time.sleep(2 ** attempt)
                retry_stats = {
                    "invoked": True,
                    "uncovered_entities": uncovered,
                }
                answer = retry_answer

        result = {
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
        if item.get("expansion_stats") is not None:
            result["entity_ref_expansion"] = item["expansion_stats"]
        if retry_stats is not None:
            result["entity_ref_retry"] = retry_stats
        if srr_stats is not None:
            result["srr"] = srr_stats
        return result

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
                # Checkpoint every 100 new completions (independent of resumed done_ids)
                ckpt_counter[0] += 1
                if ckpt_counter[0] % 100 == 0:
                    batch = new_results[-100:]
                    with open(ckpt_f, "a") as f:
                        for r in batch:
                            f.write(json.dumps(r) + "\n")
                    _upsert_results_to_db(run_db, batch)
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

    _write_results_to_db(run_db, all_results)
    _run_completed[0] = True
    print(f"\nComplete: {len(all_results)} results → {results_f} + {run_db}")


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
    data_dir    = out_dir / "data"
    results_dir = out_dir / "results"
    repo_dir    = out_dir / "GraphRAG-Benchmark"
    run_name    = getattr(args, "run_name", "contextual")
    results_f   = results_dir / f"{run_name}.jsonl"
    ckpt_f      = results_dir / f"bench_eval_ckpt_{run_name}.jsonl"
    out_f       = results_dir / f"bench_eval_{run_name}.json"

    run_db = _run_db_path(data_dir, run_name)
    if run_db.exists():
        _init_run_db(run_db)  # migrate schema (ALTER TABLE is idempotent)
        records = _read_results_from_db(run_db)
    elif results_f.exists():
        records = [json.loads(line) for line in open(results_f)]
        _init_run_db(run_db)
        _write_results_to_db(run_db, records)
    else:
        print(f"No results found: {results_f}. Run 'run' first.")
        return
    if not repo_dir.exists():
        print(f"Benchmark repo not found at {repo_dir}. Run 'download' first.")
        return
    question_ids_file = getattr(args, 'question_ids', None)
    if question_ids_file:
        with open(question_ids_file) as _f:
            allowed_ids = set(json.load(_f))
        records = [r for r in records if r["id"] in allowed_ids]
    if args.limit:
        records = records[:args.limit]

    # Load gold schemas: authoritative ground truth + typed answer schemas
    _gold_schemas: dict[str, str] = {}
    _answer_schemas: dict[str, dict] = {}
    for _schema_f in sorted(data_dir.glob("*_gold_schemas.jsonl")):
        for _line in open(_schema_f):
            _s = json.loads(_line)
            if _s.get("gold_answer"):
                _gold_schemas[_s["id"]] = _s["gold_answer"]
            if _s.get("answer_schema"):
                _answer_schemas[_s["id"]] = _s["answer_schema"]

    # Convert to benchmark format
    bench_records = [{
        "id":            r["id"],
        "question":      r["question"],
        "question_type": r["question_type"],
        "generated_answer": r["generated_answer"],
        "ground_truth":  (r.get("gold_answer") or _gold_schemas.get(r["id"])
                          or r.get("ground_truth", "")),
        "context":       [r["context"]],
        "answer_schema": _answer_schemas.get(r["id"]),
        "srr":           r.get("srr"),
    } for r in records]

    # Load checkpoint
    done: dict[str, dict] = {}
    if ckpt_f.exists():
        for line in open(ckpt_f):
            item = json.loads(line)
            done[item["id"]] = item
        print(f"Resuming bench-eval: {len(done)} already done")
        # Flush checkpoint items to DB immediately so they're visible in report
        if done and run_db.exists():
            con = _connect_with_retry(run_db)
            for item in done.values():
                con.execute("DELETE FROM eval_scores WHERE id = ?", [item.get("id")])
                con.execute(
                    "INSERT INTO eval_scores VALUES (?,?,?,?,?,?,?,?,?)",
                    [item.get("id"), item.get("question_type"),
                     item.get("answer_correctness"), item.get("rouge_score"),
                     item.get("coverage_score"), item.get("faithfulness"),
                     item.get("nan_reason"), item.get("decomposed_score"),
                     item.get("decomposed_detail")],
                )
            con.close()
            print(f"  Flushed {len(done)} checkpoint scores to DB")

    pending = [r for r in bench_records if r["id"] not in done]
    print(f"Pending: {len(pending)} samples")

    if not pending:
        print("All samples already evaluated.")
    else:
        # Add benchmark repo to path
        sys.path.insert(0, str(repo_dir))
        import httpx
        from Evaluation.metrics import (
            compute_answer_correctness,
            compute_coverage_score,
            compute_faithfulness_score,
            compute_rouge_score,
        )
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        judge_provider = getattr(args, "judge_provider", "openai")
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=20),
        )
        _judge_kwargs = dict(
            model=args.judge,
            temperature=0.0,
            top_p=1,
            seed=42,
            presence_penalty=0,
            frequency_penalty=0,
            max_retries=0,
            timeout=30,
            http_async_client=_http_client,
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
        from langchain_core.embeddings import Embeddings as LCEmbeddings
        from sentence_transformers import SentenceTransformer

        gt_cache_f = out_dir / "data" / "gt_embeddings.npy"
        gt_id_f    = out_dir / "data" / "gt_embedding_ids.json"

        embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")

        # Ground truth embeddings — cached by sorted ID set so any run subset
        # or reordering reuses the same cache file via ID lookup.
        _gt_by_id = {r["id"]: r["ground_truth"] for r in bench_records}
        _gt_ids_sorted = sorted(_gt_by_id.keys())

        _gt_cache_vecs = None
        if gt_cache_f.exists() and gt_id_f.exists():
            _cached_gt_ids = json.loads(gt_id_f.read_text())
            # Hit if cached IDs are a superset of what we need (common in full→grid direction)
            if set(_gt_ids_sorted).issubset(set(_cached_gt_ids)):
                print("Loading cached ground-truth embeddings...")
                _all_gt_vecs = np.load(str(gt_cache_f))
                _cached_id_idx = {qid: i for i, qid in enumerate(_cached_gt_ids)}
                _gt_cache_vecs = np.stack([_all_gt_vecs[_cached_id_idx[qid]] for qid in _gt_ids_sorted])

        if _gt_cache_vecs is None:
            _gt_texts_sorted = [_gt_by_id[qid] or "" for qid in _gt_ids_sorted]
            print(f"Encoding {len(_gt_texts_sorted)} ground-truth texts (batched)...")
            _gt_cache_vecs = embed_model.encode(
                _gt_texts_sorted, batch_size=32, normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            # Only expand cache if this run has more questions than what's cached
            if not gt_cache_f.exists() or len(_gt_ids_sorted) > len(json.loads(gt_id_f.read_text()) if gt_id_f.exists() else []):
                np.save(str(gt_cache_f), _gt_cache_vecs)
                gt_id_f.write_text(json.dumps(_gt_ids_sorted), encoding="utf-8")
                print(f"  Cached → {gt_cache_f}")

        # Align to bench_records order
        _gt_sorted_idx = {qid: i for i, qid in enumerate(_gt_ids_sorted)}
        gt_ids  = [r["id"] for r in bench_records]
        gt_vecs = np.stack([_gt_cache_vecs[_gt_sorted_idx[rid]] for rid in gt_ids])

        # Answer embeddings (per run — cached)
        ans_cache_f = out_dir / "data" / f"ans_embeddings_{run_name}.npy"
        ans_id_f    = out_dir / "data" / f"ans_embedding_ids_{run_name}.json"
        all_ans_ids   = [r["id"] for r in bench_records]
        all_ans_texts = [r["generated_answer"] or "" for r in bench_records]
        if ans_cache_f.exists() and ans_id_f.exists() and json.loads(ans_id_f.read_text()) == all_ans_ids:
            print("Loading cached answer embeddings...")
            all_ans_vecs = np.load(str(ans_cache_f))
        else:
            print(f"Encoding {len(all_ans_texts)} answer texts (batched)...")
            all_ans_vecs = embed_model.encode(
                all_ans_texts, batch_size=32, normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            np.save(str(ans_cache_f), all_ans_vecs)
            ans_id_f.write_text(json.dumps(all_ans_ids), encoding="utf-8")
            print(f"  Cached → {ans_cache_f}")
        # Sync embedder for typed scorer text questions — capture by value before del
        def _typed_embedder(text: str, _model=embed_model):
            return _model.encode(text, normalize_embeddings=True, show_progress_bar=False)

        del embed_model  # free memory

        # Build lookup: text → embedding vector
        emb_lookup: dict[str, np.ndarray] = {}
        for r, vec in zip(bench_records, gt_vecs):
            emb_lookup[r["ground_truth"]] = vec
        for r, vec in zip(bench_records, all_ans_vecs):
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
            "Fact Retrieval":                   ["rouge_score", "answer_correctness"],
            "Complex Reasoning":                ["rouge_score", "answer_correctness"],
            "Contextual Summarize":             ["answer_correctness", "coverage_score"],
            "Creative Generation":              ["answer_correctness", "coverage_score", "faithfulness"],
            # FANG-2026 question types — typed scorer used when answer_schema present
            "Multi-Document Join":              ["typed_score"],
            "Temporal Versioning":              ["typed_score"],
            "Cross-Domain Entity Resolution":   ["typed_score"],
            "Targeted Attribute Lookup":        ["typed_score"],
            "Descriptive Attribute Lookup":     ["typed_score"],
            "Quantitative Synthesis":           ["typed_score"],
            "Absence/Negation":                 ["typed_score"],
        }

        # Import typed scorer
        import importlib.util as _ilu
        _ts_path = data_dir / "score_typed.py"
        _score_one = None
        if _ts_path.exists():
            _ts_spec = _ilu.spec_from_file_location("score_typed", _ts_path)
            _ts_mod = _ilu.module_from_spec(_ts_spec)
            _ts_spec.loader.exec_module(_ts_mod)
            _score_one = _ts_mod.score_one

        semaphore = asyncio.Semaphore(args.concurrency)
        _judge_model = getattr(args, "judge", GEN_MODEL)
        eval_rpm: int = getattr(args, "eval_rpm", None) or _model_rpm_limit(_judge_model)
        nan_limit: int | None = getattr(args, "nan_limit", None)
        _nan_final = [0]  # count of items finalized as NaN (shared across coroutines)

        # Token-bucket throttle: at most eval_rpm tokens per 60s window.
        _rpm_tokens   = [float(eval_rpm)]
        _rpm_last_ts  = [time.monotonic()]
        _rpm_lock     = asyncio.Lock()
        print(f"[eval] RPM limit: {eval_rpm} (judge={_judge_model})", flush=True)

        async def _acquire_rpm_token():
            async with _rpm_lock:
                import asyncio as _aio
                now = time.monotonic()
                elapsed = now - _rpm_last_ts[0]
                _rpm_tokens[0] = min(float(eval_rpm), _rpm_tokens[0] + elapsed * (eval_rpm / 60.0))
                _rpm_last_ts[0] = now
                if _rpm_tokens[0] < 1.0:
                    wait = (1.0 - _rpm_tokens[0]) / (eval_rpm / 60.0)
                    print(f"[eval] RPM throttle: sleeping {wait:.1f}s", flush=True)
                    await _aio.sleep(wait)
                    _rpm_tokens[0] = 0.0
                    _rpm_last_ts[0] = time.monotonic()
                else:
                    _rpm_tokens[0] -= 1.0

        import math as _math
        _REPROMPT_TEMPLATES = {
            "answer_correctness": (
                "You are evaluating whether a generated answer is correct relative to the ground truth.\n"
                "Question: {question}\n"
                "Ground truth: {ground_truth}\n"
                "Generated answer: {generated_answer}\n\n"
                "Score from 0.0 (completely wrong) to 1.0 (perfectly correct).\n"
                'Respond with ONLY valid JSON, exactly: {{"score": <float>}}'
            ),
            "coverage_score": (
                "You are evaluating how well a generated answer covers the key information in the ground truth.\n"
                "Question: {question}\n"
                "Ground truth: {ground_truth}\n"
                "Generated answer: {generated_answer}\n\n"
                "Score from 0.0 (covers nothing) to 1.0 (covers everything).\n"
                'Respond with ONLY valid JSON, exactly: {{"score": <float>}}'
            ),
            "faithfulness": (
                "You are evaluating whether a generated answer is faithful to the provided context (no hallucinations).\n"
                "Question: {question}\n"
                "Generated answer: {generated_answer}\n"
                "Context (first 1000 chars): {context}\n\n"
                "Score from 0.0 (completely hallucinated) to 1.0 (fully grounded in context).\n"
                'Respond with ONLY valid JSON, exactly: {{"score": <float>}}'
            ),
        }

        async def _judge_reprompt(r: dict, nan_metrics: list[str]) -> dict[str, float]:
            """Simplified extraction prompt for metrics that returned malformed JSON."""
            import json as _json
            import re as _re
            recovered: dict[str, float] = {}
            ctx_snippet = (r.get("context") or "")[:1000]
            for metric in nan_metrics:
                tmpl = _REPROMPT_TEMPLATES.get(metric)
                if not tmpl:
                    continue
                prompt = tmpl.format(
                    question=r.get("question", ""),
                    ground_truth=r.get("ground_truth", ""),
                    generated_answer=r.get("generated_answer", ""),
                    context=ctx_snippet,
                )
                for _rp_attempt in range(2):
                    try:
                        resp = await llm.ainvoke(prompt)
                        text = resp.content if hasattr(resp, "content") else str(resp)
                        # Try strict JSON parse first, then regex fallback
                        try:
                            obj = _json.loads(text)
                            score = float(obj.get("score", float("nan")))
                        except Exception:
                            m = _re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
                            score = float(m.group(1)) if m else float("nan")
                        if 0.0 <= score <= 1.0:
                            recovered[metric] = score
                            break
                    except Exception:
                        pass
            return recovered

        async def _eval_one(r: dict, attempt: int = 0) -> dict:
            """Single attempt. Returns result dict, or {'_deferred': True, '_attempt': attempt} on timeout."""
            import asyncio as _aio
            import random as _random
            qtype   = r["question_type"]
            metrics = METRIC_CONFIG.get(qtype, ["answer_correctness"])
            result  = {"id": r["id"], "question_type": qtype}
            _rate_delay = [0.0]
            _caught_rate = [False]

            async with semaphore:
                _t0 = time.monotonic()
                try:
                    tasks = {}
                    if "typed_score" in metrics:
                        schema = r.get("answer_schema")
                        if schema and _score_one is not None:
                            _ts = _score_one(r["generated_answer"] or "", schema, embedder=_typed_embedder, srr_data=r.get("srr"))
                            result["typed_score"] = _ts if _ts == _ts else float("nan")
                        else:
                            # no schema — fall back to LLM judge recorded as answer_correctness
                            tasks["answer_correctness"] = compute_answer_correctness(
                                r["question"], r["generated_answer"], r["ground_truth"], llm, embedding
                            )
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
                    _rpm_weights = {"answer_correctness": 3, "coverage_score": 1, "faithfulness": 1}
                    api_count = sum(_rpm_weights.get(k, 0) for k in tasks)
                    for _ in range(api_count):
                        await _acquire_rpm_token()
                    vals = await asyncio.gather(*tasks.values(), return_exceptions=True)
                    for key, val in zip(tasks.keys(), vals):
                        if isinstance(val, BaseException):
                            print(f"[eval] {r['id']} {key} exception: {type(val).__name__}: {val}", flush=True)
                        result[key] = float(val) if isinstance(val, (int, float)) else float("nan")
                    nan_metrics = [k for k in tasks if k != "rouge_score" and result.get(k) != result.get(k)]
                    if nan_metrics:
                        # Guided judge reprompt: judge already reasoned but returned malformed JSON.
                        # Recover the score with a simplified extraction prompt.
                        recovered = await _judge_reprompt(r, nan_metrics)
                        for key, val in recovered.items():
                            if not _math.isnan(val):
                                result[key] = val
                        nan_metrics = [k for k in nan_metrics if _math.isnan(result.get(k, float("nan")))]
                        if nan_metrics:
                            result["nan_reason"] = "parse"
                            _nan_final[0] += 1
                        else:
                            print(f"[eval] {r['id']} reprompt recovered {list(recovered.keys())}", flush=True)
                    print(f"[eval] {r['id']} done in {time.monotonic()-_t0:.1f}s metrics={list(tasks.keys())}", flush=True)
                    return result
                except Exception as e:
                    is_rate    = "429" in str(e) or "rate_limit" in str(e).lower() or "RateLimit" in type(e).__name__
                    is_timeout = "timeout" in str(e).lower() or "Timeout" in type(e).__name__
                    kind = "rate-limit" if is_rate else "timeout" if is_timeout else "error"
                    print(f"[eval] {r['id']} attempt {attempt+1}/5 {kind}: {e}", flush=True)
                    if is_timeout:
                        # Defer to back of queue — do not retry inline
                        return {"id": r["id"], "question_type": qtype, "_deferred": True, "_attempt": attempt}
                    if is_rate:
                        retry_after = None
                        if hasattr(e, "response") and e.response is not None:
                            retry_after = e.response.headers.get("Retry-After")
                        _rate_delay[0] = float(retry_after) + _random.uniform(0, 1) if retry_after \
                            else 15 * (2 ** attempt) + _random.uniform(0, 1)
                        print(f"[eval] rate-limit backoff {_rate_delay[0]:.1f}s", flush=True)
                        _caught_rate[0] = True
                    else:
                        for key in METRIC_CONFIG.get(qtype, ["answer_correctness"]):
                            result.setdefault(key, float("nan"))
                        result["nan_reason"] = "error"
                        _nan_final[0] += 1
                        return result

            # Rate limit: sleep outside semaphore then recurse with incremented attempt
            if _caught_rate[0]:
                await _aio.sleep(_rate_delay[0])
                if attempt < 4:
                    return await _eval_one(r, attempt + 1)
                for key in METRIC_CONFIG.get(qtype, ["answer_correctness"]):
                    result.setdefault(key, float("nan"))
                result["nan_reason"] = "rate_exhausted"
                _nan_final[0] += 1
            return result

        async def _eval_one_safe(r: dict, attempt: int = 0) -> dict:
            import asyncio as _aio
            try:
                return await _aio.wait_for(_eval_one(r, attempt), timeout=1800)
            except TimeoutError:
                print(f"[eval] {r['id']} outer 1800s timeout — skipping", flush=True)
                qtype = r.get("question_type", "?")
                return {"id": r["id"], "question_type": qtype,
                        "nan_reason": "outer_timeout",
                        **{k: float("nan") for k in METRIC_CONFIG.get(qtype, ["answer_correctness"])}}

        async def _run_all():
            import asyncio as _aio
            _run_all_start = time.monotonic()
            _completed = [0]
            _ckpt_lock = _aio.Lock()
            _deferred: list[tuple[dict, int]] = []

            async def _save_result(result: dict):
                async with _ckpt_lock:
                    done[result["id"]] = result
                    with open(ckpt_f, "a") as f:
                        f.write(json.dumps(result) + "\n")
                    if run_db.exists():
                        con = _connect_with_retry(run_db)
                        con.execute("DELETE FROM eval_scores WHERE id = ?", [result.get("id")])
                        con.execute(
                            "INSERT INTO eval_scores VALUES (?,?,?,?,?,?,?,?,?)",
                            [
                                result.get("id"), result.get("question_type"),
                                result.get("answer_correctness"), result.get("rouge_score"),
                                result.get("coverage_score"), result.get("faithfulness"),
                                result.get("nan_reason"), result.get("decomposed_score"),
                                result.get("decomposed_detail"),
                            ],
                        )
                        con.close()
                    _completed[0] += 1
                    _total_elapsed = time.monotonic() - _run_all_start
                    _qpm = _completed[0] / (_total_elapsed / 60.0) if _total_elapsed > 0 else 0
                    if _completed[0] % 20 == 0:
                        print(f"  {_completed[0]}/{len(pending)} evaluated | elapsed={_total_elapsed:.0f}s | avg={_qpm:.1f}q/min", flush=True)
                        await _check_early_stop()

            async def _eval_and_save(r: dict, attempt: int = 0):
                result = await _eval_one_safe(r, attempt)
                if not isinstance(result, dict):
                    return
                if result.get("_deferred"):
                    next_attempt = result.get("_attempt", attempt) + 1
                    if next_attempt < 5:
                        _deferred.append((r, next_attempt))
                    else:
                        qtype = r.get("question_type", "?")
                        nan_result = {"id": r["id"], "question_type": qtype,
                                     "nan_reason": "timeout_exhausted",
                                     **{k: float("nan") for k in METRIC_CONFIG.get(qtype, ["answer_correctness"])}}
                        await _save_result(nan_result)
                        _nan_final[0] += 1
                    return
                await _save_result(result)

            _es_target = getattr(args, "early_stop_target", None)
            _es_min_n  = getattr(args, "early_stop_min_n", 500)

            async def _check_early_stop():
                import math as _math
                if _es_target is None or len(done) < _es_min_n:
                    return
                _scores = [
                    v["answer_correctness"] for v in done.values()
                    if "answer_correctness" in v
                    and isinstance(v["answer_correctness"], float)
                    and not _math.isnan(v["answer_correctness"])
                ]
                if len(_scores) < _es_min_n:
                    return
                completed = _completed[0]
                _n    = len(_scores)
                _mean = sum(_scores) / _n
                _var  = sum((s - _mean) ** 2 for s in _scores) / _n
                _se   = _math.sqrt(_var / _n) if _var > 0 else 0.0
                _uci  = _mean + 1.645 * _se
                _n_remaining = len(pending) - completed
                _max_possible = (_n * _mean + _n_remaining) / (_n + _n_remaining) if (_n + _n_remaining) > 0 else 0.0
                print(f"  [early-stop] n={_n} mean={_mean:.4f} upper_95={_uci:.4f} max_possible={_max_possible:.4f} target={_es_target:.4f}", flush=True)
                if _max_possible < _es_target:
                    print(f"\n=== EARLY STOP: max_possible {_max_possible:.4f} < target {_es_target:.4f} — mathematically impossible to win ===", flush=True)
                    sys.exit(2)
                if _uci < _es_target:
                    print(f"\n=== EARLY STOP: upper_95 CI {_uci:.4f} < target {_es_target:.4f} — aborting ===", flush=True)
                    sys.exit(2)

            await _aio.gather(*[_eval_and_save(r) for r in pending], return_exceptions=True)

            # Process deferred (timed-out) items in passes — back-of-queue retry
            for _pass in range(5):
                if not _deferred:
                    break
                print(f"\n[eval] Deferred pass {_pass+1}: {len(_deferred)} items — sleeping 30s before retry", flush=True)
                await _aio.sleep(30)
                current = _deferred[:]
                _deferred.clear()
                await _aio.gather(*[_eval_and_save(r, attempt) for r, attempt in current], return_exceptions=True)

            if _deferred:
                # Any still deferred after 5 passes: record as timeout_exhausted
                for r, _ in _deferred:
                    qtype = r.get("question_type", "?")
                    nan_result = {"id": r["id"], "question_type": qtype,
                                 "nan_reason": "timeout_exhausted",
                                 **{k: float("nan") for k in METRIC_CONFIG.get(qtype, ["answer_correctness"])}}
                    await _save_result(nan_result)
                    _nan_final[0] += 1

        asyncio.run(_run_all())

    # Aggregate results by question type
    by_type: dict[str, list] = defaultdict(list)
    for item in done.values():
        by_type[item["question_type"]].append(item)

    aggregated: dict[str, dict] = {}
    for qtype, items in by_type.items():
        agg: dict[str, float] = {}
        for key in ["rouge_score", "answer_correctness", "typed_score", "coverage_score", "faithfulness"]:
            vals = [i[key] for i in items if key in i and not (isinstance(i[key], float) and i[key] != i[key])]
            if vals:
                agg[key] = float(np.nanmean(vals))
        aggregated[qtype] = agg

    params = {
        "judge": args.judge,
        "judge_provider": getattr(args, "judge_provider", "openai"),
        "run_name": run_name,
        "n_evaluated": len(done),
    }
    with open(out_f, "w") as f:
        json.dump({"_params": params, **aggregated}, f, indent=2)

    # Write per-question eval scores to run DB
    if run_db.exists():
        con = _connect_with_retry(run_db)
        con.execute("DELETE FROM eval_scores")
        for item in done.values():
            con.execute(
                "INSERT INTO eval_scores VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    item.get("id"), item.get("question_type"),
                    item.get("answer_correctness"), item.get("rouge_score"),
                    item.get("coverage_score"), item.get("faithfulness"),
                    item.get("nan_reason"), item.get("decomposed_score"),
                    item.get("decomposed_detail"),
                ],
            )
        con.close()

    print(f"\nBench eval complete → {out_f}")
    for qtype, scores in aggregated.items():
        print(f"  {qtype}: " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))


def cmd_prep_nan_reeval(args: argparse.Namespace) -> None:
    """Prepare a run for NaN-only re-evaluation.

    Reads eval_scores from the run DuckDB, writes a checkpoint containing only
    non-NaN results, and deletes the bench_eval JSON.  Running 'eval' afterwards
    will skip the already-good questions and only evaluate the NaN ones.
    """
    import math as _math
    out_dir     = Path(args.out_dir)
    results_dir = out_dir / "results"
    run_name    = args.run_name
    run_db      = _run_db_path(out_dir / "data", run_name)
    ckpt_f      = results_dir / f"bench_eval_ckpt_{run_name}.jsonl"
    bench_eval  = results_dir / f"bench_eval_{run_name}.json"

    if not run_db.exists():
        print(f"Missing DB: {run_db}")
        return

    import duckdb as _ddb
    con = _ddb.connect(str(run_db), read_only=True)
    try:
        rows = con.execute(
            "SELECT id, question_type, answer_correctness, rouge_score, "
            "coverage_score, faithfulness, nan_reason FROM eval_scores"
        ).fetchall()
        cols = ["id", "question_type", "answer_correctness", "rouge_score",
                "coverage_score", "faithfulness", "nan_reason"]
    except Exception:
        rows = con.execute(
            "SELECT id, question_type, answer_correctness, rouge_score, "
            "coverage_score, faithfulness FROM eval_scores"
        ).fetchall()
        cols = ["id", "question_type", "answer_correctness", "rouge_score",
                "coverage_score", "faithfulness"]
    con.close()
    good, nan_ids = [], []
    for row in rows:
        r = dict(zip(cols, row))
        ac = r.get("answer_correctness")
        if ac is None or (isinstance(ac, float) and _math.isnan(ac)):
            nan_ids.append(r["id"])
        else:
            good.append(r)

    if not nan_ids:
        print(f"{run_name}: no NaN entries — nothing to do")
        return

    with open(ckpt_f, "w") as f:
        for r in good:
            f.write(json.dumps({k: v for k, v in r.items() if v is not None}) + "\n")

    if bench_eval.exists():
        bench_eval.unlink()

    print(f"{run_name}: {len(good)} good kept in checkpoint, {len(nan_ids)} NaN queued for re-eval")
    print(f"  Run: python demo/graphrag_bench.py eval --out-dir {out_dir} --run-name {run_name} ...")


def cmd_backfill_db(args: argparse.Namespace) -> None:
    """Reconstruct a run DuckDB from results jsonl + eval checkpoint jsonl.

    Use when gen ran on a different machine and no DuckDB exists, so cmd_score
    returns nan. Reads work/results/{name}.jsonl and bench_eval_ckpt_{name}.jsonl,
    creates work/data/runs/{name}.duckdb with both results and eval_scores tables.
    """
    out_dir      = Path(args.out_dir)
    results_dir  = out_dir / "results"
    run_name     = args.run_name
    results_f    = results_dir / f"{run_name}.jsonl"
    ckpt_f       = results_dir / f"bench_eval_ckpt_{run_name}.jsonl"
    run_db       = _run_db_path(out_dir / "data", run_name)

    if not results_f.exists():
        print(f"Missing: {results_f}")
        return
    if not ckpt_f.exists():
        print(f"Missing: {ckpt_f}")
        return

    records = [json.loads(l) for l in open(results_f)]
    _init_run_db(run_db)
    _write_results_to_db(run_db, records)

    import duckdb as _ddb
    con = _ddb.connect(str(run_db))
    eval_rows = [json.loads(l) for l in open(ckpt_f)]
    for r in eval_rows:
        con.execute("DELETE FROM eval_scores WHERE id = ?", [r["id"]])
        con.execute(
            "INSERT INTO eval_scores VALUES (?,?,?,?,?,?,?,?,?)",
            [r["id"], r.get("question_type"), r.get("answer_correctness"),
             r.get("rouge_score"), r.get("coverage_score"), r.get("faithfulness"),
             r.get("nan_reason"), r.get("decomposed_score"), r.get("decomposed_detail")],
        )
    con.close()
    print(f"Backfilled {len(records)} results + {len(eval_rows)} eval_scores → {run_db}")


def cmd_score(args: argparse.Namespace) -> None:
    """Print the All score (matches report: mean-of-means per subset×type) for a run.

    Outputs a single float (e.g. '0.6850') or 'nan' if the run is incomplete.
    Intended for shell scripts: SCORE=$(python ... score --run-name foo)
    """
    import math as _math
    from collections import defaultdict as _dd
    data_dir = Path(args.out_dir) / "data"
    run_db   = data_dir / "runs" / f"{args.run_name}.duckdb"
    if not run_db.exists():
        print("nan")
        return
    import duckdb as _ddb
    con = _ddb.connect(str(run_db), read_only=True)
    try:
        rows = con.execute(
            "SELECT e.question_type, r.source, e.answer_correctness "
            "FROM eval_scores e JOIN results r ON e.id = r.id"
        ).fetchall()
    except Exception:
        rows = []
    finally:
        con.close()
    # Group by (subset, question_type) — same logic as cmd_bench_report
    by_st: dict[tuple, list] = _dd(list)
    for qtype, source, ac in rows:
        if ac is None or (isinstance(ac, float) and _math.isnan(ac)):
            continue
        subset = "Med" if source and "medical" in str(source).lower() else "Nov"
        by_st[(subset, qtype)].append(float(ac))
    def _m(lst): return sum(lst) / len(lst) if lst else float("nan")  # noqa
    qtypes = ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]
    med_vals = [_m(by_st[(s, qt)]) for s in ["Med"] for qt in qtypes]
    nov_vals = [_m(by_st[(s, qt)]) for s in ["Nov"] for qt in qtypes]
    med = _m([v for v in med_vals if not _math.isnan(v)])
    nov = _m([v for v in nov_vals if not _math.isnan(v)])
    if _math.isnan(med) or _math.isnan(nov):
        print("nan")
    else:
        print(f"{(med + nov) / 2:.4f}")


def cmd_make_order(args: argparse.Namespace) -> None:
    """Generate a stratified question order file for representative full-corpus evaluation.

    Interleaves questions by (subset × question_type) so any prefix of N questions
    has the same stratum distribution as the full set.
    """
    out_dir  = Path(args.out_dir)
    data_dir = out_dir / "data"
    out_file = Path(args.output) if args.output else data_dir / "full_corpus_stratified_order.json"

    # Load questions tagged with their subset
    questions = []
    for subset in SUBSETS:
        f = data_dir / f"{subset}_questions.jsonl"
        if f.exists():
            with open(f) as fh:
                for line in fh:
                    q = json.loads(line)
                    q["_subset"] = subset
                    questions.append(q)

    # Bucket by (subset, question_type)
    buckets: dict[tuple, list] = defaultdict(list)
    for q in questions:
        key = (q.get("_subset", "unknown"), q.get("question_type", "unknown"))
        buckets[key].append(q.get("id", ""))

    # Assign fractional rank = position_within_bucket / bucket_size so that
    # sorting by this rank interleaves buckets proportionally (like a zipper).
    ranked: list[tuple] = []
    for key, ids in sorted(buckets.items()):
        n = len(ids)
        for pos, qid in enumerate(ids):
            ranked.append((pos / n, key, qid))
    ranked.sort(key=lambda x: (x[0], x[1]))
    ordered_ids = [x[2] for x in ranked]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(ordered_ids, f)

    total = len(ordered_ids)
    print(f"Stratified order: {total} questions → {out_file}")
    for key, ids in sorted(buckets.items()):
        print(f"  {key[0]}/{key[1]}: {len(ids)} ({100 * len(ids) / total:.1f}%)")


def cmd_bench_report(args: argparse.Namespace) -> None:
    """Combined matrix: our runs + full leaderboard, scored with benchmark's native metric."""
    out_dir     = Path(args.out_dir)
    data_dir    = out_dir / "data"
    results_dir = out_dir / "results"

    # Discover run DBs first, fall back to legacy bench_eval_*.json
    run_filter:  str | None = getattr(args, "filter", None)
    run_exclude: str | None = getattr(args, "exclude", None)
    run_results: dict[str, dict] = {}
    runs_dir = data_dir / "runs"
    if runs_dir.exists():
        import duckdb as _ddb
        import numpy as _np
        for run_db in sorted(runs_dir.glob("*.duckdb")):
            rn = run_db.stem
            if run_filter and run_filter not in rn:
                continue
            if run_exclude and run_exclude in rn:
                continue
            try:
                con = _ddb.connect(str(run_db), read_only=True)
                try:
                    rows = con.execute(
                        "SELECT e.question_type, r.source, e.answer_correctness, e.nan_reason "
                        "FROM eval_scores e JOIN results r ON e.id = r.id"
                    ).fetchall()
                except Exception:
                    rows = [(qt, src, ac, None) for qt, src, ac in con.execute(
                        "SELECT e.question_type, r.source, e.answer_correctness "
                        "FROM eval_scores e JOIN results r ON e.id = r.id"
                    ).fetchall()]
                total_results = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
                con.close()
            except Exception:
                continue
            if not rows:
                continue
            n_eval = len(rows)
            n_nan = sum(1 for _, _, ac, _ in rows if ac is None or ac != ac)
            from collections import Counter as _Counter
            nan_reasons = _Counter(nr for _, _, ac, nr in rows if (ac is None or ac != ac) and nr)
            # by_subset_type[(source, qtype)] -> [ac, ...]
            by_st: dict[tuple, list] = {}
            for qtype, source, ac, _ in rows:
                subset = "Med" if source and "medical" in source.lower() else "Nov"
                for key in [(subset, qtype), ("All", qtype)]:
                    if key not in by_st:
                        by_st[key] = []
                    if ac is not None and ac == ac:
                        by_st[key].append(ac)
            def _mean(lst): return float(_np.mean(lst)) if lst else float("nan")  # noqa
            scores_st = {k: _mean(v) for k, v in by_st.items()}
            run_results[rn] = {"scores_st": scores_st, "n_eval": n_eval, "n_total": total_results,
                               "n_nan": n_nan, "nan_reasons": dict(nan_reasons)}

    # Legacy bench_eval_*.json — no Med/Nov split, but load All-level scores for comparison
    import json as _json
    QTYPE_MAP = {
        "Fact Retrieval": "Fact", "Complex Reasoning": "Rsn",
        "Contextual Summarize": "Summ", "Creative Generation": "Crea",
    }
    for jf in sorted((results_dir).glob("bench_eval_*.json")):
        rn = jf.stem[len("bench_eval_"):]
        if rn in run_results:
            continue  # already loaded from DuckDB
        if run_filter and run_filter not in rn:
            continue
        if run_exclude and run_exclude in rn:
            continue
        try:
            data = _json.loads(jf.read_text())
        except Exception:
            continue
        params = data.get("_params", {})
        n_eval = params.get("n_evaluated", 0)
        scores_st: dict = {}
        type_scores = []
        for long, short in QTYPE_MAP.items():
            ac = data.get(long, {}).get("answer_correctness")
            if ac is not None:
                scores_st[("All", long)] = ac
                type_scores.append(ac)
        if type_scores:
            import numpy as _np2
            scores_st[("All", "All")] = float(_np2.mean(type_scores))
        if scores_st:
            run_results[rn] = {"scores_st": scores_st, "n_eval": n_eval, "n_total": n_eval, "legacy": True}

    QTYPES = [
        ("Fact Retrieval",       "Fact"),
        ("Complex Reasoning",    "Rsn"),
        ("Contextual Summarize", "Summ"),
        ("Creative Generation",  "Crea"),
    ]
    SUBSETS = ["Med", "Nov", "All"]

    print("\n── Benchmark Eval Results (answer_correctness) ──\n")
    # Header: Run | Med-Fact Med-Rsn Med-Summ Med-Crea Med | Nov-Fact ... Nov | All-Fact ... All
    col_w = 6
    run_w = 52
    hdr_parts = []
    for subset in SUBSETS:
        for _, short in QTYPES:
            hdr_parts.append(f"{subset[:1]}-{short:>4}")
        hdr_parts.append(f"  {subset:>3}")
    header = f"  {'Run':<{run_w}}  " + "  ".join(hdr_parts)
    print(header)
    print("  " + "-" * (len(header) - 2))

    fmt = lambda v: f"{v:.3f}" if v == v else "  — "  # noqa: E731

    def composite(scores_st, subset):
        if subset == "All":
            med = composite(scores_st, "Med")
            nov = composite(scores_st, "Nov")
            valid = [v for v in [med, nov] if v == v]
            return sum(valid) / len(valid) if valid else float("nan")
        vals = [scores_st.get((subset, qt), float("nan")) for qt, _ in QTYPES]
        valid = [v for v in vals if v == v]
        return sum(valid) / len(valid) if valid else float("nan")

    for run_name, info in sorted(run_results.items()):
        st = info.get("scores_st", {})
        n_eval = info.get("n_eval", 0)
        n_total = info.get("n_total", 0)
        n_nan = info.get("n_nan", 0)
        nan_reasons = info.get("nan_reasons", {})
        if nan_reasons:
            reason_str = " " + "+".join(f"{v}{k[0].upper()}" for k, v in sorted(nan_reasons.items()))
        else:
            reason_str = ""
        nan_str = f", {n_nan} NaN{reason_str}" if n_nan else ""
        partial = f" ({n_eval}/{n_total}{nan_str})" if n_total and (n_eval < n_total or n_nan) else (f" ({n_nan} NaN{reason_str})" if n_nan else "")
        parts = []
        for subset in SUBSETS:
            for qt, _ in QTYPES:
                parts.append(f"{fmt(st.get((subset, qt), float('nan'))):>6}")
            parts.append(f"  {fmt(composite(st, subset)):>5}")
        print(f"  {run_name:<{run_w}}  " + "  ".join(parts) + partial)

    # Leaderboard (gpt-4o-mini judge and generator per benchmark paper arXiv:2506.05690)
    # Qwen2.5-14B appendix numbers from paper (Table 7, RAG w/ rerank)
    # Combined med+nov avg per question type is not directly available — showing overall acc
    print("\n── Published Leaderboard (gpt-4o-mini generator + judge) ──\n")
    print(f"  {'Method':<28}  {'Med ACC':>8}  {'Nov ACC':>8}  {'Overall':>8}")
    print("  " + "-" * 58)
    for name, scores in PUBLISHED_BASELINES.items():
        med = f"{scores['med_acc']:.3f}" if scores["med_acc"] is not None else "   —"
        nov = f"{scores['nov_acc']:.3f}" if scores["nov_acc"] is not None else "   —"
        ov  = f"{scores['overall']:.3f}" if scores["overall"] is not None else "   —"
        print(f"  {name:<28}  {med:>8}  {nov:>8}  {ov:>8}")

    print("\n  * Leaderboard: gpt-4o-mini generator + gpt-4o-mini judge (arXiv:2506.05690).")
    print("  * Our bench-eval: gpt-4o-mini generator + gpt-4o-mini judge, answer_correctness metric.")
    print("  * Overall = mean(Med, Nov); Med/Nov = mean of 4 question-type ACC scores.")


# ─────────────────────────────────────────────────────────────────────────────
# Together dedicated endpoint lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def cmd_nan_bias_report(args: argparse.Namespace) -> None:
    """Assess whether NaN questions introduce score bias across runs.

    For each run, identifies which question IDs have NaN answer_correctness.
    Computes:
      1. Pairwise Jaccard overlap of NaN sets (high = same questions failing = less bias)
      2. Per-run NaN difficulty bias: mean score of NaN-in-X questions in all other runs
         vs overall mean — positive delta means skipping those questions inflated the score
      3. MCAR flag: whether missingness appears random or systematic
    """
    import math as _math
    from pathlib import Path as _Path

    import numpy as _np

    runs_dir = _Path(args.out_dir) / "data" / "runs"
    import duckdb as _ddb

    run_filter  = getattr(args, "filter", None)
    run_exclude = getattr(args, "exclude", None)

    # Load per-question answer_correctness for every available run
    # run_data[run_name] = {qid: float or nan}
    run_data: dict[str, dict[str, float]] = {}
    for db in sorted(runs_dir.glob("*.duckdb")):
        if str(db).endswith(".wal"):
            continue
        rn = db.stem
        if run_filter  and run_filter  not in rn: continue
        if run_exclude and run_exclude in rn:      continue
        try:
            con = _ddb.connect(str(db), read_only=True)
            rows = con.execute(
                "SELECT id, answer_correctness FROM eval_scores"
            ).fetchall()
            con.close()
        except Exception:
            continue
        if not rows:
            continue
        run_data[rn] = {qid: (float("nan") if (ac is None or (isinstance(ac, float) and _math.isnan(ac))) else float(ac))
                        for qid, ac in rows}

    if len(run_data) < 2:
        print("Need at least 2 runs with eval_scores. Aborting.")
        return

    run_names = sorted(run_data.keys())
    nan_sets: dict[str, set] = {rn: {q for q, v in run_data[rn].items() if _math.isnan(v)}
                                 for rn in run_names}

    def _short(name: str) -> str:
        return name.replace("nobc_ner_ref_rerank_", "").replace("nobc_", "").replace("_full", "").replace("_grid", "")

    # ── 1. NaN set sizes ──────────────────────────────────────────────────────
    print("\n── NaN set sizes ──\n")
    print(f"  {'Run':<48}  {'NaN':>5}  {'Total':>6}  {'%':>5}")
    print("  " + "-" * 68)
    for rn in run_names:
        total = len(run_data[rn])
        n_nan = len(nan_sets[rn])
        pct   = 100 * n_nan / total if total else 0
        print(f"  {_short(rn):<48}  {n_nan:>5}  {total:>6}  {pct:>4.1f}%")

    # ── 2. Pairwise Jaccard overlap of NaN sets ───────────────────────────────
    print("\n── Pairwise Jaccard overlap of NaN sets (1.0 = identical questions failing) ──\n")
    w = 18
    header = f"  {'':48}" + "".join(f"  {_short(rn)[:w]:>{w}}" for rn in run_names)
    print(header)
    print("  " + "-" * len(header))
    for ra in run_names:
        row = f"  {_short(ra):<48}"
        for rb in run_names:
            if ra == rb:
                row += f"  {'—':>{w}}"
            else:
                inter = len(nan_sets[ra] & nan_sets[rb])
                union = len(nan_sets[ra] | nan_sets[rb])
                j = inter / union if union else 1.0
                row += f"  {j:>{w}.3f}"
        print(row)

    # ── 3. NaN difficulty bias ────────────────────────────────────────────────
    # For each run X: find questions that NaN in X; look up their scores in all
    # other runs where they did NOT NaN; compare to overall mean of those runs.
    print("\n── NaN difficulty bias (how NaN questions score in other runs vs overall mean) ──\n")
    print(f"  {'Run':<48}  {'NaN-q mean':>10}  {'Overall mean':>12}  {'Delta':>7}  {'n_obs':>6}")
    print("  " + "-" * 90)

    for rn in run_names:
        nan_qs = nan_sets[rn]
        if not nan_qs:
            print(f"  {_short(rn):<48}  {'(no NaN)':>10}")
            continue
        # Gather scores for nan_qs from all OTHER runs (where they are non-NaN)
        nan_q_scores: list[float] = []
        all_scores:   list[float] = []
        for other in run_names:
            if other == rn:
                continue
            other_data = run_data[other]
            # overall non-NaN scores in other run
            all_scores.extend(v for v in other_data.values() if not _math.isnan(v))
            # scores of rn's NaN questions in other run
            for qid in nan_qs:
                v = other_data.get(qid, float("nan"))
                if not _math.isnan(v):
                    nan_q_scores.append(v)
        if not nan_q_scores:
            print(f"  {_short(rn):<48}  {'(no overlap)':>10}")
            continue
        nan_mean  = float(_np.mean(nan_q_scores))
        all_mean  = float(_np.mean(all_scores)) if all_scores else float("nan")
        delta     = nan_mean - all_mean
        # positive delta: NaN questions score ABOVE average elsewhere → skipping them deflated this run
        # negative delta: NaN questions score BELOW average elsewhere → skipping them inflated this run
        flag = " ▲ inflated" if delta < -0.005 else (" ▼ deflated" if delta > 0.005 else " ≈ neutral")
        print(f"  {_short(rn):<48}  {nan_mean:>10.4f}  {all_mean:>12.4f}  {delta:>+7.4f}{flag}  {len(nan_q_scores):>6}")

    # ── 4. MCAR summary ──────────────────────────────────────────────────────
    # If NaN sets are highly overlapping (mean pairwise Jaccard high), missingness
    # is systematic (same questions always fail) → MCAR across runs, bias is consistent.
    all_jaccards = []
    for i, ra in enumerate(run_names):
        for rb in run_names[i+1:]:
            inter = len(nan_sets[ra] & nan_sets[rb])
            union = len(nan_sets[ra] | nan_sets[rb])
            if union:
                all_jaccards.append(inter / union)
    mean_j = float(_np.mean(all_jaccards)) if all_jaccards else 0.0
    print("\n── Summary ──\n")
    print(f"  Mean pairwise NaN-set Jaccard: {mean_j:.3f}")
    if mean_j >= 0.5:
        print("  Interpretation: NaN questions are largely the same across runs (systematic parse failures).")
        print("  Scores are comparable — the same questions are excluded from all runs.")
    elif mean_j >= 0.2:
        print("  Interpretation: Moderate overlap — some systematic, some run-specific NaN.")
        print("  Check per-run delta above for inflation/deflation estimates.")
    else:
        print("  Interpretation: Low overlap — different questions fail in different runs.")
        print("  Scores may not be directly comparable; check delta column for bias magnitude.")


def cmd_corrected_report(args: argparse.Namespace) -> None:
    """Bias-corrected scores: impute NaN questions from other runs, recompute Med+Nov mean.

    For each run X and each NaN question q, imputed score = mean of q's score across
    all other runs where q is non-NaN.  Falls back to run X's non-NaN overall mean
    if no other run answered q.  Recomputes the balanced (Med+Nov)/2 mean-of-means
    using actual + imputed scores and reports raw vs corrected side by side.
    """
    import math as _math
    from collections import defaultdict as _dd
    from pathlib import Path as _Path

    import numpy as _np

    runs_dir = _Path(args.out_dir) / "data" / "runs"
    import duckdb as _ddb

    run_filter  = getattr(args, "filter",  None)
    run_exclude = getattr(args, "exclude", None)

    # Load per-question {id: (subset, qtype, ac)} for every run
    # subset derived from results.source ("Medical" → Med, else Nov)
    RunQ = dict  # {qid: (subset, qtype, float|nan)}
    run_data: dict[str, RunQ] = {}

    for db in sorted(runs_dir.glob("*.duckdb")):
        if str(db).endswith(".wal"):
            continue
        rn = db.stem
        if run_filter  and run_filter  not in rn: continue
        if run_exclude and run_exclude in rn:      continue
        try:
            con = _ddb.connect(str(db), read_only=True)
            try:
                rows = con.execute(
                    "SELECT e.id, r.source, e.question_type, e.answer_correctness "
                    "FROM eval_scores e JOIN results r ON e.id = r.id"
                ).fetchall()
            except Exception:
                rows = []
            con.close()
        except Exception:
            continue
        if not rows:
            continue
        d: RunQ = {}
        for qid, source, qtype, ac in rows:
            subset = "Med" if source and "medical" in str(source).lower() else "Nov"
            val    = float("nan") if (ac is None or (isinstance(ac, float) and _math.isnan(ac))) else float(ac)
            d[qid] = (subset, qtype, val)
        run_data[rn] = d

    if len(run_data) < 2:
        print("Need ≥2 runs with eval_scores joined to results. Aborting.")
        return

    run_names = sorted(run_data.keys())

    # Build cross-run score lookup: {qid: [non-NaN scores across all runs]}
    cross: dict[str, list[float]] = _dd(list)
    for rn, d in run_data.items():
        for qid, (_, _, v) in d.items():
            if not _math.isnan(v):
                cross[qid].append(v)

    QTYPES = ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]

    def _balanced_mean(scores_by_st: dict) -> float:
        """(Med_mean + Nov_mean) / 2 where each subset mean = mean of 4 qtype means."""
        def _subset(s):
            vals = [_np.mean(scores_by_st[(s, qt)]) for qt in QTYPES
                    if scores_by_st.get((s, qt))]
            return float(_np.mean(vals)) if vals else float("nan")
        med = _subset("Med")
        nov = _subset("Nov")
        if _math.isnan(med) or _math.isnan(nov):
            return float("nan")
        return (med + nov) / 2

    def _score_run(d: RunQ, impute: bool) -> tuple[float, int]:
        """Return (balanced_mean, n_imputed). impute=False gives raw score."""
        scores_by_st: dict[tuple, list] = _dd(list)
        n_imputed = 0
        overall_fallback = float(_np.mean([v for _, _, v in d.values() if not _math.isnan(v)]) or float("nan"))
        for qid, (subset, qtype, v) in d.items():
            if not _math.isnan(v):
                scores_by_st[(subset, qtype)].append(v)
            elif impute:
                others = [s for s in cross[qid] if True]  # all runs incl. self already excluded by being NaN
                imp = float(_np.mean(others)) if others else overall_fallback
                if not _math.isnan(imp):
                    scores_by_st[(subset, qtype)].append(imp)
                    n_imputed += 1
        return _balanced_mean(scores_by_st), n_imputed

    def _short(name: str) -> str:
        return (name.replace("nobc_ner_ref_rerank_", "")
                    .replace("nobc_", "")
                    .replace("_full", "")
                    .replace("_grid", ""))

    print("\n── Bias-corrected scores (cross-run imputation of NaN questions) ──\n")
    print(f"  {'Run':<48}  {'Raw':>6}  {'Corrected':>9}  {'Delta':>7}  {'N imputed':>10}")
    print("  " + "-" * 88)

    rows_out = []
    for rn in run_names:
        d = run_data[rn]
        raw,       _  = _score_run(d, impute=False)
        corrected, ni = _score_run(d, impute=True)
        delta = corrected - raw if not (_math.isnan(raw) or _math.isnan(corrected)) else float("nan")
        rows_out.append((rn, raw, corrected, delta, ni))

    # Sort by corrected score descending
    rows_out.sort(key=lambda r: -r[2] if not _math.isnan(r[2]) else -999)
    for rn, raw, corrected, delta, ni in rows_out:
        r_str = f"{raw:.4f}"       if not _math.isnan(raw)       else "  —   "
        c_str = f"{corrected:.4f}" if not _math.isnan(corrected) else "  —   "
        d_str = f"{delta:+.4f}"    if not _math.isnan(delta)     else "  —   "
        print(f"  {_short(rn):<48}  {r_str:>6}  {c_str:>9}  {d_str:>7}  {ni:>10}")

    print()
    print("  Imputation: NaN question score = mean of that question's score across all other runs.")
    print("  Fallback (question NaN in all runs): run's own non-NaN mean.")


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
    from concurrent.futures import ThreadPoolExecutor

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


def cmd_prime_cache(args: argparse.Namespace) -> None:
    """Pre-compute and persist question + entity embedding caches."""
    import hashlib
    from pathlib import Path

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from chonk.ner import SpacyMatcher

    out_dir   = Path(args.out_dir)
    data_dir  = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    embed_model_name = args.embed_model or EMBED_MODEL
    spacy_model_name = args.spacy_model or SPACY_MODEL

    _embed_device = os.environ.get("EMBED_DEVICE") or None
    if not _embed_device:
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not torch.cuda.is_available():
                _embed_device = "cpu"
        except ImportError:
            pass

    print(f"Loading embed model: {embed_model_name}" + (f" [{_embed_device}]" if _embed_device else ""))
    embed_model = SentenceTransformer(embed_model_name, device=_embed_device) if _embed_device else SentenceTransformer(embed_model_name)

    # ── Question embeddings ───────────────────────────────────────────────────
    corpus_qs  = _load_questions(data_dir)
    corpus_ids = [q.get("id", f"q{i}") for i, q in enumerate(corpus_qs)]
    q_vecs_cache = data_dir / "question_embeddings.npy"
    q_ids_cache  = data_dir / "question_ids.json"

    if q_vecs_cache.exists() and q_ids_cache.exists():
        if json.loads(q_ids_cache.read_text()) == corpus_ids:
            print(f"Question embeddings already cached ({len(corpus_ids)} questions). Skipping.")
        else:
            q_vecs_cache.unlink()
            q_ids_cache.unlink()
            print("Question ID mismatch — rebuilding question embeddings.")

    if not q_vecs_cache.exists():
        print(f"Embedding {len(corpus_qs)} questions...")
        q_vecs = embed_model.encode(
            [q["question"] for q in corpus_qs],
            normalize_embeddings=True, show_progress_bar=True, batch_size=256,
        ).astype("float32")
        np.save(str(q_vecs_cache), q_vecs)
        q_ids_cache.write_text(json.dumps(corpus_ids), encoding="utf-8")
        print(f"Saved → {q_vecs_cache.name}")

    # ── Entity embeddings ─────────────────────────────────────────────────────
    cache_key    = hashlib.md5(
        (embed_model_name + spacy_model_name + json.dumps(corpus_ids)).encode()
    ).hexdigest()[:16]
    ent_vecs_cache = data_dir / f"entity_vecs_{cache_key}.npz"
    ent_ents_cache = data_dir / f"entity_ents_{cache_key}.json"

    if ent_vecs_cache.exists() and ent_ents_cache.exists():
        print(f"Entity caches already exist ({ent_vecs_cache.name}). Skipping.")
    else:
        print(f"Running NER on {len(corpus_qs)} questions (spaCy: {spacy_model_name})...")
        ner = SpacyMatcher(model=spacy_model_name, strip_numeric=True)
        q_entities_by_id: dict[str, list[str]] = {}
        for i, q in enumerate(corpus_qs):
            qid = corpus_ids[i]
            q_entities_by_id[qid] = [m.name for m in ner.match(q["question"])]
            if (i + 1) % 500 == 0 or (i + 1) == len(corpus_qs):
                print(f"  NER {i+1}/{len(corpus_qs)}", flush=True)

        all_unique_ents = list({e for ents in q_entities_by_id.values() for e in ents})
        print(f"Batch-embedding {len(all_unique_ents)} unique entities...")
        ent_vecs = embed_model.encode(
            all_unique_ents, normalize_embeddings=True,
            show_progress_bar=True, batch_size=256,
        ).astype("float32")
        np.savez(str(ent_vecs_cache), strings=np.array(all_unique_ents), vecs=ent_vecs)
        ent_ents_cache.write_text(json.dumps(q_entities_by_id), encoding="utf-8")
        print(f"Saved → {ent_vecs_cache.name}, {ent_ents_cache.name}")

    print("prime-cache complete.")


# ─────────────────────────────────────────────────────────────────────────────
# run-all and init-config commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_run_all(args: argparse.Namespace) -> None:
    import subprocess as _subprocess
    import sys as _sys

    config_dir = Path(args.config_dir)
    toml_files = sorted(f for f in config_dir.glob("*.toml") if not f.name.startswith("."))
    if not toml_files:
        raise RuntimeError(f"No *.toml files found in {config_dir}")

    out_dir = Path(args.out_dir)
    results_dir = out_dir / "results"

    ap = _make_parser()

    for toml_path in toml_files:
        cfg = _load_config(str(toml_path))
        run_name = cfg.get("run_name")
        if not run_name:
            print(f"SKIP {toml_path.name}: no run_name")
            continue

        crash_marker = results_dir / f".failed_{run_name}"
        if crash_marker.exists():
            print(f"=== SKIP {run_name} (crashed — clear marker to retry) ===")
            continue

        eval_file = results_dir / f"bench_eval_{run_name}_rp.json"
        if eval_file.exists():
            try:
                import json as _json
                _ev = _json.loads(eval_file.read_text())
                _n = _ev.get("_params", {}).get("n_evaluated", 0)
            except Exception:
                _n = 0
            if _n > 0:
                print(f"=== SKIP {run_name}_rp (done, n={_n}) ===")
                continue
            print(f"=== STALE {run_name}_rp (n_evaluated=0, re-running) ===")
            eval_file.unlink()

        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
        except Exception:
            pass

        gen_file = results_dir / f"{run_name}.jsonl"
        if not gen_file.exists() or gen_file.stat().st_size == 0:
            if gen_file.exists():
                gen_file.unlink()
                print(f"=== RUN {run_name} (empty output, regenerating) ===")
            else:
                print(f"=== RUN {run_name} ===")
            cmd = [
                _sys.executable, __file__, "run",
                "--out-dir", str(out_dir),
                "--config", str(toml_path),
                "--run-name", run_name,
            ] + (["--question-ids", args.question_ids] if args.question_ids else [])
            ret = _subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"=== CRASH {run_name} (exit {ret.returncode}) — marking failed, skipping ===")
                crash_marker.write_text(f"exit={ret.returncode}")
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                        _torch.cuda.synchronize()
                        print("  GPU cache cleared after crash.")
                except Exception:
                    pass
                continue
        else:
            print(f"=== SKIP GEN {run_name} (output exists) ===")

        print(f"=== EVAL {run_name}_rp ===")
        rp_src = results_dir / f"{run_name}.jsonl"
        rp_dst = results_dir / f"{run_name}_rp.jsonl"
        if rp_src.exists() and not rp_dst.exists():
            import shutil
            shutil.copy(str(rp_src), str(rp_dst))
        nan_limit = str(cfg.get("eval", {}).get("nan_limit", 136))
        eval_args = ap.parse_args([
            "eval",
            "--out-dir", str(out_dir),
            "--run-name", f"{run_name}_rp",
            "--judge", "gpt-4o-mini-2024-07-18",
            "--eval-rpm", "8000",
            "--eval-batch-size", "20",
            "--concurrency", "50",
            "--nan-limit", nan_limit,
        ])
        try:
            cmd_bench_eval(eval_args)
        except Exception as _exc:
            print(f"=== CRASH EVAL {run_name}_rp: {_exc} — marking failed, skipping ===")
            crash_marker.write_text(str(_exc))
            continue


def cmd_init_config(args: argparse.Namespace) -> None:

    out_dir = Path(args.out_dir)
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    dest = configs_dir / "base.toml"
    if dest.exists() and not getattr(args, "force", False):
        print(f"Exists (use --force to overwrite): {dest}")
        return
    src = Path(__file__).parent.parent / "work" / "configs" / "base.toml"
    if src.exists():
        import shutil
        shutil.copy(str(src), str(dest))
        print(f"Written: {dest}")
    else:
        raise FileNotFoundError(f"Source base.toml not found at {src}")


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
    p = sub.add_parser("index", help="Chunk + embed + store corpus via chonk")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--force", action="store_true", help="Delete existing index and reindex")
    p.add_argument("--embed-content-only", action="store_true",
                   help="Embed content only (no breadcrumb); stores to chonk_nobc.duckdb")
    p.add_argument("--include-doc-name", action="store_true", dest="include_doc_name",
                   help="Include document name in breadcrumbs ([doc > section]); stores to chunkymonkey_bc.duckdb")
    p.add_argument("--min-chunk", type=int, default=None)
    p.add_argument("--max-chunk", type=int, default=None)
    p.add_argument("--with-ner", action="store_true",
                   help="Run NER + cluster after indexing and persist to DB")
    p.add_argument("--with-community", action="store_true", dest="with_community",
                   help="Build community index after indexing and persist to DB")
    p.add_argument("--with-svo", action="store_true", dest="with_svo",
                   help="Extract SVO triples after indexing and persist to DB (requires --gen-model / --gen-provider)")
    p.add_argument("--gen-model", default=GEN_MODEL, dest="gen_model",
                   help=f"Generation model for --with-svo (default: {GEN_MODEL})")
    p.add_argument("--gen-provider", default="openai", choices=["openai", "together", "anthropic"],
                   dest="gen_provider",
                   help="API provider for --with-svo (default: openai)")
    p.add_argument("--db-name", default=None, dest="db_name",
                   help="DB filename override for --with-community / --with-svo")
    p.set_defaults(func=cmd_index)

    # build-ner
    p = sub.add_parser("build-ner", help="Run NER + persist chunk_entities to an existing index DB")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--db-name", required=True, help="DB filename inside {out_dir}/data/")
    p.add_argument("--with-embeddings", action="store_true", dest="with_embeddings",
                   help="Also embed entity strings and store in entity_embeddings table (required for --ner-x)")
    p.add_argument("--with-schema-vocab", action="store_true", dest="with_schema_vocab",
                   help="Augment spaCy NER with SchemaMatcher built from schema/API chunks in the index")
    p.add_argument("--with-context-graph", action="store_true", dest="with_context_graph",
                   help="Build context graph immediately after NER (cooccur + cluster signals)")
    p.add_argument("--force", action="store_true", help="Rebuild even if tables already populated")
    p.set_defaults(func=cmd_build_ner)

    # index-vanilla
    p = sub.add_parser("index-vanilla", help="Build vanilla RAG index: naive 256-token chunks")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--force", action="store_true", help="Delete existing index and reindex")
    p.add_argument("--chunk-tokens", type=int, default=None)
    p.add_argument("--from-store", metavar="DB", default=None,
                   help="Build corpus from an existing Store DuckDB instead of corpus_info.json")
    p.set_defaults(func=cmd_index_vanilla)

    # run
    p = sub.add_parser("run", help="Retrieve and generate answers for all questions")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--limit",       type=int, default=None, metavar="N",
                   help="Limit to first N questions (for testing)")
    p.add_argument("--gen-model",     default=GEN_MODEL,
                   help=f"Generation model (default: {GEN_MODEL})")
    p.add_argument("--gen-provider",  default="openai", choices=["openai", "together", "anthropic"],
                   help="API provider for generation: openai, together, or anthropic (default: openai)")
    p.add_argument("--concurrency",   type=int, default=20,
                   help="Parallel workers (default: 20)")
    p.add_argument("--run-name",      default="contextual",
                   help="Output file prefix (default: contextual)")
    # ── Base ──────────────────────────────────────────────────────────────────
    g_base = p.add_argument_group("base", "Core retrieval mode")
    g_base.add_argument("--vanilla", action="store_true",
                        help=f"Use vanilla RAG index ({VANILLA_CHUNK_TOKENS}-token chunks, k={VANILLA_K})")
    g_base.add_argument("--rerank", action="store_true",
                        help=f"Rerank top-{K_FETCH} candidates to top-{K}")
    g_base.add_argument("--rerank-provider", default="local", choices=["local", "together", "cohere"],
                        help=f"Reranker: local={RERANK_MODEL}, together={RERANK_MODEL_TOGETHER} (default: local)")
    g_base.add_argument("--rerank-chunk", type=int, default=200, metavar="N",
                        help="Outer loop batch size for reranking + checkpoint interval (default: 200)")
    g_base.add_argument("--bm25", action="store_true", dest="bm25",
                        help="Enable BM25 hybrid retrieval fused with vector search via RRF")

    # ── Expansion ─────────────────────────────────────────────────────────────
    g_exp = p.add_argument_group(
        "expansion",
        "Grow the candidate pool beyond vector seed. Dependencies: --enhanced required for all expansion flags.")
    g_exp.add_argument("--enhanced", action="store_true",
                       help=f"Enable entity/structural/cluster expansion (SpacyMatcher/{SPACY_MODEL})")
    g_exp.add_argument("--ner-x", action="store_true", dest="ner_x",
                       help="Add entity embedding ANN expansion; requires --enhanced + build-ner --with-embeddings")
    g_exp.add_argument("--cluster", action="store_true",
                       help="Enable cluster-neighbor expansion (off by default); requires --enhanced")
    g_exp.add_argument("--entity-ref-expansion", action="store_true", dest="entity_ref_expansion",
                       help="Post-selection: if query entities absent from top-k, re-search and insert covering chunks")
    g_exp.add_argument("--entity-ref-expansion-per-k", type=int, default=None, dest="entity_ref_expansion_per_k",
                       help="Per-entity retrieval k for --entity-ref-expansion (default: k / n_missing)")
    g_exp.add_argument("--entity-ref-expansion-min-sim", type=float, default=None, dest="entity_ref_expansion_min_sim",
                       help="Min cosine sim for --entity-ref-expansion hits (default: no filter)")
    g_exp.add_argument("--context-graph", action="store_true", dest="context_graph",
                       help="Expand ref-expansion via context graph edges (requires build-context-graph first)")
    g_exp.add_argument("--context-graph-min-weight", type=float, default=0.1, dest="context_graph_min_weight",
                       help="Min edge weight for context graph expansion (default: 0.1)")
    g_exp.add_argument("--context-graph-top-k", type=int, default=5, dest="context_graph_top_k",
                       help="Max context graph edges to follow per missing entity (default: 5)")
    g_exp.add_argument("--breadcrumb-embed", action="store_true",
                       help="Use bc-in-embedding index (chunkymonkey_1100_2200.duckdb); improves seed quality for structured docs")

    # ── Pool Management ───────────────────────────────────────────────────────
    g_pool = p.add_argument_group(
        "pool_management",
        "Control which expanded candidates survive to the final top-k. Meaningful only when --enhanced is set.")
    g_pool.add_argument("--lane-entity-min-sim", type=float, default=None, dest="lane_entity_min_sim",
                        help="Quality gate: drop entity-adjacent chunks with query-sim < threshold before pool merge (e.g. 0.45)")
    g_pool.add_argument("--redundancy-threshold", type=float, default=None, dest="redundancy_threshold",
                        help="Dedup: drop near-duplicate chunks from merged pool before top-k selection (e.g. 0.92); fires after lane filter")
    g_pool.add_argument("--concentration-threshold", type=float, default=None, dest="concentration_threshold",
                        help="Auto-trigger --entity-ref-expansion if ≥ this fraction of top-k is from one doc (e.g. 0.6)")

    # ── Context Enrichment ────────────────────────────────────────────────────
    g_ctx = p.add_argument_group(
        "context_enrichment",
        "Modify what the generator sees. Independent of retrieval quality.")
    g_ctx.add_argument("--community-context", action="store_true", dest="community_context",
                       help="Inject community topic labels into generation prompt (requires build-community)")
    g_ctx.add_argument("--community-min-coherence", type=float, default=0.0, dest="community_min_coherence",
                       help="Suppress community labels with coherence < threshold (default: 0.0)")
    g_ctx.add_argument("--query-complexity-threshold", type=int, default=2, dest="query_complexity_threshold",
                       help="Skip community context for low-complexity queries (default: 2)")
    g_ctx.add_argument("--breadcrumb-context", action="store_true",
                       help="Prepend breadcrumb heading to each chunk in the generator context window")
    g_ctx.add_argument("--breadcrumb-style", default="markdown",
                       choices=["markdown", "literal", "symbol"], dest="breadcrumb_style",
                       help="Breadcrumb format in context: markdown (## headings), literal (Section: X.), symbol ([X > Y])")

    # ── Misc ──────────────────────────────────────────────────────────────────
    g_misc = p.add_argument_group("misc")
    g_misc.add_argument("--entity-ref-retry", action="store_true", dest="entity_ref_retry",
                        help="On refusal answer, retry with partial-answer hint (max 1 retry)")
    g_misc.add_argument("--search-mode", default="vector_first",
                        choices=["vector_first", "graph_first", "global", "map_reduce_global"],
                        dest="search_mode",
                        help="EnhancedSearch retrieval mode: vector_first (default), graph_first (entity graph traversal), global (community summaries only), map_reduce_global (MS-GraphRAG map-reduce over community summaries)")
    g_misc.add_argument("--sr", action="store_true", dest="sr",
                        help="Structured Response: generate with JSON schema (key_claims + evidence_used) once, no coverage check or gap-fill")
    g_misc.add_argument("--srr", action="store_true", dest="srr",
                        help="Structured Response Retry: generate with JSON schema, check entity coverage, gap-fill and retry (max 2 rounds)")
    g_misc.add_argument("--srr-model", default=None, dest="srr_model",
                        help="Model for SRR calls (default: same as --gen-model). Use a cheaper model e.g. gpt-4o-mini while --gen-model uses gpt-4o.")
    g_misc.add_argument("--srr-provider", default=None, choices=["openai", "together", "anthropic"], dest="srr_provider",
                        help="API provider for SRR calls (default: same as --gen-provider)")
    g_misc.add_argument("--multi-step", action="store_true", dest="multi_step",
                        help="Multi-step retrieval: decompose each question into sub-queries, retrieve for each, merge hits before generation")
    g_misc.add_argument("--structured-gen", action="store_true", dest="structured_gen",
                        help="Require ANSWER: <text> format from generator; retry once if non-compliant; strip marker before judge")
    g_misc.add_argument("--top-k", type=int, default=None, dest="top_k", metavar="K",
                        help=f"Override retrieval top-k (default: {K})")
    g_misc.add_argument("--namespaces", nargs="*", default=None, dest="namespaces",
                        help="Restrict retrieval to these namespaces (default: all namespaces)")
    g_misc.add_argument("--domain-ids", nargs="*", default=None, dest="domain_ids",
                        help="Restrict retrieval to these domain_ids (default: all domains)")
    g_misc.add_argument("--auto-domain-filter", action="store_true", default=False, dest="auto_domain_filter",
                        help="Enable per-query LLM domain routing (ADF); tags FANG corpus on first use")
    g_misc.add_argument("--db-name", default=None, dest="db_name",
                        help="Override DB filename inside {out_dir}/data/ (default: auto from --breadcrumb-embed)")
    g_misc.add_argument("--question-ids", default=None, dest="question_ids", metavar="PATH",
                        help="JSON file with list of question IDs to run (default: all)")
    g_misc.add_argument("--config", default=None, metavar="TOML",
                        help="Path to run config TOML file; CLI flags override")

    p.set_defaults(func=cmd_run)

    # build-community
    p = sub.add_parser("build-community", help="Build community index: heading vectors + weighted avg + Louvain detection")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--db-name", required=True, help="DB filename inside {out_dir}/data/")
    p.add_argument("--alpha", type=float, default=0.2,
                   help="Heading weight in weighted average (default: 0.2)")
    p.add_argument("--sim-threshold", type=float, default=0.6, dest="sim_threshold",
                   help="Min cosine sim for graph edges (default: 0.6)")
    p.add_argument("--community-label-strategy", default="ner_embedding",
                   choices=["term_freq", "ner_embedding"], dest="community_label_strategy",
                   help="Community label method: ner_embedding (default) uses entity embeddings; term_freq uses word frequency")
    p.add_argument("--gen-provider", default=None, dest="gen_provider",
                   choices=["openai", "together", "anthropic"],
                   help="LLM provider for community summary generation (omit to skip summaries)")
    p.add_argument("--gen-model", default=GEN_MODEL, dest="gen_model",
                   help=f"Model for community summary generation (default: {GEN_MODEL})")
    p.add_argument("--force", action="store_true", help="Rebuild even if already populated")
    p.set_defaults(func=cmd_build_community)

    # build-svo
    p = sub.add_parser("build-svo", help="Extract SVO triples from all chunks and persist to svo_triples table")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--db-name", default=None, dest="db_name",
                   help=f"DB filename inside {{out_dir}}/data/ (default: {DB_FILENAME})")
    p.add_argument("--force", action="store_true", help="Rebuild even if already populated")
    p.add_argument("--gen-model", default=GEN_MODEL, dest="gen_model",
                   help=f"LLM model for triple extraction (default: {GEN_MODEL})")
    p.add_argument("--gen-provider", default="openai", choices=["openai", "together", "anthropic"],
                   dest="gen_provider",
                   help="API provider for LLM (default: openai)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Parallel extraction workers (default: 4)")
    p.add_argument("--max-chunks", type=int, default=None, dest="max_chunks",
                   help="Limit to first N chunks (for testing)")
    p.add_argument("--progress-out", default=None, dest="progress_out",
                   metavar="PATH",
                   help="Write NDJSON progress events to PATH ('-' for stdout). "
                        "One JSON line per chunk: {done, total, chunk_id, triples, "
                        "descriptions, aliases, rel_descriptions}. Suitable for "
                        "tailing by a web service and forwarding as SSE.")
    p.add_argument("--with-context-graph", action="store_true", dest="with_context_graph",
                   help="Rebuild context graph after SVO extraction (adds svo_signal to all edges)")
    p.set_defaults(func=cmd_build_svo)

    # build-context-graph
    p = sub.add_parser("build-context-graph", help="Build context graph edges from chunk_entities + svo_triples")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--db-name", default=None, dest="db_name",
                   help=f"DB filename inside {{out_dir}}/data/ (default: {DB_FILENAME})")
    p.add_argument("--namespace", default="global",
                   help="Namespace to build graph for, or 'all' for every namespace (default: global)")
    p.add_argument("--min-weight", type=float, default=0.1, dest="min_weight",
                   help="Prune edges below this weight (default: 0.1)")
    p.add_argument("--algorithm", default="agglomerative", choices=["agglomerative", "dbscan", "leiden"],
                   help="Clustering algorithm for chunk clusters (default: agglomerative)")
    p.add_argument("--min-chunks", type=int, default=10, dest="min_chunks",
                   help="Minimum chunks before clustering runs (default: 10)")
    p.add_argument("--force", action="store_true", help="Rebuild even if cache is valid")
    p.set_defaults(func=cmd_build_context_graph)

    # use-endpoints — run benchmark against existing Together dedicated endpoints
    p = sub.add_parser("use-endpoints", help="Run benchmark using existing Together dedicated endpoints")
    p.add_argument("endpoint_ids", nargs="+", metavar="ENDPOINT_ID",
                   help="One or more Together dedicated endpoint IDs")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--concurrency", type=int, default=20,
                   help="Workers per endpoint (default: 20, total = N * 20)")
    p.add_argument("--judge",       default="gpt-4o-mini")
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
    p.add_argument("--question-ids",   default=None, metavar="PATH",
                   help="JSON file with list of question IDs to evaluate")
    p.add_argument("--concurrency",    type=int, default=5,
                   help="Parallel workers (default: 5)")
    p.add_argument("--eval-rpm",       type=int, default=None, dest="eval_rpm",
                   help="Max judge API requests per minute (token-bucket throttle, default: no limit)")
    p.add_argument("--eval-batch-size", type=int, default=10, dest="eval_batch_size",
                   help="Items per async batch (default: 10)")
    p.add_argument("--nan-limit",      type=int, default=None, dest="nan_limit",
                   help="Stop retrying NaN items once total finalized-NaN count is ≤ this threshold (default: always retry)")
    p.add_argument("--early-stop-target", type=float, default=None, dest="early_stop_target",
                   help="Abort eval (exit 2) when upper 95%% CI of answer_correctness falls below this score")
    p.add_argument("--early-stop-min-n",  type=int,   default=500,  dest="early_stop_min_n",
                   help="Min evaluated questions before early-stop check fires (default: 500)")
    p.set_defaults(func=cmd_bench_eval)

    # score — print All score for a completed run (for shell script use)
    p = sub.add_parser("score", help="Print All score (mean of Med+Nov answer_correctness) for a run")
    p.add_argument("--out-dir",  default="/tmp/grb", metavar="DIR")
    p.add_argument("--run-name", required=True, help="Run name to score")
    p.set_defaults(func=cmd_score)

    # backfill-db — reconstruct DuckDB from jsonl + eval checkpoint
    p = sub.add_parser("backfill-db", help="Reconstruct run DuckDB from results jsonl + eval checkpoint jsonl")
    p.add_argument("--out-dir",  default="/tmp/grb", metavar="DIR")
    p.add_argument("--run-name", required=True, help="Run name to backfill")
    p.set_defaults(func=cmd_backfill_db)

    p = sub.add_parser("prep-nan-reeval",
                       help="Prepare checkpoint keeping only non-NaN results so eval re-runs NaN questions only")
    p.add_argument("--out-dir",  default="/tmp/grb", metavar="DIR")
    p.add_argument("--run-name", required=True)
    p.set_defaults(func=cmd_prep_nan_reeval)

    # make-order — generate stratified question order for full-corpus runs
    p = sub.add_parser("make-order", help="Generate stratified question order file for full-corpus runs")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="Output JSON path (default: {out_dir}/data/full_corpus_stratified_order.json)")
    p.set_defaults(func=cmd_make_order)

    # prime-cache — pre-compute and persist all question + entity embeddings
    p = sub.add_parser("prime-cache", help="Pre-compute question and entity embedding caches (run before 'run')")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--embed-model", default=None, metavar="MODEL",
                   help=f"Embedding model (default: {EMBED_MODEL})")
    p.add_argument("--spacy-model", default=None, metavar="MODEL",
                   help=f"spaCy NER model (default: {SPACY_MODEL})")
    p.set_defaults(func=cmd_prime_cache)

    # report — combined matrix of all eval runs + leaderboard
    p = sub.add_parser("nan-bias-report", help="Assess NaN question bias across runs")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--filter", default=None, metavar="STR")
    p.add_argument("--exclude", default=None, metavar="STR")
    p.set_defaults(func=cmd_nan_bias_report)

    p = sub.add_parser("corrected-report", help="Bias-corrected scores via cross-run NaN imputation")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--filter", default=None, metavar="STR")
    p.add_argument("--exclude", default=None, metavar="STR")
    p.set_defaults(func=cmd_corrected_report)

    p = sub.add_parser("report", help="Combined matrix: our eval runs + leaderboard")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--filter", default=None, metavar="STR",
                   help="Only show runs whose name contains STR (e.g. 'grid' or 'full')")
    p.add_argument("--exclude", default=None, metavar="STR",
                   help="Exclude runs whose name contains STR (e.g. 'full' to show only grid runs)")
    p.set_defaults(func=cmd_bench_report)

    # run-all — run every TOML config in a directory, skipping completed runs
    p = sub.add_parser("run-all", help="Run all TOML configs in a directory, then eval each")
    p.add_argument("--config-dir", required=True, metavar="DIR",
                   help="Directory containing *.toml run configs")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--question-ids", default=None, dest="question_ids", metavar="PATH",
                   help="JSON file with list of question IDs to run (default: all)")
    p.set_defaults(func=cmd_run_all)

    # init-config — write base.toml to {out_dir}/configs/
    p = sub.add_parser("init-config", help="Write base.toml with all defaults to {out_dir}/configs/")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--force", action="store_true", help="Overwrite if exists")
    p.set_defaults(func=cmd_init_config)

    return ap


def main() -> None:
    ap   = _make_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
