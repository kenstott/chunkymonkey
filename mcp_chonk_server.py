#!/usr/bin/env python
"""
MCP server exposing Chonk search over one or more DuckDB indexes.

Requirements:
    pip install "chonk-rag[storage]" mcp

Transport (CHONK_TRANSPORT):
    stdio  (default) – local process; MCP host manages the subprocess
    http             – Starlette/uvicorn HTTP server; clients connect by URL

    stdio is for local/developer use. http is for centralised enterprise
    deployment where end users point their MCP client at a URL.

DB config:
    CHONK_DB_PATH         – path to DuckDB file (single DB)
    CHONK_EMBEDDING_DIM   – embedding dimension (int, default 1024)
    CHONK_DB_CONFIG       – JSON mapping name → {"path": "...", "embedding_dim": N}
                            Takes precedence over CHONK_DB_PATH when set.

HTTP transport env vars:
    CHONK_HOST            – bind host (default 0.0.0.0)
    CHONK_PORT            – bind port (default 8000)
    CHONK_API_KEY         – if set, all requests must carry
                            Authorization: Bearer <key>

Claude Desktop config (stdio):
    {
      "mcpServers": {
        "chonk": {
          "command": "python",
          "args": ["/path/to/mcp_chonk_server.py"],
          "env": {"CHONK_DB_PATH": "/data/index.duckdb"}
        }
      }
    }

Claude Desktop config (http — enterprise):
    {
      "mcpServers": {
        "chonk": {
          "url": "http://chonk.internal:8000/mcp",
          "headers": {"Authorization": "Bearer <key>"}
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import numpy as np
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from chonk.storage import Store

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_DIM = 1024

# Together.ai chat config for the `ask` (RAG synthesis) tool.
_CHAT_MODEL = os.environ.get("CHONK_CHAT_MODEL", "Qwen/Qwen3.5-9B")
_TOGETHER_BASE_URL = os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
_ANSWER_TOKEN_BUDGET = int(os.environ.get("CHONK_ANSWER_TOKEN_BUDGET", "4096"))
_ANSWER_MAX_TOKENS = int(os.environ.get("CHONK_ANSWER_MAX_TOKENS", "2048"))


def _together_chat(prompt: str) -> str:
    """Synchronous Together.ai chat completion. Returns the answer text.

    Raises (no fallback) if the API key is missing, the request fails, or the
    response carries no choices.
    """
    import httpx

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY is not set; the ask tool requires it.")

    resp = httpx.post(
        f"{_TOGETHER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": _CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer concisely. Lead with the direct answer, then only the "
                        "supporting detail the question needs. Prefer tight prose or a "
                        "short list over long exposition. Do not pad, restate the "
                        "question, or add a preamble. Ground every claim in the "
                        "provided context, but synthesize in your own words — the "
                        "sources are returned to the caller as citations, so do not "
                        "reproduce source text verbatim or quote long passages."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": _ANSWER_MAX_TOKENS,
            "temperature": 0,
            # Qwen3.5 is a thinking model; disable CoT so the answer lands in
            # message.content rather than a separate reasoning field.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices")
    if not choices:
        raise RuntimeError(f"Together response had no choices: {data}")
    return choices[0]["message"]["content"]


def _load_stores() -> dict[str, Store]:
    config_json = os.environ.get("CHONK_DB_CONFIG")
    if config_json:
        config = json.loads(config_json)
        return {
            name: Store(
                entry["path"],
                embedding_dim=int(entry.get("embedding_dim", _DEFAULT_DIM)),
                read_only=True,
            )
            for name, entry in config.items()
        }

    db_path = os.environ.get("CHONK_DB_PATH")
    if not db_path:
        raise RuntimeError("Either CHONK_DB_PATH or CHONK_DB_CONFIG env var is required")
    dim = int(os.environ.get("CHONK_EMBEDDING_DIM", str(_DEFAULT_DIM)))
    return {"default": Store(db_path, embedding_dim=dim, read_only=True)}


STORES: dict[str, Store] = _load_stores()

SERVER = Server("chonk-search")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_chunk(chunk_id: str, score: float, chunk: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "chunk_id": chunk_id,
        "score": float(score),
        "document_name": getattr(chunk, "document_name", ""),
        "section": getattr(chunk, "section", []),
        "chunk_type": getattr(chunk, "chunk_type", "document"),
        "source": getattr(chunk, "source", ""),
        "breadcrumb": getattr(chunk, "breadcrumb", None),
        "content": getattr(chunk, "content", ""),
        "source_offset": getattr(chunk, "source_offset", None),
        "source_length": getattr(chunk, "source_length", None),
        "source_detail": getattr(chunk, "source_detail", None),
    }


def _db_path_for(store_name: str) -> str:
    config_json = os.environ.get("CHONK_DB_CONFIG")
    if config_json:
        return json.loads(config_json)[store_name]["path"]
    return os.environ["CHONK_DB_PATH"]


def _fetch_chunk_by_id(store_name: str, chunk_id: str) -> dict[str, Any]:
    import duckdb

    from chonk.models import DocumentChunk
    from chonk.storage._vector import _deserialize_section  # type: ignore[attr-defined]

    conn = duckdb.connect(_db_path_for(store_name), read_only=True)
    row = conn.execute(
        """
        SELECT chunk_id, document_name, section, chunk_index, content,
               breadcrumb, chunk_type, source_offset, source_length, source_detail
        FROM embeddings WHERE chunk_id = ?
        """,
        [chunk_id],
    ).fetchone()
    conn.close()
    if row is None:
        raise KeyError(f"chunk_id not found: {chunk_id}")

    (
        cid,
        doc_name,
        section_raw,
        chunk_index,
        content,
        breadcrumb,
        chunk_type,
        source_offset,
        source_length,
        source_detail_str,
    ) = row

    chunk = DocumentChunk(
        document_name=doc_name,
        content=content,
        section=_deserialize_section(section_raw),
        chunk_index=chunk_index,
        source_offset=source_offset,
        source_length=source_length,
        breadcrumb=breadcrumb,
        chunk_type=chunk_type or "document",
        source_detail=json.loads(source_detail_str) if source_detail_str else None,
    )
    return _serialize_chunk(cid, 1.0, chunk)


def _fetch_neighbors(
    store_name: str,
    base_chunk: dict[str, Any],
    radius: int,
) -> list[dict[str, Any]]:
    import duckdb

    from chonk.models import DocumentChunk
    from chonk.storage._vector import _deserialize_section  # type: ignore[attr-defined]

    conn = duckdb.connect(_db_path_for(store_name), read_only=True)
    row = conn.execute(
        "SELECT chunk_index FROM embeddings WHERE chunk_id = ?",
        [base_chunk["chunk_id"]],
    ).fetchone()
    if row is None:
        conn.close()
        return []

    base_idx = int(row[0])
    rows = conn.execute(
        """
        SELECT chunk_id, document_name, section, chunk_index, content,
               breadcrumb, chunk_type, source_offset, source_length, source_detail
        FROM embeddings
        WHERE document_name = ? AND chunk_index BETWEEN ? AND ? AND chunk_id <> ?
        ORDER BY chunk_index
        """,
        [
            base_chunk["document_name"],
            base_idx - radius,
            base_idx + radius,
            base_chunk["chunk_id"],
        ],
    ).fetchall()
    conn.close()

    neighbors: list[dict[str, Any]] = []
    for (
        cid,
        dname,
        section_raw,
        cidx,
        content,
        breadcrumb,
        chunk_type,
        source_offset,
        source_length,
        source_detail_str,
    ) in rows:
        chunk = DocumentChunk(
            document_name=dname,
            content=content,
            section=_deserialize_section(section_raw),
            chunk_index=cidx,
            source_offset=source_offset,
            source_length=source_length,
            breadcrumb=breadcrumb,
            chunk_type=chunk_type or "document",
            source_detail=json.loads(source_detail_str) if source_detail_str else None,
        )
        neighbors.append(_serialize_chunk(cid, 1.0, chunk))
    return neighbors


# ---------------------------------------------------------------------------
# MCP: tool definitions
# ---------------------------------------------------------------------------


@SERVER.list_tools()
async def list_tools() -> list[Tool]:
    db_names = list(STORES.keys())
    return [
        Tool(
            name="search_chunks",
            description=(
                "Hybrid vector + BM25 search over the Chonk index. "
                "Supply a query_embedding (float array) computed from the user's question. "
                "Optionally include query_text for BM25 hybrid. "
                f"Available DBs: {db_names}. Omit 'db' to search all and merge by score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query_embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Embedding vector for the query, shape (dim,).",
                    },
                    "query_text": {
                        "type": "string",
                        "description": "Raw query text for BM25 hybrid search.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 5,
                    },
                    "db": {
                        "type": "string",
                        "description": (
                            f"Target a specific DB by name. One of: {db_names}. Omit to search all."
                        ),
                    },
                    "namespaces": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "chunk_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            'e.g. ["document"], ["db_table","db_column"], ["api_endpoint"]'
                        ),
                    },
                },
                "required": ["query_embedding"],
            },
        ),
        Tool(
            name="ask",
            description=(
                "RAG answer synthesis. Retrieves the most relevant chunks, then generates "
                f"a grounded natural-language answer with source citations via the Together.ai "
                f"chat model ({_CHAT_MODEL}). Use this when you want a ready-made answer rather "
                "than raw chunks. This server does not embed text itself: supply query_embedding "
                "(the embedded form of query_text) just as for search_chunks. "
                f"Available DBs: {db_names}. Omit 'db' to retrieve across all and merge by score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The natural-language question to answer.",
                    },
                    "query_embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Embedding vector for query_text, shape (dim,).",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 8,
                        "description": "Number of chunks to retrieve as context.",
                    },
                    "db": {
                        "type": "string",
                        "description": (
                            f"Target a specific DB by name. One of: {db_names}. Omit to use all."
                        ),
                    },
                    "namespaces": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "chunk_types": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["query_text", "query_embedding"],
            },
        ),
        Tool(
            name="get_chunk",
            description="Fetch a specific chunk by chunk_id, optionally with neighbors.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "db": {
                        "type": "string",
                        "description": (
                            f"Which DB to query. One of: {db_names}. Defaults to first."
                        ),
                    },
                    "include_neighbors": {"type": "boolean", "default": False},
                    "neighbor_radius": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 1,
                    },
                },
                "required": ["chunk_id"],
            },
        ),
        Tool(
            name="expand_chunk_graph",
            description=(
                "Expand a chunk into entity/relation/community overlays. "
                "Use after retrieving a chunk via search_chunks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "db": {
                        "type": "string",
                        "description": f"Which DB to query. One of: {db_names}.",
                    },
                },
                "required": ["chunk_id"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# MCP: single dispatch handler (correct API)
# ---------------------------------------------------------------------------


@SERVER.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    args = arguments or {}

    if name == "search_chunks":
        return await _search_chunks(args)
    if name == "ask":
        return await _ask(args)
    if name == "get_chunk":
        return await _get_chunk(args)
    if name == "expand_chunk_graph":
        return await _expand_chunk_graph(args)

    raise ValueError(f"Unknown tool: {name!r}")


async def _search_chunks(args: dict[str, Any]) -> list[TextContent]:
    raw = args.get("query_embedding")
    if raw is None:
        raise ValueError("query_embedding is required")

    query_embedding = np.asarray(raw, dtype="float32")
    if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
        query_embedding = query_embedding[0]
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be shape (dim,) or (1, dim)")

    limit = int(args.get("limit", 5))
    query_text: str | None = args.get("query_text") or None
    namespaces: list[str] | None = args.get("namespaces")
    chunk_types: list[str] | None = args.get("chunk_types")
    target_db: str | None = args.get("db")

    if target_db and target_db not in STORES:
        raise ValueError(f"Unknown db {target_db!r}. Available: {list(STORES)}")
    target_stores = {target_db: STORES[target_db]} if target_db else STORES

    all_results: list[dict[str, Any]] = []
    for db_name, store in target_stores.items():
        for cid, score, chunk in store.search(
            query_embedding=query_embedding,
            limit=limit,
            query_text=query_text,
            namespaces=namespaces,
            chunk_types=chunk_types,
        ):
            row = _serialize_chunk(cid, score, chunk)
            row["db"] = db_name
            all_results.append(row)

    all_results.sort(key=lambda r: r["score"], reverse=True)
    all_results = all_results[:limit]

    wrapper = {
        "results": all_results,
        "usage": {
            "instructions": (
                "Each result has 'content' (the text to rely on), 'document_name', "
                "'section', 'breadcrumb' describing origin, 'score' (higher = more similar), "
                "and 'db' indicating which store it came from. "
                "Base your answer strictly on the 'content' fields. "
                "Cite document_name/section when helpful. "
                "If no content clearly answers the question, say you don't know."
            )
        },
    }
    return [TextContent(type="text", text=json.dumps(wrapper))]


async def _ask(args: dict[str, Any]) -> list[TextContent]:
    from chonk.generation import AnswerContext, AnswerGenerator
    from chonk.models import ScoredChunk

    query_text = (args.get("query_text") or "").strip()
    if not query_text:
        raise ValueError("query_text is required")
    raw = args.get("query_embedding")
    if raw is None:
        raise ValueError("query_embedding is required")

    query_embedding = np.asarray(raw, dtype="float32")
    if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
        query_embedding = query_embedding[0]
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be shape (dim,) or (1, dim)")

    limit = int(args.get("limit", 8))
    namespaces: list[str] | None = args.get("namespaces")
    chunk_types: list[str] | None = args.get("chunk_types")
    target_db: str | None = args.get("db")

    if target_db and target_db not in STORES:
        raise ValueError(f"Unknown db {target_db!r}. Available: {list(STORES)}")
    target_stores = {target_db: STORES[target_db]} if target_db else STORES

    ranked: list[tuple[float, str, ScoredChunk]] = []
    for db_name, store in target_stores.items():
        for cid, score, chunk in store.search(
            query_embedding=query_embedding,
            limit=limit,
            query_text=query_text,
            namespaces=namespaces,
            chunk_types=chunk_types,
        ):
            ranked.append(
                (
                    score,
                    db_name,
                    ScoredChunk(chunk_id=cid, chunk=chunk, score=score, provenance=db_name),
                )
            )

    ranked.sort(key=lambda r: r[0], reverse=True)
    ranked = ranked[:limit]
    db_by_chunk = {sc.chunk_id: db_name for _, db_name, sc in ranked}
    scored = [sc for _, _, sc in ranked]

    context = AnswerContext(chunks=scored, query=query_text)
    generator = AnswerGenerator(_together_chat, token_budget=_ANSWER_TOKEN_BUDGET)
    # generate() builds the prompt and makes a blocking HTTP call; run off-loop.
    answer = await asyncio.get_event_loop().run_in_executor(None, generator.generate, context)

    citations = []
    for c in answer.citations:
        row = _serialize_chunk(c.chunk_id, c.score, c.chunk)
        row["db"] = db_by_chunk.get(c.chunk_id, "")
        citations.append(row)

    payload = {
        "answer": answer.text,
        "citations": citations,
        "meta": {
            "query": query_text,
            "db_filter": target_db,
            "model": _CHAT_MODEL,
            "chunks_retrieved": len(scored),
            "chunks_cited": len(answer.citations),
        },
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def _get_chunk(args: dict[str, Any]) -> list[TextContent]:
    chunk_id = args.get("chunk_id")
    if not chunk_id:
        raise ValueError("chunk_id is required")

    db_name = args.get("db") or next(iter(STORES))
    if db_name not in STORES:
        raise ValueError(f"Unknown db {db_name!r}. Available: {list(STORES)}")

    include_neighbors = bool(args.get("include_neighbors", False))
    neighbor_radius = int(args.get("neighbor_radius", 1))

    base = _fetch_chunk_by_id(db_name, chunk_id)
    out: dict[str, Any] = {"chunk": base, "db": db_name}
    if include_neighbors and neighbor_radius > 0:
        out["neighbors"] = _fetch_neighbors(db_name, base, neighbor_radius)

    return [TextContent(type="text", text=json.dumps(out))]


async def _expand_chunk_graph(args: dict[str, Any]) -> list[TextContent]:
    chunk_id = args.get("chunk_id", "")
    raise NotImplementedError(
        "expand_chunk_graph is not yet wired to an EntityIndex/RelationshipIndex. "
        "Implement using your graph/NER indices for chunk_id=" + chunk_id
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_TRANSPORT = os.environ.get("CHONK_TRANSPORT", "stdio").lower()


async def _run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await SERVER.run(
            read_stream,
            write_stream,
            SERVER.create_initialization_options(),
        )


async def _run_http() -> None:
    from contextlib import asynccontextmanager

    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Mount
    from starlette.types import Receive, Scope, Send

    _API_KEY = os.environ.get("CHONK_API_KEY")
    _manager = StreamableHTTPSessionManager(app=SERVER, stateless=False)

    async def handle_mcp(scope: Scope, receive: Receive, send: Send) -> None:
        if _API_KEY and scope.get("type") == "http":
            request = Request(scope, receive)
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {_API_KEY}":
                response = JSONResponse({"error": "Unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await _manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def lifespan(_app):  # noqa: ANN001, ANN202
        async with _manager.run():
            yield

    app = Starlette(
        lifespan=lifespan,
        routes=[Mount("/mcp", app=handle_mcp)],
    )

    host = os.environ.get("CHONK_HOST", "0.0.0.0")
    port = int(os.environ.get("CHONK_PORT", "8000"))
    config = uvicorn.Config(app, host=host, port=port)
    await uvicorn.Server(config).serve()


async def main() -> None:
    if _TRANSPORT == "http":
        await _run_http()
    else:
        await _run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
