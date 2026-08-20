import contextlib
import os
import sys
import json
from functools import lru_cache
from pathlib import Path

import mcp.server.stdio
from mcp import types
from mcp.server import Server

from ..index.index import get_collection
from ..retrieval.retrieve import bm25_search, cosign_simularity, search_hybrid_rrf

TOOL_DESCRIPTIONS = {
    "bm25_search": (
        "Lexical/keyword search via BM25. Best for exact terminology, rare "
        "technical terms, or identifiers that must match the source text "
        "verbatim. Weak on paraphrased or conceptually related queries with "
        "little vocabulary overlap with the source."
    ),
    "vector_search": (
        "Semantic search via embedding cosine similarity. Best for "
        "paraphrased or conceptually related queries where the wording "
        "differs from the source text."
    ),
    "hybrid_search": (
        "Combines bm25_search and vector_search via Reciprocal Rank Fusion. "
        "Use when uncertain which single method fits, or when a query mixes "
        "exact terminology with conceptual phrasing."
    ),
}

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query."},
        "top_k": {"type": "integer", "description": "How many results to return.", "default": 5},
    },
    "required": ["query"],
}

server = Server("retrieval-arena")


@lru_cache(maxsize=None)
def _resolve_collection(name: str | None):
    return get_collection(name)


def _resolve_chunks_path() -> Path | None:
    raw = os.getenv("RETRIEVAL_ARENA_MCP_CHUNKS_PATH")
    return Path(raw) if raw else None


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(name=name, description=description, inputSchema=INPUT_SCHEMA)
        for name, description in TOOL_DESCRIPTIONS.items()
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    query = arguments["query"]
    top_k = arguments.get("top_k", 5)
    collection_name = os.getenv("RETRIEVAL_ARENA_MCP_COLLECTION")
    chunks_path = _resolve_chunks_path()

    # bm25_search/cosign_simularity print() on failure; stdio MCP reserves
    # stdout for JSON-RPC framing, so any stray print corrupts the protocol
    with contextlib.redirect_stdout(sys.stderr):
        if name == "bm25_search":
            results = bm25_search(query, path=chunks_path, top_k=top_k)
        elif name == "vector_search":
            results = cosign_simularity(query, top_k=top_k, collection=_resolve_collection(collection_name))
        elif name == "hybrid_search":
            results = search_hybrid_rrf(query, top_k=top_k, bm25_path=chunks_path, collection=_resolve_collection(collection_name))
        else:
            raise ValueError(f"Unknown tool {name!r}")

    return [types.TextContent(type="text", text=json.dumps(results))]


async def serve() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(serve())

