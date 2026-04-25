"""sdk/mcp_server.py

Exposes Brave Local RAG pipeline as an MCP server for Hermes agent.

Tools:
  brave_local_index    — fetch Brave POIs and build/refresh the RAG index
  brave_local_retrieve — retrieve top-k POIs by semantic similarity
  brave_local_status   — return index health (count + TTL)

Usage (stdio MCP):
  python sdk/mcp_server.py
"""
from __future__ import annotations

import asyncio
import json

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from sdk.embeddings import strategy_for
from sdk.local_rag import DEFAULT_STORE, build_local_rag_index, retrieve_local

app = Server("brave-rag")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="brave_local_index",
            description=(
                "Fetch Brave local POIs and build or refresh the vector RAG index. "
                "Call before retrieve when the index is stale or missing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Place search term, e.g. 'photo studio'",
                    },
                    "location": {"type": "string", "default": "Bradenton, FL"},
                    "radius": {
                        "type": "integer", "default": 5000,
                        "description": "Metres, max 20000",
                    },
                    "force": {
                        "type": "boolean", "default": False,
                        "description": "Force rebuild even if index is fresh",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="brave_local_retrieve",
            description="Retrieve top-k nearby POIs by semantic similarity to a query string.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="brave_local_status",
            description=(
                "Return current RAG index health: "
                "POI count, creation time, and TTL remaining."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "brave_local_index":
        index = await build_local_rag_index(
            query=arguments["query"],
            location=arguments.get("location", "Bradenton, FL"),
            radius=arguments.get("radius", 5000),
            strategy=strategy_for("rag_poi"),
            force=arguments.get("force", False),
        )
        return [types.TextContent(
            type="text",
            text=f"Indexed {len(index['docs'])} POIs. "
                 f"Created: {index.get('metadata', {}).get('created', 'unknown')}",
        )]

    if name == "brave_local_retrieve":
        hits = await retrieve_local(
            user_query=arguments["query"],
            top_k=arguments.get("top_k", 5),
            strategy=strategy_for("rag_poi"),
        )
        return [types.TextContent(type="text", text=json.dumps(hits, indent=2))]

    if name == "brave_local_status":
        if not DEFAULT_STORE.exists():
            return [types.TextContent(
                type="text",
                text="No index found. Call brave_local_index first.",
            )]
        data = json.loads(DEFAULT_STORE.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        return [types.TextContent(
            type="text",
            text=(
                f"POIs indexed: {len(data.get('docs', []))}\n"
                f"Created: {meta.get('created', 'unknown')}\n"
                f"Query: {meta.get('query', 'unknown')}\n"
                f"Location: {meta.get('location', 'unknown')}\n"
                f"Strategy: {meta.get('strategy', 'unknown')}"
            ),
        )]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
