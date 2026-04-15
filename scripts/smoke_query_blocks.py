#!/usr/bin/env python3
"""End-to-end query-block smoke test against a local Mnemosyne cluster.

Usage:
  MNEMOSYNE_FASTAPI_URL=http://127.0.0.1:8080 \
  MNEMOSYNE_DEV_USER_ID=dev-user-001 \
  MNEMOSYNE_DEV_TOKEN=dev-user-001 \
  uv run python scripts/smoke_query_blocks.py

This script creates a scratch graph, inserts a few RDF triples, writes a
document, inserts a queryBlock, and previews it through the MCP layer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from neem.mcp.server import create_standalone_mcp_server

DEFAULT_QUERY = (
    "PREFIX ex: <http://example.com/>\n"
    "SELECT ?s ?p ?o WHERE {\n"
    "  ?s ?p ?o .\n"
    "}\n"
    "LIMIT 10"
)


def _maybe_parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def decode_tool_result(result: Any) -> Any:
    """Normalize FastMCP call_tool outputs into plain Python data."""
    if isinstance(result, tuple) and len(result) == 2:
        content, meta = result
        if isinstance(meta, dict) and "result" in meta:
            return _maybe_parse_json(meta["result"])
        return decode_tool_result(content)

    if isinstance(result, list):
        if len(result) == 1 and hasattr(result[0], "text"):
            return _maybe_parse_json(result[0].text)
        return [
            _maybe_parse_json(item.text) if hasattr(item, "text") else item
            for item in result
        ]

    if hasattr(result, "text"):
        return _maybe_parse_json(result.text)

    return result


async def call_tool(server: Any, name: str, args: dict[str, Any]) -> Any:
    payload = decode_tool_result(await server.call_tool(name, args))
    print(f"[{name}]")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, indent=2)[:4000])
    else:
        print(payload)
    print()
    return payload


async def close_server_clients(server: Any) -> None:
    hp_client = getattr(server, "_hocuspocus_client", None)
    if hp_client is not None:
        await hp_client.close()

    job_stream = getattr(server, "_job_stream", None)
    if job_stream is not None:
        await job_stream.close()


async def run_smoke(*, graph_id: str, document_id: str, keep_graph: bool) -> dict[str, Any]:
    server = create_standalone_mcp_server()
    query_block_id: str | None = None

    try:
        await call_tool(server, "create_graph", {
            "graph_id": graph_id,
            "title": graph_id,
            "description": "local query-block smoke test",
        })

        await call_tool(server, "sparql_update", {
            "graph_id": graph_id,
            "sparql": (
                "PREFIX ex: <http://example.com/> "
                "INSERT DATA { "
                "ex:alice ex:knows ex:bob . "
                "ex:bob ex:knows ex:carol . "
                "}"
            ),
        })

        written = await call_tool(server, "write_document", {
            "graph_id": graph_id,
            "document_id": document_id,
            "content": "# Query Block Smoke\n\nSeed document.",
        })

        inserted = await call_tool(server, "insert_query_block", {
            "graph_id": graph_id,
            "document_id": document_id,
            "query": DEFAULT_QUERY,
            "comment": "Smoke test query block",
            "visualization": "network",
            "display_mode": "auto",
            "max_rows": 25,
        })
        if not isinstance(inserted, dict) or "block_id" not in inserted:
            raise RuntimeError(f"Unexpected insert_query_block result: {inserted!r}")
        query_block_id = inserted["block_id"]

        await call_tool(server, "get_query_block", {
            "graph_id": graph_id,
            "document_id": document_id,
            "block_id": query_block_id,
        })

        preview = await call_tool(server, "preview_query_block", {
            "graph_id": graph_id,
            "document_id": document_id,
            "block_id": query_block_id,
        })
        if not isinstance(preview, dict):
            raise RuntimeError(f"Unexpected preview_query_block result: {preview!r}")

        summary = {
            "graph_id": graph_id,
            "document_id": document_id,
            "query_block_id": query_block_id,
            "written_block_ids": written.get("block_ids") if isinstance(written, dict) else None,
            "resolved_display": preview.get("resolved_display"),
            "row_count": preview.get("profile", {}).get("row_count"),
            "network_nodes": len(preview.get("network_data", {}).get("nodes", [])),
            "network_edges": len(preview.get("network_data", {}).get("edges", [])),
        }
        print("[summary]")
        print(json.dumps(summary, indent=2))
        print()
        return summary
    finally:
        if not keep_graph:
            try:
                await call_tool(server, "delete_graph", {"graph_id": graph_id, "hard": True})
            except Exception as exc:
                print(f"[cleanup] failed to delete graph {graph_id}: {exc}")
                print()
        await close_server_clients(server)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-id",
        default=f"qb-smoke-{int(time.time())}",
        help="Scratch graph ID to create",
    )
    parser.add_argument(
        "--document-id",
        default="scratch-doc",
        help="Scratch document ID to create",
    )
    parser.add_argument(
        "--keep-graph",
        action="store_true",
        help="Keep the scratch graph instead of deleting it during cleanup",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_smoke(
            graph_id=args.graph_id,
            document_id=args.document_id,
            keep_graph=args.keep_graph,
        )
    )


if __name__ == "__main__":
    main()
