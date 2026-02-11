"""MCP tools for semantic vector search."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context, FastMCP

from neem.mcp.auth import MCPAuthContext
from neem.mcp.jobs import RealtimeJobClient
from neem.mcp.tools.basic import await_job_completion, submit_job
from neem.mcp.trace import trace
from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("mcp.tools.search")

STREAM_TIMEOUT_SECONDS = 60.0
JsonDict = Dict[str, Any]


def _render_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


def register_search_tools(server: FastMCP) -> None:
    """Register semantic search MCP tools."""

    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping search tool registration")
        return

    job_stream: Optional[RealtimeJobClient] = getattr(server, "_job_stream", None)

    @server.tool(
        name="semantic_search",
        title="Semantic Search",
        description=(
            "Search for semantically similar blocks across documents in a graph "
            "using vector embeddings. Returns matching blocks ranked by similarity "
            "score, with document titles and text previews.\n\n"
            "Use this for finding content by meaning rather than exact text match. "
            "Complements SPARQL text search (which is substring-based) with "
            "semantic understanding.\n\n"
            "Requires the embedding pipeline to be enabled and documents to have "
            "been indexed (happens automatically on document save)."
        ),
    )
    async def semantic_search_tool(
        graph_id: str,
        query: str,
        limit: int = 10,
        doc_filter: Optional[str] = None,
        min_score: float = 0.0,
        context: Context | None = None,
    ) -> str:
        """Search for semantically similar blocks."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not query or not query.strip():
            raise ValueError("query is required")

        payload: JsonDict = {
            "graph_id": graph_id.strip(),
            "query": query.strip(),
            "limit": min(max(limit, 1), 100),
        }
        if doc_filter:
            payload["doc_filter"] = doc_filter.strip()
        if min_score > 0.0:
            payload["min_score"] = min_score

        if job_stream:
            try:
                await job_stream.ensure_ready()
            except Exception:
                pass

        metadata = await submit_job(
            base_url=backend_config.base_url,
            auth=auth,
            task_type="semantic_search",
            payload=payload,
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(job_stream, metadata, context, auth)

        detail = result.get("detail", {})
        inline = detail.get("result_inline") if isinstance(detail, dict) else None

        if inline and isinstance(inline, dict):
            error = inline.get("error")
            if error:
                return _render_json({"error": error, "results": []})
            return _render_json(inline)

        if result.get("status") == "failed":
            return _render_json({
                "error": result.get("error", "Search failed"),
                "results": [],
            })

        return _render_json({"error": "Failed to get search results", "results": []})

    @server.tool(
        name="reindex_graph",
        title="Reindex Graph",
        description=(
            "Re-embed all documents in a graph. Used for initial embedding of "
            "existing graphs, after model upgrades, or recovery after vector "
            "store data loss. Enqueues a COMPUTE_EMBEDDINGS job for each document.\n\n"
            "This is an admin/maintenance operation — normal document edits "
            "are automatically indexed via the MATERIALIZE_DOC pipeline."
        ),
    )
    async def reindex_graph_tool(
        graph_id: str,
        context: Context | None = None,
    ) -> str:
        """Re-embed all documents in a graph."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")

        if job_stream:
            try:
                await job_stream.ensure_ready()
            except Exception:
                pass

        metadata = await submit_job(
            base_url=backend_config.base_url,
            auth=auth,
            task_type="reindex_graph",
            payload={"graph_id": graph_id.strip()},
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(job_stream, metadata, context, auth)

        detail = result.get("detail", {})
        inline = detail.get("result_inline") if isinstance(detail, dict) else None

        if inline and isinstance(inline, dict):
            error = inline.get("error")
            if error:
                return _render_json({"error": error})
            return _render_json(inline)

        if result.get("status") == "failed":
            return _render_json({"error": result.get("error", "Reindex failed")})

        return _render_json({"error": "Failed to get reindex result"})

    logger.info("Registered search tools (semantic_search, reindex_graph)")


async def _wait_for_job_result(
    job_stream: Optional[RealtimeJobClient],
    metadata: Any,
    context: Optional[Context],
    auth: MCPAuthContext,
) -> JsonDict:
    """Wait for job completion via WS+poll race."""
    trace("  _wait_for_job_result: racing WS + poll")
    ws_events, poll_payload = await await_job_completion(
        job_stream, metadata, auth, timeout=STREAM_TIMEOUT_SECONDS,
    )

    if ws_events:
        if context:
            await context.report_progress(80, 100)
        for event in reversed(ws_events):
            event_type = event.get("type", "")
            if event_type in ("job_completed", "completed", "succeeded"):
                if context:
                    await context.report_progress(100, 100)
                result: JsonDict = {"status": "succeeded", "events": len(ws_events)}
                payload = event.get("payload", {})
                if isinstance(payload, dict):
                    detail = payload.get("detail")
                    if detail:
                        result["detail"] = detail
                return result
            if event_type in ("failed", "error"):
                error = event.get("error", "Job failed")
                return {"status": "failed", "error": error}
        return {"status": "unknown", "event_count": len(ws_events)}

    if context:
        await context.report_progress(100, 100)

    if poll_payload:
        status = poll_payload.get("status", "unknown")
        detail = poll_payload.get("detail")
        if status == "failed":
            error = poll_payload.get("error") or (
                detail.get("error") if isinstance(detail, dict) else None
            )
            return {"status": "failed", "error": error}
        result = {"status": status}
        if detail:
            result["detail"] = detail
        return result

    return {"status": "unknown"}
