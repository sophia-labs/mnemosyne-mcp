"""
Graph operation tools for SPARQL queries, updates, and management.

These tools provide workflow-based abstractions over the FastAPI backend,
optimized for token efficiency and ergonomic AI interaction.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Literal, Optional

import httpx
from mcp.server.fastmcp import Context, FastMCP

from neem.mcp.auth import MCPAuthContext
from neem.mcp.jobs import JobSubmitMetadata, RealtimeJobClient
from neem.mcp.utils.response_filters import (
    extract_result_from_job_detail,
    filter_graph_metadata,
    filter_job_status,
    filter_query_results,
)
from neem.mcp.utils.token_utils import estimate_tokens, render_compact_json, render_pretty_json
from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("mcp.tools.graph_operations")

HTTP_TIMEOUT = 30.0
DEFAULT_WAIT_MS = 2000
MAX_POLL_ATTEMPTS = 10
MAX_QUERY_RESULTS = 100

JsonDict = Dict[str, Any]


def register_graph_tools(server: FastMCP) -> None:
    """
    Register graph operation tools.

    Tools:
    - query_graph: Execute SPARQL queries with result filtering
    - update_graph: Execute SPARQL updates (INSERT/DELETE)
    - manage_graph: Polymorphic graph management (read/delete/stats)
    - create_graph: Create new knowledge graph
    """
    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping graph tool registration")
        return

    # Get the WebSocket streaming client (if available)
    job_stream = getattr(server, "_job_stream", None)
    if backend_config.has_websocket and job_stream is None:
        logger.warning(
            "Backend provides WebSocket but no realtime client available",
            extra_context={"websocket_url": backend_config.websocket_url},
        )

    @server.tool(
        name="query_graph",
        description="Execute SPARQL query on knowledge graph, return filtered results (max 100 rows)",
    )
    async def query_graph_tool(
        sparql: str,
        max_results: int = 10,
        result_format: Literal["json", "csv", "xml"] = "json",
        context: Context | None = None,
    ) -> str:
        """
        Execute a SPARQL SELECT/CONSTRUCT/ASK/DESCRIBE query.

        Args:
            sparql: SPARQL query string
            max_results: Maximum number of results to return (1-100, default 10)
            result_format: Response format (json, csv, xml)
            context: MCP context for progress reporting

        Returns:
            Filtered query results in compact JSON format
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        # Validate max_results
        if max_results < 1 or max_results > MAX_QUERY_RESULTS:
            return render_pretty_json({
                "error": f"max_results must be between 1 and {MAX_QUERY_RESULTS}",
                "hint": f"Requested {max_results}, allowed range: 1-{MAX_QUERY_RESULTS}",
            })

        if context:
            await context.report_progress(5, 100)

        # Submit query job
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.post(
                    f"{backend_config.base_url}/graphs/query",
                    headers=auth.http_headers(),
                    json={
                        "sparql": sparql,
                        "result_format": result_format,
                        "max_rows": max_results,
                    },
                )
                response.raise_for_status()
                job_data = response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            return render_pretty_json({
                "error": "Query submission failed",
                "status_code": e.response.status_code,
                "detail": error_detail,
                "hint": "Check SPARQL syntax. Example: SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10",
            })
        except Exception as e:
            return render_pretty_json({"error": str(e)})

        if context:
            await context.report_progress(20, 100)

        job_id = job_data.get("job_id")
        poll_url = job_data.get("poll_url", f"{backend_config.base_url}/graphs/jobs/{job_id}")

        status_payload = await _await_job_status(
            job_stream=job_stream,
            job_id=job_id,
            poll_url=poll_url,
            auth=auth,
            wait_ms=DEFAULT_WAIT_MS,
        )

        if context:
            await context.report_progress(80, 100)

        if not status_payload or status_payload.get("status") != "succeeded":
            error_msg = status_payload.get("error") if status_payload else "Job did not complete"
            return render_pretty_json({
                "error": error_msg,
                "job": filter_job_status(status_payload, include_debug=True) if status_payload else {},
                "hint": "Check SPARQL syntax and graph permissions",
            })

        # Extract and filter results
        raw_results = extract_result_from_job_detail(status_payload.get("detail", {}))

        if raw_results is None:
            result_url = job_data.get("result_url")
            if result_url:
                raw_results = await _fetch_result(result_url, auth)

        if raw_results is None:
            return render_pretty_json({
                "error": "No results available",
                "job": filter_job_status(status_payload, include_debug=True),
            })

        # Filter results for token efficiency
        if isinstance(raw_results, list):
            filtered_results = filter_query_results(raw_results, max_results=max_results)
        else:
            filtered_results = raw_results

        # Log token savings
        raw_json = render_compact_json(raw_results)
        filtered_json = render_compact_json(filtered_results)
        raw_tokens = estimate_tokens(raw_json)
        filtered_tokens = estimate_tokens(filtered_json)

        logger.info(
            "Query results optimized",
            extra_context={
                "raw_tokens": raw_tokens,
                "filtered_tokens": filtered_tokens,
                "result_count": len(filtered_results) if isinstance(filtered_results, list) else 1,
                "duration_ms": status_payload.get("processing_time_ms"),
            },
        )

        if context:
            await context.report_progress(100, 100)

        return render_compact_json({
            "results": filtered_results,
            "count": len(filtered_results) if isinstance(filtered_results, list) else 1,
            "duration_ms": status_payload.get("processing_time_ms"),
        })

    @server.tool(
        name="update_graph",
        description="Execute SPARQL update (INSERT/DELETE) on knowledge graph",
    )
    async def update_graph_tool(
        sparql: str,
        context: Context | None = None,
    ) -> str:
        """
        Execute a SPARQL INSERT/DELETE/UPDATE operation.

        Args:
            sparql: SPARQL update string
            context: MCP context for progress reporting

        Returns:
            Success confirmation with affected triple count
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if context:
            await context.report_progress(5, 100)

        # Submit update job
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.post(
                    f"{backend_config.base_url}/graphs/update",
                    headers=auth.http_headers(),
                    json={"sparql": sparql},
                )
                response.raise_for_status()
                job_data = response.json()
        except httpx.HTTPStatusError as e:
            return render_pretty_json({
                "error": "Update submission failed",
                "status_code": e.response.status_code,
                "detail": e.response.text,
                "hint": "Check SPARQL syntax. Example: INSERT DATA { <s> <p> <o> }",
            })
        except Exception as e:
            return render_pretty_json({"error": str(e)})

        if context:
            await context.report_progress(20, 100)

        job_id = job_data.get("job_id")
        poll_url = job_data.get("poll_url", f"{backend_config.base_url}/graphs/jobs/{job_id}")

        status_payload = await _await_job_status(
            job_stream=job_stream,
            job_id=job_id,
            poll_url=poll_url,
            auth=auth,
            wait_ms=DEFAULT_WAIT_MS,
        )

        if context:
            await context.report_progress(90, 100)

        if not status_payload:
            return render_pretty_json({"error": "Update job did not complete"})

        if status_payload.get("status") != "succeeded":
            return render_pretty_json({
                "error": status_payload.get("error", "Update failed"),
                "job": filter_job_status(status_payload, include_debug=True),
            })

        # Extract result details
        detail = status_payload.get("detail", {})
        result_data = extract_result_from_job_detail(detail)

        if context:
            await context.report_progress(100, 100)

        return render_compact_json({
            "status": "success",
            "result": result_data or {},
            "duration_ms": status_payload.get("processing_time_ms"),
        })

    @server.tool(
        name="manage_graph",
        description="Manage knowledge graph: read metadata, get stats, or delete graph",
    )
    async def manage_graph_tool(
        graph_id: str,
        action: Literal["read", "stats", "delete"],
        context: Context | None = None,
    ) -> str:
        """
        Polymorphic graph management tool.

        Args:
            graph_id: Graph identifier
            action: Operation to perform (read/stats/delete)
            context: MCP context for progress reporting

        Returns:
            Action-specific results in compact JSON
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if context:
            await context.report_progress(10, 100)

        # Map action to endpoint
        endpoint_map = {
            "read": f"/graphs/{graph_id}",
            "stats": f"/graphs/{graph_id}/stats",
            "delete": f"/graphs/{graph_id}",
        }
        method_map = {
            "read": "GET",
            "stats": "GET",
            "delete": "DELETE",
        }

        endpoint = endpoint_map[action]
        method = method_map[action]

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.request(
                    method,
                    f"{backend_config.base_url}{endpoint}",
                    headers=auth.http_headers(),
                    params={"wait_ms": DEFAULT_WAIT_MS} if method == "GET" else None,
                )
                response.raise_for_status()
                job_data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return render_pretty_json({
                    "error": f"Graph '{graph_id}' not found",
                    "hint": "Use list_graphs to see available graphs",
                })
            return render_pretty_json({
                "error": f"Graph {action} failed",
                "status_code": e.response.status_code,
                "detail": e.response.text,
            })
        except Exception as e:
            return render_pretty_json({"error": str(e)})

        if context:
            await context.report_progress(30, 100)

        # Poll for result if async job
        if "job_id" in job_data:
            job_id = job_data["job_id"]
            poll_url = job_data.get("poll_url", f"{backend_config.base_url}/graphs/jobs/{job_id}")

            status_payload = await _await_job_status(
                job_stream=job_stream,
                job_id=job_id,
                poll_url=poll_url,
                auth=auth,
                wait_ms=DEFAULT_WAIT_MS,
            )

            if not status_payload or status_payload.get("status") != "succeeded":
                return render_pretty_json({
                    "error": f"Graph {action} operation failed",
                    "job": filter_job_status(status_payload, include_debug=True) if status_payload else {},
                })

            result_data = extract_result_from_job_detail(status_payload.get("detail", {}))
        else:
            result_data = job_data

        if context:
            await context.report_progress(90, 100)

        # Format response based on action
        if action == "read" and isinstance(result_data, dict):
            filtered_data = filter_graph_metadata(result_data, verbose=True)
            if context:
                await context.report_progress(100, 100)
            return render_compact_json(filtered_data)
        elif action == "delete":
            if context:
                await context.report_progress(100, 100)
            return render_compact_json({
                "status": "deleted",
                "graph_id": graph_id,
            })
        else:
            if context:
                await context.report_progress(100, 100)
            return render_compact_json(result_data or {})

    @server.tool(
        name="create_graph",
        description="Create new knowledge graph with title and optional description",
    )
    async def create_graph_tool(
        graph_id: str,
        title: str,
        description: str = "",
        context: Context | None = None,
    ) -> str:
        """
        Create a new knowledge graph.

        Args:
            graph_id: Unique graph identifier (alphanumeric, dashes, underscores)
            title: Human-readable graph title
            description: Optional graph description
            context: MCP context for progress reporting

        Returns:
            Created graph metadata in compact JSON
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        # Validate graph_id format
        if not graph_id or len(graph_id) > 128:
            return render_pretty_json({
                "error": "Invalid graph_id",
                "hint": "Must be 1-128 characters, alphanumeric with dashes/underscores only",
            })

        if context:
            await context.report_progress(5, 100)

        # Submit creation job
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.post(
                    f"{backend_config.base_url}/graphs",
                    headers=auth.http_headers(),
                    json={
                        "graph_id": graph_id,
                        "title": title,
                        "description": description,
                    },
                )
                response.raise_for_status()
                job_data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                return render_pretty_json({
                    "error": f"Graph '{graph_id}' already exists",
                    "hint": "Use a different graph_id or delete the existing graph first",
                })
            return render_pretty_json({
                "error": "Graph creation failed",
                "status_code": e.response.status_code,
                "detail": e.response.text,
            })
        except Exception as e:
            return render_pretty_json({"error": str(e)})

        if context:
            await context.report_progress(20, 100)

        job_id = job_data.get("job_id")
        poll_url = job_data.get("poll_url", f"{backend_config.base_url}/graphs/jobs/{job_id}")

        status_payload = await _await_job_status(
            job_stream=job_stream,
            job_id=job_id,
            poll_url=poll_url,
            auth=auth,
            wait_ms=DEFAULT_WAIT_MS,
        )

        if context:
            await context.report_progress(90, 100)

        if not status_payload or status_payload.get("status") != "succeeded":
            return render_pretty_json({
                "error": "Graph creation failed",
                "job": filter_job_status(status_payload, include_debug=True) if status_payload else {},
            })

        if context:
            await context.report_progress(100, 100)

        return render_compact_json({
            "status": "created",
            "graph_id": graph_id,
            "title": title,
        })

    @server.tool(
        name="cancel_job",
        description="Cancel a queued job. Only jobs that haven't started running can be cancelled.",
    )
    async def cancel_job_tool(
        job_id: str,
        context: Context | None = None,
    ) -> str:
        """
        Cancel a queued job.

        Only jobs that are still in the QUEUED state can be cancelled.
        Jobs that have already started running cannot be cancelled - they will
        run to completion.

        Args:
            job_id: The job ID to cancel (e.g., "job-abc123...")
            context: MCP context

        Returns:
            Cancellation result with previous job status
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.delete(
                    f"{backend_config.base_url}/graphs/jobs/{job_id}",
                    headers=auth.http_headers(),
                )
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return render_pretty_json({
                    "error": "Job not found",
                    "job_id": job_id,
                    "hint": "Job may have already completed or does not exist",
                })
            return render_pretty_json({
                "error": "Failed to cancel job",
                "status_code": e.response.status_code,
                "detail": e.response.text,
            })
        except Exception as e:
            return render_pretty_json({"error": str(e)})

        return render_compact_json({
            "job_id": result.get("job_id", job_id),
            "cancelled": result.get("cancelled", False),
            "previous_status": result.get("previous_status"),
            "message": result.get("message"),
        })


async def _await_job_status(
    *,
    job_stream: Optional[RealtimeJobClient],
    job_id: str,
    poll_url: Optional[str],
    auth: MCPAuthContext,
    wait_ms: int,
    stream_timeout: float = 60.0,
) -> Optional[JsonDict]:
    """Wait for job completion using the per-user stream with HTTP fallback."""
    if job_stream:
        status_payload = await job_stream.wait_for_status(job_id, timeout=stream_timeout)
        if status_payload:
            return status_payload

    if poll_url:
        return await _poll_job_until_terminal(poll_url, auth, wait_ms=wait_ms)

    return None


async def _poll_job_until_terminal(
    status_url: str,
    auth: MCPAuthContext,
    *,
    wait_ms: int = DEFAULT_WAIT_MS,
) -> Optional[JsonDict]:
    """Poll job status endpoint until terminal state."""
    attempt = 0
    last_payload: Optional[JsonDict] = None

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        while attempt < MAX_POLL_ATTEMPTS:
            attempt += 1
            try:
                resp = await client.get(
                    status_url,
                    headers=auth.http_headers(),
                    params={"wait_ms": wait_ms},
                )
                resp.raise_for_status()
                payload = resp.json()
                last_payload = payload

                status = (payload.get("status") or "").lower()
                if status in {"succeeded", "failed"}:
                    return payload

                await asyncio.sleep(min(1.0 * attempt, 3.0))
            except Exception as e:
                logger.warning(
                    "Job polling attempt failed",
                    extra_context={"attempt": attempt, "error": str(e)},
                )
                await asyncio.sleep(2.0)

    return last_payload


async def _fetch_result(result_url: str, auth: MCPAuthContext) -> Optional[JsonDict]:
    """Fetch job result payload."""
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(result_url, headers=auth.http_headers())
            response.raise_for_status()
            if not response.content:
                return None
            return response.json()
    except Exception as e:
        logger.warning("Failed to fetch job result", extra_context={"error": str(e)})
        return None
