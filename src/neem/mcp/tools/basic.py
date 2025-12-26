"""
Proof-of-concept MCP tools that exercise the backend job queue + WebSockets.

The intent is to keep these helpers small and debuggable while we figure out
the richer tool surface. Once the new APIs solidify we can replace this module
with proper graph/query tools.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import Context, FastMCP

from neem.mcp.auth import MCPAuthContext
from neem.mcp.jobs import JobSubmitMetadata, RealtimeJobClient, WebSocketConnectionError
from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("mcp.tools.basic")

HTTP_TIMEOUT = 30.0
DEFAULT_WAIT_MS = 1500
MAX_POLL_ATTEMPTS = 6
STREAM_TIMEOUT_SECONDS = 60.0

JsonDict = Dict[str, Any]


def register_basic_tools(server: FastMCP) -> None:
    """
    Register the experimental `list_graphs` tool.

    The tool intentionally uses the generic `/jobs/` endpoint so we can validate
    the push channel supplied via `links.websocket`.
    """

    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping tool registration")
        return

    job_stream: Optional[RealtimeJobClient] = getattr(server, "_job_stream", None)
    if backend_config.has_websocket and job_stream is None:
        logger.warning(
            "Backend provides a WebSocket hint but no realtime client is available",
            extra_context={"websocket_url": backend_config.websocket_url},
        )

    @server.tool(
        name="list_graphs",
        title="List Mnemosyne Graphs",
        description=(
            "Lists all knowledge graphs owned by the authenticated user. "
            "Returns graph metadata including IDs, titles, and timestamps."
        ),
    )
    async def list_graphs_tool(
        include_deleted: bool = False,
        context: Context | None = None,
    ) -> str:
        """List graphs owned by the authenticated user.

        Args:
            include_deleted: If False (default), excludes graphs with status 'deleted'.
                            Set to True to include soft-deleted graphs.
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        metadata = await submit_job(
            base_url=backend_config.base_url,
            auth=auth,
            task_type="list_graphs",
            payload={},
        )

        if context:
            await context.report_progress(10, 100)

        events = None
        if job_stream and metadata.links.websocket:
            events = await stream_job(job_stream, metadata, timeout=STREAM_TIMEOUT_SECONDS)

        # Try to extract graphs from WebSocket events first
        if events:
            if context:
                await context.report_progress(80, 100)
            graphs = _extract_graphs_from_events(events)
            if graphs is not None:
                graphs = _filter_deleted_graphs(graphs, include_deleted)
                return _render_json({"graphs": graphs, "count": len(graphs)})
            # Fall back to raw events if extraction failed
            return _render_json({"job_id": metadata.job_id, "events": events})

        # Poll for job completion and extract result
        status_payload = (
            await poll_job_until_terminal(metadata.links.status, auth)
            if metadata.links.status
            else None
        )

        if context:
            await context.report_progress(100, 100)

        # Extract graphs from the job's detail.result_inline field
        graphs = _extract_graphs_from_status(status_payload)
        if graphs is not None:
            graphs = _filter_deleted_graphs(graphs, include_deleted)
            return _render_json({"graphs": graphs, "count": len(graphs)})

        # Fallback: return error with debug info
        return _render_json({
            "error": "Failed to extract graph list from job result",
            "job_id": metadata.job_id,
            "status": status_payload.get("status") if status_payload else None,
        })


async def submit_job(
    base_url: str,
    auth: MCPAuthContext,
    task_type: str,
    payload: JsonDict,
) -> JobSubmitMetadata:
    """Submit a job through the generic `/jobs/` endpoint."""
    url = f"{base_url.rstrip('/')}/jobs/"
    data = await _request_json(
        method="POST",
        url=url,
        auth=auth,
        json_payload={"type": task_type, "payload": payload},
    )
    return JobSubmitMetadata.from_api(data, base_url=base_url)


async def poll_job_until_terminal(
    status_url: Optional[str],
    auth: MCPAuthContext,
    wait_ms: int = DEFAULT_WAIT_MS,
) -> Optional[JsonDict]:
    """Poll the job status endpoint until it reaches a terminal state."""
    if not status_url:
        return None

    attempt = 0
    last_payload: Optional[JsonDict] = None
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        while attempt < MAX_POLL_ATTEMPTS:
            attempt += 1
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
            await asyncio.sleep(min(1.0 * attempt, 5.0))

    return last_payload


def _extract_graphs_from_status(status_payload: Optional[JsonDict]) -> Optional[list]:
    """Extract graph list from job status detail.result_inline field."""
    if not status_payload:
        return None

    detail = status_payload.get("detail")
    if not isinstance(detail, dict):
        return None

    result_inline = detail.get("result_inline")
    if isinstance(result_inline, list):
        return result_inline

    return None


def _filter_deleted_graphs(graphs: list, include_deleted: bool) -> list:
    """Filter out soft-deleted graphs unless include_deleted is True."""
    if include_deleted:
        return graphs
    return [g for g in graphs if g.get("status") != "deleted"]


def _extract_graphs_from_events(events: list[JsonDict]) -> Optional[list]:
    """Extract graph list from WebSocket job events."""
    for event in reversed(events):  # Check most recent events first
        event_type = event.get("type", "")
        if event_type in ("job_completed", "completed", "succeeded"):
            # Result might be in payload.detail.result_inline or directly in payload
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                detail = payload.get("detail", {})
                if isinstance(detail, dict):
                    result_inline = detail.get("result_inline")
                    if isinstance(result_inline, list):
                        return result_inline
            # Or result might be at event top level
            if "result_inline" in event:
                result_inline = event.get("result_inline")
                if isinstance(result_inline, list):
                    return result_inline
    return None


async def stream_job(
    job_client: RealtimeJobClient,
    metadata: JobSubmitMetadata,
    *,
    timeout: float,
) -> Optional[list[JsonDict]]:
    """Wait for job events from the cache and return them.

    Returns None if WebSocket connection fails, allowing fallback to polling.
    """
    job_id = metadata.job_id

    try:
        # Wait for the job to complete or timeout
        await asyncio.wait_for(
            job_client.wait_for_terminal(job_id, timeout=timeout),
            timeout=timeout,
        )
    except WebSocketConnectionError as exc:
        logger.warning(
            "WebSocket connection failed; falling back to HTTP polling",
            extra_context={"job_id": job_id, "error": str(exc)},
        )
        return None
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out waiting for job completion via WebSocket",
            extra_context={"job_id": job_id},
        )
        # Still try to get whatever events we have
        pass

    # Get all events from cache
    events = await job_client.get_events(job_id)
    if not events:
        logger.debug("No events found in cache", extra_context={"job_id": job_id})
        return None

    # Convert JobEvent objects to raw JSON dicts
    return [event.raw for event in events]


async def _request_json(
    method: str,
    url: str,
    auth: MCPAuthContext,
    *,
    json_payload: Optional[JsonDict] = None,
) -> JsonDict:
    headers = auth.http_headers()
    # Log headers being sent (redact secrets)
    safe_headers = {
        k: (v[:20] + "..." if len(v) > 20 else v) if k.lower() not in ("authorization", "x-internal-service") else "[REDACTED]"
        for k, v in headers.items()
    }
    logger.info(
        "mcp_api_request",
        extra_context={
            "method": method,
            "url": url,
            "headers": safe_headers,
            "auth_source": auth.source,
            "has_token": bool(auth.token),
            "has_user_id": bool(auth.user_id),
            "has_internal_secret": bool(auth.internal_service_secret),
        },
    )
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.request(
            method,
            url,
            headers=headers,
            json=json_payload,
        )
        if response.status_code >= 400:
            logger.error(
                "mcp_api_request_failed",
                extra_context={
                    "status_code": response.status_code,
                    "response_text": response.text[:500] if response.text else None,
                },
            )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()


def _render_json(payload: JsonDict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
