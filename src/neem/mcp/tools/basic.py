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

from neem.mcp.jobs import JobSubmitMetadata, RealtimeJobClient
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_dev_user_id, validate_token_and_load

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
            "Lists graphs via the Mnemosyne job queue. Returns streamed WebSocket events when available "
            "and falls back to HTTP polling otherwise."
        ),
    )
    async def list_graphs_tool(context: Context | None = None) -> str:
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        metadata = await submit_job(
            base_url=backend_config.base_url,
            token=token,
            task_type="list_graphs",
            payload={},
        )

        if context:
            await context.report_progress(10, 100)

        events = None
        if job_stream and metadata.links.websocket:
            events = await stream_job(job_stream, metadata, timeout=STREAM_TIMEOUT_SECONDS)

        if events:
            if context:
                await context.report_progress(80, 100)
            return _render_json({"job_id": metadata.job_id, "events": events})

        status_payload = (
            await poll_job_until_terminal(metadata.links.status, token)
            if metadata.links.status
            else None
        )
        result_payload = (
            await fetch_result(metadata.links.result, token)
            if metadata.links.result
            else None
        )

        response: JsonDict = {"job_id": metadata.job_id}
        if status_payload is not None:
            response["status"] = status_payload
        if result_payload is not None:
            response["result"] = result_payload

        if context:
            await context.report_progress(100, 100)
        return _render_json(response)


async def submit_job(
    base_url: str,
    token: str,
    task_type: str,
    payload: JsonDict,
) -> JobSubmitMetadata:
    """Submit a job through the generic `/jobs/` endpoint."""
    url = f"{base_url.rstrip('/')}/jobs/"
    data = await _request_json(
        method="POST",
        url=url,
        token=token,
        json_payload={"type": task_type, "payload": payload},
    )
    return JobSubmitMetadata.from_api(data, base_url=base_url)


async def poll_job_until_terminal(
    status_url: Optional[str],
    token: str,
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
                headers=_auth_headers(token),
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


async def fetch_result(result_url: Optional[str], token: str) -> Optional[JsonDict]:
    """Fetch the job result payload if a result URL is provided."""
    if not result_url:
        return None
    return await _request_json("GET", result_url, token=token)


async def stream_job(
    job_client: RealtimeJobClient,
    metadata: JobSubmitMetadata,
    *,
    timeout: float,
) -> Optional[list[JsonDict]]:
    """Wait for job events from the cache and return them."""
    job_id = metadata.job_id

    try:
        # Wait for the job to complete or timeout
        await asyncio.wait_for(
            job_client.wait_for_terminal(job_id, timeout=timeout),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out waiting for job completion",
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
    token: str,
    *,
    json_payload: Optional[JsonDict] = None,
) -> JsonDict:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.request(
            method,
            url,
            headers=_auth_headers(token),
            json=json_payload,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()


def _auth_headers(token: str) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    dev_user = get_dev_user_id()
    if dev_user:
        headers["X-User-ID"] = dev_user
    return headers


def _render_json(payload: JsonDict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
