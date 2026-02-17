"""
Proof-of-concept MCP tools that exercise the backend job queue + WebSockets.

The intent is to keep these helpers small and debuggable while we figure out
the richer tool surface. Once the new APIs solidify we can replace this module
with proper graph/query tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import Context, FastMCP

from neem.mcp.auth import MCPAuthContext
from neem.mcp.jobs import JobSubmitMetadata, RealtimeJobClient, WebSocketConnectionError
from neem.mcp.trace import trace, trace_separator
from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("mcp.tools.basic")

HTTP_TIMEOUT = 30.0
DEFAULT_WAIT_MS = 1500
MAX_POLL_ATTEMPTS = 6
STREAM_TIMEOUT_SECONDS = 60.0
RACE_TIMEOUT_SECONDS = 15.0

# Retry settings for transient server errors (5xx)
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5  # seconds; doubles each attempt: 0.5, 1.0, 2.0

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
        trace_separator("list_graphs CALLED")
        trace("include_deleted=%s, has_context=%s" % (include_deleted, context is not None))

        trace("Step 1: Resolving auth...")
        auth = MCPAuthContext.from_context(context)
        trace("Auth resolved", {
            "source": auth.source,
            "has_token": bool(auth.token),
            "token_prefix": (auth.token[:20] + "...") if auth.token else None,
            "user_id": auth.user_id,
            "has_internal_secret": bool(auth.internal_service_secret),
        })
        auth.require_auth()
        trace("Auth validated OK")

        # Pre-connect WebSocket so subscribe is near-instant after job submit
        trace("Step 2: Pre-connecting WebSocket...")
        if job_stream:
            try:
                await job_stream.ensure_ready()
                trace("WebSocket pre-connected OK")
            except Exception as exc:
                trace("WebSocket pre-connect failed (will use polling): %s" % exc)

        trace("Step 3: Submitting job to %s" % backend_config.base_url)
        metadata = await submit_job(
            base_url=backend_config.base_url,
            auth=auth,
            task_type="list_graphs",
            payload={},
        )
        trace("Job submitted", {
            "job_id": metadata.job_id,
            "status": metadata.status,
            "trace_id": metadata.trace_id,
            "links.status": metadata.links.status,
            "links.result": metadata.links.result,
            "links.websocket": metadata.links.websocket,
        })

        if context:
            await context.report_progress(10, 100)

        # Race WS streaming against HTTP polling
        trace("Step 4: Racing WS + poll concurrently")
        ws_events, poll_payload = await await_job_completion(
            job_stream, metadata, auth, timeout=RACE_TIMEOUT_SECONDS,
        )

        if context:
            await context.report_progress(80, 100)

        # Try to extract graphs from WebSocket events first
        if ws_events:
            trace("Step 5: Extracting graphs from %d WS events" % len(ws_events))
            for i, ev in enumerate(ws_events):
                trace("  event[%d]" % i, ev)
            graphs = _extract_graphs_from_events(ws_events)
            if graphs is not None:
                graphs = _filter_deleted_graphs(graphs, include_deleted)
                trace("SUCCESS: Got %d graphs from WS events" % len(graphs))
                if context:
                    await context.report_progress(100, 100)
                return _render_json({"graphs": graphs, "count": len(graphs)})
            trace("WARN: Could not extract graphs from WS events")

        # Try poll result
        if poll_payload:
            trace("Step 5: Extracting graphs from poll response")
            graphs = _extract_graphs_from_status(poll_payload)
            if graphs is not None:
                graphs = _filter_deleted_graphs(graphs, include_deleted)
                trace("SUCCESS: Got %d graphs from polling" % len(graphs))
                if context:
                    await context.report_progress(100, 100)
                return _render_json({"graphs": graphs, "count": len(graphs)})

        if context:
            await context.report_progress(100, 100)

        # Fallback: return error with debug info
        trace("FAIL: Could not extract graphs from any source")
        return _render_json({
            "error": "Failed to extract graph list from job result",
        })


async def submit_job(
    base_url: str,
    auth: MCPAuthContext,
    task_type: str,
    payload: JsonDict,
) -> JobSubmitMetadata:
    """Submit a job through the generic `/jobs/` endpoint."""
    url = f"{base_url.rstrip('/')}/jobs/"
    trace("  submit_job POST %s" % url, {"type": task_type, "payload": payload})
    data = await _request_json(
        method="POST",
        url=url,
        auth=auth,
        json_payload={"type": task_type, "payload": payload},
    )
    trace("  submit_job response", data)
    return JobSubmitMetadata.from_api(data, base_url=base_url)


async def poll_job_until_terminal(
    status_url: Optional[str],
    auth: MCPAuthContext,
    wait_ms: int = DEFAULT_WAIT_MS,
) -> Optional[JsonDict]:
    """Poll the job status endpoint until it reaches a terminal state."""
    if not status_url:
        trace("  poll: no status_url, returning None")
        return None

    trace("  poll: starting (max %d attempts, wait_ms=%d)" % (MAX_POLL_ATTEMPTS, wait_ms))
    attempt = 0
    last_payload: Optional[JsonDict] = None
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        while attempt < MAX_POLL_ATTEMPTS:
            attempt += 1
            trace("  poll: attempt %d/%d GET %s" % (attempt, MAX_POLL_ATTEMPTS, status_url))
            resp = await client.get(
                status_url,
                headers=auth.http_headers(),
                params={"wait_ms": wait_ms},
            )
            trace("  poll: HTTP %d" % resp.status_code)
            resp.raise_for_status()
            payload = resp.json()
            last_payload = payload
            status = (payload.get("status") or "").lower()
            trace("  poll: status=%s" % status, payload)
            if status in {"succeeded", "failed"}:
                trace("  poll: terminal status reached")
                return payload
            sleep_time = min(1.0 * attempt, 5.0)
            trace("  poll: not terminal, sleeping %.1fs" % sleep_time)
            await asyncio.sleep(sleep_time)

    trace("  poll: exhausted all attempts, returning last payload")
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
        payload = event.get("payload", {})
        payload_status = payload.get("status", "") if isinstance(payload, dict) else ""

        is_success = (
            event_type in ("job_completed", "completed", "succeeded")
            or (event_type == "job_update" and payload_status == "succeeded")
        )
        if is_success:
            # Result might be in payload.detail.result_inline or directly in payload
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
    trace("  stream_job: job_id=%s, timeout=%.1fs" % (job_id, timeout))
    trace("  stream_job: client connected=%s, ws_url=%s" % (
        job_client.is_connected, job_client.websocket_url,
    ))

    try:
        # Wait for the job to complete or timeout
        trace("  stream_job: calling wait_for_terminal...")
        await asyncio.wait_for(
            job_client.wait_for_terminal(job_id, timeout=timeout),
            timeout=timeout,
        )
        trace("  stream_job: wait_for_terminal returned (job completed)")
    except WebSocketConnectionError as exc:
        trace("  stream_job: WebSocket CONNECTION FAILED: %s" % exc)
        logger.warning(
            "WebSocket connection failed; falling back to HTTP polling",
            extra_context={"job_id": job_id, "error": str(exc)},
        )
        return None
    except asyncio.TimeoutError:
        trace("  stream_job: TIMED OUT after %.1fs" % timeout)
        logger.warning(
            "Timed out waiting for job completion via WebSocket",
            extra_context={"job_id": job_id},
        )
        # Still try to get whatever events we have
        pass

    # Get all events from cache
    trace("  stream_job: fetching events from cache...")
    events = await job_client.get_events(job_id)
    trace("  stream_job: got %d events from cache" % len(events))
    if not events:
        logger.debug("No events found in cache", extra_context={"job_id": job_id})
        trace("  stream_job: returning None (no events)")
        return None

    # Convert JobEvent objects to raw JSON dicts
    raw_events = [event.raw for event in events]
    for i, ev in enumerate(raw_events):
        trace("  stream_job: event[%d]" % i, ev)
    return raw_events


async def await_job_completion(
    job_stream: Optional[RealtimeJobClient],
    metadata: JobSubmitMetadata,
    auth: MCPAuthContext,
    *,
    timeout: float = RACE_TIMEOUT_SECONDS,
) -> tuple[Optional[list[JsonDict]], Optional[JsonDict]]:
    """Race WS streaming against HTTP polling. Returns (ws_events, poll_payload).

    Tries both WS streaming and HTTP polling, preferring whichever yields usable
    terminal data first. Unlike a strict first-completed race, this keeps waiting
    when the first completed branch returns None/non-terminal payloads.
    """
    trace("  race: starting (timeout=%.1fs, has_ws=%s)" % (timeout, job_stream is not None))

    async def _do_stream() -> Optional[list[JsonDict]]:
        if not job_stream or not metadata.links.websocket:
            # Block forever so poll wins
            await asyncio.Event().wait()
            return None  # unreachable
        return await stream_job(job_stream, metadata, timeout=timeout)

    async def _do_poll() -> Optional[JsonDict]:
        if not metadata.links.status:
            return None
        return await poll_job_until_terminal(metadata.links.status, auth)

    ws_task = asyncio.create_task(_do_stream(), name="race-ws")
    poll_task = asyncio.create_task(_do_poll(), name="race-poll")

    ws_events: Optional[list[JsonDict]] = None
    poll_payload: Optional[JsonDict] = None
    pending: set[asyncio.Task[Any]] = {ws_task, poll_task}
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while pending:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break

        done_now, pending = await asyncio.wait(
            pending,
            timeout=remaining,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done_now:
            break

        for task in done_now:
            if task is ws_task:
                try:
                    ws_events = task.result()
                    trace("  race: WS completed (events=%s)" % (len(ws_events) if ws_events else None))
                except Exception as exc:
                    trace("  race: WS task failed: %s" % exc)
            elif task is poll_task:
                try:
                    poll_payload = task.result()
                    trace("  race: poll completed (status=%s)" % (
                        poll_payload.get("status") if poll_payload else None,
                    ))
                except Exception as exc:
                    trace("  race: poll task failed: %s" % exc)

        poll_status = (poll_payload.get("status") or "").lower() if isinstance(poll_payload, dict) else ""
        if poll_status in {"succeeded", "failed"}:
            # Terminal poll result is authoritative and usually includes detail.
            break

        # If one side returned None/empty quickly, keep waiting for the other side.
        if ws_task not in pending and poll_task in pending and ws_events is None:
            continue
        if poll_task not in pending and ws_task in pending and poll_payload is None:
            continue

        # Both sides have produced some signal (or one side finished and other unavailable).
        if ws_task not in pending and poll_task not in pending:
            break

    # Cancel any still-pending task(s)
    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    if pending and ws_events is None and poll_payload is None:
        trace("  race: timed out after %.1fs with no result" % timeout)

    return ws_events, poll_payload


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
    last_exc: Optional[Exception] = None
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_payload,
                )
                if response.status_code >= 500 and attempt < MAX_RETRIES:
                    delay = RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "mcp_api_request_retrying",
                        extra_context={
                            "status_code": response.status_code,
                            "attempt": attempt + 1,
                            "max_retries": MAX_RETRIES,
                            "delay_s": delay,
                            "url": url,
                        },
                    )
                    await asyncio.sleep(delay)
                    continue
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
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    delay = RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "mcp_api_request_timeout_retrying",
                        extra_context={
                            "attempt": attempt + 1,
                            "max_retries": MAX_RETRIES,
                            "delay_s": delay,
                            "url": url,
                        },
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
    # Should not reach here, but satisfy type checker
    raise last_exc or RuntimeError("_request_json exhausted retries")


def _render_json(payload: JsonDict) -> str:
    return json.dumps(payload, sort_keys=True)
