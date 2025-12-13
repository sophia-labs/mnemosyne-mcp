#!/usr/bin/env python3
"""
Mnemosyne MCP benchmark harness (Phase 1 foundations).

Implements the baseline harness described in docs/mcp-benchmark-spec.ttl:
 - CLI configurable load (duration, users, workers, concurrency, RPS, mix)
 - Backend + auth resolution shared with MCP tools
 - Optional per-user WebSocket streaming with polling fallback
 - Per-call metrics collection (latency, backend status, ttfb, transport path)
 - JSON/NDJSON export plus percentile summaries

Usage examples:
  uv run scripts/bench_mcp.py --duration 30 --users 1 --workers 2
  uv run scripts/bench_mcp.py --duration 20 --no-ws --mix "list_graphs:0.5,query_graph:0.5"
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from urllib.parse import urljoin

import httpx
import typer

from neem.mcp.jobs import JobLinks, JobSubmitMetadata, RealtimeJobClient
from neem.mcp.server.standalone_server import resolve_backend_config
from neem.mcp.tools.basic import fetch_result
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_dev_user_id, validate_token_and_load

JsonDict = Dict[str, Any]
ToolName = Literal["list_graphs", "query_graph"]

SUPPORTED_TOOLS: Tuple[ToolName, ...] = ("list_graphs", "query_graph")
HTTP_TIMEOUT = 30.0
STREAM_TIMEOUT_SECONDS = 60.0
MAX_POLL_ATTEMPTS = 12

app = typer.Typer(add_completion=False, help="Mnemosyne MCP benchmark harness (Phase 1 baseline).")


@dataclass
class BenchmarkConfig:
    duration: float
    users: int
    workers_per_user: int
    concurrency: int
    rps_per_worker: float
    mix_weights: Dict[ToolName, float]
    sparql: str
    enable_ws: bool
    wait_ms: int
    ws_cache_ttl: float
    ws_cache_size: int
    visualize: bool
    output_path: Optional[Path]
    ndjson_path: Optional[Path]
    log_level: str
    seed: Optional[int]


@dataclass
class ToolCallRecord:
    tool: ToolName
    user_id: str
    job_id: Optional[str]
    trace_id: Optional[str]
    started_at: datetime
    completed_at: datetime
    latency_ms: float
    backend_status: Optional[str]
    http_status: Optional[int]
    ok: bool
    error_category: Optional[str]
    error_detail: Optional[str]
    path: Literal["stream", "poll"]
    poll_attempts: int
    backend_processing_time_ms: Optional[float]
    ttfb_ms: Optional[float]
    mix_weight: float

    def to_dict(self) -> JsonDict:
        return {
            "tool": self.tool,
            "user_id": self.user_id,
            "job_id": self.job_id,
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "latency_ms": self.latency_ms,
            "backend_status": self.backend_status,
            "http_status": self.http_status,
            "ok": self.ok,
            "error_category": self.error_category,
            "error_detail": self.error_detail,
            "path": self.path,
            "poll_attempts": self.poll_attempts,
            "backend_processing_time_ms": self.backend_processing_time_ms,
            "ttfb_ms": self.ttfb_ms,
            "mix_weight": self.mix_weight,
        }


@dataclass
class UserContext:
    index: int
    user_id: str
    token: str
    job_stream: Optional[RealtimeJobClient]
    mix_weight_lookup: Dict[ToolName, float]


class WorkloadSelector:
    """Weighted random selector for tool invocations."""

    def __init__(self, weights: Dict[ToolName, float]) -> None:
        self._total = sum(weights.values())
        cumulative: List[Tuple[float, ToolName]] = []
        running = 0.0
        for tool, weight in weights.items():
            running += weight
            cumulative.append((running, tool))
        self._cumulative = cumulative

    def choose(self) -> ToolName:
        target = random.random() * self._total
        for threshold, tool in self._cumulative:
            if target <= threshold:
                return tool
        return self._cumulative[-1][1]


class BenchmarkHarness:
    """Coordinates multi-user workloads, tool execution, and metrics aggregation."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.backend = resolve_backend_config()
        self.logger = LoggerFactory.get_logger("bench", base_context={"component": "bench"})
        self._records: List[ToolCallRecord] = []
        self._records_lock = asyncio.Lock()
        self._concurrency = asyncio.Semaphore(config.concurrency)
        self._stop_event = asyncio.Event()
        self._selector = WorkloadSelector(config.mix_weights)
        self._user_contexts: List[UserContext] = []

    async def run(self) -> List[ToolCallRecord]:
        """Run the benchmark to completion."""
        await self._initialize_users()
        end_time = time.monotonic() + self.config.duration
        tasks = [
            asyncio.create_task(
                self._worker_loop(user_ctx, worker_id, end_time),
                name=f"user-{user_ctx.index}-worker-{worker_id}",
            )
            for user_ctx in self._user_contexts
            for worker_id in range(self.config.workers_per_user)
        ]

        self.logger.info(
            "Benchmark started",
            extra_context={
                "users": len(self._user_contexts),
                "workers_per_user": self.config.workers_per_user,
                "duration": self.config.duration,
                "concurrency": self.config.concurrency,
                "rps_per_worker": self.config.rps_per_worker,
            },
        )

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.warning("Benchmark cancelled")
        finally:
            self._stop_event.set()
            await self._shutdown_streams()

        return self._records

    async def _initialize_users(self) -> None:
        """Create per-user contexts (token + optional WebSocket client)."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Authentication token missing. Run `neem init` to log in.")

        base_dev_user = get_dev_user_id()

        for index in range(self.config.users):
            if base_dev_user:
                user_id = f"{base_dev_user}-{index+1}" if self.config.users > 1 else base_dev_user
            else:
                user_id = f"user-{index+1}"

            job_stream: Optional[RealtimeJobClient] = None
            if self.config.enable_ws and self.backend.websocket_url:
                job_stream = RealtimeJobClient(
                    self.backend.websocket_url,
                    token_provider=lambda tok=token: tok,
                    dev_user_id=user_id,  # Use per-user ID, not base user
                    cache_ttl_seconds=self.config.ws_cache_ttl,
                    cache_max_size=self.config.ws_cache_size,
                )

            self._user_contexts.append(
                UserContext(
                    index=index,
                    user_id=user_id,
                    token=token,
                    job_stream=job_stream,
                    mix_weight_lookup=self.config.mix_weights,
                )
            )

    async def _shutdown_streams(self) -> None:
        """Close all WebSocket clients."""
        for user_ctx in self._user_contexts:
            if user_ctx.job_stream:
                with suppress(Exception):
                    await user_ctx.job_stream.close()

    async def _worker_loop(self, user_ctx: UserContext, worker_id: int, end_time: float) -> None:
        """Closed-loop worker that schedules calls until duration elapses."""
        interval = (1.0 / self.config.rps_per_worker) if self.config.rps_per_worker > 0 else None
        next_slot = time.perf_counter()

        while not self._stop_event.is_set():
            if time.monotonic() >= end_time:
                break

            if interval:
                if next_slot > time.perf_counter():
                    await asyncio.sleep(next_slot - time.perf_counter())
                next_slot += interval

            tool = self._selector.choose()

            async with self._concurrency:
                record = await self._execute_tool(tool, user_ctx)

            async with self._records_lock:
                self._records.append(record)

    async def _execute_tool(self, tool: ToolName, user_ctx: UserContext) -> ToolCallRecord:
        """Execute a tool call and capture metrics."""
        started_at = datetime.now(timezone.utc)
        start_monotonic = time.perf_counter()
        start_wall = time.time()

        if tool == "list_graphs":
            result = await self._run_list_graphs(user_ctx, start_wall)
        elif tool == "query_graph":
            result = await self._run_query_graph(user_ctx, start_wall)
        else:
            raise ValueError(f"Unsupported tool: {tool}")

        completed_at = datetime.now(timezone.utc)
        latency_ms = (time.perf_counter() - start_monotonic) * 1000.0

        mix_weight = user_ctx.mix_weight_lookup.get(tool, 0.0)

        return ToolCallRecord(
            tool=tool,
            user_id=user_ctx.user_id,
            job_id=result.job_id,
            trace_id=result.trace_id,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            backend_status=result.backend_status,
            http_status=result.http_status,
            ok=result.ok,
            error_category=result.error_category,
            error_detail=result.error_detail,
            path=result.path,
            poll_attempts=result.poll_attempts,
            backend_processing_time_ms=result.backend_processing_time_ms,
            ttfb_ms=result.ttfb_ms,
            mix_weight=mix_weight,
        )

    async def _run_list_graphs(self, user_ctx: UserContext, start_wall: float) -> "ToolExecutionResult":
        try:
            metadata, status_code = await submit_job_request(
                base_url=self.backend.base_url,
                token=user_ctx.token,
                task_type="list_graphs",
                payload={},
                user_id=user_ctx.user_id,
            )
        except httpx.HTTPStatusError as exc:
            return ToolExecutionResult(
                ok=False,
                backend_status=None,
                job_id=None,
                trace_id=None,
                http_status=exc.response.status_code,
                error_category=classify_http_error(exc.response.status_code),
                error_detail=exc.response.text,
            )
        except Exception as exc:
            return ToolExecutionResult(
                ok=False,
                backend_status=None,
                job_id=None,
                trace_id=None,
                http_status=None,
                error_category="unknown",
                error_detail=str(exc),
            )

        try:
            status_payload, path, poll_attempts, ttfb_ms, status_http = await await_job_status(
                metadata,
                token=user_ctx.token,
                job_stream=user_ctx.job_stream,
                wait_ms=self.config.wait_ms,
                start_wall=start_wall,
                user_id=user_ctx.user_id,
            )
        except httpx.HTTPStatusError as exc:
            return ToolExecutionResult(
                ok=False,
                backend_status=None,
                job_id=metadata.job_id,
                trace_id=metadata.trace_id,
                http_status=exc.response.status_code,
                error_category=classify_http_error(exc.response.status_code),
                error_detail=exc.response.text,
            )

        backend_status = (status_payload or {}).get("status")
        backend_processing_ms = extract_processing_time(status_payload)

        if backend_status == "succeeded" and metadata.links.result:
            with suppress(Exception):
                await fetch_result(metadata.links.result, user_ctx.token)

        ok = backend_status == "succeeded"
        error_category = None if ok else classify_backend_error(status_payload)
        error_detail = None if ok else json.dumps(status_payload or {}, default=str)

        http_status = status_http or status_code

        return ToolExecutionResult(
            ok=ok,
            backend_status=backend_status,
            job_id=metadata.job_id,
            trace_id=metadata.trace_id,
            http_status=http_status,
            error_category=error_category,
            error_detail=error_detail,
            path=path,
            poll_attempts=poll_attempts,
            backend_processing_time_ms=backend_processing_ms,
            ttfb_ms=ttfb_ms,
        )

    async def _run_query_graph(self, user_ctx: UserContext, start_wall: float) -> "ToolExecutionResult":
        url = f"{self.backend.base_url.rstrip('/')}/graphs/query"
        body = {
            "sparql": self.config.sparql,
            "result_format": "json",
            "max_rows": 25,
        }

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.post(url, headers=auth_headers(user_ctx.token, user_ctx.user_id), json=body)
                response.raise_for_status()
                status_code = response.status_code
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            return ToolExecutionResult(
                ok=False,
                backend_status=None,
                job_id=None,
                trace_id=None,
                http_status=exc.response.status_code,
                error_category=classify_http_error(exc.response.status_code),
                error_detail=exc.response.text,
            )
        except Exception as exc:
            return ToolExecutionResult(
                ok=False,
                backend_status=None,
                job_id=None,
                trace_id=None,
                http_status=None,
                error_category="unknown",
                error_detail=str(exc),
            )

        job_id = payload.get("job_id")
        trace_id = payload.get("trace_id")
        if not job_id:
            return ToolExecutionResult(
                ok=False,
                backend_status=None,
                job_id=None,
                trace_id=None,
                http_status=status_code,
                error_category="unknown",
                error_detail="Backend response missing job_id",
            )

        poll_url = payload.get("poll_url") or f"{self.backend.base_url.rstrip('/')}/graphs/jobs/{job_id}"
        result_url = payload.get("result_url")

        links = JobLinks(
            status=absolute_url(self.backend.base_url, poll_url),
            result=absolute_url(self.backend.base_url, result_url),
            websocket=None,
        )

        metadata = JobSubmitMetadata(
            job_id=job_id,
            status=payload.get("status", "queued"),
            trace_id=trace_id,
            links=links,
        )

        try:
            status_payload, path, poll_attempts, ttfb_ms, status_http = await await_job_status(
                metadata,
                token=user_ctx.token,
                job_stream=user_ctx.job_stream,
                wait_ms=self.config.wait_ms,
                start_wall=start_wall,
                user_id=user_ctx.user_id,
            )
        except httpx.HTTPStatusError as exc:
            return ToolExecutionResult(
                ok=False,
                backend_status=None,
                job_id=job_id,
                trace_id=trace_id,
                http_status=exc.response.status_code,
                error_category=classify_http_error(exc.response.status_code),
                error_detail=exc.response.text,
            )

        backend_status = (status_payload or {}).get("status")
        backend_processing_ms = extract_processing_time(status_payload)
        http_status = status_http or status_code
        ok = backend_status == "succeeded"
        error_category = None if ok else classify_backend_error(status_payload)
        error_detail = None if ok else json.dumps(status_payload or {}, default=str)

        return ToolExecutionResult(
            ok=ok,
            backend_status=backend_status,
            job_id=job_id,
            trace_id=trace_id,
            http_status=http_status,
            error_category=error_category,
            error_detail=error_detail,
            path=path,
            poll_attempts=poll_attempts,
            backend_processing_time_ms=backend_processing_ms,
            ttfb_ms=ttfb_ms,
        )


@dataclass
class ToolExecutionResult:
    ok: bool
    backend_status: Optional[str]
    job_id: Optional[str]
    trace_id: Optional[str]
    http_status: Optional[int]
    error_category: Optional[str]
    error_detail: Optional[str]
    path: Literal["stream", "poll"] = "poll"
    poll_attempts: int = 0
    backend_processing_time_ms: Optional[float] = None
    ttfb_ms: Optional[float] = None


async def submit_job_request(
    *,
    base_url: str,
    token: str,
    task_type: str,
    payload: JsonDict,
    user_id: Optional[str] = None,
) -> Tuple[JobSubmitMetadata, int]:
    """Submit a job through the FastAPI backend."""
    url = f"{base_url.rstrip('/')}/jobs/"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(
            url,
            headers=auth_headers(token, user_id),
            json={"type": task_type, "payload": payload},
        )
        response.raise_for_status()
        data = response.json()
    return JobSubmitMetadata.from_api(data, base_url=base_url), response.status_code


async def await_job_status(
    metadata: JobSubmitMetadata,
    *,
    token: str,
    job_stream: Optional[RealtimeJobClient],
    wait_ms: int,
    start_wall: float,
    user_id: Optional[str] = None,
) -> Tuple[Optional[JsonDict], Literal["stream", "poll"], int, Optional[float], Optional[int]]:
    """Wait for job completion via streaming where possible, else HTTP poll."""
    path: Literal["stream", "poll"] = "poll"
    poll_attempts = 0
    status_payload: Optional[JsonDict] = None
    http_status: Optional[int] = None
    ttfb_ms: Optional[float] = None
    first_event_task: Optional[asyncio.Task[Optional[float]]] = None

    if job_stream:
        first_event_task = asyncio.create_task(
            job_stream.wait_for_first_event(metadata.job_id, timeout=STREAM_TIMEOUT_SECONDS + 5),
            name=f"ttfb-{metadata.job_id}",
        )
        status_payload = await job_stream.wait_for_status(metadata.job_id, timeout=STREAM_TIMEOUT_SECONDS)
        if status_payload:
            path = "stream"
            if first_event_task:
                with suppress(asyncio.CancelledError):
                    first_event_at = await first_event_task
                    if first_event_at is not None:
                        ttfb_ms = max((first_event_at - start_wall) * 1000.0, 0.0)
        else:
            if first_event_task:
                first_event_task.cancel()
                with suppress(asyncio.CancelledError):
                    await first_event_task

    if status_payload:
        return status_payload, path, poll_attempts, ttfb_ms, http_status

    status_payload, poll_attempts, http_status = await poll_job_status(
        status_url=metadata.links.status,
        token=token,
        wait_ms=wait_ms,
        user_id=user_id,
    )
    return status_payload, "poll", poll_attempts, None, http_status


async def poll_job_status(
    *,
    status_url: Optional[str],
    token: str,
    wait_ms: int,
    user_id: Optional[str] = None,
) -> Tuple[Optional[JsonDict], int, Optional[int]]:
    """HTTP poll fallback with attempt tracking."""
    if not status_url:
        return None, 0, None

    attempts = 0
    last_payload: Optional[JsonDict] = None
    last_status_code: Optional[int] = None

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        while attempts < MAX_POLL_ATTEMPTS:
            attempts += 1
            response = await client.get(
                status_url,
                headers=auth_headers(token, user_id),
                params={"wait_ms": wait_ms},
            )
            last_status_code = response.status_code
            response.raise_for_status()
            payload = response.json()
            last_payload = payload
            status = (payload.get("status") or "").lower()
            if status in {"succeeded", "failed"}:
                return payload, attempts, last_status_code
            await asyncio.sleep(min(1.0 * attempts, 5.0))

    return last_payload, attempts, last_status_code


def extract_processing_time(status_payload: Optional[JsonDict]) -> Optional[float]:
    """Best-effort extraction of backend processing time."""
    if not status_payload:
        return None
    for key in ("processing_time_ms", "processing_ms", "duration_ms"):
        if key in status_payload:
            return status_payload.get(key)
    detail = status_payload.get("detail") if isinstance(status_payload, dict) else None
    if isinstance(detail, dict):
        for key in ("processing_time_ms", "processing_ms", "duration_ms"):
            if key in detail:
                return detail[key]
    return None


def classify_http_error(status_code: int) -> str:
    if 400 <= status_code < 500:
        return "http_4xx"
    if status_code >= 500:
        return "http_5xx"
    return "unknown"


def classify_backend_error(status_payload: Optional[JsonDict]) -> str:
    if not status_payload:
        return "timeout"
    status = (status_payload.get("status") or "").lower()
    if status == "failed":
        detail = status_payload.get("detail") or {}
        code = (detail.get("error_code") or "").lower()
        if "auth" in code:
            return "auth_missing"
    return "unknown"


def auth_headers(token: str, user_id: Optional[str] = None) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    dev_user = user_id or get_dev_user_id()
    if dev_user:
        headers["X-User-ID"] = dev_user
    return headers


def absolute_url(base_url: str, candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate
    return urljoin(f"{base_url.rstrip('/')}/", candidate.lstrip("/"))


def parse_mix(mix: str) -> Dict[ToolName, float]:
    weights: Dict[ToolName, float] = {}
    mix = mix.strip()
    if not mix:
        weights["list_graphs"] = 1.0
        return weights

    for chunk in mix.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, weight_str = chunk.partition(":")
        tool = name.strip()
        if tool not in SUPPORTED_TOOLS:
            raise typer.BadParameter(f"Unsupported tool '{tool}'. Supported: {SUPPORTED_TOOLS}")
        try:
            weight = float(weight_str or "1.0")
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid weight for '{tool}': {weight_str}") from exc
        if weight <= 0:
            continue
        weights[tool] = weight

    if not weights:
        raise typer.BadParameter("Mix must include at least one tool with positive weight.")

    return weights


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return sorted_vals[int(k)]
    lower_val = sorted_vals[lower]
    upper_val = sorted_vals[upper]
    return lower_val + (upper_val - lower_val) * (k - lower)


def summarize(records: Sequence[ToolCallRecord]) -> List[str]:
    """Generate per-tool summary strings."""
    summaries: List[str] = []
    by_tool: Dict[str, List[ToolCallRecord]] = {}
    for record in records:
        by_tool.setdefault(record.tool, []).append(record)

    for tool, tool_records in sorted(by_tool.items()):
        latencies = [r.latency_ms for r in tool_records if r.ok]
        successes = sum(1 for r in tool_records if r.ok)
        errors = len(tool_records) - successes
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        p99 = percentile(latencies, 99)
        summaries.append(
            f"{tool}: calls={len(tool_records)} success={successes} error={errors} "
            f"p50={format_ms(p50)} p95={format_ms(p95)} p99={format_ms(p99)}"
        )
    return summaries


def format_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}ms"


def write_output(path: Optional[Path], records: Sequence[ToolCallRecord], ndjson: bool = False) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if ndjson:
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict()))
                fh.write("\n")
    else:
        serialized = [record.to_dict() for record in records]
        path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def render_visualizations(records: Sequence[ToolCallRecord]) -> None:
    """Render latency histograms and scatter plots per tool."""
    if not records:
        typer.echo("No records to visualize.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - import varies per environment
        typer.echo(f"Matplotlib is required for --visualize ({exc}).")
        return

    by_tool: Dict[str, List[ToolCallRecord]] = {}
    for record in records:
        by_tool.setdefault(record.tool, []).append(record)

    tool_count = len(by_tool)
    if tool_count == 0:
        typer.echo("No tool data available for visualization.")
        return

    global_start = min(record.started_at for record in records)
    fig, axes = plt.subplots(tool_count, 2, figsize=(12, 4 * tool_count), squeeze=False)

    for row_idx, (tool, tool_records) in enumerate(sorted(by_tool.items())):
        latencies = [r.latency_ms for r in tool_records]
        hist_ax = axes[row_idx][0]
        scatter_ax = axes[row_idx][1]

        if latencies:
            bins = min(40, max(10, len(latencies) // 2))
            hist_ax.hist(latencies, bins=bins, color="tab:blue", alpha=0.7)
            for mark in (50, 95, 99):
                value = percentile(latencies, mark)
                if value is not None:
                    hist_ax.axvline(value, linestyle="--", label=f"p{mark}={value:.1f}ms", alpha=0.8)
            if hist_ax.get_legend():
                hist_ax.legend(fontsize="small")
        else:
            hist_ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=hist_ax.transAxes)

        hist_ax.set_title(f"{tool} latency histogram")
        hist_ax.set_xlabel("Latency (ms)")
        hist_ax.set_ylabel("Calls")

        x_values = [(record.completed_at - global_start).total_seconds() for record in tool_records]
        colors = ["tab:green" if record.ok else "tab:red" for record in tool_records]

        if latencies:
            scatter_ax.scatter(x_values, latencies, c=colors, alpha=0.7, s=20)
        else:
            scatter_ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=scatter_ax.transAxes)

        scatter_ax.set_title(f"{tool} latency over time")
        scatter_ax.set_xlabel("Seconds since start")
        scatter_ax.set_ylabel("Latency (ms)")
        scatter_ax.grid(alpha=0.25)

    fig.suptitle("Mnemosyne MCP Benchmark", fontsize=16)
    fig.tight_layout()

    try:
        plt.show()
    except Exception as exc:  # pragma: no cover - backend dependent
        fallback_path = Path("benchmark_visualization.png")
        with suppress(Exception):
            fig.savefig(fallback_path, dpi=150, bbox_inches="tight")
            typer.echo(f"Saved visualization to {fallback_path} (could not display window: {exc})")


def build_config(
    duration: float,
    users: int,
    concurrency: int,
    workers: int,
    rps: float,
    mix: str,
    sparql: str,
    no_ws: bool,
    wait_ms: int,
    ws_ttl: float,
    ws_cache_size: int,
    visualize: bool,
    output: Optional[str],
    ndjson: Optional[str],
    log_level: str,
    seed: Optional[int],
) -> BenchmarkConfig:
    if duration <= 0:
        raise typer.BadParameter("Duration must be positive.")
    if users <= 0:
        raise typer.BadParameter("Users must be at least 1.")
    if concurrency <= 0:
        raise typer.BadParameter("Concurrency must be at least 1.")
    if workers <= 0:
        raise typer.BadParameter("Workers per user must be at least 1.")
    if wait_ms <= 0:
        raise typer.BadParameter("wait-ms must be positive.")

    mix_weights = parse_mix(mix)

    output_path = Path(output) if output else None
    ndjson_path = Path(ndjson) if ndjson else None

    return BenchmarkConfig(
        duration=duration,
        users=users,
        workers_per_user=workers,
        concurrency=concurrency,
        rps_per_worker=rps,
        mix_weights=mix_weights,
        sparql=sparql,
        enable_ws=not no_ws,
        wait_ms=wait_ms,
        ws_cache_ttl=ws_ttl,
        ws_cache_size=ws_cache_size,
        visualize=visualize,
        output_path=output_path,
        ndjson_path=ndjson_path,
        log_level=log_level,
        seed=seed,
    )


@app.command()
def main(
    duration: float = typer.Option(30.0, "--duration", help="Benchmark duration in seconds."),
    users: int = typer.Option(1, "--users", help="Number of logical users."),
    concurrency: int = typer.Option(8, "--concurrency", help="Global in-flight concurrency limit."),
    workers: int = typer.Option(1, "--workers", help="Workers per user."),
    rps: float = typer.Option(0.0, "--rps", help="Target requests/sec per worker (0=unpaced)."),
    mix: str = typer.Option("list_graphs:1.0", "--mix", help="Weighted tool mix, e.g. list_graphs:0.7,query_graph:0.3"),
    sparql: str = typer.Option("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10", "--sparql", help="SPARQL query for query_graph tool."),
    no_ws: bool = typer.Option(False, "--no-ws", help="Disable WebSocket streaming and force HTTP polling."),
    wait_ms: int = typer.Option(1500, "--wait-ms", help="Polling wait_ms for HTTP fallback."),
    ws_ttl: float = typer.Option(3600.0, "--ws-ttl", help="WebSocket cache TTL per user (seconds)."),
    ws_cache_size: int = typer.Option(1000, "--ws-cache-size", help="WebSocket cache size per user."),
    visualize: bool = typer.Option(False, "--visualize", help="Render Matplotlib latency charts."),
    output: Optional[str] = typer.Option(None, "--output", help="Write JSON array of call records."),
    ndjson: Optional[str] = typer.Option(None, "--ndjson", help="Write NDJSON stream of call records."),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level for benchmark logger."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for workload selection."),
) -> None:
    """Run the Mnemosyne MCP benchmark harness."""
    config = build_config(
        duration=duration,
        users=users,
        concurrency=concurrency,
        workers=workers,
        rps=rps,
        mix=mix,
        sparql=sparql,
        no_ws=no_ws,
        wait_ms=wait_ms,
        ws_ttl=ws_ttl,
        ws_cache_size=ws_cache_size,
        visualize=visualize,
        output=output,
        ndjson=ndjson,
        log_level=log_level,
        seed=seed,
    )

    LoggerFactory.configure_logging(level=config.log_level.upper())

    if config.seed is not None:
        random.seed(config.seed)

    harness = BenchmarkHarness(config)
    records = asyncio.run(harness.run())

    if not records:
        typer.echo("Benchmark finished with no records.")  # pragma: no cover
        return

    for line in summarize(records):
        typer.echo(line)

    write_output(config.output_path, records, ndjson=False)
    write_output(config.ndjson_path, records, ndjson=True)

    if config.visualize:
        render_visualizations(records)


if __name__ == "__main__":
    app()
