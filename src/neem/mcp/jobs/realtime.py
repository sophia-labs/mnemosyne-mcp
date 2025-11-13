"""
Realtime job subscription client.

Wires the FastAPI `/ws` gateway into an asyncio-friendly helper that tools can
use to stream job progress without writing bespoke WebSocket plumbing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from neem.mcp.jobs.models import WebSocketSubscriptionHint
from neem.utils.logging import LoggerFactory

JsonDict = Dict[str, Any]
TokenProvider = Callable[[], Optional[str]]

logger = LoggerFactory.get_logger("mcp.realtime_jobs")


@dataclass
class JobEvent:
    """Envelope emitted for everything received over the WebSocket."""

    job_id: str
    event_type: str
    payload: JsonDict
    raw: JsonDict


@dataclass
class JobCacheEntry:
    """Cache entry for a job's events."""

    job_id: str
    events: List[JobEvent] = field(default_factory=list)
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)


class RealtimeJobClient:
    """Maintains a single authenticated WebSocket connection for job push events."""

    def __init__(
        self,
        websocket_url: str,
        token_provider: TokenProvider,
        *,
        dev_user_id: Optional[str] = None,
        heartbeat: float = 20.0,
        connect_timeout: float = 10.0,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        session_factory: Optional[Callable[[], aiohttp.ClientSession]] = None,
        cache_ttl_seconds: float = 3600.0,
        cache_max_size: int = 1000,
        cleanup_interval_seconds: float = 60.0,
    ) -> None:
        self.websocket_url = websocket_url
        self._token_provider = token_provider
        self._heartbeat = heartbeat
        self._connect_timeout = connect_timeout
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._session_factory = session_factory
        self._dev_user_id = dev_user_id
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache_max_size = cache_max_size
        self._cleanup_interval_seconds = cleanup_interval_seconds

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._stop_requested = False

        self._event_cache: Dict[str, JobCacheEntry] = {}

    @property
    def is_connected(self) -> bool:
        return bool(self._ws) and not self._ws.closed

    async def close(self) -> None:
        """Shut down the WebSocket cleanly."""
        self._stop_requested = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
        await self._teardown_connection()
        self._event_cache.clear()

    async def get_events(self, job_id: str, from_index: int = 0) -> List[JobEvent]:
        """Get events for a job starting from the given index."""
        async with self._lock:
            entry = self._event_cache.get(job_id)
            if not entry:
                return []
            entry.last_accessed = time.time()
            return entry.events[from_index:]

    async def wait_for_terminal(self, job_id: str, timeout: float) -> None:
        """Wait until a job reaches terminal state or timeout."""
        entry = await self._ensure_cache_entry(job_id)
        await asyncio.wait_for(entry.completed.wait(), timeout=timeout)

    def is_job_complete(self, job_id: str) -> bool:
        """Check if a job has reached terminal state."""
        entry = self._event_cache.get(job_id)
        return entry.completed.is_set() if entry else False

    async def _ensure_cache_entry(self, job_id: str) -> JobCacheEntry:
        """Ensure a cache entry exists for the given job_id."""
        async with self._lock:
            if job_id not in self._event_cache:
                self._event_cache[job_id] = JobCacheEntry(job_id=job_id)
            return self._event_cache[job_id]

    async def _ensure_connected(self) -> None:
        """Establish the WebSocket connection if needed."""
        if self.is_connected:
            return
        await self._connect_with_backoff()

    async def _connect_with_backoff(self) -> None:
        """Attempt to connect, retrying with exponential backoff on failure."""
        attempt = 0
        while not self._stop_requested:
            attempt += 1
            try:
                token = self._token_provider()
                if not token:
                    raise RuntimeError(
                        "Authentication token not available; run `neem init` again to refresh credentials."
                    )

                self._session = self._session_factory() if self._session_factory else aiohttp.ClientSession()
                headers = {"Authorization": f"Bearer {token}"}
                protocols = None
                if self._dev_user_id:
                    headers["X-User-ID"] = self._dev_user_id
                    protocols = [f"Bearer.{self._dev_user_id}"]

                self._ws = await self._session.ws_connect(
                    self.websocket_url,
                    headers=headers,
                    heartbeat=self._heartbeat,
                    timeout=self._connect_timeout,
                    protocols=protocols or None,
                )

                logger.info(
                    "Connected to FastAPI WebSocket gateway",
                    extra_context={"url": self.websocket_url},
                )

                self._receiver_task = asyncio.create_task(self._receiver_loop(), name="mnemosyne-ws-listener")

                # Start cleanup task if not already running
                if not self._cleanup_task or self._cleanup_task.done():
                    self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="mnemosyne-cache-cleanup")

                return
            except Exception as exc:
                logger.warning(
                    "WebSocket connection attempt failed",
                    extra_context={"attempt": attempt, "error": str(exc)},
                )
                await self._teardown_connection()
                delay = min(self._reconnect_max_delay, self._reconnect_base_delay * (2 ** (attempt - 1)))
                await asyncio.sleep(delay)

    async def _receiver_loop(self) -> None:
        """Dispatch incoming WS messages to job queues."""
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = msg.json(loads=json.loads)
                    except json.JSONDecodeError:
                        logger.warning("Received invalid JSON payload from WebSocket", extra_context={"data": msg.data})
                        continue
                    await self._handle_payload(payload)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    logger.debug("Ignoring binary WebSocket payload")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket transport error", extra_context={"error": self._ws.exception()})
                    break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("WebSocket listener crashed", extra_context={"error": str(exc)})
        finally:
            await self._teardown_connection()
            if not self._stop_requested:
                await self._connect_with_backoff()

    async def _handle_payload(self, payload: JsonDict) -> None:
        """Push incoming events to the cache."""
        job_id = payload.get("job_id")
        if not job_id:
            logger.debug("Dropping WebSocket payload without job_id", extra_context={"payload": payload})
            return

        event_type = _extract_event_type(payload)
        job_event = JobEvent(
            job_id=job_id,
            event_type=event_type,
            payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else payload,
            raw=payload,
        )

        async with self._lock:
            if job_id not in self._event_cache:
                self._event_cache[job_id] = JobCacheEntry(job_id=job_id)

            entry = self._event_cache[job_id]
            entry.events.append(job_event)
            entry.last_accessed = time.time()

            if _is_terminal(payload):
                entry.completed.set()
                logger.debug(
                    "Job marked as completed in cache",
                    extra_context={"job_id": job_id, "event_count": len(entry.events)},
                )

    async def _cleanup_loop(self) -> None:
        """Periodically clean up stale cache entries using TTL and LRU strategies."""
        try:
            while not self._stop_requested:
                await asyncio.sleep(self._cleanup_interval_seconds)

                now = time.time()
                async with self._lock:
                    # TTL cleanup: remove entries older than TTL
                    expired = [
                        job_id
                        for job_id, entry in self._event_cache.items()
                        if (now - entry.created_at) > self._cache_ttl_seconds
                    ]
                    for job_id in expired:
                        del self._event_cache[job_id]

                    if expired:
                        logger.debug(
                            "Cleaned up expired cache entries",
                            extra_context={"count": len(expired)},
                        )

                    # LRU cleanup: if still over limit, remove least recently used
                    if len(self._event_cache) > self._cache_max_size:
                        sorted_entries = sorted(
                            self._event_cache.items(),
                            key=lambda x: x[1].last_accessed,
                        )
                        to_remove = len(self._event_cache) - self._cache_max_size
                        removed_ids = []
                        for job_id, _ in sorted_entries[:to_remove]:
                            del self._event_cache[job_id]
                            removed_ids.append(job_id)

                        logger.debug(
                            "Cleaned up LRU cache entries",
                            extra_context={"count": len(removed_ids)},
                        )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Cache cleanup loop crashed", extra_context={"error": str(exc)})

    async def _teardown_connection(self) -> None:
        current = asyncio.current_task()
        receiver_task = self._receiver_task
        if receiver_task:
            if receiver_task is current:
                receiver_task.cancel()
            else:
                receiver_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver_task
        self._receiver_task = None

        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()

        self._ws = None
        self._session = None


def _extract_event_type(payload: JsonDict) -> str:
    for key in ("event", "type", "status", "state"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return "message"


def _is_terminal(payload: JsonDict) -> bool:
    terminal_markers = {"succeeded", "failed", "completed", "complete"}
    for key in ("status", "state", "event", "type"):
        value = payload.get(key)
        if isinstance(value, str) and value.lower() in terminal_markers:
            return True
    return False
