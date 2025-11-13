"""Integration-style tests for the realtime job WebSocket client."""

import asyncio
import json

import pytest
import pytest_asyncio
from aiohttp import web

from neem.mcp.jobs.realtime import RealtimeJobClient


@pytest_asyncio.fixture
async def websocket_server(unused_tcp_port):
    """Spin up a temporary aiohttp WebSocket endpoint for testing."""

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Basic auth check so we know headers make it through.
        assert request.headers.get("Authorization") == "Bearer test-token"
        assert request.headers.get("X-User-ID") == "dev-user"
        assert request.headers.get("Sec-WebSocket-Protocol") == "Bearer.dev-user"

        # Simulate backend: immediately send events for job-123
        # (mimics the user channel broadcasting all job events)
        await asyncio.sleep(0.1)  # Small delay to let client get ready
        await ws.send_json({"job_id": "job-123", "type": "progress", "payload": {"step": 0}})
        await ws.send_json({"job_id": "job-123", "type": "completed", "payload": {"result": "ok"}})

        # Keep connection open until client closes
        async for msg in ws:
            pass

        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/ws", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()

    yield f"ws://127.0.0.1:{unused_tcp_port}/ws"

    await runner.cleanup()


@pytest.mark.asyncio
async def test_realtime_client_receives_events(websocket_server):
    """Test that events are cached and can be retrieved."""
    client = RealtimeJobClient(
        websocket_url=websocket_server,
        token_provider=lambda: "test-token",
        dev_user_id="dev-user",
        heartbeat=2.0,
    )

    try:
        # Ensure client connects (triggers background connection)
        await client._ensure_connected()

        # Wait for job to complete (events pushed to cache)
        await asyncio.wait_for(client.wait_for_terminal("job-123", timeout=5.0), timeout=5.0)

        # Verify job is marked as complete
        assert client.is_job_complete("job-123")

        # Get all events from cache
        events = await client.get_events("job-123")
        assert len(events) == 2

        # Check first event
        assert events[0].event_type == "progress"
        assert events[0].job_id == "job-123"
        assert events[0].payload["step"] == 0

        # Check second event
        assert events[1].event_type == "completed"
        assert events[1].payload["result"] == "ok"

    finally:
        await client.close()
