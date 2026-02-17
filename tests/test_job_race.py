"""Regression tests for WS+poll job race behavior."""

import asyncio
from typing import Any

import pytest

from neem.mcp.jobs.models import JobLinks, JobSubmitMetadata, WebSocketSubscriptionHint
from neem.mcp.tools import basic


class _DummyAuth:
    def http_headers(self) -> dict[str, str]:
        return {}


def _metadata() -> JobSubmitMetadata:
    return JobSubmitMetadata(
        job_id="job-1",
        status="queued",
        trace_id="trace-1",
        links=JobLinks(
            status="https://backend/jobs/job-1",
            result=None,
            websocket=WebSocketSubscriptionHint(
                description="subscribe",
                payload={"type": "subscribe", "job_id": "job-1"},
            ),
        ),
    )


@pytest.mark.asyncio
async def test_await_job_completion_keeps_poll_when_ws_returns_none(monkeypatch: Any):
    async def _fake_stream(*args: Any, **kwargs: Any):
        await asyncio.sleep(0.01)
        return None

    async def _fake_poll(*args: Any, **kwargs: Any):
        await asyncio.sleep(0.02)
        return {"status": "succeeded", "detail": {"result_inline": {"ok": True}}}

    monkeypatch.setattr(basic, "stream_job", _fake_stream)
    monkeypatch.setattr(basic, "poll_job_until_terminal", _fake_poll)

    ws_events, poll_payload = await basic.await_job_completion(
        object(),
        _metadata(),
        _DummyAuth(),  # type: ignore[arg-type]
        timeout=0.2,
    )

    assert ws_events is None
    assert poll_payload is not None
    assert poll_payload.get("status") == "succeeded"


@pytest.mark.asyncio
async def test_await_job_completion_keeps_poll_even_after_ws_events(monkeypatch: Any):
    async def _fake_stream(*args: Any, **kwargs: Any):
        await asyncio.sleep(0.01)
        return [{"type": "job_update", "payload": {"status": "succeeded"}}]

    async def _fake_poll(*args: Any, **kwargs: Any):
        await asyncio.sleep(0.02)
        return {"status": "succeeded", "detail": {"result_inline": {"ok": True}}}

    monkeypatch.setattr(basic, "stream_job", _fake_stream)
    monkeypatch.setattr(basic, "poll_job_until_terminal", _fake_poll)

    ws_events, poll_payload = await basic.await_job_completion(
        object(),
        _metadata(),
        _DummyAuth(),  # type: ignore[arg-type]
        timeout=0.2,
    )

    assert ws_events is not None
    assert len(ws_events) == 1
    assert poll_payload is not None
    assert poll_payload.get("status") == "succeeded"
