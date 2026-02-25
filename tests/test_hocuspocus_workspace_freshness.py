"""Workspace channel freshness regression tests for HocuspocusClient."""

import time
from typing import Any

import pytest
import pycrdt

from neem.hocuspocus.client import ChannelState, HocuspocusClient
from neem.hocuspocus.protocol import encode_sync_update


class _FakeWS:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _make_client() -> HocuspocusClient:
    return HocuspocusClient(
        base_url="http://localhost:8080",
        token_provider=lambda: None,
        dev_user_id="u1",
    )


@pytest.mark.asyncio
async def test_connect_workspace_reconnects_when_stale(monkeypatch: Any) -> None:
    client = _make_client()
    key = "u1:g1"

    stale = ChannelState()
    stale.ws = _FakeWS()
    stale.synced.set()
    stale.synced_at = time.monotonic() - 60
    client._workspace_channels[key] = stale  # noqa: SLF001 - test internal state

    connect_calls = 0

    async def _fake_connect_channel(
        channel: ChannelState,
        path: str,
        channel_name: str,
        user_id: str | None = None,
    ) -> None:
        nonlocal connect_calls
        connect_calls += 1
        channel.ws = _FakeWS()
        channel.synced.set()
        channel.synced_at = time.monotonic()

    monkeypatch.setattr(client, "_connect_channel", _fake_connect_channel)

    await client.connect_workspace("g1", user_id="u1", max_age=1.0)

    assert connect_calls == 1
    assert stale.ws is not None and stale.ws.closed is True
    assert key in client._workspace_channels  # noqa: SLF001
    assert client._workspace_channels[key] is not stale  # noqa: SLF001


@pytest.mark.asyncio
async def test_connect_workspace_reuses_fresh_channel(monkeypatch: Any) -> None:
    client = _make_client()
    key = "u1:g1"

    fresh = ChannelState()
    fresh.ws = _FakeWS()
    fresh.synced.set()
    fresh.synced_at = time.monotonic()
    client._workspace_channels[key] = fresh  # noqa: SLF001

    async def _fail_connect(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("should not reconnect fresh workspace channel")

    monkeypatch.setattr(client, "_connect_channel", _fail_connect)

    await client.connect_workspace("g1", user_id="u1", max_age=5.0)

    assert client._workspace_channels[key] is fresh  # noqa: SLF001


@pytest.mark.asyncio
async def test_handle_sync_update_refreshes_synced_timestamp() -> None:
    client = _make_client()
    channel = ChannelState()
    channel.synced.set()
    channel.synced_at = time.monotonic() - 60

    source = pycrdt.Doc()
    payload_map: pycrdt.Map = source.get("m", type=pycrdt.Map)
    payload_map["k"] = "v"
    message = encode_sync_update(source.get_update())

    before = channel.synced_at
    await client._handle_message(channel, message, "workspace:g1")  # noqa: SLF001

    assert channel.synced_at > before

