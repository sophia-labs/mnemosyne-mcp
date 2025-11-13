"""Tests for backend configuration helpers."""

import os

import pytest

from neem.mcp.server.standalone_server import resolve_backend_config, DEFAULT_LOCAL_BACKEND_URL


def _clear_env(monkeypatch):
    for key in [
        "MNEMOSYNE_FASTAPI_URL",
        "MNEMOSYNE_API_URL",
        "MNEMOSYNE_FASTAPI_HOST",
        "MNEMOSYNE_FASTAPI_PORT",
        "FASTAPI_SERVICE_HOST",
        "FASTAPI_SERVICE_PORT",
        "MNEMOSYNE_FASTAPI_WS_URL",
        "MNEMOSYNE_FASTAPI_WS_PATH",
        "MNEMOSYNE_FASTAPI_WS_DISABLE",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_backend_config_derives_ws_url(monkeypatch):
    """Default configuration converts HTTP base to ws:// path."""
    _clear_env(monkeypatch)
    config = resolve_backend_config()

    assert config.base_url == DEFAULT_LOCAL_BACKEND_URL
    assert config.websocket_url == "ws://127.0.0.1:8001/ws"
    assert config.has_websocket is True


def test_backend_config_respects_explicit_ws_url(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MNEMOSYNE_FASTAPI_URL", "https://api.example.com")
    monkeypatch.setenv("MNEMOSYNE_FASTAPI_WS_URL", "wss://ws.example.com/stream")

    config = resolve_backend_config()
    assert config.base_url == "https://api.example.com"
    assert config.websocket_url == "wss://ws.example.com/stream"


def test_backend_config_can_disable_ws(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MNEMOSYNE_FASTAPI_WS_DISABLE", "true")

    config = resolve_backend_config()
    assert config.websocket_url is None
    assert config.has_websocket is False
