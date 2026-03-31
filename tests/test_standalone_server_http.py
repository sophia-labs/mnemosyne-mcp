"""HTTP app helpers for the standalone MCP server."""

import pytest
from starlette.testclient import TestClient

from neem.mcp.server.standalone_server import build_streamable_http_app
from mcp.server.fastmcp import FastMCP


def test_build_streamable_http_app_exposes_health() -> None:
    app = build_streamable_http_app(FastMCP("test"))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_build_streamable_http_app_starts_inner_lifespan() -> None:
    app = build_streamable_http_app(FastMCP("test"))

    with TestClient(app) as client:
        response = client.post("/mcp", json={})

    assert response.status_code != 500


def test_build_streamable_http_app_supports_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_ROOT_PATH_PREFIX", "/chatgpt-demo")
    app = build_streamable_http_app(FastMCP("test"))

    with TestClient(app) as client:
        response = client.post("/chatgpt-demo/mcp", json={})

    assert response.status_code != 500


def test_build_streamable_http_app_exposes_protected_resource_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "chatgpt_oauth")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_AUTH_SERVER_URL", "https://api.example.com/oauth/chatgpt")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_RESOURCE_URL", "https://api.example.com/chatgpt-demo/mcp")

    app = build_streamable_http_app(FastMCP("test"))

    with TestClient(app) as client:
        response = client.get("/.well-known/oauth-protected-resource")

    assert response.status_code == 200
    assert response.json() == {
        "resource": "https://api.example.com/chatgpt-demo/mcp",
        "authorization_servers": ["https://api.example.com/oauth/chatgpt"],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["mnemosyne.mcp.read"],
    }


def test_chatgpt_oauth_challenge_uses_configured_https_metadata_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "chatgpt_oauth")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_AUTH_SERVER_URL", "https://api.example.com/oauth/chatgpt")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_RESOURCE_URL", "https://api.example.com/chatgpt-auth/mcp")
    monkeypatch.setenv("MCP_ROOT_PATH_PREFIX", "/chatgpt-auth")

    app = build_streamable_http_app(FastMCP("test"))

    with TestClient(app) as client:
        response = client.get("/chatgpt-auth/mcp")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == (
        'Bearer realm="mnemosyne", '
        'resource_metadata="https://api.example.com/chatgpt-auth/.well-known/oauth-protected-resource"'
    )
