"""HTTP app helpers for the standalone MCP server."""

from starlette.routing import Mount
from starlette.testclient import TestClient

from neem.mcp.server.standalone_server import build_streamable_http_app
from mcp.server.fastmcp import FastMCP


def test_build_streamable_http_app_exposes_health() -> None:
    app = build_streamable_http_app(FastMCP("test"))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_build_streamable_http_app_mounts_transport_at_mcp() -> None:
    app = build_streamable_http_app(FastMCP("test"))

    mounts = [route for route in app.routes if isinstance(route, Mount)]

    assert [mount.path for mount in mounts] == ["/mcp"]
