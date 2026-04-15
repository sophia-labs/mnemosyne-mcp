"""Tests for MCP auth context resolution."""

from __future__ import annotations

from types import SimpleNamespace

from neem.mcp.auth import MCPAuthContext


class _ContextWithoutRequest:
    @property
    def request_context(self):
        raise ValueError("Context is not available outside of a request")


def test_from_context_tolerates_fastmcp_context_without_request(monkeypatch) -> None:
    monkeypatch.setenv("MNEMOSYNE_DEV_USER_ID", "dev-user")
    monkeypatch.delenv("MNEMOSYNE_DEV_TOKEN", raising=False)

    auth = MCPAuthContext.from_context(_ContextWithoutRequest())
    assert auth.user_id in {"dev-user", "dev-user-001"}
    assert auth.source in {"dev_env", "local_storage"}


def test_from_context_reads_headers_when_request_present(monkeypatch) -> None:
    monkeypatch.delenv("MNEMOSYNE_DEV_USER_ID", raising=False)
    monkeypatch.delenv("MNEMOSYNE_DEV_TOKEN", raising=False)

    request = SimpleNamespace(headers={"Authorization": "Bearer token-123", "X-User-ID": "alice"})
    ctx = SimpleNamespace(request_context=SimpleNamespace(request=request))

    auth = MCPAuthContext.from_context(ctx)
    assert auth.token == "token-123"
    assert auth.user_id == "alice"
    assert auth.source == "http_header"
