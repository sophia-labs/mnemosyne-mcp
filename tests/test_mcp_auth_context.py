"""Tests for request-scoped MCP auth context behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from neem.mcp.auth import (
    MCPAuthContext,
    clear_current_auth_context,
    get_current_auth_context,
    get_current_auth_token,
)


@dataclass
class _FakeRequest:
    headers: dict[str, str]


@dataclass
class _FakeRequestContext:
    request: _FakeRequest


@dataclass
class _FakeContext:
    request_context: _FakeRequestContext


@pytest.fixture(autouse=True)
def _clear_context() -> None:
    clear_current_auth_context()
    yield
    clear_current_auth_context()


def _ctx(headers: dict[str, str]) -> _FakeContext:
    return _FakeContext(request_context=_FakeRequestContext(request=_FakeRequest(headers=headers)))


def test_from_context_uses_http_headers_and_binds_current_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "hosted")
    monkeypatch.setattr("neem.mcp.auth.validate_token_and_load", lambda: "local-token")

    auth = MCPAuthContext.from_context(
        _ctx(
            {
                "authorization": "Bearer request-token",
                "x-user-id": "user-123",
            }
        )
    )

    assert auth.token == "request-token"
    assert auth.user_id == "user-123"
    assert auth.source == "http_header"
    assert get_current_auth_context() is not None
    assert get_current_auth_token() == "request-token"


def test_hosted_mode_disables_local_token_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "hosted")
    monkeypatch.setattr("neem.mcp.auth.validate_token_and_load", lambda: "local-token")

    auth = MCPAuthContext.from_context(_ctx({}))
    assert auth.token is None
    assert get_current_auth_token() is None


def test_auto_mode_allows_local_token_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MNEMOSYNE_MCP_AUTH_MODE", raising=False)
    monkeypatch.setattr("neem.mcp.auth.validate_token_and_load", lambda: "local-token")

    auth = MCPAuthContext.from_context(_ctx({}))
    assert auth.token == "local-token"
    assert get_current_auth_token() == "local-token"
