"""Auth-mode-specific HocuspocusClient header and protocol behavior."""

from __future__ import annotations

import pytest

from neem.hocuspocus.client import HocuspocusClient


def test_public_mode_uses_bearer_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "public")
    client = HocuspocusClient(
        base_url="http://localhost:8080",
        token_provider=lambda: "request-token",
        dev_user_id="dev-user",
        internal_service_secret="internal-secret",
    )

    headers = client._build_auth_headers(user_id="forwarded-user")  # noqa: SLF001 - auth helper coverage
    protocols = client._build_ws_protocols(user_id="forwarded-user")  # noqa: SLF001 - auth helper coverage

    assert headers == {"Authorization": "Bearer request-token"}
    assert protocols == ["Bearer.request-token"]


def test_hosted_mode_uses_bearer_ws_auth_and_internal_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "hosted")
    client = HocuspocusClient(
        base_url="http://localhost:8080",
        token_provider=lambda: "request-token",
        dev_user_id="dev-user",
        internal_service_secret="internal-secret",
    )

    headers = client._build_auth_headers(user_id="forwarded-user")  # noqa: SLF001 - auth helper coverage
    protocols = client._build_ws_protocols(user_id="forwarded-user")  # noqa: SLF001 - auth helper coverage

    assert headers == {
        "Authorization": "Bearer request-token",
        "X-Internal-Service": "internal-secret",
    }
    assert protocols == ["Bearer.request-token"]


def test_public_mode_disables_user_id_subprotocol_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "public")
    client = HocuspocusClient(
        base_url="http://localhost:8080",
        token_provider=lambda: None,
        dev_user_id="dev-user",
    )

    headers = client._build_auth_headers(user_id="forwarded-user")  # noqa: SLF001 - auth helper coverage
    protocols = client._build_ws_protocols(user_id="forwarded-user")  # noqa: SLF001 - auth helper coverage

    assert headers == {}
    assert protocols is None
