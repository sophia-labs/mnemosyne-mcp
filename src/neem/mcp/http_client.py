"""Shared httpx client for MCP tool HTTP calls.

Replaces per-request ``async with httpx.AsyncClient() as client:`` with a
single long-lived client that reuses TCP connections via connection pooling.
This dramatically reduces connection churn when many tool calls run in
parallel (e.g. pathfinder agents making 30+ concurrent requests).

Usage::

    from neem.mcp.http_client import get_http_client

    # In any tool function — uses the shared pooled client:
    resp = await get_http_client().post(url, headers=h, timeout=httpx.Timeout(15.0))

The shared client is initialised once at server startup via
``set_http_client(create_http_client())``.  If no client has been set when
``get_http_client()`` is first called, a fallback client is created
automatically (useful for tests and non-server entry points).
"""

from __future__ import annotations

from typing import Optional

import httpx

from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("mcp.http_client")

# Default timeout — individual calls override as needed via timeout= kwarg.
DEFAULT_TIMEOUT = 30.0

# Module-level singleton.
_client: Optional[httpx.AsyncClient] = None


def create_http_client(
    max_connections: int = 20,
    max_keepalive_connections: int = 10,
    keepalive_expiry: float = 30.0,
) -> httpx.AsyncClient:
    """Create a new :class:`httpx.AsyncClient` with connection pooling."""
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry=keepalive_expiry,
    )
    client = httpx.AsyncClient(
        limits=limits,
        timeout=httpx.Timeout(DEFAULT_TIMEOUT),
    )
    logger.info(
        "Shared HTTP client created",
        extra_context={
            "max_connections": max_connections,
            "max_keepalive": max_keepalive_connections,
            "keepalive_expiry_s": keepalive_expiry,
        },
    )
    return client


def set_http_client(client: httpx.AsyncClient) -> None:
    """Set the module-level shared client (called once at server startup).

    If a previous client exists it is closed best-effort to avoid leaking
    connections (relevant for tests that spin up multiple servers).
    """
    global _client
    old = _client
    _client = client
    if old is not None and old is not client:
        try:
            old.close()  # sync close — enough for cleanup
        except Exception:
            pass


def clear_http_client() -> None:
    """Clear the global pointer (call after closing the client at shutdown)."""
    global _client
    _client = None


def get_http_client() -> httpx.AsyncClient:
    """Return the shared client.  Creates a fallback if none was set."""
    global _client
    if _client is None:
        _client = create_http_client()
        logger.warning("No shared HTTP client was set; created fallback client")
    return _client
