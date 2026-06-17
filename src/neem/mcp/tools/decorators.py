"""Decorators for MCP tool functions.

Provides reusable decorators for common patterns like authentication
and parameter validation.
"""

from __future__ import annotations

import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from mcp.server.fastmcp import Context
from neem.mcp.auth import MCPAuthContext, get_current_auth_token

# Type variables for generic decorator typing
F = TypeVar("F", bound=Callable[..., Any])


def require_auth(func: F) -> F:
    """Decorator that validates authentication before tool execution.

    Raises RuntimeError if no valid token is available.

    Usage:
        @require_auth
        async def my_tool(...) -> dict:
            # Token is guaranteed to be valid here
            ...
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        context = kwargs.get("context")
        if context is not None:
            auth = MCPAuthContext.from_context(context)
            token = auth.require_auth()
        else:
            token = get_current_auth_token()
        if not token:
            raise RuntimeError(
                "Not authenticated. Run `neem init` to refresh your token."
            )
        return await func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def validate_required(*required_params: str) -> Callable[[F], F]:
    """Decorator that validates required string parameters are non-empty.

    Args:
        *required_params: Names of parameters that must be non-empty strings

    Usage:
        @validate_required("graph_id", "document_id")
        async def my_tool(graph_id: str, document_id: str, ...) -> dict:
            # graph_id and document_id are guaranteed non-empty
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for param in required_params:
                value = kwargs.get(param)
                if not value or not str(value).strip():
                    raise ValueError(f"{param} is required and cannot be empty")
            return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def strip_params(*params_to_strip: str) -> Callable[[F], F]:
    """Decorator that strips whitespace from specified string parameters.

    Args:
        *params_to_strip: Names of string parameters to strip

    Usage:
        @strip_params("graph_id", "document_id")
        async def my_tool(graph_id: str, document_id: str, ...) -> dict:
            # graph_id and document_id are already stripped
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for param in params_to_strip:
                if param in kwargs and isinstance(kwargs[param], str):
                    kwargs[param] = kwargs[param].strip()
            return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Home graph — session-scoped default graph_id
# ---------------------------------------------------------------------------
# Persisted to disk so that the home-graph hint survives MCP server reconnects
# and Claude Code /compact restarts within a 24-hour TTL. The in-memory dict
# remains authoritative for hot lookups; disk is only read at module load and
# written on set/clear.

_HOME_GRAPH_TTL_SECONDS = 24 * 60 * 60


def _home_graph_state_path() -> Path:
    override = os.environ.get("MNEMOSYNE_MCP_HOME_GRAPH_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".mnemosyne" / "mcp_home_graphs.json"


def _load_persisted_home_graphs() -> dict[str, tuple[str, float]]:
    path = _home_graph_state_path()
    try:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            return {}
    except Exception:
        return {}

    now = time.time()
    live: dict[str, tuple[str, float]] = {}
    for user_id, entry in raw.items():
        if not isinstance(user_id, str) or not isinstance(entry, dict):
            continue
        graph_id = entry.get("graph_id")
        ts = entry.get("ts")
        if not isinstance(graph_id, str) or not graph_id:
            continue
        if not isinstance(ts, (int, float)):
            continue
        if now - float(ts) > _HOME_GRAPH_TTL_SECONDS:
            continue
        live[user_id] = (graph_id, float(ts))
    return live


def _persist_home_graphs() -> None:
    path = _home_graph_state_path()
    # Per-PID tmp suffix avoids collision when two MCP processes for the same
    # user persist concurrently. Atomic rename guarantees no torn writes; the
    # remaining race (one process's in-memory dict overwriting another's on
    # disk) is documented and acceptable for the single-user case.
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            user_id: {"graph_id": gid, "ts": ts}
            for user_id, (gid, ts) in _home_graphs_with_ts.items()
        }
        tmp_path.write_text(json.dumps(payload))
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of an orphaned tmp file if rename never landed.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


_home_graphs_with_ts: dict[str, tuple[str, float]] = _load_persisted_home_graphs()
_home_graphs: dict[str, str] = {uid: gid for uid, (gid, _ts) in _home_graphs_with_ts.items()}


def set_home_graph(user_id: str, graph_id: str) -> None:
    """Set the session's default graph for a user.

    Persists to disk with a 24h TTL so the home-graph hint survives MCP
    reconnects and Claude Code /compact restarts.
    """
    now = time.time()
    _home_graphs[user_id] = graph_id
    _home_graphs_with_ts[user_id] = (graph_id, now)
    _persist_home_graphs()


def get_home_graph(user_id: str) -> str | None:
    """Get the session's default graph for a user, or None if not set."""
    return _home_graphs.get(user_id)


def clear_home_graph(user_id: str) -> None:
    """Clear the session's default graph for a user."""
    _home_graphs.pop(user_id, None)
    _home_graphs_with_ts.pop(user_id, None)
    _persist_home_graphs()


def resolve_home_graph(func: F) -> F:
    """Decorator that resolves graph_id from the session's home graph if not provided.

    When graph_id is None or empty, looks up the home graph for the authenticated
    user. If found, injects it into kwargs. If not found, raises ValueError with
    instructions to call set_home_graph.

    Usage:
        @server.tool(name="read_document", ...)
        @resolve_home_graph
        async def read_document_tool(graph_id: str | None = None, ...) -> dict:
            # graph_id is guaranteed non-None here
            ...
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        graph_id = kwargs.get("graph_id")
        if not graph_id or not str(graph_id).strip():
            context = kwargs.get("context")
            if context is not None:
                auth = MCPAuthContext.from_context(context)
                if auth.user_id:
                    home = get_home_graph(auth.user_id)
                    if home:
                        kwargs["graph_id"] = home
            if not kwargs.get("graph_id") or not str(kwargs.get("graph_id", "")).strip():
                raise ValueError(
                    "graph_id is required. Either pass it explicitly or set a home graph "
                    "via set_home_graph(graph_id='your-graph-id'). Home graph is persisted "
                    "for 24h and survives MCP reconnects."
                )
        return await func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


__all__ = [
    "require_auth",
    "validate_required",
    "strip_params",
    "set_home_graph",
    "get_home_graph",
    "clear_home_graph",
    "resolve_home_graph",
]
