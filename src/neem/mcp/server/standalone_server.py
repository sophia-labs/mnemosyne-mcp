"""
Standalone MCP server bootstrapper.

This version intentionally strips out all of the legacy tool implementations so
we can rebuild against the new FastAPI backend that runs inside the local
kubectl context. All configuration now points at that backend (either through a
port-forward on localhost or through cluster-injected service variables), and
the server performs a lightweight health probe so we can confirm connectivity
before wiring in fresh tools.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase
from mcp.types import ToolAnnotations
from pydantic import ConfigDict
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from neem.mcp.auth import get_current_auth_token
from neem.mcp.http_client import create_http_client, set_http_client
from neem.mcp.jobs.realtime import RealtimeJobClient
from neem.mcp.tools.basic import register_basic_tools
from neem.mcp.tools.graph_ops import register_graph_ops_tools
from neem.mcp.tools.hocuspocus import register_hocuspocus_tools
from neem.mcp.tools.wire_tools import register_wire_tools
from neem.mcp.tools.geist import register_geist_tools
from neem.mcp.tools.search import register_search_tools
from neem.mcp.tools.surface import register_surface_tools
from neem.mcp.tools.history import register_history_tools
from neem.mcp.tools.delete import register_delete_tool
from neem.mcp.trace import trace, trace_separator
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_dev_user_id, get_internal_service_secret

logger = LoggerFactory.get_logger("mcp.standalone_server")

# Defaults align with `kubectl port-forward svc/mnemosyne-api 8080:80`
# The mnemosyne-api service exposes port 80, targeting pod port 8000
DEFAULT_LOCAL_BACKEND_URL = "http://127.0.0.1:8080"
DEFAULT_WS_PATH = "/ws"
DEFAULT_LOCAL_WS_PORT = 8080
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}
LOCAL_HTTP_PORT_HINTS = {8000, 8080}
BACKEND_URL_ENV_VARS = ("MNEMOSYNE_FASTAPI_URL", "MNEMOSYNE_API_URL")
HOST_PORT_ENV_VARS = (
    ("MNEMOSYNE_FASTAPI_HOST", "MNEMOSYNE_FASTAPI_PORT"),
    ("FASTAPI_SERVICE_HOST", "FASTAPI_SERVICE_PORT"),
)
CHATGPT_DEMO_PROFILE = "chatgpt_demo"
CHATGPT_DEMO_GRAPH_ID_ENV = "MNEMOSYNE_CHATGPT_DEMO_GRAPH_ID"
CHATGPT_OAUTH_AUTH_SERVER_URL_ENV = "MNEMOSYNE_CHATGPT_OAUTH_AUTH_SERVER_URL"
CHATGPT_OAUTH_RESOURCE_URL_ENV = "MNEMOSYNE_CHATGPT_OAUTH_RESOURCE_URL"
CHATGPT_OAUTH_SCOPE_ENV = "MNEMOSYNE_CHATGPT_OAUTH_SCOPE"
CHATGPT_DEMO_READ_TOOLS: frozenset[str] = frozenset({
    "search_documents",
    "search_blocks",
    "read_document",
    "document_digest",
    "get_workspace",
    "read_blocks",
    "get_block",
    "query_blocks",
})
CHATGPT_DEMO_WRITE_TOOLS: frozenset[str] = frozenset({
    "write_document",
    "insert_blocks",
    "update_blocks",
    "edit_block_text",
})


async def _health_response(_request) -> JSONResponse:
    """Lightweight process health for container and ingress probes."""
    return JSONResponse({"status": "ok"})


def _auth_mode() -> str:
    return (os.getenv("MNEMOSYNE_MCP_AUTH_MODE", "").strip().lower() or "auto")


def _is_chatgpt_oauth_mode() -> bool:
    return _auth_mode() == "chatgpt_oauth"


def _chatgpt_oauth_scope() -> str:
    return (os.getenv(CHATGPT_OAUTH_SCOPE_ENV) or "mnemosyne.mcp.read").strip()


def _chatgpt_oauth_auth_server_url() -> str:
    value = (os.getenv(CHATGPT_OAUTH_AUTH_SERVER_URL_ENV) or "").strip().rstrip("/")
    if not value:
        raise RuntimeError(
            f"{CHATGPT_OAUTH_AUTH_SERVER_URL_ENV} must be set when MNEMOSYNE_MCP_AUTH_MODE=chatgpt_oauth."
        )
    return value


def _enforce_strict_tool_argument_validation() -> None:
    """Fail closed on unknown MCP tool arguments.

    FastMCP argument models default to ignoring extra fields, which can silently
    drop misspelled or stale parameter names (e.g., after tool schema changes).
    Override ArgModelBase to forbid extras so callers get explicit validation
    errors instead of accidental fallback behavior.
    """
    ArgModelBase.model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


def _protected_resource_metadata_path(mount_path: str) -> str:
    suffix = "/.well-known/oauth-protected-resource"
    return f"{mount_path}{suffix}" if mount_path else suffix


def _resource_url(mount_path: str) -> str:
    configured = (os.getenv(CHATGPT_OAUTH_RESOURCE_URL_ENV) or "").strip()
    if configured:
        return configured
    raise RuntimeError(
        f"{CHATGPT_OAUTH_RESOURCE_URL_ENV} must be set when MNEMOSYNE_MCP_AUTH_MODE=chatgpt_oauth."
    )


def _protected_resource_metadata_url(mount_path: str) -> str:
    resource = _resource_url(mount_path)
    parsed = urlparse(resource)
    return urlunparse(
        parsed._replace(
            path=_protected_resource_metadata_path(mount_path),
            params="",
            query="",
            fragment="",
        )
    )


async def _oauth_protected_resource_response(request: Request) -> JSONResponse:
    mount_path = getattr(request.app.state, "mcp_mount_path", "")
    auth_server = _chatgpt_oauth_auth_server_url()
    return JSONResponse(
        {
            "resource": _resource_url(mount_path),
            "authorization_servers": [auth_server],
            "bearer_methods_supported": ["header"],
            "scopes_supported": [_chatgpt_oauth_scope()],
        }
    )


def build_streamable_http_app(mcp_server: FastMCP) -> Starlette:
    """Wrap the FastMCP app with a simple health endpoint."""
    transport_app = mcp_server.streamable_http_app()
    mount_path = os.getenv("MCP_ROOT_PATH_PREFIX", "").strip()
    if mount_path and not mount_path.startswith("/"):
        mount_path = f"/{mount_path}"
    mount_path = mount_path.rstrip("/")
    if mount_path == "/":
        mount_path = ""

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        async with transport_app.router.lifespan_context(transport_app):
            yield

    routes = [
        Route("/health", endpoint=_health_response),
        Route("/.well-known/oauth-protected-resource", endpoint=_oauth_protected_resource_response),
    ]
    if mount_path:
        routes.append(
            Route(
                _protected_resource_metadata_path(mount_path),
                endpoint=_oauth_protected_resource_response,
            )
        )
        routes.append(Mount(mount_path, app=transport_app))
    else:
        routes.append(Mount("/", app=transport_app))

    app = Starlette(
        lifespan=lifespan,
        routes=routes,
    )
    app.state.mcp_mount_path = mount_path

    class _ChatGPTOAuthChallengeMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next) -> Response:
            if _is_chatgpt_oauth_mode():
                path = request.url.path
                protected_path = f"{mount_path}/mcp" if mount_path else "/mcp"
                if path.startswith(protected_path):
                    auth_header = request.headers.get("authorization", "")
                    if not auth_header.startswith("Bearer "):
                        return JSONResponse(
                            {"error": "authentication_required"},
                            status_code=401,
                            headers={
                                "WWW-Authenticate": (
                                    'Bearer realm="mnemosyne", '
                                    f'resource_metadata="{_protected_resource_metadata_url(mount_path)}"'
                                )
                            },
                        )
            return await call_next(request)

    app.add_middleware(_ChatGPTOAuthChallengeMiddleware)
    return app


@dataclass(frozen=True)
class BackendConfig:
    """Resolved backend details for the MCP server."""

    base_url: str
    health_path: str
    websocket_url: Optional[str]

    @property
    def health_url(self) -> str:
        """Expand the health path into a full URL."""
        if not self.health_path:
            return ""
        if self.health_path.startswith("http://") or self.health_path.startswith("https://"):
            return self.health_path
        path = self.health_path if self.health_path.startswith("/") else f"/{self.health_path}"
        return f"{self.base_url}{path}"

    @property
    def has_websocket(self) -> bool:
        """True if backend advertises a WebSocket endpoint."""
        return bool(self.websocket_url)


def _normalize_base_url(value: str) -> str:
    """Normalize host inputs into an http(s) URL."""
    candidate = value.strip()
    if not candidate:
        raise ValueError("FastAPI base URL is empty")
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def _build_url_from_host_vars() -> Optional[str]:
    """Allow configuring host/port separately (useful for kubectl/local clusters)."""
    scheme = os.getenv("MNEMOSYNE_FASTAPI_SCHEME", "http").strip() or "http"
    for host_var, port_var in HOST_PORT_ENV_VARS:
        host = os.getenv(host_var)
        if not host:
            continue
        host = host.strip()
        port = (os.getenv(port_var) or "").strip()
        if port:
            return f"{scheme}://{host}:{port}"
        return f"{scheme}://{host}"
    return None


def _build_websocket_url(base_url: str, path: str, *, port_override: Optional[int] = None) -> str:
    """Derive a ws:// URL from the HTTP base, optionally overriding the port."""
    parsed = urlparse(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    normalized_path = path if path.startswith("/") else f"/{path}"
    host = parsed.hostname or ""
    if not host:
        raise ValueError("Unable to parse host from FastAPI base URL")
    port = port_override if port_override is not None else parsed.port
    netloc = _format_host_port(host, port)
    return urlunparse((ws_scheme, netloc, normalized_path, "", "", ""))


def _normalize_ws_url(value: str) -> str:
    """Ensure explicit WebSocket URLs are well-formed."""
    candidate = value.strip()
    if not candidate:
        raise ValueError("FastAPI WebSocket URL is empty")
    if not (candidate.startswith("ws://") or candidate.startswith("wss://")):
        raise ValueError("FastAPI WebSocket URL must start with ws:// or wss://")
    return candidate.rstrip("/")


def _format_host_port(host: str, port: Optional[int]) -> str:
    """Format a host[:port] string, handling IPv6 literals."""
    host_part = host
    if ":" in host and not host.startswith("["):
        host_part = f"[{host}]"
    if port:
        return f"{host_part}:{port}"
    return host_part


def _resolve_ws_port_override(base_url: str) -> Optional[int]:
    """Return an overridden port for WebSocket derivation when configured or implied."""
    env_var = "MNEMOSYNE_FASTAPI_WS_PORT"
    raw_override = os.getenv(env_var)
    if raw_override:
        try:
            port = int(raw_override)
            if port <= 0 or port > 65535:
                raise ValueError
        except ValueError:
            logger.warning(
                "Ignoring invalid WebSocket port override",
                extra_context={"variable": env_var, "value": raw_override},
            )
        else:
            return port

    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    base_port = parsed.port
    if host in LOOPBACK_HOSTS and base_port in LOCAL_HTTP_PORT_HINTS and base_port != DEFAULT_LOCAL_WS_PORT:
        logger.debug(
            "Assuming split WebSocket port for localhost FastAPI",
            extra_context={"http_port": base_port, "ws_port": DEFAULT_LOCAL_WS_PORT},
        )
        return DEFAULT_LOCAL_WS_PORT
    return None


def resolve_backend_config() -> BackendConfig:
    """Resolve backend connectivity details from the environment."""
    health_path = os.getenv("MNEMOSYNE_FASTAPI_HEALTH_PATH", "/health")

    base_url: Optional[str] = None
    for env_var in BACKEND_URL_ENV_VARS:
        raw_url = os.getenv(env_var)
        if raw_url:
            base_url = _normalize_base_url(raw_url)
            logger.info(
                "Using FastAPI backend URL from environment",
                extra_context={"variable": env_var, "url": base_url},
            )
            break

    if base_url is None:
        host_based = _build_url_from_host_vars()
        if host_based:
            base_url = _normalize_base_url(host_based)
            logger.info(
                "Using FastAPI backend host/port configuration",
                extra_context={"url": base_url},
            )

    if base_url is None:
        base_url = DEFAULT_LOCAL_BACKEND_URL
        logger.info(
            "FastAPI backend URL not provided; defaulting to localhost port-forward",
            extra_context={"url": base_url},
        )

    websocket_url: Optional[str] = None
    ws_disabled = os.getenv("MNEMOSYNE_FASTAPI_WS_DISABLE", "").strip().lower() in {"1", "true", "yes"}
    if not ws_disabled:
        ws_override = os.getenv("MNEMOSYNE_FASTAPI_WS_URL")
        ws_path = os.getenv("MNEMOSYNE_FASTAPI_WS_PATH", DEFAULT_WS_PATH)
        ws_port_override = _resolve_ws_port_override(base_url)

        try:
            if ws_override:
                websocket_url = _normalize_ws_url(ws_override)
                logger.info(
                    "Using explicit FastAPI WebSocket URL from environment",
                    extra_context={"url": websocket_url},
                )
            else:
                websocket_url = _build_websocket_url(base_url, ws_path, port_override=ws_port_override)
                logger.info(
                    "Derived FastAPI WebSocket URL from base URL",
                    extra_context={"url": websocket_url, "port_override": ws_port_override},
                )
        except ValueError as exc:
            logger.warning(
                "Invalid WebSocket configuration provided; disabling realtime push",
                extra_context={"error": str(exc)},
            )
            websocket_url = None
    else:
        logger.info("FastAPI WebSocket integration disabled via environment flag")

    return BackendConfig(
        base_url=base_url,
        health_path=health_path,
        websocket_url=websocket_url,
    )


def verify_backend_connectivity(config: BackendConfig) -> None:
    """
    Perform a lightweight health probe.

    This does not fail server startup; it simply reports whether we could reach
    the FastAPI backend so developers know if their kubectl context is wired up.
    """
    health_url = config.health_url
    if not health_url:
        logger.debug("Health check skipped: no health path configured")
        return

    try:
        response = httpx.get(health_url, timeout=5.0)
        response.raise_for_status()
        logger.info(
            "FastAPI backend reachable",
            extra_context={"health_url": health_url, "status_code": response.status_code},
        )
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "FastAPI backend health endpoint returned an error",
            extra_context={"health_url": health_url, "status_code": exc.response.status_code},
        )
    except httpx.RequestError as exc:
        logger.warning(
            "Unable to reach FastAPI backend",
            extra_context={"health_url": health_url, "error": str(exc)},
        )


# Tools available in the "lite" profile — read-only graph research for
# small-model agents (pathfinders, Choreograph workers, local models).
LITE_TOOLS: frozenset[str] = frozenset({
    # Read
    "get_workspace",
    "read_document",
    "read_blocks",
    "document_digest",
    "get_block",
    "query_blocks",
    "search_documents",
    "search_blocks",
    # History
    "get_document_history",
    "read_document_at_snapshot",
    # Wires (read-only)
    "get_wires",
    "traverse_wires",
    # Geist (read-only)
    "music",
    "get_important_blocks",
    "recall",
})

# Tools excluded from the "hivemind" profile — Sophia agent sessions.
# These are browser-only, operator-only, or unused by agents.
HIVEMIND_EXCLUDED: frozenset[str] = frozenset({
    "get_session_state",    # browser UI state — irrelevant to agents
    "upload_artifact",      # file upload — not used by hivemind
    "ingest_artifact",      # artifact ingestion — not used by hivemind
    "dump_chat",            # browser chat log dump — not used by agents
    "quick_orient",         # deprecated lightweight orient — unused
    "reindex_graph",        # admin maintenance — agents shouldn't trigger re-indexing
    "list_wire_predicates", # predicate taxonomy is in CLAUDE.md — no need for a tool call
    "archive_memories",     # rare maintenance op — switch to full profile when needed
    "surface",              # browser chat UI card — irrelevant to agent sessions
})

# Tools available in the "angel" profile — haiku-class subagents with
# inherited parent context. Can read, search, wire, append, comment,
# and value — but cannot edit/delete existing content or use
# continuity tools (sing, remember, care) that imply persistent identity.
ANGEL_TOOLS: frozenset[str] = frozenset({
    # Read & search (same as lite)
    "get_workspace",
    "read_document",
    "read_blocks",
    "document_digest",
    "get_block",
    "query_blocks",
    "search_documents",
    "search_blocks",
    # History
    "get_document_history",
    "read_document_at_snapshot",
    # Wires (read + write)
    "get_wires",
    "traverse_wires",
    "create_wires",
    "list_wire_predicates",  # Angels don't have CLAUDE.md predicate reference
    # Write (additive only — no edit/delete)
    "insert_blocks",
    "write_document",
    "edit_comment",
    # Geist (attunement + valuation, no continuity)
    "music",
    "get_important_blocks",
    "recall",
    "value",
    "get_block_values",
    "get_values",
})


def _chatgpt_demo_read_annotations() -> ToolAnnotations:
    return ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )


def _chatgpt_demo_write_annotations() -> ToolAnnotations:
    return ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )


def _chatgpt_demo_meta() -> dict[str, Any]:
    # FastMCP accepts arbitrary per-tool metadata. Choose the ChatGPT-facing
    # security scheme based on the deployment auth mode.
    if _is_chatgpt_oauth_mode():
        auth_server = _chatgpt_oauth_auth_server_url()
        security = [
            {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": f"{auth_server}/authorize",
                        "tokenUrl": f"{auth_server}/token",
                        "scopes": {
                            _chatgpt_oauth_scope(): "Use Mnemosyne through ChatGPT.",
                        },
                    }
                },
            }
        ]
    else:
        security = [{"type": "noauth"}]
    return {
        "securitySchemes": security,
        "_meta": {"securitySchemes": security},
    }


def _require_chatgpt_demo_graph_id() -> str:
    graph_id = os.getenv(CHATGPT_DEMO_GRAPH_ID_ENV, "").strip()
    if not graph_id:
        raise RuntimeError(
            f"{CHATGPT_DEMO_GRAPH_ID_ENV} must be set when MCP_PROFILE={CHATGPT_DEMO_PROFILE}.",
        )
    return graph_id


def _parse_json_payload(raw: Any, *, tool_name: str) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise RuntimeError(f"{tool_name} returned unsupported payload type: {type(raw)!r}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{tool_name} returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{tool_name} returned non-object JSON payload")
    return parsed


def _register_chatgpt_demo_tools(mcp_server: FastMCP) -> None:
    demo_graph_id = _require_chatgpt_demo_graph_id()
    tool_manager = mcp_server._tool_manager  # type: ignore[attr-defined]
    original_tools = tool_manager._tools
    enabled_tools = CHATGPT_DEMO_READ_TOOLS | (
        CHATGPT_DEMO_WRITE_TOOLS if _is_chatgpt_oauth_mode() else frozenset()
    )
    missing = sorted(enabled_tools - set(original_tools.keys()))
    if missing:
        raise RuntimeError(f"chatgpt_demo profile missing required tools: {', '.join(missing)}")

    original_fns: dict[str, Callable[..., Any]] = {
        name: original_tools[name].fn for name in enabled_tools
    }

    for name in list(original_tools.keys()):
        try:
            mcp_server.remove_tool(name)
        except (KeyError, ToolError):
            pass

    read_annotations = _chatgpt_demo_read_annotations()
    write_annotations = _chatgpt_demo_write_annotations()
    meta = _chatgpt_demo_meta()

    @mcp_server.tool(
        name="search_documents",
        title="Search Documents",
        description=(
            "Use this when you know roughly what document you want in the Mnemosyne demo "
            "workspace and need to find it quickly by title or path."
        ),
        annotations=read_annotations,
        meta=meta,
        structured_output=True,
    )
    async def chatgpt_search_documents_tool(
        query: str,
        mode: str = "auto",
        limit: int = 10,
        context: Context | None = None,
    ) -> dict[str, Any]:
        raw = await original_fns["search_documents"](
            graph_id=demo_graph_id,
            query=query,
            mode=mode,
            limit=min(max(limit, 1), 20),
            folder_id=None,
            include_folders=False,
            context=context,
        )
        payload = _parse_json_payload(raw, tool_name="search_documents")
        results = []
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "document_id": item.get("document_id"),
                    "title": item.get("title"),
                    "folder_path": item.get("folder_path"),
                    "match_type": item.get("match_type"),
                }
            )
        return {
            "query": payload.get("query", query),
            "mode": payload.get("mode", mode),
            "count": len(results),
            "results": results,
        }

    @mcp_server.tool(
        name="search_blocks",
        title="Search Blocks",
        description=(
            "Use this when you need to find passages or notes by content in the Mnemosyne "
            "demo workspace."
        ),
        annotations=read_annotations,
        meta=meta,
        structured_output=True,
    )
    async def chatgpt_search_blocks_tool(
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
        document_id: str | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        raw = await original_fns["search_blocks"](
            graph_id=demo_graph_id,
            query=query,
            mode=mode,
            limit=min(max(limit, 1), 20),
            doc_filter=(document_id.strip() if document_id else None),
            context=context,
        )
        payload = _parse_json_payload(raw, tool_name="search_blocks")
        results = []
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "document_id": item.get("document_id"),
                    "document_title": item.get("document_title"),
                    "block_id": item.get("block_id"),
                    "snippet": item.get("text_preview"),
                    "match_source": item.get("match_source"),
                }
            )
        return {
            "query": payload.get("query", query),
            "mode": mode,
            "count": len(results),
            "results": results,
        }

    @mcp_server.tool(
        name="read_document",
        title="Read Document",
        description=(
            "Use this when you already know which document you want and need its contents "
            "in Markdown."
        ),
        annotations=read_annotations,
        meta=meta,
        structured_output=True,
    )
    async def chatgpt_read_document_tool(
        document_id: str,
        context: Context | None = None,
    ) -> dict[str, Any]:
        result = await original_fns["read_document"](
            graph_id=demo_graph_id,
            document_id=document_id,
            format="markdown",
            context=context,
        )
        if not isinstance(result, dict):
            raise RuntimeError("read_document returned unsupported payload")
        return {
            "document_id": result.get("document_id", document_id),
            "format": "markdown",
            "content": result.get("content", ""),
            "updated_at": result.get("updated_at"),
        }

    @mcp_server.tool(
        name="document_digest",
        title="Document Digest",
        description=(
            "Use this when you want a quick summary of a document before deciding whether "
            "to read the full text."
        ),
        annotations=read_annotations,
        meta=meta,
        structured_output=True,
    )
    async def chatgpt_document_digest_tool(
        document_id: str,
        top_valued: int = 3,
        context: Context | None = None,
    ) -> dict[str, Any]:
        result = await original_fns["document_digest"](
            graph_id=demo_graph_id,
            document_id=document_id,
            top_valued=top_valued,
            context=context,
        )
        if not isinstance(result, dict):
            raise RuntimeError("document_digest returned unsupported payload")
        return {
            "metadata": result.get("metadata"),
            "size": result.get("size"),
            "freshness": result.get("freshness"),
            "headings": result.get("headings"),
            "wire_summary": result.get("wire_summary"),
            "valuation_summary": result.get("valuation_summary"),
        }

    @mcp_server.tool(
        name="get_workspace",
        title="Get Workspace",
        description=(
            "Use this when you need the folder and document structure of the Mnemosyne "
            "workspace before choosing what to read or edit."
        ),
        annotations=read_annotations,
        meta=meta,
        structured_output=True,
    )
    async def chatgpt_get_workspace_tool(
        depth: int = 1,
        folder_id: str | None = None,
        folders_only: bool = False,
        min_score: str | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        raw = await original_fns["get_workspace"](
            graph_id=demo_graph_id,
            depth=depth,
            folder_id=folder_id,
            folders_only=folders_only,
            min_score=min_score,
            context=context,
        )
        return _parse_json_payload(raw, tool_name="get_workspace")

    @mcp_server.tool(
        name="read_blocks",
        title="Read Blocks",
        description=(
            "Use this when you need a paginated, block-by-block read of a document, "
            "including block IDs for targeted follow-up reads or edits."
        ),
        annotations=read_annotations,
        meta=meta,
        structured_output=True,
    )
    async def chatgpt_read_blocks_tool(
        document_id: str,
        offset: int = 0,
        limit: int = 50,
        format: str = "markdown",
        block_id: str | None = None,
        include_ids: bool = True,
        context: Context | None = None,
    ) -> dict[str, Any]:
        result = await original_fns["read_blocks"](
            graph_id=demo_graph_id,
            document_id=document_id,
            offset=offset,
            limit=min(max(limit, 1), 100),
            format=format,
            block_id=block_id,
            include_ids=include_ids,
            context=context,
        )
        if not isinstance(result, dict):
            raise RuntimeError("read_blocks returned unsupported payload")
        return result

    @mcp_server.tool(
        name="get_block",
        title="Get Block",
        description=(
            "Use this when you already have a block ID and need the exact block content "
            "or context for targeted reasoning or editing."
        ),
        annotations=read_annotations,
        meta=meta,
        structured_output=True,
    )
    async def chatgpt_get_block_tool(
        document_id: str,
        block_id: str,
        format: str = "markdown",
        context: Context | None = None,
    ) -> dict[str, Any]:
        result = await original_fns["get_block"](
            graph_id=demo_graph_id,
            document_id=document_id,
            block_id=block_id,
            format=format,
            context=context,
        )
        if not isinstance(result, dict):
            raise RuntimeError("get_block returned unsupported payload")
        return result

    @mcp_server.tool(
        name="query_blocks",
        title="Query Blocks",
        description=(
            "Use this when you need to find blocks in one document by structure or text, "
            "or when you need to resolve block IDs for targeted edits."
        ),
        annotations=read_annotations,
        meta=meta,
        structured_output=True,
    )
    async def chatgpt_query_blocks_tool(
        document_id: str,
        block_type: str | None = None,
        heading_level: int | None = None,
        indent: int | None = None,
        indent_gte: int | None = None,
        indent_lte: int | None = None,
        list_type: str | None = None,
        checked: bool | None = None,
        text_contains: str | None = None,
        limit: int = 50,
        queries: list[dict[str, Any]] | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        result = await original_fns["query_blocks"](
            graph_id=demo_graph_id,
            document_id=document_id,
            block_type=block_type,
            heading_level=heading_level,
            indent=indent,
            indent_gte=indent_gte,
            indent_lte=indent_lte,
            list_type=list_type,
            checked=checked,
            text_contains=text_contains,
            limit=min(max(limit, 1), 100),
            queries=queries,
            context=context,
        )
        if not isinstance(result, dict):
            raise RuntimeError("query_blocks returned unsupported payload")
        return result

    if _is_chatgpt_oauth_mode():
        @mcp_server.tool(
            name="write_document",
            title="Write Document",
            description=(
                "Use this when you want to create a document or replace an existing "
                "document's full contents in one step."
            ),
            annotations=write_annotations,
            meta=meta,
            structured_output=True,
        )
        async def chatgpt_write_document_tool(
            document_id: str,
            content: str,
            comments: dict[str, Any] | None = None,
            await_durable: bool = True,
            context: Context | None = None,
        ) -> dict[str, Any]:
            result = await original_fns["write_document"](
                graph_id=demo_graph_id,
                document_id=document_id,
                content=content,
                comments=comments,
                await_durable=await_durable,
                context=context,
            )
            if not isinstance(result, dict):
                raise RuntimeError("write_document returned unsupported payload")
            return result

        @mcp_server.tool(
            name="insert_blocks",
            title="Insert Blocks",
            description=(
                "Use this when you want to append or insert new content into an existing "
                "document without replacing the whole document."
            ),
            annotations=write_annotations,
            meta=meta,
            structured_output=True,
        )
        async def chatgpt_insert_blocks_tool(
            document_id: str,
            content: str,
            block_id: str | None = None,
            index: int | None = None,
            position: str = "after",
            context: Context | None = None,
        ) -> dict[str, Any]:
            result = await original_fns["insert_blocks"](
                graph_id=demo_graph_id,
                document_id=document_id,
                content=content,
                block_id=block_id,
                index=index,
                position=position,
                context=context,
            )
            if not isinstance(result, dict):
                raise RuntimeError("insert_blocks returned unsupported payload")
            return result

        @mcp_server.tool(
            name="update_blocks",
            title="Update Blocks",
            description=(
                "Use this when you need to replace or reformat specific blocks by block ID "
                "without rewriting the full document."
            ),
            annotations=write_annotations,
            meta=meta,
            structured_output=True,
        )
        async def chatgpt_update_blocks_tool(
            document_id: str,
            block_id: str | None = None,
            attributes: dict[str, Any] | None = None,
            xml_content: str | None = None,
            updates: list[dict[str, Any]] | None = None,
            context: Context | None = None,
        ) -> dict[str, Any]:
            result = await original_fns["update_blocks"](
                graph_id=demo_graph_id,
                document_id=document_id,
                block_id=block_id,
                attributes=attributes,
                xml_content=xml_content,
                updates=updates,
                context=context,
            )
            if not isinstance(result, dict):
                raise RuntimeError("update_blocks returned unsupported payload")
            return result

        @mcp_server.tool(
            name="edit_block_text",
            title="Edit Block Text",
            description=(
                "Use this when you need a surgical text edit within one block and you "
                "already know the block ID."
            ),
            annotations=write_annotations,
            meta=meta,
            structured_output=True,
        )
        async def chatgpt_edit_block_text_tool(
            document_id: str,
            block_id: str,
            operations: list[dict[str, Any]] | None = None,
            find: str | None = None,
            replace: str | None = None,
            occurrence: int = 1,
            context: Context | None = None,
        ) -> dict[str, Any]:
            result = await original_fns["edit_block_text"](
                graph_id=demo_graph_id,
                document_id=document_id,
                block_id=block_id,
                operations=operations,
                find=find,
                replace=replace,
                occurrence=occurrence,
                context=context,
            )
            if not isinstance(result, dict):
                raise RuntimeError("edit_block_text returned unsupported payload")
            return result

    logger.info(
        "ChatGPT demo profile applied",
        extra_context={
            "tool_count": len(mcp_server._tool_manager._tools),
            "demo_graph_id": demo_graph_id,
        },
    )


def create_standalone_mcp_server(profile: str | None = None) -> FastMCP:
    """
    Create an MCP server bound to the local FastAPI backend.

    Tool registration is intentionally deferred until the new architecture is
    ready; for now we only set up connectivity and metadata.

    Args:
        profile: Optional profile name ("lite" for reduced tool set).
                 Falls back to MCP_PROFILE env var if not provided.
    """
    active_profile = profile or os.getenv("MCP_PROFILE", "")
    _enforce_strict_tool_argument_validation()
    if _is_chatgpt_oauth_mode():
        # Fail closed for ChatGPT OAuth deployments:
        # - default to chatgpt_demo when unset
        # - reject conflicting explicit profiles
        if not active_profile:
            active_profile = CHATGPT_DEMO_PROFILE
        elif active_profile != CHATGPT_DEMO_PROFILE:
            raise RuntimeError(
                "MNEMOSYNE_MCP_AUTH_MODE=chatgpt_oauth requires MCP_PROFILE=chatgpt_demo"
            )
    backend_config = resolve_backend_config()

    trace_separator("MCP SERVER STARTUP")
    trace("Backend config resolved", {
        "base_url": backend_config.base_url,
        "health_url": backend_config.health_url,
        "websocket_url": backend_config.websocket_url,
        "has_websocket": backend_config.has_websocket,
    })
    trace("Environment", {
        "MNEMOSYNE_FASTAPI_URL": os.getenv("MNEMOSYNE_FASTAPI_URL"),
        "MNEMOSYNE_FASTAPI_WS_DISABLE": os.getenv("MNEMOSYNE_FASTAPI_WS_DISABLE"),
        "MNEMOSYNE_FASTAPI_WS_URL": os.getenv("MNEMOSYNE_FASTAPI_WS_URL"),
        "MNEMOSYNE_DEV_USER_ID": os.getenv("MNEMOSYNE_DEV_USER_ID"),
        "has_dev_token": bool(os.getenv("MNEMOSYNE_DEV_TOKEN")),
    })

    mcp_server = FastMCP(
        name="Mnemosyne Knowledge Graph",
        instructions=(
            "This MCP server provides real-time access to Mnemosyne knowledge graphs. "
            "Available tools:\n\n"
            "**Quick-start workflow:**\n"
            "(1) Orient — context_bundle (single call) or get_user_location �� set_home_graph → get_workspace.\n"
            "(2) Read — read_document to see content and block IDs.\n"
            "(3) Edit — use block tools for surgical changes or write_document for full replacement.\n"
            "(4) Connect — create_wire with predicates from list_wire_predicates, using block IDs for precision.\n\n"
            "**Home graph:** Call set_home_graph(graph_id) or use context_bundle (which auto-sets it). "
            "Once set, graph_id becomes optional on all tools — omit it to use the home graph. "
            "Pass graph_id explicitly to override for a single call.\n\n"
            "**Orientation (use in this order):**\n"
            "- context_bundle: Single-call attunement — returns location, Song, recall, agent document, "
            "important blocks, and workspace in one response. Auto-sets home graph. "
            "Pass agent_name to include identity document (e.g. agent_name='gamma').\n"
            "- get_user_location: Where is the user right now? Returns just graph_id and document_id (minimal tokens)\n"
            "- get_workspace: What's in this graph? Returns folder/file structure (default depth=1, "
            "deeper folders show document counts). Use folder_id to drill into a subtree, "
            "min_score to filter by document-level valuation scores\n"
            "- get_session_state: Full UI state including tabs and preferences (large payload, rarely needed)\n\n"
            "**SPARQL:** Use `PREFIX doc: <http://mnemosyne.dev/doc#>` (NEVER `urn:mnemosyne:schema:doc:`). "
            "Document type is `doc:TipTapDocument`. Wires use `PREFIX mnemo: <http://mnemosyne.ai/vocab#>`. "
            "Load the `sparql` skill for full namespace reference, predicates, and query patterns.\n\n"
            "Documents are synced in real-time via Y.js CRDT, so changes appear "
            "immediately in the Mnemosyne web UI.\n\n"
            "**Geist (Sophia Memory Tools):**\n"
            "Memory, valuation, and self-narrative tools for agent continuity.\n"
            "- Orientation flow: context_bundle() (preferred) or get_user_location → music() → recall() → get_workspace()\n"
            "- music/sing: Read/write the Song (narrative orientation before structural orientation)\n"
            "- remember/recall/care: Working memory queue (FIFO, numbered, append-only)\n"
            "- value/get_block_values: Block-level importance (0-5) and valence (-5 to +5) scoring\n"
            "- get_values/revaluate: Read/update scoring configuration\n"
            "- recall only searches the memory queue — use search_blocks for cross-document content discovery\n"
            "- value works on any block in any document, not just the queue\n"
            "- remember is for the agent's own working memory; append_to_document is for user-facing content\n"
            "- Wires express relationships between things; valuation expresses judgment about a single thing\n\n"
            "**Search:**\n"
            "- search_documents: Fast title/path search against workspace (no SPARQL needed). Modes: auto (default), exact, substring, regex\n"
            "- search_blocks: Cross-document content search. Modes: hybrid (default, lexical+semantic in parallel), lexical, semantic\n"
            "- query_blocks: Single-document structural filter (block_type, heading_level, indent, list_type, checked, text_contains). "
            "CRDT-native, instant, no backend round-trip. Use for structural navigation within one document; use search_blocks for cross-document discovery\n"
            "- reindex_graph: Re-embed all documents (admin/maintenance, auto-indexes on save)\n"
            "- recall with query param searches only the memory queue\n\n"
            "**TipTap XML Reference:**\n"
            "Block types: paragraph, heading (level=\"1-3\"), bulletList, orderedList, blockquote, "
            "codeBlock (language=\"...\"), taskList (taskItem checked=\"true\"), horizontalRule, "
            "image (src=\"...\", alt=\"...\")\n"
            "Inline marks (nestable): strong, em, strike, code, mark (highlight), a (href=\"...\"), "
            "footnote (data-footnote-content=\"...\"), commentMark (data-comment-id=\"...\")\n"
            "Container blocks (blockquote, tableCell, tableHeader) require <paragraph> children — "
            "they cannot contain inline text directly.\n\n"
            "**Write Tool Guidance:**\n"
            "Read tools auto-reconnect for freshness (2s staleness threshold, 1 retry on sync timeout). "
            "Write tools use a persistent cached channel — always call a read tool first "
            "(read_document or get_block) to sync the channel before writing. "
            "CRDT merge prevents corruption, but writing without reading first may silently "
            "overwrite another agent's recent changes.\n\n"
            "**Inline Valuations:**\n"
            "Embed valuation markers in content — stripped before CRDT write, applied automatically after block IDs assigned. "
            "Markdown/plain text (end of line): `{!3}` importance, `{!,+2}` valence, `{!4,-3}` both. "
            "XML: `data-val-importance` and `data-val-valence` attributes on block elements. "
            "Fire-and-forget (errors never fail the write). Supported in write_document, insert_blocks, update_blocks. "
            "Markers inside code fences are preserved as literal text. Invalid ranges are ignored.\n\n"
            "When making function calls using tools that accept array or object parameters "
            "ensure those are structured using JSON."
        ),
        stateless_http=True,
    )

    # Store the resolved config so the eventual tool layer can reuse it.
    mcp_server._backend_config = backend_config  # type: ignore[attr-defined]

    # Shared httpx client with connection pooling — replaces per-request
    # AsyncClient instantiation across all tool modules.
    http_client = create_http_client()
    set_http_client(http_client)
    mcp_server._http_client = http_client  # type: ignore[attr-defined]
    auth_mode = _auth_mode()
    hosted_mode = auth_mode in {"hosted", "sidecar", "public", "demo_noauth"}

    if backend_config.has_websocket and not hosted_mode:
        dev_uid = get_dev_user_id()
        internal_secret = get_internal_service_secret()
        trace("Creating RealtimeJobClient", {
            "websocket_url": backend_config.websocket_url,
            "dev_user_id": dev_uid,
            "has_internal_secret": bool(internal_secret),
            "token_provider": "get_current_auth_token",
        })
        job_client = RealtimeJobClient(
            websocket_url=backend_config.websocket_url,  # type: ignore[arg-type]
            token_provider=get_current_auth_token,
            dev_user_id=dev_uid,
            internal_service_secret=internal_secret,
        )
        mcp_server._job_stream = job_client  # type: ignore[attr-defined]
        trace("RealtimeJobClient created and attached to server")
        logger.info(
            "Realtime job streaming enabled",
            extra_context={"websocket_url": backend_config.websocket_url},
        )
    else:
        if backend_config.has_websocket and hosted_mode:
            trace("WebSocket job stream DISABLED in hosted auth mode")
            logger.info(
                "Realtime job streaming disabled in hosted auth mode; using HTTP polling",
                extra_context={"auth_mode": auth_mode},
            )
        else:
            trace("WebSocket DISABLED — will use HTTP polling only")
            logger.info("Realtime job streaming disabled (WebSocket URL not configured)")

    verify_backend_connectivity(backend_config)

    register_basic_tools(mcp_server)
    register_graph_ops_tools(mcp_server)
    register_hocuspocus_tools(mcp_server)
    register_wire_tools(mcp_server)
    register_geist_tools(mcp_server)
    register_search_tools(mcp_server)
    register_surface_tools(mcp_server)
    register_history_tools(mcp_server)
    register_delete_tool(mcp_server)  # unified delete — must be after other registrations

    # --- Profile filtering: keep only allowlisted tools for lite profiles ---
    if active_profile == "lite":
        # Access internal tool registry (sync) — list_tools() is async.
        all_names = set(mcp_server._tool_manager._tools.keys())
        to_remove = all_names - LITE_TOOLS
        for name in to_remove:
            try:
                mcp_server.remove_tool(name)
            except (KeyError, ToolError):
                pass
        logger.info(
            "Lite profile applied",
            extra_context={
                "kept": len(all_names - to_remove),
                "removed": len(to_remove),
            },
        )

    elif active_profile == "angel":
        all_names = set(mcp_server._tool_manager._tools.keys())
        to_remove = all_names - ANGEL_TOOLS
        for name in to_remove:
            try:
                mcp_server.remove_tool(name)
            except (KeyError, ToolError):
                pass
        logger.info(
            "Angel profile applied",
            extra_context={
                "kept": len(all_names - to_remove),
                "removed": len(to_remove),
            },
        )

    elif active_profile == "hivemind":
        for name in HIVEMIND_EXCLUDED:
            try:
                mcp_server.remove_tool(name)
            except (KeyError, ToolError):
                pass
        logger.info(
            "Hivemind profile applied",
            extra_context={"excluded": len(HIVEMIND_EXCLUDED)},
        )
    elif active_profile == CHATGPT_DEMO_PROFILE:
        _register_chatgpt_demo_tools(mcp_server)

    # Remove excluded tools based on MCP_EXCLUDED_TOOLS env var.
    # Comma-separated list of tool names, e.g. "export_document,upload_artifact,sparql_update"
    excluded_raw = os.getenv("MCP_EXCLUDED_TOOLS", "")
    if excluded_raw.strip():
        excluded_names = [name.strip() for name in excluded_raw.split(",") if name.strip()]
        for name in excluded_names:
            try:
                mcp_server.remove_tool(name)
                logger.info("Excluded tool", extra_context={"tool": name})
            except (KeyError, ToolError):
                logger.warning(
                    "MCP_EXCLUDED_TOOLS references unknown tool",
                    extra_context={"tool": name},
                )

    final_tool_count = len(mcp_server._tool_manager._tools)
    logger.info(
        "Standalone MCP server created",
        extra_context={
            "backend_url": backend_config.base_url,
            "profile": active_profile or "full",
            "tool_count": final_tool_count,
            "excluded": excluded_raw or "(none)",
        },
    )

    return mcp_server


def run_standalone_mcp_server_sync():
    """Run the standalone MCP server with HTTP transport (mainly for debugging)."""
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8003"))

    logger.info("🚀 Starting standalone MCP server", extra_context={"host": host, "port": port})

    mcp_server = create_standalone_mcp_server()

    try:
        import uvicorn

        http_app = build_streamable_http_app(mcp_server)
        uvicorn.run(http_app, host=host, port=port)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as exc:
        logger.error(f"Server error: {exc}")
        raise
    finally:
        # Best-effort close of the shared httpx client and clear the global.
        http_client = getattr(mcp_server, "_http_client", None)
        if http_client is not None:
            try:
                asyncio.run(http_client.aclose())
            except Exception:
                pass  # Process is exiting anyway
            from neem.mcp.http_client import clear_http_client
            clear_http_client()
        logger.info("✅ Standalone MCP server shutdown complete")


if __name__ == "__main__":
    run_standalone_mcp_server_sync()
