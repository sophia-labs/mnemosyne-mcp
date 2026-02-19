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
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urlunparse

import httpx
from mcp.server.fastmcp import FastMCP

from neem.mcp.jobs.realtime import RealtimeJobClient
from neem.mcp.tools.basic import register_basic_tools
from neem.mcp.tools.graph_ops import register_graph_ops_tools
from neem.mcp.tools.hocuspocus import register_hocuspocus_tools
from neem.mcp.tools.wire_tools import register_wire_tools
from neem.mcp.tools.geist import register_geist_tools
from neem.mcp.tools.search import register_search_tools
from neem.mcp.trace import trace, trace_separator
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_dev_user_id, get_internal_service_secret, validate_token_and_load

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


def create_standalone_mcp_server() -> FastMCP:
    """
    Create an MCP server bound to the local FastAPI backend.

    Tool registration is intentionally deferred until the new architecture is
    ready; for now we only set up connectivity and metadata.
    """
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
            "(1) Orient — get_user_location then get_workspace.\n"
            "(2) Read — read_document to see content and block IDs.\n"
            "(3) Edit — use block tools for surgical changes or write_document for full replacement.\n"
            "(4) Connect — create_wire with predicates from list_wire_predicates, using block IDs for precision.\n\n"
            "**Orientation (use in this order):**\n"
            "- get_user_location: Where is the user right now? Returns just graph_id and document_id (minimal tokens)\n"
            "- get_workspace: What's in this graph? Returns folder/file structure (default depth=2, "
            "deeper folders show document counts). Use folder_id to drill into a subtree, "
            "min_score to filter by document-level valuation scores\n"
            "- get_session_state: Full UI state including tabs and preferences (large payload, rarely needed)\n\n"
            "**SPARQL Namespace Reference (IMPORTANT):**\n"
            "When writing SPARQL queries, use these exact prefixes:\n"
            "  PREFIX doc: <http://mnemosyne.dev/doc#>\n"
            "  PREFIX dcterms: <http://purl.org/dc/terms/>\n"
            "  PREFIX nfo: <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>\n"
            "  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            "  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "  PREFIX nie: <http://www.semanticdesktop.org/ontologies/2007/01/19/nie#>\n"
            "  PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
            "WARNING: Do NOT use 'urn:mnemosyne:schema:doc:' as the doc namespace - "
            "it will match nothing. The correct namespace is 'http://mnemosyne.dev/doc#'.\n\n"
            "**Common RDF Types and Predicates:**\n"
            "- doc:TipTapDocument - Document entity type\n"
            "- doc:Folder - Folder entity type\n"
            "- doc:Artifact - Uploaded file entity type\n"
            "- doc:XmlFragment - Document content root\n"
            "- doc:Paragraph, doc:Heading, doc:TextNode - Block/node types\n"
            "- dcterms:title - Document/entity title\n"
            "- nfo:fileName - Artifact/folder display name\n"
            "- nfo:belongsToContainer - Parent folder relationship\n"
            "- doc:order - Sort order (float timestamp)\n"
            "- doc:section - Sidebar section ('documents' or 'artifacts')\n"
            "- doc:content - Text content of a node\n"
            "- doc:childNode - Parent-to-child block relationship\n"
            "- doc:siblingOrder - Order among sibling blocks\n\n"
            "**Entity URI Pattern:**\n"
            "  urn:mnemosyne:user:{user_id}:graph:{graph_id}:{type}:{entity_id}\n"
            "  Content fragments use # suffixes: ...doc:{id}#frag, ...doc:{id}#block-{block_id}\n\n"
            "Documents are synced in real-time via Y.js CRDT, so changes appear "
            "immediately in the Mnemosyne web UI.\n\n"
            "**Geist (Sophia Memory Tools):**\n"
            "Memory, valuation, and self-narrative tools for agent continuity.\n"
            "- Orientation flow: get_user_location → music() → recall() → get_workspace()\n"
            "- music/sing: Read/write the Song (narrative orientation before structural orientation)\n"
            "- store_memory/recall/care: Working memory queue (FIFO, numbered, append-only)\n"
            "- valuate/get_block_values: Block-level importance (0-5) and valence (-5 to +5) scoring\n"
            "- get_values/revaluate: Read/update scoring configuration\n"
            "- recall only searches the memory queue — use get_block_values for graph-wide retrieval\n"
            "- valuate works on any block in any document, not just the queue\n"
            "- store_memory is for the agent's own working memory; append_to_document is for user-facing content\n"
            "- Wires express relationships between things; valuation expresses judgment about a single thing\n\n"
            "**Search:**\n"
            "- search_documents: Fast title/path search against workspace (no SPARQL needed). Modes: auto (default), exact, substring, regex\n"
            "- search_blocks: Cross-document content search. Modes: hybrid (default, lexical+semantic in parallel), lexical, semantic\n"
            "- reindex_graph: Re-embed all documents (admin/maintenance, auto-indexes on save)\n"
            "- recall with query param uses hybrid search: memory queue + vector similarity merged via RRF\n\n"
            "When making function calls using tools that accept array or object parameters "
            "ensure those are structured using JSON."
        ),
        stateless_http=True,
    )

    # Store the resolved config so the eventual tool layer can reuse it.
    mcp_server._backend_config = backend_config  # type: ignore[attr-defined]

    if backend_config.has_websocket:
        dev_uid = get_dev_user_id()
        internal_secret = get_internal_service_secret()
        trace("Creating RealtimeJobClient", {
            "websocket_url": backend_config.websocket_url,
            "dev_user_id": dev_uid,
            "has_internal_secret": bool(internal_secret),
            "token_provider": "validate_token_and_load",
        })
        job_client = RealtimeJobClient(
            websocket_url=backend_config.websocket_url,  # type: ignore[arg-type]
            token_provider=validate_token_and_load,
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
        trace("WebSocket DISABLED — will use HTTP polling only")
        logger.info("Realtime job streaming disabled (WebSocket URL not configured)")

    verify_backend_connectivity(backend_config)

    register_basic_tools(mcp_server)
    register_graph_ops_tools(mcp_server)
    register_hocuspocus_tools(mcp_server)
    register_wire_tools(mcp_server)
    register_geist_tools(mcp_server)
    register_search_tools(mcp_server)

    # LOCAL STOPGAP: throttle on every tool call to reduce backend pressure
    # when multiple Sophia instances hit the cluster simultaneously.
    _original_call_tool = mcp_server.call_tool

    async def _throttled_call_tool(*args, **kwargs):
        await asyncio.sleep(0.2)
        return await _original_call_tool(*args, **kwargs)

    mcp_server.call_tool = _throttled_call_tool  # type: ignore[method-assign]
    logger.info("Tool throttle enabled: 0.2s sleep before every tool call")

    # Remove excluded tools based on MCP_EXCLUDED_TOOLS env var.
    # Comma-separated list of tool names, e.g. "export_document,upload_artifact,sparql_update"
    excluded_raw = os.getenv("MCP_EXCLUDED_TOOLS", "")
    if excluded_raw.strip():
        excluded_names = [name.strip() for name in excluded_raw.split(",") if name.strip()]
        for name in excluded_names:
            try:
                mcp_server.remove_tool(name)
                logger.info("Excluded tool", extra_context={"tool": name})
            except KeyError:
                logger.warning(
                    "MCP_EXCLUDED_TOOLS references unknown tool",
                    extra_context={"tool": name},
                )

    registered_tools = mcp_server.list_tools()
    logger.info(
        "Standalone MCP server created",
        extra_context={
            "backend_url": backend_config.base_url,
            "tool_count": len(registered_tools) if hasattr(registered_tools, '__len__') else "unknown",
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

        http_app = mcp_server.streamable_http_app()
        uvicorn.run(http_app, host=host, port=port)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as exc:
        logger.error(f"Server error: {exc}")
        raise
    finally:
        logger.info("✅ Standalone MCP server shutdown complete")


if __name__ == "__main__":
    run_standalone_mcp_server_sync()
