"""MCP tools for document edit history (git-style snapshot access)."""

from __future__ import annotations

import httpx
from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context, FastMCP

from neem.mcp.auth import MCPAuthContext
from neem.mcp.http_client import get_http_client
from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("mcp.tools.history")

_HTTP_TIMEOUT = httpx.Timeout(15.0)

_TIER_LABELS: Dict[str, str] = {
    "20min": "20 min",
    "2h": "2 hr",
    "12h": "12 hr",
    "daily": "daily",
    "weekly": "weekly",
}


def register_history_tools(server: FastMCP) -> None:
    """Register document history MCP tools."""

    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping history tool registration")
        return

    # ==================================================================
    # get_document_history — git log equivalent
    # ==================================================================

    @server.tool(
        name="get_document_history",
        title="Get Document History",
        description=(
            "Returns the edit history of a document as a flat list of snapshots, "
            "newest first. Like `git log`.\n\n"
            "Each snapshot entry includes:\n"
            "- **snapshot_id**: identifier for this snapshot\n"
            "- **created_at**: ISO 8601 timestamp\n"
            "- **tier**: resolution window this snapshot represents "
            "(20min / 2hr / 12hr / daily / weekly)\n"
            "- **snapshot_count**: how many raw 20-min snapshots were absorbed "
            "into this entry (>1 means it represents a compressed window)\n"
            "- **chars_added / chars_removed**: edit volume since the prior snapshot\n"
            "- **blocks_added / blocks_removed / blocks_modified**: structural changes\n"
            "- **is_manual**: true if this is a named manual save\n"
            "- **label**: user-provided name for manual saves\n\n"
            "Use this to assess metabolic rate — how actively is this document "
            "being revised? Look at chars_added+chars_removed frequency and magnitude "
            "to detect burst patterns, settling, or anabolic/catabolic ratio.\n\n"
            "Use `read_document_at_snapshot` to read the document content at a "
            "specific snapshot_id."
        ),
    )
    async def get_document_history_tool(
        graph_id: str,
        document_id: str,
        limit: int = 20,
        context: Context | None = None,
    ) -> dict:
        """Return the edit history of a document.

        Args:
            graph_id: The graph containing the document
            document_id: The document to inspect
            limit: Maximum number of snapshots to return (default 20, max 200)
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        limit = max(1, min(200, limit))
        url = (
            f"{backend_config.base_url}/v1/documents/{graph_id}/{document_id}"
            f"/snapshots?limit={limit}"
        )
        try:
            resp = await get_http_client().get(
                url,
                headers=auth.http_headers(),
                timeout=_HTTP_TIMEOUT,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch document history: {exc}") from exc

        if resp.status_code == 404:
            raise RuntimeError(
                f"Document '{document_id}' not found in graph '{graph_id}'."
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"History request failed ({resp.status_code}): {resp.text[:300]}"
            )

        data = resp.json()
        snapshots = data.get("snapshots", [])

        formatted: list[Dict[str, Any]] = []
        for snap in snapshots:
            entry: Dict[str, Any] = {
                "snapshot_id": snap["snapshot_id"],
                "created_at": snap["created_at"],
                "tier": snap.get("tier", "20min"),
                "tier_label": _TIER_LABELS.get(snap.get("tier", "20min"), snap.get("tier", "")),
                "snapshot_count": snap.get("snapshot_count", 1),
                "chars_added": snap.get("chars_added", 0),
                "chars_removed": snap.get("chars_removed", 0),
                "blocks_added": snap.get("blocks_added", 0),
                "blocks_removed": snap.get("blocks_removed", 0),
                "blocks_modified": snap.get("blocks_modified", 0),
                "is_manual": snap.get("is_manual", False),
            }
            if snap.get("label"):
                entry["label"] = snap["label"]
            formatted.append(entry)

        return {
            "graph_id": graph_id,
            "document_id": document_id,
            "snapshot_count": len(formatted),
            "snapshots": formatted,
        }

    # ==================================================================
    # read_document_at_snapshot — git show equivalent
    # ==================================================================

    @server.tool(
        name="read_document_at_snapshot",
        title="Read Document at Snapshot",
        description=(
            "Returns the document content at a historical snapshot as plain text / "
            "markdown. Like `git show`.\n\n"
            "Content is rendered from the stored block list: headings become `#` "
            "markers, bullet/ordered lists are rendered idiomatically, code blocks "
            "use fences, blockquotes use `>`. Inline formatting (bold, links, "
            "comments) is stripped to plain text.\n\n"
            "Use `get_document_history` first to get a list of snapshot_ids, then "
            "call this to read the document at a specific point in time."
        ),
    )
    async def read_document_at_snapshot_tool(
        graph_id: str,
        document_id: str,
        snapshot_id: str,
        context: Context | None = None,
    ) -> dict:
        """Return document content at a historical snapshot.

        Args:
            graph_id: The graph containing the document
            document_id: The document ID
            snapshot_id: The snapshot to read (from get_document_history)
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        url = (
            f"{backend_config.base_url}/v1/documents/{graph_id}/{document_id}"
            f"/snapshots/{snapshot_id}/text"
        )
        try:
            resp = await get_http_client().get(
                url,
                headers=auth.http_headers(),
                timeout=_HTTP_TIMEOUT,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch snapshot content: {exc}") from exc

        if resp.status_code == 404:
            raise RuntimeError(
                f"Snapshot '{snapshot_id}' not found for document '{document_id}'."
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Snapshot content request failed ({resp.status_code}): {resp.text[:300]}"
            )

        return {
            "graph_id": graph_id,
            "document_id": document_id,
            "snapshot_id": snapshot_id,
            "content": resp.text,
        }
