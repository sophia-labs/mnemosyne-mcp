"""Surface tool — lets Sophia show users what she did as clickable links in chat."""

from __future__ import annotations

import json
from typing import Any, List, Optional

from mcp.server.fastmcp import Context, FastMCP

from neem.hocuspocus import HocuspocusClient, WorkspaceReader
from neem.mcp.auth import MCPAuthContext
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_dev_user_id, get_internal_service_secret, validate_token_and_load

logger = LoggerFactory.get_logger("mcp.tools.surface")


def register_surface_tools(server: FastMCP) -> None:
    """Register the surface tool for showing completed actions in chat."""

    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping surface tool registration")
        return

    hp_client: Optional[HocuspocusClient] = getattr(server, "_hocuspocus_client", None)
    if hp_client is None:
        hp_client = HocuspocusClient(
            base_url=backend_config.base_url,
            token_provider=validate_token_and_load,
            dev_user_id=get_dev_user_id(),
            internal_service_secret=get_internal_service_secret(),
        )
        server._hocuspocus_client = hp_client  # type: ignore[attr-defined]

    @server.tool(
        name="surface",
        title="Surface Actions",
        description=(
            "Show the user what you did as clickable links in chat. "
            "Call this after completing work to give the user navigable references "
            "to documents and blocks you created, edited, or connected.\n\n"
            "Each action needs a document_id and a short action description. "
            "Optionally include a block_id for block-level navigation. "
            "Document titles are resolved automatically.\n\n"
            "Example:\n"
            '  surface(graph_id="default", actions=[{"document_id": "meeting-notes", '
            '"action": "created"}, {"document_id": "project-plan", '
            '"block_id": "block-abc", "action": "added timeline section"}])'
        ),
    )
    async def surface(
        graph_id: str,
        actions: List[dict[str, Any]],
        ctx: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(ctx)
        auth.require_auth()

        if not actions:
            return json.dumps({"type": "surface", "actions": []})

        # Resolve document titles from workspace Y.Doc
        titles: dict[str, str | None] = {}
        try:
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
            if channel is not None:
                reader = WorkspaceReader(channel.doc)
                for action in actions:
                    doc_id = action.get("document_id", "")
                    if doc_id and doc_id not in titles:
                        doc_entry = reader.get_document(doc_id)
                        titles[doc_id] = doc_entry.get("title") if doc_entry else None
        except Exception as exc:
            logger.warning(
                "surface_title_resolution_failed",
                extra_context={"graph_id": graph_id, "error": str(exc)},
            )

        # Build resolved action list
        resolved: list[dict[str, Any]] = []
        for action in actions:
            doc_id = action.get("document_id", "")
            entry: dict[str, Any] = {
                "document_id": doc_id,
                "title": titles.get(doc_id) or doc_id,
                "action": action.get("action", ""),
            }
            block_id = action.get("block_id")
            if block_id:
                entry["block_id"] = block_id
            resolved.append(entry)

        result = {"type": "surface", "actions": resolved}

        logger.info(
            "surface_actions",
            extra_context={
                "graph_id": graph_id,
                "action_count": len(resolved),
                "user_id": auth.user_id,
            },
        )

        return json.dumps(result)
