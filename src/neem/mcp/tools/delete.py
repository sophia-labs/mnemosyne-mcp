"""Unified delete tool — dispatches to type-specific delete handlers.

Consolidates delete_folder, delete_documents, delete_blocks, delete_wires,
and delete_graph into a single `delete` tool with a `type` discriminator.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import Context, FastMCP

from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("mcp.tools.delete")


def register_delete_tool(server: FastMCP) -> None:
    """Register the unified delete tool.

    Must be called AFTER all other tool registration functions, since it
    reads _delete_handlers stored by register_hocuspocus_tools,
    register_wire_tools, and register_graph_ops_tools.
    """
    handlers: dict = getattr(server, "_delete_handlers", {})
    if not handlers:
        logger.warning("No delete handlers found — unified delete tool not registered")
        return

    @server.tool(
        name="delete",
        title="Delete",
        description=(
            "Delete resources from the knowledge graph. Pass `type` to specify what to delete.\n\n"
            "**Types:**\n"
            "- `documents`: Delete document(s). Pass `document_id` (single) or `document_ids` (batch). "
            "`hard=true` (default) permanently deletes; `hard=false` removes from workspace only.\n"
            "- `blocks`: Delete block(s) within a document. Pass `document_id` + `block_id` or `block_ids`. "
            "`cascade=true` deletes indent-children too. Read the document first.\n"
            "- `folder`: Delete a folder. Pass `folder_id`. `cascade=true` deletes all contents. "
            "`hard=true` (default) permanently deletes.\n"
            "- `wires`: Delete semantic wires. By ID: `wire_id` or `wire_ids`. "
            "By document: `document_id`. By block: `document_id` + `block_id`.\n"
            "- `graph`: Delete an entire graph. `hard=true` permanently deletes (cannot be undone)."
        ),
    )
    async def delete_tool(
        type: str = "",
        graph_id: str | None = None,
        # Documents / blocks / wires / folders
        document_id: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        # Blocks
        block_id: Optional[str] = None,
        block_ids: Optional[list[str]] = None,
        # Folders
        folder_id: Optional[str] = None,
        # Wires
        wire_id: Optional[str] = None,
        wire_ids: Optional[list[str]] = None,
        # Shared flags
        cascade: bool = False,
        hard: bool = True,
        context: Context | None = None,
    ) -> str | dict:
        """Unified delete dispatcher."""
        if not type or not type.strip():
            raise ValueError(
                "type is required. One of: documents, blocks, folder, wires, graph"
            )
        type = type.strip().lower()

        handler = handlers.get(type)
        if handler is None:
            raise ValueError(
                f"Unknown delete type: '{type}'. "
                f"Must be one of: {', '.join(sorted(handlers.keys()))}"
            )

        # Build kwargs for the specific handler
        kwargs: dict = {"graph_id": graph_id, "context": context}

        if type == "documents":
            kwargs["document_id"] = document_id
            kwargs["document_ids"] = document_ids
            kwargs["hard"] = hard
        elif type == "blocks":
            kwargs["document_id"] = document_id or ""
            kwargs["block_id"] = block_id
            kwargs["block_ids"] = block_ids
            kwargs["cascade"] = cascade
        elif type == "folder":
            kwargs["folder_id"] = folder_id or ""
            kwargs["cascade"] = cascade
            kwargs["hard"] = hard
        elif type == "wires":
            kwargs["wire_id"] = wire_id
            kwargs["wire_ids"] = wire_ids
            kwargs["document_id"] = document_id
            kwargs["block_id"] = block_id
        elif type == "graph":
            kwargs["graph_id"] = graph_id
            kwargs["hard"] = hard

        return await handler(**kwargs)

    logger.info(
        "Registered unified delete tool",
        extra_context={"types": sorted(handlers.keys())},
    )
