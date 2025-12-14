"""
MCP tools that use Hocuspocus/Y.js for real-time document access.

These tools provide direct read/write access to Mnemosyne documents via Y.js
CRDT synchronization, bypassing the job queue for lower latency operations.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

from mcp.server.fastmcp import Context, FastMCP

from neem.hocuspocus import HocuspocusClient, DocumentReader, DocumentWriter, WorkspaceWriter, WorkspaceReader
from neem.hocuspocus.document import extract_title_from_xml
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_dev_user_id, get_user_id_from_token, validate_token_and_load

logger = LoggerFactory.get_logger("mcp.tools.hocuspocus")

JsonDict = Dict[str, Any]


def register_hocuspocus_tools(server: FastMCP) -> None:
    """Register document tools that use the Hocuspocus WebSocket client."""

    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping hocuspocus tool registration")
        return

    # Get or create the HocuspocusClient
    hp_client: Optional[HocuspocusClient] = getattr(server, "_hocuspocus_client", None)
    if hp_client is None:
        hp_client = HocuspocusClient(
            base_url=backend_config.base_url,
            token_provider=validate_token_and_load,
            dev_user_id=get_dev_user_id(),
        )
        server._hocuspocus_client = hp_client  # type: ignore[attr-defined]
        logger.info(
            "Created HocuspocusClient for real-time document access",
            extra_context={"base_url": backend_config.base_url},
        )

    @server.tool(
        name="get_active_context",
        title="Get Active Graph and Document",
        description=(
            "Returns the currently active graph ID and document ID from the user's session. "
            "Use this to understand what the user is currently working on in the Mnemosyne UI. "
            "The user_id is automatically derived from authentication if not provided."
        ),
    )
    async def get_active_context_tool(
        user_id: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        """Get the active graph and document from session state."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        # Auto-derive user_id if not provided
        if not user_id:
            user_id = get_user_id_from_token(token)
            if not user_id:
                raise RuntimeError(
                    "Could not determine user ID. Either provide it explicitly or "
                    "ensure your token contains a 'sub' claim."
                )

        try:
            await hp_client.ensure_session_connected(user_id)

            active_graph = hp_client.get_active_graph_id()
            active_doc = hp_client.get_active_document_id()
            session_snapshot = hp_client.get_session_snapshot()

            result = {
                "active_graph_id": active_graph,
                "active_document_id": active_doc,
                "session": session_snapshot,
            }
            return _render_json(result)

        except Exception as e:
            logger.error(
                "Failed to get active context",
                extra_context={"error": str(e)},
            )
            raise RuntimeError(f"Failed to get active context: {e}")

    @server.tool(
        name="read_document",
        title="Read Document Content",
        description="""Reads document content as TipTap XML with full formatting.

Blocks: paragraph, heading (level="1-3"), bulletList, orderedList, blockquote, codeBlock (language="..."), taskList (taskItem checked="true"), horizontalRule
Marks (nestable): strong, em, strike, code, mark (highlight), a (href="..."), footnote (data-footnote-content="..."), commentMark (data-comment-id="...")
Lists: <bulletList><listItem><paragraph>item</paragraph></listItem></bulletList>""",
    )
    async def read_document_tool(
        graph_id: str,
        document_id: str,
        context: Context | None = None,
    ) -> str:
        """Read document content as TipTap XML."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        try:
            # Connect to the document channel
            await hp_client.connect_document(graph_id, document_id)

            # Get the channel and read content
            channel = hp_client.get_document_channel(graph_id, document_id)
            if channel is None:
                raise RuntimeError(f"Document channel not found: {graph_id}/{document_id}")

            reader = DocumentReader(channel.doc)
            xml_content = reader.to_xml()
            comments = reader.get_all_comments()

            result = {
                "graph_id": graph_id,
                "document_id": document_id,
                "content": xml_content,
                "comments": comments,
            }
            return _render_json(result)

        except Exception as e:
            logger.error(
                "Failed to read document",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to read document: {e}")

    @server.tool(
        name="write_document",
        title="Write Document Content",
        description="""Replaces document content with TipTap XML. Syncs to UI in real-time.

WARNING: This REPLACES all content. For collaborative editing, prefer append_to_document.

Blocks: paragraph, heading (level="1-3"), bulletList, orderedList, blockquote, codeBlock (language="..."), taskList (taskItem checked="true"), horizontalRule
Marks (nestable): strong, em, strike, code, mark (highlight), a (href="..."), footnote (data-footnote-content="..."), commentMark (data-comment-id="...")
Example: <paragraph>Text with <mark>highlight</mark> and a note<footnote data-footnote-content="This is a footnote"/></paragraph>

Comments: Pass a dict mapping comment IDs to metadata. Comment IDs must match data-comment-id attributes in the content.
Example comments: {"comment-1": {"text": "Great point!", "author": "Claude"}}""",
    )
    async def write_document_tool(
        graph_id: str,
        document_id: str,
        content: str,
        comments: Optional[Dict[str, Any]] = None,
        context: Context | None = None,
    ) -> str:
        """Write TipTap XML content to a document."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        try:
            # 1. Write document content and comments
            await hp_client.connect_document(graph_id, document_id)

            def write_content_and_comments(doc: Any) -> None:
                writer = DocumentWriter(doc)
                writer.replace_all_content(content)
                # Write comments if provided
                if comments:
                    for comment_id, comment_data in comments.items():
                        writer.set_comment(
                            comment_id=comment_id,
                            text=comment_data.get("text", ""),
                            author=comment_data.get("author", "MCP Agent"),
                            author_id=comment_data.get("authorId", "mcp-agent"),
                            resolved=comment_data.get("resolved", False),
                            quoted_text=comment_data.get("quotedText"),
                        )

            await hp_client.transact_document(
                graph_id,
                document_id,
                write_content_and_comments,
            )

            # 2. Update workspace navigation so document appears in file tree
            # Extract title from first heading, fallback to document_id
            title = extract_title_from_xml(content) or document_id
            await hp_client.connect_workspace(graph_id)
            await hp_client.transact_workspace(
                graph_id,
                lambda doc: WorkspaceWriter(doc).upsert_document(document_id, title),
            )

            # 3. Read back document content and comments to confirm
            channel = hp_client.get_document_channel(graph_id, document_id)
            if channel is None:
                raise RuntimeError(f"Document channel not found: {graph_id}/{document_id}")

            reader = DocumentReader(channel.doc)
            xml_content = reader.to_xml()
            result_comments = reader.get_all_comments()

            result = {
                "success": True,
                "graph_id": graph_id,
                "document_id": document_id,
                "title": title,
                "content": xml_content,
                "comments": result_comments,
            }
            return _render_json(result)

        except Exception as e:
            logger.error(
                "Failed to write document",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to write document: {e}")

    @server.tool(
        name="append_to_document",
        title="Append Paragraph to Document",
        description=(
            "Appends a new paragraph to the end of a document. "
            "Use this for incremental additions without replacing existing content."
        ),
    )
    async def append_to_document_tool(
        graph_id: str,
        document_id: str,
        text: str,
        context: Context | None = None,
    ) -> str:
        """Append a paragraph to a document."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        try:
            # Connect to the document channel
            await hp_client.connect_document(graph_id, document_id)

            # Escape XML special characters in the text
            import html
            escaped_text = html.escape(text)

            # Use transact_document for proper incremental update handling
            await hp_client.transact_document(
                graph_id,
                document_id,
                lambda doc: DocumentWriter(doc).append_block(
                    f"<paragraph>{escaped_text}</paragraph>"
                ),
            )

            result = {
                "success": True,
                "graph_id": graph_id,
                "document_id": document_id,
            }
            return _render_json(result)

        except Exception as e:
            logger.error(
                "Failed to append to document",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to append to document: {e}")

    @server.tool(
        name="get_workspace",
        title="Get Workspace Structure",
        description=(
            "Returns the folder and file structure of a graph's workspace. "
            "Use this to understand the organization of documents in a graph."
        ),
    )
    async def get_workspace_tool(
        graph_id: str,
        context: Context | None = None,
    ) -> str:
        """Get workspace folder structure."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        try:
            # Connect to the workspace channel
            await hp_client.connect_workspace(graph_id)

            # Get workspace snapshot
            snapshot = hp_client.get_workspace_snapshot(graph_id)

            result = {
                "graph_id": graph_id,
                "workspace": snapshot,
            }
            return _render_json(result)

        except Exception as e:
            logger.error(
                "Failed to get workspace",
                extra_context={
                    "graph_id": graph_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to get workspace: {e}")

    # -------------------------------------------------------------------------
    # Folder Operations (Y.js-based, replacing HTTP job-based navigation.py)
    # -------------------------------------------------------------------------

    @server.tool(
        name="create_folder",
        title="Create Folder",
        description=(
            "Create a new folder in the workspace. "
            "Use parent_id to nest inside another folder (null for root level). "
            "The section parameter determines which sidebar section the folder appears in."
        ),
    )
    async def create_folder_tool(
        graph_id: str,
        folder_id: str,
        label: str,
        parent_id: Optional[str] = None,
        order: Optional[float] = None,
        section: str = "documents",
        context: Context | None = None,
    ) -> str:
        """Create a new folder in the workspace via Y.js."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")
        if not label or not label.strip():
            raise ValueError("label is required and cannot be empty")
        if section not in ("documents", "artifacts"):
            raise ValueError("section must be 'documents' or 'artifacts'")

        try:
            await hp_client.connect_workspace(graph_id.strip())

            # Create folder via Y.js transact
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).upsert_folder(
                    folder_id.strip(),
                    label.strip(),  # 'label' param → 'name' in Y.js
                    parent_id=parent_id.strip() if parent_id else None,
                    section=section,
                    order=order,
                ),
            )

            # Return workspace snapshot for confirmation
            snapshot = hp_client.get_workspace_snapshot(graph_id.strip())

            result = {
                "success": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "label": label.strip(),
                "parent_id": parent_id.strip() if parent_id else None,
                "section": section,
                "workspace": snapshot,
            }
            return _render_json(result)

        except Exception as e:
            logger.error(
                "Failed to create folder",
                extra_context={
                    "graph_id": graph_id,
                    "folder_id": folder_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to create folder: {e}")

    @server.tool(
        name="move_folder",
        title="Move Folder",
        description=(
            "Move a folder to a new parent folder. "
            "Set new_parent_id to null to move to root level. "
            "Optionally update the order for positioning among siblings."
        ),
    )
    async def move_folder_tool(
        graph_id: str,
        folder_id: str,
        new_parent_id: Optional[str] = None,
        new_order: Optional[float] = None,
        context: Context | None = None,
    ) -> str:
        """Move a folder to a new parent via Y.js."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip())

            # Read current folder state from Y.js
            channel = hp_client._workspace_channels.get(graph_id.strip())
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)
            current = reader.get_folder(folder_id.strip())

            if not current:
                raise RuntimeError(f"Folder '{folder_id}' not found in graph '{graph_id}'")

            # Update folder with new parent/order via Y.js
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).update_folder(
                    folder_id.strip(),
                    parent_id=new_parent_id.strip() if new_parent_id else None,
                    order=new_order,
                ),
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip())

            result = {
                "success": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "new_parent_id": new_parent_id.strip() if new_parent_id else None,
                "workspace": snapshot,
            }
            return _render_json(result)

        except Exception as e:
            logger.error(
                "Failed to move folder",
                extra_context={
                    "graph_id": graph_id,
                    "folder_id": folder_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to move folder: {e}")

    @server.tool(
        name="rename_folder",
        title="Rename Folder",
        description="Rename a folder's display label.",
    )
    async def rename_folder_tool(
        graph_id: str,
        folder_id: str,
        new_label: str,
        context: Context | None = None,
    ) -> str:
        """Rename a folder via Y.js."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")
        if not new_label or not new_label.strip():
            raise ValueError("new_label is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip())

            # Verify folder exists
            channel = hp_client._workspace_channels.get(graph_id.strip())
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)
            current = reader.get_folder(folder_id.strip())

            if not current:
                raise RuntimeError(f"Folder '{folder_id}' not found in graph '{graph_id}'")

            # Update folder name via Y.js
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).update_folder(
                    folder_id.strip(),
                    name=new_label.strip(),
                ),
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip())

            result = {
                "success": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "new_label": new_label.strip(),
                "workspace": snapshot,
            }
            return _render_json(result)

        except Exception as e:
            logger.error(
                "Failed to rename folder",
                extra_context={
                    "graph_id": graph_id,
                    "folder_id": folder_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to rename folder: {e}")

    @server.tool(
        name="delete_folder",
        title="Delete Folder",
        description=(
            "Delete a folder from the workspace. "
            "Set cascade=true to delete all contents (subfolders, documents, artifacts). "
            "Without cascade, deletion fails if the folder has children."
        ),
    )
    async def delete_folder_tool(
        graph_id: str,
        folder_id: str,
        cascade: bool = False,
        context: Context | None = None,
    ) -> str:
        """Delete a folder via Y.js."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip())

            # Delete folder via Y.js
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).delete_folder(folder_id.strip(), cascade=cascade),
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip())

            result = {
                "success": True,
                "deleted": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "cascade": cascade,
                "workspace": snapshot,
            }
            return _render_json(result)

        except ValueError as ve:
            # Cascade error - folder has children
            raise RuntimeError(str(ve))
        except Exception as e:
            logger.error(
                "Failed to delete folder",
                extra_context={
                    "graph_id": graph_id,
                    "folder_id": folder_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to delete folder: {e}")

    # -------------------------------------------------------------------------
    # Artifact Operations
    # -------------------------------------------------------------------------

    @server.tool(
        name="move_artifact",
        title="Move Artifact",
        description=(
            "Move an artifact to a different folder. "
            "Set new_parent_id to null to move to root level."
        ),
    )
    async def move_artifact_tool(
        graph_id: str,
        artifact_id: str,
        new_parent_id: Optional[str] = None,
        new_order: Optional[float] = None,
        context: Context | None = None,
    ) -> str:
        """Move an artifact to a different folder via Y.js."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not artifact_id or not artifact_id.strip():
            raise ValueError("artifact_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip())

            # Read current artifact state from Y.js
            channel = hp_client._workspace_channels.get(graph_id.strip())
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)
            current = reader.get_artifact(artifact_id.strip())

            if not current:
                raise RuntimeError(f"Artifact '{artifact_id}' not found in graph '{graph_id}'")

            # Update artifact with new parent/order via Y.js
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).update_artifact(
                    artifact_id.strip(),
                    parent_id=new_parent_id.strip() if new_parent_id else None,
                    order=new_order,
                ),
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip())

            result = {
                "success": True,
                "artifact_id": artifact_id.strip(),
                "graph_id": graph_id.strip(),
                "new_parent_id": new_parent_id.strip() if new_parent_id else None,
                "workspace": snapshot,
            }
            return _render_json(result)

        except Exception as e:
            logger.error(
                "Failed to move artifact",
                extra_context={
                    "graph_id": graph_id,
                    "artifact_id": artifact_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to move artifact: {e}")

    @server.tool(
        name="rename_artifact",
        title="Rename Artifact",
        description="Rename an artifact's display label.",
    )
    async def rename_artifact_tool(
        graph_id: str,
        artifact_id: str,
        new_label: str,
        context: Context | None = None,
    ) -> str:
        """Rename an artifact via Y.js."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not artifact_id or not artifact_id.strip():
            raise ValueError("artifact_id is required and cannot be empty")
        if not new_label or not new_label.strip():
            raise ValueError("new_label is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip())

            # Verify artifact exists
            channel = hp_client._workspace_channels.get(graph_id.strip())
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)
            current = reader.get_artifact(artifact_id.strip())

            if not current:
                raise RuntimeError(f"Artifact '{artifact_id}' not found in graph '{graph_id}'")

            # Update artifact name via Y.js
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).update_artifact(
                    artifact_id.strip(),
                    name=new_label.strip(),
                ),
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip())

            result = {
                "success": True,
                "artifact_id": artifact_id.strip(),
                "graph_id": graph_id.strip(),
                "new_label": new_label.strip(),
                "workspace": snapshot,
            }
            return _render_json(result)

        except Exception as e:
            logger.error(
                "Failed to rename artifact",
                extra_context={
                    "graph_id": graph_id,
                    "artifact_id": artifact_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to rename artifact: {e}")

    # -------------------------------------------------------------------------
    # Document Navigation Operations
    # -------------------------------------------------------------------------

    @server.tool(
        name="move_document",
        title="Move Document",
        description=(
            "Move a document to a folder. "
            "Set new_parent_id to null to move to root level (unfiled). "
            "Note: This updates the document's folder assignment in workspace navigation."
        ),
    )
    async def move_document_tool(
        graph_id: str,
        document_id: str,
        new_parent_id: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        """Move a document to a folder via Y.js."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip())

            # Verify document exists in workspace
            channel = hp_client._workspace_channels.get(graph_id.strip())
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)
            current = reader.get_document(document_id.strip())

            if not current:
                raise RuntimeError(f"Document '{document_id}' not found in workspace '{graph_id}'")

            # Update document parent via Y.js
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).update_document(
                    document_id.strip(),
                    parent_id=new_parent_id.strip() if new_parent_id else None,
                ),
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip())

            result = {
                "success": True,
                "document_id": document_id.strip(),
                "graph_id": graph_id.strip(),
                "new_parent_id": new_parent_id.strip() if new_parent_id else None,
                "workspace": snapshot,
            }
            return _render_json(result)

        except Exception as e:
            logger.error(
                "Failed to move document",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to move document: {e}")

    logger.info("Registered hocuspocus tools (documents and navigation)")


def _render_json(payload: JsonDict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
