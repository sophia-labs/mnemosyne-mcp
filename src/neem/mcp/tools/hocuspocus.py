"""
MCP tools that use Hocuspocus/Y.js for real-time document access.

These tools provide direct read/write access to Mnemosyne documents via Y.js
CRDT synchronization, bypassing the job queue for lower latency operations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context, FastMCP

from neem.hocuspocus import HocuspocusClient, DocumentReader, DocumentWriter, WorkspaceWriter, WorkspaceReader
from neem.hocuspocus.document import extract_title_from_xml
from neem.mcp.auth import MCPAuthContext
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_dev_user_id, get_internal_service_secret, get_user_id_from_token, validate_token_and_load

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
            internal_service_secret=get_internal_service_secret(),
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
    ) -> dict:
        """Get the active graph and document from session state."""
        auth = MCPAuthContext.from_context(context)
        token = auth.require_auth()

        # Auto-derive user_id if not provided
        if not user_id:
            # Try auth context first, then token
            user_id = auth.user_id or (get_user_id_from_token(token) if token else None)
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
            return result

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
    ) -> dict:
        """Read document content as TipTap XML."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        try:
            # Connect to the document channel with user context
            await hp_client.connect_document(graph_id, document_id, user_id=auth.user_id)

            # Get the channel and read content
            channel = hp_client.get_document_channel(graph_id, document_id, user_id=auth.user_id)
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
            return result

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
    ) -> dict:
        """Write TipTap XML content to a document."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        try:
            # 1. Write document content and comments (with user context)
            await hp_client.connect_document(graph_id, document_id, user_id=auth.user_id)

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
                user_id=auth.user_id,
            )

            # 2. Update workspace navigation so document appears in file tree
            # Extract title from first heading, fallback to document_id
            title = extract_title_from_xml(content) or document_id
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            await hp_client.transact_workspace(
                graph_id,
                lambda doc: WorkspaceWriter(doc).upsert_document(document_id, title),
                user_id=auth.user_id,
            )

            # 3. Read back document content and comments to confirm
            channel = hp_client.get_document_channel(graph_id, document_id, user_id=auth.user_id)
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
            return result

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
        title="Append Block to Document",
        description=(
            "Appends a block to the end of a document. Accepts TipTap XML for any block type. "
            "Use this for incremental additions without replacing existing content. "
            "For plain text, wrap in <paragraph>text</paragraph>. For structured content, "
            "provide full XML like <heading level=\"2\">Title</heading> or <listItem listType=\"bullet\">...</listItem>."
        ),
    )
    async def append_to_document_tool(
        graph_id: str,
        document_id: str,
        text: str,
        context: Context | None = None,
    ) -> dict:
        """Append a block to a document.

        Args:
            graph_id: The graph containing the document
            document_id: The document to append to
            text: TipTap XML content. If it doesn't start with '<', it's wrapped in <paragraph>.
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not text:
            raise ValueError("text is required")

        try:
            # Connect to the document channel with user context
            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            # Determine if text is XML or plain text
            content = text.strip()
            if not content.startswith("<"):
                # Plain text - escape and wrap in paragraph
                import html
                escaped_text = html.escape(content)
                content = f"<paragraph>{escaped_text}</paragraph>"

            new_block_id: str = ""

            def perform_append(doc: Any) -> None:
                nonlocal new_block_id
                writer = DocumentWriter(doc)
                writer.append_block(content)
                # Get the last block's ID
                reader = DocumentReader(doc)
                count = reader.get_block_count()
                if count > 0:
                    block = reader.get_block_at(count - 1)
                    if block and hasattr(block, "attributes"):
                        # pycrdt XmlAttributesView.get() doesn't support default arg
                        attrs = block.attributes
                        new_block_id = attrs.get("data-block-id") if "data-block-id" in attrs else ""

            await hp_client.transact_document(
                graph_id.strip(),
                document_id.strip(),
                perform_append,
                user_id=auth.user_id,
            )

            result = {
                "success": True,
                "graph_id": graph_id.strip(),
                "document_id": document_id.strip(),
                "new_block_id": new_block_id,
            }
            return result

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
    ) -> dict:
        """Get workspace folder structure."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        try:
            # Connect to the workspace channel with user context
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)

            # Get workspace snapshot
            snapshot = hp_client.get_workspace_snapshot(graph_id, user_id=auth.user_id)

            result = {
                "graph_id": graph_id,
                "workspace": snapshot,
            }
            return result

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
    ) -> dict:
        """Create a new folder in the workspace via Y.js."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")
        if not label or not label.strip():
            raise ValueError("label is required and cannot be empty")
        if section not in ("documents", "artifacts"):
            raise ValueError("section must be 'documents' or 'artifacts'")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

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
                user_id=auth.user_id,
            )

            # Return workspace snapshot for confirmation
            snapshot = hp_client.get_workspace_snapshot(graph_id.strip(), user_id=auth.user_id)

            result = {
                "success": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "label": label.strip(),
                "parent_id": parent_id.strip() if parent_id else None,
                "section": section,
                "workspace": snapshot,
            }
            return result

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
    ) -> dict:
        """Move a folder to a new parent via Y.js."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Read current folder state from Y.js
            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
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
                user_id=auth.user_id,
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip(), user_id=auth.user_id)

            result = {
                "success": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "new_parent_id": new_parent_id.strip() if new_parent_id else None,
                "workspace": snapshot,
            }
            return result

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
    ) -> dict:
        """Rename a folder via Y.js."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")
        if not new_label or not new_label.strip():
            raise ValueError("new_label is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Verify folder exists
            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
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
                user_id=auth.user_id,
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip(), user_id=auth.user_id)

            result = {
                "success": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "new_label": new_label.strip(),
                "workspace": snapshot,
            }
            return result

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
    ) -> dict:
        """Delete a folder via Y.js."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Delete folder via Y.js
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).delete_folder(folder_id.strip(), cascade=cascade),
                user_id=auth.user_id,
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip(), user_id=auth.user_id)

            result = {
                "success": True,
                "deleted": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "cascade": cascade,
                "workspace": snapshot,
            }
            return result

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
    ) -> dict:
        """Move an artifact to a different folder via Y.js."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not artifact_id or not artifact_id.strip():
            raise ValueError("artifact_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Read current artifact state from Y.js
            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
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
                user_id=auth.user_id,
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip(), user_id=auth.user_id)

            result = {
                "success": True,
                "artifact_id": artifact_id.strip(),
                "graph_id": graph_id.strip(),
                "new_parent_id": new_parent_id.strip() if new_parent_id else None,
                "workspace": snapshot,
            }
            return result

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
    ) -> dict:
        """Rename an artifact via Y.js."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not artifact_id or not artifact_id.strip():
            raise ValueError("artifact_id is required and cannot be empty")
        if not new_label or not new_label.strip():
            raise ValueError("new_label is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Verify artifact exists
            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
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
                user_id=auth.user_id,
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip(), user_id=auth.user_id)

            result = {
                "success": True,
                "artifact_id": artifact_id.strip(),
                "graph_id": graph_id.strip(),
                "new_label": new_label.strip(),
                "workspace": snapshot,
            }
            return result

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
    ) -> dict:
        """Move a document to a folder via Y.js."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Verify document exists in workspace
            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
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
                user_id=auth.user_id,
            )

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip(), user_id=auth.user_id)

            result = {
                "success": True,
                "document_id": document_id.strip(),
                "graph_id": graph_id.strip(),
                "new_parent_id": new_parent_id.strip() if new_parent_id else None,
                "workspace": snapshot,
            }
            return result

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

    @server.tool(
        name="delete_document",
        title="Delete Document",
        description=(
            "Delete a document from workspace navigation. "
            "This removes the document from the file tree but does not destroy the underlying data. "
            "The document can be recreated by writing to the same document_id."
        ),
    )
    async def delete_document_tool(
        graph_id: str,
        document_id: str,
        context: Context | None = None,
    ) -> dict:
        """Delete a document from workspace navigation via Y.js."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Verify document exists in workspace
            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)
            current = reader.get_document(document_id.strip())

            if not current:
                raise RuntimeError(f"Document '{document_id}' not found in workspace '{graph_id}'")

            # Delete document from workspace via Y.js
            deleted = False
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).delete_document(document_id.strip()),
                user_id=auth.user_id,
            )
            deleted = True

            snapshot = hp_client.get_workspace_snapshot(graph_id.strip(), user_id=auth.user_id)

            result = {
                "success": True,
                "deleted": deleted,
                "document_id": document_id.strip(),
                "graph_id": graph_id.strip(),
                "workspace": snapshot,
            }
            return result

        except Exception as e:
            logger.error(
                "Failed to delete document",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to delete document: {e}")

    # -------------------------------------------------------------------------
    # Block-Level Document Operations
    # -------------------------------------------------------------------------

    @server.tool(
        name="get_block",
        title="Get Block by ID",
        description=(
            "Read a specific block by its data-block-id. Returns detailed info including "
            "the block's XML content, attributes, text content, and context (prev/next block IDs). "
            "Use this for targeted reads without fetching the entire document."
        ),
    )
    async def get_block_tool(
        graph_id: str,
        document_id: str,
        block_id: str,
        context: Context | None = None,
    ) -> dict:
        """Get detailed information about a block by its ID."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not block_id or not block_id.strip():
            raise ValueError("block_id is required")

        try:
            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            channel = hp_client.get_document_channel(graph_id.strip(), document_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Document channel not found: {graph_id}/{document_id}")

            reader = DocumentReader(channel.doc)
            block_info = reader.get_block_info(block_id.strip())

            if block_info is None:
                raise RuntimeError(f"Block not found: {block_id}")

            result = {
                "graph_id": graph_id.strip(),
                "document_id": document_id.strip(),
                "block": block_info,
            }
            return result

        except Exception as e:
            logger.error(
                "Failed to get block",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "block_id": block_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to get block: {e}")

    @server.tool(
        name="query_blocks",
        title="Query Blocks",
        description=(
            "Search for blocks matching specific criteria. Filter by block type, indent level, "
            "list type, checked state, or text content. Returns a list of matching block summaries. "
            "Use this to find blocks without reading the entire document."
        ),
    )
    async def query_blocks_tool(
        graph_id: str,
        document_id: str,
        block_type: Optional[str] = None,
        indent: Optional[int] = None,
        indent_gte: Optional[int] = None,
        indent_lte: Optional[int] = None,
        list_type: Optional[str] = None,
        checked: Optional[bool] = None,
        text_contains: Optional[str] = None,
        limit: int = 50,
        context: Context | None = None,
    ) -> dict:
        """Query blocks matching specific criteria."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")

        try:
            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            channel = hp_client.get_document_channel(graph_id.strip(), document_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Document channel not found: {graph_id}/{document_id}")

            reader = DocumentReader(channel.doc)
            matches = reader.query_blocks(
                block_type=block_type,
                indent=indent,
                indent_gte=indent_gte,
                indent_lte=indent_lte,
                list_type=list_type,
                checked=checked,
                text_contains=text_contains,
                limit=limit,
            )

            result = {
                "graph_id": graph_id.strip(),
                "document_id": document_id.strip(),
                "count": len(matches),
                "blocks": matches,
            }
            return result

        except Exception as e:
            logger.error(
                "Failed to query blocks",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to query blocks: {e}")

    @server.tool(
        name="update_block",
        title="Update Block",
        description=(
            "Update a block by its ID. Can update attributes (indent, checked, listType) "
            "without changing content, or replace the entire block content. "
            "This is the most surgical edit - only modifies what you specify."
        ),
    )
    async def update_block_tool(
        graph_id: str,
        document_id: str,
        block_id: str,
        attributes: Optional[Dict[str, Any]] = None,
        xml_content: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Update a block's attributes or content.

        Args:
            graph_id: The graph containing the document
            document_id: The document containing the block
            block_id: The block to update
            attributes: Dict of attributes to update (indent, checked, listType, collapsed)
            xml_content: If provided, replaces the entire block content (preserves block_id)
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not block_id or not block_id.strip():
            raise ValueError("block_id is required")
        if attributes is None and xml_content is None:
            raise ValueError("Either attributes or xml_content must be provided")

        try:
            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            def perform_update(doc: Any) -> None:
                writer = DocumentWriter(doc)
                if xml_content:
                    # Full content replacement (preserves block_id)
                    writer.replace_block_by_id(block_id.strip(), xml_content)
                if attributes:
                    # Surgical attribute update
                    writer.update_block_attributes(block_id.strip(), attributes)

            await hp_client.transact_document(
                graph_id.strip(),
                document_id.strip(),
                perform_update,
                user_id=auth.user_id,
            )

            # Read back the updated block
            channel = hp_client.get_document_channel(graph_id.strip(), document_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Document channel not found: {graph_id}/{document_id}")

            reader = DocumentReader(channel.doc)
            block_info = reader.get_block_info(block_id.strip())

            result = {
                "success": True,
                "graph_id": graph_id.strip(),
                "document_id": document_id.strip(),
                "block": block_info,
            }
            return result

        except Exception as e:
            logger.error(
                "Failed to update block",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "block_id": block_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to update block: {e}")

    @server.tool(
        name="edit_block_text",
        title="Edit Block Text",
        description=(
            "Insert or delete text at specific character offsets within a block, using "
            "CRDT-native operations that merge cleanly with concurrent browser edits. "
            "Unlike update_block (which replaces entire content), this enables true "
            "collaborative editing without data loss.\n\n"
            "Workflow: 1) Call get_block to read current text and length, "
            "2) Determine offset(s) for edits, "
            "3) Call edit_block_text with operations, "
            "4) Response includes updated text for verification.\n\n"
            "Each operation has: type ('insert' or 'delete'), offset (0-indexed char position), "
            "text (for insert), length (for delete), attrs (optional formatting like {\"bold\": {}}), "
            "inherit_format (default true - inherit formatting from preceding character).\n\n"
            "Multiple operations are applied in a single transaction. "
            "Insert beyond text length appends at end. "
            "Delete beyond text length raises an error."
        ),
    )
    async def edit_block_text_tool(
        graph_id: str,
        document_id: str,
        block_id: str,
        operations: list[Dict[str, Any]],
        context: Context | None = None,
    ) -> dict:
        """Edit text within a block at specific character offsets.

        Args:
            graph_id: The graph containing the document
            document_id: The document containing the block
            block_id: The block to edit
            operations: List of insert/delete operations with offsets
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not block_id or not block_id.strip():
            raise ValueError("block_id is required")
        if not operations:
            raise ValueError("operations list is required and cannot be empty")

        try:
            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            updated_text_info: dict = {}

            def perform_edit(doc: Any) -> None:
                nonlocal updated_text_info
                writer = DocumentWriter(doc)
                updated_text_info = writer.edit_block_text(
                    block_id.strip(), operations
                )

            await hp_client.transact_document(
                graph_id.strip(),
                document_id.strip(),
                perform_edit,
                user_id=auth.user_id,
            )

            result = {
                "success": True,
                "graph_id": graph_id.strip(),
                "document_id": document_id.strip(),
                "block": updated_text_info,
            }
            return result

        except ValueError as ve:
            # Validation errors - return as-is for clear agent feedback
            raise RuntimeError(str(ve))
        except Exception as e:
            logger.error(
                "Failed to edit block text",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "block_id": block_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to edit block text: {e}")

    @server.tool(
        name="insert_block",
        title="Insert Block",
        description=(
            "Insert a new block relative to an existing block. Use position='after' or 'before' "
            "to specify where to insert. Returns the new block's generated ID. "
            "For appending to the end, use append_to_document instead."
        ),
    )
    async def insert_block_tool(
        graph_id: str,
        document_id: str,
        reference_block_id: str,
        xml_content: str,
        position: str = "after",
        context: Context | None = None,
    ) -> dict:
        """Insert a new block before or after a reference block.

        Args:
            graph_id: The graph containing the document
            document_id: The document to insert into
            reference_block_id: The block to insert relative to
            xml_content: TipTap XML for the new block
            position: 'after' or 'before' the reference block
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not reference_block_id or not reference_block_id.strip():
            raise ValueError("reference_block_id is required")
        if not xml_content or not xml_content.strip():
            raise ValueError("xml_content is required")
        if position not in ("after", "before"):
            raise ValueError("position must be 'after' or 'before'")

        try:
            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            new_block_id: str = ""

            def perform_insert(doc: Any) -> None:
                nonlocal new_block_id
                writer = DocumentWriter(doc)
                if position == "after":
                    new_block_id = writer.insert_block_after_id(
                        reference_block_id.strip(), xml_content.strip()
                    )
                else:
                    new_block_id = writer.insert_block_before_id(
                        reference_block_id.strip(), xml_content.strip()
                    )

            await hp_client.transact_document(
                graph_id.strip(),
                document_id.strip(),
                perform_insert,
                user_id=auth.user_id,
            )

            # Read back the new block
            channel = hp_client.get_document_channel(graph_id.strip(), document_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Document channel not found: {graph_id}/{document_id}")

            reader = DocumentReader(channel.doc)
            block_info = reader.get_block_info(new_block_id) if new_block_id else None

            result = {
                "success": True,
                "graph_id": graph_id.strip(),
                "document_id": document_id.strip(),
                "new_block_id": new_block_id,
                "block": block_info,
            }
            return result

        except Exception as e:
            logger.error(
                "Failed to insert block",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "reference_block_id": reference_block_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to insert block: {e}")

    @server.tool(
        name="delete_block",
        title="Delete Block",
        description=(
            "Delete a block by its ID. Use cascade=true to also delete all subsequent blocks "
            "with higher indent (indent-based children). Returns the list of deleted block IDs."
        ),
    )
    async def delete_block_tool(
        graph_id: str,
        document_id: str,
        block_id: str,
        cascade: bool = False,
        context: Context | None = None,
    ) -> dict:
        """Delete a block and optionally its indent-children.

        Args:
            graph_id: The graph containing the document
            document_id: The document containing the block
            block_id: The block to delete
            cascade: If True, also delete indent-children
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not block_id or not block_id.strip():
            raise ValueError("block_id is required")

        try:
            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            deleted_ids: list[str] = []

            def perform_delete(doc: Any) -> None:
                nonlocal deleted_ids
                writer = DocumentWriter(doc)
                deleted_ids = writer.delete_block_by_id(
                    block_id.strip(), cascade_children=cascade
                )

            await hp_client.transact_document(
                graph_id.strip(),
                document_id.strip(),
                perform_delete,
                user_id=auth.user_id,
            )

            result = {
                "success": True,
                "graph_id": graph_id.strip(),
                "document_id": document_id.strip(),
                "deleted_block_ids": deleted_ids,
                "cascade": cascade,
            }
            return result

        except Exception as e:
            logger.error(
                "Failed to delete block",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "block_id": block_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to delete block: {e}")

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    @server.tool(
        name="batch_update_blocks",
        title="Batch Update Blocks",
        description=(
            "Update multiple blocks in a single transaction. More efficient than "
            "individual update_block calls. Each update can specify attributes to change "
            "and/or new XML content. Returns results for each update."
        ),
    )
    async def batch_update_blocks_tool(
        graph_id: str,
        document_id: str,
        updates: list[Dict[str, Any]],
        context: Context | None = None,
    ) -> dict:
        """Batch update multiple blocks atomically.

        Args:
            graph_id: The graph containing the document
            document_id: The document containing the blocks
            updates: List of update specs, each with:
                - block_id (required): The block to update
                - attributes (optional): Dict of attributes to update
                - content (optional): New XML content for the block
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not updates:
            raise ValueError("updates list is required and cannot be empty")

        try:
            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            results: list[Dict[str, Any]] = []

            def perform_batch(doc: Any) -> None:
                writer = DocumentWriter(doc)
                for update in updates:
                    block_id = update.get("block_id")
                    if not block_id:
                        results.append({"error": "missing block_id"})
                        continue

                    try:
                        if "content" in update:
                            writer.replace_block_by_id(block_id, update["content"])
                        if "attributes" in update:
                            writer.update_block_attributes(block_id, update["attributes"])
                        results.append({"block_id": block_id, "success": True})
                    except Exception as e:
                        results.append({"block_id": block_id, "error": str(e)})

            await hp_client.transact_document(
                graph_id.strip(),
                document_id.strip(),
                perform_batch,
                user_id=auth.user_id,
            )

            return {
                "success": all(r.get("success") for r in results),
                "graph_id": graph_id.strip(),
                "document_id": document_id.strip(),
                "results": results,
                "updated_count": sum(1 for r in results if r.get("success")),
                "error_count": sum(1 for r in results if "error" in r),
            }

        except Exception as e:
            logger.error(
                "Failed to batch update blocks",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "update_count": len(updates),
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to batch update blocks: {e}")

    logger.info("Registered hocuspocus tools (documents, navigation, and block operations)")
