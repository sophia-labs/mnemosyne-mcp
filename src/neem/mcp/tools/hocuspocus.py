"""
MCP tools that use Hocuspocus/Y.js for real-time document access.

These tools provide direct read/write access to Mnemosyne documents via Y.js
CRDT synchronization, bypassing the job queue for lower latency operations.
"""

from __future__ import annotations

import html as html_mod
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import Context, FastMCP

from neem.hocuspocus import HocuspocusClient, DocumentReader, DocumentWriter, WorkspaceWriter, WorkspaceReader
from neem.hocuspocus.converters import looks_like_markdown, markdown_to_tiptap_xml, tiptap_xml_to_html, tiptap_xml_to_markdown
from neem.hocuspocus.document import extract_title_from_xml
from neem.mcp.auth import MCPAuthContext
from neem.mcp.tools.wire_tools import _get_wires_for_document
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_dev_user_id, get_internal_service_secret, get_user_id_from_token, validate_token_and_load

logger = LoggerFactory.get_logger("mcp.tools.hocuspocus")

JsonDict = Dict[str, Any]


def _ensure_xml(text: str) -> str:
    """Convert text to TipTap XML if needed.

    - If text starts with '<', it's returned as-is (assumed XML).
    - If text looks like markdown, it's parsed via markdown_to_tiptap_xml().
    - Otherwise, it's wrapped in a <paragraph> element as plain text.
    """
    content = text.strip()
    if not content:
        return content
    if content.startswith("<"):
        return content
    if looks_like_markdown(content):
        return markdown_to_tiptap_xml(content)
    return f"<paragraph>{html_mod.escape(content)}</paragraph>"


def _ensure_xml_multiblock(text: str) -> str:
    """Like _ensure_xml, but for write_document where input may contain
    multiple paragraphs or full markdown documents.

    - If text starts with '<', it's returned as-is (assumed XML).
    - If text looks like markdown, it's parsed via markdown_to_tiptap_xml().
    - Otherwise, plain text is split on double-newlines into <paragraph> blocks.
    """
    content = text.strip()
    if not content:
        return content
    if content.startswith("<"):
        return content
    if looks_like_markdown(content):
        return markdown_to_tiptap_xml(content)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return "".join(
        f"<paragraph>{html_mod.escape(p)}</paragraph>" for p in paragraphs
    )


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

    # ------------------------------------------------------------------
    # Helper: resolve user_id from auth context
    # ------------------------------------------------------------------
    def _resolve_user_id(auth: MCPAuthContext, token: str, user_id: Optional[str] = None) -> str:
        if not user_id:
            user_id = auth.user_id or (get_user_id_from_token(token) if token else None)
            if not user_id:
                raise RuntimeError(
                    "Could not determine user ID. Either provide it explicitly or "
                    "ensure your token contains a 'sub' claim."
                )
        return user_id

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    async def _validate_document_in_workspace(
        graph_id: str, document_id: str, user_id: str,
    ) -> None:
        """Verify a document exists in the workspace metadata.

        Raises RuntimeError with a helpful message listing available
        documents when the requested document is not found.
        """
        await hp_client.connect_workspace(graph_id, user_id=user_id)
        ws_channel = hp_client.get_workspace_channel(graph_id, user_id=user_id)
        if ws_channel is None:
            raise RuntimeError(
                f"Could not connect to workspace for graph '{graph_id}'. "
                f"The graph may not exist or the backend may be unreachable. "
                f"Use list_graphs to see available graphs."
            )
        reader = WorkspaceReader(ws_channel.doc)
        if reader.get_document(document_id) is not None:
            return  # Document exists

        # Build a helpful error with available documents
        snapshot = hp_client.get_workspace_snapshot(graph_id, user_id=user_id)
        available: list[str] = []
        if snapshot:
            docs = snapshot.get("documents") or {}
            for doc_id, doc_info in docs.items():
                title = doc_info.get("title", doc_id) if isinstance(doc_info, dict) else doc_id
                available.append(f"  - {title} ({doc_id})")

        msg = f"Document '{document_id}' not found in graph '{graph_id}'."
        if available:
            shown = available[:15]
            msg += "\n\nAvailable documents:\n" + "\n".join(shown)
            if len(available) > 15:
                msg += f"\n  ... and {len(available) - 15} more"
        else:
            msg += (
                " The graph workspace is empty — this may mean the graph "
                "doesn't exist or you're connected to the wrong backend."
            )
        msg += "\n\nUse get_workspace to see the full graph structure."
        raise RuntimeError(msg)

    @server.tool(
        name="get_user_location",
        title="Get Current Graph and Document",
        description=(
            "Returns the graph ID and document ID the user is currently viewing. "
            "This is the lightest-weight context tool — use it when you just need to know "
            "where the user is. Follow up with get_workspace to see the full graph structure, "
            "or read_document to see the document content."
        ),
    )
    async def get_user_location_tool(
        user_id: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Get the user's current graph and document IDs."""
        auth = MCPAuthContext.from_context(context)
        token = auth.require_auth()
        user_id = _resolve_user_id(auth, token, user_id)

        try:
            await hp_client.ensure_session_connected(user_id)

            return {
                "graph_id": hp_client.get_active_graph_id(),
                "document_id": hp_client.get_active_document_id(),
            }

        except Exception as e:
            logger.error(
                "Failed to get user location",
                extra_context={"error": str(e)},
            )
            raise RuntimeError(f"Failed to get user location: {e}")

    @server.tool(
        name="get_session_state",
        title="Get Full Session State",
        description=(
            "Returns the complete session state including all tabs, preferences, and UI settings. "
            "WARNING: This returns a large payload. Prefer get_user_location (for current graph/document) "
            "or get_workspace (for graph structure) unless you specifically need full session details."
        ),
    )
    async def get_session_state_tool(
        user_id: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Get the full session state (tabs, preferences, UI settings)."""
        auth = MCPAuthContext.from_context(context)
        token = auth.require_auth()
        user_id = _resolve_user_id(auth, token, user_id)

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
                "Failed to get session state",
                extra_context={"error": str(e)},
            )
            raise RuntimeError(f"Failed to get session state: {e}")

    @server.tool(
        name="read_document",
        title="Read Document Content",
        description="""Reads document content as TipTap XML with full formatting.

Blocks: paragraph, heading (level="1-3"), bulletList, orderedList, blockquote, codeBlock (language="..."), taskList (taskItem checked="true"), horizontalRule
Marks (nestable): strong, em, strike, code, mark (highlight), a (href="..."), footnote (data-footnote-content="..."), commentMark (data-comment-id="...")
Lists: <bulletList><listItem><paragraph>item</paragraph></listItem></bulletList>

Each block element includes a data-block-id attribute. Use these IDs with block-level tools (get_block, update_block, insert_block, delete_block) and for block-level wire connections.

Also returns wire counts: document-level (outgoing, incoming, total) and block-level (which blocks have wires attached). Use get_wires for full wire details.""",
    )
    async def read_document_tool(
        graph_id: str,
        document_id: str,
        format: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Read document content as TipTap XML or markdown.

        Args:
            graph_id: The graph containing the document
            document_id: The document to read
            format: Output format - "xml" (default) or "markdown"
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if format and format not in ("xml", "markdown"):
            raise ValueError("format must be 'xml' or 'markdown'")

        try:
            # Validate document exists in workspace before connecting
            await _validate_document_in_workspace(graph_id, document_id, auth.user_id)

            # Connect to the document channel with user context
            try:
                await hp_client.connect_document(graph_id, document_id, user_id=auth.user_id)
            except TimeoutError:
                logger.warning(
                    "read_document initial sync timed out, retrying once",
                    extra_context={
                        "graph_id": graph_id,
                        "document_id": document_id,
                    },
                )
                await hp_client.disconnect_document(graph_id, document_id, user_id=auth.user_id)
                await hp_client.connect_document(graph_id, document_id, user_id=auth.user_id)

            # Get the channel and read content
            channel = hp_client.get_document_channel(graph_id, document_id, user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Document channel not found: {graph_id}/{document_id}")

            reader = DocumentReader(channel.doc)
            xml_content = reader.to_xml()
            comments = reader.get_all_comments()

            # Convert to requested format
            if format == "markdown":
                content = tiptap_xml_to_markdown(xml_content)
            elif format == "html":
                title = extract_title_from_xml(xml_content) or document_id
                content = tiptap_xml_to_html(xml_content, title=title, themed=True)
            else:
                content = xml_content

            result = {
                "graph_id": graph_id,
                "document_id": document_id,
                "format": format or "xml",
                "content": content,
                "comments": comments,
            }

            # Add wire counts (document-level and block-level)
            try:
                await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
                ws_channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
                if ws_channel and ws_channel.doc:
                    outgoing = _get_wires_for_document(ws_channel.doc, document_id, "outgoing")
                    incoming = _get_wires_for_document(ws_channel.doc, document_id, "incoming")

                    # Separate document-level vs block-level wires
                    doc_level = 0
                    block_wire_counts: Dict[str, int] = {}
                    for wire in outgoing:
                        block_id = wire.get("sourceBlockId")
                        if block_id:
                            block_wire_counts[block_id] = block_wire_counts.get(block_id, 0) + 1
                        else:
                            doc_level += 1
                    for wire in incoming:
                        block_id = wire.get("targetBlockId")
                        if block_id:
                            block_wire_counts[block_id] = block_wire_counts.get(block_id, 0) + 1
                        else:
                            doc_level += 1

                    result["wires"] = {
                        "outgoing": len(outgoing),
                        "incoming": len(incoming),
                        "total": len(outgoing) + len(incoming),
                        "document_level": doc_level,
                        "by_block": block_wire_counts if block_wire_counts else None,
                    }
                else:
                    result["wires"] = {"outgoing": 0, "incoming": 0, "total": 0, "by_block": None}
            except Exception as e:
                logger.debug(
                    "Failed to fetch wire counts for read_document (non-fatal)",
                    extra_context={"document_id": document_id, "error": str(e)},
                )
                result["wires"] = None

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
        name="export_document",
        title="Export Document",
        description=(
            "Export a document in a specified format. Returns the document content "
            "converted to the requested format along with the document title.\n\n"
            "Formats:\n"
            "- 'xml': Raw TipTap XML (lossless, includes block IDs and all attributes)\n"
            "- 'markdown': Clean Markdown with headings, lists, code blocks, bold/italic/etc.\n"
            "- 'html': Self-contained HTML with Garden theming (serif typography, dark/light mode)\n\n"
            "Use this for clipboard export, backups, or feeding document content to other tools."
        ),
    )
    async def export_document_tool(
        graph_id: str,
        document_id: str,
        format: str = "markdown",
        context: Context | None = None,
    ) -> dict:
        """Export a document in xml, markdown, or html format.

        Args:
            graph_id: The graph containing the document
            document_id: The document to export
            format: Export format - "xml", "markdown" (default), or "html"
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if format not in ("xml", "markdown", "html"):
            raise ValueError("format must be 'xml', 'markdown', or 'html'")

        try:
            # Validate document exists in workspace before connecting
            await _validate_document_in_workspace(graph_id, document_id, auth.user_id)

            # Connect to the document channel
            await hp_client.connect_document(graph_id, document_id, user_id=auth.user_id)

            channel = hp_client.get_document_channel(graph_id, document_id, user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Document channel not found: {graph_id}/{document_id}")

            reader = DocumentReader(channel.doc)
            xml_content = reader.to_xml()

            # Extract title from content
            title = extract_title_from_xml(xml_content) or document_id

            # Convert to requested format
            if format == "markdown":
                content = tiptap_xml_to_markdown(xml_content)
            elif format == "html":
                content = tiptap_xml_to_html(xml_content, title=title, themed=True)
            else:
                content = xml_content

            return {
                "graph_id": graph_id,
                "document_id": document_id,
                "title": title,
                "format": format,
                "content": content,
            }

        except Exception as e:
            logger.error(
                "Failed to export document",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "format": format,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to export document: {e}")

    @server.tool(
        name="write_document",
        title="Write Document Content",
        description="""Replaces document content with TipTap XML. Syncs to UI in real-time.

WARNING: This REPLACES all content. For collaborative editing, prefer append_to_document.

Plain text is accepted: if the content doesn't start with '<', each paragraph (separated by blank lines) is auto-wrapped in <paragraph> tags. Use XML when you need formatting or specific block types.

Blocks: paragraph, heading (level="1-3"), bulletList, orderedList, blockquote, codeBlock (language="..."), taskList (taskItem checked="true"), horizontalRule
Marks (nestable): strong, em, strike, code, mark (highlight), a (href="..."), footnote (data-footnote-content="..."), commentMark (data-comment-id="...")
Example: <paragraph>Text with <mark>highlight</mark> and a note<footnote data-footnote-content="This is a footnote"/></paragraph>

Comments: Pass a dict mapping comment IDs to metadata. Comment IDs must match data-comment-id attributes in the content.
Example comments: {"comment-1": {"text": "Great point!", "author": "Claude"}}

Markdown is also accepted and auto-converted to TipTap XML.

NOT for: editing existing documents (use edit_block_text, update_block, or insert_block instead). Only use write_document for brand-new documents or when the user explicitly asks for a full rewrite.""",
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

            xml_content = _ensure_xml_multiblock(content)

            def write_content_and_comments(doc: Any) -> None:
                writer = DocumentWriter(doc)
                writer.replace_all_content(xml_content)
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
            title = extract_title_from_xml(xml_content) or document_id
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            await hp_client.transact_workspace(
                graph_id,
                lambda doc: WorkspaceWriter(doc).upsert_document(document_id, title),
                user_id=auth.user_id,
            )

            return {
                "success": True,
                "graph_id": graph_id,
                "document_id": document_id,
                "title": title,
            }

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
            "provide full XML like <heading level=\"2\">Title</heading> or <listItem listType=\"bullet\">...</listItem>. "
            "Plain text without XML tags is auto-wrapped in a <paragraph>. "
            "Only accepts a single top-level XML block element per call. To append multiple blocks, make multiple calls. "
            "Markdown is also accepted and auto-converted."
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
            # Validate document exists in workspace before appending
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id)

            # Connect to the document channel with user context
            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            # Auto-wrap plain text in a paragraph element
            content = _ensure_xml(text)

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
            "Returns the folder and file structure of a graph's workspace, including all documents, "
            "artifacts, and folders with their titles and organization. This is the primary tool for "
            "understanding what's in a graph. Use get_user_location first if you need to know which "
            "graph the user is in.\n\n"
            "This is always complete — prefer it over sparql_query for discovering what documents exist."
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

            result: dict[str, Any] = {
                "graph_id": graph_id,
                "workspace": snapshot,
            }

            # Fail if workspace is completely empty — almost certainly a
            # nonexistent graph or wrong backend connection.
            has_docs = bool(snapshot.get("documents"))
            has_folders = bool(snapshot.get("folders"))
            has_artifacts = bool(snapshot.get("artifacts"))
            if not has_docs and not has_folders and not has_artifacts:
                raise RuntimeError(
                    f"Graph '{graph_id}' not found — workspace is completely empty "
                    f"(no documents, folders, or artifacts). This usually means the "
                    f"graph ID is wrong or you're connected to the wrong backend. "
                    f"Use list_graphs to see available graphs."
                )

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

    @server.tool(
        name="read_artifact",
        title="Read Artifact Content",
        description=(
            "Read an artifact's metadata and, if the artifact has been ingested, its document content. "
            "Returns metadata (name, fileType, mimeType, status, ingestedDocId, etc.) always. "
            "If the artifact has been ingested into a document (ingestedDocId is set), also returns "
            "the ingested document's TipTap XML content and comments.\n\n"
            "If the artifact has not been ingested yet, the response will not include 'ingested_document'. "
            "Use ingest_artifact to convert the artifact into a readable document first."
        ),
    )
    async def read_artifact_tool(
        graph_id: str,
        artifact_id: str,
        context: Context | None = None,
    ) -> dict:
        """Read artifact metadata and ingested document content if available."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not artifact_id or not artifact_id.strip():
            raise ValueError("artifact_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)
            metadata = reader.get_artifact(artifact_id.strip())

            if not metadata:
                raise RuntimeError(f"Artifact '{artifact_id}' not found in graph '{graph_id}'")

            result: dict[str, Any] = {
                "graph_id": graph_id.strip(),
                "artifact_id": artifact_id.strip(),
                "metadata": metadata,
            }

            # If the artifact has been ingested, read the linked document content
            ingested_doc_id = metadata.get("ingestedDocId")
            if ingested_doc_id:
                try:
                    await hp_client.connect_document(graph_id.strip(), ingested_doc_id, user_id=auth.user_id)
                    doc_channel = hp_client.get_document_channel(graph_id.strip(), ingested_doc_id, user_id=auth.user_id)
                    if doc_channel is not None:
                        doc_reader = DocumentReader(doc_channel.doc)
                        result["ingested_document"] = {
                            "document_id": ingested_doc_id,
                            "content": doc_reader.to_xml(),
                            "comments": doc_reader.get_all_comments(),
                        }
                except Exception as doc_err:
                    logger.warning(
                        "Artifact has ingestedDocId but document could not be read",
                        extra_context={
                            "artifact_id": artifact_id,
                            "ingested_doc_id": ingested_doc_id,
                            "error": str(doc_err),
                        },
                    )

            return result

        except Exception as e:
            logger.error(
                "Failed to read artifact",
                extra_context={
                    "graph_id": graph_id,
                    "artifact_id": artifact_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to read artifact: {e}")

    @server.tool(
        name="upload_artifact",
        title="Upload Artifact",
        description=(
            "Upload a local file as a new artifact in a graph. "
            "Reads the file from the local filesystem and uploads it to the platform.\n\n"
            "Supported formats: TXT, MD, HTML, and text-like files "
            "(.log, .csv, .json, .xml, .yaml, .py, .js, .ts, .sql, etc.)\n"
            "Maximum size: 50MB\n\n"
            "After uploading, use ingest_artifact to convert the artifact into a "
            "readable/editable document."
        ),
    )
    async def upload_artifact_tool(
        graph_id: str,
        file_path: str,
        parent_id: Optional[str] = None,
        label: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Upload a local file as a new artifact via the backend API."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not file_path or not file_path.strip():
            raise ValueError("file_path is required and cannot be empty")

        path = Path(file_path.strip())
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        file_size = path.stat().st_size
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            raise ValueError(
                f"File too large ({file_size / (1024 * 1024):.1f}MB). Maximum size: 50MB"
            )

        try:
            content = path.read_bytes()
            filename = label or path.name
            mime_type, _ = mimetypes.guess_type(path.name)
            if mime_type is None:
                mime_type = "application/octet-stream"

            url = f"{backend_config.base_url}/artifacts/{graph_id.strip()}/upload"

            data: dict[str, str] = {}
            if parent_id is not None and parent_id.strip():
                data["parent_id"] = parent_id.strip()

            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                resp = await client.post(
                    url,
                    files={"file": (filename, content, mime_type)},
                    data=data,
                    headers=auth.http_headers(),
                )
                if resp.status_code != 201:
                    raise RuntimeError(
                        f"Backend returned {resp.status_code}: {resp.text[:300]}"
                    )
                result = resp.json()

            return {
                "success": True,
                "artifact_id": result.get("artifactId"),
                "file_type": result.get("fileType"),
                "label": result.get("label"),
                "storage_key": result.get("storageKey"),
            }

        except Exception as e:
            logger.error(
                "Failed to upload artifact",
                extra_context={
                    "graph_id": graph_id,
                    "file_path": file_path,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to upload artifact: {e}")

    @server.tool(
        name="ingest_artifact",
        title="Ingest Artifact",
        description=(
            "Ingest an artifact into a readable document. Converts uploaded files (PDFs, etc.) "
            "into TipTap documents that can be read and searched.\n\n"
            "Modes:\n"
            "- 'ingest' (default): Creates a read-only e-reader document. Best for reference material.\n"
            "- 'import': Creates a fully editable document. Best when you need to modify the content.\n\n"
            "After ingestion, use read_artifact or read_document with the returned document_id "
            "to access the content."
        ),
    )
    async def ingest_artifact_tool(
        graph_id: str,
        artifact_id: str,
        mode: str = "ingest",
        title: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Ingest an artifact into a document via the backend API."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not artifact_id or not artifact_id.strip():
            raise ValueError("artifact_id is required and cannot be empty")
        if mode not in ("ingest", "import"):
            raise ValueError("mode must be 'ingest' or 'import'")

        try:
            url = f"{backend_config.base_url}/artifacts/{graph_id.strip()}/{artifact_id.strip()}/import"
            body: dict[str, Any] = {
                "readOnly": mode == "ingest",
                "useYdocPath": True,
            }
            if title is not None:
                body["title"] = title

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                resp = await client.post(url, json=body, headers=auth.http_headers())
                if resp.status_code not in (200, 201):
                    raise RuntimeError(
                        f"Backend returned {resp.status_code}: {resp.text[:300]}"
                    )
                data = resp.json()

            return {
                "success": True,
                "graph_id": graph_id.strip(),
                "artifact_id": artifact_id.strip(),
                "document_id": data.get("documentId"),
                "title": data.get("title"),
                "mode": mode,
                "read_only": mode == "ingest",
            }

        except Exception as e:
            logger.error(
                "Failed to ingest artifact",
                extra_context={
                    "graph_id": graph_id,
                    "artifact_id": artifact_id,
                    "mode": mode,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to ingest artifact: {e}")

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
                raise RuntimeError(
                    f"Document '{document_id}' not found in graph '{graph_id}'. "
                    f"Use get_workspace to see available documents."
                )

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
                raise RuntimeError(
                    f"Document '{document_id}' not found in graph '{graph_id}'. "
                    f"Use get_workspace to see available documents."
                )

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
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id)

            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            channel = hp_client.get_document_channel(graph_id.strip(), document_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Document channel not found: {graph_id}/{document_id}")

            reader = DocumentReader(channel.doc)
            block_info = reader.get_block_info(block_id.strip())

            if block_info is None:
                raise RuntimeError(f"Block '{block_id}' not found in document '{document_id}'.")

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
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id)

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
            "This is the most surgical edit - only modifies what you specify. "
            "Plain text in xml_content is auto-wrapped in a <paragraph>. "
            "Markdown is also accepted and auto-converted."
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

        # Auto-wrap plain text in a paragraph element
        resolved_xml = _ensure_xml(xml_content) if xml_content else None

        try:
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id)

            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            def perform_update(doc: Any) -> None:
                writer = DocumentWriter(doc)
                if resolved_xml:
                    # Full content replacement (preserves block_id)
                    writer.replace_block_by_id(block_id.strip(), resolved_xml)
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
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id)

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
            "For appending to the end, use append_to_document instead. "
            "Plain text in xml_content is auto-wrapped in a <paragraph>. "
            "Markdown is also accepted and auto-converted."
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

        # Auto-wrap plain text in a paragraph element
        resolved_xml = _ensure_xml(xml_content)

        try:
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id)

            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            new_block_id: str = ""

            def perform_insert(doc: Any) -> None:
                nonlocal new_block_id
                writer = DocumentWriter(doc)
                if position == "after":
                    new_block_id = writer.insert_block_after_id(
                        reference_block_id.strip(), resolved_xml
                    )
                else:
                    new_block_id = writer.insert_block_before_id(
                        reference_block_id.strip(), resolved_xml
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
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id)

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
            "individual update_block calls. Each update object should have a block_id (required), "
            "and optionally attributes (object) and/or xml_content (string), matching the "
            "parameters of update_block. Returns results for each update."
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
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id)

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
                        # Accept both "xml_content" (matches update_block param name)
                        # and "content" (legacy) for the XML content key
                        xml_content = update.get("xml_content") or update.get("content")
                        attrs = update.get("attributes")

                        if xml_content is None and attrs is None:
                            results.append({
                                "block_id": block_id,
                                "error": "No xml_content or attributes provided — nothing to update",
                            })
                            continue

                        if xml_content:
                            writer.replace_block_by_id(block_id, xml_content)
                        if attrs:
                            writer.update_block_attributes(block_id, attrs)
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
