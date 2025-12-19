#!/usr/bin/env python3
"""
MCP Scripting Interface - Call MCP tools directly from Python.

This module provides a simple interface to script MCP operations without
going through the JSON-RPC protocol. Perfect for testing and automation.

Usage:
    from playground.mcp_script import MCP

    async def main():
        mcp = MCP("http://localhost:8765")  # Point to playground or real backend

        # Read a document
        content = await mcp.read_document("my-graph", "my-doc")
        print(content)

        # Write content
        await mcp.write_document("my-graph", "my-doc", "<paragraph>Hello!</paragraph>")

        # Append a block
        await mcp.append("<paragraph>New paragraph</paragraph>")

        # Query blocks
        blocks = await mcp.query_blocks(block_type="paragraph")

        # Block operations
        await mcp.insert_block_after("block-123", "<paragraph>After</paragraph>")
        await mcp.update_block("block-123", indent=1, checked=True)
        await mcp.delete_block("block-123")

        await mcp.close()

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neem.hocuspocus.client import HocuspocusClient
from neem.hocuspocus.document import DocumentReader, DocumentWriter
from neem.hocuspocus.workspace import WorkspaceReader, WorkspaceWriter


@dataclass
class BlockInfo:
    """Information about a document block."""
    block_id: str
    index: int
    type: str
    text: str
    xml: str
    attributes: Dict[str, Any]
    prev_id: Optional[str] = None
    next_id: Optional[str] = None


class MCP:
    """
    Scripting interface for MCP document operations.

    Provides a high-level Python API for all MCP tools, bypassing the JSON-RPC
    protocol for direct, scriptable access to document operations.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        user_id: str = "test-user",
        token: str = "test-token",
    ):
        """
        Initialize MCP scripting interface.

        Args:
            base_url: Backend URL (playground or real platform)
            user_id: User ID for dev mode
            token: Token for dev mode
        """
        self._base_url = base_url
        self._user_id = user_id
        self._token = token
        self._client: Optional[HocuspocusClient] = None

        # Current document context (set by read_document or connect)
        self._current_graph: Optional[str] = None
        self._current_doc: Optional[str] = None

    async def _ensure_client(self) -> HocuspocusClient:
        """Get or create the HocuspocusClient."""
        if self._client is None:
            self._client = HocuspocusClient(
                base_url=self._base_url,
                token_provider=lambda: self._token,
                dev_user_id=self._user_id,
            )
        return self._client

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None

    # -------------------------------------------------------------------------
    # Document Operations
    # -------------------------------------------------------------------------

    async def connect(self, graph_id: str, doc_id: str) -> None:
        """
        Connect to a document (sets it as current context).

        After connecting, you can use shorthand methods without specifying
        graph_id and doc_id each time.
        """
        client = await self._ensure_client()
        await client.connect_document(graph_id, doc_id)
        self._current_graph = graph_id
        self._current_doc = doc_id

    async def read_document(
        self,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Read document content as TipTap XML.

        Returns the full XML content of the document.
        """
        graph_id = graph_id or self._current_graph
        doc_id = doc_id or self._current_doc
        if not graph_id or not doc_id:
            raise ValueError("graph_id and doc_id required (connect first or specify)")

        client = await self._ensure_client()
        await client.connect_document(graph_id, doc_id)
        self._current_graph = graph_id
        self._current_doc = doc_id

        channel = client.get_document_channel(graph_id, doc_id)
        if not channel:
            raise RuntimeError(f"Document not found: {graph_id}/{doc_id}")

        reader = DocumentReader(channel.doc)
        return reader.to_xml()

    async def write_document(
        self,
        content: str,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> None:
        """
        Replace document content with TipTap XML.

        WARNING: This replaces all existing content!
        """
        graph_id = graph_id or self._current_graph
        doc_id = doc_id or self._current_doc
        if not graph_id or not doc_id:
            raise ValueError("graph_id and doc_id required")

        client = await self._ensure_client()
        await client.connect_document(graph_id, doc_id)

        await client.transact_document(
            graph_id, doc_id,
            lambda doc: DocumentWriter(doc).replace_all_content(content)
        )

    async def append(
        self,
        content: str,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Append a block to the document.

        Args:
            content: TipTap XML (or plain text which gets wrapped in <paragraph>)

        Returns:
            The new block's ID
        """
        graph_id = graph_id or self._current_graph
        doc_id = doc_id or self._current_doc
        if not graph_id or not doc_id:
            raise ValueError("graph_id and doc_id required")

        # Wrap plain text in paragraph
        if not content.strip().startswith("<"):
            import html
            content = f"<paragraph>{html.escape(content)}</paragraph>"

        client = await self._ensure_client()
        await client.connect_document(graph_id, doc_id)

        new_block_id = ""

        def do_append(doc):
            nonlocal new_block_id
            writer = DocumentWriter(doc)
            writer.append_block(content)
            reader = DocumentReader(doc)
            count = reader.get_block_count()
            if count > 0:
                block = reader.get_block_at(count - 1)
                if block and hasattr(block, "attributes"):
                    attrs = block.attributes
                    new_block_id = attrs.get("data-block-id") if "data-block-id" in attrs else ""

        await client.transact_document(graph_id, doc_id, do_append)
        return new_block_id

    async def clear(
        self,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> None:
        """Clear all content from the document."""
        graph_id = graph_id or self._current_graph
        doc_id = doc_id or self._current_doc
        if not graph_id or not doc_id:
            raise ValueError("graph_id and doc_id required")

        client = await self._ensure_client()
        await client.connect_document(graph_id, doc_id)

        await client.transact_document(
            graph_id, doc_id,
            lambda doc: DocumentWriter(doc).clear_content()
        )

    # -------------------------------------------------------------------------
    # Block Operations
    # -------------------------------------------------------------------------

    async def get_block(
        self,
        block_id: str,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Optional[BlockInfo]:
        """Get detailed information about a block."""
        graph_id = graph_id or self._current_graph
        doc_id = doc_id or self._current_doc
        if not graph_id or not doc_id:
            raise ValueError("graph_id and doc_id required")

        client = await self._ensure_client()
        await client.connect_document(graph_id, doc_id)

        channel = client.get_document_channel(graph_id, doc_id)
        if not channel:
            return None

        reader = DocumentReader(channel.doc)
        info = reader.get_block_info(block_id)
        if not info:
            return None

        return BlockInfo(
            block_id=info["block_id"],
            index=info["index"],
            type=info["type"],
            text=info["text_content"],
            xml=info["xml"],
            attributes=info["attributes"],
            prev_id=info["context"]["prev_block_id"],
            next_id=info["context"]["next_block_id"],
        )

    async def query_blocks(
        self,
        block_type: Optional[str] = None,
        indent: Optional[int] = None,
        indent_gte: Optional[int] = None,
        indent_lte: Optional[int] = None,
        list_type: Optional[str] = None,
        checked: Optional[bool] = None,
        text_contains: Optional[str] = None,
        limit: int = 50,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query blocks matching criteria.

        Args:
            block_type: paragraph, heading, listItem, etc.
            indent: Exact indent level
            indent_gte: Indent >= value
            indent_lte: Indent <= value
            list_type: bullet, ordered, task
            checked: For task items
            text_contains: Search text
            limit: Max results

        Returns:
            List of block summaries
        """
        graph_id = graph_id or self._current_graph
        doc_id = doc_id or self._current_doc
        if not graph_id or not doc_id:
            raise ValueError("graph_id and doc_id required")

        client = await self._ensure_client()
        await client.connect_document(graph_id, doc_id)

        channel = client.get_document_channel(graph_id, doc_id)
        if not channel:
            return []

        reader = DocumentReader(channel.doc)
        return reader.query_blocks(
            block_type=block_type,
            indent=indent,
            indent_gte=indent_gte,
            indent_lte=indent_lte,
            list_type=list_type,
            checked=checked,
            text_contains=text_contains,
            limit=limit,
        )

    async def insert_block_after(
        self,
        block_id: str,
        content: str,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Insert a block after the specified block. Returns new block ID."""
        graph_id = graph_id or self._current_graph
        doc_id = doc_id or self._current_doc
        if not graph_id or not doc_id:
            raise ValueError("graph_id and doc_id required")

        client = await self._ensure_client()
        await client.connect_document(graph_id, doc_id)

        new_id = ""

        def do_insert(doc):
            nonlocal new_id
            writer = DocumentWriter(doc)
            new_id = writer.insert_block_after_id(block_id, content)

        await client.transact_document(graph_id, doc_id, do_insert)
        return new_id

    async def insert_block_before(
        self,
        block_id: str,
        content: str,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Insert a block before the specified block. Returns new block ID."""
        graph_id = graph_id or self._current_graph
        doc_id = doc_id or self._current_doc
        if not graph_id or not doc_id:
            raise ValueError("graph_id and doc_id required")

        client = await self._ensure_client()
        await client.connect_document(graph_id, doc_id)

        new_id = ""

        def do_insert(doc):
            nonlocal new_id
            writer = DocumentWriter(doc)
            new_id = writer.insert_block_before_id(block_id, content)

        await client.transact_document(graph_id, doc_id, do_insert)
        return new_id

    async def update_block(
        self,
        block_id: str,
        content: Optional[str] = None,
        indent: Optional[int] = None,
        checked: Optional[bool] = None,
        list_type: Optional[str] = None,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        **attrs,
    ) -> None:
        """
        Update a block's content and/or attributes.

        Args:
            block_id: Block to update
            content: New XML content (replaces entire block)
            indent: Set indent level
            checked: Set checked state (for tasks)
            list_type: Set list type (bullet, ordered, task)
            **attrs: Additional attributes to set
        """
        graph_id = graph_id or self._current_graph
        doc_id = doc_id or self._current_doc
        if not graph_id or not doc_id:
            raise ValueError("graph_id and doc_id required")

        client = await self._ensure_client()
        await client.connect_document(graph_id, doc_id)

        # Build attributes dict
        attributes = {}
        if indent is not None:
            attributes["indent"] = indent
        if checked is not None:
            attributes["checked"] = checked
        if list_type is not None:
            attributes["listType"] = list_type
        attributes.update(attrs)

        def do_update(doc):
            writer = DocumentWriter(doc)
            if content:
                writer.replace_block_by_id(block_id, content)
            if attributes:
                writer.update_block_attributes(block_id, attributes)

        await client.transact_document(graph_id, doc_id, do_update)

    async def delete_block(
        self,
        block_id: str,
        cascade: bool = False,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[str]:
        """
        Delete a block. Returns list of deleted block IDs.

        Args:
            block_id: Block to delete
            cascade: Also delete indent-children
        """
        graph_id = graph_id or self._current_graph
        doc_id = doc_id or self._current_doc
        if not graph_id or not doc_id:
            raise ValueError("graph_id and doc_id required")

        client = await self._ensure_client()
        await client.connect_document(graph_id, doc_id)

        deleted = []

        def do_delete(doc):
            nonlocal deleted
            writer = DocumentWriter(doc)
            deleted = writer.delete_block_by_id(block_id, cascade_children=cascade)

        await client.transact_document(graph_id, doc_id, do_delete)
        return deleted

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    async def blocks(
        self,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all blocks in the document."""
        return await self.query_blocks(limit=1000, graph_id=graph_id, doc_id=doc_id)

    async def paragraphs(
        self,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all paragraphs in the document."""
        return await self.query_blocks(block_type="paragraph", graph_id=graph_id, doc_id=doc_id)

    async def headings(
        self,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all headings in the document."""
        return await self.query_blocks(block_type="heading", graph_id=graph_id, doc_id=doc_id)

    async def tasks(
        self,
        completed: Optional[bool] = None,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get task items, optionally filtered by completion status."""
        return await self.query_blocks(
            list_type="task",
            checked=completed,
            graph_id=graph_id,
            doc_id=doc_id,
        )

    async def search(
        self,
        text: str,
        graph_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for blocks containing text."""
        return await self.query_blocks(text_contains=text, graph_id=graph_id, doc_id=doc_id)


# -------------------------------------------------------------------------
# Script-friendly functions (no class instantiation needed)
# -------------------------------------------------------------------------

_default_mcp: Optional[MCP] = None


def get_mcp(
    base_url: str = "http://localhost:8765",
    user_id: str = "test-user",
    token: str = "test-token",
) -> MCP:
    """Get or create the default MCP instance."""
    global _default_mcp
    if _default_mcp is None:
        _default_mcp = MCP(base_url, user_id, token)
    return _default_mcp


async def read(graph_id: str, doc_id: str) -> str:
    """Quick read function."""
    return await get_mcp().read_document(graph_id, doc_id)


async def write(graph_id: str, doc_id: str, content: str) -> None:
    """Quick write function."""
    await get_mcp().write_document(content, graph_id, doc_id)


async def append(graph_id: str, doc_id: str, content: str) -> str:
    """Quick append function."""
    mcp = get_mcp()
    await mcp.connect(graph_id, doc_id)
    return await mcp.append(content)


# -------------------------------------------------------------------------
# Demo / CLI
# -------------------------------------------------------------------------

async def demo():
    """Run a demo of MCP scripting."""
    print("=" * 60)
    print("MCP Scripting Demo")
    print("=" * 60)

    mcp = MCP()

    try:
        # Connect to test document
        print("\n1. Connecting to test-graph/test-doc...")
        await mcp.connect("test-graph", "test-doc")

        # Clear and write initial content
        print("\n2. Writing initial content...")
        await mcp.write_document("""
<heading level="1">MCP Scripting Demo</heading>
<paragraph>This document was created by a script!</paragraph>
<listItem listType="task">First task</listItem>
<listItem listType="task">Second task</listItem>
<listItem listType="task" checked="true">Completed task</listItem>
""".strip())

        # Read back
        print("\n3. Reading content:")
        content = await mcp.read_document()
        print(f"   {content[:200]}...")

        # Query blocks
        print("\n4. Querying blocks:")
        blocks = await mcp.blocks()
        for b in blocks:
            print(f"   [{b['index']}] {b['type']}: {b['text_preview'][:40]}")

        # Append a block
        print("\n5. Appending a paragraph...")
        new_id = await mcp.append("This was appended by the script!")
        print(f"   New block ID: {new_id}")

        # Query tasks
        print("\n6. Finding incomplete tasks:")
        incomplete = await mcp.tasks(completed=False)
        for t in incomplete:
            print(f"   - {t['text_preview']}")

        # Mark a task complete
        if incomplete:
            task_id = incomplete[0]["block_id"]
            print(f"\n7. Marking task {task_id} as complete...")
            await mcp.update_block(task_id, checked=True)

        # Final content
        print("\n8. Final content:")
        content = await mcp.read_document()
        print(content)

        print("\n" + "=" * 60)
        print("Demo complete! Check http://localhost:8765 to see changes.")
        print("=" * 60)

    finally:
        await mcp.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP Scripting Interface")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--read", nargs=2, metavar=("GRAPH", "DOC"), help="Read a document")
    parser.add_argument("--append", nargs=3, metavar=("GRAPH", "DOC", "TEXT"), help="Append text")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(demo())
    elif args.read:
        async def do_read():
            content = await read(args.read[0], args.read[1])
            print(content)
        asyncio.run(do_read())
    elif args.append:
        async def do_append():
            block_id = await append(args.append[0], args.append[1], args.append[2])
            print(f"Appended block: {block_id}")
        asyncio.run(do_append())
    else:
        parser.print_help()
