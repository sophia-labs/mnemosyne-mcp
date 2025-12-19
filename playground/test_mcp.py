#!/usr/bin/env python3
"""
Test MCP document operations against the playground server.

This script demonstrates how to use the MCP's HocuspocusClient and
document tools against the local playground server.

Usage:
    # First, start the playground server in another terminal:
    uv run python playground/server.py

    # Then run this test script:
    uv run python playground/test_mcp.py
"""

import asyncio
import os
import sys

# Ensure we're using the local playground server
os.environ["MNEMOSYNE_FASTAPI_URL"] = "http://localhost:8765"
os.environ["MNEMOSYNE_DEV_USER_ID"] = "test-user"
os.environ["MNEMOSYNE_DEV_TOKEN"] = "test-token"

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neem.hocuspocus.client import HocuspocusClient
from neem.hocuspocus.document import DocumentReader, DocumentWriter


async def test_basic_operations():
    """Test basic document read/write operations."""
    print("\n" + "=" * 60)
    print("Testing MCP Document Operations")
    print("=" * 60)

    # Create client pointing to playground server
    client = HocuspocusClient(
        base_url="http://localhost:8765",
        token_provider=lambda: "test-token",
        dev_user_id="test-user",
    )

    graph_id = "test-graph"
    doc_id = "test-doc"

    try:
        print(f"\n1. Connecting to document {graph_id}/{doc_id}...")
        await client.connect_document(graph_id, doc_id)
        print("   Connected!")

        # Get the document channel
        channel = client.get_document_channel(graph_id, doc_id)
        if not channel:
            print("   ERROR: No channel found!")
            return

        # Read current content
        print("\n2. Reading current content...")
        reader = DocumentReader(channel.doc)
        content = reader.to_xml()
        print(f"   Current content: {content[:200] if content else '(empty)'}")
        print(f"   Block count: {reader.get_block_count()}")

        # Append a paragraph
        print("\n3. Appending a paragraph...")
        await client.transact_document(
            graph_id, doc_id,
            lambda doc: DocumentWriter(doc).append_block(
                "<paragraph>Hello from MCP test script!</paragraph>"
            )
        )
        print("   Paragraph appended!")

        # Read updated content
        content = reader.to_xml()
        print(f"   Updated content: {content[:300] if content else '(empty)'}")

        # Add a heading
        print("\n4. Adding a heading...")
        await client.transact_document(
            graph_id, doc_id,
            lambda doc: DocumentWriter(doc).append_block(
                '<heading level="2">MCP Test Section</heading>'
            )
        )
        print("   Heading added!")

        # Add a bullet list
        print("\n5. Adding a bullet list...")
        await client.transact_document(
            graph_id, doc_id,
            lambda doc: DocumentWriter(doc).append_block(
                '<bulletList><listItem><paragraph>Item one</paragraph></listItem><listItem><paragraph>Item two</paragraph></listItem></bulletList>'
            )
        )
        print("   Bullet list added!")

        # Final content
        print("\n6. Final document state:")
        content = reader.to_xml()
        print(f"   {content}")
        print(f"   Total blocks: {reader.get_block_count()}")

        # Query blocks
        print("\n7. Querying blocks...")
        all_blocks = reader.query_blocks(limit=10)
        for block in all_blocks:
            print(f"   [{block['index']}] {block['type']}: {block['text_preview'][:50]}")

        print("\n" + "=" * 60)
        print("SUCCESS! Check the browser UI to see the changes.")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await client.close()


async def test_block_operations():
    """Test surgical block operations."""
    print("\n" + "=" * 60)
    print("Testing Block-Level Operations")
    print("=" * 60)

    client = HocuspocusClient(
        base_url="http://localhost:8765",
        token_provider=lambda: "test-token",
        dev_user_id="test-user",
    )

    graph_id = "test-graph"
    doc_id = "test-doc"

    try:
        await client.connect_document(graph_id, doc_id)
        channel = client.get_document_channel(graph_id, doc_id)
        reader = DocumentReader(channel.doc)

        # Find a block by ID
        print("\n1. Getting first block info...")
        block = reader.get_block_at(0)
        if block:
            block_id = block.attributes.get("data-block-id")
            print(f"   Block ID: {block_id}")
            print(f"   Type: {block.tag}")

            # Get detailed info
            info = reader.get_block_info(block_id)
            if info:
                print(f"   Text: {info['text_content'][:50]}")
                print(f"   Context: prev={info['context']['prev_block_id']}, next={info['context']['next_block_id']}")

            # Insert a block after it
            print("\n2. Inserting block after first block...")
            await client.transact_document(
                graph_id, doc_id,
                lambda doc: DocumentWriter(doc).insert_block_after_id(
                    block_id,
                    "<paragraph>Inserted after first block!</paragraph>"
                )
            )
            print("   Block inserted!")

        # Update a block's attributes
        blocks = reader.query_blocks(block_type="listItem", limit=1)
        if blocks:
            print("\n3. Updating list item indent...")
            list_block_id = blocks[0]["block_id"]
            await client.transact_document(
                graph_id, doc_id,
                lambda doc: DocumentWriter(doc).update_block_attributes(
                    list_block_id,
                    {"indent": 1}
                )
            )
            print(f"   Updated indent for block {list_block_id}")

        print("\n" + "=" * 60)
        print("Block operations complete!")
        print("=" * 60)

    finally:
        await client.close()


async def interactive_mode():
    """Interactive mode for manual testing."""
    print("\n" + "=" * 60)
    print("Interactive MCP Testing Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  read     - Read document content")
    print("  append   - Append a paragraph")
    print("  blocks   - List all blocks")
    print("  clear    - Clear document")
    print("  quit     - Exit")
    print("=" * 60)

    client = HocuspocusClient(
        base_url="http://localhost:8765",
        token_provider=lambda: "test-token",
        dev_user_id="test-user",
    )

    graph_id = "test-graph"
    doc_id = "test-doc"

    try:
        await client.connect_document(graph_id, doc_id)
        channel = client.get_document_channel(graph_id, doc_id)
        reader = DocumentReader(channel.doc)

        while True:
            try:
                cmd = input("\n> ").strip().lower()

                if cmd == "quit":
                    break
                elif cmd == "read":
                    content = reader.to_xml()
                    print(f"Content:\n{content if content else '(empty)'}")
                elif cmd == "append":
                    text = input("Text: ")
                    await client.transact_document(
                        graph_id, doc_id,
                        lambda doc: DocumentWriter(doc).append_block(
                            f"<paragraph>{text}</paragraph>"
                        )
                    )
                    print("Appended!")
                elif cmd == "blocks":
                    blocks = reader.query_blocks(limit=20)
                    for b in blocks:
                        print(f"  [{b['index']}] {b['type']}: {b['text_preview'][:40]}")
                elif cmd == "clear":
                    await client.transact_document(
                        graph_id, doc_id,
                        lambda doc: DocumentWriter(doc).clear_content()
                    )
                    print("Cleared!")
                else:
                    print("Unknown command")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

    finally:
        await client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test MCP document operations")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--blocks", "-b", action="store_true", help="Test block operations")
    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_mode())
    elif args.blocks:
        asyncio.run(test_block_operations())
    else:
        asyncio.run(test_basic_operations())
