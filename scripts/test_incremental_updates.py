#!/usr/bin/env python3
"""Test script to validate incremental Y.js updates.

This script tests that the new transact_document() pattern correctly:
1. Captures state BEFORE changes
2. Generates INCREMENTAL diffs (not full state)
3. Produces smaller update sizes for surgical edits vs full replacement

Usage:
    cd mnemosyne-mcp
    uv run python scripts/test_incremental_updates.py
"""

import pycrdt

from neem.hocuspocus.document import DocumentWriter, DocumentReader


def test_incremental_vs_full_update():
    """Compare update sizes: incremental append vs full replacement."""
    print("=" * 60)
    print("TEST: Incremental vs Full Update Size Comparison")
    print("=" * 60)

    # Create a document with some initial content
    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)

    # Add 10 paragraphs of content
    initial_content = "\n".join(
        f"<paragraph>This is paragraph {i} with some content.</paragraph>"
        for i in range(10)
    )
    writer.replace_all_content(initial_content)

    reader = DocumentReader(doc)
    print(f"\nInitial document has {writer.get_block_count()} blocks")
    print(f"Initial content preview: {reader.to_xml()[:200]}...")

    # Now test incremental append
    print("\n--- Test 1: Incremental Append ---")

    # Capture state BEFORE changes (this is what transact_document does)
    old_state = doc.get_state()

    # Make a small change: append one paragraph
    writer.append_block("<paragraph>New paragraph added incrementally!</paragraph>")

    # Get INCREMENTAL update (diff from old state)
    incremental_update = doc.get_update(old_state)

    # Get FULL state update (what the old code did)
    full_update = doc.get_update()

    print(f"Incremental update size: {len(incremental_update)} bytes")
    print(f"Full state update size:  {len(full_update)} bytes")
    print(f"Ratio: {len(full_update) / len(incremental_update):.1f}x larger for full state")

    # The incremental update should be MUCH smaller
    assert len(incremental_update) < len(full_update), "Incremental should be smaller!"
    print("✓ Incremental update is smaller than full state update")

    # Verify the incremental update can be applied to a fresh doc
    print("\n--- Test 2: Apply Incremental Update to Fresh Doc ---")

    # Test this properly by simulating a sync scenario
    # Doc A has content, Doc B starts empty, they sync
    doc_a = pycrdt.Doc()
    doc_b = pycrdt.Doc()

    # Doc A creates initial content
    writer_a = DocumentWriter(doc_a)
    writer_a.replace_all_content("<paragraph>Hello from Doc A</paragraph>")

    # Sync A -> B (initial sync) - get full state from A
    full_state_a = doc_a.get_update()
    doc_b.apply_update(full_state_a)

    reader_b = DocumentReader(doc_b)
    print(f"Doc B after initial sync: {reader_b.to_xml()}")
    assert "Hello from Doc A" in reader_b.to_xml()
    print("✓ Initial sync works")

    # Now Doc A makes an incremental change
    old_state_a = doc_a.get_state()
    writer_a.append_block("<paragraph>Incremental update from Doc A</paragraph>")
    incremental_a = doc_a.get_update(old_state_a)

    print(f"\nIncremental update size: {len(incremental_a)} bytes")

    # Apply incremental update to Doc B
    doc_b.apply_update(incremental_a)

    reader_b = DocumentReader(doc_b)
    print(f"Doc B after incremental sync: {reader_b.to_xml()}")
    assert "Incremental update from Doc A" in reader_b.to_xml()
    print("✓ Incremental sync works")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def test_surgical_edits():
    """Test various surgical edit methods."""
    print("\n" + "=" * 60)
    print("TEST: Surgical Edit Methods")
    print("=" * 60)

    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)
    reader = DocumentReader(doc)

    # Test append_block
    print("\n--- append_block ---")
    old_state = doc.get_state()
    writer.append_block("<paragraph>First paragraph</paragraph>")
    update = doc.get_update(old_state)
    print(f"Append block update size: {len(update)} bytes")
    print(f"Content: {reader.to_xml()}")
    assert writer.get_block_count() == 1
    print("✓ append_block works")

    # Test insert_block_at
    print("\n--- insert_block_at ---")
    old_state = doc.get_state()
    writer.insert_block_at(0, "<paragraph>Inserted at beginning</paragraph>")
    update = doc.get_update(old_state)
    print(f"Insert block update size: {len(update)} bytes")
    print(f"Content: {reader.to_xml()}")
    assert writer.get_block_count() == 2
    assert "Inserted at beginning" in reader.to_xml()[:100]
    print("✓ insert_block_at works")

    # Test delete_block_at
    print("\n--- delete_block_at ---")
    old_state = doc.get_state()
    writer.delete_block_at(0)
    update = doc.get_update(old_state)
    print(f"Delete block update size: {len(update)} bytes")
    print(f"Content: {reader.to_xml()}")
    assert writer.get_block_count() == 1
    assert "Inserted at beginning" not in reader.to_xml()
    print("✓ delete_block_at works")

    # Test get_block_count
    print("\n--- get_block_count ---")
    writer.append_block("<paragraph>Second</paragraph>")
    writer.append_block("<paragraph>Third</paragraph>")
    assert writer.get_block_count() == 3
    print(f"Block count: {writer.get_block_count()}")
    print("✓ get_block_count works")

    print("\n" + "=" * 60)
    print("All surgical edit tests passed!")
    print("=" * 60)


def test_formatted_content():
    """Test that formatted content (marks) work with surgical edits."""
    print("\n" + "=" * 60)
    print("TEST: Formatted Content with Marks")
    print("=" * 60)

    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)
    reader = DocumentReader(doc)

    # Test adding formatted content
    print("\n--- Adding formatted paragraph ---")
    old_state = doc.get_state()
    writer.append_block(
        '<paragraph>Hello <strong>bold</strong> and <em>italic</em> text</paragraph>'
    )
    update = doc.get_update(old_state)
    print(f"Update size: {len(update)} bytes")
    print(f"Content: {reader.to_xml()}")
    print("✓ Formatted content added")

    # Test adding a heading
    print("\n--- Adding heading ---")
    old_state = doc.get_state()
    writer.append_block('<heading level="2">Section Header</heading>')
    update = doc.get_update(old_state)
    print(f"Update size: {len(update)} bytes")
    print(f"Content: {reader.to_xml()}")
    print("✓ Heading added")

    # Test adding a link
    print("\n--- Adding link ---")
    old_state = doc.get_state()
    writer.append_block(
        '<paragraph>Check out <a href="https://example.com">this link</a>!</paragraph>'
    )
    update = doc.get_update(old_state)
    print(f"Update size: {len(update)} bytes")
    print(f"Content: {reader.to_xml()}")
    print("✓ Link added")

    print("\n" + "=" * 60)
    print("All formatted content tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_incremental_vs_full_update()
    test_surgical_edits()
    test_formatted_content()

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
