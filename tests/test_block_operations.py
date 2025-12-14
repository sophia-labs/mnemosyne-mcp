"""Tests for block-level document operations.

These tests validate DocumentReader and DocumentWriter methods for
surgical block editing without requiring a running backend.
"""

import pytest
import pycrdt

from neem.hocuspocus.document import DocumentReader, DocumentWriter


@pytest.fixture
def empty_doc():
    """Create an empty Y.Doc."""
    return pycrdt.Doc()


@pytest.fixture
def populated_doc():
    """Create a Y.Doc with sample content for testing."""
    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)

    # Create a document with various block types and indentation
    writer.replace_all_content("""
<heading level="1">Test Document</heading>
<paragraph>First paragraph at indent 0.</paragraph>
<paragraph data-indent="1">Indented child paragraph.</paragraph>
<paragraph data-indent="2">Deeply indented paragraph.</paragraph>
<paragraph data-indent="1">Another indent-1 paragraph.</paragraph>
<paragraph>Back to indent 0.</paragraph>
<listItem listType="bullet">Bullet item</listItem>
<listItem listType="bullet" data-indent="1">Nested bullet</listItem>
<listItem listType="task" checked="true">Completed task</listItem>
<listItem listType="task">Uncompleted task</listItem>
<paragraph>Final paragraph.</paragraph>
""".strip())

    return doc


class TestDocumentReader:
    """Tests for DocumentReader block operations."""

    def test_find_block_by_id(self, populated_doc):
        """Test finding a block by its ID."""
        reader = DocumentReader(populated_doc)

        # Get the first block's ID
        block = reader.get_block_at(0)
        assert block is not None
        block_id = block.attributes.get("data-block-id")
        assert block_id is not None
        assert block_id.startswith("block-")

        # Find it by ID
        result = reader.find_block_by_id(block_id)
        assert result is not None
        index, elem = result
        assert index == 0
        assert elem.tag == "heading"

    def test_find_block_by_id_not_found(self, populated_doc):
        """Test that missing block returns None."""
        reader = DocumentReader(populated_doc)
        result = reader.find_block_by_id("block-nonexistent")
        assert result is None

    def test_get_block_at(self, populated_doc):
        """Test getting blocks by index."""
        reader = DocumentReader(populated_doc)

        # First block is heading
        block0 = reader.get_block_at(0)
        assert block0 is not None
        assert block0.tag == "heading"

        # Second block is paragraph
        block1 = reader.get_block_at(1)
        assert block1 is not None
        assert block1.tag == "paragraph"

        # Out of bounds returns None
        block_oob = reader.get_block_at(999)
        assert block_oob is None

    def test_get_block_info(self, populated_doc):
        """Test getting detailed block info."""
        reader = DocumentReader(populated_doc)

        # Get the heading's ID
        block = reader.get_block_at(0)
        block_id = block.attributes.get("data-block-id")

        info = reader.get_block_info(block_id)
        assert info is not None
        assert info["block_id"] == block_id
        assert info["index"] == 0
        assert info["type"] == "heading"
        assert "Test Document" in info["text_content"]
        assert info["context"]["total_blocks"] == 11
        assert info["context"]["prev_block_id"] is None  # First block
        assert info["context"]["next_block_id"] is not None  # Has next

    def test_get_block_info_with_context(self, populated_doc):
        """Test that block info includes prev/next context."""
        reader = DocumentReader(populated_doc)

        # Get the second block's ID
        block1 = reader.get_block_at(1)
        block1_id = block1.attributes.get("data-block-id")

        info = reader.get_block_info(block1_id)
        assert info["context"]["prev_block_id"] is not None
        assert info["context"]["next_block_id"] is not None

    def test_query_blocks_by_type(self, populated_doc):
        """Test querying blocks by type."""
        reader = DocumentReader(populated_doc)

        # Query for headings
        headings = reader.query_blocks(block_type="heading")
        assert len(headings) == 1
        assert headings[0]["type"] == "heading"

        # Query for listItems
        list_items = reader.query_blocks(block_type="listItem")
        assert len(list_items) == 4

    def test_query_blocks_by_indent(self, populated_doc):
        """Test querying blocks by indent level."""
        reader = DocumentReader(populated_doc)

        # Exact indent match
        indent1 = reader.query_blocks(indent=1)
        assert len(indent1) == 3  # Two paragraphs + one nested bullet

        # Indent greater than or equal
        indent_gte1 = reader.query_blocks(indent_gte=1)
        assert len(indent_gte1) >= 3

    def test_query_blocks_by_list_type(self, populated_doc):
        """Test querying by listType attribute."""
        reader = DocumentReader(populated_doc)

        bullets = reader.query_blocks(list_type="bullet")
        assert len(bullets) == 2

        tasks = reader.query_blocks(list_type="task")
        assert len(tasks) == 2

    def test_query_blocks_by_checked(self, populated_doc):
        """Test querying task items by checked state."""
        reader = DocumentReader(populated_doc)

        completed = reader.query_blocks(list_type="task", checked=True)
        assert len(completed) == 1

        uncompleted = reader.query_blocks(list_type="task", checked=False)
        assert len(uncompleted) == 1

    def test_query_blocks_by_text(self, populated_doc):
        """Test querying blocks by text content."""
        reader = DocumentReader(populated_doc)

        matches = reader.query_blocks(text_contains="indent")
        assert len(matches) >= 2  # Should find "Indented" paragraphs

    def test_query_blocks_limit(self, populated_doc):
        """Test query limit parameter."""
        reader = DocumentReader(populated_doc)

        # Without limit, gets all paragraphs
        all_paras = reader.query_blocks(block_type="paragraph")

        # With limit
        limited = reader.query_blocks(block_type="paragraph", limit=2)
        assert len(limited) == 2


class TestDocumentWriter:
    """Tests for DocumentWriter block operations."""

    def test_delete_block_by_id(self, populated_doc):
        """Test deleting a single block by ID."""
        writer = DocumentWriter(populated_doc)
        reader = DocumentReader(populated_doc)

        initial_count = reader.get_block_count()

        # Get first paragraph's ID (index 1)
        block = reader.get_block_at(1)
        block_id = block.attributes.get("data-block-id")

        deleted = writer.delete_block_by_id(block_id)
        assert deleted == [block_id]

        # Verify count decreased
        assert reader.get_block_count() == initial_count - 1

        # Verify block is gone
        assert reader.find_block_by_id(block_id) is None

    def test_delete_block_by_id_cascade(self, populated_doc):
        """Test cascade delete removes indent-children."""
        writer = DocumentWriter(populated_doc)
        reader = DocumentReader(populated_doc)

        # Get the first indent-0 paragraph (index 1) which has indent children
        block = reader.get_block_at(1)
        block_id = block.attributes.get("data-block-id")

        # The next 3 blocks are indented children
        initial_count = reader.get_block_count()

        deleted = writer.delete_block_by_id(block_id, cascade_children=True)

        # Should delete the parent + its 3 indent-children
        assert len(deleted) == 4
        assert reader.get_block_count() == initial_count - 4

    def test_delete_block_by_id_not_found(self, populated_doc):
        """Test that deleting nonexistent block raises."""
        writer = DocumentWriter(populated_doc)

        with pytest.raises(ValueError, match="Block not found"):
            writer.delete_block_by_id("block-nonexistent")

    def test_update_block_attributes(self, populated_doc):
        """Test updating block attributes."""
        writer = DocumentWriter(populated_doc)
        reader = DocumentReader(populated_doc)

        # Get an uncompleted task
        tasks = reader.query_blocks(list_type="task", checked=False)
        assert len(tasks) > 0
        task_id = tasks[0]["block_id"]

        # Mark it as checked
        writer.update_block_attributes(task_id, {"checked": True})

        # Verify update
        info = reader.get_block_info(task_id)
        assert info["attributes"].get("checked") == True

    def test_update_block_indent(self, populated_doc):
        """Test updating indent attribute."""
        writer = DocumentWriter(populated_doc)
        reader = DocumentReader(populated_doc)

        # Get first paragraph
        block = reader.get_block_at(1)
        block_id = block.attributes.get("data-block-id")

        # Increase indent
        writer.update_block_attributes(block_id, {"indent": 3})

        # Verify
        info = reader.get_block_info(block_id)
        assert info["attributes"].get("indent") == 3

    def test_replace_block_by_id(self, populated_doc):
        """Test replacing block content while preserving ID."""
        writer = DocumentWriter(populated_doc)
        reader = DocumentReader(populated_doc)

        # Get heading
        block = reader.get_block_at(0)
        block_id = block.attributes.get("data-block-id")

        # Replace with new content
        returned_id = writer.replace_block_by_id(
            block_id,
            '<heading level="2">New Title</heading>'
        )

        # ID should be preserved
        assert returned_id == block_id

        # Content should be updated
        info = reader.get_block_info(block_id)
        assert "New Title" in info["text_content"]
        # Level attribute should be updated
        assert info["attributes"].get("level") == "2"

    def test_insert_block_after_id(self, populated_doc):
        """Test inserting a block after another."""
        writer = DocumentWriter(populated_doc)
        reader = DocumentReader(populated_doc)

        initial_count = reader.get_block_count()

        # Get heading ID
        block = reader.get_block_at(0)
        heading_id = block.attributes.get("data-block-id")

        # Insert new block after heading
        new_id = writer.insert_block_after_id(
            heading_id,
            '<paragraph>Inserted paragraph</paragraph>'
        )

        assert new_id.startswith("block-")
        assert reader.get_block_count() == initial_count + 1

        # New block should be at index 1 (after heading)
        new_block = reader.get_block_at(1)
        assert new_block.attributes.get("data-block-id") == new_id

    def test_insert_block_before_id(self, populated_doc):
        """Test inserting a block before another."""
        writer = DocumentWriter(populated_doc)
        reader = DocumentReader(populated_doc)

        initial_count = reader.get_block_count()

        # Get first paragraph ID (index 1)
        block = reader.get_block_at(1)
        para_id = block.attributes.get("data-block-id")

        # Insert new block before it
        new_id = writer.insert_block_before_id(
            para_id,
            '<paragraph>Inserted before</paragraph>'
        )

        assert reader.get_block_count() == initial_count + 1

        # Original paragraph should now be at index 2
        result = reader.find_block_by_id(para_id)
        assert result is not None
        index, _ = result
        assert index == 2  # Pushed down by 1

    def test_insert_list_container_flattens(self, populated_doc):
        """Test that inserting list containers flattens them."""
        writer = DocumentWriter(populated_doc)
        reader = DocumentReader(populated_doc)

        initial_count = reader.get_block_count()

        # Get last block ID
        last_block = reader.get_block_at(initial_count - 1)
        last_id = last_block.attributes.get("data-block-id")

        # Insert a bulletList (should flatten to individual listItems)
        writer.insert_block_after_id(
            last_id,
            '<bulletList><listItem><paragraph>Item 1</paragraph></listItem><listItem><paragraph>Item 2</paragraph></listItem></bulletList>'
        )

        # Should add 2 blocks (flattened listItems)
        assert reader.get_block_count() == initial_count + 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_document(self, empty_doc):
        """Test operations on empty document."""
        reader = DocumentReader(empty_doc)

        assert reader.get_block_count() == 0
        assert reader.get_block_at(0) is None
        assert reader.find_block_by_id("block-any") is None
        assert reader.query_blocks() == []

    def test_block_id_generation(self, empty_doc):
        """Test that block IDs are automatically generated."""
        writer = DocumentWriter(empty_doc)
        reader = DocumentReader(empty_doc)

        # Add block without explicit ID
        writer.append_block('<paragraph>Test</paragraph>')

        block = reader.get_block_at(0)
        block_id = block.attributes.get("data-block-id")

        assert block_id is not None
        assert block_id.startswith("block-")
        assert len(block_id) == 14  # "block-" + 8 hex chars

    def test_preserve_block_id_on_replace(self, populated_doc):
        """Test that replace preserves the original block ID."""
        writer = DocumentWriter(populated_doc)
        reader = DocumentReader(populated_doc)

        # Get a block
        block = reader.get_block_at(0)
        original_id = block.attributes.get("data-block-id")

        # Replace it
        writer.replace_block_by_id(original_id, '<paragraph>Replaced</paragraph>')

        # Same ID should still exist
        result = reader.find_block_by_id(original_id)
        assert result is not None

    def test_query_combined_filters(self, populated_doc):
        """Test combining multiple query filters."""
        reader = DocumentReader(populated_doc)

        # Find indented bullets
        results = reader.query_blocks(
            block_type="listItem",
            list_type="bullet",
            indent=1
        )

        assert len(results) == 1
        assert results[0]["attributes"]["listType"] == "bullet"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
