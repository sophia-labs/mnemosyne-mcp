"""Tests for edit_block_text character-level editing operations.

These tests validate DocumentWriter.edit_block_text() and the supporting
DocumentReader.get_block_text_info() method for CRDT-native text editing
without requiring a running backend.
"""

import pytest
import pycrdt

from neem.hocuspocus.document import DocumentReader, DocumentWriter


@pytest.fixture
def plain_doc():
    """Create a Y.Doc with a single plain paragraph."""
    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)
    writer.replace_all_content(
        '<paragraph>Hello world</paragraph>'
    )
    return doc


@pytest.fixture
def formatted_doc():
    """Create a Y.Doc with a paragraph containing bold formatting."""
    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)
    writer.replace_all_content(
        '<paragraph>Hello <strong>bold</strong> world</paragraph>'
    )
    return doc


@pytest.fixture
def multi_block_doc():
    """Create a Y.Doc with multiple block types."""
    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)
    writer.replace_all_content("""
<heading level="1">Title</heading>
<paragraph>First paragraph.</paragraph>
<codeBlock language="python">def hello(): pass</codeBlock>
<paragraph>Last paragraph.</paragraph>
""".strip())
    return doc


def _get_block_id(doc, index):
    """Helper to get block ID at index."""
    reader = DocumentReader(doc)
    block = reader.get_block_at(index)
    return block.attributes.get("data-block-id")


class TestGetBlockTextInfo:
    """Tests for DocumentReader.get_block_text_info()."""

    def test_plain_text(self, plain_doc):
        """Test text info for plain paragraph."""
        reader = DocumentReader(plain_doc)
        block_id = _get_block_id(plain_doc, 0)
        info = reader.get_block_text_info(block_id)

        assert info is not None
        assert info["block_id"] == block_id
        assert info["text"] == "Hello world"
        assert info["length"] == 11
        assert info["has_inline_nodes"] is False
        assert len(info["runs"]) == 1
        assert info["runs"][0]["text"] == "Hello world"
        assert info["runs"][0]["offset"] == 0
        assert info["runs"][0]["length"] == 11
        assert info["runs"][0]["attrs"] is None

    def test_formatted_text(self, formatted_doc):
        """Test text info for paragraph with bold formatting."""
        reader = DocumentReader(formatted_doc)
        block_id = _get_block_id(formatted_doc, 0)
        info = reader.get_block_text_info(block_id)

        assert info is not None
        assert info["text"] == "Hello bold world"
        assert info["length"] == 16
        assert info["has_inline_nodes"] is False

        # Should have 3 runs: plain, bold, plain
        assert len(info["runs"]) == 3
        assert info["runs"][0]["text"] == "Hello "
        assert info["runs"][0]["attrs"] is None
        assert info["runs"][1]["text"] == "bold"
        assert "bold" in info["runs"][1]["attrs"]
        assert info["runs"][2]["text"] == " world"
        assert info["runs"][2]["attrs"] is None

    def test_not_found(self, plain_doc):
        """Test that missing block returns None."""
        reader = DocumentReader(plain_doc)
        info = reader.get_block_text_info("block-nonexistent")
        assert info is None

    def test_text_length_in_get_block_info(self, plain_doc):
        """Test that get_block_info includes text_length."""
        reader = DocumentReader(plain_doc)
        block_id = _get_block_id(plain_doc, 0)
        info = reader.get_block_info(block_id)

        assert info is not None
        assert "text_length" in info
        assert info["text_length"] == 11


class TestInsertOperations:
    """Tests for insert operations via edit_block_text."""

    def test_insert_at_start(self, plain_doc):
        """Test inserting text at the beginning."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 0, "text": "Hey! "}
        ])

        assert result["text"] == "Hey! Hello world"
        assert result["length"] == 16

    def test_insert_at_middle(self, plain_doc):
        """Test inserting text in the middle."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 5, "text": " beautiful"}
        ])

        assert result["text"] == "Hello beautiful world"

    def test_insert_at_end(self, plain_doc):
        """Test inserting text at the end."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 11, "text": "!"}
        ])

        assert result["text"] == "Hello world!"

    def test_insert_beyond_end_clamps(self, plain_doc):
        """Test that insert offset beyond text length appends at end."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 999, "text": "!"}
        ])

        assert result["text"] == "Hello world!"

    def test_insert_empty_text_skipped(self, plain_doc):
        """Test that empty insert text is a no-op."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 0, "text": ""}
        ])

        assert result["text"] == "Hello world"


class TestDeleteOperations:
    """Tests for delete operations via edit_block_text."""

    def test_delete_from_start(self, plain_doc):
        """Test deleting text from the beginning."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "delete", "offset": 0, "length": 6}
        ])

        assert result["text"] == "world"

    def test_delete_from_middle(self, plain_doc):
        """Test deleting text from the middle."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "delete", "offset": 5, "length": 1}
        ])

        assert result["text"] == "Helloworld"

    def test_delete_from_end(self, plain_doc):
        """Test deleting text from the end."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "delete", "offset": 5, "length": 6}
        ])

        assert result["text"] == "Hello"

    def test_delete_clamps_length(self, plain_doc):
        """Test that delete length is clamped to text end."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        # Delete from offset 5 with length 999 — should clamp to end
        result = writer.edit_block_text(block_id, [
            {"type": "delete", "offset": 5, "length": 999}
        ])

        assert result["text"] == "Hello"

    def test_delete_offset_beyond_end_errors(self, plain_doc):
        """Test that delete offset beyond text length raises error."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        with pytest.raises(ValueError, match="beyond text length"):
            writer.edit_block_text(block_id, [
                {"type": "delete", "offset": 999, "length": 1}
            ])

    def test_delete_zero_length_skipped(self, plain_doc):
        """Test that zero-length delete is a no-op."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "delete", "offset": 0, "length": 0}
        ])

        assert result["text"] == "Hello world"


class TestFormatInheritance:
    """Tests for formatting behavior during inserts."""

    def test_insert_inside_bold_inherits_format(self, formatted_doc):
        """Test that inserting inside a bold run inherits bold formatting."""
        writer = DocumentWriter(formatted_doc)
        block_id = _get_block_id(formatted_doc, 0)

        # "Hello bold world" — "bold" is at offsets 6-9
        # Insert "er" at offset 10 (right after "bold", preceding char is 'd' which is bold)
        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 10, "text": "er"}
        ])

        assert result["text"] == "Hello bolder world"
        # Verify the inserted text got bold formatting
        bold_runs = [r for r in result["runs"] if r.get("attrs") and "bold" in r["attrs"]]
        bold_text = "".join(r["text"] for r in bold_runs)
        assert "bolder" in bold_text

    def test_insert_with_explicit_attrs(self, plain_doc):
        """Test inserting with explicit formatting attrs."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 5, "text": " brave", "attrs": {"bold": {}}}
        ])

        assert result["text"] == "Hello brave world"
        # Verify explicit bold was applied
        bold_runs = [r for r in result["runs"] if r.get("attrs") and "bold" in r["attrs"]]
        assert len(bold_runs) > 0
        bold_text = "".join(r["text"] for r in bold_runs)
        assert "brave" in bold_text

    def test_insert_with_inherit_false(self, formatted_doc):
        """Test that inherit_format=False inserts plain text."""
        writer = DocumentWriter(formatted_doc)
        block_id = _get_block_id(formatted_doc, 0)

        # Insert inside bold run with inherit_format=False
        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 8, "text": "XX", "inherit_format": False}
        ])

        assert result["text"] == "Hello boXXld world"

    def test_insert_at_offset_zero_no_inheritance(self, formatted_doc):
        """Test that inserting at offset 0 doesn't inherit (no preceding char)."""
        writer = DocumentWriter(formatted_doc)
        block_id = _get_block_id(formatted_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 0, "text": "Hey "}
        ])

        assert result["text"] == "Hey Hello bold world"
        # First run should be plain (no inherited formatting)
        assert result["runs"][0]["attrs"] is None


class TestMultipleOperations:
    """Tests for multiple operations in a single call."""

    def test_insert_and_delete(self, plain_doc):
        """Test combining insert and delete in one call."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        # "Hello world" → delete " world" (offset 5, length 6), insert "!" at offset 5
        # Operations are sorted by offset descending, so both work correctly
        result = writer.edit_block_text(block_id, [
            {"type": "delete", "offset": 5, "length": 6},
            {"type": "insert", "offset": 5, "text": " there!"},
        ])

        assert result["text"] == "Hello there!"

    def test_multiple_inserts(self, plain_doc):
        """Test multiple inserts at different offsets."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        # "Hello world" (H=0..o=4, ' '=5, w=6..d=10)
        # All offsets relative to ORIGINAL text; applied descending.
        # Insert "!" at 11 (end), "beautiful " at 6 (before 'w'), "," at 5 (before ' ')
        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 5, "text": ","},
            {"type": "insert", "offset": 6, "text": "beautiful "},
            {"type": "insert", "offset": 11, "text": "!"},
        ])

        assert result["text"] == "Hello, beautiful world!"

    def test_operations_sorted_descending(self, plain_doc):
        """Test that operations are correctly sorted by offset descending."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        # Provide operations in ascending order — should still work
        # "Hello world" → applied descending: C at 11, B at 5, A at 0
        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 0, "text": "A"},
            {"type": "insert", "offset": 5, "text": "B"},
            {"type": "insert", "offset": 11, "text": "C"},
        ])

        # C at 11 (end): "Hello worldC"
        # B at 5 (between 'o' and ' '): "HelloB worldC"
        # A at 0 (start): "AHelloB worldC"
        assert result["text"] == "AHelloB worldC"


class TestCodeBlocks:
    """Tests for editing code blocks (no formatting)."""

    def test_insert_in_code_block(self, multi_block_doc):
        """Test inserting text in a code block."""
        writer = DocumentWriter(multi_block_doc)
        block_id = _get_block_id(multi_block_doc, 2)  # codeBlock

        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 10, "text": "\n    "}
        ])

        assert "def hello(" in result["text"]
        assert result["length"] > 17  # Original was 17

    def test_delete_in_code_block(self, multi_block_doc):
        """Test deleting text from a code block."""
        writer = DocumentWriter(multi_block_doc)
        block_id = _get_block_id(multi_block_doc, 2)  # codeBlock

        # "def hello(): pass" → delete " pass"
        result = writer.edit_block_text(block_id, [
            {"type": "delete", "offset": 12, "length": 5}
        ])

        assert result["text"] == "def hello():"


class TestErrorHandling:
    """Tests for error cases."""

    def test_block_not_found(self, plain_doc):
        """Test that editing a nonexistent block raises."""
        writer = DocumentWriter(plain_doc)

        with pytest.raises(ValueError, match="Block not found"):
            writer.edit_block_text("block-nonexistent", [
                {"type": "insert", "offset": 0, "text": "x"}
            ])

    def test_empty_operations(self, plain_doc):
        """Test that empty operations list raises."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        with pytest.raises(ValueError, match="cannot be empty"):
            writer.edit_block_text(block_id, [])

    def test_invalid_operation_type(self, plain_doc):
        """Test that invalid operation type raises."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        with pytest.raises(ValueError, match="type must be"):
            writer.edit_block_text(block_id, [
                {"type": "replace", "offset": 0, "text": "x"}
            ])

    def test_missing_offset(self, plain_doc):
        """Test that missing offset raises."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        with pytest.raises(ValueError, match="'offset' is required"):
            writer.edit_block_text(block_id, [
                {"type": "insert", "text": "x"}
            ])

    def test_missing_text_for_insert(self, plain_doc):
        """Test that missing text for insert raises."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        with pytest.raises(ValueError, match="'text' is required"):
            writer.edit_block_text(block_id, [
                {"type": "insert", "offset": 0}
            ])

    def test_missing_length_for_delete(self, plain_doc):
        """Test that missing length for delete raises."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        with pytest.raises(ValueError, match="'length' is required"):
            writer.edit_block_text(block_id, [
                {"type": "delete", "offset": 0}
            ])


class TestEdgeCases:
    """Additional edge case tests."""

    def test_delete_all_text(self, plain_doc):
        """Test deleting all text from a block."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        result = writer.edit_block_text(block_id, [
            {"type": "delete", "offset": 0, "length": 11}
        ])

        assert result["text"] == ""
        assert result["length"] == 0

    def test_insert_into_empty_block(self, plain_doc):
        """Test inserting into a block after deleting all text."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        # First delete all text
        writer.edit_block_text(block_id, [
            {"type": "delete", "offset": 0, "length": 11}
        ])

        # Then insert
        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 0, "text": "Fresh start"}
        ])

        assert result["text"] == "Fresh start"

    def test_sequential_edits(self, plain_doc):
        """Test multiple sequential edit_block_text calls."""
        writer = DocumentWriter(plain_doc)
        block_id = _get_block_id(plain_doc, 0)

        # First edit
        writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 5, "text": ","}
        ])

        # Second edit (offsets are relative to current state)
        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 6, "text": " beautiful"}
        ])

        assert result["text"] == "Hello, beautiful world"

    def test_heading_block_edit(self, multi_block_doc):
        """Test editing a heading block."""
        writer = DocumentWriter(multi_block_doc)
        block_id = _get_block_id(multi_block_doc, 0)  # heading

        result = writer.edit_block_text(block_id, [
            {"type": "insert", "offset": 5, "text": " Page"}
        ])

        assert result["text"] == "Title Page"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
