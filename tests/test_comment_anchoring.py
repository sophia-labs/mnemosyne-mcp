"""Tests for automatic comment anchoring primitives."""

from __future__ import annotations

import pytest
import pycrdt

from neem.hocuspocus.document import DocumentReader, DocumentWriter


def _get_block_id(doc: pycrdt.Doc, index: int = 0) -> str:
    reader = DocumentReader(doc)
    block = reader.get_block_at(index)
    assert block is not None
    return str(block.attributes.get("data-block-id"))


def test_add_comment_mark_applies_mark_without_dropping_existing_marks() -> None:
    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)
    writer.replace_all_content("<paragraph>Hello <strong>bold</strong> world</paragraph>")
    block_id = _get_block_id(doc)

    # "bold" starts at code-point offset 6.
    writer.add_comment_mark(block_id=block_id, start_offset=6, length_cp=4, comment_id="c-1")

    xml_out = DocumentReader(doc).to_xml()
    assert 'data-comment-id="c-1"' in xml_out
    assert "<strong>" in xml_out  # original formatting should remain
    assert "bold" in xml_out


def test_add_comment_mark_rejects_ranges_that_cross_inline_nodes() -> None:
    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)
    writer.replace_all_content(
        '<paragraph>Hi <footnote data-footnote-content="note"/> there</paragraph>',
    )
    block_id = _get_block_id(doc)

    with pytest.raises(ValueError, match="intersects inline element"):
        writer.add_comment_mark(
            block_id=block_id,
            start_offset=0,
            length_cp=100,
            comment_id="c-inline",
        )


def test_set_comment_preserves_anchor_metadata_on_update() -> None:
    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)

    writer.set_comment(
        comment_id="c-anchor",
        text="first",
        author="Sophia",
        author_id="mcp-agent",
        resolved=False,
        quoted_text="quoted",
        block_id="block-123",
        document_position=7,
    )
    writer.set_comment(
        comment_id="c-anchor",
        text="second",
        author="Sophia",
        author_id="mcp-agent",
        resolved=True,
    )

    comments = writer.get_all_comments()
    updated = comments["c-anchor"]
    assert updated["text"] == "second"
    assert updated["resolved"] is True
    assert updated["quotedText"] == "quoted"
    assert updated["blockId"] == "block-123"
    assert updated["documentPosition"] == 7

