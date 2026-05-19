"""Tests for the server-side tag-chip sync helper.

`_sync_tag_chips_to_block` ensures that, for each (tag, date) pair the
caller cares about, the source block's inline content carries a
`<mn-tag-chip>` atom matching the spec. Used by
``apply_tags_with_side_effects`` so MCP-written / value()-applied tags
appear as chips inline on the source block immediately, without waiting
for the frontend sync plugin to react.
"""

from __future__ import annotations

import pycrdt
import pytest

from neem.hocuspocus.document import DocumentWriter
from neem.mcp.tools.hocuspocus import _sync_tag_chips_to_block


def _block_with_id(doc: pycrdt.Doc, block_id: str) -> pycrdt.XmlElement:
    """Create a block in a fresh doc and return its XmlElement."""
    writer = DocumentWriter(doc)
    writer.replace_all_content(
        f'<paragraph data-block-id="{block_id}">hello</paragraph>'
    )
    result = writer.find_block_by_id(block_id)
    assert result is not None, "block not found after creation"
    _idx, elem = result
    return elem


def _attr_or_empty(attrs, key: str) -> str:
    """pycrdt's XmlAttributesView.get() takes only key, no default."""
    try:
        return attrs[key] or ""
    except (KeyError, TypeError):
        return ""


def _chip_children(elem: pycrdt.XmlElement) -> list[dict]:
    """Return list of {name, date} for chip children of a block."""
    out: list[dict] = []
    for child in elem.children:
        if not isinstance(child, pycrdt.XmlElement):
            continue
        if child.tag != "mn-tag-chip":
            continue
        out.append({
            "name": _attr_or_empty(child.attributes, "name"),
            "date": _attr_or_empty(child.attributes, "date"),
        })
    return out


class TestSyncTagChips:
    def test_appends_chip_when_absent(self) -> None:
        doc = pycrdt.Doc()
        elem = _block_with_id(doc, "b1")

        with doc.transaction():
            count = _sync_tag_chips_to_block(elem, {"event": "2026-05-15"})

        assert count == 1
        chips = _chip_children(elem)
        assert chips == [{"name": "event", "date": "2026-05-15"}]

    def test_appends_chip_with_no_date(self) -> None:
        doc = pycrdt.Doc()
        elem = _block_with_id(doc, "b1")

        with doc.transaction():
            count = _sync_tag_chips_to_block(elem, {"decision": None})

        assert count == 1
        chips = _chip_children(elem)
        # The chip exists but has no date attribute (empty string when
        # read back via attributes.get default).
        assert chips == [{"name": "decision", "date": ""}]

    def test_idempotent_on_repeat(self) -> None:
        doc = pycrdt.Doc()
        elem = _block_with_id(doc, "b1")

        with doc.transaction():
            _sync_tag_chips_to_block(elem, {"event": "2026-05-15"})
        with doc.transaction():
            count = _sync_tag_chips_to_block(elem, {"event": "2026-05-15"})

        assert count == 0
        chips = _chip_children(elem)
        assert chips == [{"name": "event", "date": "2026-05-15"}]

    def test_updates_existing_chip_date(self) -> None:
        doc = pycrdt.Doc()
        elem = _block_with_id(doc, "b1")

        with doc.transaction():
            _sync_tag_chips_to_block(elem, {"event": "2026-05-15"})
        with doc.transaction():
            count = _sync_tag_chips_to_block(elem, {"event": "2026-05-20"})

        assert count == 1
        chips = _chip_children(elem)
        # Single chip — updated in place to the new date.
        assert chips == [{"name": "event", "date": "2026-05-20"}]

    def test_multiple_tags(self) -> None:
        doc = pycrdt.Doc()
        elem = _block_with_id(doc, "b1")

        with doc.transaction():
            count = _sync_tag_chips_to_block(elem, {
                "event": "2026-05-15",
                "decision": None,
                "todo": "2026-05-20",
            })

        assert count == 3
        chips = _chip_children(elem)
        names = {c["name"] for c in chips}
        assert names == {"event", "decision", "todo"}

    def test_mixed_existing_and_new(self) -> None:
        doc = pycrdt.Doc()
        elem = _block_with_id(doc, "b1")

        with doc.transaction():
            _sync_tag_chips_to_block(elem, {"event": "2026-05-15"})
        with doc.transaction():
            count = _sync_tag_chips_to_block(elem, {
                "event": "2026-05-15",  # already present, same date — no-op
                "todo": "2026-05-20",   # new
            })

        assert count == 1
        chips = _chip_children(elem)
        names = {c["name"] for c in chips}
        assert names == {"event", "todo"}

    def test_blank_tag_skipped(self) -> None:
        doc = pycrdt.Doc()
        elem = _block_with_id(doc, "b1")

        with doc.transaction():
            count = _sync_tag_chips_to_block(elem, {"": "2026-05-15"})

        assert count == 0
        assert _chip_children(elem) == []

    def test_does_not_touch_non_chip_children(self) -> None:
        """Existing text content should be preserved; chips append after it."""
        doc = pycrdt.Doc()
        elem = _block_with_id(doc, "b1")
        # Block already has its initial "hello" text node.

        with doc.transaction():
            _sync_tag_chips_to_block(elem, {"event": "2026-05-15"})

        # Walk children: text node first, chip second.
        children_summary = []
        for child in elem.children:
            if isinstance(child, pycrdt.XmlElement):
                children_summary.append(("element", child.tag))
            elif isinstance(child, pycrdt.XmlText):
                children_summary.append(("text", str(child)))
        # Initial text should still be present.
        assert any(t == "text" for t, _ in children_summary)
        # Chip should be among the elements.
        assert ("element", "mn-tag-chip") in children_summary
