"""Tests for inline tag marker preprocessing, extraction, and persistence.

Inline tags ({#tag}, {#tag:dur}) are applied as CRDT data-tags block
attributes after block IDs are assigned. These tests cover the converter
layer plus the Bug B persistence contract (see TestDataTagsPersistenceContract):
inline tags were silently dropped because they were applied with a bare
DocumentWriter(channel.doc) mutation OUTSIDE transact_document, so the
incremental diff was never computed or sent to the server.
"""

from __future__ import annotations

import pycrdt

from neem.hocuspocus.converters.inline_tags import (
    PendingTag,
    extract_xml_tags,
    map_tags_to_block_ids,
    postprocess_tags,
    preprocess_tags,
)
from neem.hocuspocus.converters.inline_valuations import preprocess_valuations
from neem.hocuspocus.document import DocumentReader, DocumentWriter


# ------------------------------------------------------------------
# preprocess_tags
# ------------------------------------------------------------------


class TestPreprocessTags:
    def test_no_markers(self):
        text = "Hello world\nSecond line"
        assert preprocess_tags(text) == text

    def test_single_tag(self):
        result = preprocess_tags("A decision {#decision}")
        assert "{#decision}" not in result
        assert "TAG:decision" in result

    def test_multiple_tags(self):
        result = preprocess_tags("Both {#decision}{#pragma}")
        assert "{#decision}" not in result
        assert "{#pragma}" not in result
        assert "TAG:decision,pragma" in result

    def test_tag_with_duration(self):
        result = preprocess_tags("A task {#todo:7d}")
        assert "{#todo:7d}" not in result
        assert "TAG:todo" in result
        assert "todo=7d" in result

    def test_code_fence_skipped(self):
        text = "Before\n```\ncode {#decision}\n```\nAfter {#pragma}"
        result = preprocess_tags(text)
        # Inside the fence the marker is preserved
        assert "{#decision}" in result.split("```")[1]
        # Outside the fence it is converted
        assert "TAG:pragma" in result

    def test_tag_after_valuation_placeholder(self):
        """Real pipeline order: preprocess_valuations runs first and leaves a
        VAL placeholder, then preprocess_tags must still pick up the trailing
        {#tag}. This is the combined `{!4,+2}{#decision}` form."""
        pre = preprocess_valuations("Combined {!4,+2}{#decision}")
        result = preprocess_tags(pre)
        assert "{#decision}" not in result
        assert "TAG:decision" in result
        # The valuation placeholder survived intact
        assert "VAL:4:2" in result


# ------------------------------------------------------------------
# extract_xml_tags (XML attribute path)
# ------------------------------------------------------------------


class TestExtractXmlTags:
    def test_extracts_data_tags_attribute(self):
        xml = '<paragraph data-tags=\'["decision","pragma"]\'>Content</paragraph>'
        clean, pending = extract_xml_tags(xml)
        assert len(pending) == 1
        assert pending[0].tags == ["decision", "pragma"]
        assert "data-tags" not in clean

    def test_no_tags_passthrough(self):
        xml = "<paragraph>Plain</paragraph>"
        clean, pending = extract_xml_tags(xml)
        assert pending == []
        assert clean == xml


# ------------------------------------------------------------------
# map_tags_to_block_ids
# ------------------------------------------------------------------


class TestMapTagsToBlockIds:
    def test_maps_index_to_block_id(self):
        pending = [PendingTag(block_index=0, tags=["decision"])]
        entries = map_tags_to_block_ids(pending, ["block-x"])
        assert entries == [{"block_id": "block-x", "tags": ["decision"]}]

    def test_out_of_range_index_skipped(self):
        pending = [PendingTag(block_index=5, tags=["decision"])]
        entries = map_tags_to_block_ids(pending, ["block-x"])
        assert entries == []

    def test_includes_expirations(self):
        pending = [PendingTag(block_index=0, tags=["todo"], expirations={"todo": "7d"})]
        entries = map_tags_to_block_ids(pending, ["block-x"])
        assert entries[0]["expirations"] == {"todo": "7d"}


# ------------------------------------------------------------------
# End-to-end: preprocess → markdown_to_xml → postprocess
# ------------------------------------------------------------------


class TestEndToEndTags:
    def test_markdown_with_tags(self):
        from neem.hocuspocus.converters import markdown_to_tiptap_xml

        md = "A tagged paragraph {#decision}\n\nPlain paragraph"
        pre = preprocess_tags(md)
        xml = markdown_to_tiptap_xml(pre)
        clean, pending = postprocess_tags(xml)

        assert "{#" not in clean
        assert len(pending) == 1
        assert pending[0].tags == ["decision"]


# ------------------------------------------------------------------
# Bug B: data-tags write is a real CRDT op (must go through transact_document)
# ------------------------------------------------------------------


class TestDataTagsPersistenceContract:
    """The original bug: inline tags were applied with a bare
    DocumentWriter(channel.doc).update_block_attributes(...) OUTSIDE
    transact_document, so the incremental diff was never computed/sent and the
    tags were discarded when the channel idled out. These tests pin the
    contract the fix relies on: a data-tags write is a real CRDT mutation that
    produces a non-empty update — therefore it MUST be applied inside
    transact_document (which computes get_update(old_state) and sends it)."""

    @staticmethod
    def _first_block_id(doc) -> str:
        reader = DocumentReader(doc)
        for child in reader.get_content_fragment().children:
            if hasattr(child, "attributes"):
                bid = child.attributes.get("data-block-id")
                if bid:
                    return bid
        raise AssertionError("no block id found")

    def test_update_block_attributes_sets_data_tags(self):
        doc = pycrdt.Doc()
        writer = DocumentWriter(doc)
        writer.replace_all_content("<paragraph>Hello</paragraph>")
        bid = self._first_block_id(doc)

        writer.update_block_attributes(bid, {"data-tags": '["decision"]'})

        _idx, elem = writer.find_block_by_id(bid)
        assert elem.attributes.get("data-tags") == '["decision"]'

    def test_data_tags_write_produces_nonempty_diff(self):
        doc = pycrdt.Doc()
        writer = DocumentWriter(doc)
        writer.replace_all_content("<paragraph>Hello</paragraph>")
        bid = self._first_block_id(doc)

        old_state = doc.get_state()
        writer.update_block_attributes(bid, {"data-tags": '["decision"]'})
        update = doc.get_update(old_state)

        # Non-empty: this is exactly the diff transact_document must send for
        # the tag to persist. Applying the write outside a transaction (the
        # bug) computes/sends nothing, silently dropping the tag.
        assert update
