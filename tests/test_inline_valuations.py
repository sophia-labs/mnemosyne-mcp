"""Tests for inline valuation marker preprocessing and extraction."""

from __future__ import annotations

import pytest

from neem.hocuspocus.converters.inline_valuations import (
    PendingValuation,
    extract_xml_valuations,
    map_valuations_to_block_ids,
    postprocess_valuations,
    preprocess_valuations,
)


# ------------------------------------------------------------------
# preprocess_valuations
# ------------------------------------------------------------------


class TestPreprocessValuations:
    def test_no_markers(self):
        text = "Hello world\nSecond line"
        assert preprocess_valuations(text) == text

    def test_importance_only(self):
        text = "Important block {!3}"
        result = preprocess_valuations(text)
        assert "{!3}" not in result
        assert "\uE000VAL:3:\uE001" in result

    def test_valence_only_positive(self):
        text = "Good vibes {!,+2}"
        result = preprocess_valuations(text)
        assert "{!,+2}" not in result
        assert "\uE000VAL::2\uE001" in result

    def test_valence_only_negative(self):
        text = "Tension here {!,-4}"
        result = preprocess_valuations(text)
        assert "{!,-4}" not in result
        assert "\uE000VAL::-4\uE001" in result

    def test_both_importance_and_valence(self):
        text = "Key insight {!4,+2}"
        result = preprocess_valuations(text)
        assert "{!4,+2}" not in result
        assert "\uE000VAL:4:2\uE001" in result

    def test_both_with_negative_valence(self):
        text = "Problem area {!3,-5}"
        result = preprocess_valuations(text)
        assert "\uE000VAL:3:-5\uE001" in result

    def test_multiple_blocks(self):
        text = "First block {!5}\nSecond block\nThird block {!2,+1}"
        result = preprocess_valuations(text)
        assert "\uE000VAL:5:\uE001" in result
        assert "\uE000VAL:2:1\uE001" in result
        assert "Second block" in result

    def test_code_fence_skipped(self):
        text = "Before\n```\ncode block {!3}\n```\nAfter {!2}"
        result = preprocess_valuations(text)
        # Marker inside code fence should be preserved as-is
        assert "{!3}" in result
        # Marker outside code fence should be replaced
        assert "{!3}" in result.split("```")[1]  # still in code block
        assert "\uE000VAL:2:\uE001" in result

    def test_marker_stripped_from_content(self):
        text = "Content here {!4}"
        result = preprocess_valuations(text)
        assert "Content here" in result
        assert "{!4}" not in result

    def test_importance_zero(self):
        """importance=0 is valid (active forgetting)."""
        text = "Forget this {!0}"
        result = preprocess_valuations(text)
        assert "\uE000VAL:0:\uE001" in result

    def test_importance_out_of_range(self):
        """importance=9 is out of range, marker left as-is."""
        text = "Bad range {!9}"
        result = preprocess_valuations(text)
        assert "{!9}" in result

    def test_valence_out_of_range(self):
        """valence=-9 is out of range, marker left as-is."""
        text = "Bad range {!,-9}"
        result = preprocess_valuations(text)
        assert "{!,-9}" in result

    def test_trailing_whitespace(self):
        text = "Content {!3}  "
        result = preprocess_valuations(text)
        assert "\uE000VAL:3:\uE001" in result

    def test_heading_with_marker(self):
        text = "## Important heading {!5}"
        result = preprocess_valuations(text)
        assert "\uE000VAL:5:\uE001" in result
        assert "## Important heading" in result

    def test_list_item_with_marker(self):
        text = "- List item {!3,+1}"
        result = preprocess_valuations(text)
        assert "\uE000VAL:3:1\uE001" in result

    def test_empty_marker_ignored(self):
        """Bare {!} should not match (no importance or valence)."""
        text = "No values {!}"
        result = preprocess_valuations(text)
        # The regex requires at least one capture, bare {!} has neither
        assert text == result

    def test_marker_not_at_end(self):
        """Markers mid-line should not match."""
        text = "Some {!3} text here"
        result = preprocess_valuations(text)
        assert result == text


# ------------------------------------------------------------------
# postprocess_valuations
# ------------------------------------------------------------------


class TestPostprocessValuations:
    def test_no_placeholders(self):
        xml = "<paragraph>Hello world</paragraph>"
        clean, pending = postprocess_valuations(xml)
        assert clean == xml
        assert pending == []

    def test_single_placeholder(self):
        xml = "<paragraph>Content \uE000VAL:3:\uE001</paragraph>"
        clean, pending = postprocess_valuations(xml)
        assert "\uE000" not in clean
        assert "\uE001" not in clean
        assert len(pending) == 1
        assert pending[0] == PendingValuation(block_index=0, importance=3, valence=None)

    def test_valence_placeholder(self):
        xml = "<paragraph>Content \uE000VAL::+2\uE001</paragraph>"
        clean, pending = postprocess_valuations(xml)
        assert len(pending) == 1
        assert pending[0] == PendingValuation(block_index=0, importance=None, valence=2)

    def test_multiple_blocks_different_indices(self):
        xml = (
            "<paragraph>First \uE000VAL:5:\uE001</paragraph>"
            "<paragraph>Second</paragraph>"
            "<heading level=\"2\">Third \uE000VAL:2:1\uE001</heading>"
        )
        clean, pending = postprocess_valuations(xml)
        assert len(pending) == 2
        assert pending[0] == PendingValuation(block_index=0, importance=5, valence=None)
        assert pending[1] == PendingValuation(block_index=2, importance=2, valence=1)

    def test_placeholder_in_nested_element(self):
        """Placeholder inside a <strong> inside a <paragraph>."""
        xml = "<paragraph><strong>Bold text \uE000VAL:4:\uE001</strong></paragraph>"
        clean, pending = postprocess_valuations(xml)
        assert len(pending) == 1
        assert pending[0].block_index == 0
        assert pending[0].importance == 4

    def test_malformed_xml_fallback(self):
        """Malformed XML should strip placeholders and return empty list."""
        xml = "<paragraph>Content \uE000VAL:3:\uE001<unclosed"
        clean, pending = postprocess_valuations(xml)
        assert "\uE000" not in clean
        assert pending == []

    def test_listitem_block(self):
        xml = '<listItem listType="bullet"><paragraph>Item \uE000VAL:3:2\uE001</paragraph></listItem>'
        clean, pending = postprocess_valuations(xml)
        assert len(pending) == 1
        assert pending[0] == PendingValuation(block_index=0, importance=3, valence=2)


# ------------------------------------------------------------------
# extract_xml_valuations
# ------------------------------------------------------------------


class TestExtractXmlValuations:
    def test_no_val_attrs(self):
        xml = "<paragraph>Content</paragraph>"
        clean, pending = extract_xml_valuations(xml)
        assert clean == xml
        assert pending == []

    def test_importance_attr(self):
        xml = '<paragraph data-val-importance="4">Content</paragraph>'
        clean, pending = extract_xml_valuations(xml)
        assert "data-val-importance" not in clean
        assert "Content" in clean
        assert len(pending) == 1
        assert pending[0] == PendingValuation(block_index=0, importance=4, valence=None)

    def test_valence_attr(self):
        xml = '<paragraph data-val-valence="-3">Content</paragraph>'
        clean, pending = extract_xml_valuations(xml)
        assert "data-val-valence" not in clean
        assert len(pending) == 1
        assert pending[0] == PendingValuation(block_index=0, importance=None, valence=-3)

    def test_both_attrs(self):
        xml = '<paragraph data-val-importance="5" data-val-valence="3">Key</paragraph>'
        clean, pending = extract_xml_valuations(xml)
        assert "data-val-importance" not in clean
        assert "data-val-valence" not in clean
        assert len(pending) == 1
        assert pending[0] == PendingValuation(block_index=0, importance=5, valence=3)

    def test_multiple_blocks(self):
        xml = (
            '<paragraph data-val-importance="3">First</paragraph>'
            '<paragraph>Second</paragraph>'
            '<heading level="2" data-val-valence="-2">Third</heading>'
        )
        clean, pending = extract_xml_valuations(xml)
        assert len(pending) == 2
        assert pending[0] == PendingValuation(block_index=0, importance=3, valence=None)
        assert pending[1] == PendingValuation(block_index=2, importance=None, valence=-2)

    def test_out_of_range_skipped(self):
        xml = '<paragraph data-val-importance="9">Content</paragraph>'
        clean, pending = extract_xml_valuations(xml)
        assert "data-val-importance" not in clean
        assert pending == []


# ------------------------------------------------------------------
# map_valuations_to_block_ids
# ------------------------------------------------------------------


class TestMapValuationsToBlockIds:
    def test_single_mapping(self):
        pending = [PendingValuation(block_index=0, importance=3, valence=None)]
        block_ids = ["block-abc123"]
        result = map_valuations_to_block_ids(pending, block_ids, "doc-1")
        assert result == [{
            "document_id": "doc-1",
            "block_id": "block-abc123",
            "importance": 3,
        }]

    def test_multiple_mappings(self):
        pending = [
            PendingValuation(block_index=0, importance=5, valence=None),
            PendingValuation(block_index=2, importance=None, valence=-3),
        ]
        block_ids = ["block-a", "block-b", "block-c"]
        result = map_valuations_to_block_ids(pending, block_ids, "doc-1")
        assert len(result) == 2
        assert result[0]["block_id"] == "block-a"
        assert result[0]["importance"] == 5
        assert "valence" not in result[0]
        assert result[1]["block_id"] == "block-c"
        assert result[1]["valence"] == -3

    def test_out_of_range_skipped(self):
        pending = [PendingValuation(block_index=5, importance=3, valence=None)]
        block_ids = ["block-a", "block-b"]
        result = map_valuations_to_block_ids(pending, block_ids, "doc-1")
        assert result == []

    def test_empty_pending(self):
        result = map_valuations_to_block_ids([], ["block-a"], "doc-1")
        assert result == []

    def test_both_importance_and_valence(self):
        pending = [PendingValuation(block_index=0, importance=4, valence=2)]
        block_ids = ["block-x"]
        result = map_valuations_to_block_ids(pending, block_ids, "doc-1")
        assert result[0]["importance"] == 4
        assert result[0]["valence"] == 2


# ------------------------------------------------------------------
# End-to-end: preprocess → markdown_to_xml → postprocess
# ------------------------------------------------------------------


class TestEndToEnd:
    def test_markdown_with_valuations(self):
        """Full pipeline: markdown content with markers → clean XML + valuations."""
        from neem.hocuspocus.converters import markdown_to_tiptap_xml

        md = "## Important heading {!5}\n\nA paragraph with insight {!3,+2}\n\nPlain paragraph"
        preprocessed = preprocess_valuations(md)
        xml = markdown_to_tiptap_xml(preprocessed)
        clean, pending = postprocess_valuations(xml)

        # Markers should be stripped
        assert "{!" not in clean
        assert "\uE000" not in clean

        # Should have 2 valuations
        assert len(pending) == 2
        assert pending[0].importance == 5  # heading
        assert pending[1].importance == 3  # paragraph
        assert pending[1].valence == 2

    def test_plain_text_with_valuations(self):
        """Plain text (not markdown) with markers."""
        import html as html_mod

        text = "Simple content {!4}"
        preprocessed = preprocess_valuations(text)
        # Simulate _ensure_xml for plain text: wrap in paragraph
        xml = f"<paragraph>{html_mod.escape(preprocessed)}</paragraph>"
        clean, pending = postprocess_valuations(xml)

        assert "{!" not in clean
        assert len(pending) == 1
        assert pending[0] == PendingValuation(block_index=0, importance=4, valence=None)

    def test_no_markers_passthrough(self):
        """Content without markers passes through unchanged."""
        from neem.hocuspocus.converters import markdown_to_tiptap_xml

        md = "## Heading\n\nA paragraph"
        preprocessed = preprocess_valuations(md)
        xml = markdown_to_tiptap_xml(preprocessed)
        clean, pending = postprocess_valuations(xml)

        assert pending == []
        # XML should be identical to what markdown_to_tiptap_xml produces
        assert clean == markdown_to_tiptap_xml(md)
