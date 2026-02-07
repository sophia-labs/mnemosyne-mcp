"""Tests for TipTap XML attribute normalization on read."""

from __future__ import annotations

import pycrdt

from neem.hocuspocus.document import DocumentReader, DocumentWriter


def _roundtrip(xml: str) -> str:
    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)
    writer.replace_all_content(xml)
    reader = DocumentReader(doc)
    return reader.to_xml()


def test_footnote_attribute_serializes_to_data_attribute() -> None:
    xml = '<paragraph>Hello<footnote data-footnote-content="note"/></paragraph>'
    out = _roundtrip(xml)
    assert 'data-footnote-content="note"' in out
    assert 'footnote content="' not in out


def test_comment_mark_attribute_serializes_to_data_attribute() -> None:
    xml = '<paragraph>Hi <commentMark data-comment-id="c-1">there</commentMark></paragraph>'
    out = _roundtrip(xml)
    assert 'data-comment-id="c-1"' in out
    assert "commentId=" not in out


def test_block_indent_serializes_back_to_data_indent() -> None:
    xml = '<paragraph data-indent="2">Indented</paragraph>'
    out = _roundtrip(xml)
    assert 'data-indent="2"' in out
    assert "<paragraph" in out
