"""Tests for the tag-page doc-id contract on the MCP side.

The MCP server's write tools all funnel through ``_assert_document_writable``
which rejects any doc id that addresses a synthesized read-only tag-page.
The reject path is two thin module-level helpers — those are what we test
here. Integration tests for the full write_document_tool path are out of
scope; the helpers are the contract.

The platform side has the parallel contract in
``app.services.documents.tag_pages``; the two prefixes must agree.
"""

from __future__ import annotations

import pytest

from neem.mcp.tools.hocuspocus import (
    TAG_PAGE_DOC_ID_PREFIX,
    is_tag_page_doc_id,
    tag_page_read_only_error,
)


class TestIsTagPageDocId:
    def test_matches_prefix(self) -> None:
        assert is_tag_page_doc_id("tag-page-pragma") is True
        assert is_tag_page_doc_id("tag-page-decision") is True
        assert is_tag_page_doc_id("tag-page-q2-planning") is True

    def test_rejects_unrelated(self) -> None:
        assert is_tag_page_doc_id("doc-abc") is False
        assert is_tag_page_doc_id("daily-note-2026-05-15") is False
        assert is_tag_page_doc_id("folder-xyz") is False

    def test_rejects_non_string(self) -> None:
        assert is_tag_page_doc_id(None) is False  # type: ignore[arg-type]
        assert is_tag_page_doc_id(123) is False  # type: ignore[arg-type]


class TestTagPageReadOnlyError:
    def test_returns_runtime_error(self) -> None:
        err = tag_page_read_only_error("tag-page-pragma")
        assert isinstance(err, RuntimeError)

    def test_message_names_the_doc_and_explains_read_only(self) -> None:
        err = tag_page_read_only_error("tag-page-pragma")
        msg = str(err).lower()
        assert "tag-page-pragma" in msg
        assert "read-only" in msg or "read only" in msg

    def test_message_directs_user_to_source_doc(self) -> None:
        # Don't just say "denied" — point them to where the edit can
        # actually land.
        err = tag_page_read_only_error("tag-page-pragma")
        msg = str(err).lower()
        assert "source" in msg or "open" in msg


class TestPrefixContract:
    def test_prefix_constant(self) -> None:
        # Lockstep with the platform's TAG_PAGE_PREFIX. If you change one,
        # change the other; this test is a tripwire.
        assert TAG_PAGE_DOC_ID_PREFIX == "tag-page-"


class TestRaisesAreCallable:
    def test_can_actually_raise(self) -> None:
        with pytest.raises(RuntimeError, match="tag-page"):
            raise tag_page_read_only_error("tag-page-pragma")
