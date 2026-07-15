"""Tests for the ID normalization helpers used by the MCP tool layer.

Audit refs:
- P1 #5: Drop `block-` prefix and bare-echo overhead
- P1 #13: Strip `doc-` prefix and accept bare UUIDs everywhere
"""

from __future__ import annotations

from neem.mcp.tools._id_normalize import (
    bare_block_id,
    bare_block_ids,
    bare_ids_in_result,
    normalize_block_id_for_lookup,
    normalize_document_id_for_lookup,
    prefixed_block_id,
)


# Block ID — output direction


def test_bare_block_id_strips_prefix() -> None:
    assert bare_block_id("block-abc123") == "abc123"


def test_bare_block_id_passes_bare_unchanged() -> None:
    assert bare_block_id("abc123") == "abc123"


def test_bare_block_id_handles_none() -> None:
    assert bare_block_id(None) is None


def test_bare_block_id_handles_non_string() -> None:
    assert bare_block_id(42) == 42


def test_bare_block_ids_handles_list() -> None:
    out = bare_block_ids(["block-a", "block-b", "c"])
    assert out == ["a", "b", "c"]


def test_bare_block_ids_handles_none() -> None:
    assert bare_block_ids(None) == []


# Block ID — input direction


def test_prefixed_block_id_adds_prefix() -> None:
    assert prefixed_block_id("abc123") == "block-abc123"


def test_prefixed_block_id_idempotent_on_prefixed() -> None:
    assert prefixed_block_id("block-abc123") == "block-abc123"


def test_normalize_block_id_for_lookup_accepts_both_forms() -> None:
    assert normalize_block_id_for_lookup("abc") == "block-abc"
    assert normalize_block_id_for_lookup("block-abc") == "block-abc"


# Document ID — no-op pass-through (P1 #13 regression, root-caused 2026-07-15)
#
# This used to strip a leading `doc-` prefix whenever the tail looked
# UUID-shaped, on the theory that the bare form was "the canonical form
# most likely to match a workspace entry." That was false: documents
# created via the web UI are commonly keyed by the full `doc-<uuid>` form,
# so the strip corrupted an already-correct ID into one that matched
# nothing, and every such document silently became unreadable via
# read_document/read_blocks/document_digest/etc. Tolerance for callers
# supplying the "wrong" form now lives at the actual lookup point (see
# test_workspace_reader_compat.py), where the real keyspace is known.


def test_normalize_document_id_leaves_doc_prefixed_uuid_unchanged() -> None:
    uuid = "550e8400-e29b-41d4-a716-446655440000"
    assert normalize_document_id_for_lookup(f"doc-{uuid}") == f"doc-{uuid}"


def test_normalize_document_id_passes_slug_unchanged() -> None:
    assert normalize_document_id_for_lookup("garden-design-antinomies") == "garden-design-antinomies"
    assert normalize_document_id_for_lookup("agent-delta") == "agent-delta"


def test_normalize_document_id_passes_bare_uuid_unchanged() -> None:
    uuid = "550e8400-e29b-41d4-a716-446655440000"
    assert normalize_document_id_for_lookup(uuid) == uuid


def test_normalize_document_id_keeps_doc_prefix_on_non_uuid() -> None:
    assert normalize_document_id_for_lookup("doc-garden") == "doc-garden"


# Recursive walker over result dicts


def test_bare_ids_in_result_strips_block_id_at_top_level() -> None:
    out = bare_ids_in_result({"block_id": "block-abc"})
    assert out == {"block_id": "abc"}


def test_bare_ids_in_result_strips_block_ids_list() -> None:
    out = bare_ids_in_result({"block_ids": ["block-a", "block-b"]})
    assert out == {"block_ids": ["a", "b"]}


def test_bare_ids_in_result_nested_in_lists() -> None:
    out = bare_ids_in_result({
        "results": [
            {"block_id": "block-a", "success": True},
            {"block_id": "block-b", "success": True},
        ],
    })
    assert out["results"][0]["block_id"] == "a"
    assert out["results"][1]["block_id"] == "b"


def test_bare_ids_in_result_nested_in_dicts() -> None:
    out = bare_ids_in_result({
        "block": {"block_id": "block-xyz", "type": "paragraph"},
    })
    assert out["block"]["block_id"] == "xyz"


def test_bare_ids_in_result_preserves_other_fields() -> None:
    out = bare_ids_in_result({
        "block_id": "block-a",
        "graph_id": "default",
        "document_id": "agent-delta",
        "content": "hello",
    })
    assert out["graph_id"] == "default"
    assert out["document_id"] == "agent-delta"
    assert out["content"] == "hello"


def test_bare_ids_in_result_handles_empty_input() -> None:
    assert bare_ids_in_result({}) == {}
    assert bare_ids_in_result([]) == []
    assert bare_ids_in_result(None) is None


# Wire-shape coverage — epsilon's #3 finding


def test_bare_ids_in_result_strips_source_and_target_block_id() -> None:
    """get_wires/traverse_wires emit sourceBlockId/targetBlockId; walker must
    strip those just like block_id."""
    out = bare_ids_in_result({
        "sourceBlockId": "block-aaa",
        "targetBlockId": "block-bbb",
        "predicate": "supports",
    })
    assert out["sourceBlockId"] == "aaa"
    assert out["targetBlockId"] == "bbb"
    assert out["predicate"] == "supports"


def test_bare_ids_in_result_strips_snake_case_wire_block_ids() -> None:
    out = bare_ids_in_result({
        "source_block_id": "block-aaa",
        "target_block_id": "block-bbb",
    })
    assert out["source_block_id"] == "aaa"
    assert out["target_block_id"] == "bbb"


def test_bare_ids_in_result_strips_by_block_dict_keys() -> None:
    """read_document().wires.by_block has block IDs as dict KEYS — these
    require key rewriting since the walker can't recurse through them."""
    out = bare_ids_in_result({
        "wires": {
            "outgoing": 2,
            "incoming": 1,
            "by_block": {"block-abc": 2, "block-def": 1},
        },
    })
    assert out["wires"]["by_block"] == {"abc": 2, "def": 1}


def test_bare_ids_in_result_strips_nested_wire_blocks() -> None:
    out = bare_ids_in_result({
        "wires": [
            {"id": "w-1", "sourceBlockId": "block-x", "targetBlockId": "block-y"},
            {"id": "w-2", "sourceBlockId": "block-z"},
        ],
    })
    assert out["wires"][0]["sourceBlockId"] == "x"
    assert out["wires"][0]["targetBlockId"] == "y"
    assert out["wires"][1]["sourceBlockId"] == "z"
