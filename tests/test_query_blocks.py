"""Tests for MCP query-block helpers."""

from __future__ import annotations

import pytest

from neem.mcp.query_blocks import (
    build_query_block_xml,
    extract_query_block_network_data,
    infer_query_block_query_kind,
    normalize_query_block_attrs,
    normalize_query_result,
    profile_query_block_result,
    resolve_query_block_display,
)


def test_infer_query_block_query_kind_rejects_updates() -> None:
    with pytest.raises(ValueError, match="read-only"):
        infer_query_block_query_kind("DELETE WHERE { ?s ?p ?o }")


def test_build_query_block_xml_preserves_multiline_query() -> None:
    xml = build_query_block_xml({
        "query": 'SELECT ?s WHERE {\n  FILTER(?label = "garden")\n}\nLIMIT 5',
        "visualization": "network",
        "displayMode": "agent",
        "maxRows": 25,
    })
    assert "<queryBlock" in xml
    assert "&#10;" in xml
    assert "&quot;garden&quot;" in xml
    assert 'visualization="network"' in xml
    assert 'displayMode="agent"' in xml
    assert 'maxRows="25"' in xml


def test_normalize_query_block_attrs_clamps_rows_and_remaps_bar() -> None:
    attrs = normalize_query_block_attrs({
        "visualization": "bar",
        "maxRows": 9999,
        "collapsed": "true",
    })
    assert attrs["visualization"] == "vega"
    assert attrs["maxRows"] == 500
    assert attrs["collapsed"] is True


def test_profile_and_network_extraction_for_triple_bindings() -> None:
    raw = {
        "head": {"vars": ["subject", "predicate", "object"]},
        "results": {
            "bindings": [
                {
                    "subject": {"type": "uri", "value": "http://example.com/alice"},
                    "predicate": {"type": "uri", "value": "http://example.com/friend"},
                    "object": {"type": "uri", "value": "http://example.com/bob"},
                }
            ]
        },
    }
    result = normalize_query_result(raw, "select", 12, "application/sparql-results+json")
    profile = profile_query_block_result(result)
    network = extract_query_block_network_data(result, profile)
    resolved = resolve_query_block_display("network", result, display_mode="manual", profile=profile)

    assert profile["shape"] == "triples"
    assert len(network["nodes"]) == 2
    assert len(network["edges"]) == 1
    assert resolved["kind"] == "network"
    assert resolved["source"] == "override"


def test_normalize_query_result_for_serialized_construct() -> None:
    result = normalize_query_result(
        "<http://example.com/alice> <http://example.com/friend> <http://example.com/bob> .",
        "construct",
        18,
        "application/n-quads",
    )
    assert result["resultKind"] == "serialized"
    assert result["mediaType"] == "application/n-quads"
    assert "friend" in result["value"]
