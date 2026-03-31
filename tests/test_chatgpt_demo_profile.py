"""Tests for the chatgpt_demo MCP profile."""

from __future__ import annotations

import pytest

from neem.mcp.server.standalone_server import create_standalone_mcp_server


@pytest.fixture
def _demo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_PROFILE", "chatgpt_demo")
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "demo_noauth")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_DEMO_GRAPH_ID", "demo-graph")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_DEMO_TOKEN", "demo-token")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_DEMO_USER_ID", "demo-user")


def test_chatgpt_demo_profile_exposes_only_expected_tools(_demo_env: None) -> None:
    server = create_standalone_mcp_server()

    tools = server._tool_manager._tools
    assert set(tools.keys()) == {
        "search_documents",
        "search_blocks",
        "read_document",
        "document_digest",
    }


def test_chatgpt_demo_profile_hides_graph_id_and_marks_tools_read_only(_demo_env: None) -> None:
    server = create_standalone_mcp_server()
    tools = server._tool_manager._tools

    search_documents = tools["search_documents"]
    assert "graph_id" not in search_documents.parameters["properties"]
    assert set(search_documents.parameters["required"]) == {"query"}
    assert search_documents.annotations is not None
    assert search_documents.annotations.readOnlyHint is True
    assert search_documents.meta == {
        "securitySchemes": [{"type": "noauth"}],
        "_meta": {"securitySchemes": [{"type": "noauth"}]},
    }

    search_blocks = tools["search_blocks"]
    assert "graph_id" not in search_blocks.parameters["properties"]
    assert "document_id" in search_blocks.parameters["properties"]
    assert set(search_blocks.parameters["required"]) == {"query"}

    read_document = tools["read_document"]
    assert "graph_id" not in read_document.parameters["properties"]
    assert set(read_document.parameters["required"]) == {"document_id"}

    document_digest = tools["document_digest"]
    assert "graph_id" not in document_digest.parameters["properties"]
    assert set(document_digest.parameters["required"]) == {"document_id"}


def test_chatgpt_demo_profile_requires_demo_graph_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_PROFILE", "chatgpt_demo")
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "demo_noauth")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_DEMO_TOKEN", "demo-token")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_DEMO_USER_ID", "demo-user")
    monkeypatch.delenv("MNEMOSYNE_CHATGPT_DEMO_GRAPH_ID", raising=False)

    with pytest.raises(RuntimeError, match="MNEMOSYNE_CHATGPT_DEMO_GRAPH_ID"):
        create_standalone_mcp_server()
