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
        "get_workspace",
        "read_blocks",
        "get_block",
        "query_blocks",
        "search_documents",
        "search_blocks",
        "read_document",
        "document_digest",
        "check_document",
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

    get_workspace = tools["get_workspace"]
    assert "graph_id" not in get_workspace.parameters["properties"]

    read_blocks = tools["read_blocks"]
    assert "graph_id" not in read_blocks.parameters["properties"]
    assert set(read_blocks.parameters["required"]) == {"document_id"}

    get_block = tools["get_block"]
    assert "graph_id" not in get_block.parameters["properties"]
    assert set(get_block.parameters["required"]) == {"document_id", "block_id"}

    query_blocks = tools["query_blocks"]
    assert "graph_id" not in query_blocks.parameters["properties"]
    assert set(query_blocks.parameters["required"]) == {"document_id"}


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


def test_chatgpt_demo_profile_uses_oauth2_security_in_chatgpt_oauth_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_PROFILE", "chatgpt_demo")
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "chatgpt_oauth")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_DEMO_GRAPH_ID", "demo-graph")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_AUTH_SERVER_URL", "https://api.example.com/oauth/chatgpt")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_RESOURCE_URL", "https://api.example.com/chatgpt-demo/mcp")

    server = create_standalone_mcp_server()
    tools = server._tool_manager._tools

    assert tools["search_documents"].meta == {
        "securitySchemes": [
            {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": "https://api.example.com/oauth/chatgpt/authorize",
                        "tokenUrl": "https://api.example.com/oauth/chatgpt/token",
                        "scopes": {
                            "mnemosyne.mcp.read": "Use Mnemosyne through ChatGPT.",
                        },
                    }
                },
            }
        ],
        "_meta": {
            "securitySchemes": [
                {
                    "type": "oauth2",
                    "flows": {
                        "authorizationCode": {
                            "authorizationUrl": "https://api.example.com/oauth/chatgpt/authorize",
                            "tokenUrl": "https://api.example.com/oauth/chatgpt/token",
                            "scopes": {
                                "mnemosyne.mcp.read": "Use Mnemosyne through ChatGPT.",
                            },
                        }
                    },
                }
            ]
        },
    }

    assert {"write_document", "insert_blocks", "update_blocks", "edit_block_text"} <= set(tools.keys())
    assert tools["write_document"].annotations is not None
    assert tools["write_document"].annotations.readOnlyHint is False
    assert tools["insert_blocks"].annotations is not None
    assert tools["insert_blocks"].annotations.readOnlyHint is False


def test_chatgpt_oauth_defaults_to_demo_profile_when_mcp_profile_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MCP_PROFILE", raising=False)
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "chatgpt_oauth")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_DEMO_GRAPH_ID", "demo-graph")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_AUTH_SERVER_URL", "https://api.example.com/oauth/chatgpt")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_RESOURCE_URL", "https://api.example.com/chatgpt-demo/mcp")

    server = create_standalone_mcp_server()
    tools = server._tool_manager._tools

    assert set(tools.keys()) == {
        "get_workspace",
        "read_blocks",
        "get_block",
        "query_blocks",
        "search_documents",
        "search_blocks",
        "read_document",
        "document_digest",
        "check_document",
        "write_document",
        "insert_blocks",
        "update_blocks",
        "edit_block_text",
    }
    assert "delete_graph" not in tools
    assert "delete_documents" not in tools
    assert "sparql_update" not in tools


def test_chatgpt_oauth_rejects_non_demo_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_PROFILE", "hivemind")
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "chatgpt_oauth")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_AUTH_SERVER_URL", "https://api.example.com/oauth/chatgpt")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_RESOURCE_URL", "https://api.example.com/chatgpt-demo/mcp")

    with pytest.raises(RuntimeError, match="requires MCP_PROFILE=chatgpt_demo"):
        create_standalone_mcp_server()
