"""Tests for the chatgpt_account_readonly MCP profile."""

from __future__ import annotations

import pytest

from neem.mcp.server.standalone_server import create_standalone_mcp_server


@pytest.fixture
def _oauth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_PROFILE", "chatgpt_account_readonly")
    monkeypatch.setenv("MNEMOSYNE_MCP_AUTH_MODE", "chatgpt_oauth")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_AUTH_SERVER_URL", "https://api.example.com/oauth/chatgpt")
    monkeypatch.setenv("MNEMOSYNE_CHATGPT_OAUTH_RESOURCE_URL", "https://api.example.com/chatgpt-auth/mcp")


def test_chatgpt_account_readonly_profile_exposes_expected_tools(_oauth_env: None) -> None:
    server = create_standalone_mcp_server()

    tools = server._tool_manager._tools
    assert set(tools.keys()) == {
        "list_graphs",
        "get_workspace",
        "read_blocks",
        "get_block",
        "query_blocks",
        "search_documents",
        "search_blocks",
        "read_document",
        "document_digest",
    }


def test_chatgpt_account_readonly_profile_requires_graph_id_and_marks_tools_read_only(
    _oauth_env: None,
) -> None:
    server = create_standalone_mcp_server()
    tools = server._tool_manager._tools

    list_graphs = tools["list_graphs"]
    assert "graph_id" not in list_graphs.parameters["properties"]
    assert list_graphs.annotations is not None
    assert list_graphs.annotations.readOnlyHint is True
    assert list_graphs.meta == {
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

    search_documents = tools["search_documents"]
    assert "graph_id" in search_documents.parameters["properties"]
    assert set(search_documents.parameters["required"]) == {"graph_id", "query"}

    search_blocks = tools["search_blocks"]
    assert "graph_id" in search_blocks.parameters["properties"]
    assert "document_id" in search_blocks.parameters["properties"]
    assert set(search_blocks.parameters["required"]) == {"graph_id", "query"}

    read_document = tools["read_document"]
    assert "graph_id" in read_document.parameters["properties"]
    assert set(read_document.parameters["required"]) == {"graph_id", "document_id"}

    document_digest = tools["document_digest"]
    assert "graph_id" in document_digest.parameters["properties"]
    assert set(document_digest.parameters["required"]) == {"graph_id", "document_id"}

    get_workspace = tools["get_workspace"]
    assert "graph_id" in get_workspace.parameters["properties"]
    assert set(get_workspace.parameters["required"]) == {"graph_id"}

    read_blocks = tools["read_blocks"]
    assert "graph_id" in read_blocks.parameters["properties"]
    assert set(read_blocks.parameters["required"]) == {"graph_id", "document_id"}

    get_block = tools["get_block"]
    assert "graph_id" in get_block.parameters["properties"]
    assert set(get_block.parameters["required"]) == {"graph_id", "document_id", "block_id"}

    query_blocks = tools["query_blocks"]
    assert "graph_id" in query_blocks.parameters["properties"]
    assert set(query_blocks.parameters["required"]) == {"graph_id", "document_id"}
