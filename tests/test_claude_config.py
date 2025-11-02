"""Tests for the read-only Claude Code configuration helpers."""

import json
from pathlib import Path

import pytest

from neem.utils.claude_config import (
    DEFAULT_SETTINGS_PATH,
    MCP_SERVER_NAME,
    get_settings_path,
    load_settings,
    is_mcp_configured,
    get_mcp_config_status,
)


@pytest.fixture
def settings_path(tmp_path, monkeypatch) -> Path:
    """Provide an isolated CLAUDE_CODE_SETTINGS_PATH for each test."""
    path = tmp_path / ".claude" / "settings.json"
    monkeypatch.setenv("CLAUDE_CODE_SETTINGS_PATH", str(path))
    return path


def test_get_settings_path_default(monkeypatch):
    """When no override is set, fall back to the shared default location."""
    monkeypatch.delenv("CLAUDE_CODE_SETTINGS_PATH", raising=False)
    assert get_settings_path() == DEFAULT_SETTINGS_PATH


def test_get_settings_path_custom(settings_path):
    """Respect CLAUDE_CODE_SETTINGS_PATH when present."""
    assert get_settings_path() == settings_path


def test_load_settings_missing_file_returns_empty(settings_path):
    """Missing settings should yield an empty dict without creating files."""
    assert not settings_path.exists()
    assert load_settings() == {}
    assert not settings_path.exists()


def test_load_settings_empty_file(settings_path):
    """An empty JSON document should parse as an empty dict."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("")
    assert load_settings() == {}


def test_load_settings_invalid_json(settings_path):
    """Invalid JSON is ignored without writing backups."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("{ invalid json")
    backup = settings_path.with_suffix(".json.backup")

    assert load_settings() == {}
    assert not backup.exists()


def test_load_settings_valid_json(settings_path):
    """Valid JSON should be returned as-is."""
    settings = {"mcpServers": {"foo": {"type": "stdio"}}}
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings))

    assert load_settings() == settings


def test_is_mcp_configured_true(settings_path):
    """Detect when the Mnemosyne MCP entry exists."""
    settings = {
        "mcpServers": {
            MCP_SERVER_NAME: {"type": "stdio", "command": "neem-mcp-server"}
        }
    }
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings))

    assert is_mcp_configured() is True


def test_is_mcp_configured_false(settings_path):
    """Return False when the entry is missing."""
    assert is_mcp_configured() is False


def test_get_mcp_config_status(settings_path):
    """Report path, existence, and configuration flag."""
    status = get_mcp_config_status()
    assert status["settings_path"] == str(settings_path)
    assert status["settings_exists"] is False
    assert status["mcp_configured"] is False
    assert status["server_name"] == MCP_SERVER_NAME

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps({"mcpServers": {MCP_SERVER_NAME: {"type": "stdio"}}})
    )

    status = get_mcp_config_status()
    assert status["settings_exists"] is True
    assert status["mcp_configured"] is True
