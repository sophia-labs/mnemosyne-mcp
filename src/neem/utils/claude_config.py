"""
Lightweight helpers for inspecting Claude Code MCP configuration.

These utilities intentionally avoid mutating the user's settings file; our CLI
now only guides people through manual setup using the Claude CLI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
MCP_SERVER_NAME = "mnemosyne-graph"


def get_settings_path() -> Path:
    """
    Return the Claude Code settings path, honoring CLAUDE_CODE_SETTINGS_PATH.
    """
    custom_path = os.getenv("CLAUDE_CODE_SETTINGS_PATH")
    if custom_path:
        return Path(custom_path)
    return DEFAULT_SETTINGS_PATH


def _read_settings_text(path: Path) -> str:
    """
    Read the raw settings text if available, returning an empty string on error.
    """
    try:
        return path.read_text()
    except FileNotFoundError:
        logger.debug("Claude settings file not found", path=str(path))
    except Exception as exc:  # pragma: no cover - rare I/O edge cases
        logger.warning("Failed to read Claude settings", path=str(path), error=str(exc))
    return ""


def load_settings() -> Dict[str, Any]:
    """
    Load the Claude settings file into a dictionary without modifying it.
    """
    settings_path = get_settings_path()

    raw = _read_settings_text(settings_path)
    if not raw.strip():
        if raw:
            logger.debug("Claude settings file is empty", path=str(settings_path))
        return {}

    try:
        settings = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Claude settings JSON invalid", path=str(settings_path), error=str(exc))
        return {}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Unexpected error parsing Claude settings", path=str(settings_path), error=str(exc))
        return {}

    if not isinstance(settings, dict):
        logger.warning("Claude settings JSON is not an object", path=str(settings_path))
        return {}

    return settings


def _has_mnemosyne_server(settings: Dict[str, Any]) -> bool:
    """Return True when the Mnemosyne MCP server entry exists."""
    servers = settings.get("mcpServers")
    if not isinstance(servers, dict):
        return False

    entry = servers.get(MCP_SERVER_NAME)
    return isinstance(entry, dict)


def is_mcp_configured() -> bool:
    """
    Check if the Mnemosyne MCP server is registered with Claude Code.
    """
    settings = load_settings()
    return _has_mnemosyne_server(settings)


def get_mcp_config_status() -> Dict[str, Any]:
    """
    Return a small status blob for CLI display purposes.
    """
    settings_path = get_settings_path()
    exists = settings_path.exists()
    settings: Dict[str, Any] = {}

    if exists:
        settings = load_settings()

    configured = _has_mnemosyne_server(settings)

    return {
        "settings_path": str(settings_path),
        "settings_exists": exists,
        "mcp_configured": configured,
        "server_name": MCP_SERVER_NAME,
    }

