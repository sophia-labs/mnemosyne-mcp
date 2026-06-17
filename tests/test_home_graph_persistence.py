"""Tests for persisted home-graph state across MCP reconnects."""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pytest


def _reload_decorators(monkeypatch: pytest.MonkeyPatch, state_path: Path):
    monkeypatch.setenv("MNEMOSYNE_MCP_HOME_GRAPH_PATH", str(state_path))
    from neem.mcp.tools import decorators as _decorators
    return importlib.reload(_decorators)


def test_set_and_get_home_graph_in_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_path = tmp_path / "mcp_home_graphs.json"
    decorators = _reload_decorators(monkeypatch, state_path)

    decorators.set_home_graph("user-1", "default")
    assert decorators.get_home_graph("user-1") == "default"


def test_home_graph_persists_to_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_path = tmp_path / "mcp_home_graphs.json"
    decorators = _reload_decorators(monkeypatch, state_path)

    decorators.set_home_graph("user-1", "default")

    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert payload["user-1"]["graph_id"] == "default"
    assert isinstance(payload["user-1"]["ts"], (int, float))


def test_home_graph_restored_across_module_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate an MCP reconnect: write state, reload module, expect state to survive."""
    state_path = tmp_path / "mcp_home_graphs.json"
    decorators = _reload_decorators(monkeypatch, state_path)
    decorators.set_home_graph("user-1", "default")

    decorators_after_reconnect = _reload_decorators(monkeypatch, state_path)
    assert decorators_after_reconnect.get_home_graph("user-1") == "default"


def test_home_graph_expires_after_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "mcp_home_graphs.json"
    decorators = _reload_decorators(monkeypatch, state_path)
    decorators.set_home_graph("user-1", "default")

    # Backdate the persisted timestamp beyond the 24h TTL
    payload = json.loads(state_path.read_text())
    payload["user-1"]["ts"] = time.time() - (25 * 60 * 60)
    state_path.write_text(json.dumps(payload))

    decorators_after_reconnect = _reload_decorators(monkeypatch, state_path)
    assert decorators_after_reconnect.get_home_graph("user-1") is None


def test_clear_home_graph_removes_disk_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "mcp_home_graphs.json"
    decorators = _reload_decorators(monkeypatch, state_path)

    decorators.set_home_graph("user-1", "default")
    decorators.clear_home_graph("user-1")

    payload = json.loads(state_path.read_text())
    assert "user-1" not in payload


def test_per_user_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_path = tmp_path / "mcp_home_graphs.json"
    decorators = _reload_decorators(monkeypatch, state_path)

    decorators.set_home_graph("user-1", "default")
    decorators.set_home_graph("user-2", "sophia-labs")

    assert decorators.get_home_graph("user-1") == "default"
    assert decorators.get_home_graph("user-2") == "sophia-labs"

    decorators.clear_home_graph("user-1")
    assert decorators.get_home_graph("user-1") is None
    assert decorators.get_home_graph("user-2") == "sophia-labs"


def test_corrupt_state_file_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "mcp_home_graphs.json"
    state_path.write_text("not valid json {{{")

    decorators = _reload_decorators(monkeypatch, state_path)
    assert decorators.get_home_graph("user-1") is None

    decorators.set_home_graph("user-1", "default")
    assert decorators.get_home_graph("user-1") == "default"
