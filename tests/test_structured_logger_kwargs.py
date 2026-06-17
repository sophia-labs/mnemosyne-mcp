"""Tests for StructuredLogger / ContextualLogger kwarg tolerance.

Audit P1 #10: callsites use the structlog idiom `logger.warning("event", key=value)`
and were crashing with `StructuredLogger.warning() got an unexpected keyword
argument 'error'`. The fix merges arbitrary kwargs into extra_context so both
the canonical `extra_context={...}` style and the structlog-style work.
"""

from __future__ import annotations

from typing import Any

import pytest

from neem.utils.logging import LoggerFactory, StructuredLogger


def _capture_extra_context(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Return a dict that's populated with the extra_context of the most recent
    call to StructuredLogger._log_with_context."""
    captured: dict[str, Any] = {}

    real = StructuredLogger._log_with_context

    def spy(self, level, message, *, extra_context=None, exception=None, lazy_context=None):
        captured["level"] = level
        captured["message"] = message
        captured["extra_context"] = extra_context
        captured["exception"] = exception
        # Do NOT call real — we don't want to spam the real logger during tests.

    monkeypatch.setattr(StructuredLogger, "_log_with_context", spy)
    return captured


def test_warning_accepts_extra_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_extra_context(monkeypatch)
    logger = LoggerFactory.get_logger("test-warning-kwargs")
    logger.warning("foo_event", error="boom", graph_id="default")
    assert captured["level"] == "WARNING"
    assert captured["extra_context"]["error"] == "boom"
    assert captured["extra_context"]["graph_id"] == "default"


def test_info_accepts_extra_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_extra_context(monkeypatch)
    logger = LoggerFactory.get_logger("test-info-kwargs")
    logger.info("foo_event", removed=3, kept=10)
    assert captured["extra_context"]["removed"] == 3
    assert captured["extra_context"]["kept"] == 10


def test_error_accepts_extra_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_extra_context(monkeypatch)
    logger = LoggerFactory.get_logger("test-error-kwargs")
    logger.error("foo_event", path="/tmp/x", error="nope")
    assert captured["extra_context"]["path"] == "/tmp/x"
    assert captured["extra_context"]["error"] == "nope"


def test_extra_context_dict_and_kwargs_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_extra_context(monkeypatch)
    logger = LoggerFactory.get_logger("test-merge")
    logger.warning("foo_event", extra_context={"already": "here"}, also="there")
    assert captured["extra_context"]["already"] == "here"
    assert captured["extra_context"]["also"] == "there"


def test_kwargs_override_extra_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_extra_context(monkeypatch)
    logger = LoggerFactory.get_logger("test-override")
    logger.warning("foo_event", extra_context={"k": "old"}, k="new")
    assert captured["extra_context"]["k"] == "new"


def test_contextual_logger_accepts_extra_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_extra_context(monkeypatch)
    base = LoggerFactory.get_logger("test-contextual")
    ctx = base.with_context(req_id="abc-123")
    ctx.warning("ctx_event", error="boom")
    # extra_context should include both the request-context AND the extra kwarg
    assert captured["extra_context"]["req_id"] == "abc-123"
    assert captured["extra_context"]["error"] == "boom"


def test_canonical_extra_context_style_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard — make sure the kwarg-merge didn't break canonical usage."""
    captured = _capture_extra_context(monkeypatch)
    logger = LoggerFactory.get_logger("test-canonical")
    logger.warning("foo_event", extra_context={"a": 1, "b": 2})
    assert captured["extra_context"] == {"a": 1, "b": 2}


def test_no_kwargs_no_extra_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_extra_context(monkeypatch)
    logger = LoggerFactory.get_logger("test-empty")
    logger.warning("foo_event")
    assert captured["extra_context"] is None
