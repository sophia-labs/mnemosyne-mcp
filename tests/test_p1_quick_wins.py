"""Light coverage for P1 quick-win fixes — alias tools, XML constraint docs,
and read-side retry policy. Heavier behavioral tests live alongside the
implementations they cover."""

from __future__ import annotations

import pytest


def test_xml_constraint_note_in_write_tools_descriptions() -> None:
    """Audit P1 #14: write tools should call out the XML well-formedness
    constraint so 'junk after document element' becomes self-diagnosable."""
    src = open(
        "/Users/eschaton/Documents/GitHub/mnemosyne-mcp/src/neem/mcp/tools/hocuspocus.py"
    ).read()
    # write_document
    assert "junk after document element" in src, "write_document description should mention the specific error string"
    assert "entity-escape" in src or "entity-escaped" in src
    # update_blocks
    assert "XML constraint" in src
    # insert_blocks
    assert "XML constraints (when content is raw XML)" in src


def test_store_memory_alias_registered() -> None:
    """Audit P1 #6: server-side alias for the hallucinated `store_memory` name."""
    src = open(
        "/Users/eschaton/Documents/GitHub/mnemosyne-mcp/src/neem/mcp/tools/geist.py"
    ).read()
    assert 'name="store_memory"' in src
    assert "DEPRECATED" in src or "deprecated" in src
    assert "remember" in src


def test_read_retry_backoffs_defined() -> None:
    """Audit P1 #8: read tools should retry transient errors with bounded backoff."""
    src = open(
        "/Users/eschaton/Documents/GitHub/mnemosyne-mcp/src/neem/mcp/tools/hocuspocus.py"
    ).read()
    assert "_READ_RETRY_BACKOFFS = (0.5, 2.0)" in src
    # Retry only on 5xx + transient network exceptions
    assert "500 <= response.status_code < 600" in src
    assert "httpx.TimeoutException" in src
    assert "httpx.ConnectError" in src


def test_structured_logger_module_loads_with_kwarg_merge() -> None:
    """Audit P1 #10: StructuredLogger merges arbitrary kwargs."""
    from neem.utils.logging import StructuredLogger

    assert hasattr(StructuredLogger, "_merge_extra")
    # Smoke: build a logger and call all methods with extra kwargs
    from neem.utils.logging import LoggerFactory

    logger = LoggerFactory.get_logger("test-smoke")
    # These should not raise (canonical or kwarg styles).
    logger.warning("event", error="x")
    logger.info("event", removed=1)
    logger.error("event", path="/tmp", code=500)
    ctx = logger.with_context(req="abc")
    ctx.warning("event2", extra_field="ok")
