"""Helpers for optional MCP progress reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


async def safe_report_progress(context: Context | None, current: int, total: int) -> None:
    """Report progress when possible, but tolerate in-process tool calls.

    FastMCP exposes a Context object for direct `call_tool()` usage, but that
    context may not have an active request backing it. In that case
    `context.report_progress()` raises ValueError. For local smoke harnesses we
    want progress reporting to be non-fatal.
    """
    if context is None:
        return

    try:
        await context.report_progress(current, total)
    except ValueError:
        return
