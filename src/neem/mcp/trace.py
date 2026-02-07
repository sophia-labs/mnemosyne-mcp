"""
Dead-simple trace logger for debugging MCP tool calls.

Writes plain timestamped lines to /tmp/mcp-trace.log.
No structlog, no JSON, no key=value — just readable text.

Usage:
    from neem.mcp.trace import trace
    trace("step description")
    trace("got response", {"key": "value"})
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_TRACE_FILE = Path(os.environ.get("MCP_TRACE_FILE", "/tmp/mcp-trace.log"))
_start_time: Optional[float] = None


def _ensure_file():
    global _start_time
    _TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _start_time is None:
        _start_time = time.monotonic()
        # Write header on first call
        with open(_TRACE_FILE, "a") as f:
            f.write("\n" + "=" * 72 + "\n")
            f.write(f"  MCP TRACE — session started {datetime.now(timezone.utc).isoformat()}\n")
            f.write("=" * 72 + "\n\n")


def trace(step: str, data: Any = None) -> None:
    """Write a trace line to the log file.

    Args:
        step: Human-readable description of what's happening.
        data: Optional dict/list/str to dump alongside.
    """
    _ensure_file()
    elapsed = time.monotonic() - _start_time
    ts = f"+{elapsed:7.2f}s"
    now = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]

    lines = [f"[{now}] {ts}  {step}"]
    if data is not None:
        if isinstance(data, (dict, list)):
            try:
                formatted = json.dumps(data, indent=2, default=str)
            except (TypeError, ValueError):
                formatted = repr(data)
        else:
            formatted = str(data)
        # Indent data under the step
        for line in formatted.splitlines():
            lines.append(f"                      {line}")

    with open(_TRACE_FILE, "a") as f:
        for line in lines:
            f.write(line + "\n")


def trace_separator(label: str = "") -> None:
    """Write a visual separator."""
    _ensure_file()
    elapsed = time.monotonic() - _start_time
    ts = f"+{elapsed:7.2f}s"
    now = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    with open(_TRACE_FILE, "a") as f:
        if label:
            f.write(f"\n[{now}] {ts}  ---- {label} ----\n")
        else:
            f.write(f"\n[{now}] {ts}  " + "-" * 40 + "\n")
