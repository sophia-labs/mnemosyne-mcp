"""
Token estimation and JSON rendering utilities.

Provides compact JSON rendering and rough token estimation for monitoring
MCP response efficiency.
"""

from __future__ import annotations

import json
from typing import Any, Dict

JsonDict = Dict[str, Any]


def render_compact_json(data: Any) -> str:
    """
    Render JSON in compact format (no whitespace).

    Use this for all data responses to minimize token usage.
    Reserve pretty-printing for human-facing error messages.

    Args:
        data: Data to serialize

    Returns:
        Compact JSON string
    """
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False)


def render_pretty_json(data: Any) -> str:
    """
    Render JSON in pretty format (2-space indent).

    Use sparingly, only for human-facing messages where readability
    is more important than token efficiency.

    Args:
        data: Data to serialize

    Returns:
        Pretty-printed JSON string
    """
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (1 token ≈ 4 characters).

    This is a simplified heuristic. Actual tokenization varies by model,
    but this gives a ballpark figure for optimization tracking.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return max(1, len(text) // 4)


def render_with_stats(data: Any, *, compact: bool = True) -> str:
    """
    Render JSON with token usage comment.

    For development/debugging only. In production, use render_compact_json directly.

    Args:
        data: Data to serialize
        compact: Use compact rendering

    Returns:
        JSON string with embedded stats comment
    """
    json_str = render_compact_json(data) if compact else render_pretty_json(data)
    token_count = estimate_tokens(json_str)

    # Add stats as a trailing comment (not valid JSON, for logging only)
    return f"{json_str}\n# Estimated tokens: {token_count}, Bytes: {len(json_str)}"
