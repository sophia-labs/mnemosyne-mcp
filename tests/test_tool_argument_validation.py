"""Tests for strict tool argument validation behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, func_metadata
from neem.mcp.server.standalone_server import _enforce_strict_tool_argument_validation


def test_strict_tool_argument_validation_rejects_unknown_fields() -> None:
    """Unknown tool params should fail instead of being silently ignored."""
    original_config = ArgModelBase.model_config
    try:
        _enforce_strict_tool_argument_validation()

        def insert_blocks_like(block_id: str) -> str:
            return block_id

        meta = func_metadata(insert_blocks_like)

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            meta.arg_model.model_validate(
                {
                    "block_id": "block-123",
                    "after_block_id": "block-legacy-param",
                }
            )
    finally:
        ArgModelBase.model_config = original_config
