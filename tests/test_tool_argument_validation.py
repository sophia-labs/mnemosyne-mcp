"""Tests for strict tool argument validation behavior."""

from __future__ import annotations

import pytest

from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, func_metadata
from neem.mcp.server.standalone_server import _enforce_strict_tool_argument_validation


def test_strict_tool_argument_validation_rejects_unknown_fields() -> None:
    """Unknown tool params should fail with a helpful message naming permitted fields."""
    original_config = ArgModelBase.model_config
    try:
        _enforce_strict_tool_argument_validation()

        def insert_blocks_like(block_id: str) -> str:
            return block_id

        meta = func_metadata(insert_blocks_like)

        with pytest.raises(ValueError, match="Permitted fields") as excinfo:
            meta.arg_model.model_validate(
                {
                    "block_id": "block-123",
                    "after_block_id": "block-legacy-param",
                }
            )
        msg = str(excinfo.value)
        assert "after_block_id" in msg
        assert "block_id" in msg
    finally:
        ArgModelBase.model_config = original_config
