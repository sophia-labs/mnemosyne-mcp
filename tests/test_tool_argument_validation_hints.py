"""Tests for the enhanced pydantic extra_forbidden error formatter.

The enhancement turns `Extra inputs are not permitted` into
`Unknown field 'X'. Did you mean 'Y'? Permitted fields: [a, b, c].`
so the agent does not cycle 3-4 wrong kwarg names guessing the right one.
"""

from __future__ import annotations

from typing import Any

import pytest
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase
from pydantic import create_model

from neem.mcp.server.standalone_server import (
    _enforce_strict_tool_argument_validation,
    _format_validation_error_with_field_hints,
    _levenshtein,
    _nearest_field_name,
)


@pytest.fixture(autouse=True)
def _enforce_strict() -> None:
    _enforce_strict_tool_argument_validation()


def _make_model() -> type[ArgModelBase]:
    return create_model(
        "TestModel",
        __base__=ArgModelBase,
        text=(str, ...),
        graph_id=(str | None, None),
        limit=(int, 10),
    )


def test_levenshtein_basic() -> None:
    assert _levenshtein("text", "text") == 0
    assert _levenshtein("text", "tex") == 1
    assert _levenshtein("content", "text") == 4
    assert _levenshtein("", "abc") == 3


def test_nearest_field_name_finds_typo() -> None:
    assert _nearest_field_name("tex", ["text", "graph_id"]) == "text"
    assert _nearest_field_name("limt", ["limit", "text"]) == "limit"
    assert _nearest_field_name("xyz", ["text"]) is None  # too far


def test_nearest_field_name_handles_synonyms() -> None:
    # "content" vs "text" — distance 6, target_len=7, threshold = max(2, 3) = 3
    # So it should NOT suggest text. Test conservatism.
    assert _nearest_field_name("content", ["text", "graph_id"]) is None


def test_extra_forbidden_error_includes_permitted_fields_and_suggestion() -> None:
    Model = _make_model()
    with pytest.raises(ValueError) as excinfo:
        Model.model_validate({"text": "hello", "limt": 5})
    msg = str(excinfo.value)
    assert "limt" in msg
    assert "Did you mean 'limit'" in msg
    assert "Permitted fields" in msg
    assert "limit" in msg
    assert "text" in msg
    assert "graph_id" in msg


def test_extra_forbidden_without_close_match_still_lists_permitted() -> None:
    Model = _make_model()
    with pytest.raises(ValueError) as excinfo:
        Model.model_validate({"text": "hello", "blorpazon": 5})
    msg = str(excinfo.value)
    assert "blorpazon" in msg
    assert "Permitted fields" in msg
    # No suggestion since distance is too large
    assert "Did you mean" not in msg


def test_valid_arguments_pass_through() -> None:
    Model = _make_model()
    obj = Model.model_validate({"text": "hello", "graph_id": "default", "limit": 5})
    assert obj.text == "hello"
    assert obj.graph_id == "default"
    assert obj.limit == 5


def test_missing_required_field_uses_pydantic_default_message() -> None:
    Model = _make_model()
    with pytest.raises(Exception) as excinfo:
        Model.model_validate({"graph_id": "default"})
    # Missing required field is NOT an extra_forbidden error;
    # should fall through to pydantic's default ValidationError.
    msg = str(excinfo.value)
    assert "text" in msg
    # Should NOT be wrapped in our ValueError formatting
    assert "Permitted fields" not in msg


def test_multiple_extras_all_reported() -> None:
    Model = _make_model()
    with pytest.raises(ValueError) as excinfo:
        Model.model_validate({"text": "hi", "foo": 1, "bar": 2})
    msg = str(excinfo.value)
    assert "foo" in msg
    assert "bar" in msg
    assert "Permitted fields" in msg


def test_format_helper_returns_none_when_no_extras() -> None:
    Model = _make_model()
    from pydantic import ValidationError as PydanticValidationError

    try:
        Model.model_validate({"graph_id": "default"})
    except PydanticValidationError as exc:
        # Required-field error has no extras — formatter should return None
        result = _format_validation_error_with_field_hints(Model, exc)
        assert result is None
