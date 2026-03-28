"""Unit tests for edit_block_text auto matching helpers."""

from neem.mcp.tools.hocuspocus import _find_positions_with_auto_match


def test_auto_match_strips_list_and_bold_markers():
    plain = (
        "Critique predicates (supports, contradicts, qualifies, exemplifies) "
        "practice Truth."
    )
    requested = (
        "- **Critique predicates** (supports, contradicts, qualifies, exemplifies) "
        "practice **Truth**."
    )
    positions, matched = _find_positions_with_auto_match(plain, requested)
    assert positions == [0]
    assert matched == plain


def test_auto_match_collapses_whitespace():
    plain = "Line one  \n\nLine two"
    requested = "Line one Line two"
    positions, matched = _find_positions_with_auto_match(plain, requested)
    assert positions == [0]
    assert matched == requested
