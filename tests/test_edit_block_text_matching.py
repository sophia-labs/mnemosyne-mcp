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
    matches, matched = _find_positions_with_auto_match(plain, requested)
    assert matches == [(0, len(plain))]
    assert matched == plain


def test_auto_match_collapses_whitespace():
    plain = "Line one  \n\nLine two"
    requested = "Line one Line two"
    matches, matched = _find_positions_with_auto_match(plain, requested)
    assert matches == [(0, len(plain))]
    assert matched == requested


def test_auto_match_collapsed_span_lengths_follow_original_text():
    plain = "A  \n\nB"
    requested = "A B"
    matches, matched = _find_positions_with_auto_match(plain, requested)
    assert matches == [(0, len(plain))]
    assert matched == requested
