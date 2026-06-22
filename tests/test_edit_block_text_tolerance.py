"""Tests for the extra tolerance edit_block_text gained in P2 B:
typographic-punctuation folding and the closest-substring hint shown when
find/replace misses."""

from __future__ import annotations

import pytest

from neem.mcp.tools.hocuspocus import (
    _build_find_candidates,
    _closest_substring_hint,
    _find_positions_with_auto_match,
    _fold_punctuation,
)


# ── Punctuation folding ────────────────────────────────────────────────────


def test_fold_punctuation_smart_quotes_to_ascii() -> None:
    assert _fold_punctuation("don’t") == "don't"
    assert _fold_punctuation("“hello”") == '"hello"'
    assert _fold_punctuation("‘a’ and “b”") == "'a' and \"b\""


def test_fold_punctuation_dashes_to_hyphen() -> None:
    assert _fold_punctuation("now — later") == "now - later"
    assert _fold_punctuation("9–12") == "9-12"
    assert _fold_punctuation("a−b") == "a-b"


def test_fold_punctuation_ellipsis_to_three_dots() -> None:
    assert _fold_punctuation("wait…") == "wait..."


def test_fold_punctuation_idempotent_on_ascii() -> None:
    assert _fold_punctuation("plain text") == "plain text"


def test_build_find_candidates_includes_folded_form() -> None:
    cands = _build_find_candidates("don’t go")
    assert "don't go" in cands


def test_find_positions_matches_when_only_punctuation_differs() -> None:
    haystack = "she said “hello” quietly"
    matches, matched = _find_positions_with_auto_match(
        plain_text=haystack,
        requested_find='"hello"',
    )
    assert matches, "should match across smart-quote boundary"
    assert matched == '"hello"'


def test_find_positions_matches_em_dash_with_hyphen() -> None:
    haystack = "ship — then ask"
    matches, _matched = _find_positions_with_auto_match(
        plain_text=haystack,
        requested_find="ship - then",
    )
    assert matches, "should match em-dash against ASCII hyphen"


def test_find_positions_matches_apostrophe() -> None:
    haystack = "it’s fine"
    matches, _matched = _find_positions_with_auto_match(
        plain_text=haystack,
        requested_find="it's fine",
    )
    assert matches


# ── Closest-substring hint ─────────────────────────────────────────────────


def test_closest_substring_returns_window_around_match() -> None:
    block = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs."
    )
    hint = _closest_substring_hint(block, "quick brawn fox", window_chars=40)
    assert hint is not None
    assert "quick" in hint["snippet"] or "brown" in hint["snippet"] or "fox" in hint["snippet"]
    assert hint["matched_length"] >= 3


def test_closest_substring_none_when_no_overlap_at_all() -> None:
    # Two strings that share NO characters → no_match
    hint = _closest_substring_hint("aaaa", "zzzz")
    assert hint is None


def test_closest_substring_handles_empty_inputs() -> None:
    assert _closest_substring_hint("", "x") is None
    assert _closest_substring_hint("x", "") is None
    assert _closest_substring_hint("", "") is None


def test_closest_substring_snippet_respects_window() -> None:
    block = "a" * 200 + "needle" + "a" * 200
    hint = _closest_substring_hint(block, "needle", window_chars=20)
    assert hint is not None
    # Window should be roughly window_chars + needle length, not the whole block.
    assert len(hint["snippet"]) <= 50  # 20/2 + 6 + 20/2 = 26-ish, plus slack


def test_closest_substring_returns_matched_chars() -> None:
    block = "alpha beta gamma delta"
    hint = _closest_substring_hint(block, "betta")
    assert hint is not None
    # "beta" is the longest common substring with "betta"
    assert hint["matched_substring"] == "bet" or "bet" in hint["matched_substring"]
