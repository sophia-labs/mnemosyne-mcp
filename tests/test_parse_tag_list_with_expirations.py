"""Tests for parse_tag_list_with_expirations.

This is the helper that lets value() (and other callers that take a flat
``tags=[...]`` list) accept inline-marker syntax for expirations:

    tags=["event:2026-05-15"]    →  (["event"], {"event": "2026-05-15"})
    tags=["todo:7d"]             →  (["todo"], {"todo": "7d"})
    tags=["decision"]            →  (["decision"], {})

Same colon-suffix as the ``{#event:7d}`` inline markers so users learn
the syntax once and can use it everywhere.
"""

from __future__ import annotations

from neem.mcp.tools.hocuspocus import parse_tag_list_with_expirations


class TestParseTagListWithExpirations:
    def test_bare_tag(self) -> None:
        tags, exp = parse_tag_list_with_expirations(["decision"])
        assert tags == ["decision"]
        assert exp == {}

    def test_tag_with_iso_date(self) -> None:
        tags, exp = parse_tag_list_with_expirations(["event:2026-05-15"])
        assert tags == ["event"]
        assert exp == {"event": "2026-05-15"}

    def test_tag_with_relative_duration(self) -> None:
        tags, exp = parse_tag_list_with_expirations(["todo:7d"])
        assert tags == ["todo"]
        assert exp == {"todo": "7d"}

    def test_mixed_bare_and_dated(self) -> None:
        tags, exp = parse_tag_list_with_expirations(
            ["decision", "event:2026-05-15", "todo:7d"]
        )
        assert tags == ["decision", "event", "todo"]
        assert exp == {"event": "2026-05-15", "todo": "7d"}

    def test_strips_leading_hash(self) -> None:
        tags, exp = parse_tag_list_with_expirations(["#event:2026-05-15"])
        assert tags == ["event"]
        assert exp == {"event": "2026-05-15"}

    def test_lowercases_tag_name(self) -> None:
        tags, exp = parse_tag_list_with_expirations(["EVENT:7d", "Decision"])
        assert tags == ["event", "decision"]
        assert exp == {"event": "7d"}

    def test_empty_input(self) -> None:
        assert parse_tag_list_with_expirations(None) == ([], {})
        assert parse_tag_list_with_expirations([]) == ([], {})

    def test_blank_entries_dropped(self) -> None:
        tags, exp = parse_tag_list_with_expirations(["", "  ", "#"])
        assert tags == []
        assert exp == {}

    def test_dedup_same_tag(self) -> None:
        """Same tag listed twice (e.g. once bare, once dated) — dedupes the
        name, but the dated form's expiration is recorded."""
        tags, exp = parse_tag_list_with_expirations(["event", "event:2026-05-15"])
        assert tags == ["event"]
        assert exp == {"event": "2026-05-15"}

    def test_later_expiration_wins(self) -> None:
        """If the same tag appears with two different dates, the later wins."""
        tags, exp = parse_tag_list_with_expirations(
            ["todo:7d", "todo:14d"]
        )
        assert tags == ["todo"]
        assert exp == {"todo": "14d"}

    def test_colon_with_no_value(self) -> None:
        """Trailing colon with no duration falls back to bare tag."""
        tags, exp = parse_tag_list_with_expirations(["event:"])
        assert tags == ["event"]
        assert exp == {}

    def test_unknown_duration_format_kept_for_resolver(self) -> None:
        """The parser doesn't validate the duration — that happens at
        resolution time. Anything after the first colon is the duration."""
        tags, exp = parse_tag_list_with_expirations(["event:tomorrow"])
        assert tags == ["event"]
        assert exp == {"event": "tomorrow"}

    def test_strip_whitespace_around_components(self) -> None:
        tags, exp = parse_tag_list_with_expirations(["  event : 2026-05-15  "])
        # Leading whitespace stripped on the whole string;
        # but we partition on first ':' — value stripped too.
        assert tags == ["event"]
        assert exp == {"event": "2026-05-15"}
