"""Tests for the tag-expiration resolution helper.

The helper takes a raw {tag → value} dict from inline tag markers
({#todo:7d}, {#event:2026-05-15}) and normalizes values to absolute
ISO calendar dates. Anything we cannot normalize is dropped.
"""

from __future__ import annotations

from datetime import date

from neem.mcp.tools.hocuspocus import _resolve_tag_expirations


class TestResolveTagExpirations:
    def test_iso_date_passes_through(self):
        out = _resolve_tag_expirations({"event": "2026-05-15"})
        assert out == {"event": "2026-05-15"}

    def test_relative_duration_resolves_to_absolute(self):
        out = _resolve_tag_expirations(
            {"todo": "7d"},
            today=date(2026, 5, 1),
        )
        assert out == {"todo": "2026-05-08"}

    def test_zero_days_is_today(self):
        out = _resolve_tag_expirations(
            {"event": "0d"},
            today=date(2026, 5, 1),
        )
        assert out == {"event": "2026-05-01"}

    def test_empty_input(self):
        assert _resolve_tag_expirations({}) == {}
        assert _resolve_tag_expirations(None) == {}

    def test_invalid_value_dropped(self):
        out = _resolve_tag_expirations(
            {
                "event": "tomorrow",  # not ISO, not <N>d
                "decision": "next-friday",
                "todo": "7d",  # valid
            },
            today=date(2026, 5, 1),
        )
        assert out == {"todo": "2026-05-08"}

    def test_tag_normalized(self):
        """Tag keys are lowercased and stripped of leading '#'."""
        out = _resolve_tag_expirations(
            {"#Event": "2026-05-15", " TODO ": "7d"},
            today=date(2026, 5, 1),
        )
        assert out == {"event": "2026-05-15", "todo": "2026-05-08"}

    def test_duplicate_tag_first_wins(self):
        """If somehow the same normalized tag key appears twice, first wins."""
        out = _resolve_tag_expirations(
            {"event": "2026-05-15", "#Event": "2026-05-20"},
            today=date(2026, 5, 1),
        )
        assert out == {"event": "2026-05-15"}

    def test_blank_value_dropped(self):
        out = _resolve_tag_expirations({"event": "", "todo": None})
        assert out == {}

    def test_blank_tag_dropped(self):
        out = _resolve_tag_expirations({"": "2026-05-15", "#": "2026-05-15"})
        assert out == {}

    def test_partial_iso_rejected(self):
        """Strings like '2026-5-1' (no zero-padding) must not be accepted."""
        out = _resolve_tag_expirations({"event": "2026-5-15"})
        assert out == {}

    def test_large_relative_duration(self):
        out = _resolve_tag_expirations(
            {"event": "365d"},
            today=date(2026, 5, 1),
        )
        assert out == {"event": "2027-05-01"}
