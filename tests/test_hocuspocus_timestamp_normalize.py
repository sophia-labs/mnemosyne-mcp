"""Unit tests for timestamp normalization helpers in hocuspocus tools."""

from neem.mcp.tools.hocuspocus import _normalize_timestamp_to_iso


def test_normalize_iso_with_five_digit_fraction_z() -> None:
    value = "2026-02-10T02:10:43.64423Z"
    normalized = _normalize_timestamp_to_iso(value)
    assert normalized == "2026-02-10T02:10:43.644230+00:00"


def test_normalize_iso_with_long_fraction_and_tz() -> None:
    value = "2026-02-10T02:10:43.12345678+00:00"
    normalized = _normalize_timestamp_to_iso(value)
    assert normalized == "2026-02-10T02:10:43.123456+00:00"


def test_normalize_iso_with_compact_tz_offset() -> None:
    value = "2026-02-10T02:10:43.5+0530"
    normalized = _normalize_timestamp_to_iso(value)
    assert normalized == "2026-02-09T20:40:43.500000+00:00"


def test_normalize_epoch_milliseconds() -> None:
    normalized = _normalize_timestamp_to_iso(1770689439404.0)
    assert normalized == "2026-02-10T02:10:39.404000+00:00"


def test_normalize_invalid_timestamp_returns_none() -> None:
    assert _normalize_timestamp_to_iso("not-a-timestamp") is None
