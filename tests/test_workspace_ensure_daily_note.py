"""Tests for WorkspaceWriter.ensure_daily_note.

Mirrors the frontend ensureDailyNote helper: deterministic doc id
`daily-note-{YYYY-MM-DD}`, idempotent on re-call, daily-note metadata
written via the upsert_document `extra` channel so other surfaces (the
sidebar, search) recognize the doc as a daily note.
"""

from __future__ import annotations

import pycrdt

from neem.hocuspocus.workspace import WorkspaceReader, WorkspaceWriter


def _fresh_doc() -> pycrdt.Doc:
    doc = pycrdt.Doc()
    doc.get("documents", type=pycrdt.Map)
    doc.get("folders", type=pycrdt.Map)
    return doc


class TestEnsureDailyNote:
    def test_creates_daily_note_when_absent(self) -> None:
        doc = _fresh_doc()
        writer = WorkspaceWriter(doc)
        reader = WorkspaceReader(doc)

        doc_id = writer.ensure_daily_note("2026-05-15")

        assert doc_id == "daily-note-2026-05-15"
        entry = reader.get_document(doc_id)
        assert entry is not None
        assert entry["documentKind"] == "daily-note"
        assert entry["dailyNoteDate"] == "2026-05-15"
        # Title format mirrors Vera's frontend formatDailyNoteTitle.
        assert entry["title"] == "May 15, 2026"

    def test_idempotent_on_repeat(self) -> None:
        doc = _fresh_doc()
        writer = WorkspaceWriter(doc)

        first_id = writer.ensure_daily_note("2026-05-15")
        # Mutate the entry to a sentinel to detect re-creation.
        documents_map = doc.get("documents", type=pycrdt.Map)
        documents_map[first_id]["title"] = "Manually Renamed"

        second_id = writer.ensure_daily_note("2026-05-15")

        assert first_id == second_id
        # Entry should NOT have been overwritten by the second call.
        assert documents_map[first_id]["title"] == "Manually Renamed"

    def test_time_zone_optional(self) -> None:
        doc = _fresh_doc()
        writer = WorkspaceWriter(doc)
        reader = WorkspaceReader(doc)

        writer.ensure_daily_note("2026-05-15", time_zone="America/Los_Angeles")
        entry = reader.get_document("daily-note-2026-05-15")
        assert entry["dailyNoteTimeZone"] == "America/Los_Angeles"

        writer.ensure_daily_note("2026-05-16")  # no time_zone
        entry2 = reader.get_document("daily-note-2026-05-16")
        assert entry2 is not None
        assert "dailyNoteTimeZone" not in entry2

    def test_invalid_date_does_not_corrupt_workspace(self) -> None:
        """Malformed dates produce a no-op rather than a half-created entry."""
        doc = _fresh_doc()
        writer = WorkspaceWriter(doc)
        reader = WorkspaceReader(doc)

        doc_id = writer.ensure_daily_note("not-a-date")
        # We still return the deterministic id, but the doc was not created.
        assert doc_id == "daily-note-not-a-date"
        assert reader.get_document(doc_id) is None

    def test_distinct_dates_create_distinct_docs(self) -> None:
        doc = _fresh_doc()
        writer = WorkspaceWriter(doc)
        reader = WorkspaceReader(doc)

        writer.ensure_daily_note("2026-05-15")
        writer.ensure_daily_note("2026-05-16")
        writer.ensure_daily_note("2026-05-17")

        docs_map = doc.get("documents", type=pycrdt.Map)
        assert len(list(docs_map.keys())) == 3
        assert reader.get_document("daily-note-2026-05-15")["dailyNoteDate"] == "2026-05-15"
        assert reader.get_document("daily-note-2026-05-16")["dailyNoteDate"] == "2026-05-16"
        assert reader.get_document("daily-note-2026-05-17")["dailyNoteDate"] == "2026-05-17"

    def test_title_handles_single_digit_day(self) -> None:
        """Day in title is unpadded — '5' not '05', matching Vera's frontend."""
        doc = _fresh_doc()
        writer = WorkspaceWriter(doc)
        reader = WorkspaceReader(doc)

        writer.ensure_daily_note("2026-05-05")
        entry = reader.get_document("daily-note-2026-05-05")
        assert entry["title"] == "May 5, 2026"

    def test_title_january(self) -> None:
        doc = _fresh_doc()
        writer = WorkspaceWriter(doc)
        reader = WorkspaceReader(doc)

        writer.ensure_daily_note("2026-01-31")
        entry = reader.get_document("daily-note-2026-01-31")
        assert entry["title"] == "January 31, 2026"
