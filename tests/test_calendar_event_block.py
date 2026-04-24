"""Tests for the calendarEvent atom block type.

Validates that XML with <calendarEvent> survives the _xml_to_pycrdt pipeline
with the right attrs + defaults, and that the TipTap-internal attr names
(camelCase) match what y-prosemirror expects on round-trip.
"""

import pycrdt
import pytest

from neem.hocuspocus.document import (
    BLOCK_DEFAULTS,
    BLOCK_TYPES,
    DocumentReader,
    DocumentWriter,
)


@pytest.fixture
def doc():
    return pycrdt.Doc()


def test_calendar_event_is_registered_block_type():
    assert "calendarEvent" in BLOCK_TYPES


def test_calendar_event_has_tiptap_defaults():
    defaults = BLOCK_DEFAULTS["calendarEvent"]
    # Every attr the frontend addAttributes() declares must have a default
    # here or y-prosemirror normalizes the node on client load.
    for key in (
        "timeStart",
        "timeEnd",
        "allDay",
        "title",
        "location",
        "annotation",
        "source",
        "externalEventId",
    ):
        assert key in defaults, f"missing default: {key}"


def test_inserting_a_calendar_event_preserves_attrs(doc):
    writer = DocumentWriter(doc)
    writer.replace_all_content("<paragraph>anchor</paragraph>")

    xml = (
        '<calendarEvent '
        'title="Design review" '
        'timeStart="2026-04-24T09:30" '
        'timeEnd="2026-04-24T10:15" '
        'allDay="false" '
        'location="Office 2B" '
        'annotation="prep doc attached" '
        'source="manual" />'
    )
    new_ids = writer.insert_blocks_at(1, [xml])
    assert len(new_ids) == 1

    reader = DocumentReader(doc)
    fragment = reader.get_content_fragment()
    children = list(fragment.children)
    event = children[-1]

    assert event.tag == "calendarEvent"
    attrs = dict(event.attributes)
    assert attrs["title"] == "Design review"
    assert attrs["timeStart"] == "2026-04-24T09:30"
    assert attrs["timeEnd"] == "2026-04-24T10:15"
    assert attrs["allDay"] == "false"
    assert attrs["location"] == "Office 2B"
    assert attrs["annotation"] == "prep doc attached"
    assert attrs["source"] == "manual"
    # data-block-id must be auto-assigned since calendarEvent is in BLOCK_TYPES.
    assert attrs.get("data-block-id", "").strip() != ""


def test_all_day_event_omits_time_fields(doc):
    writer = DocumentWriter(doc)
    writer.replace_all_content("<paragraph>anchor</paragraph>")
    writer.insert_blocks_at(
        1,
        ['<calendarEvent title="Offsite" allDay="true" source="gcal" externalEventId="abc123" />'],
    )

    reader = DocumentReader(doc)
    children = list(reader.get_content_fragment().children)
    event = children[-1]
    attrs = dict(event.attributes)
    assert attrs["allDay"] == "true"
    assert attrs["source"] == "gcal"
    assert attrs["externalEventId"] == "abc123"
    # Defaults inject empty strings for time attrs on all-day events so
    # y-prosemirror doesn't see missing-attr normalization on client load.
    assert attrs.get("timeStart", "") == ""
    assert attrs.get("timeEnd", "") == ""


def test_calendar_event_gets_unique_block_id_per_insert(doc):
    writer = DocumentWriter(doc)
    writer.replace_all_content("<paragraph>anchor</paragraph>")
    xml = '<calendarEvent title="A" allDay="true" source="manual" />'
    writer.insert_blocks_at(1, [xml])
    writer.insert_blocks_at(2, [xml])

    reader = DocumentReader(doc)
    children = list(reader.get_content_fragment().children)
    events = [c for c in children if c.tag == "calendarEvent"]
    assert len(events) == 2
    ids = [dict(e.attributes).get("data-block-id") for e in events]
    assert ids[0] and ids[1] and ids[0] != ids[1]
