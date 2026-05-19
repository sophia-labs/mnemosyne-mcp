"""Tests for the calendarEvent / todoItem XML builders used by the
inline-tag write path to materialize atoms into target daily-notes.

These produce TipTap XML strings that get appended to the daily-note's
content y.Doc via DocumentWriter.append_block. Block ids must be
deterministic from (source_block_id, date) so re-firing the same tag
write is idempotent (find_block_by_id catches the dup).
"""

from __future__ import annotations

from neem.mcp.tools.hocuspocus import (
    _build_calendar_event_xml,
    _build_todo_item_xml,
)


class TestCalendarEventBuilder:
    def test_includes_deterministic_id(self) -> None:
        xml = _build_calendar_event_xml(
            source_block_id="block-abc",
            date_str="2026-05-15",
            title="dentist appointment",
        )
        assert 'id="evt-block-abc-2026-05-15"' in xml
        assert 'data-block-id="evt-block-abc-2026-05-15"' in xml

    def test_carries_source_block_reference(self) -> None:
        xml = _build_calendar_event_xml(
            source_block_id="block-abc",
            date_str="2026-05-15",
            title="dentist",
        )
        assert 'data-source-block-id="block-abc"' in xml
        # externalEventId tracks the source so cleanup flows can find atoms
        # from the source side.
        assert 'externalEventId="src:block-abc"' in xml

    def test_default_all_day_and_manual_source(self) -> None:
        xml = _build_calendar_event_xml(
            source_block_id="b1",
            date_str="2026-05-15",
            title="x",
        )
        assert 'allDay="true"' in xml
        assert 'source="manual"' in xml

    def test_title_escapes_quotes_and_brackets(self) -> None:
        xml = _build_calendar_event_xml(
            source_block_id="b1",
            date_str="2026-05-15",
            title='Meeting with "Bob" <urgent>',
        )
        # Quotes and angle brackets must be entity-escaped — otherwise the
        # title attribute closes early or the XML parses as nested elements.
        assert 'title="Meeting with &quot;Bob&quot; &lt;urgent&gt;"' in xml

    def test_renders_as_self_closing_tag(self) -> None:
        xml = _build_calendar_event_xml(
            source_block_id="b1",
            date_str="2026-05-15",
            title="x",
        )
        # Tag MUST be the PM schema node name `calendarEvent`, not the
        # HTML render tag `mn-calendar-event`. y-prosemirror maps Y →
        # PM via schema.node(el.nodeName, ...) directly; a tag mismatch
        # raises and the atom is silently deleted on next sync.
        assert xml.startswith("<calendarEvent ")
        assert xml.endswith("/>")

    def test_empty_title_produces_empty_string(self) -> None:
        xml = _build_calendar_event_xml(
            source_block_id="b1",
            date_str="2026-05-15",
            title="",
        )
        assert 'title=""' in xml

    def test_id_isolation_across_dates(self) -> None:
        """Same source block on two different dates → distinct ids."""
        xml1 = _build_calendar_event_xml(
            source_block_id="b1", date_str="2026-05-15", title="x",
        )
        xml2 = _build_calendar_event_xml(
            source_block_id="b1", date_str="2026-05-16", title="x",
        )
        assert "evt-b1-2026-05-15" in xml1
        assert "evt-b1-2026-05-16" in xml2


class TestTodoItemBuilder:
    def test_includes_deterministic_id_and_listType(self) -> None:
        xml = _build_todo_item_xml(
            source_block_id="b1",
            date_str="2026-05-15",
            content="follow up with vendor",
        )
        assert 'data-block-id="todo-b1-2026-05-15"' in xml
        # listType MUST be `task` — only that value triggers the
        # checkbox-rendering branch in the listItem extension.
        assert 'listType="task"' in xml
        # checked must be ABSENT — string "false" is truthy in JS, so
        # any value (incl. "false") would render as pre-checked.
        assert 'checked="' not in xml

    def test_carries_source_reference(self) -> None:
        xml = _build_todo_item_xml(
            source_block_id="b1",
            date_str="2026-05-15",
            content="x",
        )
        assert 'data-source-block-id="b1"' in xml

    def test_content_wrapped_in_paragraph(self) -> None:
        xml = _build_todo_item_xml(
            source_block_id="b1",
            date_str="2026-05-15",
            content="follow up",
        )
        assert "<paragraph>follow up</paragraph>" in xml

    def test_content_escapes_xml_special_chars(self) -> None:
        xml = _build_todo_item_xml(
            source_block_id="b1",
            date_str="2026-05-15",
            content="check <a> tag & ensure",
        )
        # Inside a paragraph, & and < must escape so the y.XmlElement parser
        # doesn't see fake child elements.
        assert "<paragraph>check &lt;a&gt; tag &amp; ensure</paragraph>" in xml
