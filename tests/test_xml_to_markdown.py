"""Tests for TipTap XML → Markdown converter."""

from __future__ import annotations

from neem.hocuspocus.converters import tiptap_xml_to_markdown


# ---------------------------------------------------------------------------
# Empty / Edge Cases
# ---------------------------------------------------------------------------

def test_empty_string() -> None:
    assert tiptap_xml_to_markdown("") == ""


def test_whitespace_only() -> None:
    assert tiptap_xml_to_markdown("   \n  ") == ""


def test_none_input() -> None:
    assert tiptap_xml_to_markdown(None) == ""  # type: ignore[arg-type]


def test_invalid_xml_fallback() -> None:
    result = tiptap_xml_to_markdown("<broken <xml>")
    # Should fallback to stripped text
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

def test_heading_level_1() -> None:
    xml = '<heading level="1">Hello World</heading>'
    assert tiptap_xml_to_markdown(xml) == "# Hello World\n"


def test_heading_level_2() -> None:
    xml = '<heading level="2">Section</heading>'
    assert tiptap_xml_to_markdown(xml) == "## Section\n"


def test_heading_level_3() -> None:
    xml = '<heading level="3">Subsection</heading>'
    assert tiptap_xml_to_markdown(xml) == "### Subsection\n"


def test_heading_with_float_level() -> None:
    """pycrdt stores level as float (e.g. "2.0")."""
    xml = '<heading level="2.0">Float Level</heading>'
    assert tiptap_xml_to_markdown(xml) == "## Float Level\n"


def test_heading_with_inline_marks() -> None:
    xml = '<heading level="2">Hello <strong>bold</strong> world</heading>'
    assert tiptap_xml_to_markdown(xml) == "## Hello **bold** world\n"


# ---------------------------------------------------------------------------
# Paragraphs
# ---------------------------------------------------------------------------

def test_simple_paragraph() -> None:
    xml = "<paragraph>Hello world</paragraph>"
    assert tiptap_xml_to_markdown(xml) == "Hello world\n"


def test_empty_paragraph() -> None:
    xml = "<paragraph></paragraph>"
    assert tiptap_xml_to_markdown(xml) == "\n"


def test_multiple_paragraphs() -> None:
    xml = "<paragraph>First</paragraph><paragraph>Second</paragraph>"
    result = tiptap_xml_to_markdown(xml)
    assert "First" in result
    assert "Second" in result
    # Should have blank line between paragraphs
    assert "First\n\nSecond" in result


# ---------------------------------------------------------------------------
# Inline Marks: Bold
# ---------------------------------------------------------------------------

def test_bold_strong() -> None:
    xml = "<paragraph><strong>bold text</strong></paragraph>"
    assert "**bold text**" in tiptap_xml_to_markdown(xml)


def test_bold_tag() -> None:
    xml = "<paragraph><bold>bold text</bold></paragraph>"
    assert "**bold text**" in tiptap_xml_to_markdown(xml)


# ---------------------------------------------------------------------------
# Inline Marks: Italic
# ---------------------------------------------------------------------------

def test_italic_em() -> None:
    xml = "<paragraph><em>italic text</em></paragraph>"
    assert "*italic text*" in tiptap_xml_to_markdown(xml)


def test_italic_tag() -> None:
    xml = "<paragraph><italic>italic text</italic></paragraph>"
    assert "*italic text*" in tiptap_xml_to_markdown(xml)


# ---------------------------------------------------------------------------
# Inline Marks: Strikethrough
# ---------------------------------------------------------------------------

def test_strikethrough_s() -> None:
    xml = "<paragraph><s>deleted</s></paragraph>"
    assert "~~deleted~~" in tiptap_xml_to_markdown(xml)


def test_strikethrough_strike() -> None:
    xml = "<paragraph><strike>deleted</strike></paragraph>"
    assert "~~deleted~~" in tiptap_xml_to_markdown(xml)


# ---------------------------------------------------------------------------
# Inline Marks: Code
# ---------------------------------------------------------------------------

def test_inline_code() -> None:
    xml = "<paragraph><code>foo()</code></paragraph>"
    assert "`foo()`" in tiptap_xml_to_markdown(xml)


def test_inline_code_with_backtick() -> None:
    xml = "<paragraph><code>use `backticks`</code></paragraph>"
    result = tiptap_xml_to_markdown(xml)
    assert "`` use `backticks` ``" in result


# ---------------------------------------------------------------------------
# Inline Marks: Links
# ---------------------------------------------------------------------------

def test_link() -> None:
    xml = '<paragraph><a href="https://example.com">click here</a></paragraph>'
    assert "[click here](https://example.com)" in tiptap_xml_to_markdown(xml)


def test_link_with_bold() -> None:
    xml = '<paragraph><a href="https://example.com"><strong>bold link</strong></a></paragraph>'
    result = tiptap_xml_to_markdown(xml)
    assert "[**bold link**](https://example.com)" in result


# ---------------------------------------------------------------------------
# Inline Marks: Highlight
# ---------------------------------------------------------------------------

def test_highlight() -> None:
    xml = "<paragraph><mark>highlighted</mark></paragraph>"
    assert "==highlighted==" in tiptap_xml_to_markdown(xml)


# ---------------------------------------------------------------------------
# Inline Marks: Comments (stripped)
# ---------------------------------------------------------------------------

def test_comment_mark_stripped() -> None:
    xml = '<paragraph>before <commentMark data-comment-id="c1">commented</commentMark> after</paragraph>'
    result = tiptap_xml_to_markdown(xml)
    assert "before commented after" in result
    assert "commentMark" not in result
    assert "c1" not in result


# ---------------------------------------------------------------------------
# Inline Marks: Nested
# ---------------------------------------------------------------------------

def test_nested_bold_italic() -> None:
    xml = "<paragraph><strong><em>bold italic</em></strong></paragraph>"
    result = tiptap_xml_to_markdown(xml)
    assert "***bold italic***" in result


def test_bold_with_tail_text() -> None:
    xml = "<paragraph>start <strong>bold</strong> end</paragraph>"
    result = tiptap_xml_to_markdown(xml)
    assert "start **bold** end" in result


def test_multiple_marks_in_paragraph() -> None:
    xml = "<paragraph><strong>bold</strong> and <em>italic</em></paragraph>"
    result = tiptap_xml_to_markdown(xml)
    assert "**bold**" in result
    assert "*italic*" in result
    assert "**bold** and *italic*" in result


# ---------------------------------------------------------------------------
# Footnotes
# ---------------------------------------------------------------------------

def test_footnote() -> None:
    xml = '<paragraph>Some text<footnote data-footnote-content="This is a note"/></paragraph>'
    result = tiptap_xml_to_markdown(xml)
    assert "[^1]" in result
    assert "[^1]: This is a note" in result


def test_multiple_footnotes() -> None:
    xml = (
        '<paragraph>First<footnote data-footnote-content="Note one"/> '
        'and second<footnote data-footnote-content="Note two"/></paragraph>'
    )
    result = tiptap_xml_to_markdown(xml)
    assert "[^1]" in result
    assert "[^2]" in result
    assert "[^1]: Note one" in result
    assert "[^2]: Note two" in result


# ---------------------------------------------------------------------------
# Wikilinks
# ---------------------------------------------------------------------------

def test_wikilink() -> None:
    xml = '<paragraph>See <wikilink label="My Document"/> for details</paragraph>'
    result = tiptap_xml_to_markdown(xml)
    assert "My Document" in result


# ---------------------------------------------------------------------------
# Hard Breaks
# ---------------------------------------------------------------------------

def test_hard_break() -> None:
    xml = "<paragraph>line one<hardBreak/>line two</paragraph>"
    result = tiptap_xml_to_markdown(xml)
    assert "line one" in result
    assert "line two" in result


# ---------------------------------------------------------------------------
# Code Blocks
# ---------------------------------------------------------------------------

def test_code_block_no_language() -> None:
    xml = "<codeBlock>print('hello')</codeBlock>"
    result = tiptap_xml_to_markdown(xml)
    assert "```\nprint('hello')\n```" in result


def test_code_block_with_language() -> None:
    xml = '<codeBlock language="python">def foo():\n    pass</codeBlock>'
    result = tiptap_xml_to_markdown(xml)
    assert "```python" in result
    assert "def foo():" in result
    assert "```" in result


def test_code_block_preserves_content_literally() -> None:
    """Code block content should not have marks processed."""
    xml = "<codeBlock>**not bold** *not italic*</codeBlock>"
    result = tiptap_xml_to_markdown(xml)
    assert "**not bold** *not italic*" in result


# ---------------------------------------------------------------------------
# Bullet Lists (flat TipTap representation)
# ---------------------------------------------------------------------------

def test_bullet_list_simple() -> None:
    xml = (
        '<bulletList>'
        '<listItem listType="bullet"><paragraph>Item one</paragraph></listItem>'
        '<listItem listType="bullet"><paragraph>Item two</paragraph></listItem>'
        '</bulletList>'
    )
    result = tiptap_xml_to_markdown(xml)
    assert "- Item one" in result
    assert "- Item two" in result


def test_bullet_list_with_indent() -> None:
    xml = (
        '<bulletList>'
        '<listItem listType="bullet"><paragraph>Top</paragraph></listItem>'
        '<listItem listType="bullet" data-indent="1"><paragraph>Nested</paragraph></listItem>'
        '<listItem listType="bullet" data-indent="2"><paragraph>Deep</paragraph></listItem>'
        '</bulletList>'
    )
    result = tiptap_xml_to_markdown(xml)
    assert "- Top" in result
    assert "  - Nested" in result
    assert "    - Deep" in result


# ---------------------------------------------------------------------------
# Ordered Lists
# ---------------------------------------------------------------------------

def test_ordered_list() -> None:
    xml = (
        '<orderedList>'
        '<listItem listType="ordered"><paragraph>First</paragraph></listItem>'
        '<listItem listType="ordered"><paragraph>Second</paragraph></listItem>'
        '</orderedList>'
    )
    result = tiptap_xml_to_markdown(xml)
    assert "1. First" in result
    assert "1. Second" in result


def test_ordered_list_with_indent() -> None:
    xml = (
        '<orderedList>'
        '<listItem listType="ordered"><paragraph>Outer</paragraph></listItem>'
        '<listItem listType="ordered" data-indent="1"><paragraph>Inner</paragraph></listItem>'
        '</orderedList>'
    )
    result = tiptap_xml_to_markdown(xml)
    assert "1. Outer" in result
    assert "  1. Inner" in result


# ---------------------------------------------------------------------------
# Task Lists
# ---------------------------------------------------------------------------

def test_task_list_unchecked() -> None:
    xml = (
        '<taskList>'
        '<taskItem checked="false"><paragraph>Todo</paragraph></taskItem>'
        '</taskList>'
    )
    result = tiptap_xml_to_markdown(xml)
    assert "- [ ] Todo" in result


def test_task_list_checked() -> None:
    xml = (
        '<taskList>'
        '<taskItem checked="true"><paragraph>Done</paragraph></taskItem>'
        '</taskList>'
    )
    result = tiptap_xml_to_markdown(xml)
    assert "- [x] Done" in result


def test_task_list_mixed() -> None:
    xml = (
        '<taskList>'
        '<taskItem checked="true"><paragraph>Done</paragraph></taskItem>'
        '<taskItem checked="false"><paragraph>Not done</paragraph></taskItem>'
        '</taskList>'
    )
    result = tiptap_xml_to_markdown(xml)
    assert "- [x] Done" in result
    assert "- [ ] Not done" in result


# ---------------------------------------------------------------------------
# Blockquotes
# ---------------------------------------------------------------------------

def test_simple_blockquote() -> None:
    xml = "<blockquote><paragraph>Quoted text</paragraph></blockquote>"
    result = tiptap_xml_to_markdown(xml)
    assert "> Quoted text" in result


def test_blockquote_with_multiple_paragraphs() -> None:
    xml = (
        "<blockquote>"
        "<paragraph>First line</paragraph>"
        "<paragraph>Second line</paragraph>"
        "</blockquote>"
    )
    result = tiptap_xml_to_markdown(xml)
    assert "> First line" in result
    assert "> Second line" in result


def test_blockquote_with_marks() -> None:
    xml = "<blockquote><paragraph><strong>Bold</strong> quote</paragraph></blockquote>"
    result = tiptap_xml_to_markdown(xml)
    assert "> **Bold** quote" in result


# ---------------------------------------------------------------------------
# Horizontal Rule
# ---------------------------------------------------------------------------

def test_horizontal_rule() -> None:
    xml = "<paragraph>Above</paragraph><horizontalRule/><paragraph>Below</paragraph>"
    result = tiptap_xml_to_markdown(xml)
    assert "---" in result
    assert "Above" in result
    assert "Below" in result


# ---------------------------------------------------------------------------
# Block IDs (should be ignored)
# ---------------------------------------------------------------------------

def test_block_ids_ignored() -> None:
    xml = '<paragraph data-block-id="abc-123">Content</paragraph>'
    result = tiptap_xml_to_markdown(xml)
    assert "Content" in result
    assert "abc-123" not in result
    assert "data-block-id" not in result


# ---------------------------------------------------------------------------
# Mixed Document
# ---------------------------------------------------------------------------

def test_full_document() -> None:
    """Test a realistic document with mixed block types."""
    xml = (
        '<heading level="1">My Document</heading>'
        "<paragraph>This is the intro with <strong>bold</strong> and <em>italic</em>.</paragraph>"
        '<heading level="2">Section One</heading>'
        '<bulletList>'
        '<listItem listType="bullet"><paragraph>Point A</paragraph></listItem>'
        '<listItem listType="bullet"><paragraph>Point B</paragraph></listItem>'
        '</bulletList>'
        '<codeBlock language="python">x = 42</codeBlock>'
        "<blockquote><paragraph>A wise quote</paragraph></blockquote>"
        "<horizontalRule/>"
        "<paragraph>The end.</paragraph>"
    )
    result = tiptap_xml_to_markdown(xml)
    assert result.startswith("# My Document\n")
    assert "**bold**" in result
    assert "*italic*" in result
    assert "## Section One" in result
    assert "- Point A" in result
    assert "- Point B" in result
    assert "```python" in result
    assert "x = 42" in result
    assert "> A wise quote" in result
    assert "---" in result
    assert "The end." in result


# ---------------------------------------------------------------------------
# Non-ASCII Content
# ---------------------------------------------------------------------------

def test_non_ascii_em_dash() -> None:
    xml = "<paragraph>Hello \u2014 world</paragraph>"
    result = tiptap_xml_to_markdown(xml)
    assert "Hello \u2014 world" in result


def test_emoji() -> None:
    xml = "<paragraph>Hello \U0001f44b world</paragraph>"
    result = tiptap_xml_to_markdown(xml)
    assert "\U0001f44b" in result


def test_cjk_characters() -> None:
    xml = "<paragraph>\u4f60\u597d\u4e16\u754c</paragraph>"
    result = tiptap_xml_to_markdown(xml)
    assert "\u4f60\u597d\u4e16\u754c" in result


# ---------------------------------------------------------------------------
# Float data-indent (pycrdt stores as float)
# ---------------------------------------------------------------------------

def test_list_item_float_indent() -> None:
    xml = (
        '<bulletList>'
        '<listItem listType="bullet" data-indent="1.0"><paragraph>Indented</paragraph></listItem>'
        '</bulletList>'
    )
    result = tiptap_xml_to_markdown(xml)
    assert "  - Indented" in result
