"""Tests for Markdown → TipTap XML converter."""

from __future__ import annotations

from neem.hocuspocus.converters import looks_like_markdown, markdown_to_tiptap_xml


# ---------------------------------------------------------------------------
# Empty / Edge Cases
# ---------------------------------------------------------------------------

def test_empty_string() -> None:
    assert markdown_to_tiptap_xml("") == ""


def test_whitespace_only() -> None:
    assert markdown_to_tiptap_xml("   \n  ") == ""


def test_none_input() -> None:
    assert markdown_to_tiptap_xml(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

def test_heading_1() -> None:
    result = markdown_to_tiptap_xml("# Hello")
    assert '<heading level="1">Hello</heading>' in result


def test_heading_2() -> None:
    result = markdown_to_tiptap_xml("## Section")
    assert '<heading level="2">Section</heading>' in result


def test_heading_3() -> None:
    result = markdown_to_tiptap_xml("### Sub")
    assert '<heading level="3">Sub</heading>' in result


def test_heading_with_inline() -> None:
    result = markdown_to_tiptap_xml("# Hello **bold** world")
    assert '<heading level="1">Hello <strong>bold</strong> world</heading>' in result


# ---------------------------------------------------------------------------
# Paragraphs
# ---------------------------------------------------------------------------

def test_simple_paragraph() -> None:
    result = markdown_to_tiptap_xml("Hello world")
    assert "<paragraph>Hello world</paragraph>" in result


def test_multiple_paragraphs() -> None:
    result = markdown_to_tiptap_xml("First\n\nSecond")
    assert "<paragraph>First</paragraph>" in result
    assert "<paragraph>Second</paragraph>" in result


# ---------------------------------------------------------------------------
# Bold
# ---------------------------------------------------------------------------

def test_bold() -> None:
    result = markdown_to_tiptap_xml("Some **bold** text")
    assert "<strong>bold</strong>" in result


def test_bold_underscore() -> None:
    result = markdown_to_tiptap_xml("Some __bold__ text")
    assert "<strong>bold</strong>" in result


# ---------------------------------------------------------------------------
# Italic
# ---------------------------------------------------------------------------

def test_italic() -> None:
    result = markdown_to_tiptap_xml("Some *italic* text")
    assert "<em>italic</em>" in result


def test_italic_underscore() -> None:
    result = markdown_to_tiptap_xml("Some _italic_ text")
    assert "<em>italic</em>" in result


# ---------------------------------------------------------------------------
# Strikethrough
# ---------------------------------------------------------------------------

def test_strikethrough() -> None:
    result = markdown_to_tiptap_xml("Some ~~struck~~ text")
    assert "<s>struck</s>" in result


# ---------------------------------------------------------------------------
# Inline Code
# ---------------------------------------------------------------------------

def test_inline_code() -> None:
    result = markdown_to_tiptap_xml("Use `foo()` here")
    assert "<code>foo()</code>" in result


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def test_link() -> None:
    result = markdown_to_tiptap_xml("[click](https://example.com)")
    assert '<a href="https://example.com">click</a>' in result


def test_link_with_bold() -> None:
    result = markdown_to_tiptap_xml("[**bold link**](https://example.com)")
    assert "<strong>bold link</strong>" in result
    assert 'href="https://example.com"' in result


# ---------------------------------------------------------------------------
# Nested Marks
# ---------------------------------------------------------------------------

def test_bold_italic() -> None:
    result = markdown_to_tiptap_xml("***bold italic***")
    # Could be <strong><em> or <em><strong> depending on parser
    assert "<strong>" in result
    assert "<em>" in result
    assert "bold italic" in result


def test_bold_with_code() -> None:
    result = markdown_to_tiptap_xml("**bold with `code`**")
    assert "<strong>" in result
    assert "<code>code</code>" in result


# ---------------------------------------------------------------------------
# Code Blocks
# ---------------------------------------------------------------------------

def test_code_block_no_language() -> None:
    md = "```\nprint('hello')\n```"
    result = markdown_to_tiptap_xml(md)
    assert "<codeBlock>" in result
    assert "print(&#x27;hello&#x27;)" in result or "print('hello')" in result
    assert "</codeBlock>" in result


def test_code_block_with_language() -> None:
    md = "```python\ndef foo():\n    pass\n```"
    result = markdown_to_tiptap_xml(md)
    assert 'language="python"' in result
    assert "def foo():" in result


def test_code_block_content_not_formatted() -> None:
    """Code block content should be escaped, not parsed for marks."""
    md = "```\n**not bold** *not italic*\n```"
    result = markdown_to_tiptap_xml(md)
    # Content should be escaped text, not wrapped in <strong>/<em>
    assert "<strong>" not in result
    assert "<em>" not in result


# ---------------------------------------------------------------------------
# Bullet Lists
# ---------------------------------------------------------------------------

def test_bullet_list() -> None:
    md = "- Item one\n- Item two"
    result = markdown_to_tiptap_xml(md)
    assert 'listType="bullet"' in result
    assert "<paragraph>Item one</paragraph>" in result
    assert "<paragraph>Item two</paragraph>" in result


def test_bullet_list_nested() -> None:
    md = "- Top\n  - Nested\n    - Deep"
    result = markdown_to_tiptap_xml(md)
    # Top item at indent 0 (no attribute)
    assert '<listItem listType="bullet"><paragraph>Top</paragraph></listItem>' in result
    # Nested at indent 1
    assert 'data-indent="1"' in result
    assert "<paragraph>Nested</paragraph>" in result
    # Deep at indent 2
    assert 'data-indent="2"' in result
    assert "<paragraph>Deep</paragraph>" in result


def test_bullet_list_with_marks() -> None:
    md = "- **Bold** item\n- *Italic* item"
    result = markdown_to_tiptap_xml(md)
    assert "<strong>Bold</strong>" in result
    assert "<em>Italic</em>" in result


# ---------------------------------------------------------------------------
# Ordered Lists
# ---------------------------------------------------------------------------

def test_ordered_list() -> None:
    md = "1. First\n2. Second"
    result = markdown_to_tiptap_xml(md)
    assert 'listType="ordered"' in result
    assert "<paragraph>First</paragraph>" in result
    assert "<paragraph>Second</paragraph>" in result


def test_ordered_list_nested() -> None:
    md = "1. Outer\n   1. Inner"
    result = markdown_to_tiptap_xml(md)
    assert 'listType="ordered"' in result
    assert 'data-indent="1"' in result
    assert "<paragraph>Inner</paragraph>" in result


# ---------------------------------------------------------------------------
# Task Lists
# ---------------------------------------------------------------------------

def test_task_list_checked() -> None:
    md = "- [x] Done"
    result = markdown_to_tiptap_xml(md)
    assert 'checked="true"' in result
    assert "<paragraph>Done</paragraph>" in result


def test_task_list_unchecked() -> None:
    md = "- [ ] Todo"
    result = markdown_to_tiptap_xml(md)
    assert 'checked="false"' in result
    assert "<paragraph>Todo</paragraph>" in result


def test_task_list_mixed() -> None:
    md = "- [x] Done\n- [ ] Todo"
    result = markdown_to_tiptap_xml(md)
    assert 'checked="true"' in result
    assert 'checked="false"' in result


# ---------------------------------------------------------------------------
# Blockquotes
# ---------------------------------------------------------------------------

def test_blockquote() -> None:
    md = "> Quoted text"
    result = markdown_to_tiptap_xml(md)
    assert "<blockquote>" in result
    assert "<paragraph>Quoted text</paragraph>" in result
    assert "</blockquote>" in result


def test_blockquote_multiline() -> None:
    md = "> Line one\n> Line two"
    result = markdown_to_tiptap_xml(md)
    assert "<blockquote>" in result
    assert "Line one" in result
    assert "Line two" in result


def test_blockquote_with_marks() -> None:
    md = "> **Bold** quote"
    result = markdown_to_tiptap_xml(md)
    assert "<blockquote>" in result
    assert "<strong>Bold</strong>" in result


# ---------------------------------------------------------------------------
# Horizontal Rules
# ---------------------------------------------------------------------------

def test_horizontal_rule_dashes() -> None:
    md = "Above\n\n---\n\nBelow"
    result = markdown_to_tiptap_xml(md)
    assert "<horizontalRule/>" in result
    assert "<paragraph>Above</paragraph>" in result
    assert "<paragraph>Below</paragraph>" in result


# ---------------------------------------------------------------------------
# Footnotes
# ---------------------------------------------------------------------------

def test_footnote() -> None:
    md = "Some text[^1].\n\n[^1]: The footnote content."
    result = markdown_to_tiptap_xml(md)
    assert 'data-footnote-content="The footnote content."' in result


def test_multiple_footnotes() -> None:
    md = "First[^1] and second[^2].\n\n[^1]: Note one.\n[^2]: Note two."
    result = markdown_to_tiptap_xml(md)
    assert 'data-footnote-content="Note one."' in result
    assert 'data-footnote-content="Note two."' in result


# ---------------------------------------------------------------------------
# Hard Breaks
# ---------------------------------------------------------------------------

def test_hard_break() -> None:
    md = "Line one  \nLine two"  # Two trailing spaces = hard break
    result = markdown_to_tiptap_xml(md)
    assert "<hardBreak/>" in result
    assert "Line one" in result
    assert "Line two" in result


# ---------------------------------------------------------------------------
# HTML Escaping
# ---------------------------------------------------------------------------

def test_html_entities_escaped() -> None:
    result = markdown_to_tiptap_xml("Use <div> & \"quotes\"")
    assert "&lt;div&gt;" in result
    assert "&amp;" in result


def test_code_block_html_escaped() -> None:
    md = "```\n<div class=\"test\">&amp;</div>\n```"
    result = markdown_to_tiptap_xml(md)
    assert "&lt;div" in result


# ---------------------------------------------------------------------------
# Non-ASCII
# ---------------------------------------------------------------------------

def test_non_ascii_em_dash() -> None:
    result = markdown_to_tiptap_xml("Hello \u2014 world")
    assert "\u2014" in result


def test_emoji() -> None:
    result = markdown_to_tiptap_xml("Hello \U0001f44b")
    assert "\U0001f44b" in result


# ---------------------------------------------------------------------------
# Full Document
# ---------------------------------------------------------------------------

def test_full_document() -> None:
    md = """# My Document

This is the intro with **bold** and *italic*.

## Section One

- Point A
- Point B

```python
x = 42
```

> A wise quote

---

The end.
"""
    result = markdown_to_tiptap_xml(md)
    assert '<heading level="1">My Document</heading>' in result
    assert "<strong>bold</strong>" in result
    assert "<em>italic</em>" in result
    assert '<heading level="2">Section One</heading>' in result
    assert 'listType="bullet"' in result
    assert "<paragraph>Point A</paragraph>" in result
    assert 'language="python"' in result
    assert "x = 42" in result
    assert "<blockquote>" in result
    assert "<horizontalRule/>" in result
    assert "<paragraph>The end.</paragraph>" in result


# ---------------------------------------------------------------------------
# Markdown Detection Heuristic: looks_like_markdown()
# ---------------------------------------------------------------------------

class TestLooksLikeMarkdown:

    def test_plain_text_no(self) -> None:
        assert looks_like_markdown("Hello world") is False

    def test_plain_text_with_period(self) -> None:
        assert looks_like_markdown("This is a sentence.") is False

    def test_heading_yes(self) -> None:
        assert looks_like_markdown("# Heading") is True

    def test_heading_h2(self) -> None:
        assert looks_like_markdown("## Section") is True

    def test_bold_yes(self) -> None:
        assert looks_like_markdown("Some **bold** text") is True

    def test_italic_yes(self) -> None:
        assert looks_like_markdown("Some *italic* text") is True

    def test_bullet_list_yes(self) -> None:
        assert looks_like_markdown("- item one\n- item two") is True

    def test_ordered_list_yes(self) -> None:
        assert looks_like_markdown("1. first\n2. second") is True

    def test_code_fence_yes(self) -> None:
        assert looks_like_markdown("```python\ncode\n```") is True

    def test_link_yes(self) -> None:
        assert looks_like_markdown("See [this](https://example.com)") is True

    def test_blockquote_yes(self) -> None:
        assert looks_like_markdown("> quoted text") is True

    def test_horizontal_rule_yes(self) -> None:
        assert looks_like_markdown("---") is True

    def test_task_list_yes(self) -> None:
        assert looks_like_markdown("- [x] done task") is True

    def test_strikethrough_yes(self) -> None:
        assert looks_like_markdown("~~struck~~") is True

    def test_xml_no(self) -> None:
        assert looks_like_markdown("<paragraph>hello</paragraph>") is False

    def test_empty_no(self) -> None:
        assert looks_like_markdown("") is False

    def test_number_with_period_not_list(self) -> None:
        """A number followed by period mid-sentence is not a list."""
        # This pattern requires start-of-line
        assert looks_like_markdown("I have 3. things") is False

    def test_asterisk_in_math_not_bold(self) -> None:
        """Single asterisk with space after is not markdown."""
        # Our pattern requires non-space after **
        assert looks_like_markdown("a * b = c") is False
