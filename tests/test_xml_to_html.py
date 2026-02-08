"""Tests for TipTap XML → HTML converter with Garden theming.

Tests verify:
1. All block types produce correct semantic HTML
2. Inline marks produce correct HTML elements
3. Flat lists are reconstructed into proper nested HTML
4. Garden CSS is included when themed=True
5. Dark mode support via prefers-color-scheme
6. Footnotes render as linked section
7. Edge cases (empty input, unknown tags, etc.)
"""

import pytest

from neem.hocuspocus.converters.xml_to_html import (
    tiptap_xml_to_html,
    GARDEN_CSS,
    MINIMAL_CSS,
)


# ==================================================================
# Document structure
# ==================================================================


class TestDocumentStructure:
    def test_full_document_wrapper(self):
        html = tiptap_xml_to_html("<paragraph>Hello</paragraph>")
        assert html.startswith("<!DOCTYPE html>")
        assert "<html lang=" in html
        assert "<meta charset=" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "</html>" in html

    def test_title_in_head(self):
        html = tiptap_xml_to_html("<paragraph>Text</paragraph>", title="My Doc")
        assert "<title>My Doc</title>" in html

    def test_title_escaped(self):
        html = tiptap_xml_to_html("<paragraph>x</paragraph>", title='A "B" & <C>')
        assert "A &quot;B&quot; &amp; &lt;C&gt;" in html

    def test_default_title_untitled(self):
        html = tiptap_xml_to_html("<paragraph>Text</paragraph>")
        assert "<title>Untitled</title>" in html

    def test_themed_includes_garden_css(self):
        html = tiptap_xml_to_html("<paragraph>Text</paragraph>", themed=True)
        assert "Literata" in html
        assert "--page:" in html
        assert "--fern:" in html
        assert "prefers-color-scheme: dark" in html

    def test_unthemed_uses_minimal_css(self):
        html = tiptap_xml_to_html("<paragraph>Text</paragraph>", themed=False)
        assert "system-ui" in html
        assert "Literata" not in html

    def test_fragment_mode(self):
        html = tiptap_xml_to_html(
            "<paragraph>Hello</paragraph>", full_document=False
        )
        assert "<!DOCTYPE" not in html
        assert "<p>Hello</p>" in html

    def test_empty_input(self):
        html = tiptap_xml_to_html("")
        assert "<!DOCTYPE html>" in html
        assert "<body>" in html

    def test_whitespace_only(self):
        html = tiptap_xml_to_html("   \n  ")
        assert "<!DOCTYPE html>" in html

    def test_viewport_meta(self):
        html = tiptap_xml_to_html("<paragraph>x</paragraph>")
        assert 'name="viewport"' in html


# ==================================================================
# Block types
# ==================================================================


class TestHeadings:
    def test_h1(self):
        html = tiptap_xml_to_html(
            '<heading level="1">Title</heading>', full_document=False
        )
        assert "<h1>Title</h1>" in html

    def test_h2(self):
        html = tiptap_xml_to_html(
            '<heading level="2">Sub</heading>', full_document=False
        )
        assert "<h2>Sub</h2>" in html

    def test_h3(self):
        html = tiptap_xml_to_html(
            '<heading level="3">Sub</heading>', full_document=False
        )
        assert "<h3>Sub</h3>" in html

    def test_h6(self):
        html = tiptap_xml_to_html(
            '<heading level="6">Small</heading>', full_document=False
        )
        assert "<h6>Small</h6>" in html

    def test_heading_with_formatting(self):
        html = tiptap_xml_to_html(
            '<heading level="2"><strong>Bold</strong> heading</heading>',
            full_document=False,
        )
        assert "<h2><strong>Bold</strong> heading</h2>" in html


class TestParagraphs:
    def test_simple_paragraph(self):
        html = tiptap_xml_to_html(
            "<paragraph>Hello world.</paragraph>", full_document=False
        )
        assert "<p>Hello world.</p>" in html

    def test_multiple_paragraphs(self):
        xml = "<paragraph>First.</paragraph><paragraph>Second.</paragraph>"
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "<p>First.</p>" in html
        assert "<p>Second.</p>" in html

    def test_paragraph_escapes_html(self):
        html = tiptap_xml_to_html(
            "<paragraph>A &amp; B &lt; C</paragraph>", full_document=False
        )
        assert "<p>A &amp; B &lt; C</p>" in html


class TestCodeBlocks:
    def test_code_block_with_language(self):
        html = tiptap_xml_to_html(
            '<codeBlock language="python">def hello():\n    pass</codeBlock>',
            full_document=False,
        )
        assert '<pre><code class="language-python">' in html
        assert "def hello():" in html
        assert "</code></pre>" in html

    def test_code_block_no_language(self):
        html = tiptap_xml_to_html(
            "<codeBlock>some code</codeBlock>", full_document=False
        )
        assert "<pre><code>" in html
        assert "some code" in html

    def test_code_block_escapes_content(self):
        html = tiptap_xml_to_html(
            "<codeBlock>x &lt; y &amp; z</codeBlock>", full_document=False
        )
        assert "&lt;" in html
        assert "&amp;" in html


class TestBlockquotes:
    def test_simple_blockquote(self):
        html = tiptap_xml_to_html(
            "<blockquote><paragraph>Quoted.</paragraph></blockquote>",
            full_document=False,
        )
        assert "<blockquote>" in html
        assert "<p>Quoted.</p>" in html
        assert "</blockquote>" in html

    def test_multi_paragraph_blockquote(self):
        xml = "<blockquote><paragraph>A</paragraph><paragraph>B</paragraph></blockquote>"
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "<p>A</p>" in html
        assert "<p>B</p>" in html


class TestHorizontalRule:
    def test_horizontal_rule(self):
        xml = "<paragraph>Above</paragraph><horizontalRule/><paragraph>Below</paragraph>"
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "<hr" in html
        assert "<p>Above</p>" in html
        assert "<p>Below</p>" in html


# ==================================================================
# Inline marks
# ==================================================================


class TestInlineMarks:
    def test_bold(self):
        html = tiptap_xml_to_html(
            "<paragraph>A <strong>bold</strong> word.</paragraph>",
            full_document=False,
        )
        assert "<strong>bold</strong>" in html

    def test_bold_alias(self):
        html = tiptap_xml_to_html(
            "<paragraph><bold>text</bold></paragraph>", full_document=False
        )
        assert "<strong>text</strong>" in html

    def test_italic(self):
        html = tiptap_xml_to_html(
            "<paragraph><em>italic</em></paragraph>", full_document=False
        )
        assert "<em>italic</em>" in html

    def test_italic_alias(self):
        html = tiptap_xml_to_html(
            "<paragraph><italic>text</italic></paragraph>", full_document=False
        )
        assert "<em>text</em>" in html

    def test_strike(self):
        html = tiptap_xml_to_html(
            "<paragraph><s>struck</s></paragraph>", full_document=False
        )
        assert "<s>struck</s>" in html

    def test_strike_alias(self):
        html = tiptap_xml_to_html(
            "<paragraph><strike>text</strike></paragraph>", full_document=False
        )
        assert "<s>text</s>" in html

    def test_inline_code(self):
        html = tiptap_xml_to_html(
            "<paragraph>Use <code>var</code> here.</paragraph>",
            full_document=False,
        )
        assert "<code>var</code>" in html

    def test_link(self):
        html = tiptap_xml_to_html(
            '<paragraph><a href="https://example.com">click</a></paragraph>',
            full_document=False,
        )
        assert '<a href="https://example.com">click</a>' in html

    def test_highlight(self):
        html = tiptap_xml_to_html(
            "<paragraph><mark>highlighted</mark></paragraph>",
            full_document=False,
        )
        assert "<mark>highlighted</mark>" in html

    def test_nested_marks(self):
        html = tiptap_xml_to_html(
            "<paragraph><strong><em>bold italic</em></strong></paragraph>",
            full_document=False,
        )
        assert "<strong><em>bold italic</em></strong>" in html

    def test_comment_mark_stripped(self):
        html = tiptap_xml_to_html(
            '<paragraph><commentMark data-comment-id="c1">text</commentMark></paragraph>',
            full_document=False,
        )
        assert "commentMark" not in html
        assert "text" in html

    def test_hard_break(self):
        html = tiptap_xml_to_html(
            "<paragraph>Line 1<hardBreak/>Line 2</paragraph>",
            full_document=False,
        )
        assert "<br />" in html
        assert "Line 1" in html
        assert "Line 2" in html


# ==================================================================
# Lists (flat → nested reconstruction)
# ==================================================================


class TestFlatLists:
    def test_simple_bullet_list(self):
        xml = (
            '<listItem listType="bullet"><paragraph>A</paragraph></listItem>'
            '<listItem listType="bullet"><paragraph>B</paragraph></listItem>'
        )
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "<ul>" in html
        assert "<li>A</li>" in html
        assert "<li>B</li>" in html
        assert "</ul>" in html

    def test_simple_ordered_list(self):
        xml = (
            '<listItem listType="ordered"><paragraph>First</paragraph></listItem>'
            '<listItem listType="ordered"><paragraph>Second</paragraph></listItem>'
        )
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "<ol>" in html
        assert "<li>First</li>" in html
        assert "<li>Second</li>" in html
        assert "</ol>" in html

    def test_nested_list(self):
        xml = (
            '<listItem listType="bullet"><paragraph>Parent</paragraph></listItem>'
            '<listItem listType="bullet" data-indent="1"><paragraph>Child</paragraph></listItem>'
        )
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "<ul>" in html
        assert "<li>Parent</li>" in html
        assert "<li>Child</li>" in html
        # Should have nested ul
        assert html.count("<ul>") >= 2

    def test_task_list(self):
        xml = (
            '<taskItem checked="false"><paragraph>Todo</paragraph></taskItem>'
            '<taskItem checked="true"><paragraph>Done</paragraph></taskItem>'
        )
        html = tiptap_xml_to_html(xml, full_document=False)
        assert 'type="checkbox"' in html
        assert "disabled" in html
        assert "checked" in html
        assert "Todo" in html
        assert "Done" in html

    def test_mixed_list_types_nested(self):
        xml = (
            '<listItem listType="bullet"><paragraph>Bullet</paragraph></listItem>'
            '<listItem listType="ordered" data-indent="1"><paragraph>Ordered child</paragraph></listItem>'
        )
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "<ul>" in html
        assert "<ol>" in html
        assert "Bullet" in html
        assert "Ordered child" in html

    def test_list_item_with_formatting(self):
        xml = '<listItem listType="bullet"><paragraph>A <strong>bold</strong> item</paragraph></listItem>'
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "<strong>bold</strong>" in html


class TestContainerLists:
    """Test lists wrapped in bulletList/orderedList container elements."""

    def test_bullet_list_container(self):
        xml = '<bulletList><listItem><paragraph>A</paragraph></listItem><listItem><paragraph>B</paragraph></listItem></bulletList>'
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "<ul>" in html
        assert "<li>A</li>" in html
        assert "<li>B</li>" in html

    def test_ordered_list_container(self):
        xml = '<orderedList><listItem><paragraph>One</paragraph></listItem></orderedList>'
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "<ol>" in html
        assert "<li>One</li>" in html


# ==================================================================
# Footnotes
# ==================================================================


class TestFootnotes:
    def test_footnote_reference(self):
        xml = '<paragraph>Text<footnote data-footnote-content="A note."/></paragraph>'
        html = tiptap_xml_to_html(xml, full_document=False)
        assert 'class="fn-ref"' in html
        assert 'href="#fn-1"' in html
        assert "[1]" in html

    def test_footnote_section(self):
        xml = '<paragraph>Text<footnote data-footnote-content="A note."/></paragraph>'
        html = tiptap_xml_to_html(xml, full_document=False)
        assert 'class="footnotes"' in html
        assert "A note." in html
        assert 'id="fn-1"' in html

    def test_multiple_footnotes(self):
        xml = (
            '<paragraph>'
            'First<footnote data-footnote-content="Note 1."/> '
            'second<footnote data-footnote-content="Note 2."/>'
            '</paragraph>'
        )
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "[1]" in html
        assert "[2]" in html
        assert "Note 1." in html
        assert "Note 2." in html

    def test_footnote_backref(self):
        xml = '<paragraph>Text<footnote data-footnote-content="Note."/></paragraph>'
        html = tiptap_xml_to_html(xml, full_document=False)
        # Footnote section should have back-reference link
        assert "\u21a9" in html  # ↩ character
        assert 'href="#fnref-1"' in html


# ==================================================================
# Block IDs
# ==================================================================


class TestBlockIds:
    def test_block_ids_excluded_by_default(self):
        xml = '<paragraph data-block-id="abc123">Text</paragraph>'
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "data-block-id" not in html

    def test_block_ids_included_when_requested(self):
        xml = '<paragraph data-block-id="abc123">Text</paragraph>'
        html = tiptap_xml_to_html(xml, full_document=False, include_block_ids=True)
        assert 'data-block-id="abc123"' in html


# ==================================================================
# Edge cases
# ==================================================================


class TestEdgeCases:
    def test_non_ascii_content(self):
        html = tiptap_xml_to_html(
            "<paragraph>Überschrift — em dash</paragraph>",
            full_document=False,
        )
        assert "Überschrift" in html
        assert "—" in html

    def test_emoji(self):
        html = tiptap_xml_to_html(
            "<paragraph>Hello 🌍</paragraph>", full_document=False
        )
        assert "🌍" in html

    def test_invalid_xml_fallback(self):
        html = tiptap_xml_to_html("not <valid> xml <here", full_document=False)
        # Should still produce some output
        assert len(html) > 0

    def test_unknown_block_as_paragraph(self):
        html = tiptap_xml_to_html(
            "<customBlock>Content</customBlock>", full_document=False
        )
        assert "Content" in html

    def test_wikilink(self):
        html = tiptap_xml_to_html(
            '<paragraph><wikilink label="My Page"/></paragraph>',
            full_document=False,
        )
        assert "My Page" in html


# ==================================================================
# Complex documents
# ==================================================================


class TestComplexDocuments:
    def test_full_document(self):
        xml = (
            '<heading level="1">Document Title</heading>'
            "<paragraph>An intro with <strong>bold</strong> and <em>italic</em>.</paragraph>"
            '<listItem listType="bullet"><paragraph>Item one</paragraph></listItem>'
            '<listItem listType="bullet"><paragraph>Item two</paragraph></listItem>'
            '<codeBlock language="python">x = 42</codeBlock>'
            "<blockquote><paragraph>A wise quote.</paragraph></blockquote>"
            "<horizontalRule/>"
            "<paragraph>The end.</paragraph>"
        )
        html = tiptap_xml_to_html(xml, title="Test Doc")
        assert "<title>Test Doc</title>" in html
        assert "<h1>Document Title</h1>" in html
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html
        assert "<ul>" in html
        assert "<li>Item one</li>" in html
        assert 'class="language-python"' in html
        assert "<blockquote>" in html
        assert "<hr" in html
        assert "<p>The end.</p>" in html

    def test_mixed_content_between_lists(self):
        """Lists separated by paragraphs should be separate list groups."""
        xml = (
            '<listItem listType="bullet"><paragraph>A</paragraph></listItem>'
            "<paragraph>Break</paragraph>"
            '<listItem listType="ordered"><paragraph>B</paragraph></listItem>'
        )
        html = tiptap_xml_to_html(xml, full_document=False)
        assert "<ul>" in html
        assert "<ol>" in html
        assert "<p>Break</p>" in html
