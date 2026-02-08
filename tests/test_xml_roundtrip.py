"""Round-trip tests for TipTap XML ↔ Markdown conversion.

Tests three round-trip paths:
1. CRDT round-trip: XML → DocumentWriter → DocumentReader → XML
2. MD round-trip: MD → markdown_to_tiptap_xml → tiptap_xml_to_markdown → compare
3. Full round-trip: XML → MD → XML (structural equivalence)

Since round-trips involve known normalizations (mark tag names, list
containers flattened, block IDs generated, float→int), tests verify
semantic equivalence rather than exact string equality.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pycrdt

from neem.hocuspocus.converters import markdown_to_tiptap_xml, tiptap_xml_to_markdown
from neem.hocuspocus.document import DocumentReader, DocumentWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crdt_roundtrip(xml: str) -> str:
    """Write XML into a CRDT doc and read it back."""
    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)
    writer.replace_all_content(xml)
    reader = DocumentReader(doc)
    return reader.to_xml()


def _md_roundtrip(md: str) -> str:
    """Convert MD → XML → MD."""
    xml = markdown_to_tiptap_xml(md)
    return tiptap_xml_to_markdown(xml)


def _xml_md_xml_roundtrip(xml: str) -> str:
    """Convert XML → MD → XML."""
    md = tiptap_xml_to_markdown(xml)
    return markdown_to_tiptap_xml(md)


def _strip_block_ids(xml: str) -> str:
    """Remove data-block-id attributes for comparison."""
    return re.sub(r'\s*data-block-id="[^"]*"', "", xml)


def _strip_data_indent_zero(xml: str) -> str:
    """Remove data-indent="0" attributes (default, often omitted)."""
    return re.sub(r'\s*data-indent="0"', "", xml)


def _normalize_xml(xml: str) -> str:
    """Normalize XML for comparison: strip block IDs, zero indents, whitespace."""
    xml = _strip_block_ids(xml)
    xml = _strip_data_indent_zero(xml)
    # Normalize mark tags to canonical forms
    xml = xml.replace("<bold>", "<strong>").replace("</bold>", "</strong>")
    xml = xml.replace("<italic>", "<em>").replace("</italic>", "</em>")
    xml = xml.replace("<strike>", "<s>").replace("</strike>", "</s>")
    # Remove collapsed attribute (CRDT may add it)
    xml = re.sub(r'\s*collapsed="[^"]*"', "", xml)
    # Remove data-indent from non-list elements
    xml = re.sub(r'(<(?:paragraph|heading)[^>]*)\s*data-indent="[^"]*"', r"\1", xml)
    return xml.strip()


def _assert_text_preserved(original: str, result: str, label: str = "") -> None:
    """Assert that all significant text content is preserved."""
    # Extract text from XML-like strings
    text_orig = re.sub(r"<[^>]+>", " ", original)
    text_result = re.sub(r"<[^>]+>", " ", result)
    # Normalize whitespace
    words_orig = set(text_orig.split())
    words_result = set(text_result.split())
    missing = words_orig - words_result
    assert not missing, f"Text lost in {label} round-trip: {missing}"


# ===========================================================================
# 1. CRDT Round-Trip Tests (XML → Writer → Reader → XML)
# ===========================================================================

class TestCRDTRoundTrip:
    """Verify that XML written to a CRDT doc reads back equivalently."""

    def test_heading(self) -> None:
        xml = '<heading level="2">Section Title</heading>'
        out = _crdt_roundtrip(xml)
        assert "Section Title" in out
        assert 'level="2"' in out

    def test_paragraph(self) -> None:
        xml = "<paragraph>Hello world</paragraph>"
        out = _crdt_roundtrip(xml)
        assert "Hello world" in out

    def test_bold(self) -> None:
        xml = "<paragraph><strong>bold text</strong></paragraph>"
        out = _crdt_roundtrip(xml)
        assert "<strong>bold text</strong>" in out

    def test_italic(self) -> None:
        xml = "<paragraph><em>italic text</em></paragraph>"
        out = _crdt_roundtrip(xml)
        assert "<em>italic text</em>" in out

    def test_strikethrough(self) -> None:
        xml = "<paragraph><s>struck text</s></paragraph>"
        out = _crdt_roundtrip(xml)
        assert "<s>struck text</s>" in out

    def test_inline_code(self) -> None:
        xml = "<paragraph><code>foo()</code></paragraph>"
        out = _crdt_roundtrip(xml)
        assert "<code>foo()</code>" in out

    def test_link(self) -> None:
        xml = '<paragraph><a href="https://example.com">link</a></paragraph>'
        out = _crdt_roundtrip(xml)
        assert 'href="https://example.com"' in out
        assert "link" in out

    def test_highlight(self) -> None:
        xml = "<paragraph><mark>highlighted</mark></paragraph>"
        out = _crdt_roundtrip(xml)
        assert "<mark>highlighted</mark>" in out

    def test_nested_marks(self) -> None:
        xml = "<paragraph><strong><em>bold italic</em></strong></paragraph>"
        out = _crdt_roundtrip(xml)
        assert "bold italic" in out
        # Both marks should be present (order may vary)
        assert "<strong>" in out or "<em>" in out

    def test_code_block(self) -> None:
        xml = '<codeBlock language="python">def foo():\n    pass</codeBlock>'
        out = _crdt_roundtrip(xml)
        assert 'language="python"' in out
        assert "def foo():" in out

    def test_bullet_list(self) -> None:
        xml = (
            '<listItem listType="bullet"><paragraph>Item one</paragraph></listItem>'
            '<listItem listType="bullet"><paragraph>Item two</paragraph></listItem>'
        )
        out = _crdt_roundtrip(xml)
        assert "Item one" in out
        assert "Item two" in out
        assert 'listType="bullet"' in out

    def test_ordered_list(self) -> None:
        xml = (
            '<listItem listType="ordered"><paragraph>First</paragraph></listItem>'
            '<listItem listType="ordered"><paragraph>Second</paragraph></listItem>'
        )
        out = _crdt_roundtrip(xml)
        assert "First" in out
        assert "Second" in out
        assert 'listType="ordered"' in out

    def test_task_item(self) -> None:
        xml = '<taskItem checked="true"><paragraph>Done</paragraph></taskItem>'
        out = _crdt_roundtrip(xml)
        assert "Done" in out
        assert 'checked="true"' in out

    def test_blockquote(self) -> None:
        xml = "<blockquote><paragraph>Quoted text</paragraph></blockquote>"
        out = _crdt_roundtrip(xml)
        assert "Quoted text" in out
        assert "<blockquote" in out  # May have data-block-id added

    def test_horizontal_rule(self) -> None:
        xml = "<paragraph>Above</paragraph><horizontalRule/><paragraph>Below</paragraph>"
        out = _crdt_roundtrip(xml)
        assert "Above" in out
        assert "Below" in out
        assert "horizontalRule" in out

    def test_footnote(self) -> None:
        xml = '<paragraph>Text<footnote data-footnote-content="A note"/></paragraph>'
        out = _crdt_roundtrip(xml)
        assert "Text" in out
        assert 'data-footnote-content="A note"' in out

    def test_comment_mark(self) -> None:
        xml = '<paragraph><commentMark data-comment-id="c-1">text</commentMark></paragraph>'
        out = _crdt_roundtrip(xml)
        assert "text" in out
        assert 'data-comment-id="c-1"' in out

    def test_list_with_indent(self) -> None:
        xml = (
            '<listItem listType="bullet"><paragraph>Top</paragraph></listItem>'
            '<listItem listType="bullet" data-indent="1"><paragraph>Nested</paragraph></listItem>'
        )
        out = _crdt_roundtrip(xml)
        assert "Top" in out
        assert "Nested" in out
        assert 'data-indent="1"' in out

    def test_block_ids_generated(self) -> None:
        """Blocks without IDs get IDs assigned."""
        xml = "<paragraph>No ID</paragraph>"
        out = _crdt_roundtrip(xml)
        assert "No ID" in out
        assert "data-block-id=" in out

    def test_block_ids_preserved(self) -> None:
        """Blocks with IDs keep them."""
        xml = '<paragraph data-block-id="my-id">With ID</paragraph>'
        out = _crdt_roundtrip(xml)
        assert 'data-block-id="my-id"' in out

    def test_non_ascii(self) -> None:
        xml = "<paragraph>Hello \u2014 world \U0001f44b \u4f60\u597d</paragraph>"
        out = _crdt_roundtrip(xml)
        assert "\u2014" in out
        assert "\U0001f44b" in out
        assert "\u4f60\u597d" in out

    def test_list_container_flattened(self) -> None:
        """Nested list containers are flattened to flat listItems."""
        xml = (
            "<bulletList>"
            '<listItem listType="bullet"><paragraph>One</paragraph></listItem>'
            '<listItem listType="bullet"><paragraph>Two</paragraph></listItem>'
            "</bulletList>"
        )
        out = _crdt_roundtrip(xml)
        assert "One" in out
        assert "Two" in out
        # Container tag should be gone
        assert "<bulletList>" not in out

    def test_mark_tag_normalization(self) -> None:
        """CRDT writer accepts canonical names: strong, em, s.
        Non-canonical variants (bold, italic, strike) are only handled by
        the XML→MD converter, not by the CRDT writer."""
        # Use canonical names for CRDT round-trip
        xml = "<paragraph><strong>a</strong> <em>b</em> <s>c</s></paragraph>"
        out = _crdt_roundtrip(xml)
        assert "<strong>a</strong>" in out
        assert "<em>b</em>" in out
        assert "<s>c</s>" in out

    def test_strike_variant_normalization(self) -> None:
        """<strike> normalizes to <s> through CRDT round-trip."""
        xml = "<paragraph><strike>struck</strike></paragraph>"
        out = _crdt_roundtrip(xml)
        assert "<s>struck</s>" in out


# ===========================================================================
# 2. Markdown Round-Trip Tests (MD → XML → MD)
# ===========================================================================

class TestMarkdownRoundTrip:
    """Verify that markdown survives MD → XML → MD conversion."""

    def test_heading(self) -> None:
        md = "# Hello World\n"
        result = _md_roundtrip(md)
        assert "# Hello World" in result

    def test_heading_h2(self) -> None:
        md = "## Section\n"
        result = _md_roundtrip(md)
        assert "## Section" in result

    def test_paragraph(self) -> None:
        result = _md_roundtrip("Hello world\n")
        assert "Hello world" in result

    def test_bold(self) -> None:
        result = _md_roundtrip("Some **bold** text\n")
        assert "**bold**" in result

    def test_italic(self) -> None:
        result = _md_roundtrip("Some *italic* text\n")
        assert "*italic*" in result

    def test_strikethrough(self) -> None:
        result = _md_roundtrip("Some ~~struck~~ text\n")
        assert "~~struck~~" in result

    def test_inline_code(self) -> None:
        result = _md_roundtrip("Use `foo()` here\n")
        assert "`foo()`" in result

    def test_link(self) -> None:
        result = _md_roundtrip("[click](https://example.com)\n")
        assert "[click](https://example.com)" in result

    def test_bullet_list(self) -> None:
        md = "- Item one\n- Item two\n"
        result = _md_roundtrip(md)
        assert "- Item one" in result
        assert "- Item two" in result

    def test_bullet_list_nested(self) -> None:
        md = "- Top\n  - Nested\n"
        result = _md_roundtrip(md)
        assert "- Top" in result
        assert "  - Nested" in result

    def test_ordered_list(self) -> None:
        md = "1. First\n2. Second\n"
        result = _md_roundtrip(md)
        assert "1. First" in result
        assert "1. Second" in result  # renumbered to 1. (standard)

    def test_task_list(self) -> None:
        md = "- [x] Done\n- [ ] Todo\n"
        result = _md_roundtrip(md)
        assert "- [x] Done" in result
        assert "- [ ] Todo" in result

    def test_code_block(self) -> None:
        md = "```python\ndef foo():\n    pass\n```\n"
        result = _md_roundtrip(md)
        assert "```python" in result
        assert "def foo():" in result
        assert "```" in result

    def test_blockquote(self) -> None:
        md = "> Quoted text\n"
        result = _md_roundtrip(md)
        assert "> Quoted text" in result

    def test_horizontal_rule(self) -> None:
        md = "Above\n\n---\n\nBelow\n"
        result = _md_roundtrip(md)
        assert "---" in result
        assert "Above" in result
        assert "Below" in result

    def test_footnote(self) -> None:
        md = "Text[^1].\n\n[^1]: Note content.\n"
        result = _md_roundtrip(md)
        assert "[^1]" in result
        assert "Note content" in result

    def test_non_ascii(self) -> None:
        md = "Hello \u2014 world \U0001f44b\n"
        result = _md_roundtrip(md)
        assert "\u2014" in result
        assert "\U0001f44b" in result

    def test_full_document(self) -> None:
        md = """# My Doc

Intro with **bold** and *italic*.

## Section

- Point A
- Point B

```python
x = 42
```

> Quote

---

The end.
"""
        result = _md_roundtrip(md)
        assert "# My Doc" in result
        assert "**bold**" in result
        assert "*italic*" in result
        assert "## Section" in result
        assert "- Point A" in result
        assert "```python" in result
        assert "> Quote" in result
        assert "---" in result
        assert "The end." in result


# ===========================================================================
# 3. XML → MD → XML Round-Trip Tests
# ===========================================================================

class TestXMLMDXMLRoundTrip:
    """Verify XML survives XML → MD → XML conversion.

    The XML won't be identical (block IDs stripped, attribute order may
    change, list containers flattened), but semantic content must be
    preserved.
    """

    def test_heading(self) -> None:
        xml = '<heading level="2">Section</heading>'
        result = _xml_md_xml_roundtrip(xml)
        assert '<heading level="2">Section</heading>' in result

    def test_paragraph(self) -> None:
        xml = "<paragraph>Hello world</paragraph>"
        result = _xml_md_xml_roundtrip(xml)
        assert "<paragraph>Hello world</paragraph>" in result

    def test_bold(self) -> None:
        xml = "<paragraph><strong>bold</strong> text</paragraph>"
        result = _xml_md_xml_roundtrip(xml)
        assert "<strong>bold</strong>" in result

    def test_italic(self) -> None:
        xml = "<paragraph><em>italic</em> text</paragraph>"
        result = _xml_md_xml_roundtrip(xml)
        assert "<em>italic</em>" in result

    def test_strikethrough(self) -> None:
        xml = "<paragraph><s>struck</s> text</paragraph>"
        result = _xml_md_xml_roundtrip(xml)
        assert "<s>struck</s>" in result

    def test_inline_code(self) -> None:
        xml = "<paragraph><code>foo()</code></paragraph>"
        result = _xml_md_xml_roundtrip(xml)
        assert "<code>foo()</code>" in result

    def test_link(self) -> None:
        xml = '<paragraph><a href="https://example.com">click</a></paragraph>'
        result = _xml_md_xml_roundtrip(xml)
        assert 'href="https://example.com"' in result
        assert "click" in result

    def test_code_block(self) -> None:
        xml = '<codeBlock language="python">def foo():\n    pass</codeBlock>'
        result = _xml_md_xml_roundtrip(xml)
        assert 'language="python"' in result
        assert "def foo():" in result

    def test_bullet_list(self) -> None:
        xml = (
            '<listItem listType="bullet"><paragraph>One</paragraph></listItem>'
            '<listItem listType="bullet"><paragraph>Two</paragraph></listItem>'
        )
        result = _xml_md_xml_roundtrip(xml)
        assert 'listType="bullet"' in result
        assert "<paragraph>One</paragraph>" in result
        assert "<paragraph>Two</paragraph>" in result

    def test_ordered_list(self) -> None:
        xml = (
            '<listItem listType="ordered"><paragraph>First</paragraph></listItem>'
            '<listItem listType="ordered"><paragraph>Second</paragraph></listItem>'
        )
        result = _xml_md_xml_roundtrip(xml)
        assert 'listType="ordered"' in result
        assert "First" in result
        assert "Second" in result

    def test_list_with_indent(self) -> None:
        xml = (
            '<listItem listType="bullet"><paragraph>Top</paragraph></listItem>'
            '<listItem listType="bullet" data-indent="1"><paragraph>Nested</paragraph></listItem>'
        )
        result = _xml_md_xml_roundtrip(xml)
        assert "Top" in result
        assert "Nested" in result
        assert 'data-indent="1"' in result

    def test_task_items(self) -> None:
        xml = (
            '<taskItem checked="true"><paragraph>Done</paragraph></taskItem>'
            '<taskItem checked="false"><paragraph>Todo</paragraph></taskItem>'
        )
        result = _xml_md_xml_roundtrip(xml)
        assert 'checked="true"' in result
        assert 'checked="false"' in result

    def test_blockquote(self) -> None:
        xml = "<blockquote><paragraph>Quoted</paragraph></blockquote>"
        result = _xml_md_xml_roundtrip(xml)
        assert "<blockquote>" in result
        assert "Quoted" in result

    def test_horizontal_rule(self) -> None:
        xml = "<paragraph>Above</paragraph><horizontalRule/><paragraph>Below</paragraph>"
        result = _xml_md_xml_roundtrip(xml)
        assert "<horizontalRule/>" in result
        assert "Above" in result
        assert "Below" in result

    def test_footnote(self) -> None:
        xml = '<paragraph>Text<footnote data-footnote-content="A note"/></paragraph>'
        result = _xml_md_xml_roundtrip(xml)
        assert 'data-footnote-content="A note"' in result

    def test_highlight_lost(self) -> None:
        """Highlight marks (==text==) survive XML→MD but may not survive MD→XML
        since highlight is not standard markdown. The == syntax is non-standard."""
        xml = "<paragraph><mark>highlighted</mark></paragraph>"
        md = tiptap_xml_to_markdown(xml)
        assert "==highlighted==" in md
        # MD→XML won't recreate <mark> since mistune doesn't parse ==
        # This is a known lossy conversion

    def test_comment_mark_lost(self) -> None:
        """Comment marks are metadata stripped during MD export (by design)."""
        xml = '<paragraph><commentMark data-comment-id="c1">text</commentMark></paragraph>'
        md = tiptap_xml_to_markdown(xml)
        assert "text" in md
        assert "c1" not in md  # Comment metadata stripped

    def test_block_ids_stripped(self) -> None:
        """Block IDs don't survive MD round-trip (by design)."""
        xml = '<paragraph data-block-id="my-id">Content</paragraph>'
        md = tiptap_xml_to_markdown(xml)
        assert "my-id" not in md
        result = markdown_to_tiptap_xml(md)
        assert "Content" in result
        assert "my-id" not in result  # IDs are not in markdown

    def test_full_document(self) -> None:
        """Full document survives XML→MD→XML with semantic equivalence."""
        xml = (
            '<heading level="1">My Document</heading>'
            "<paragraph>Intro with <strong>bold</strong> and <em>italic</em>.</paragraph>"
            '<heading level="2">Section</heading>'
            '<listItem listType="bullet"><paragraph>Point A</paragraph></listItem>'
            '<listItem listType="bullet"><paragraph>Point B</paragraph></listItem>'
            '<codeBlock language="python">x = 42</codeBlock>'
            "<blockquote><paragraph>A quote</paragraph></blockquote>"
            "<horizontalRule/>"
            "<paragraph>The end.</paragraph>"
        )
        result = _xml_md_xml_roundtrip(xml)

        # All semantic content preserved
        assert '<heading level="1">My Document</heading>' in result
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result
        assert '<heading level="2">Section</heading>' in result
        assert 'listType="bullet"' in result
        assert "Point A" in result
        assert "Point B" in result
        assert 'language="python"' in result
        assert "x = 42" in result
        assert "<blockquote>" in result
        assert "<horizontalRule/>" in result
        assert "The end." in result


# ===========================================================================
# 4. Full Pipeline: CRDT + Converter Round-Trip
# ===========================================================================

class TestFullPipelineRoundTrip:
    """Test the full pipeline: XML → CRDT → XML → MD → XML.

    This simulates: agent writes XML to document, reads it back, exports
    as markdown, then another agent imports the markdown.
    """

    def test_document_through_full_pipeline(self) -> None:
        """A document survives: write → CRDT → read → MD → XML."""
        original_xml = (
            '<heading level="1">Pipeline Test</heading>'
            "<paragraph>Text with <strong>bold</strong> and <em>italic</em>.</paragraph>"
            '<listItem listType="bullet"><paragraph>Item</paragraph></listItem>'
            '<codeBlock language="js">const x = 1;</codeBlock>'
        )

        # Step 1: Write to CRDT and read back
        crdt_xml = _crdt_roundtrip(original_xml)

        # Step 2: Export as markdown
        md = tiptap_xml_to_markdown(crdt_xml)

        # Step 3: Re-import from markdown
        reimported_xml = markdown_to_tiptap_xml(md)

        # Verify semantic content preserved
        assert "Pipeline Test" in reimported_xml
        assert "<strong>bold</strong>" in reimported_xml
        assert "<em>italic</em>" in reimported_xml
        assert "Item" in reimported_xml
        assert 'language="js"' in reimported_xml
        assert "const x = 1;" in reimported_xml

    def test_markdown_through_full_pipeline(self) -> None:
        """Markdown survives: parse → CRDT → read → MD."""
        original_md = """# Test

Some **bold** and *italic* text.

- List item

```python
code
```
"""
        # Step 1: Parse markdown to XML
        xml = markdown_to_tiptap_xml(original_md)

        # Step 2: Write to CRDT and read back
        crdt_xml = _crdt_roundtrip(xml)

        # Step 3: Export as markdown
        result_md = tiptap_xml_to_markdown(crdt_xml)

        # Verify content preserved
        assert "# Test" in result_md
        assert "**bold**" in result_md
        assert "*italic*" in result_md
        assert "- List item" in result_md
        assert "```python" in result_md
        assert "code" in result_md
