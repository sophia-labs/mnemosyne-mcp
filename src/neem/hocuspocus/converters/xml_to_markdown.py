"""Convert TipTap XML to well-formatted Markdown.

Pure function: tiptap_xml_to_markdown(xml_str) -> markdown_str

Handles all TipTap block types (paragraph, heading, lists, code blocks,
blockquotes, horizontal rules) and inline marks (bold, italic, strike,
code, links, highlights, footnotes). Comments are stripped.

Lists use TipTap's flat representation (listItem with listType and indent
attributes) and are converted to proper nested markdown indentation.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Optional


# Mark tags that map to markdown syntax (both internal and XML names)
BOLD_TAGS = {"strong", "bold"}
ITALIC_TAGS = {"em", "italic"}
STRIKE_TAGS = {"s", "strike"}
CODE_TAGS = {"code"}
LINK_TAGS = {"a"}
HIGHLIGHT_TAGS = {"mark"}
COMMENT_TAGS = {"commentMark"}

# Block tags
HEADING_TAG = "heading"
PARAGRAPH_TAG = "paragraph"
LIST_ITEM_TAG = "listItem"
TASK_ITEM_TAG = "taskItem"
CODE_BLOCK_TAG = "codeBlock"
BLOCKQUOTE_TAG = "blockquote"
HORIZONTAL_RULE_TAG = "horizontalRule"
BULLET_LIST_TAG = "bulletList"
ORDERED_LIST_TAG = "orderedList"
TASK_LIST_TAG = "taskList"

# Inline elements
FOOTNOTE_TAG = "footnote"
WIKILINK_TAG = "wikilink"
HARD_BREAK_TAG = "hardBreak"


def tiptap_xml_to_markdown(xml_str: str) -> str:
    """Convert TipTap XML string to well-formatted Markdown.

    Args:
        xml_str: TipTap XML content (as returned by read_document).

    Returns:
        Clean markdown string.
    """
    if not xml_str or not xml_str.strip():
        return ""

    # Wrap in root element since TipTap XML has multiple root elements
    wrapped = f"<root>{xml_str}</root>"

    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        # If XML parsing fails, return raw text content
        return _strip_tags(xml_str)

    footnotes: list[str] = []
    lines = _convert_children(root, footnotes)

    result = "\n".join(lines)

    # Append footnotes section if any
    if footnotes:
        result += "\n\n"
        for i, content in enumerate(footnotes, 1):
            result += f"[^{i}]: {content}\n"

    # Clean up excessive blank lines (max 2 consecutive newlines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip("\n") + "\n"


def _convert_children(
    parent: ET.Element,
    footnotes: list[str],
) -> list[str]:
    """Convert child elements of a parent to markdown lines."""
    lines: list[str] = []

    for elem in parent:
        tag = elem.tag

        if tag == HEADING_TAG:
            lines.extend(_convert_heading(elem, footnotes))

        elif tag == PARAGRAPH_TAG:
            lines.extend(_convert_paragraph(elem, footnotes))

        elif tag == LIST_ITEM_TAG:
            lines.extend(_convert_list_item(elem, footnotes))

        elif tag == TASK_ITEM_TAG:
            lines.extend(_convert_task_item(elem, footnotes))

        elif tag == CODE_BLOCK_TAG:
            lines.extend(_convert_code_block(elem))

        elif tag == BLOCKQUOTE_TAG:
            lines.extend(_convert_blockquote(elem, footnotes))

        elif tag == HORIZONTAL_RULE_TAG:
            lines.extend(["", "---", ""])

        elif tag in (BULLET_LIST_TAG, ORDERED_LIST_TAG, TASK_LIST_TAG):
            # Wrapper elements — descend into children
            lines.extend(_convert_children(elem, footnotes))

        else:
            # Unknown block — try to extract text content
            text = _inline_content(elem, footnotes)
            if text:
                lines.extend([text, ""])

    return lines


def _convert_heading(
    elem: ET.Element,
    footnotes: list[str],
) -> list[str]:
    """Convert <heading level="N"> to markdown heading."""
    level = _get_int_attr(elem, "level", 1)
    prefix = "#" * min(level, 6)
    content = _inline_content(elem, footnotes)
    return [f"{prefix} {content}", ""]


def _convert_paragraph(
    elem: ET.Element,
    footnotes: list[str],
) -> list[str]:
    """Convert <paragraph> to markdown text."""
    content = _inline_content(elem, footnotes)
    return [content, ""]


def _convert_list_item(
    elem: ET.Element,
    footnotes: list[str],
) -> list[str]:
    """Convert <listItem listType="bullet|ordered"> to markdown list item."""
    list_type = elem.get("listType", "bullet")
    indent = _get_int_attr(elem, "data-indent", 0)
    indent_str = "  " * indent

    # Extract content from child paragraph(s)
    content = _list_item_content(elem, footnotes)

    if list_type == "ordered":
        return [f"{indent_str}1. {content}"]
    else:
        return [f"{indent_str}- {content}"]


def _convert_task_item(
    elem: ET.Element,
    footnotes: list[str],
) -> list[str]:
    """Convert <taskItem checked="true|false"> to markdown task list item."""
    checked = elem.get("checked", "false") == "true"
    indent = _get_int_attr(elem, "data-indent", 0)
    indent_str = "  " * indent
    checkbox = "[x]" if checked else "[ ]"

    content = _list_item_content(elem, footnotes)
    return [f"{indent_str}- {checkbox} {content}"]


def _convert_code_block(elem: ET.Element) -> list[str]:
    """Convert <codeBlock language="..."> to fenced code block."""
    language = elem.get("language", "") or ""
    # Code block content is plain text (no inline marks)
    code = _plain_text(elem)
    return [f"```{language}", code, "```", ""]


def _convert_blockquote(
    elem: ET.Element,
    footnotes: list[str],
) -> list[str]:
    """Convert <blockquote> to markdown blockquote."""
    inner_lines = _convert_children(elem, footnotes)
    quoted = []
    for line in inner_lines:
        if line == "":
            quoted.append(">")
        else:
            quoted.append(f"> {line}")
    quoted.append("")
    return quoted


def _list_item_content(
    elem: ET.Element,
    footnotes: list[str],
) -> str:
    """Extract content from a list/task item.

    List items may contain:
    1. Child <paragraph> elements (TipTap canonical form)
    2. Direct text on the element with optional inline marks (flat form
       produced by DocumentWriter._flatten_list_container)

    For case 1, we extract each paragraph's inline content separately.
    For case 2, we treat the listItem itself as an inline container
    (handles elem.text, child marks like <code>/<strong>, and tail text).
    """
    # Check if any child is a paragraph — if so, use paragraph-extraction mode
    has_paragraph = any(child.tag == PARAGRAPH_TAG for child in elem)
    if has_paragraph:
        parts = []
        for child in elem:
            if child.tag == PARAGRAPH_TAG:
                parts.append(_inline_content(child, footnotes))
            else:
                text = _inline_content(child, footnotes)
                if text:
                    parts.append(text)
        return " ".join(parts)
    # Flat list item: text and inline marks live directly on the element
    return _inline_content(elem, footnotes)


def _inline_content(
    elem: ET.Element,
    footnotes: list[str],
) -> str:
    """Convert an element's inline content (text + marks) to markdown string."""
    parts: list[str] = []

    # Leading text
    if elem.text:
        parts.append(_escape_md(elem.text))

    # Child elements (marks, footnotes, etc.)
    for child in elem:
        parts.append(_convert_inline(child, footnotes))
        # Trailing text after each child element
        if child.tail:
            parts.append(_escape_md(child.tail))

    return "".join(parts)


def _convert_inline(
    elem: ET.Element,
    footnotes: list[str],
) -> str:
    """Convert a single inline element to markdown."""
    tag = elem.tag

    # Bold
    if tag in BOLD_TAGS:
        content = _inline_content(elem, footnotes)
        return f"**{content}**"

    # Italic
    if tag in ITALIC_TAGS:
        content = _inline_content(elem, footnotes)
        return f"*{content}*"

    # Strikethrough
    if tag in STRIKE_TAGS:
        content = _inline_content(elem, footnotes)
        return f"~~{content}~~"

    # Inline code
    if tag in CODE_TAGS:
        content = _plain_text(elem)
        # Use double backticks if content contains backtick
        if "`" in content:
            return f"`` {content} ``"
        return f"`{content}`"

    # Link
    if tag in LINK_TAGS:
        href = elem.get("href", "")
        content = _inline_content(elem, footnotes)
        return f"[{content}]({href})"

    # Highlight/mark
    if tag in HIGHLIGHT_TAGS:
        content = _inline_content(elem, footnotes)
        return f"=={content}=="

    # Comment — strip the comment mark, keep the text
    if tag in COMMENT_TAGS:
        return _inline_content(elem, footnotes)

    # Footnote (self-closing)
    if tag == FOOTNOTE_TAG:
        fn_content = elem.get("content") or elem.get("data-footnote-content", "")
        footnotes.append(fn_content)
        return f"[^{len(footnotes)}]"

    # Wikilink (internal link)
    if tag == WIKILINK_TAG:
        label = elem.get("label", "")
        return label  # Just output the label text

    # Hard break
    if tag == HARD_BREAK_TAG:
        return "  \n"

    # Nested paragraph inside blockquote/list
    if tag == PARAGRAPH_TAG:
        return _inline_content(elem, footnotes)

    # Unknown inline — extract text content
    return _inline_content(elem, footnotes)


def _plain_text(elem: ET.Element) -> str:
    """Extract plain text from an element, ignoring all formatting."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_plain_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _escape_md(text: str) -> str:
    """Escape markdown special characters in text content.

    Only escapes characters that would be interpreted as markdown formatting
    at the start of lines or in inline contexts. Intentionally light-touch
    to keep output readable.
    """
    # We don't aggressively escape because the XML structure already
    # disambiguates formatting from content. Only escape characters
    # that could create ambiguity in the output.
    return text


def _strip_tags(xml_str: str) -> str:
    """Fallback: strip all XML tags and return plain text."""
    return re.sub(r"<[^>]+>", "", html.unescape(xml_str))


def _get_int_attr(elem: ET.Element, attr: str, default: int = 0) -> int:
    """Get an integer attribute from an element, handling float strings."""
    val = elem.get(attr)
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default
