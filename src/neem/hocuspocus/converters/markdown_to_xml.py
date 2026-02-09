"""Convert Markdown to TipTap XML.

Pure function: markdown_to_tiptap_xml(md_str) -> xml_str

Uses mistune 3.x in AST mode to parse markdown, then walks the AST to
emit TipTap XML. No HTML intermediate step.

Handles: headings, paragraphs, lists (bullet/ordered/task with nesting),
code blocks, blockquotes, horizontal rules, and inline marks (bold, italic,
strikethrough, inline code, links). Footnotes are converted to TipTap's
self-closing <footnote/> elements.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

import mistune


# Singleton parser with plugins enabled
_md_parser = mistune.create_markdown(
    renderer="ast",
    plugins=["strikethrough", "task_lists", "footnotes"],
)


def markdown_to_tiptap_xml(md_str: str) -> str:
    """Convert a Markdown string to TipTap XML.

    Args:
        md_str: Markdown content.

    Returns:
        TipTap XML string suitable for write_document / update_block.
    """
    if not md_str or not md_str.strip():
        return ""

    ast = _md_parser(md_str)
    if not ast:
        return ""

    # Collect footnote definitions for reference
    footnotes = _collect_footnotes(ast)

    parts: list[str] = []
    for node in ast:
        xml = _convert_block(node, footnotes)
        if xml:
            parts.append(xml)

    return "".join(parts)


# ---------------------------------------------------------------------------
# Footnote collection
# ---------------------------------------------------------------------------

def _collect_footnotes(ast: list[dict]) -> dict[str, str]:
    """Extract footnote definitions from the AST into a key→content map."""
    footnotes: dict[str, str] = {}
    for node in ast:
        if node.get("type") == "footnotes":
            for item in node.get("children", []):
                if item.get("type") == "footnote_item":
                    key = str(item.get("attrs", {}).get("key", ""))
                    # Render footnote body as plain text
                    content_parts: list[str] = []
                    for child in item.get("children", []):
                        content_parts.append(_plain_text_from_node(child))
                    footnotes[key] = " ".join(content_parts).strip()
    return footnotes


def _plain_text_from_node(node: dict) -> str:
    """Extract plain text from an AST node, ignoring formatting."""
    ntype = node.get("type", "")
    if ntype == "text":
        return node.get("raw", "")
    if ntype in ("softbreak", "linebreak"):
        return " "
    children = node.get("children", [])
    return "".join(_plain_text_from_node(c) for c in children)


# ---------------------------------------------------------------------------
# Block-level conversion
# ---------------------------------------------------------------------------

def _convert_block(node: dict, footnotes: dict[str, str]) -> str:
    """Convert a block-level AST node to TipTap XML."""
    ntype = node.get("type", "")

    if ntype == "heading":
        return _convert_heading(node, footnotes)

    if ntype == "paragraph":
        return _convert_paragraph(node, footnotes)

    if ntype == "list":
        return _convert_list(node, footnotes)

    if ntype == "block_code":
        return _convert_code_block(node)

    if ntype == "block_quote":
        return _convert_blockquote(node, footnotes)

    if ntype == "thematic_break":
        return "<horizontalRule/>"

    if ntype in ("blank_line", "footnotes"):
        return ""

    # Unknown block — try to extract as paragraph
    children = node.get("children")
    if children:
        inline = _convert_inline_children(children, footnotes)
        if inline:
            return f"<paragraph>{inline}</paragraph>"

    return ""


def _convert_heading(node: dict, footnotes: dict[str, str]) -> str:
    level = node.get("attrs", {}).get("level", 1)
    content = _convert_inline_children(node.get("children", []), footnotes)
    return f'<heading level="{level}">{content}</heading>'


def _convert_paragraph(node: dict, footnotes: dict[str, str]) -> str:
    children = node.get("children", [])
    # If the paragraph contains only a single image, lift it out as a block-level image
    if len(children) == 1 and children[0].get("type") == "image":
        img = children[0]
        src = html.escape(img.get("attrs", {}).get("url", img.get("destination", "")))
        alt = html.escape(_plain_text_from_node(img))
        return f'<image src="{src}" alt="{alt}"/>'
    content = _convert_inline_children(children, footnotes)
    return f"<paragraph>{content}</paragraph>"


def _convert_code_block(node: dict) -> str:
    info = node.get("attrs", {}).get("info", "") or ""
    # Strip any extra info string tokens (e.g. "python title=foo")
    language = info.split()[0] if info else ""
    raw = node.get("raw", "")
    # Remove trailing newline that mistune adds
    if raw.endswith("\n"):
        raw = raw[:-1]
    escaped = html.escape(raw)
    if language:
        return f'<codeBlock language="{html.escape(language)}">{escaped}</codeBlock>'
    return f"<codeBlock>{escaped}</codeBlock>"


def _convert_blockquote(node: dict, footnotes: dict[str, str]) -> str:
    parts: list[str] = []
    for child in node.get("children", []):
        xml = _convert_block(child, footnotes)
        if xml:
            parts.append(xml)
    return f'<blockquote>{"".join(parts)}</blockquote>'


# ---------------------------------------------------------------------------
# List flattening
# ---------------------------------------------------------------------------

def _convert_list(node: dict, footnotes: dict[str, str], base_indent: int = 0) -> str:
    """Flatten a nested list into TipTap's flat listItem representation."""
    ordered = node.get("attrs", {}).get("ordered", False)
    items = _flatten_list(node, footnotes, base_indent, ordered)
    return "".join(items)


def _flatten_list(
    node: dict,
    footnotes: dict[str, str],
    indent: int,
    ordered: bool,
) -> list[str]:
    """Recursively flatten a list node into flat listItem XML strings."""
    items: list[str] = []

    for child in node.get("children", []):
        ctype = child.get("type", "")

        if ctype == "task_list_item":
            items.extend(_flatten_task_item(child, footnotes, indent))
        elif ctype == "list_item":
            items.extend(_flatten_list_item(child, footnotes, indent, ordered))

    return items


def _flatten_list_item(
    node: dict,
    footnotes: dict[str, str],
    indent: int,
    ordered: bool,
) -> list[str]:
    """Convert a single list_item (possibly with nested lists) to flat items."""
    items: list[str] = []
    list_type = "ordered" if ordered else "bullet"

    # Separate inline content from nested lists
    inline_children: list[dict] = []
    nested_lists: list[dict] = []

    for child in node.get("children", []):
        ctype = child.get("type", "")
        if ctype == "list":
            nested_lists.append(child)
        elif ctype == "block_text":
            # block_text contains inline children
            inline_children.extend(child.get("children", []))
        elif ctype == "paragraph":
            inline_children.extend(child.get("children", []))
        else:
            inline_children.append(child)

    # Emit the list item with inline content
    content = _convert_inline_children(inline_children, footnotes) if inline_children else ""
    indent_attr = f' data-indent="{indent}"' if indent > 0 else ""
    items.append(
        f'<listItem listType="{list_type}"{indent_attr}>'
        f"<paragraph>{content}</paragraph>"
        f"</listItem>"
    )

    # Recursively flatten nested lists at indent+1
    for nested in nested_lists:
        nested_ordered = nested.get("attrs", {}).get("ordered", False)
        items.extend(_flatten_list(nested, footnotes, indent + 1, nested_ordered))

    return items


def _flatten_task_item(
    node: dict,
    footnotes: dict[str, str],
    indent: int,
) -> list[str]:
    """Convert a task_list_item to a TipTap taskItem."""
    checked = node.get("attrs", {}).get("checked", False)
    checked_str = "true" if checked else "false"

    inline_children: list[dict] = []
    for child in node.get("children", []):
        ctype = child.get("type", "")
        if ctype == "block_text":
            inline_children.extend(child.get("children", []))
        elif ctype == "paragraph":
            inline_children.extend(child.get("children", []))
        else:
            inline_children.append(child)

    content = _convert_inline_children(inline_children, footnotes) if inline_children else ""
    indent_attr = f' data-indent="{indent}"' if indent > 0 else ""
    return [
        f'<taskItem checked="{checked_str}"{indent_attr}>'
        f"<paragraph>{content}</paragraph>"
        f"</taskItem>"
    ]


# ---------------------------------------------------------------------------
# Inline conversion
# ---------------------------------------------------------------------------

def _convert_inline_children(
    children: list[dict],
    footnotes: dict[str, str],
) -> str:
    """Convert a list of inline AST nodes to TipTap XML inline content."""
    parts: list[str] = []
    for child in children:
        parts.append(_convert_inline(child, footnotes))
    return "".join(parts)


def _convert_inline(node: dict, footnotes: dict[str, str]) -> str:
    """Convert a single inline AST node to TipTap XML."""
    ntype = node.get("type", "")

    if ntype == "text":
        return html.escape(node.get("raw", ""))

    if ntype == "strong":
        content = _convert_inline_children(node.get("children", []), footnotes)
        return f"<strong>{content}</strong>"

    if ntype == "emphasis":
        content = _convert_inline_children(node.get("children", []), footnotes)
        return f"<em>{content}</em>"

    if ntype == "strikethrough":
        content = _convert_inline_children(node.get("children", []), footnotes)
        return f"<s>{content}</s>"

    if ntype == "codespan":
        raw = node.get("raw", "")
        return f"<code>{html.escape(raw)}</code>"

    if ntype == "link":
        url = node.get("attrs", {}).get("url", "")
        content = _convert_inline_children(node.get("children", []), footnotes)
        return f'<a href="{html.escape(url)}">{content}</a>'

    if ntype == "image":
        src = html.escape(node.get("attrs", {}).get("url", node.get("destination", "")))
        alt = html.escape(_plain_text_from_node(node))
        return f'<image src="{src}" alt="{alt}"/>'

    if ntype == "softbreak":
        # Soft line break within a paragraph — TipTap ignores these
        return " "

    if ntype == "linebreak":
        return "<hardBreak/>"

    if ntype == "footnote_ref":
        key = str(node.get("raw", ""))
        fn_content = footnotes.get(key, "")
        return f'<footnote data-footnote-content="{html.escape(fn_content)}"/>'

    # Unknown inline — try to extract text
    children = node.get("children")
    if children:
        return _convert_inline_children(children, footnotes)

    raw = node.get("raw", "")
    if raw:
        return html.escape(raw)

    return ""


# ---------------------------------------------------------------------------
# Markdown detection heuristic
# ---------------------------------------------------------------------------

# Patterns that unambiguously indicate markdown (not plain text)
_MARKDOWN_PATTERNS = [
    # ATX headings
    r"^#{1,6}\s",
    # Bold/italic (must have non-space after opener)
    r"\*\*\S",
    r"\*\S",
    # Unordered list items at start of line
    r"^\s*[-*+]\s",
    # Ordered list items at start of line
    r"^\s*\d+\.\s",
    # Fenced code blocks
    r"^```",
    # Links [text](url)
    r"\[.+?\]\(.+?\)",
    # Blockquotes
    r"^>\s",
    # Horizontal rules (3+ dashes/asterisks/underscores alone on line)
    r"^-{3,}\s*$",
    r"^\*{3,}\s*$",
    r"^_{3,}\s*$",
    # Task lists
    r"^\s*[-*+]\s+\[[ xX]\]",
    # Strikethrough
    r"~~\S",
]

import re

_MARKDOWN_RE = [re.compile(p, re.MULTILINE) for p in _MARKDOWN_PATTERNS]


def looks_like_markdown(text: str) -> bool:
    """Conservative heuristic: returns True only if the text contains
    unambiguous markdown patterns.

    This is used by _ensure_xml() to decide whether to parse markdown
    or treat the input as plain text. False negatives (markdown treated
    as plain text) are safe — the text just gets wrapped in <paragraph>.
    False positives (plain text parsed as markdown) would mangle content,
    so we err on the side of caution.
    """
    if not text or not text.strip():
        return False

    # If it already looks like XML, it's not markdown
    if text.strip().startswith("<"):
        return False

    for pattern in _MARKDOWN_RE:
        if pattern.search(text):
            return True

    return False
