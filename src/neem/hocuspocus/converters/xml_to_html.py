"""Convert TipTap XML to self-contained HTML with Garden theming.

Pure function: tiptap_xml_to_html(xml_str, title, themed) -> html_str

Produces semantic HTML5 output with optional Garden-themed CSS that
includes the full color palette, serif typography, and dark/light mode
support via prefers-color-scheme. The output is a fully self-contained
HTML document with no external dependencies (except Google Fonts).

Lists use TipTap's flat representation (listItem with listType and indent)
and are reconstructed into proper nested <ul>/<ol>/<li> HTML structure.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Optional


# ---------------------------------------------------------------------------
# Tag constants (shared with xml_to_markdown)
# ---------------------------------------------------------------------------

BOLD_TAGS = {"strong", "bold"}
ITALIC_TAGS = {"em", "italic"}
STRIKE_TAGS = {"s", "strike"}
CODE_TAGS = {"code"}
LINK_TAGS = {"a"}
HIGHLIGHT_TAGS = {"mark"}
COMMENT_TAGS = {"commentMark"}

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
TABLE_TAG = "table"
TABLE_ROW_TAG = "tableRow"
TABLE_HEADER_TAG = "tableHeader"
TABLE_CELL_TAG = "tableCell"

FOOTNOTE_TAG = "footnote"
WIKILINK_TAG = "wikilink"
HARD_BREAK_TAG = "hardBreak"


# ---------------------------------------------------------------------------
# Garden Theme CSS
# ---------------------------------------------------------------------------

GARDEN_CSS = """\
/* Garden — The Living Codex (exported document theme) */
@import url('https://fonts.googleapis.com/css2?family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,500;0,7..72,600;1,7..72,400;1,7..72,500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --page: #f7f5f0;
  --ink: #1a1918;
  --ink-secondary: #5c5954;
  --ink-muted: #7d7973;
  --border: #e0ddd6;
  --fern: #4a8b6f;
  --fern-light: #f1f8f4;
  --indigo: #4338ca;
  --pollen-bg: #fffbeb;
  --pollen: #f59e0b;
  --code-bg: #f2f0eb;
  --highlight-bg: #fef3c7;
  --font-serif: 'Literata', Georgia, 'Times New Roman', serif;
  --font-mono: 'JetBrains Mono', 'Menlo', 'Consolas', monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --page: #1a1918;
    --ink: #f2f0eb;
    --ink-secondary: #a8a49d;
    --ink-muted: #7d7973;
    --border: #383532;
    --fern: #6fa588;
    --fern-light: rgba(74, 139, 111, 0.12);
    --indigo: #818cf8;
    --code-bg: #2b2926;
    --highlight-bg: rgba(245, 158, 11, 0.15);
  }
}

*, *::before, *::after { box-sizing: border-box; }

html {
  font-size: 16px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

body {
  font-family: var(--font-serif);
  line-height: 1.7;
  color: var(--ink);
  background: var(--page);
  max-width: 42rem;
  margin: 2.5rem auto;
  padding: 0 1.5rem 4rem;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
  font-weight: 600;
  line-height: 1.3;
  color: var(--ink);
  margin-top: 2rem;
  margin-bottom: 0.75rem;
}
h1 { font-size: 2rem; margin-top: 0; }
h2 { font-size: 1.5rem; }
h3 { font-size: 1.25rem; }
h4 { font-size: 1.1rem; }
h5 { font-size: 1rem; font-weight: 500; }
h6 { font-size: 0.9rem; font-weight: 500; color: var(--ink-secondary); }

/* Paragraphs */
p { margin: 0 0 1rem; }
p:empty { min-height: 1.7em; }

/* Links */
a {
  color: var(--indigo);
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
}
a:hover { text-decoration-thickness: 2px; }

/* Lists */
ul, ol {
  margin: 0 0 1rem;
  padding-left: 1.5em;
}
li { margin-bottom: 0.25em; }
li > ul, li > ol { margin-top: 0.25em; margin-bottom: 0; }

/* Task lists */
ul.task-list {
  list-style: none;
  padding-left: 0;
}
ul.task-list > li {
  display: flex;
  align-items: baseline;
  gap: 0.5em;
}
ul.task-list input[type="checkbox"] {
  accent-color: var(--fern);
  margin: 0;
  flex-shrink: 0;
}

/* Blockquote */
blockquote {
  border-left: 3px solid var(--fern);
  margin: 0 0 1rem;
  padding: 0.25rem 0 0.25rem 1.25rem;
  color: var(--ink-secondary);
}
blockquote p:last-child { margin-bottom: 0; }

/* Code */
code {
  font-family: var(--font-mono);
  font-size: 0.88em;
  background: var(--code-bg);
  padding: 0.15em 0.35em;
  border-radius: 3px;
}

pre {
  background: var(--code-bg);
  padding: 1rem 1.25rem;
  border-radius: 6px;
  overflow-x: auto;
  margin: 0 0 1rem;
  border: 1px solid var(--border);
}
pre code {
  background: none;
  padding: 0;
  font-size: 0.85em;
  line-height: 1.6;
}

/* Horizontal rule */
hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 2rem 0;
}

/* Highlight / mark */
mark {
  background: var(--highlight-bg);
  padding: 0.1em 0.2em;
  border-radius: 2px;
  color: inherit;
}

/* Strikethrough */
s, del { color: var(--ink-muted); }

/* Footnotes */
.footnotes {
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
  font-size: 0.9em;
  color: var(--ink-secondary);
}
.footnotes ol { padding-left: 1.5em; }
.footnotes li { margin-bottom: 0.5em; }
sup.fn-ref a {
  color: var(--fern);
  text-decoration: none;
  font-weight: 500;
}
sup.fn-ref a:hover { text-decoration: underline; }

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.5rem 0;
  font-size: 0.95em;
}
thead {
  border-bottom: 2px solid var(--border);
}
th {
  font-weight: 600;
  text-align: left;
  padding: 0.75rem;
  background: var(--code-bg);
  border: 1px solid var(--border);
}
td {
  padding: 0.75rem;
  border: 1px solid var(--border);
}
tr:hover {
  background: var(--fern-light);
}

@media print {
  table { page-break-inside: avoid; }
  tr { page-break-inside: avoid; }
}
"""

# Minimal CSS for unthemed export
MINIMAL_CSS = """\
body { font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; }
h1, h2, h3 { font-weight: 600; }
code { background: #f4f4f4; padding: 0.2em 0.4em; border-radius: 3px; font-family: monospace; }
pre { background: #f4f4f4; padding: 1rem; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 1rem; color: #666; }
mark { background: #fff3cd; }
hr { border: none; border-top: 1px solid #ddd; margin: 2rem 0; }
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tiptap_xml_to_html(
    xml_str: str,
    *,
    title: str | None = None,
    themed: bool = True,
    include_block_ids: bool = False,
    full_document: bool = True,
) -> str:
    """Convert TipTap XML to semantic HTML.

    Args:
        xml_str: TipTap XML content (as returned by read_document).
        title: Document title for <title> and optional <h1> header.
        themed: If True, include Garden-themed CSS. If False, minimal CSS.
        include_block_ids: Preserve data-block-id attributes on elements.
        full_document: If True, wrap in <!DOCTYPE html>...; if False,
            return just the body content HTML fragment.

    Returns:
        HTML string (complete document or fragment).
    """
    if not xml_str or not xml_str.strip():
        if full_document:
            return _wrap_document("", title=title, themed=themed)
        return ""

    wrapped = f"<root>{xml_str}</root>"

    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        # Fallback: strip tags
        content = f"<p>{html.escape(_strip_tags(xml_str))}</p>"
        if full_document:
            return _wrap_document(content, title=title, themed=themed)
        return content

    footnotes: list[str] = []
    body_parts: list[str] = []

    # Collect all blocks, handling flat list grouping
    elements = list(root)
    i = 0
    while i < len(elements):
        elem = elements[i]
        tag = elem.tag

        if tag in (LIST_ITEM_TAG, TASK_ITEM_TAG):
            # Start of a flat list run — collect consecutive items
            run, end = _collect_list_run(elements, i)
            body_parts.append(_render_list_group(run, footnotes, include_block_ids))
            i = end
        elif tag in (BULLET_LIST_TAG, ORDERED_LIST_TAG, TASK_LIST_TAG):
            # Container-wrapped list — descend into children
            body_parts.append(_convert_block(elem, footnotes, include_block_ids))
            i += 1
        else:
            body_parts.append(_convert_block(elem, footnotes, include_block_ids))
            i += 1

    body_html = "\n".join(body_parts)

    # Append footnotes section
    if footnotes:
        body_html += _render_footnotes(footnotes)

    if full_document:
        return _wrap_document(body_html, title=title, themed=themed)
    return body_html


# ---------------------------------------------------------------------------
# Document wrapper
# ---------------------------------------------------------------------------


def _wrap_document(
    body_html: str,
    *,
    title: str | None = None,
    themed: bool = True,
) -> str:
    css = GARDEN_CSS if themed else MINIMAL_CSS
    safe_title = html.escape(title) if title else "Untitled"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
{css}</style>
</head>
<body>
{body_html}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Block conversion
# ---------------------------------------------------------------------------


def _convert_block(
    elem: ET.Element,
    footnotes: list[str],
    include_ids: bool,
) -> str:
    tag = elem.tag
    bid = _block_id_attr(elem, include_ids)

    if tag == HEADING_TAG:
        level = _get_int_attr(elem, "level", 1)
        level = max(1, min(level, 6))
        content = _inline_content(elem, footnotes)
        return f"<h{level}{bid}>{content}</h{level}>"

    if tag == PARAGRAPH_TAG:
        content = _inline_content(elem, footnotes)
        return f"<p{bid}>{content}</p>"

    if tag == CODE_BLOCK_TAG:
        language = elem.get("language", "") or ""
        code = html.escape(_plain_text(elem))
        lang_cls = f' class="language-{html.escape(language)}"' if language else ""
        return f"<pre{bid}><code{lang_cls}>{code}</code></pre>"

    if tag == BLOCKQUOTE_TAG:
        inner = "\n".join(
            _convert_block(child, footnotes, include_ids) for child in elem
        )
        return f"<blockquote{bid}>\n{inner}\n</blockquote>"

    if tag == HORIZONTAL_RULE_TAG:
        return f"<hr{bid} />"

    if tag in (BULLET_LIST_TAG, ORDERED_LIST_TAG, TASK_LIST_TAG):
        # Wrapper list element — convert children
        items = "\n".join(
            _convert_block(child, footnotes, include_ids) for child in elem
        )
        if tag == ORDERED_LIST_TAG:
            return f"<ol{bid}>\n{items}\n</ol>"
        return f"<ul{bid}>\n{items}\n</ul>"

    if tag == LIST_ITEM_TAG:
        content = _list_item_inline(elem, footnotes)
        return f"<li{bid}>{content}</li>"

    if tag == TASK_ITEM_TAG:
        checked = elem.get("checked", "false") == "true"
        chk = " checked" if checked else ""
        content = _list_item_inline(elem, footnotes)
        return f'<li{bid}><input type="checkbox" disabled{chk} /> {content}</li>'

    if tag == TABLE_TAG:
        rows = "\n".join(
            _convert_block(child, footnotes, include_ids) for child in elem
        )
        return f"<table{bid}>\n{rows}\n</table>"

    if tag == TABLE_ROW_TAG:
        cells = "\n".join(
            _convert_block(child, footnotes, include_ids) for child in elem
        )
        return f"<tr{bid}>\n{cells}\n</tr>"

    if tag == TABLE_HEADER_TAG:
        content = _inline_content(elem, footnotes)
        return f"<th{bid}>{content}</th>"

    if tag == TABLE_CELL_TAG:
        content = _inline_content(elem, footnotes)
        return f"<td{bid}>{content}</td>"

    # Unknown — try inline content
    content = _inline_content(elem, footnotes)
    if content:
        return f"<p{bid}>{content}</p>"
    return ""


# ---------------------------------------------------------------------------
# Flat list → nested HTML reconstruction
# ---------------------------------------------------------------------------


def _collect_list_run(
    elements: list[ET.Element], start: int
) -> tuple[list[ET.Element], int]:
    """Collect consecutive listItem/taskItem elements from a flat stream."""
    run: list[ET.Element] = []
    i = start
    while i < len(elements) and elements[i].tag in (LIST_ITEM_TAG, TASK_ITEM_TAG):
        run.append(elements[i])
        i += 1
    return run, i


def _render_list_group(
    items: list[ET.Element],
    footnotes: list[str],
    include_ids: bool,
) -> str:
    """Render a flat list run into properly nested HTML lists.

    TipTap stores lists as flat items with listType and data-indent.
    This reconstructs the nested <ul>/<ol>/<li> HTML structure.
    """
    if not items:
        return ""

    lines: list[str] = []
    stack: list[str] = []  # tracks open list tags for nesting

    for elem in items:
        indent = _get_int_attr(elem, "data-indent", 0)
        is_task = elem.tag == TASK_ITEM_TAG
        list_type = elem.get("listType", "bullet")
        bid = _block_id_attr(elem, include_ids)

        # Determine the list container tag
        if is_task:
            container_tag = 'ul class="task-list"'
            close_tag = "ul"
        elif list_type == "ordered":
            container_tag = "ol"
            close_tag = "ol"
        else:
            container_tag = "ul"
            close_tag = "ul"

        target_depth = indent + 1  # depth 1 = top-level list

        # Close lists that are deeper than needed
        while len(stack) > target_depth:
            closed = stack.pop()
            lines.append(f"{'  ' * len(stack)}</{closed}>")
            # Close the parent <li> that contained this nested list
            if stack and len(stack) >= target_depth:
                pass  # li was already written

        # Open lists to reach target depth
        while len(stack) < target_depth:
            lines.append(f"{'  ' * len(stack)}<{container_tag}>")
            stack.append(close_tag)

        # Render the item
        pad = "  " * len(stack)
        if is_task:
            checked = elem.get("checked", "false") == "true"
            chk = " checked" if checked else ""
            content = _list_item_inline(elem, footnotes)
            lines.append(f'{pad}<li{bid}><input type="checkbox" disabled{chk} /> {content}</li>')
        else:
            content = _list_item_inline(elem, footnotes)
            lines.append(f"{pad}<li{bid}>{content}</li>")

    # Close remaining open lists
    while stack:
        closed = stack.pop()
        lines.append(f"{'  ' * len(stack)}</{closed}>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inline content
# ---------------------------------------------------------------------------


def _inline_content(elem: ET.Element, footnotes: list[str]) -> str:
    """Convert element's inline content to HTML."""
    parts: list[str] = []

    if elem.text:
        parts.append(html.escape(elem.text))

    for child in elem:
        parts.append(_convert_inline(child, footnotes))
        if child.tail:
            parts.append(html.escape(child.tail))

    return "".join(parts)


def _list_item_inline(elem: ET.Element, footnotes: list[str]) -> str:
    """Extract inline content from a list/task item's paragraph children."""
    parts = []
    for child in elem:
        if child.tag == PARAGRAPH_TAG:
            parts.append(_inline_content(child, footnotes))
        else:
            t = _inline_content(child, footnotes)
            if t:
                parts.append(t)
    return " ".join(parts) if parts else ""


def _convert_inline(elem: ET.Element, footnotes: list[str]) -> str:
    """Convert a single inline element to HTML."""
    tag = elem.tag

    if tag in BOLD_TAGS:
        content = _inline_content(elem, footnotes)
        return f"<strong>{content}</strong>"

    if tag in ITALIC_TAGS:
        content = _inline_content(elem, footnotes)
        return f"<em>{content}</em>"

    if tag in STRIKE_TAGS:
        content = _inline_content(elem, footnotes)
        return f"<s>{content}</s>"

    if tag in CODE_TAGS:
        content = html.escape(_plain_text(elem))
        return f"<code>{content}</code>"

    if tag in LINK_TAGS:
        href = html.escape(elem.get("href", ""))
        content = _inline_content(elem, footnotes)
        return f'<a href="{href}">{content}</a>'

    if tag in HIGHLIGHT_TAGS:
        content = _inline_content(elem, footnotes)
        return f"<mark>{content}</mark>"

    if tag in COMMENT_TAGS:
        # Strip comment marks, keep content
        return _inline_content(elem, footnotes)

    if tag == FOOTNOTE_TAG:
        fn_content = elem.get("content") or elem.get("data-footnote-content", "")
        footnotes.append(fn_content)
        idx = len(footnotes)
        return f'<sup class="fn-ref"><a href="#fn-{idx}" id="fnref-{idx}">[{idx}]</a></sup>'

    if tag == WIKILINK_TAG:
        label = elem.get("label", "")
        return html.escape(label)

    if tag == HARD_BREAK_TAG:
        return "<br />"

    if tag == PARAGRAPH_TAG:
        return _inline_content(elem, footnotes)

    # Unknown inline — extract content
    return _inline_content(elem, footnotes)


# ---------------------------------------------------------------------------
# Footnotes section
# ---------------------------------------------------------------------------


def _render_footnotes(footnotes: list[str]) -> str:
    items = []
    for i, content in enumerate(footnotes, 1):
        safe = html.escape(content)
        items.append(
            f'  <li id="fn-{i}">{safe} '
            f'<a href="#fnref-{i}">\u21a9</a></li>'
        )
    return (
        '\n<section class="footnotes">\n'
        "<ol>\n"
        + "\n".join(items)
        + "\n</ol>\n"
        "</section>\n"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plain_text(elem: ET.Element) -> str:
    """Extract plain text, ignoring formatting."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_plain_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _block_id_attr(elem: ET.Element, include: bool) -> str:
    """Return data-block-id attribute string if requested."""
    if not include:
        return ""
    bid = elem.get("data-block-id")
    if bid:
        return f' data-block-id="{html.escape(bid)}"'
    return ""


def _strip_tags(xml_str: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(xml_str))


def _get_int_attr(elem: ET.Element, attr: str, default: int = 0) -> int:
    val = elem.get(attr)
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default
