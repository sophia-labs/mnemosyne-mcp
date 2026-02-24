"""Document editing helpers for TipTap/ProseMirror Y.js documents.

Provides high-level operations for reading and modifying collaborative documents
stored as Y.XmlFragment("content") - TipTap's native format.

Documents are exposed as XML strings, preserving full formatting fidelity
(bold, italic, highlight, links, etc.) without lossy markdown conversion.

IMPORTANT: y-prosemirror encodes marks (bold, italic, etc.) as **attributes on
Y.XmlText nodes**, not as nested Y.XmlElement wrappers. This matches how TipTap
and ProseMirror represent formatting internally.

Example: "Hello <strong>bold</strong> world" becomes:
  XmlElement("paragraph", contents=[
    XmlText("Hello bold world")  # with format(6, 10, {"bold": {}})
  ])

NOT:
  XmlElement("paragraph", contents=[
    XmlText("Hello "),
    XmlElement("strong", contents=[XmlText("bold")]),  # WRONG
    XmlText(" world"),
  ])
"""

from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from typing import Any

import pycrdt

from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("hocuspocus.document")

# Mark names that y-prosemirror uses (maps XML element names to Y.js attributes)
# These are represented as formatting attributes on XmlText nodes
MARK_ELEMENTS = frozenset({
    "strong",      # bold
    "em",          # italic
    "code",        # inline code
    "strike",      # strikethrough
    "s",           # strikethrough alt
    "mark",        # highlight
    "a",           # link
    "commentMark", # comment annotation - wraps text (data-comment-id)
})

# Inline node elements - these become XmlElement children, NOT text marks
# Unlike marks, these are atomic nodes that don't wrap text content
# The frontend TipTap extensions define these with `atom: true`
INLINE_NODE_ELEMENTS = frozenset({
    "footnote",    # self-contained annotation (data-footnote-content)
})

# Map XML attribute names to Y.js/TipTap internal attribute names
# y-prosemirror passes Y.js attributes directly to TipTap, so we need
# to store using TipTap's internal attribute names
INLINE_NODE_ATTR_MAP: dict[str, dict[str, str]] = {
    "footnote": {
        "data-footnote-content": "content",  # XML attr → TipTap attr
    },
}

# Map XML attribute names to TipTap internal attribute names for marks
# (similar to INLINE_NODE_ATTR_MAP but for mark formatting attributes)
MARK_ATTR_MAP: dict[str, dict[str, str]] = {
    "commentMark": {
        "data-comment-id": "commentId",  # XML attr → TipTap attr
    },
    "a": {
        "href": "href",  # Pass through (already same)
        "target": "target",
    },
}

# Map HTML/XML element names to TipTap's internal mark names
# TipTap uses different names internally than the HTML tags we accept in XML
MARK_NAME_MAP: dict[str, str] = {
    "strong": "bold",      # HTML <strong> → TipTap "bold" mark
    "em": "italic",        # HTML <em> → TipTap "italic" mark
    "s": "strike",         # HTML <s> → TipTap "strike" mark
    "strike": "strike",    # Also accept <strike>
    "mark": "highlight",   # HTML <mark> → TipTap "highlight" mark
    "a": "link",           # HTML <a> → TipTap "link" mark
    "code": "code",        # Same name
    "commentMark": "commentMark",  # Comment annotation - same name
}

# Reverse mapping: TipTap internal mark names back to XML element names
MARK_NAME_TO_XML: dict[str, str] = {
    "bold": "strong",
    "italic": "em",
    "strike": "s",
    "highlight": "mark",
    "link": "a",
    "code": "code",
    "commentMark": "commentMark",
}

# Map TipTap internal attribute names back to XML attribute names (reverse of MARK_ATTR_MAP)
MARK_ATTR_TO_XML: dict[str, dict[str, str]] = {
    "commentMark": {
        "commentId": "data-comment-id",
    },
    "a": {
        "href": "href",
        "target": "target",
    },
}

# Reverse mapping: TipTap internal inline-node attr names back to XML names
INLINE_NODE_ATTR_TO_XML: dict[str, dict[str, str]] = {
    tag: {tiptap_key: xml_key for xml_key, tiptap_key in attr_map.items()}
    for tag, attr_map in INLINE_NODE_ATTR_MAP.items()
}

# Map XML attribute names to TipTap internal attribute names for block elements
# y-prosemirror stores ProseMirror internal attribute names, not HTML attribute names
BLOCK_ATTR_MAP: dict[str, dict[str, str]] = {
    "paragraph": {
        "data-indent": "indent",  # XML attr → TipTap attr
    },
    "heading": {
        "data-indent": "indent",  # XML attr → TipTap attr
        "level": "level",         # Pass through (same name)
    },
    "listItem": {
        "data-indent": "indent",  # XML attr → TipTap attr
        "listType": "listType",   # Pass through (bullet/ordered/task)
        "checked": "checked",     # Pass through (for task items)
    },
}

# Reverse mapping: TipTap internal block attr names back to XML names
BLOCK_ATTR_TO_XML: dict[str, dict[str, str]] = {
    tag: {tiptap_key: xml_key for xml_key, tiptap_key in attr_map.items()}
    for tag, attr_map in BLOCK_ATTR_MAP.items()
}

# Block types that need data-block-id (matches TipTap's BlockId extension)
# Note: bulletList, orderedList, taskList are NOT block types - they're converted
# to flat listItem blocks with listType attribute during XML processing.
BLOCK_TYPES = frozenset({
    "paragraph",
    "heading",
    "listItem",      # Flat list item with listType attribute (bullet/ordered/task)
    "blockquote",
    "codeBlock",
    "horizontalRule",
    "table",
    "tableRow",
    "tableHeader",
    "tableCell",
})

# List container elements that should be flattened to listItem blocks
LIST_CONTAINER_TYPES = frozenset({
    "bulletList",
    "orderedList",
    "taskList",
})

# Block types that require paragraph children — cannot contain inline text directly.
# TipTap's schema rejects XmlText nodes inside these; they must be wrapped in <paragraph>.
# If bare text is written inside these, _xml_to_pycrdt auto-wraps it rather than silently dropping it.
PARAGRAPH_REQUIRED_CONTAINERS = frozenset({
    "blockquote",
    "tableCell",
    "tableHeader",
})


def _generate_block_id() -> str:
    """Generate a unique block ID matching TipTap's format."""
    return f"block-{uuid.uuid4().hex[:8]}"


def _get_attr_safe(attrs: Any, key: str, default: Any = None) -> Any:
    """Safely get an attribute from XmlAttributesView or dict.

    pycrdt's XmlAttributesView.get() doesn't accept a default value,
    so we need this wrapper.
    """
    try:
        if key in attrs:
            return attrs[key]
        return default
    except (TypeError, KeyError):
        return default


def _get_list_type_from_container(tag: str) -> str:
    """Map list container tag to listType attribute value."""
    mapping = {
        "bulletList": "bullet",
        "orderedList": "ordered",
        "taskList": "task",
    }
    return mapping.get(tag, "bullet")


def _map_inline_node_attrs(tag: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """Map XML attribute names to TipTap internal attribute names.

    y-prosemirror passes Y.js XmlElement attributes directly to TipTap,
    so we need to store them using TipTap's internal attribute names
    rather than the HTML/XML attribute names.

    Example:
        <footnote data-footnote-content="note"/> in XML becomes
        XmlElement("footnote", {"content": "note"}) in Y.js
        which TipTap reads as node.attrs.content
    """
    attr_map = INLINE_NODE_ATTR_MAP.get(tag, {})
    if not attr_map:
        return attrs

    result = {}
    for key, value in attrs.items():
        # Map the attribute name if there's a mapping, otherwise keep original
        mapped_key = attr_map.get(key, key)
        result[mapped_key] = value
    return result


def _map_mark_attrs(tag: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """Map XML attribute names to TipTap internal attribute names for marks.

    Similar to _map_inline_node_attrs but for mark formatting attributes.
    y-prosemirror stores mark attributes in the delta format, and TipTap
    expects specific attribute names.

    Example:
        <commentMark data-comment-id="c-123">text</commentMark> in XML becomes
        XmlText with format {commentMark: {commentId: "c-123"}}
        which TipTap reads as mark.attrs.commentId
    """
    attr_map = MARK_ATTR_MAP.get(tag, {})
    if not attr_map:
        return attrs

    result = {}
    for key, value in attrs.items():
        # Map the attribute name if there's a mapping, otherwise keep original
        mapped_key = attr_map.get(key, key)
        result[mapped_key] = value
    return result


def _escape_xml(text: str) -> str:
    """Escape special XML characters in text content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _mark_name_to_xml_tag(mark_name: str) -> str | None:
    """Convert TipTap internal mark name to XML element tag.

    Args:
        mark_name: TipTap's internal name (e.g., "bold", "italic")

    Returns:
        XML tag name (e.g., "strong", "em") or None if unknown
    """
    return MARK_NAME_TO_XML.get(mark_name)


def _build_mark_attrs_string(xml_tag: str, mark_attrs: dict[str, Any] | None) -> str:
    """Build XML attribute string for a mark element.

    Maps TipTap internal attribute names back to XML attribute names.

    Args:
        xml_tag: The XML tag name (e.g., "a", "commentMark")
        mark_attrs: Dict of TipTap internal attribute names and values

    Returns:
        Attribute string like ' href="..."' or empty string
    """
    if not mark_attrs:
        return ""

    attr_map = MARK_ATTR_TO_XML.get(xml_tag, {})
    parts = []

    for key, value in mark_attrs.items():
        if value is None:
            continue
        # Map internal name to XML name
        xml_key = attr_map.get(key, key)
        # Escape attribute value
        escaped_value = str(value).replace('"', "&quot;")
        parts.append(f'{xml_key}="{escaped_value}"')

    return " " + " ".join(parts) if parts else ""


def _map_block_attrs(tag: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """Map XML attribute names to TipTap internal attribute names for blocks.

    Similar to _map_inline_node_attrs but for block-level node attributes.
    y-prosemirror stores block attributes using TipTap's internal names.

    Example:
        <paragraph data-indent="2">text</paragraph> in XML becomes
        XmlElement("paragraph", {"indent": 2}) in Y.js
        which TipTap reads as node.attrs.indent

    Note: data-block-id is handled separately and preserved as-is since
    the BlockId extension uses that exact attribute name.
    """
    attr_map = BLOCK_ATTR_MAP.get(tag, {})

    result = {}
    for key, value in attrs.items():
        # data-block-id is special - preserve as-is
        if key == "data-block-id":
            result[key] = value
            continue
        # Map the attribute name if there's a mapping, otherwise keep original
        mapped_key = attr_map.get(key, key)
        # Convert numeric attributes from XML strings to integers
        if mapped_key in ("indent", "level") and value is not None:
            try:
                value = int(value)
            except (ValueError, TypeError):
                value = 0
        result[mapped_key] = value
    return result


def _map_attrs_to_xml(tag: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """Map TipTap/internal attribute names back to XML attribute names."""
    inline_attr_map = INLINE_NODE_ATTR_TO_XML.get(tag, {})
    block_attr_map = BLOCK_ATTR_TO_XML.get(tag, {})

    result: dict[str, Any] = {}
    for key, value in attrs.items():
        if key == "data-block-id":
            result[key] = value
            continue
        xml_key = inline_attr_map.get(key, block_attr_map.get(key, key))
        result[xml_key] = value
    return result


def extract_title_from_xml(xml_str: str) -> str | None:
    """Extract title from first heading element in TipTap XML.

    Searches for the first <heading> element and returns its text content.
    Used to derive document titles for workspace navigation.

    Args:
        xml_str: TipTap XML content string

    Returns:
        The text content of the first heading, or None if no heading found.

    Example:
        >>> extract_title_from_xml('<heading level="1">My Title</heading><paragraph>...</paragraph>')
        'My Title'
    """
    try:
        # Wrap for parsing (handles multiple root elements)
        wrapped = f"<root>{xml_str}</root>"
        root = ET.fromstring(wrapped)

        # Find first heading element (depth-first search)
        def find_heading(elem: ET.Element) -> ET.Element | None:
            if elem.tag == "heading":
                return elem
            for child in elem:
                result = find_heading(child)
                if result is not None:
                    return result
            return None

        heading = find_heading(root)
        if heading is not None:
            # Get all text content (handles marks inside heading)
            text = "".join(heading.itertext()).strip()
            return text if text else None
        return None
    except ET.ParseError:
        logger.warning("Failed to parse XML for title extraction")
        return None


class DocumentReader:
    """Reads TipTap document structure from a Y.Doc.

    Uses Y.XmlFragment("content") which is the native TipTap format,
    matching the platform backend and browser client.
    """

    def __init__(self, doc: pycrdt.Doc) -> None:
        self._doc = doc

    def get_content_fragment(self) -> pycrdt.XmlFragment:
        """Get the content XmlFragment for native TipTap collaboration."""
        return self._doc.get("content", type=pycrdt.XmlFragment)

    def has_content(self) -> bool:
        """Check if the document has any content."""
        try:
            fragment = self.get_content_fragment()
            return len(list(fragment.children)) > 0
        except Exception:
            return False

    def to_xml(self) -> str:
        """Return document content as TipTap XML.

        Properly reconstructs mark elements (strong, em, etc.) from Y.js
        formatting attributes on XmlText nodes.

        Example output:
            <paragraph>Hello <strong>bold</strong> world</paragraph>
            <heading level="2">Section</heading>
        """
        fragment = self.get_content_fragment()
        return self._serialize_fragment(fragment)

    def _serialize_fragment(self, fragment: pycrdt.XmlFragment) -> str:
        """Serialize an XmlFragment to TipTap XML string."""
        parts = []
        for child in fragment.children:
            parts.append(self._serialize_element(child))
        return "".join(parts)

    def _serialize_element(self, elem: Any) -> str:
        """Serialize an XmlElement or XmlText to TipTap XML string."""
        if isinstance(elem, pycrdt.XmlText):
            return self._serialize_text_with_marks(elem)
        elif isinstance(elem, pycrdt.XmlElement):
            return self._serialize_xml_element(elem)
        else:
            # Fallback for unknown types
            return str(elem)

    def _serialize_xml_element(self, elem: pycrdt.XmlElement) -> str:
        """Serialize an XmlElement to TipTap XML string."""
        tag = elem.tag

        # Build attributes string (convert XmlAttributesView to dict)
        attrs_parts = []
        raw_attrs = dict(elem.attributes)
        xml_attrs = _map_attrs_to_xml(tag, raw_attrs)
        for key, value in xml_attrs.items():
            if value is None:
                continue
            # Convert Python booleans to lowercase strings
            if isinstance(value, bool):
                value = "true" if value else "false"
            # Convert floats to ints if they're whole numbers (pycrdt stores ints as floats)
            elif isinstance(value, float) and value == int(value):
                value = int(value)
            attrs_parts.append(f'{key}="{value}"')

        attrs_str = " " + " ".join(attrs_parts) if attrs_parts else ""

        # Serialize children
        children_parts = []
        for child in elem.children:
            children_parts.append(self._serialize_element(child))
        children_str = "".join(children_parts)

        if children_str:
            return f"<{tag}{attrs_str}>{children_str}</{tag}>"
        else:
            # Self-closing tag for empty elements
            return f"<{tag}{attrs_str}/>"

    def _serialize_text_with_marks(self, text_node: pycrdt.XmlText) -> str:
        """Serialize an XmlText node with its formatting marks.

        Uses diff() runs so we can normalize both mark names and mark attributes
        (e.g. commentId -> data-comment-id) when building TipTap XML.
        """
        try:
            runs = text_node.diff()
        except Exception:
            # Fallback for unexpected pycrdt behavior
            text_content = str(text_node)
            return _escape_xml(text_content) if text_content else ""

        parts: list[str] = []
        for text_or_embed, attrs in runs:
            if not isinstance(text_or_embed, str) or not text_or_embed:
                continue

            segment = _escape_xml(text_or_embed)
            if isinstance(attrs, dict) and attrs:
                open_tags: list[str] = []
                close_tags: list[str] = []
                # Stable order so output is deterministic
                for mark_name in sorted(attrs.keys()):
                    xml_tag = _mark_name_to_xml_tag(mark_name) or mark_name
                    mark_attrs = attrs.get(mark_name)
                    mark_attrs_dict = mark_attrs if isinstance(mark_attrs, dict) else {}
                    attrs_str = _build_mark_attrs_string(xml_tag, mark_attrs_dict)
                    open_tags.append(f"<{xml_tag}{attrs_str}>")
                    close_tags.append(f"</{xml_tag}>")
                segment = "".join(open_tags) + segment + "".join(reversed(close_tags))

            parts.append(segment)

        return "".join(parts)

    def get_block_count(self) -> int:
        """Get the number of top-level blocks in the document."""
        fragment = self.get_content_fragment()
        return len(list(fragment.children))

    def find_block_by_id(self, block_id: str) -> tuple[int, Any] | None:
        """Find a block by its data-block-id attribute.

        Args:
            block_id: The block ID to search for (e.g., "block-abc12345")

        Returns:
            Tuple of (index, XmlElement) if found, None otherwise.
        """
        fragment = self.get_content_fragment()
        for i, child in enumerate(fragment.children):
            if hasattr(child, "attributes"):
                if child.attributes.get("data-block-id") == block_id:
                    return (i, child)
        return None

    def get_block_at(self, index: int) -> Any | None:
        """Get the block at a specific index.

        Args:
            index: The index of the block (0-based)

        Returns:
            The XmlElement at that index, or None if out of bounds.
        """
        fragment = self.get_content_fragment()
        children = list(fragment.children)
        if 0 <= index < len(children):
            return children[index]
        return None

    def get_block_info(self, block_id: str) -> dict[str, Any] | None:
        """Get detailed information about a block by its ID.

        Args:
            block_id: The block ID to search for

        Returns:
            Dict with block info, or None if not found:
            {
                "block_id": "block-abc123",
                "index": 3,
                "type": "paragraph",
                "xml": "<paragraph ...>content</paragraph>",
                "attributes": {"indent": 1, ...},
                "text_content": "Plain text content",
                "context": {
                    "total_blocks": 15,
                    "prev_block_id": "block-xyz",
                    "next_block_id": "block-def"
                }
            }
        """
        result = self.find_block_by_id(block_id)
        if result is None:
            return None

        index, elem = result
        fragment = self.get_content_fragment()
        children = list(fragment.children)
        total = len(children)

        # Get prev/next block IDs
        prev_id = None
        next_id = None
        if index > 0:
            prev_elem = children[index - 1]
            if hasattr(prev_elem, "attributes"):
                prev_id = prev_elem.attributes.get("data-block-id")
        if index < total - 1:
            next_elem = children[index + 1]
            if hasattr(next_elem, "attributes"):
                next_id = next_elem.attributes.get("data-block-id")

        # Extract attributes
        attrs = dict(elem.attributes) if hasattr(elem, "attributes") else {}

        # Get text content
        text_content = str(elem) if elem else ""
        # Strip XML tags for plain text (simple extraction)
        import re
        plain_text = re.sub(r"<[^>]+>", "", text_content)

        # Compute text_length from XmlText children
        text_length = self._compute_text_length(elem)

        return {
            "block_id": block_id,
            "index": index,
            "type": elem.tag if hasattr(elem, "tag") else "unknown",
            "xml": str(elem),
            "attributes": attrs,
            "text_content": plain_text.strip(),
            "text_length": text_length,
            "context": {
                "total_blocks": total,
                "prev_block_id": prev_id,
                "next_block_id": next_id,
            },
        }

    def query_blocks(
        self,
        block_type: str | None = None,
        indent: int | None = None,
        indent_gte: int | None = None,
        indent_lte: int | None = None,
        list_type: str | None = None,
        checked: bool | None = None,
        text_contains: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query blocks matching specific criteria.

        Args:
            block_type: Filter by block type (paragraph, heading, listItem, etc.)
            indent: Filter by exact indent level
            indent_gte: Filter by indent >= value
            indent_lte: Filter by indent <= value
            list_type: For listItems, filter by listType (bullet, ordered, task)
            checked: For task items, filter by checked state
            text_contains: Filter by text content containing this string
            limit: Maximum number of results to return

        Returns:
            List of matching block summaries.
        """
        fragment = self.get_content_fragment()
        children_list = list(fragment.children)
        total = len(children_list)
        matches = []
        import re

        for i, child in enumerate(children_list):
            if len(matches) >= limit:
                break

            if not hasattr(child, "attributes"):
                continue

            attrs = dict(child.attributes)
            tag = child.tag if hasattr(child, "tag") else "unknown"

            # Filter by type
            if block_type and tag != block_type:
                continue

            # Filter by indent
            elem_indent = attrs.get("indent", 0)
            if isinstance(elem_indent, str):
                try:
                    elem_indent = int(elem_indent)
                except ValueError:
                    elem_indent = 0

            if indent is not None and elem_indent != indent:
                continue
            if indent_gte is not None and elem_indent < indent_gte:
                continue
            if indent_lte is not None and elem_indent > indent_lte:
                continue

            # Filter by listType
            if list_type and attrs.get("listType") != list_type:
                continue

            # Filter by checked
            if checked is not None:
                elem_checked = attrs.get("checked", False)
                if isinstance(elem_checked, str):
                    elem_checked = elem_checked.lower() == "true"
                if elem_checked != checked:
                    continue

            # Filter by text content
            text = str(child)
            plain_text = re.sub(r"<[^>]+>", "", text).strip()
            if text_contains and text_contains.lower() not in plain_text.lower():
                continue

            # Get prev/next block IDs for navigation
            prev_id = None
            next_id = None
            if i > 0:
                prev_elem = children_list[i - 1]
                if hasattr(prev_elem, "attributes"):
                    prev_id = prev_elem.attributes.get("data-block-id")
            if i < total - 1:
                next_elem = children_list[i + 1]
                if hasattr(next_elem, "attributes"):
                    next_id = next_elem.attributes.get("data-block-id")

            # Build match summary
            matches.append({
                "block_id": attrs.get("data-block-id"),
                "index": i,
                "type": tag,
                "text_preview": plain_text[:100] + ("..." if len(plain_text) > 100 else ""),
                "attributes": {
                    k: v for k, v in attrs.items()
                    if k not in ("data-block-id",)  # Exclude redundant fields
                },
                "prev_block_id": prev_id,
                "next_block_id": next_id,
            })

        return matches

    def _compute_text_length(self, elem: Any) -> int:
        """Compute the total text length of a block element.

        Walks XmlText children (counting characters) and XmlElement children
        (counting 1 position each for inline nodes like footnotes).

        Args:
            elem: A pycrdt.XmlElement block

        Returns:
            Total character count in the offset space.
        """
        total = 0
        if not hasattr(elem, "children"):
            return 0
        for child in elem.children:
            if isinstance(child, pycrdt.XmlText):
                total += len(child)
            elif isinstance(child, pycrdt.XmlElement):
                # Inline elements (footnote, etc.) occupy 1 position
                total += 1
        return total

    def get_block_text_info(self, block_id: str) -> dict[str, Any] | None:
        """Get detailed text info for a block, optimized for the editing tool.

        Returns formatting runs with offsets so agents can determine where
        to insert/delete text and what formatting exists at each position.

        Args:
            block_id: The block ID to get text info for

        Returns:
            Dict with text info, or None if not found:
            {
                "block_id": "block-abc123",
                "text": "Hello bold world",
                "length": 16,
                "runs": [
                    {"text": "Hello ", "offset": 0, "length": 6, "attrs": None},
                    {"text": "bold", "offset": 6, "length": 4, "attrs": {"bold": {}}},
                    {"text": " world", "offset": 10, "length": 6, "attrs": None},
                ],
                "has_inline_nodes": False,
            }
        """
        result = self.find_block_by_id(block_id)
        if result is None:
            return None

        _, elem = result
        children = list(elem.children)

        has_inline_nodes = any(
            isinstance(child, pycrdt.XmlElement) for child in children
        )

        # Build runs by walking children
        runs: list[dict[str, Any]] = []
        full_text_parts: list[str] = []
        global_offset = 0

        for child in children:
            if isinstance(child, pycrdt.XmlText):
                # Get formatting runs from diff()
                diff_runs = child.diff()
                for text_or_embed, attrs in diff_runs:
                    if isinstance(text_or_embed, str):
                        run_len = len(text_or_embed)
                        runs.append({
                            "text": text_or_embed,
                            "offset": global_offset,
                            "length": run_len,
                            "attrs": attrs if attrs else None,
                        })
                        full_text_parts.append(text_or_embed)
                        global_offset += run_len
            elif isinstance(child, pycrdt.XmlElement):
                # Inline element (footnote, etc.) — 1 position
                runs.append({
                    "text": None,
                    "offset": global_offset,
                    "length": 1,
                    "attrs": None,
                    "inline_element": child.tag,
                })
                full_text_parts.append("\ufffc")  # Object replacement char
                global_offset += 1

        full_text = "".join(full_text_parts)

        return {
            "block_id": block_id,
            "text": full_text,
            "length": global_offset,
            "runs": runs,
            "has_inline_nodes": has_inline_nodes,
        }

    def get_comments_map(self) -> "pycrdt.Map[dict[str, Any]]":
        """Get the comments Y.Map for this document."""
        return self._doc.get("comments", type=pycrdt.Map)

    def get_all_comments(self) -> dict[str, dict[str, Any]]:
        """Get all comments from the Y.Map('comments').

        Returns:
            Dict mapping commentId to comment metadata:
            {
                "comment-123": {
                    "text": "Great point here",
                    "author": "Alice",
                    "authorId": "user-1",
                    "createdAt": 1699999999000,
                    "updatedAt": 1699999999000,
                    "resolved": false
                },
                ...
            }
        """
        comments_map = self.get_comments_map()
        return dict(comments_map.items())


class DocumentWriter:
    """Writes content to TipTap Y.js documents.

    Uses Y.XmlFragment("content") which is the native TipTap format,
    matching the platform backend and browser client.

    IMPORTANT: Methods in this class modify the Y.Doc in place. Use with
    HocuspocusClient.transact_document() to properly capture and broadcast
    incremental updates:

        await client.transact_document(graph_id, doc_id, lambda doc:
            DocumentWriter(doc).append_block("<paragraph>Hello</paragraph>")
        )
    """

    def __init__(self, doc: pycrdt.Doc) -> None:
        self._doc = doc
        self._pending_formats: list[tuple[pycrdt.XmlText, list[dict[str, Any]]]] = []
        self._seen_block_ids: set[str] = set()  # Track IDs to detect duplicates

    def get_content_fragment(self) -> pycrdt.XmlFragment:
        """Get the content XmlFragment for native TipTap collaboration."""
        return self._doc.get("content", type=pycrdt.XmlFragment)

    # -------------------------------------------------------------------------
    # Surgical Edit Methods (collaborative-safe)
    # -------------------------------------------------------------------------

    def append_block(self, xml_str: str) -> None:
        """Append a block element to the end of the document.

        This is collaborative-safe - it only adds content, never removes.

        Note: List containers (bulletList, orderedList, taskList) are automatically
        flattened to individual listItem blocks with listType attributes.

        Args:
            xml_str: TipTap XML for a single block element, e.g.:
                     "<paragraph>Hello world</paragraph>"
                     "<heading level=\"2\">Section</heading>"
                     "<bulletList><listItem><paragraph>Item</paragraph></listItem></bulletList>"
        """
        logger.info(
            "append_block: starting",
            extra_context={"xml_str": xml_str[:200]},
        )

        fragment = self.get_content_fragment()
        block_count_before = len(list(fragment.children))
        try:
            elem = ET.fromstring(xml_str)
        except ET.ParseError as e:
            if "junk after document element" in str(e):
                raise ValueError(
                    "append_block accepts a single top-level XML block element per call. "
                    "To append multiple blocks, make multiple calls. "
                    f"Original error: {e}"
                ) from e
            raise

        logger.info(
            "append_block: parsed XML",
            extra_context={
                "elem_tag": elem.tag,
                "elem_attribs": dict(elem.attrib),
                "block_count_before": block_count_before,
            },
        )

        with self._doc.transaction():
            # Process element - may return multiple blocks for list containers
            blocks = self._process_element(elem)
            logger.info(
                "append_block: processed element into blocks",
                extra_context={
                    "num_blocks": len(blocks),
                    "source_tag": elem.tag,
                },
            )
            for block in blocks:
                fragment.children.append(block)
            self._apply_pending_formats()

        block_count_after = len(list(fragment.children))
        logger.info(
            "append_block: completed",
            extra_context={
                "block_count_before": block_count_before,
                "block_count_after": block_count_after,
                "content_after": str(fragment)[:500],
            },
        )

    # -------------------------------------------------------------------------
    # Character-Level Edit Methods (CRDT-native, collaborative-safe)
    # -------------------------------------------------------------------------

    def edit_block_text(
        self,
        block_id: str,
        operations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply character-level insert/delete operations to a block's text.

        Uses pycrdt's native XmlText.insert() and del text[start:end] for
        CRDT-safe edits that merge cleanly with concurrent browser changes.

        Operations are sorted by offset descending and applied in one
        transaction so earlier offsets don't shift later ones.

        Args:
            block_id: The block to edit
            operations: List of operation dicts, each with:
                - type: "insert" or "delete" (required)
                - offset: global character position, 0-indexed (required)
                - text: string to insert (required for insert)
                - length: number of chars to delete (required for delete)
                - attrs: optional formatting dict, e.g. {"bold": {}} (for insert)
                - inherit_format: bool, default True — inherit formatting from
                  preceding character (for insert)

        Returns:
            Dict with updated block text info (same format as
            DocumentReader.get_block_text_info).

        Raises:
            ValueError: If block not found, operations empty/invalid, or
                       delete offset is out of bounds.
        """
        if not operations:
            raise ValueError("Operations list cannot be empty")

        result = self.find_block_by_id(block_id)
        if result is None:
            raise ValueError(f"Block not found: {block_id}")

        _, block = result

        # Validate all operations upfront
        for i, op in enumerate(operations):
            op_type = op.get("type")
            if op_type not in ("insert", "delete"):
                raise ValueError(
                    f"Operation {i}: type must be 'insert' or 'delete', got {op_type!r}"
                )
            if "offset" not in op:
                raise ValueError(f"Operation {i}: 'offset' is required")
            if op_type == "insert" and "text" not in op:
                raise ValueError(f"Operation {i}: 'text' is required for insert")
            if op_type == "delete" and "length" not in op:
                raise ValueError(f"Operation {i}: 'length' is required for delete")

        # Sort by offset descending so earlier offsets don't shift later ones
        sorted_ops = sorted(operations, key=lambda op: op["offset"], reverse=True)

        with self._doc.transaction():
            for op in sorted_ops:
                op_type = op["type"]
                offset = op["offset"]

                if op_type == "insert":
                    text = op["text"]
                    if not text:
                        continue  # Skip empty inserts

                    text_node, local_offset = self._resolve_offset_to_text_node(
                        block, offset, clamp=True
                    )

                    # Determine formatting attrs
                    explicit_attrs = op.get("attrs")
                    inherit_format = op.get("inherit_format", True)

                    if explicit_attrs is not None:
                        attrs = explicit_attrs
                    elif inherit_format and local_offset > 0:
                        attrs = self._get_format_at_offset(text_node, local_offset)
                    else:
                        attrs = None

                    text_node.insert(local_offset, text, attrs=attrs)

                elif op_type == "delete":
                    length = op["length"]  # code-point length from agent
                    if length <= 0:
                        continue  # Skip zero-length deletes

                    text_node, local_byte_offset = self._resolve_offset_to_text_node(
                        block, offset, clamp=False
                    )

                    # Validate delete doesn't start past end
                    text_byte_len = len(text_node)
                    if local_byte_offset >= text_byte_len:
                        raise ValueError(
                            f"Delete offset {offset} is beyond text length"
                        )

                    # Convert code-point delete length to UTF-8 byte length.
                    # Use _get_plain_text (not str()) since str() includes markup tags.
                    plain_text = self._get_plain_text(text_node)
                    plain_bytes = plain_text.encode("utf-8")
                    # Find the code-point position corresponding to byte offset
                    cp_start = len(plain_bytes[:local_byte_offset].decode("utf-8"))
                    # Slice the string by code points, then get byte length
                    delete_substr = plain_text[cp_start:cp_start + length]
                    byte_length = len(delete_substr.encode("utf-8"))

                    # Clamp to not exceed text end
                    end = min(local_byte_offset + byte_length, text_byte_len)
                    del text_node[local_byte_offset:end]

        # Read back updated text info
        reader = DocumentReader(self._doc)
        return reader.get_block_text_info(block_id)

    @staticmethod
    def _get_plain_text(text_node: pycrdt.XmlText) -> str:
        """Get plain text content of an XmlText node (without markup tags).

        str(XmlText) includes XML markup (e.g., '<bold>text</bold>'), so it
        cannot be used for offset calculations. This method extracts only the
        text content from diff() runs.
        """
        parts: list[str] = []
        for text_or_embed, _attrs in text_node.diff():
            if isinstance(text_or_embed, str):
                parts.append(text_or_embed)
        return "".join(parts)

    def _resolve_offset_to_text_node(
        self,
        block: Any,
        global_offset: int,
        clamp: bool = True,
    ) -> tuple[pycrdt.XmlText, int]:
        """Resolve a global code-point offset to a specific XmlText node and local UTF-8 byte offset.

        Accepts offsets in Unicode code points (what agents/humans count) and
        returns byte offsets (what pycrdt/Yrs uses internally). This bridges the
        gap between the MCP API (code points) and pycrdt operations (UTF-8 bytes).

        IMPORTANT: str(XmlText) includes XML markup tags, so we must use diff()
        to get the plain text for offset calculation.

        Walks the block's children. Common case (single XmlText child) is a
        direct lookup. For blocks with inline nodes (footnote, etc.), each
        XmlText contributes its plain-text code-point count, each inline
        XmlElement contributes 1 position.

        Args:
            block: The pycrdt.XmlElement block
            global_offset: The global code-point offset (0-indexed)
            clamp: If True, clamp offset to text end (for inserts).
                   If False, raise on out-of-bounds (for deletes).

        Returns:
            Tuple of (XmlText node, local UTF-8 byte offset within that node).

        Raises:
            ValueError: If offset points at an inline element, or if clamp=False
                       and offset is out of bounds.
        """
        children = list(block.children)

        # Fast path: single text node (vast majority of blocks)
        if len(children) == 1 and isinstance(children[0], pycrdt.XmlText):
            text_node = children[0]
            plain_text = self._get_plain_text(text_node)
            text_len_cp = len(plain_text)
            if clamp:
                cp_offset = min(global_offset, text_len_cp)
            elif global_offset > text_len_cp:
                raise ValueError(
                    f"Offset {global_offset} is beyond text length {text_len_cp}"
                )
            else:
                cp_offset = global_offset
            # Convert code-point offset to UTF-8 byte offset
            byte_offset = len(plain_text[:cp_offset].encode("utf-8"))
            return (text_node, byte_offset)

        # Walk children for blocks with inline nodes
        running_cp = 0
        last_text_node = None

        for child in children:
            if isinstance(child, pycrdt.XmlText):
                last_text_node = child
                child_plain = self._get_plain_text(child)
                child_len_cp = len(child_plain)
                if global_offset <= running_cp + child_len_cp:
                    local_cp = global_offset - running_cp
                    byte_offset = len(child_plain[:local_cp].encode("utf-8"))
                    return (child, byte_offset)
                running_cp += child_len_cp
            elif isinstance(child, pycrdt.XmlElement):
                if global_offset == running_cp:
                    raise ValueError(
                        f"Offset {global_offset} points at an inline element "
                        f"({child.tag}). Use offsets before or after inline elements."
                    )
                running_cp += 1

        # Beyond end — return byte length for clamping
        if clamp and last_text_node is not None:
            return (last_text_node, len(last_text_node))

        if not clamp:
            raise ValueError(
                f"Offset {global_offset} is beyond total text length {running_cp}"
            )

        # Fallback: no text nodes at all (empty block or block with only inline elements)
        raise ValueError(f"Block has no text content to edit")

    def _get_format_at_offset(
        self, text_node: pycrdt.XmlText, local_byte_offset: int
    ) -> dict[str, Any] | None:
        """Get formatting attributes at a given byte offset by reading diff() runs.

        Returns the attrs of the run containing the character just before
        the offset (i.e., "inherit from preceding character"). This matches
        browser typing behavior.

        Args:
            text_node: The XmlText node
            local_byte_offset: The local UTF-8 byte offset within the text node

        Returns:
            Formatting attrs dict, or None if no formatting at that position.
        """
        if local_byte_offset <= 0:
            return None

        diff_runs = text_node.diff()
        running = 0
        for text_or_embed, attrs in diff_runs:
            if isinstance(text_or_embed, str):
                # Use UTF-8 byte length to match pycrdt's offset system
                run_len = len(text_or_embed.encode("utf-8"))
            else:
                run_len = 1  # embed

            if local_byte_offset <= running + run_len:
                # The preceding character is in this run
                return dict(attrs) if attrs else None
            running += run_len

        return None

    def insert_block_at(self, index: int, xml_str: str) -> None:
        """Insert a block element at a specific position.

        This is collaborative-safe - it inserts without removing existing content.

        Note: List containers are flattened, so multiple blocks may be inserted.

        Args:
            index: Position to insert at (0 = beginning)
            xml_str: TipTap XML for a single block element
        """
        fragment = self.get_content_fragment()
        elem = ET.fromstring(xml_str)

        with self._doc.transaction():
            # Process element - may return multiple blocks for list containers
            blocks = self._process_element(elem)
            # Insert in order at the specified position
            for i, block in enumerate(blocks):
                fragment.children.insert(index + i, block)
            self._apply_pending_formats()

    def delete_block_at(self, index: int) -> None:
        """Delete a block at a specific position.

        Args:
            index: Position of the block to delete
        """
        fragment = self.get_content_fragment()

        with self._doc.transaction():
            del fragment.children[index]

    def get_block_count(self) -> int:
        """Get the number of blocks in the document."""
        fragment = self.get_content_fragment()
        return len(list(fragment.children))

    # -------------------------------------------------------------------------
    # Block-by-ID Operations (collaborative-safe, targeted updates)
    # -------------------------------------------------------------------------

    def find_block_by_id(self, block_id: str) -> tuple[int, Any] | None:
        """Find a block by its data-block-id attribute.

        Args:
            block_id: The block ID to search for (e.g., "block-abc12345")

        Returns:
            Tuple of (index, XmlElement) if found, None otherwise.
        """
        fragment = self.get_content_fragment()
        for i, child in enumerate(fragment.children):
            if hasattr(child, "attributes"):
                if child.attributes.get("data-block-id") == block_id:
                    return (i, child)
        return None

    def delete_block_by_id(self, block_id: str, cascade_children: bool = False) -> list[str]:
        """Delete a block by its ID, optionally cascading to indent-children.

        Args:
            block_id: The block ID to delete
            cascade_children: If True, also delete all subsequent blocks with
                             higher indent (indent-based children)

        Returns:
            List of deleted block IDs.

        Raises:
            ValueError: If block not found.
        """
        result = self.find_block_by_id(block_id)
        if result is None:
            raise ValueError(f"Block not found: {block_id}")

        index, elem = result
        deleted_ids = [block_id]

        fragment = self.get_content_fragment()

        with self._doc.transaction():
            if cascade_children:
                # Find children by indent
                parent_indent = _get_attr_safe(elem.attributes, "indent", 0)
                if isinstance(parent_indent, str):
                    try:
                        parent_indent = int(parent_indent)
                    except ValueError:
                        parent_indent = 0

                children = list(fragment.children)
                # Collect indices to delete (in reverse order to maintain positions)
                indices_to_delete = [index]

                for i in range(index + 1, len(children)):
                    child = children[i]
                    if not hasattr(child, "attributes"):
                        break
                    child_indent = _get_attr_safe(child.attributes, "indent", 0)
                    if isinstance(child_indent, str):
                        try:
                            child_indent = int(child_indent)
                        except ValueError:
                            child_indent = 0

                    if child_indent <= parent_indent:
                        break  # No longer a child

                    indices_to_delete.append(i)
                    child_id = _get_attr_safe(child.attributes, "data-block-id", None)
                    if child_id:
                        deleted_ids.append(child_id)

                # Delete in reverse order to maintain indices
                for idx in reversed(indices_to_delete):
                    del fragment.children[idx]
            else:
                del fragment.children[index]

        return deleted_ids

    def update_block_attributes(self, block_id: str, attributes: dict[str, Any]) -> None:
        """Update specific attributes on a block without replacing its content.

        This is the most surgical update - it only modifies the specified
        attributes, leaving content and other attributes untouched.

        Args:
            block_id: The block ID to update
            attributes: Dict of attributes to set. Common attributes:
                       - indent: int (0-6)
                       - checked: bool (for task items)
                       - listType: str (bullet/ordered/task)
                       - collapsed: bool (for outliner)

        Raises:
            ValueError: If block not found.
        """
        result = self.find_block_by_id(block_id)
        if result is None:
            raise ValueError(f"Block not found: {block_id}")

        index, elem = result

        with self._doc.transaction():
            for key, value in attributes.items():
                # Handle special cases
                if key == "indent" and value is not None:
                    value = int(value)
                elif key == "checked":
                    value = bool(value)

                elem.attributes[key] = value

    def replace_block_by_id(self, block_id: str, xml_str: str) -> str:
        """Replace a block's content entirely while preserving its block ID.

        The new block will keep the same data-block-id as the original.

        Args:
            block_id: The block ID to replace
            xml_str: New TipTap XML for the block

        Returns:
            The block_id (unchanged).

        Raises:
            ValueError: If block not found.
        """
        result = self.find_block_by_id(block_id)
        if result is None:
            raise ValueError(f"Block not found: {block_id}")

        index, _ = result
        fragment = self.get_content_fragment()

        # Parse new content
        elem = ET.fromstring(xml_str)

        # Override the block ID in the XML so _process_element uses it
        elem.set("data-block-id", block_id)

        with self._doc.transaction():
            # Delete old block
            del fragment.children[index]

            # Process new element (handles list container flattening)
            blocks = self._process_element(elem)

            # Insert new block(s)
            for i, block in enumerate(blocks):
                fragment.children.insert(index + i, block)

            self._apply_pending_formats()

        return block_id

    def insert_block_after_id(self, after_block_id: str, xml_str: str) -> str:
        """Insert a new block after the specified block.

        Args:
            after_block_id: The block ID to insert after
            xml_str: TipTap XML for the new block

        Returns:
            The new block's generated ID.

        Raises:
            ValueError: If reference block not found.
        """
        logger.info(
            "insert_block_after_id: starting",
            extra_context={
                "after_block_id": after_block_id,
                "xml_str": xml_str[:200],
            },
        )

        result = self.find_block_by_id(after_block_id)
        if result is None:
            logger.error(
                "insert_block_after_id: reference block not found",
                extra_context={"after_block_id": after_block_id},
            )
            raise ValueError(f"Block not found: {after_block_id}")

        index, _ = result
        fragment = self.get_content_fragment()
        block_count_before = len(list(fragment.children))
        elem = ET.fromstring(xml_str)

        # Pre-generate block ID if not already set
        new_block_id = elem.get("data-block-id")
        if not new_block_id:
            new_block_id = _generate_block_id()
            elem.set("data-block-id", new_block_id)

        logger.info(
            "insert_block_after_id: inserting at position",
            extra_context={
                "insert_after_index": index,
                "new_block_id": new_block_id,
                "elem_tag": elem.tag,
            },
        )

        with self._doc.transaction():
            blocks = self._process_element(elem)

            logger.info(
                "insert_block_after_id: processed into blocks",
                extra_context={
                    "num_blocks": len(blocks),
                    "source_tag": elem.tag,
                },
            )

            for i, block in enumerate(blocks):
                fragment.children.insert(index + 1 + i, block)

            self._apply_pending_formats()

        block_count_after = len(list(fragment.children))
        logger.info(
            "insert_block_after_id: completed",
            extra_context={
                "new_block_id": new_block_id,
                "block_count_before": block_count_before,
                "block_count_after": block_count_after,
                "content_after": str(fragment)[:500],
            },
        )

        return new_block_id

    def insert_block_before_id(self, before_block_id: str, xml_str: str) -> str:
        """Insert a new block before the specified block.

        Args:
            before_block_id: The block ID to insert before
            xml_str: TipTap XML for the new block

        Returns:
            The new block's generated ID.

        Raises:
            ValueError: If reference block not found.
        """
        result = self.find_block_by_id(before_block_id)
        if result is None:
            raise ValueError(f"Block not found: {before_block_id}")

        index, _ = result
        fragment = self.get_content_fragment()
        elem = ET.fromstring(xml_str)

        # Pre-generate block ID if not already set
        new_block_id = elem.get("data-block-id")
        if not new_block_id:
            new_block_id = _generate_block_id()
            elem.set("data-block-id", new_block_id)

        with self._doc.transaction():
            blocks = self._process_element(elem)

            for i, block in enumerate(blocks):
                fragment.children.insert(index + i, block)

            self._apply_pending_formats()

        return new_block_id

    # -------------------------------------------------------------------------
    # Destructive Methods (use with caution in collaborative contexts)
    # -------------------------------------------------------------------------

    def clear_content(self) -> None:
        """Clear all content from the document.

        WARNING: This is destructive - it removes all existing content.
        Concurrent edits from other clients will be lost.
        """
        fragment = self.get_content_fragment()

        with self._doc.transaction():
            while list(fragment.children):
                del fragment.children[0]

    def replace_all_content(self, xml_str: str) -> None:
        """Replace entire document content with new TipTap XML.

        WARNING: This is DESTRUCTIVE - it clears all existing content first.
        Any concurrent edits from other clients will be lost.

        Note: List containers (bulletList, orderedList, taskList) are automatically
        flattened to individual listItem blocks with listType attributes.

        For collaborative editing, prefer surgical methods:
        - append_block() to add content
        - insert_block_at() to insert at position
        - delete_block_at() to remove specific blocks

        Args:
            xml_str: TipTap XML content, e.g.:
                     "<paragraph>Hello</paragraph><paragraph>World</paragraph>"

        Raises:
            ValueError: If content is not valid TipTap XML
        """
        content = xml_str.strip()

        # Empty content is valid - just clear
        if not content:
            self.clear_content()
            return

        # Validate XML structure
        if not content.startswith("<"):
            raise ValueError(
                "Content must be valid TipTap XML (got plain text). "
                "Wrap plain text in <paragraph>...</paragraph>."
            )

        # Wrap for parsing (handles multiple root elements)
        wrapped = f"<root>{content}</root>"
        try:
            root = ET.fromstring(wrapped)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")

        self.clear_content()
        fragment = self.get_content_fragment()

        with self._doc.transaction():
            for child in root:
                # Process element - may return multiple blocks for list containers
                blocks = self._process_element(child)
                for block in blocks:
                    fragment.children.append(block)
            self._apply_pending_formats()

    # -------------------------------------------------------------------------
    # Legacy API (deprecated)
    # -------------------------------------------------------------------------

    def clear_document(self) -> bytes:
        """DEPRECATED: Use clear_content() with transact_document() instead."""
        import warnings

        warnings.warn(
            "clear_document() is deprecated. Use clear_content() with "
            "HocuspocusClient.transact_document() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.clear_content()
        return self._doc.get_update()

    def set_content_from_xml(self, xml_str: str) -> bytes:
        """DEPRECATED: Use replace_all_content() with transact_document() instead."""
        import warnings

        warnings.warn(
            "set_content_from_xml() is deprecated. Use replace_all_content() with "
            "HocuspocusClient.transact_document() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.replace_all_content(xml_str)
        return self._doc.get_update()

    def _flatten_list_container(
        self, elem: ET.Element, list_type: str, base_indent: int = 0
    ) -> list[pycrdt.XmlElement]:
        """Flatten a list container (bulletList/orderedList/taskList) into flat listItem blocks.

        Converts nested list structure to flat listItems with attributes:
        - listType: 'bullet' | 'ordered' | 'task'
        - indent: hierarchy level (0-based)
        - checked: boolean (for task items)

        Args:
            elem: The list container element (bulletList, orderedList, taskList)
            list_type: The type of list ('bullet', 'ordered', 'task')
            base_indent: The starting indent level for items in this list

        Returns:
            List of pycrdt.XmlElement for each flattened listItem
        """
        items: list[pycrdt.XmlElement] = []

        for child in elem:
            # Handle listItem or taskItem
            if child.tag in ("listItem", "taskItem"):
                # Collect content and nested lists separately
                content_children: list[ET.Element] = []
                nested_lists: list[tuple[ET.Element, str]] = []

                for subchild in child:
                    if subchild.tag in LIST_CONTAINER_TYPES:
                        # This is a nested list - process after the item content
                        nested_type = _get_list_type_from_container(subchild.tag)
                        nested_lists.append((subchild, nested_type))
                    else:
                        # This is content (paragraph, etc.)
                        content_children.append(subchild)

                # Build the listItem element with content
                contents: list[Any] = []
                for content_child in content_children:
                    if content_child.tag in BLOCK_TYPES:
                        contents.append(self._xml_to_pycrdt(content_child))
                    else:
                        # Inline content directly in listItem
                        content_items = self._extract_inline_content(content_child)
                        contents.extend(content_items)

                # If no block children, extract inline content from the listItem itself
                if not contents:
                    content_items = self._extract_inline_content(child)
                    contents.extend(content_items)

                # Build attributes for the flattened listItem
                # Check for existing block ID from source element
                existing_id = child.get("data-block-id", "").strip()
                if existing_id and existing_id not in self._seen_block_ids:
                    block_id = existing_id
                else:
                    if existing_id:
                        logger.warning(
                            "Duplicate block ID in list item, regenerating",
                            extra_context={"original_id": existing_id},
                        )
                    block_id = _generate_block_id()
                self._seen_block_ids.add(block_id)

                attrs: dict[str, Any] = {
                    "listType": list_type,
                    "data-block-id": block_id,
                }
                if base_indent > 0:
                    attrs["indent"] = base_indent

                # For task items, handle checked state
                if list_type == "task" or child.tag == "taskItem":
                    attrs["listType"] = "task"
                    checked = child.get("data-checked") == "true" or child.get("checked") == "true"
                    if checked:
                        attrs["checked"] = True

                items.append(pycrdt.XmlElement(
                    "listItem",
                    attrs,
                    contents=contents or None,
                ))

                # Process nested lists at increased indent
                for nested_elem, nested_type in nested_lists:
                    nested_items = self._flatten_list_container(
                        nested_elem, nested_type, base_indent + 1
                    )
                    items.extend(nested_items)

        return items

    def _xml_to_pycrdt(self, elem: ET.Element) -> pycrdt.XmlElement:
        """Convert XML element to pycrdt XmlElement.

        Handles three cases:
        1. Block with nested blocks (listItem > paragraph): Recursively build children
        2. Block with inline nodes (paragraph with footnotes): Mixed XmlText/XmlElement children
        3. Block with only marks (paragraph with bold/italic): Single XmlText with formatting

        Note: List containers (bulletList, orderedList, taskList) are NOT handled here.
        They should be pre-processed via _flatten_list_container() or _process_element().

        Marks (strong, em, etc.) are encoded as formatting attributes on XmlText.
        Inline nodes (footnote, commentMark) become XmlElement children.

        Auto-assigns data-block-id to block types that need it (matches
        TipTap's BlockId extension).
        """
        contents: list[Any] = []

        # Check if this element has any nested block children (excluding list containers)
        has_block_children = any(
            child.tag in BLOCK_TYPES or child.tag in LIST_CONTAINER_TYPES
            for child in elem
        )

        if has_block_children:
            # Handle nested block structure (e.g., listItem > paragraph)
            # Recursively process each block child
            for child in elem:
                if child.tag in BLOCK_TYPES:
                    contents.append(self._xml_to_pycrdt(child))
                elif child.tag in LIST_CONTAINER_TYPES:
                    # Flatten nested list and add items
                    list_type = _get_list_type_from_container(child.tag)
                    items = self._flatten_list_container(child, list_type, 0)
                    contents.extend(items)
                # Note: We ignore non-block children in block containers
        elif elem.tag in PARAGRAPH_REQUIRED_CONTAINERS:
            # blockquote, tableCell, tableHeader require paragraph children.
            # If an agent writes <blockquote>bare text</blockquote>, auto-wrap the
            # inline content in a paragraph rather than silently dropping it.
            has_content = bool(elem.text and elem.text.strip()) or len(elem) > 0
            if has_content:
                para = ET.Element("paragraph")
                para.text = elem.text
                for child in elem:
                    para.append(child)
                contents.append(self._xml_to_pycrdt(para))
        else:
            # Handle inline content (paragraph, heading with text/marks/inline nodes)
            # This produces a list of content items: XmlText and XmlElement mixed
            content_items = self._extract_inline_content(elem)
            contents.extend(content_items)

        # Build attributes, mapping XML names to TipTap internal names
        attrs = _map_block_attrs(elem.tag, dict(elem.attrib))

        # Ensure valid unique data-block-id for block types
        if elem.tag in BLOCK_TYPES:
            block_id = attrs.get("data-block-id", "").strip()

            # Empty or missing - generate new
            if not block_id:
                block_id = _generate_block_id()

            # Duplicate - regenerate with warning
            if block_id in self._seen_block_ids:
                logger.warning(
                    "Duplicate block ID detected, regenerating",
                    extra_context={"original_id": block_id},
                )
                block_id = _generate_block_id()

            self._seen_block_ids.add(block_id)
            attrs["data-block-id"] = block_id

        return pycrdt.XmlElement(
            elem.tag,
            attrs,
            contents=contents or None,
        )

    def _process_element(self, elem: ET.Element) -> list[pycrdt.XmlElement]:
        """Process a single XML element, returning one or more pycrdt elements.

        This handles the top-level case where list containers need to be flattened
        into multiple listItem elements.

        Args:
            elem: The XML element to process

        Returns:
            List of pycrdt.XmlElement (usually one, but multiple for list containers)
        """
        if elem.tag in LIST_CONTAINER_TYPES:
            # Flatten list container into multiple listItem elements
            list_type = _get_list_type_from_container(elem.tag)
            return self._flatten_list_container(elem, list_type, 0)
        else:
            # Regular block element - return as single-item list
            return [self._xml_to_pycrdt(elem)]

    def _extract_text_runs(
        self, elem: ET.Element, inherited_marks: dict[str, dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Extract text runs with their marks from an element.

        Returns a list of dicts: [{"text": str, "marks": {mark_name: attrs}}]

        Marks are accumulated through nested elements (e.g., <strong><em>text</em></strong>
        produces a single run with both bold and italic marks).

        NOTE: This method only handles MARK_ELEMENTS (bold, italic, commentMark, etc.).
        INLINE_NODE_ELEMENTS (footnote) are handled by _extract_inline_content.
        """
        runs: list[dict[str, Any]] = []
        marks = dict(inherited_marks or {})

        # If this element is a mark, add it to the current marks
        if elem.tag in MARK_ELEMENTS:
            mark_attrs = dict(elem.attrib) if elem.attrib else {}
            # Map XML attribute names to TipTap internal names
            mark_attrs = _map_mark_attrs(elem.tag, mark_attrs)
            marks[elem.tag] = mark_attrs

        # Text before first child
        if elem.text:
            runs.append({"text": elem.text, "marks": dict(marks)})

        # Process children
        for child in elem:
            if child.tag in MARK_ELEMENTS:
                # Recurse into mark element, inheriting current marks
                child_runs = self._extract_text_runs(child, marks)
                runs.extend(child_runs)
            elif child.tag not in INLINE_NODE_ELEMENTS:
                # Non-mark, non-inline-node child - extract its text
                child_runs = self._extract_text_runs(child, marks)
                runs.extend(child_runs)
            # Note: INLINE_NODE_ELEMENTS are skipped here - they're handled
            # by _extract_inline_content which creates XmlElement nodes for them

            # Tail text (after this child element)
            # Use marks (not inherited_marks): tail is outside the CHILD but
            # still inside the CURRENT element, so it carries the current marks.
            if child.tail:
                runs.append({"text": child.tail, "marks": dict(marks)})

        return runs

    def _apply_pending_formats(self) -> None:
        """Apply formatting to XmlText nodes after they're integrated."""
        for text_node, runs in self._pending_formats:
            offset = 0
            for run in runs:
                text = run["text"]
                marks = run["marks"]
                # pycrdt/Yrs uses UTF-8 byte offsets, not Unicode code points.
                # Python's len() counts code points, which differs for non-ASCII
                # characters (e.g., em dash U+2014 is 1 code point but 3 UTF-8 bytes).
                length = len(text.encode("utf-8"))

                if marks:
                    # Apply each mark as a format attribute
                    for mark_name, mark_attrs in marks.items():
                        # Map HTML element name to TipTap's internal mark name
                        # e.g., "strong" → "bold", "em" → "italic"
                        mapped_name = MARK_NAME_MAP.get(mark_name, mark_name)
                        # y-prosemirror uses empty dict {} for marks without attrs
                        text_node.format(offset, offset + length, {mapped_name: mark_attrs or {}})

                offset += length

        self._pending_formats.clear()

    # -------------------------------------------------------------------------
    # Comment Metadata Methods (stored in Y.Map('comments'))
    # -------------------------------------------------------------------------

    def get_comments_map(self) -> "pycrdt.Map[dict[str, Any]]":
        """Get the comments Y.Map for this document.

        Comments are stored as a Y.Map with commentId as key and metadata as value.
        The metadata includes: text, author, authorId, createdAt, updatedAt, resolved.
        """
        return self._doc.get("comments", type=pycrdt.Map)

    def set_comment(
        self,
        comment_id: str,
        text: str,
        author: str = "MCP Agent",
        author_id: str = "mcp-agent",
        resolved: bool = False,
        quoted_text: str | None = None,
    ) -> None:
        """Set or update a comment in the Y.Map('comments').

        Args:
            comment_id: Unique ID matching data-comment-id in the document
            text: The comment text content
            author: Display name of the comment author
            author_id: User ID of the author
            resolved: Whether the comment has been resolved
            quoted_text: The highlighted/quoted text from the document
        """
        import time

        comments_map = self.get_comments_map()
        now = int(time.time() * 1000)  # milliseconds timestamp like JS Date.now()

        existing = comments_map.get(comment_id)
        created_at = existing.get("createdAt", now) if existing else now
        # Preserve existing quotedText if not provided
        existing_quoted = existing.get("quotedText") if existing else None

        comment_data: dict[str, Any] = {
            "text": text,
            "author": author,
            "authorId": author_id,
            "createdAt": created_at,
            "updatedAt": now,
            "resolved": resolved,
        }
        # Only include quotedText if provided or exists
        if quoted_text is not None:
            comment_data["quotedText"] = quoted_text
        elif existing_quoted is not None:
            comment_data["quotedText"] = existing_quoted

        comments_map[comment_id] = comment_data

    def delete_comment(self, comment_id: str) -> None:
        """Delete a comment from the Y.Map('comments').

        Args:
            comment_id: ID of the comment to delete
        """
        comments_map = self.get_comments_map()
        if comment_id in comments_map:
            del comments_map[comment_id]

    def get_all_comments(self) -> dict[str, dict[str, Any]]:
        """Get all comments from the Y.Map('comments').

        Returns:
            Dict mapping commentId to comment metadata
        """
        comments_map = self.get_comments_map()
        return dict(comments_map.items())

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _extract_inline_content(self, elem: ET.Element) -> list[Any]:
        """Extract inline content as a list of XmlText and XmlElement items.

        For blocks containing inline nodes (footnote, commentMark), we need to
        create separate XmlText nodes around each inline XmlElement. This differs
        from the mark-only case where all text goes into a single XmlText.

        Example: "<paragraph>Text <footnote .../> more</paragraph>" becomes:
          [XmlText("Text "), XmlElement("footnote", ...), XmlText(" more")]

        Returns:
            List of pycrdt.XmlText and pycrdt.XmlElement items
        """
        items: list[Any] = []
        current_runs: list[dict[str, Any]] = []

        def flush_text_runs() -> None:
            """Convert accumulated text runs to an XmlText node."""
            if not current_runs:
                return

            full_text = "".join(run["text"] for run in current_runs)
            if full_text:
                text_node = pycrdt.XmlText(full_text)
                items.append(text_node)
                # Store formatting info for later application
                self._pending_formats.append((text_node, list(current_runs)))
            current_runs.clear()

        def process_element(
            el: ET.Element,
            inherited_marks: dict[str, dict[str, Any]] | None = None
        ) -> None:
            """Process an element, handling text, marks, and inline nodes."""
            marks = dict(inherited_marks or {})

            # If this is a mark element, add to current marks
            if el.tag in MARK_ELEMENTS:
                mark_attrs = dict(el.attrib) if el.attrib else {}
                # Map XML attribute names to TipTap internal names
                mark_attrs = _map_mark_attrs(el.tag, mark_attrs)
                marks[el.tag] = mark_attrs

            # If this is an inline node element, flush text and add the element
            if el.tag in INLINE_NODE_ELEMENTS:
                flush_text_runs()
                # Create the inline node element (empty contents for atom nodes)
                # Map XML attribute names to TipTap internal names
                mapped_attrs = _map_inline_node_attrs(el.tag, dict(el.attrib))
                inline_elem = pycrdt.XmlElement(el.tag, mapped_attrs, contents=[])
                items.append(inline_elem)
                # Process tail text (text after the inline node)
                if el.tail:
                    current_runs.append({"text": el.tail, "marks": dict(inherited_marks or {})})
                return

            # Text before first child
            if el.text:
                current_runs.append({"text": el.text, "marks": dict(marks)})

            # Process children
            for child in el:
                if child.tag in INLINE_NODE_ELEMENTS:
                    # Inline node - flush and add element
                    flush_text_runs()
                    mapped_attrs = _map_inline_node_attrs(child.tag, dict(child.attrib))
                    inline_elem = pycrdt.XmlElement(child.tag, mapped_attrs, contents=[])
                    items.append(inline_elem)
                elif child.tag in MARK_ELEMENTS:
                    # Mark element - recurse to extract text with marks
                    process_element(child, marks)
                else:
                    # Unknown element - try to extract text
                    process_element(child, marks)

                # Tail text (after this child, outside the child element)
                # Use marks (not inherited_marks): tail is outside the CHILD but
                # still inside the CURRENT element. E.g., in <strong>a <em>b</em> c</strong>,
                # "c" is the tail of <em> — it's outside <em> but inside <strong>,
                # so it should carry the strong mark.
                if child.tail:
                    current_runs.append({"text": child.tail, "marks": dict(marks)})

        # Process the root element (but don't treat the root itself as a mark)
        if elem.text:
            current_runs.append({"text": elem.text, "marks": {}})

        for child in elem:
            if child.tag in INLINE_NODE_ELEMENTS:
                flush_text_runs()
                mapped_attrs = _map_inline_node_attrs(child.tag, dict(child.attrib))
                inline_elem = pycrdt.XmlElement(child.tag, mapped_attrs, contents=[])
                items.append(inline_elem)
            elif child.tag in MARK_ELEMENTS:
                process_element(child, {})
            else:
                process_element(child, {})

            if child.tail:
                current_runs.append({"text": child.tail, "marks": {}})

        # Flush any remaining text
        flush_text_runs()

        return items

    def append_paragraph(self, text: str) -> bytes:
        """DEPRECATED: Use append_block() with transact_document() instead.

        Example:
            await client.transact_document(graph_id, doc_id, lambda doc:
                DocumentWriter(doc).append_block(f"<paragraph>{text}</paragraph>")
            )
        """
        import warnings

        warnings.warn(
            "append_paragraph() is deprecated. Use append_block() with "
            "HocuspocusClient.transact_document() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.append_block(f"<paragraph>{text}</paragraph>")
        return self._doc.get_update()


__all__ = [
    "DocumentReader",
    "DocumentWriter",
    "extract_title_from_xml",
]
