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
}

# Block types that need data-block-id (matches TipTap's BlockId extension)
BLOCK_TYPES = frozenset({
    "paragraph",
    "heading",
    "bulletList",
    "orderedList",
    "listItem",
    "blockquote",
    "codeBlock",
    "taskList",
    "taskItem",
    "horizontalRule",
})


def _generate_block_id() -> str:
    """Generate a unique block ID matching TipTap's format."""
    return f"block-{uuid.uuid4().hex[:8]}"


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
        # Convert indent to integer if present
        if mapped_key == "indent" and value is not None:
            try:
                value = int(value)
            except (ValueError, TypeError):
                value = 0
        result[mapped_key] = value
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

        Example output:
            <paragraph>Hello <strong>bold</strong> world</paragraph>
            <heading level="2">Section</heading>
        """
        fragment = self.get_content_fragment()
        return str(fragment)

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

    def get_content_fragment(self) -> pycrdt.XmlFragment:
        """Get the content XmlFragment for native TipTap collaboration."""
        return self._doc.get("content", type=pycrdt.XmlFragment)

    # -------------------------------------------------------------------------
    # Surgical Edit Methods (collaborative-safe)
    # -------------------------------------------------------------------------

    def append_block(self, xml_str: str) -> None:
        """Append a block element to the end of the document.

        This is collaborative-safe - it only adds content, never removes.

        Args:
            xml_str: TipTap XML for a single block element, e.g.:
                     "<paragraph>Hello world</paragraph>"
                     "<heading level=\"2\">Section</heading>"
        """
        fragment = self.get_content_fragment()
        elem = ET.fromstring(xml_str)

        with self._doc.transaction():
            block = self._xml_to_pycrdt(elem)
            fragment.children.append(block)
            self._apply_pending_formats()

    def insert_block_at(self, index: int, xml_str: str) -> None:
        """Insert a block element at a specific position.

        This is collaborative-safe - it inserts without removing existing content.

        Args:
            index: Position to insert at (0 = beginning)
            xml_str: TipTap XML for a single block element
        """
        fragment = self.get_content_fragment()
        elem = ET.fromstring(xml_str)

        with self._doc.transaction():
            block = self._xml_to_pycrdt(elem)
            fragment.children.insert(index, block)
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

        For collaborative editing, prefer surgical methods:
        - append_block() to add content
        - insert_block_at() to insert at position
        - delete_block_at() to remove specific blocks

        Args:
            xml_str: TipTap XML content, e.g.:
                     "<paragraph>Hello</paragraph><paragraph>World</paragraph>"
        """
        self.clear_content()
        fragment = self.get_content_fragment()

        # Wrap for parsing (handles multiple root elements)
        wrapped = f"<root>{xml_str}</root>"
        root = ET.fromstring(wrapped)

        with self._doc.transaction():
            for child in root:
                elem = self._xml_to_pycrdt(child)
                fragment.children.append(elem)
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

    def _xml_to_pycrdt(self, elem: ET.Element) -> pycrdt.XmlElement:
        """Convert XML element to pycrdt XmlElement.

        Handles three cases:
        1. Block with nested blocks (list > listItem > paragraph): Recursively build children
        2. Block with inline nodes (paragraph with footnotes): Mixed XmlText/XmlElement children
        3. Block with only marks (paragraph with bold/italic): Single XmlText with formatting

        Marks (strong, em, etc.) are encoded as formatting attributes on XmlText.
        Inline nodes (footnote, commentMark) become XmlElement children.

        Auto-assigns data-block-id to block types that need it (matches
        TipTap's BlockId extension).
        """
        contents: list[Any] = []

        # Check if this element has any nested block children
        has_block_children = any(child.tag in BLOCK_TYPES for child in elem)

        if has_block_children:
            # Handle nested block structure (e.g., bulletList > listItem > paragraph)
            # Recursively process each block child
            for child in elem:
                if child.tag in BLOCK_TYPES:
                    contents.append(self._xml_to_pycrdt(child))
                # Note: We ignore non-block children in block containers
                # TipTap structure is always: container > block > inline content
        else:
            # Handle inline content (paragraph, heading with text/marks/inline nodes)
            # This produces a list of content items: XmlText and XmlElement mixed
            content_items = self._extract_inline_content(elem)
            contents.extend(content_items)

        # Build attributes, mapping XML names to TipTap internal names
        attrs = _map_block_attrs(elem.tag, dict(elem.attrib))

        # Add data-block-id for block types if not present
        if elem.tag in BLOCK_TYPES and "data-block-id" not in attrs:
            attrs["data-block-id"] = _generate_block_id()

        return pycrdt.XmlElement(
            elem.tag,
            attrs,
            contents=contents or None,
        )

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
            # Use inherited_marks (not marks) because tail text is OUTSIDE the child element
            if child.tail:
                runs.append({"text": child.tail, "marks": dict(inherited_marks or {})})

        return runs

    def _apply_pending_formats(self) -> None:
        """Apply formatting to XmlText nodes after they're integrated."""
        for text_node, runs in self._pending_formats:
            offset = 0
            for run in runs:
                text = run["text"]
                marks = run["marks"]
                length = len(text)

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
                # Use inherited_marks, not marks, since tail is outside the child
                if child.tail:
                    current_runs.append({"text": child.tail, "marks": dict(inherited_marks or {})})

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
