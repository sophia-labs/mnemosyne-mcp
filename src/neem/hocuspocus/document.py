"""Document editing helpers for TipTap/ProseMirror Y.js documents.

Provides high-level operations for reading and modifying collaborative documents
stored as Y.js XmlFragment (TipTap's native format).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

import y_py as Y  # type: ignore[import-untyped]

from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("hocuspocus.document")


@dataclass
class TextSpan:
    """A span of text with optional marks (bold, italic, etc.)."""

    text: str
    marks: List[str] = None  # type: ignore

    def __post_init__(self):
        if self.marks is None:
            self.marks = []


@dataclass
class Block:
    """A document block (paragraph, heading, code block, etc.)."""

    type: str
    content: List[Union[TextSpan, "Block"]]
    attrs: Dict[str, Any] = None  # type: ignore

    def __post_init__(self):
        if self.attrs is None:
            self.attrs = {}

    def to_text(self) -> str:
        """Convert block content to plain text."""
        parts = []
        for item in self.content:
            if isinstance(item, TextSpan):
                parts.append(item.text)
            elif isinstance(item, Block):
                parts.append(item.to_text())
        return "".join(parts)

    def to_markdown(self) -> str:
        """Convert block to markdown."""
        text = self.to_text()

        if self.type == "heading":
            level = int(self.attrs.get("level", 1))
            return "#" * level + " " + text

        if self.type == "bulletList":
            # Nested list handling
            lines = []
            for item in self.content:
                if isinstance(item, Block) and item.type == "listItem":
                    lines.append("- " + item.to_text())
            return "\n".join(lines)

        if self.type == "orderedList":
            lines = []
            for i, item in enumerate(self.content, 1):
                if isinstance(item, Block) and item.type == "listItem":
                    lines.append(f"{i}. " + item.to_text())
            return "\n".join(lines)

        if self.type == "codeBlock":
            lang = self.attrs.get("language", "")
            return f"```{lang}\n{text}\n```"

        if self.type == "blockquote":
            lines = text.split("\n")
            return "\n".join("> " + line for line in lines)

        # Default: paragraph
        return text


class DocumentReader:
    """Reads TipTap document structure from a Y.Doc."""

    def __init__(self, doc: Y.YDoc) -> None:
        self._doc = doc

    def get_xml_element(self, name: str = "prosemirror") -> Y.YXmlElement:
        """Get the main XmlElement containing document content.

        Args:
            name: The element name (default: "prosemirror" for TipTap)

        Returns:
            The Y.js XmlElement root container
        """
        return self._doc.get_xml_element(name)

    def get_blocks(self) -> List[Block]:
        """Extract blocks from the document."""
        root = self.get_xml_element()

        # If root is empty, try fallback to Y.Array named "blocks"
        if len(root) == 0:
            try:
                blocks_array = self._doc.get_array("blocks")
                return self._parse_blocks_array(blocks_array)
            except Exception:
                return []

        return self._parse_xml_element_children(root)

    def _parse_xml_element_children(self, parent: Y.YXmlElement) -> List[Block]:
        """Parse children of an XmlElement into Block objects."""
        blocks = []
        child = parent.first_child
        index = 0
        while child is not None:
            try:
                if isinstance(child, Y.YXmlElement):
                    block = self._parse_xml_element(child)
                    if block:
                        blocks.append(block)
            except Exception as e:
                logger.debug(f"Failed to parse element child {index}: {e}")
            child = child.next_sibling
            index += 1
        return blocks

    def _parse_xml_element(self, element: Any) -> Optional[Block]:
        """Parse a single XmlElement into a Block."""
        if element is None:
            return None

        # Handle Y.YXmlElement
        if isinstance(element, Y.YXmlElement):
            block_type = element.name
            attrs = {}

            # Extract attributes (attributes() returns (key, value) tuples)
            for key, value in element.attributes():
                attrs[key] = value

            # Extract content by traversing children
            content = []
            child = element.first_child
            while child is not None:
                if isinstance(child, Y.YXmlElement):
                    # Nested element
                    nested = self._parse_xml_element(child)
                    if nested:
                        content.append(nested)
                elif isinstance(child, Y.YXmlText):
                    # Text node - convert to string
                    text = str(child)
                    if text:
                        content.append(TextSpan(text=text))
                child = child.next_sibling

            return Block(type=block_type, content=content, attrs=attrs)

        # Handle Y.YXmlText directly
        if isinstance(element, Y.YXmlText):
            text = str(element)
            return Block(type="text", content=[TextSpan(text=text)])

        # Handle plain string
        if isinstance(element, str):
            return Block(type="text", content=[TextSpan(text=element)])

        return None

    def _parse_blocks_array(self, blocks_array: Y.YArray) -> List[Block]:
        """Parse a Y.Array of block objects (legacy format)."""
        blocks = []
        for item in blocks_array:
            if isinstance(item, dict):
                block = self._dict_to_block(item)
                if block:
                    blocks.append(block)
        return blocks

    def _dict_to_block(self, data: Dict[str, Any]) -> Optional[Block]:
        """Convert a dict representation to a Block."""
        block_type = data.get("type", "paragraph")
        attrs = data.get("attrs", {})
        content_data = data.get("content", [])

        content = []
        for item in content_data:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    marks = [m.get("type") for m in item.get("marks", [])]
                    content.append(TextSpan(text=text, marks=marks))
                else:
                    nested = self._dict_to_block(item)
                    if nested:
                        content.append(nested)
            elif isinstance(item, str):
                content.append(TextSpan(text=item))

        return Block(type=block_type, content=content, attrs=attrs)

    def to_markdown(self) -> str:
        """Convert the entire document to markdown."""
        blocks = self.get_blocks()
        lines = []
        for block in blocks:
            md = block.to_markdown()
            if md:
                lines.append(md)
        return "\n\n".join(lines)

    def to_plain_text(self) -> str:
        """Convert the document to plain text."""
        blocks = self.get_blocks()
        return "\n".join(block.to_text() for block in blocks)


class DocumentWriter:
    """Writes content to TipTap Y.js documents.

    Uses the y-py API which differs from y.js JavaScript:
    - Elements are created via parent.push_xml_element(txn, tag)
    - Text nodes via parent.push_xml_text(txn), then text_node.push(txn, content)
    - No standalone constructors like Y.XmlElement("tag")
    """

    def __init__(self, doc: Y.YDoc, element_name: str = "prosemirror") -> None:
        self._doc = doc
        self._element_name = element_name

    def _get_root(self) -> Y.YXmlElement:
        """Get the root XML element."""
        return self._doc.get_xml_element(self._element_name)

    def append_paragraph(self, text: str) -> bytes:
        """Append a paragraph to the document.

        Returns the Y.js update bytes for broadcasting.
        """
        root = self._get_root()

        with self._doc.begin_transaction() as txn:
            para = root.push_xml_element(txn, "paragraph")
            text_node = para.push_xml_text(txn)
            text_node.push(txn, text)

        return Y.encode_state_as_update(self._doc)

    def insert_text_at(self, position: int, text: str) -> bytes:
        """Insert text at a position in the document.

        This is a simplified implementation - full cursor-aware editing
        would need ProseMirror position mapping.

        Returns the Y.js update bytes for broadcasting.
        """
        root = self._get_root()

        if len(root) == 0:
            return self.append_paragraph(text)

        with self._doc.begin_transaction() as txn:
            first = root.first_child
            if first is not None:
                # Find the text node inside the first element
                text_child = first.first_child
                if isinstance(text_child, Y.YXmlText):
                    text_child.insert(txn, position, text)

        return Y.encode_state_as_update(self._doc)

    def clear_document(self) -> bytes:
        """Clear all content from the document.

        Returns the Y.js update bytes for broadcasting.
        """
        root = self._get_root()

        with self._doc.begin_transaction() as txn:
            # Delete all children
            while len(root) > 0:
                root.delete(txn, 0, 1)

        return Y.encode_state_as_update(self._doc)

    def set_content_from_markdown(self, markdown: str) -> bytes:
        """Set document content from markdown.

        This is a basic implementation - full markdown parsing would need
        a proper markdown parser.

        Returns the Y.js update bytes for broadcasting.
        """
        # Clear existing content
        self.clear_document()

        root = self._get_root()

        with self._doc.begin_transaction() as txn:
            lines = markdown.split("\n")
            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if not line:
                    i += 1
                    continue

                # Heading
                if line.startswith("#"):
                    level = 0
                    while level < len(line) and line[level] == "#":
                        level += 1
                    text = line[level:].strip()
                    heading = root.push_xml_element(txn, "heading")
                    heading.set_attribute(txn, "level", str(level))
                    text_node = heading.push_xml_text(txn)
                    text_node.push(txn, text)

                # Code block
                elif line.startswith("```"):
                    lang = line[3:].strip()
                    code_lines = []
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("```"):
                        code_lines.append(lines[i])
                        i += 1
                    code_block = root.push_xml_element(txn, "codeBlock")
                    if lang:
                        code_block.set_attribute(txn, "language", lang)
                    text_node = code_block.push_xml_text(txn)
                    text_node.push(txn, "\n".join(code_lines))

                # Bullet list item
                elif line.startswith("- ") or line.startswith("* "):
                    text = line[2:]
                    list_item = root.push_xml_element(txn, "listItem")
                    para = list_item.push_xml_element(txn, "paragraph")
                    text_node = para.push_xml_text(txn)
                    text_node.push(txn, text)

                # Regular paragraph
                else:
                    para = root.push_xml_element(txn, "paragraph")
                    text_node = para.push_xml_text(txn)
                    text_node.push(txn, line)

                i += 1

        return Y.encode_state_as_update(self._doc)


__all__ = [
    "Block",
    "DocumentReader",
    "DocumentWriter",
    "TextSpan",
]
