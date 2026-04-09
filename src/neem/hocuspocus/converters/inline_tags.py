"""Inline tag markers for MCP write tools.

Allows embedding tag directives in content that are automatically
applied as block-level CRDT attributes after block IDs are assigned.

Markdown/plain text syntax (end of line, after valuations if present):
    {#decision}         — tag "decision"
    {#todo:7d}          — tag "todo" with 7-day expiration
    {#pragma}           — tag "pragma"
    {!4,+2}{#decision}  — valuation + tag in the same breath

XML syntax (block-level attributes):
    <paragraph data-tags='["decision","pragma"]'>Content</paragraph>

Markers are stripped from content before CRDT write. Tags are applied
as data-tags JSON array attributes on blocks after IDs are assigned.
Follows the same preprocess→placeholder→postprocess pattern as
inline_valuations.py.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

# PUA delimiters — distinct from valuation placeholders
_PH_START = "\uE002"
_PH_END = "\uE003"

# Matches {#tag} or {#tag:duration} at end of line (optional trailing whitespace)
# Can appear multiple times: {#decision}{#pragma}
# Duration: 7d, 14d, 30d, 90d, or ISO date 2026-04-21
_TAG_MARKER_RE = re.compile(
    r"\{#([a-zA-Z0-9_-]+)(?::([a-zA-Z0-9_-]+))?\}",
)

# Matches all tag markers at end of line (after optional valuation marker)
_TAG_MARKERS_SUFFIX_RE = re.compile(
    r"(\{#[a-zA-Z0-9_-]+(?::[a-zA-Z0-9_-]+)?\}\s*)+$",
)

# Matches PUA-delimited placeholder in XML text
_PH_RE = re.compile(
    re.escape(_PH_START) + r"TAG:([^" + re.escape(_PH_END) + r"]*)" + re.escape(_PH_END)
)

# Block-level XML elements that can carry tags
_BLOCK_TAGS = frozenset({
    "paragraph", "heading", "listItem", "codeBlock",
    "blockquote", "horizontalRule", "table",
})


@dataclass
class PendingTag:
    """Tags extracted from content, waiting for block ID assignment."""
    block_index: int
    tags: list[str] = field(default_factory=list)
    expirations: dict[str, str] = field(default_factory=dict)  # tag → duration/date


def preprocess_tags(text: str) -> str:
    """Replace {#...} markers with PUA-delimited placeholders.

    Skips markers inside code fences. The placeholders survive through
    markdown→XML conversion and are extracted by postprocess_tags().
    """
    lines = text.split("\n")
    result: list[str] = []
    in_code_fence = False

    for line in lines:
        stripped = line.lstrip()

        # Track code fences
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            result.append(line)
            continue

        if in_code_fence:
            result.append(line)
            continue

        # Find all tag markers in the line
        suffix_match = _TAG_MARKERS_SUFFIX_RE.search(line)
        if not suffix_match:
            result.append(line)
            continue

        # Extract individual tags from the suffix
        suffix = suffix_match.group(0)
        tags: list[str] = []
        expirations: dict[str, str] = {}

        for m in _TAG_MARKER_RE.finditer(suffix):
            tag_name = m.group(1).lower()
            duration = m.group(2)
            tags.append(tag_name)
            if duration:
                expirations[tag_name] = duration

        if not tags:
            result.append(line)
            continue

        # Build placeholder
        tag_part = ",".join(tags)
        exp_part = ";".join(f"{k}={v}" for k, v in expirations.items()) if expirations else ""
        placeholder_content = f"{tag_part}|{exp_part}" if exp_part else tag_part
        placeholder = f"{_PH_START}TAG:{placeholder_content}{_PH_END}"

        # Replace marker suffix with placeholder
        line = line[:suffix_match.start()] + placeholder + line[suffix_match.end():]

        result.append(line)

    return "\n".join(result)


def postprocess_tags(xml_str: str) -> tuple[str, list[PendingTag]]:
    """Extract PUA placeholders from XML, recording block indices.

    Returns (clean_xml, pending_tags). Placeholders are stripped from
    the XML content. Each PendingTag records the block index and tags.
    """
    pending: list[PendingTag] = []

    try:
        wrapped = f"<root>{xml_str}</root>"
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        clean = _PH_RE.sub("", xml_str)
        return clean, []

    block_index = 0
    for child in root:
        if child.tag in _BLOCK_TAGS:
            full_text = _get_all_text(child)
            m = _PH_RE.search(full_text)
            if m:
                pt = _parse_placeholder(m.group(1), block_index)
                if pt:
                    pending.append(pt)
            block_index += 1

    # Strip all placeholders from XML string
    clean = _PH_RE.sub("", xml_str)
    clean = re.sub(r"\s+</", "</", clean)

    return clean, pending


def extract_xml_tags(xml_str: str) -> tuple[str, list[PendingTag]]:
    """Extract data-tags attributes from XML block elements.

    Returns (clean_xml, pending_tags) with the attributes removed.
    """
    pending: list[PendingTag] = []

    try:
        wrapped = f"<root>{xml_str}</root>"
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return xml_str, []

    block_index = 0
    modified = False
    for child in root:
        if child.tag in _BLOCK_TAGS:
            raw_tags = child.attrib.pop("data-tags", None)

            if raw_tags is not None:
                modified = True
                try:
                    parsed = json.loads(raw_tags)
                    if isinstance(parsed, list):
                        tags = [str(t).lower() for t in parsed if t]
                        if tags:
                            pending.append(PendingTag(
                                block_index=block_index,
                                tags=tags,
                            ))
                except (json.JSONDecodeError, TypeError):
                    pass
            block_index += 1

    if not modified:
        return xml_str, []

    parts = []
    for child in root:
        parts.append(ET.tostring(child, encoding="unicode"))
    return "".join(parts), pending


def map_tags_to_block_ids(
    pending: list[PendingTag],
    block_ids: list[str],
) -> list[dict]:
    """Map block indices to block IDs, producing entries for tag application.

    Returns list of dicts: {"block_id": str, "tags": list[str], "expirations": dict}
    Entries with out-of-range block indices are silently skipped.
    """
    entries: list[dict] = []
    for pt in pending:
        if 0 <= pt.block_index < len(block_ids):
            entry: dict = {
                "block_id": block_ids[pt.block_index],
                "tags": pt.tags,
            }
            if pt.expirations:
                entry["expirations"] = pt.expirations
            entries.append(entry)
    return entries


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _get_all_text(elem: ET.Element) -> str:
    """Get all text content from an element subtree."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_get_all_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _parse_placeholder(content: str, block_index: int) -> Optional[PendingTag]:
    """Parse TAG:tag1,tag2|exp1=val1;exp2=val2 placeholder content."""
    if not content:
        return None

    parts = content.split("|", 1)
    tag_str = parts[0]
    exp_str = parts[1] if len(parts) > 1 else ""

    tags = [t.strip() for t in tag_str.split(",") if t.strip()]
    if not tags:
        return None

    expirations: dict[str, str] = {}
    if exp_str:
        for pair in exp_str.split(";"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                expirations[k.strip()] = v.strip()

    return PendingTag(
        block_index=block_index,
        tags=tags,
        expirations=expirations,
    )
