"""Inline valuation markers for MCP write tools.

Allows embedding valuation directives in content that are automatically
applied after block IDs are assigned during the CRDT write.

Markdown/plain text syntax (end of line):
    {!3}        — importance 3
    {!,+2}      — valence +2
    {!4,-3}     — importance 4, valence -3

XML syntax (block-level attributes):
    <paragraph data-val-importance="4" data-val-valence="-3">Content</paragraph>

Markers are stripped from content before CRDT write. Follows the same
preprocess→placeholder→postprocess pattern as tab-indent handling in
markdown_to_xml.py.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

# PUA delimiters — survive html.escape(), ET.fromstring(), and mistune
_PH_START = "\uE000"
_PH_END = "\uE001"

# Matches {!N}, {!,+N}, {!N,+M}, {!N,-M} at end of line (optional trailing
# whitespace). A trailing run of inline tag markers ({#tag}, {#tag:dur}) is
# tolerated via lookahead so the combined "value and tag in one breath" form
# `{!4,+2}{#decision}` still matches — the valuation marker is consumed, the
# tag markers are left intact for preprocess_tags (which runs next).
# Group 1: importance digit (optional)
# Group 2: valence with sign (optional)
_MARKER_RE = re.compile(
    r"\{!(\d)?(?:,([+-]?\d))?\}(?=(?:\s*\{#[a-zA-Z0-9_-]+(?::[a-zA-Z0-9_-]+)?\})*\s*$)",
    re.MULTILINE,
)

# Matches PUA-delimited placeholder in XML text
_PH_RE = re.compile(
    re.escape(_PH_START) + r"VAL:([^" + re.escape(_PH_END) + r"]*)" + re.escape(_PH_END)
)

# Block-level XML elements that can carry valuations
_BLOCK_TAGS = frozenset({
    "paragraph", "heading", "listItem", "codeBlock",
    "blockquote", "horizontalRule", "table",
})


@dataclass
class PendingValuation:
    """A valuation extracted from content, waiting for block ID assignment."""
    block_index: int
    importance: Optional[int] = None  # 0-5
    valence: Optional[int] = None     # -5 to +5


def preprocess_valuations(text: str) -> str:
    """Replace {!...} markers with PUA-delimited placeholders.

    Skips markers inside code fences. The placeholders survive through
    markdown→XML conversion and are extracted by postprocess_valuations().
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

        # Try to match a valuation marker at end of line
        m = _MARKER_RE.search(line)
        if m:
            imp_str = m.group(1)  # digit or None
            val_str = m.group(2)  # signed digit or None

            # Validate ranges
            imp = int(imp_str) if imp_str is not None else None
            val = int(val_str) if val_str is not None else None

            if imp is not None and not (0 <= imp <= 5):
                result.append(line)
                continue
            if val is not None and not (-5 <= val <= 5):
                result.append(line)
                continue
            if imp is None and val is None:
                result.append(line)
                continue

            # Build placeholder: VAL:imp:val (empty string for None)
            imp_part = str(imp) if imp is not None else ""
            val_part = str(val) if val is not None else ""
            placeholder = f"{_PH_START}VAL:{imp_part}:{val_part}{_PH_END}"

            # Replace marker with placeholder
            line = line[:m.start()] + placeholder + line[m.end():]

        result.append(line)

    return "\n".join(result)


def postprocess_valuations(xml_str: str) -> tuple[str, list[PendingValuation]]:
    """Extract PUA placeholders from XML, recording block indices.

    Returns (clean_xml, pending_valuations). Placeholders are stripped from
    the XML content. Each PendingValuation records the block index (0-based)
    where the marker appeared.
    """
    pending: list[PendingValuation] = []

    # Parse XML to find placeholders by block position
    try:
        wrapped = f"<root>{xml_str}</root>"
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        # Fallback: strip placeholders by regex, return empty valuations
        clean = _PH_RE.sub("", xml_str)
        return clean, []

    block_index = 0
    for child in root:
        if child.tag in _BLOCK_TAGS:
            # Check all text in this block subtree for placeholder
            full_text = _get_all_text(child)
            m = _PH_RE.search(full_text)
            if m:
                pv = _parse_placeholder(m.group(1), block_index)
                if pv:
                    pending.append(pv)
            block_index += 1

    # Strip all placeholders from XML string
    clean = _PH_RE.sub("", xml_str)
    # Clean up any trailing whitespace before closing tags caused by stripping
    clean = re.sub(r"\s+</", "</", clean)

    return clean, pending


def extract_xml_valuations(xml_str: str) -> tuple[str, list[PendingValuation]]:
    """Extract data-val-* attributes from XML block elements.

    Returns (clean_xml, pending_valuations) with the attributes removed.
    """
    pending: list[PendingValuation] = []

    try:
        wrapped = f"<root>{xml_str}</root>"
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return xml_str, []

    block_index = 0
    modified = False
    for child in root:
        if child.tag in _BLOCK_TAGS:
            imp_str = child.attrib.pop("data-val-importance", None)
            val_str = child.attrib.pop("data-val-valence", None)

            if imp_str is not None or val_str is not None:
                modified = True
                imp = _safe_int(imp_str, 0, 5) if imp_str is not None else None
                val = _safe_int(val_str, -5, 5) if val_str is not None else None
                if imp is not None or val is not None:
                    pending.append(PendingValuation(
                        block_index=block_index,
                        importance=imp,
                        valence=val,
                    ))
            block_index += 1

    if not modified:
        return xml_str, []

    # Reconstruct XML without the data-val-* attributes
    parts = []
    for child in root:
        parts.append(ET.tostring(child, encoding="unicode"))
    return "".join(parts), pending


def map_valuations_to_block_ids(
    pending: list[PendingValuation],
    block_ids: list[str],
    document_id: str,
) -> list[dict]:
    """Map block indices to block IDs, producing batch entries for value_tool.

    Entries with out-of-range block indices are silently skipped.
    """
    entries: list[dict] = []
    for pv in pending:
        if 0 <= pv.block_index < len(block_ids):
            entry: dict = {
                "document_id": document_id,
                "block_id": block_ids[pv.block_index],
            }
            if pv.importance is not None:
                entry["importance"] = pv.importance
            if pv.valence is not None:
                entry["valence"] = pv.valence
            entries.append(entry)
    return entries


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _get_all_text(elem: ET.Element) -> str:
    """Recursively extract all text from an XML element."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_get_all_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _parse_placeholder(payload: str, block_index: int) -> Optional[PendingValuation]:
    """Parse 'imp:val' from a PUA placeholder payload."""
    parts = payload.split(":", 1)
    if len(parts) != 2:
        return None

    imp_str, val_str = parts
    imp = int(imp_str) if imp_str else None
    val = int(val_str) if val_str else None

    if imp is None and val is None:
        return None

    return PendingValuation(block_index=block_index, importance=imp, valence=val)


def _safe_int(s: str, lo: int, hi: int) -> Optional[int]:
    """Parse an integer string, returning None if out of range."""
    try:
        v = int(s)
    except (TypeError, ValueError):
        return None
    if lo <= v <= hi:
        return v
    return None
