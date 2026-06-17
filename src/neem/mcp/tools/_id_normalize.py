"""ID normalization helpers for the MCP tool layer.

The CRDT layer stores block IDs as `block-<hex>` and document IDs as
either slugs or `doc-<uuid>` depending on how they were created. Agents
working through the MCP surface should never need to care about the
prefix: the tool layer accepts either form on input and returns bare
form on output. These helpers centralize that translation.

Audit refs:
- P1 #5: Drop `block-` prefix and bare-echo overhead from block returns
- P1 #13: Strip `doc-` prefix and accept bare UUIDs everywhere
"""

from __future__ import annotations

from typing import Any, Iterable

_BLOCK_PREFIX = "block-"
_DOC_PREFIX = "doc-"


def bare_block_id(value: Any) -> Any:
    """Return the bare-hex form of a block ID.

    Accepts None / non-string values unchanged so this is safe to apply
    over loosely-typed maps before serialization. Strings without the
    prefix are returned unchanged (already bare).
    """
    if isinstance(value, str) and value.startswith(_BLOCK_PREFIX):
        return value[len(_BLOCK_PREFIX):]
    return value


def prefixed_block_id(value: Any) -> Any:
    """Return the `block-<hex>` form for CRDT-internal lookup.

    Inverse of `bare_block_id`. Accepts already-prefixed values without
    double-prefixing. Strings that look like nothing (empty, None,
    non-hex) pass through so callers can produce clear lookup errors.
    """
    if isinstance(value, str) and value and not value.startswith(_BLOCK_PREFIX):
        return f"{_BLOCK_PREFIX}{value}"
    return value


def bare_block_ids(values: Iterable[Any] | None) -> list[Any]:
    """Apply `bare_block_id` element-wise. Returns [] for None."""
    if not values:
        return []
    return [bare_block_id(v) for v in values]


def prefixed_block_ids(values: Iterable[Any] | None) -> list[Any]:
    """Apply `prefixed_block_id` element-wise. Returns [] for None."""
    if not values:
        return []
    return [prefixed_block_id(v) for v in values]


def bare_document_id(value: Any) -> Any:
    """Return the bare form of a document ID (strip `doc-` prefix)."""
    if isinstance(value, str) and value.startswith(_DOC_PREFIX):
        return value[len(_DOC_PREFIX):]
    return value


def normalize_document_id_for_lookup(value: Any) -> Any:
    """Accept either `doc-<uuid>` or `<uuid>` from callers.

    The CRDT layer stores some documents under slugs (e.g. `agent-delta`,
    `garden-design-antinomies`) and others under UUID-prefixed names
    (`doc-<uuid>`). Workspace lookups expect whichever form was used at
    creation time, so we don't unconditionally strip `doc-` — we only
    strip it if the remaining tail looks like a UUID-shaped string.

    Returns the canonical form most likely to match a workspace entry.
    Currently a pass-through for slug-shaped IDs; strips `doc-` only when
    the remainder is plausibly a UUID. Callers should treat the result
    as a hint, not a guarantee — workspace lookup remains authoritative.
    """
    if not isinstance(value, str) or not value:
        return value
    if value.startswith(_DOC_PREFIX):
        tail = value[len(_DOC_PREFIX):]
        # UUID-shaped: 8-4-4-4-12 hex with dashes, total 36 chars
        if len(tail) == 36 and tail.count("-") == 4:
            return tail
    return value


def normalize_block_id_for_lookup(value: Any) -> Any:
    """Accept either `block-<hex>` or `<hex>` from callers; emit prefixed form."""
    return prefixed_block_id(value)


# Keys whose values are block IDs in the response shape and should be stripped
# of the `block-` prefix before serialization. List keys whose values are lists
# of block IDs go in `_BLOCK_ID_LIST_KEYS`.
_BLOCK_ID_SINGULAR_KEYS = frozenset({"block_id", "blockId"})
_BLOCK_ID_LIST_KEYS = frozenset({"block_ids", "blockIds"})


def bare_ids_in_result(obj: Any) -> Any:
    """In-place: strip `block-` from common block ID keys throughout obj.

    Walks dicts and lists. Modifies in place. Returns the same object for
    chainable convenience. Safe to apply at the return boundary of any
    tool that may emit block IDs.

    Only acts on keys named `block_id` / `block_ids` (and JSON-style
    camelCase variants). Does NOT touch the canonical `id` key, since
    that can mean many things; tools that want to use `id` for blocks
    should pass through `bare_block_id` at the construction site.
    """
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k in _BLOCK_ID_SINGULAR_KEYS:
                if isinstance(v, str):
                    obj[k] = bare_block_id(v)
            elif k in _BLOCK_ID_LIST_KEYS:
                if isinstance(v, list):
                    obj[k] = [bare_block_id(x) if isinstance(x, str) else x for x in v]
            elif isinstance(v, (dict, list)):
                bare_ids_in_result(v)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                bare_ids_in_result(item)
    return obj
