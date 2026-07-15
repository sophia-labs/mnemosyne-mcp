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
    """Pass a document ID through unchanged.

    Historically (P1 #13) this stripped a leading `doc-` prefix whenever the
    remaining tail looked UUID-shaped, on the theory that the bare form was
    "the canonical form most likely to match a workspace entry." That theory
    was wrong: workspace document entries are keyed by whatever ID form was
    used at creation time — bare UUID, `doc-<uuid>`, or slug — with no fixed
    rule across documents. Documents created via the web UI are commonly
    keyed by the full `doc-<uuid>` form, so unconditionally stripping the
    prefix corrupted an already-correct, already-canonical ID into one that
    matched nothing — every such document read as "not found" even though
    get_workspace / search_documents / get_user_location all report it as
    existing under the prefixed ID. That silently broke read_document and
    every other document_id-taking tool for any document created with a
    `doc-`-prefixed ID (bug reported informally several times over the
    prior month; root-caused 2026-07-15).

    Callers should pass document_id through exactly as returned by other
    tools. Tolerance for the (rarer) case of a caller supplying the "wrong"
    form now lives at the actual lookup point, where the real keyspace is
    known — see `neem.hocuspocus.workspace.WorkspaceReader.get_document`,
    which tries both forms against the live workspace map.
    """
    return value


def normalize_block_id_for_lookup(value: Any) -> Any:
    """Accept either `block-<hex>` or `<hex>` from callers; emit prefixed form."""
    return prefixed_block_id(value)


# Keys whose values are block IDs in the response shape and should be stripped
# of the `block-` prefix before serialization. List keys whose values are lists
# of block IDs go in `_BLOCK_ID_LIST_KEYS`. Includes wire-snapshot field names
# (sourceBlockId / targetBlockId) so wire payloads round-trip through the same
# bare-hex convention as everything else.
_BLOCK_ID_SINGULAR_KEYS = frozenset({
    "block_id",
    "blockId",
    "sourceBlockId",
    "targetBlockId",
    "source_block_id",
    "target_block_id",
})
_BLOCK_ID_LIST_KEYS = frozenset({"block_ids", "blockIds"})
# Keys whose VALUES are dicts whose KEYS are block IDs (e.g. read_document's
# wires.by_block: {"block-abc": count}). The walker can't rewrite keys
# anywhere it pleases, but it can target the specific containers where this
# convention is documented.
_BLOCK_ID_KEYED_DICT_KEYS = frozenset({"by_block", "byBlock"})


def bare_ids_in_result(obj: Any) -> Any:
    """In-place: strip `block-` from common block ID keys throughout obj.

    Walks dicts and lists. Modifies in place. Returns the same object for
    chainable convenience. Safe to apply at the return boundary of any
    tool that may emit block IDs.

    Handles three shapes:
    - `block_id` / `sourceBlockId` / `targetBlockId` etc. as singular values
    - `block_ids` / `blockIds` as list values
    - `by_block` / `byBlock` as a dict whose KEYS are block IDs
    Does NOT touch the canonical `id` key, since that can mean many things;
    tools that want to use `id` for blocks should pass through
    `bare_block_id` at the construction site.
    """
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k in _BLOCK_ID_SINGULAR_KEYS:
                if isinstance(v, str):
                    obj[k] = bare_block_id(v)
            elif k in _BLOCK_ID_LIST_KEYS:
                if isinstance(v, list):
                    obj[k] = [bare_block_id(x) if isinstance(x, str) else x for x in v]
            elif k in _BLOCK_ID_KEYED_DICT_KEYS:
                if isinstance(v, dict):
                    obj[k] = {bare_block_id(kk) if isinstance(kk, str) else kk: vv for kk, vv in v.items()}
            elif isinstance(v, (dict, list)):
                bare_ids_in_result(v)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                bare_ids_in_result(item)
    return obj
