"""
MCP tools for creating, querying, and traversing Wires (semantic connections).

Wires are semantic links between documents/blocks stored in the workspace Y.Doc.
These tools provide CRDT-native access to wire data — writes go through Y.js
transactions so changes sync in real-time to the browser and other clients.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

import pycrdt
from mcp.server.fastmcp import Context, FastMCP

from neem.hocuspocus import HocuspocusClient, WorkspaceReader, WorkspaceWriter
from neem.mcp.auth import MCPAuthContext
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_dev_user_id, get_internal_service_secret, validate_token_and_load

logger = LoggerFactory.get_logger("mcp.tools.wire_tools")

# ─────────────────────────────────────────────────────────────────────────────
# Built-in predicate taxonomy (mirrored from mnemosyne-platform models)
# ─────────────────────────────────────────────────────────────────────────────

MNEMO_NS = "http://mnemosyne.ai/vocab#"

BUILTIN_PREDICATES: Dict[str, Dict[str, str]] = {
    f"{MNEMO_NS}isWiredTo": {"label": "is wired to", "category": "Default"},
    # Quantity (Kant)
    f"{MNEMO_NS}partOf": {"label": "is part of", "category": "Quantity"},
    f"{MNEMO_NS}contains": {"label": "contains", "category": "Quantity"},
    f"{MNEMO_NS}exemplifies": {"label": "is an example of", "category": "Quantity"},
    # Quality (Kant)
    f"{MNEMO_NS}supports": {"label": "supports", "category": "Quality"},
    f"{MNEMO_NS}contradicts": {"label": "contradicts", "category": "Quality"},
    f"{MNEMO_NS}qualifies": {"label": "qualifies", "category": "Quality"},
    # Relation (Kant)
    f"{MNEMO_NS}causeOf": {"label": "causes", "category": "Relation"},
    f"{MNEMO_NS}consequenceOf": {"label": "is consequence of", "category": "Relation"},
    f"{MNEMO_NS}relatedTo": {"label": "is related to", "category": "Relation"},
    # Modality (Kant)
    f"{MNEMO_NS}requires": {"label": "requires", "category": "Modality"},
    f"{MNEMO_NS}enables": {"label": "enables", "category": "Modality"},
    f"{MNEMO_NS}precedes": {"label": "precedes", "category": "Modality"},
    # Connective Synthesis (Deleuze/Guattari)
    f"{MNEMO_NS}flowsInto": {"label": "flows into", "category": "Synthesis"},
    f"{MNEMO_NS}produces": {"label": "produces", "category": "Synthesis"},
    # Disjunctive Synthesis
    f"{MNEMO_NS}divergesFrom": {"label": "diverges from", "category": "Synthesis"},
    f"{MNEMO_NS}branchesTo": {"label": "branches to", "category": "Synthesis"},
    # Conjunctive Synthesis
    f"{MNEMO_NS}consumesWith": {"label": "consumes with", "category": "Synthesis"},
    f"{MNEMO_NS}intensifiesWith": {"label": "intensifies with", "category": "Synthesis"},
}


def _read_wires_map(doc: pycrdt.Doc) -> pycrdt.Map:
    """Get the wires Map from a workspace Y.Doc."""
    return doc.get("wires", type=pycrdt.Map)


def _wire_to_dict(wire_id: str, wire_map: pycrdt.Map) -> Dict[str, Any]:
    """Convert a pycrdt.Map wire entry to a plain dict."""
    data: Dict[str, Any] = {"id": wire_id}
    for key in wire_map.keys():
        data[key] = wire_map.get(key)
    return data


def _get_predicate_label(uri: str) -> str:
    """Get human-readable label for a predicate URI."""
    info = BUILTIN_PREDICATES.get(uri)
    if info:
        return info["label"]
    return uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]


def _get_predicate_short_name(uri: str) -> str:
    """Extract short name from a predicate URI (e.g. 'supports' from 'http://mnemosyne.ai/vocab#supports')."""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.split("/")[-1]


# Reverse lookup: short name -> full URI
_SHORT_TO_URI: Dict[str, str] = {
    _get_predicate_short_name(uri): uri for uri in BUILTIN_PREDICATES
}


def _resolve_predicate(predicate: str) -> str:
    """Resolve a predicate that may be a short name or full URI to a full URI."""
    if predicate.startswith("http://") or predicate.startswith("https://"):
        return predicate
    return _SHORT_TO_URI.get(predicate, f"{MNEMO_NS}{predicate}")


def _get_all_wires(doc: pycrdt.Doc, include_tombstoned: bool = False) -> List[Dict[str, Any]]:
    """Read all wires from a workspace Y.Doc.

    By default, wires with a ``_tombstonedAt`` timestamp are excluded.
    These are wires pending deletion after a block was removed — they'll
    be restored if the user undoes the deletion within the grace period.
    """
    wires_map = _read_wires_map(doc)
    result: List[Dict[str, Any]] = []
    for wire_id in wires_map.keys():
        wire = wires_map.get(wire_id)
        if isinstance(wire, pycrdt.Map):
            if not include_tombstoned and wire.get("_tombstonedAt") is not None:
                continue
            result.append(_wire_to_dict(wire_id, wire))
    return result


def _get_wires_for_document(
    doc: pycrdt.Doc,
    document_id: str,
    direction: str = "both",
) -> List[Dict[str, Any]]:
    """Get wires connected to a document, filtered by direction."""
    all_wires = _get_all_wires(doc)
    result: List[Dict[str, Any]] = []

    for wire in all_wires:
        # Skip inverse wires to avoid duplicates — the canonical wire already
        # carries the bidirectional flag and the inverse is an implementation detail
        if wire["id"].endswith("-inv"):
            continue

        is_source = wire.get("sourceDocumentId") == document_id
        is_target = wire.get("targetDocumentId") == document_id

        if direction == "outgoing" and is_source:
            result.append(wire)
        elif direction == "incoming" and is_target:
            result.append(wire)
        elif direction == "both" and (is_source or is_target):
            result.append(wire)

    return result


def _resolve_title_from_workspace(
    ws_doc: Optional[pycrdt.Doc], doc_id: Optional[str],
) -> Optional[str]:
    """Look up a document's title from the workspace documents map.

    This mirrors how the frontend resolves wire titles — the wire snapshot
    fields (sourceTitle/targetTitle) are often null because the backend's
    SPARQL-based title fetch depends on RDF materialization which can lag.
    The workspace Y.Doc documents map always has current titles.
    """
    if ws_doc is None or not doc_id:
        return None
    reader = WorkspaceReader(ws_doc)
    doc_data = reader.get_document(doc_id)
    if doc_data:
        return doc_data.get("title")
    return None


def _format_wire_summary(
    wire: Dict[str, Any],
    ws_doc: Optional[pycrdt.Doc] = None,
) -> Dict[str, Any]:
    """Format a wire dict for display, adding predicate label.

    When ws_doc is provided, resolves titles from the workspace documents map
    as a fallback when wire snapshot titles are null (matching frontend behavior).
    """
    predicate_uri = wire.get("predicate", "")
    source_title = wire.get("sourceTitle") or _resolve_title_from_workspace(
        ws_doc, wire.get("sourceDocumentId"),
    )
    target_title = wire.get("targetTitle") or _resolve_title_from_workspace(
        ws_doc, wire.get("targetDocumentId"),
    )
    result: Dict[str, Any] = {
        "id": wire["id"],
        "sourceDocumentId": wire.get("sourceDocumentId"),
        "targetDocumentId": wire.get("targetDocumentId"),
        "predicate": _get_predicate_short_name(predicate_uri),
        "predicateLabel": _get_predicate_label(predicate_uri),
        "bidirectional": bool(wire.get("bidirectional")),
    }
    if wire.get("sourceBlockId"):
        result["sourceBlockId"] = wire["sourceBlockId"]
    if wire.get("targetBlockId"):
        result["targetBlockId"] = wire["targetBlockId"]
    if source_title:
        result["sourceTitle"] = source_title
    if target_title:
        result["targetTitle"] = target_title
    if wire.get("sourceSnippet"):
        result["sourceSnippet"] = wire["sourceSnippet"]
    if wire.get("targetSnippet"):
        result["targetSnippet"] = wire["targetSnippet"]
    return result


async def _refresh_wire_snapshot(
    base_url: str,
    auth: "MCPAuthContext",
    graph_id: str,
    wire_id: str,
) -> bool:
    """Call the backend refresh endpoint to populate title/snippet previews.

    This is best-effort — the wire already exists in the Y.Doc. If the refresh
    fails (e.g., RDF hasn't materialized yet), the wire still works; previews
    will just be empty until the user triggers a manual refresh.

    Returns True if refresh succeeded, False otherwise.
    """
    url = f"{base_url}/wires/{graph_id}/{wire_id}/refresh"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.post(url, headers=auth.http_headers())
            if resp.status_code == 200:
                logger.debug(
                    "Wire snapshot refreshed",
                    extra_context={"wire_id": wire_id, "graph_id": graph_id},
                )
                return True
            else:
                logger.debug(
                    "Wire snapshot refresh returned non-200",
                    extra_context={
                        "wire_id": wire_id,
                        "status": resp.status_code,
                        "body": resp.text[:200],
                    },
                )
                return False
    except Exception as e:
        logger.debug(
            "Wire snapshot refresh failed (best-effort)",
            extra_context={"wire_id": wire_id, "error": str(e)},
        )
        return False


def _create_wire_in_doc(
    doc: pycrdt.Doc,
    wire_id: str,
    source_document_id: str,
    target_graph_id: str,
    target_document_id: str,
    predicate: str,
    *,
    source_block_id: Optional[str] = None,
    target_block_id: Optional[str] = None,
    bidirectional: bool = False,
    inverse_of: Optional[str] = None,
) -> None:
    """Create a wire entry in the workspace Y.Doc's wires Map.

    IMPORTANT: pycrdt.Map cannot be mutated after construction until it is
    integrated into a Y.Doc. We MUST build the complete dict upfront and
    pass it to pycrdt.Map() in one shot. See commit 5f6f489.
    """
    wires_map: pycrdt.Map = doc.get("wires", type=pycrdt.Map)
    now = datetime.now(timezone.utc).isoformat()

    wire_data: dict[str, Any] = {
        "sourceDocumentId": source_document_id,
        "targetGraphId": target_graph_id,
        "targetDocumentId": target_document_id,
        "predicate": predicate,
        "bidirectional": bidirectional,
        "createdAt": now,
        "snapshotAt": now,
    }

    if source_block_id:
        wire_data["sourceBlockId"] = source_block_id
    if target_block_id:
        wire_data["targetBlockId"] = target_block_id
    if inverse_of:
        wire_data["inverseOf"] = inverse_of

    wires_map[wire_id] = pycrdt.Map(wire_data)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registration
# ─────────────────────────────────────────────────────────────────────────────


def register_wire_tools(server: FastMCP) -> None:
    """Register wire-related MCP tools."""

    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping wire tool registration")
        return

    hp_client: Optional[HocuspocusClient] = getattr(server, "_hocuspocus_client", None)
    if hp_client is None:
        hp_client = HocuspocusClient(
            base_url=backend_config.base_url,
            token_provider=validate_token_and_load,
            dev_user_id=get_dev_user_id(),
            internal_service_secret=get_internal_service_secret(),
        )
        server._hocuspocus_client = hp_client  # type: ignore[attr-defined]

    # ─────────────────────────────────────────────────────────────────────────
    # Validation helper
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_document_in_ws(
        ws_doc: pycrdt.Doc, graph_id: str, document_id: str,
    ) -> None:
        """Verify a document exists in workspace. Raises RuntimeError if not."""
        reader = WorkspaceReader(ws_doc)
        if reader.get_document(document_id) is None:
            raise RuntimeError(
                f"Document '{document_id}' not found in graph '{graph_id}'. "
                f"Use get_workspace to see available documents."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # list_wire_predicates
    # ─────────────────────────────────────────────────────────────────────────

    @server.tool(
        name="list_wire_predicates",
        title="List Wire Predicates",
        description=(
            "Returns the taxonomy of semantic predicates available for wires. "
            "Predicates are organized by philosophical category "
            "(Quantity, Quality, Relation, Modality from Kant; Synthesis from Deleuze & Guattari). "
            "Categories: Quantity (partOf, contains, exemplifies), Quality (supports, contradicts, qualifies), "
            "Relation (causeOf, consequenceOf, relatedTo), Modality (requires, enables, precedes), "
            "Synthesis (flowsInto, produces, divergesFrom, branchesTo, consumesWith, intensifiesWith). "
            "Also includes any custom predicates found in the graph's wires. "
            "The returned short names can be passed directly to create_wires's predicate parameter."
        ),
    )
    async def list_wire_predicates_tool(
        graph_id: str,
        context: Context | None = None,
    ) -> str:
        """List all wire predicates (built-in + custom in use)."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        # Start with built-in predicates (short names only)
        predicates: List[Dict[str, Any]] = []
        for uri, info in BUILTIN_PREDICATES.items():
            predicates.append({
                "name": _get_predicate_short_name(uri),
                "label": info["label"],
                "category": info["category"],
            })

        # Scan workspace wires for any custom predicates not already in built-ins.
        # Dedup by short name to prevent the same predicate appearing twice
        # (e.g. a custom URI whose short name matches a built-in).
        try:
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
            if channel and channel.doc:
                all_wires = _get_all_wires(channel.doc)
                builtin_uris = set(BUILTIN_PREDICATES.keys())
                seen_names = {p["name"] for p in predicates}
                custom_uris: set[str] = set()

                for wire in all_wires:
                    pred = wire.get("predicate")
                    if pred and pred not in builtin_uris:
                        custom_uris.add(pred)

                for uri in sorted(custom_uris):
                    short_name = _get_predicate_short_name(uri)
                    if short_name not in seen_names:
                        seen_names.add(short_name)
                        predicates.append({
                            "name": short_name,
                            "label": _get_predicate_label(uri),
                            "category": "Custom",
                        })
        except Exception as e:
            logger.warning(
                "Failed to scan custom predicates",
                extra_context={"graph_id": graph_id, "error": str(e)},
            )

        return json.dumps({"predicates": predicates, "count": len(predicates)})

    # ─────────────────────────────────────────────────────────────────────────
    # create_wires
    # ─────────────────────────────────────────────────────────────────────────

    @server.tool(
        name="create_wires",
        title="Create Wires",
        description=(
            "Create semantic wire(s) between documents or blocks. "
            "For a single wire, pass source_document_id and target_document_id directly. "
            "For multiple wires, pass a `wires` list where each entry has "
            "source_document_id, target_document_id, and optional predicate, "
            "source_block_id, target_block_id, bidirectional, target_graph_id.\n\n"
            "Wires are written to the workspace Y.Doc via CRDT and sync in real-time. "
            "Use list_wire_predicates to see available predicates. "
            "Accepts short predicate names (e.g. 'supports', 'intensifiesWith').\n\n"
            "Always read both documents before wiring — choose predicates based on actual content, not just titles. "
            "Prefer block-level wires for precise links. Don't default to 'relatedTo' when a more specific "
            "predicate fits (supports, exemplifies, flowsInto, etc.)."
        ),
    )
    async def create_wires_tool(
        graph_id: str,
        source_document_id: Optional[str] = None,
        target_document_id: Optional[str] = None,
        predicate: Optional[str] = None,
        source_block_id: Optional[str] = None,
        target_block_id: Optional[str] = None,
        bidirectional: bool = False,
        target_graph_id: Optional[str] = None,
        wires: list[dict[str, Any]] | None = None,
        context: Context | None = None,
    ) -> str:
        """Create one or more wires between documents/blocks via Y.js CRDT."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")

        # Resolve single vs batch
        if wires is not None and source_document_id is not None:
            raise ValueError("Provide either 'source_document_id' (single) or 'wires' (batch), not both")

        graph_id = graph_id.strip()

        if wires is not None:
            # ── Batch mode ──
            if not wires:
                raise ValueError("wires list must not be empty")

            try:
                await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
                channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
                if channel is None:
                    raise RuntimeError(f"Workspace not connected: {graph_id}")

                created: List[Dict[str, Any]] = []
                errors: List[Dict[str, Any]] = []

                for i, spec in enumerate(wires):
                    try:
                        src_doc = (spec.get("source_document_id") or "").strip()
                        tgt_doc = (spec.get("target_document_id") or "").strip()
                        if not src_doc or not tgt_doc:
                            errors.append({"index": i, "error": "source_document_id and target_document_id are required"})
                            continue

                        pred_input = spec.get("predicate")
                        eff_pred = _resolve_predicate(pred_input) if pred_input else f"{MNEMO_NS}isWiredTo"
                        eff_tgt_graph = (spec.get("target_graph_id") or graph_id).strip()
                        bidir = bool(spec.get("bidirectional", False))
                        src_block = (spec.get("source_block_id") or "").strip() or None
                        tgt_block = (spec.get("target_block_id") or "").strip() or None

                        _validate_document_in_ws(channel.doc, graph_id, src_doc)
                        if eff_tgt_graph == graph_id:
                            _validate_document_in_ws(channel.doc, graph_id, tgt_doc)

                        wid = f"w-{uuid4().hex[:12]}"
                        inv_id = f"{wid}-inv" if bidir else None

                        def _do_create(doc: pycrdt.Doc, *, _wid=wid, _sd=src_doc, _td=tgt_doc,
                                       _tg=eff_tgt_graph, _pred=eff_pred, _sb=src_block,
                                       _tb=tgt_block, _bi=bidir, _inv=inv_id) -> None:
                            _create_wire_in_doc(
                                doc, wire_id=_wid, source_document_id=_sd,
                                target_graph_id=_tg, target_document_id=_td,
                                predicate=_pred, source_block_id=_sb,
                                target_block_id=_tb, bidirectional=_bi, inverse_of=_inv,
                            )
                            if _bi and _inv:
                                _create_wire_in_doc(
                                    doc, wire_id=_inv, source_document_id=_td,
                                    target_graph_id=graph_id, target_document_id=_sd,
                                    predicate=_pred, source_block_id=_tb,
                                    target_block_id=_sb, bidirectional=True, inverse_of=_wid,
                                )

                        await hp_client.transact_workspace(graph_id, _do_create, user_id=auth.user_id)

                        result_entry: Dict[str, Any] = {
                            "wire_id": wid,
                            "predicate": _get_predicate_short_name(eff_pred),
                        }
                        if bidir:
                            result_entry["bidirectional"] = True
                        created.append(result_entry)

                        await _refresh_wire_snapshot(backend_config.base_url, auth, graph_id, wid)

                    except Exception as e:
                        errors.append({"index": i, "error": str(e)})

                output: Dict[str, Any] = {"created": created, "created_count": len(created)}
                if errors:
                    output["errors"] = errors
                    output["error_count"] = len(errors)
                return json.dumps(output)

            except Exception as e:
                raise RuntimeError(f"Failed to create wires: {e}")

        else:
            # ── Single mode ──
            if not source_document_id or not source_document_id.strip():
                raise ValueError("source_document_id is required (or pass 'wires' for batch)")
            if not target_document_id or not target_document_id.strip():
                raise ValueError("target_document_id is required")

            effective_predicate = _resolve_predicate(predicate) if predicate else f"{MNEMO_NS}isWiredTo"
            effective_target_graph = target_graph_id or graph_id
            wire_id = f"w-{uuid4().hex[:12]}"
            inverse_wire_id = f"{wire_id}-inv" if bidirectional else None

            try:
                await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
                channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
                if channel is None:
                    raise RuntimeError(f"Workspace not connected: {graph_id}")
                _validate_document_in_ws(channel.doc, graph_id, source_document_id.strip())
                if effective_target_graph.strip() == graph_id:
                    _validate_document_in_ws(channel.doc, graph_id, target_document_id.strip())

                def _do_create(doc: pycrdt.Doc) -> None:
                    _create_wire_in_doc(
                        doc,
                        wire_id=wire_id,
                        source_document_id=source_document_id.strip(),
                        target_graph_id=effective_target_graph.strip(),
                        target_document_id=target_document_id.strip(),
                        predicate=effective_predicate,
                        source_block_id=source_block_id.strip() if source_block_id else None,
                        target_block_id=target_block_id.strip() if target_block_id else None,
                        bidirectional=bidirectional,
                        inverse_of=inverse_wire_id,
                    )
                    if bidirectional and inverse_wire_id:
                        _create_wire_in_doc(
                            doc,
                            wire_id=inverse_wire_id,
                            source_document_id=target_document_id.strip(),
                            target_graph_id=graph_id,
                            target_document_id=source_document_id.strip(),
                            predicate=effective_predicate,
                            source_block_id=target_block_id.strip() if target_block_id else None,
                            target_block_id=source_block_id.strip() if source_block_id else None,
                            bidirectional=True,
                            inverse_of=wire_id,
                        )

                await hp_client.transact_workspace(graph_id, _do_create, user_id=auth.user_id)

                result: Dict[str, Any] = {
                    "wire_id": wire_id,
                    "predicate": _get_predicate_short_name(effective_predicate),
                    "predicateLabel": _get_predicate_label(effective_predicate),
                }
                if bidirectional:
                    result["bidirectional"] = True
                if inverse_wire_id:
                    result["inverse_wire_id"] = inverse_wire_id

                await _refresh_wire_snapshot(backend_config.base_url, auth, graph_id, wire_id)

                return json.dumps(result)

            except Exception as e:
                raise RuntimeError(f"Failed to create wire: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # get_wires
    # ─────────────────────────────────────────────────────────────────────────

    @server.tool(
        name="get_wires",
        title="Get Wires for Document",
        description=(
            "Returns all semantic wires (connections) for a document. "
            "Filter by direction: 'outgoing' (from this doc), 'incoming' (to this doc), "
            "or 'both' (default). Optionally filter by predicate URI. "
            "Each wire includes source/target document IDs, predicate, "
            "bidirectional flag, and cached title/snippet previews."
        ),
    )
    async def get_wires_tool(
        graph_id: str,
        document_id: str,
        direction: str = "both",
        predicate: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        """Get wires connected to a document."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(f"direction must be 'outgoing', 'incoming', or 'both', got '{direction}'")

        try:
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            # Validate document exists
            _validate_document_in_ws(channel.doc, graph_id, document_id)

            wires = _get_wires_for_document(channel.doc, document_id, direction)

            # Filter by predicate if specified (accept short names or full URIs)
            if predicate:
                resolved = _resolve_predicate(predicate)
                wires = [w for w in wires if w.get("predicate") == resolved]

            formatted = [_format_wire_summary(w, ws_doc=channel.doc) for w in wires]

            return json.dumps({"wires": formatted, "count": len(formatted)})

        except Exception as e:
            raise RuntimeError(f"Failed to get wires: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # traverse_wires
    # ─────────────────────────────────────────────────────────────────────────

    @server.tool(
        name="traverse_wires",
        title="Traverse Wire Graph",
        description=(
            "Traverse the wire graph starting from a document, following connections "
            "up to a specified depth. Returns all reachable documents and the paths "
            "connecting them. Useful for discovering related content, understanding "
            "document clusters, and exploring knowledge structure. "
            "Results include document titles and wire metadata, making this useful for "
            "quickly mapping the conceptual neighborhood of a document. "
            "Optionally filter by predicate to follow only specific relationship types."
        ),
    )
    async def traverse_wires_tool(
        graph_id: str,
        document_id: str,
        max_depth: int = 2,
        predicate: Optional[str] = None,
        direction: str = "both",
        context: Context | None = None,
    ) -> str:
        """Traverse wire connections from a starting document."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if max_depth < 1 or max_depth > 10:
            raise ValueError("max_depth must be between 1 and 10")

        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(f"direction must be 'outgoing', 'incoming', or 'both', got '{direction}'")

        try:
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            # Validate starting document exists
            _validate_document_in_ws(channel.doc, graph_id, document_id)

            doc = channel.doc

            # BFS traversal
            visited: set[str] = {document_id}
            seen_wires: set[str] = set()
            start_title = _resolve_title_from_workspace(doc, document_id)
            nodes: Dict[str, Dict[str, Any]] = {
                document_id: {"id": document_id, "depth": 0, "title": start_title},
            }
            edges: List[Dict[str, Any]] = []
            queue: deque[tuple[str, int]] = deque([(document_id, 0)])

            while queue:
                current_doc_id, depth = queue.popleft()
                if depth >= max_depth:
                    continue

                wires = _get_wires_for_document(doc, current_doc_id, direction)

                if predicate:
                    resolved = _resolve_predicate(predicate)
                    wires = [w for w in wires if w.get("predicate") == resolved]

                for wire in wires:
                    wire_id = wire["id"]

                    # Deduplicate: each wire appears once regardless of traversal path
                    if wire_id in seen_wires:
                        # Still need to enqueue the other end for BFS even if edge is deduped
                        source_id = wire.get("sourceDocumentId")
                        target_id = wire.get("targetDocumentId")
                        other_id = target_id if source_id == current_doc_id else source_id
                        if other_id and other_id not in visited:
                            visited.add(other_id)
                            other_title = _resolve_title_from_workspace(doc, other_id)
                            nodes[other_id] = {
                                "id": other_id,
                                "depth": depth + 1,
                                "title": other_title,
                            }
                            queue.append((other_id, depth + 1))
                        continue

                    seen_wires.add(wire_id)

                    source_id = wire.get("sourceDocumentId")
                    target_id = wire.get("targetDocumentId")
                    pred_uri = wire.get("predicate", "")

                    # Determine which end is "the other document"
                    other_id = target_id if source_id == current_doc_id else source_id

                    if not other_id:
                        continue

                    edge: Dict[str, Any] = {
                        "wireId": wire_id,
                        "source": source_id,
                        "target": target_id,
                        "predicate": _get_predicate_short_name(pred_uri),
                        "predicateLabel": _get_predicate_label(pred_uri),
                        "bidirectional": bool(wire.get("bidirectional")),
                    }

                    # Include block IDs when present — block-level precision matters
                    if wire.get("sourceBlockId"):
                        edge["sourceBlockId"] = wire["sourceBlockId"]
                    if wire.get("targetBlockId"):
                        edge["targetBlockId"] = wire["targetBlockId"]

                    # Include truncated snippets for semantic context without full payload
                    src_snip = wire.get("sourceSnippet")
                    tgt_snip = wire.get("targetSnippet")
                    if src_snip:
                        edge["sourceSnippet"] = src_snip[:80] + ("..." if len(src_snip) > 80 else "")
                    if tgt_snip:
                        edge["targetSnippet"] = tgt_snip[:80] + ("..." if len(tgt_snip) > 80 else "")

                    edges.append(edge)

                    if other_id not in visited:
                        visited.add(other_id)
                        other_title = _resolve_title_from_workspace(doc, other_id)
                        nodes[other_id] = {
                            "id": other_id,
                            "depth": depth + 1,
                            "title": other_title,
                        }
                        queue.append((other_id, depth + 1))

            return json.dumps({
                "nodes": list(nodes.values()),
                "edges": edges,
            })

        except Exception as e:
            raise RuntimeError(f"Failed to traverse wires: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # delete_wires
    # ─────────────────────────────────────────────────────────────────────────

    @server.tool(
        name="delete_wires",
        title="Delete Wires",
        description=(
            "Delete semantic wires. Three modes:\n"
            "1. By ID: pass wire_id (single) or wire_ids (batch) to delete specific wires.\n"
            "2. By document: pass document_id to delete ALL wires connected to that document.\n"
            "3. By block: pass document_id + block_id to delete wires connected to that specific block.\n\n"
            "Bidirectional wires are automatically cleaned up — deleting either "
            "the canonical or inverse wire removes both.\n"
            "Use get_wires to find wire IDs before deleting."
        ),
    )
    async def delete_wires_tool(
        graph_id: str,
        wire_id: Optional[str] = None,
        wire_ids: list[str] | None = None,
        document_id: Optional[str] = None,
        block_id: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        """Delete wires by ID, by document, or by block."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")

        graph_id = graph_id.strip()

        # Validate parameter combinations
        has_ids = wire_id is not None or wire_ids is not None
        has_doc = document_id is not None

        if has_ids and has_doc:
            raise ValueError(
                "Provide wire IDs (wire_id/wire_ids) OR document_id, not both"
            )
        if block_id is not None and not has_doc:
            raise ValueError("block_id requires document_id")
        if not has_ids and not has_doc:
            raise ValueError(
                "Provide wire_id, wire_ids, or document_id (+ optional block_id)"
            )

        try:
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)

            # Mode: delete by document/block match
            if has_doc:
                doc_id = document_id.strip()
                blk_id = block_id.strip() if block_id else None
                deleted: list[str] = []

                def _do_delete_matching(doc: pycrdt.Doc) -> None:
                    writer = WorkspaceWriter(doc)
                    deleted.extend(writer.delete_wires_matching(doc_id, blk_id))

                await hp_client.transact_workspace(
                    graph_id, _do_delete_matching, user_id=auth.user_id
                )

                return json.dumps({
                    "deleted": deleted,
                    "deleted_count": len(deleted),
                    "scope": {"document_id": doc_id, "block_id": blk_id},
                })

            # Mode: delete by explicit IDs
            if wire_ids is not None and wire_id is not None:
                raise ValueError(
                    "Provide either 'wire_id' (single) or 'wire_ids' (batch), not both"
                )

            ids_to_delete: List[str]
            if wire_ids is not None:
                if not wire_ids:
                    raise ValueError("wire_ids list must not be empty")
                ids_to_delete = wire_ids
            else:
                ids_to_delete = [wire_id]  # type: ignore[list-item]

            all_deleted: List[str] = []
            errors: List[Dict[str, Any]] = []

            for i, wid in enumerate(ids_to_delete):
                try:
                    wid = wid.strip()
                    deleted_ids: list[str] = []

                    def _do_delete(doc: pycrdt.Doc, *, _wid=wid) -> None:
                        writer = WorkspaceWriter(doc)
                        deleted_ids.extend(writer.delete_wire(_wid))

                    await hp_client.transact_workspace(
                        graph_id, _do_delete, user_id=auth.user_id
                    )
                    all_deleted.extend(deleted_ids)

                except Exception as e:
                    errors.append({"wire_id": wid, "error": str(e)})

            output: Dict[str, Any] = {
                "deleted": all_deleted,
                "deleted_count": len(all_deleted),
            }
            if errors:
                output["errors"] = errors
                output["error_count"] = len(errors)
            return json.dumps(output)

        except Exception as e:
            raise RuntimeError(f"Failed to delete wires: {e}")
