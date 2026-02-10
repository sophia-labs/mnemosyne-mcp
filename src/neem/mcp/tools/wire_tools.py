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

from neem.hocuspocus import HocuspocusClient, WorkspaceReader
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


def _get_all_wires(doc: pycrdt.Doc) -> List[Dict[str, Any]]:
    """Read all wires from a workspace Y.Doc."""
    wires_map = _read_wires_map(doc)
    result: List[Dict[str, Any]] = []
    for wire_id in wires_map.keys():
        wire = wires_map.get(wire_id)
        if isinstance(wire, pycrdt.Map):
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
            "Also includes any custom predicates found in the graph's wires."
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

        # Scan workspace wires for any custom predicates
        try:
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
            if channel and channel.doc:
                all_wires = _get_all_wires(channel.doc)
                builtin_uris = set(BUILTIN_PREDICATES.keys())
                custom_uris: set[str] = set()

                for wire in all_wires:
                    pred = wire.get("predicate")
                    if pred and pred not in builtin_uris:
                        custom_uris.add(pred)

                for uri in sorted(custom_uris):
                    predicates.append({
                        "name": _get_predicate_short_name(uri),
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
    # create_wire
    # ─────────────────────────────────────────────────────────────────────────

    @server.tool(
        name="create_wire",
        title="Create Wire",
        description=(
            "Create a semantic wire (connection) between two documents or blocks. "
            "The wire is written directly to the workspace Y.Doc via CRDT, so it "
            "syncs in real-time to the browser and materializes to RDF automatically. "
            "Use list_wire_predicates to see available predicate URIs. "
            "Title/snippet previews are not populated at creation time — use the "
            "UI's refresh action or the backend API to fill them in later.\n\n"
            "Supports four connection modes: document-to-document (default), document-to-block, "
            "block-to-document, and block-to-block. Use source_block_id and/or target_block_id "
            "to wire at block granularity for precise conceptual connections.\n\n"
            "Always read both documents before wiring — choose predicates based on actual content, not just titles. "
            "Prefer block-level wires for precise links. Don't default to 'relatedTo' when a more specific "
            "predicate fits (supports, exemplifies, flowsInto, etc.)."
        ),
    )
    async def create_wire_tool(
        graph_id: str,
        source_document_id: str,
        target_document_id: str,
        predicate: Optional[str] = None,
        source_block_id: Optional[str] = None,
        target_block_id: Optional[str] = None,
        bidirectional: bool = False,
        target_graph_id: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        """Create a wire between two documents/blocks via Y.js CRDT."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not source_document_id or not source_document_id.strip():
            raise ValueError("source_document_id is required")
        if not target_document_id or not target_document_id.strip():
            raise ValueError("target_document_id is required")

        effective_predicate = _resolve_predicate(predicate) if predicate else f"{MNEMO_NS}isWiredTo"
        effective_target_graph = target_graph_id or graph_id
        wire_id = f"w-{uuid4().hex[:12]}"
        inverse_wire_id = f"{wire_id}-inv" if bidirectional else None

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Validate source document exists in workspace
            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")
            _validate_document_in_ws(channel.doc, graph_id.strip(), source_document_id.strip())
            # Validate target document if it's in the same graph
            if effective_target_graph.strip() == graph_id.strip():
                _validate_document_in_ws(
                    channel.doc, graph_id.strip(), target_document_id.strip(),
                )

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

                # Bidirectional wires create an inverse with source↔target swapped
                if bidirectional and inverse_wire_id:
                    _create_wire_in_doc(
                        doc,
                        wire_id=inverse_wire_id,
                        source_document_id=target_document_id.strip(),
                        target_graph_id=graph_id.strip(),
                        target_document_id=source_document_id.strip(),
                        predicate=effective_predicate,
                        source_block_id=target_block_id.strip() if target_block_id else None,
                        target_block_id=source_block_id.strip() if source_block_id else None,
                        bidirectional=True,
                        inverse_of=wire_id,
                    )

            await hp_client.transact_workspace(
                graph_id.strip(),
                _do_create,
                user_id=auth.user_id,
            )

            result: Dict[str, Any] = {
                "wire_id": wire_id,
                "predicate": _get_predicate_short_name(effective_predicate),
                "predicateLabel": _get_predicate_label(effective_predicate),
            }
            if bidirectional:
                result["bidirectional"] = True
            if inverse_wire_id:
                result["inverse_wire_id"] = inverse_wire_id

            # Trigger snippet refresh via backend API (best-effort)
            await _refresh_wire_snapshot(
                backend_config.base_url, auth, graph_id.strip(), wire_id
            )

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
