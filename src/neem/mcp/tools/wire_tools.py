"""
MCP tools for querying and traversing Wires (semantic connections).

Wires are semantic links between documents/blocks stored in the workspace Y.Doc.
These tools provide read access to wire data for graph exploration, knowledge
discovery, and understanding document relationships.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Dict, List, Optional

import pycrdt
from mcp.server.fastmcp import Context, FastMCP

from neem.hocuspocus import HocuspocusClient
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


def _format_wire_summary(wire: Dict[str, Any]) -> Dict[str, Any]:
    """Format a wire dict for display, adding predicate label."""
    predicate = wire.get("predicate", "")
    return {
        "id": wire["id"],
        "sourceDocumentId": wire.get("sourceDocumentId"),
        "targetDocumentId": wire.get("targetDocumentId"),
        "predicate": predicate,
        "predicateLabel": _get_predicate_label(predicate),
        "bidirectional": bool(wire.get("bidirectional")),
        "sourceBlockId": wire.get("sourceBlockId"),
        "targetBlockId": wire.get("targetBlockId"),
        "sourceTitle": wire.get("sourceTitle"),
        "targetTitle": wire.get("targetTitle"),
        "sourceSnippet": wire.get("sourceSnippet"),
        "targetSnippet": wire.get("targetSnippet"),
        "createdAt": wire.get("createdAt"),
    }


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
    # list_wire_predicates
    # ─────────────────────────────────────────────────────────────────────────

    @server.tool(
        name="list_wire_predicates",
        title="List Wire Predicates",
        description=(
            "Returns the taxonomy of semantic predicates available for wires. "
            "Predicates are organized by philosophical category (Quantity, Quality, "
            "Relation, Modality from Kant; Synthesis from Deleuze & Guattari). "
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

        # Start with built-in predicates
        predicates: List[Dict[str, Any]] = []
        for uri, info in BUILTIN_PREDICATES.items():
            predicates.append({
                "uri": uri,
                "label": info["label"],
                "category": info["category"],
                "builtin": True,
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
                        "uri": uri,
                        "label": _get_predicate_label(uri),
                        "category": "Custom",
                        "builtin": False,
                    })
        except Exception as e:
            logger.warning(
                "Failed to scan custom predicates",
                extra_context={"graph_id": graph_id, "error": str(e)},
            )

        return json.dumps({"predicates": predicates, "count": len(predicates)}, indent=2)

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

            wires = _get_wires_for_document(channel.doc, document_id, direction)

            # Filter by predicate if specified
            if predicate:
                wires = [w for w in wires if w.get("predicate") == predicate]

            formatted = [_format_wire_summary(w) for w in wires]

            return json.dumps({
                "document_id": document_id,
                "direction": direction,
                "wires": formatted,
                "count": len(formatted),
            }, indent=2)

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

            doc = channel.doc

            # BFS traversal
            visited: set[str] = {document_id}
            nodes: Dict[str, Dict[str, Any]] = {
                document_id: {"id": document_id, "depth": 0, "title": None},
            }
            edges: List[Dict[str, Any]] = []
            queue: deque[tuple[str, int]] = deque([(document_id, 0)])

            while queue:
                current_doc_id, depth = queue.popleft()
                if depth >= max_depth:
                    continue

                wires = _get_wires_for_document(doc, current_doc_id, direction)

                if predicate:
                    wires = [w for w in wires if w.get("predicate") == predicate]

                for wire in wires:
                    source_id = wire.get("sourceDocumentId")
                    target_id = wire.get("targetDocumentId")
                    pred = wire.get("predicate", "")

                    # Determine which end is "the other document"
                    if source_id == current_doc_id:
                        other_id = target_id
                        other_title = wire.get("targetTitle")
                    else:
                        other_id = source_id
                        other_title = wire.get("sourceTitle")

                    if not other_id:
                        continue

                    edges.append({
                        "wireId": wire["id"],
                        "source": source_id,
                        "target": target_id,
                        "predicate": pred,
                        "predicateLabel": _get_predicate_label(pred),
                        "bidirectional": bool(wire.get("bidirectional")),
                        "depth": depth + 1,
                    })

                    if other_id not in visited:
                        visited.add(other_id)
                        nodes[other_id] = {
                            "id": other_id,
                            "depth": depth + 1,
                            "title": other_title,
                        }
                        queue.append((other_id, depth + 1))

            # Try to fill in the starting document's title
            all_wires = _get_all_wires(doc)
            for w in all_wires:
                if w.get("sourceDocumentId") == document_id and w.get("sourceTitle"):
                    nodes[document_id]["title"] = w["sourceTitle"]
                    break
                if w.get("targetDocumentId") == document_id and w.get("targetTitle"):
                    nodes[document_id]["title"] = w["targetTitle"]
                    break

            return json.dumps({
                "startDocument": document_id,
                "maxDepth": max_depth,
                "nodes": list(nodes.values()),
                "edges": edges,
                "nodeCount": len(nodes),
                "edgeCount": len(edges),
            }, indent=2)

        except Exception as e:
            raise RuntimeError(f"Failed to traverse wires: {e}")
