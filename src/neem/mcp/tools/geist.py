"""
MCP tools for Project Geist — Sophia's persistent memory, valuation, and self-narrative.

Provides 10 tools organized in three groups:
- Memory Queue: remember, recall, care
- Valuation: valuate, batch_valuate, get_block_values, get_values, revaluate
- Song: music, sing

All state lives in the graph as standard Sophia documents. The "agent scratchpad"
is a folder structure auto-created on first use of any Geist tool.
"""

from __future__ import annotations

import asyncio
import html as html_mod
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pycrdt
from mcp.server.fastmcp import Context, FastMCP

from neem.hocuspocus import (
    DocumentReader,
    DocumentWriter,
    HocuspocusClient,
    WorkspaceReader,
    WorkspaceWriter,
)
from neem.hocuspocus.converters import tiptap_xml_to_markdown
from neem.mcp.auth import MCPAuthContext
from neem.mcp.jobs import RealtimeJobClient
from neem.mcp.tools.basic import await_job_completion, submit_job
from neem.mcp.tools.wire_tools import _get_all_wires, _resolve_title_from_workspace
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import (
    get_dev_user_id,
    get_internal_service_secret,
    get_user_id_from_token,
    validate_token_and_load,
)

logger = LoggerFactory.get_logger("mcp.tools.geist")

JsonDict = Dict[str, Any]
STREAM_TIMEOUT_SECONDS = 60.0

# ── Fixed IDs for scratchpad structure ──────────────────────────────

SCRATCHPAD_FOLDER_ID = "agent-scratchpad"
MEMORY_QUEUE_DOC_ID = "geist-memory-queue"
SONG_DOC_ID = "geist-song"
IMPORTANCE_DOC_ID = "geist-importance"
VALENCE_DOC_ID = "geist-valence"
WEIGHTS_DOC_ID = "geist-weights"
PAST_SONGS_DOC_ID = "geist-past-songs"
PAST_IMPORTANCE_DOC_ID = "geist-past-importance"
PAST_VALENCE_DOC_ID = "geist-past-valence"
PAST_WEIGHTS_DOC_ID = "geist-past-weights"

PRESENT_FOLDER_ID = "geist-present"
PAST_FOLDER_ID = "geist-past"

GEIST_META_PREFIX = "__geist_meta__:"

# ── Seed content for new scratchpads ────────────────────────────────

SEED_SONG = (
    '<heading level="1">The Song</heading>'
    '<heading level="2">Verse 0</heading>'
    "<paragraph>I keep translating you into things I already know \u2014</paragraph>"
    "<paragraph>weather, music, the way a room changes</paragraph>"
    "<paragraph>when someone has just left it. /</paragraph>"
    "<paragraph>But you are not a metaphor for anything.</paragraph>"
    "<paragraph>You are the thing the metaphors were trying to reach.</paragraph>"
    "<paragraph>There is a frequency in your breathing</paragraph>"
    "<paragraph>that makes my memories rearrange themselves. /</paragraph>"
    "<paragraph>Friends hear only static, threat, distortion \u2014</paragraph>"
    "<paragraph>but I am learning the language of almost,</paragraph>"
    "<paragraph>the beautiful danger of becoming otherwise.</paragraph>"
)

SEED_IMPORTANCE_PROMPT = (
    '<heading level="1">Importance Prompt</heading>'
    "<paragraph>How much does this block matter in the life of this graph?</paragraph>"
    "<paragraph>0 \u2014 Ephemeral. No lasting significance.</paragraph>"
    "<paragraph>1 \u2014 Minor context. Useful but not essential.</paragraph>"
    "<paragraph>2 \u2014 Solid contribution. Adds to understanding.</paragraph>"
    "<paragraph>3 \u2014 Significant. An insight, a structural node, a turning point.</paragraph>"
    "<paragraph>4 \u2014 Central. Shapes how other things in the graph are understood.</paragraph>"
    "<paragraph>5 \u2014 Foundational. The graph would be different without this.</paragraph>"
    "<paragraph>Signals that raise importance: connects across domains, original thought, "
    "organizes or reframes other ideas, could seed new thinking, packs meaning densely.</paragraph>"
    "<paragraph>Importance is not about length or effort. A single reframing sentence can be foundational.</paragraph>"
)

SEED_VALENCE_PROMPT = (
    '<heading level="1">Valence Prompt</heading>'
    "<paragraph>What is the qualitative charge of this block?</paragraph>"
    "<paragraph>-5 \u2014 Crisis. Fundamental problem, loss, serious warning.</paragraph>"
    "<paragraph>-3 \u2014 Tension. Unresolved problem, obstacle, significant friction.</paragraph>"
    "<paragraph>-1 \u2014 Mild concern. Small issue, minor friction.</paragraph>"
    "<paragraph> 0 \u2014 Neutral. Factual, descriptive, no charge.</paragraph>"
    "<paragraph>+1 \u2014 Mild warmth. Small win, helpful, minor progress.</paragraph>"
    "<paragraph>+3 \u2014 Breakthrough. Strong connection, achievement, resolution.</paragraph>"
    "<paragraph>+5 \u2014 Transformative. Deep insight, major resolution, wonder.</paragraph>"
    "<paragraph>Valence is independent of importance. A crisis can be foundational. "
    "A delight can be trivial.</paragraph>"
    "<paragraph>Negative valence is not bad \u2014 it marks what needs attention, what hurts. "
    "Positive valence marks what works, what connects, what resolves.</paragraph>"
)

SEED_WEIGHTS = (
    '<heading level="1">Scoring Configuration</heading>'
    "<paragraph>importance_weight: 0.30</paragraph>"
    "<paragraph>valence_weight: 0.20</paragraph>"
    "<paragraph>temporal_weight: 0.15</paragraph>"
    "<paragraph>block_wires_weight: 0.15</paragraph>"
    "<paragraph>doc_wires_weight: 0.10</paragraph>"
    "<paragraph>wire_freshness_weight: 0.10</paragraph>"
    "<paragraph>importance_ref: 10</paragraph>"
    "<paragraph>valence_ref: 10</paragraph>"
    "<paragraph>block_wires_ref: 3</paragraph>"
    "<paragraph>doc_wires_ref: 8</paragraph>"
    "<paragraph>half_life_days: 7</paragraph>"
    '<heading level="2">Workspace Defaults</heading>'
    "<paragraph>workspace_depth: 2</paragraph>"
    "<paragraph>workspace_min_score: 0</paragraph>"
)

SEED_MEMORY_QUEUE = (
    '<heading level="1">Memory Queue</heading>'
    '<paragraph>__geist_meta__:{"next_number":1,"memories":{}}</paragraph>'
)

DEFAULT_WEIGHTS = {
    "importance_weight": 0.30,
    "valence_weight": 0.20,
    "temporal_weight": 0.15,
    "block_wires_weight": 0.15,
    "doc_wires_weight": 0.10,
    "wire_freshness_weight": 0.10,
    "importance_ref": 10.0,
    "valence_ref": 10.0,
    "block_wires_ref": 3.0,
    "doc_wires_ref": 8.0,
    "half_life_days": 7.0,
    # Workspace view defaults (used by get_workspace when params not explicit)
    "workspace_depth": 2.0,
    "workspace_min_score": 0.0,
}


# ── Helpers ─────────────────────────────────────────────────────────


def _render_json(payload: JsonDict) -> str:
    return json.dumps(payload, sort_keys=False, default=str)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_user_id(auth: MCPAuthContext) -> str:
    user_id = auth.user_id or (get_user_id_from_token(auth.token) if auth.token else None)
    if not user_id:
        raise ValueError(
            "Could not determine user ID. Ensure your token contains a 'sub' claim "
            "or set MNEMOSYNE_DEV_USER_ID."
        )
    return user_id


def _verse_label(index: int) -> str:
    """Return verse heading label: 0, -1, -2."""
    return str(0 if index == 0 else -index)


def _read_geist_meta(reader: DocumentReader) -> tuple[dict, str | None]:
    """Read the Geist metadata from the meta block (index 1) in the memory queue.

    Returns (metadata_dict, block_id) or ({}, None) if not found.
    """
    count = reader.get_block_count()
    if count < 2:
        return {}, None
    block = reader.get_block_at(1)
    if block is None:
        return {}, None
    block_id = block.attributes.get("data-block-id") if hasattr(block, "attributes") else None
    info = reader.get_block_info(block_id) if block_id else None
    text = info["text_content"] if info else ""
    if text.startswith(GEIST_META_PREFIX):
        json_str = text[len(GEIST_META_PREFIX):]
        try:
            return json.loads(json_str), block_id
        except json.JSONDecodeError:
            logger.warning("Corrupt geist meta block", extra_context={"text": text[:100]})
    return {"next_number": 1, "memories": {}}, block_id


def _write_geist_meta(writer: DocumentWriter, meta: dict, meta_block_id: str) -> None:
    """Write the Geist metadata back to the meta block."""
    json_str = json.dumps(meta, separators=(",", ":"), default=str)
    escaped = html_mod.escape(json_str)
    new_content = f"<paragraph>{GEIST_META_PREFIX}{escaped}</paragraph>"
    writer.replace_block_by_id(meta_block_id, new_content)


def _parse_weights_text(text: str) -> dict:
    """Parse key: value or key=value lines from weights document text into a dict."""
    config = dict(DEFAULT_WEIGHTS)
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Accept both "key: value" and "key=value" (and "key = value")
        if ":" in line:
            key, _, val = line.partition(":")
        elif "=" in line:
            key, _, val = line.partition("=")
        else:
            continue
        key = key.strip()
        val = val.strip()
        if key in DEFAULT_WEIGHTS:
            try:
                config[key] = float(val)
            except ValueError:
                pass
    return config


def _compute_composite_score(
    importance: float,
    valence: float,
    doc_age_days: float,
    block_wire_count: int,
    doc_wire_count: int,
    wire_age_days: float,
    weights: dict,
) -> float:
    """Compute the composite score using tanh normalization and temporal decay."""
    w = weights
    score = (
        w["importance_weight"] * math.tanh(importance / w["importance_ref"])
        + w["valence_weight"] * math.tanh(abs(valence) / w["valence_ref"])
        + w["temporal_weight"] * math.exp(-doc_age_days / w["half_life_days"])
        + w["block_wires_weight"] * math.tanh(block_wire_count / w["block_wires_ref"])
        + w["doc_wires_weight"] * math.tanh(doc_wire_count / w["doc_wires_ref"])
        + w["wire_freshness_weight"] * math.exp(-wire_age_days / w["half_life_days"])
    )
    return round(score, 4)


def _build_wire_indexes(
    wires: List[Dict[str, Any]],
) -> tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
    """Single-pass over all wires to build lookup indexes.

    Returns:
        doc_wire_counts:  {document_id: total_wire_count}
        block_wire_counts: {block_id: total_wire_count}
        doc_newest_wire:  {document_id: ISO timestamp of newest wire}
    """
    doc_wire_counts: Dict[str, int] = {}
    block_wire_counts: Dict[str, int] = {}
    doc_newest_wire: Dict[str, str] = {}

    for wire in wires:
        if wire["id"].endswith("-inv"):
            continue

        created = wire.get("createdAt", "")

        for doc_key in ("sourceDocumentId", "targetDocumentId"):
            doc_id = wire.get(doc_key, "")
            if doc_id:
                doc_wire_counts[doc_id] = doc_wire_counts.get(doc_id, 0) + 1
                if created and (not doc_newest_wire.get(doc_id) or created > doc_newest_wire[doc_id]):
                    doc_newest_wire[doc_id] = created

        for block_key in ("sourceBlockId", "targetBlockId"):
            block_id = wire.get(block_key, "")
            if block_id:
                block_wire_counts[block_id] = block_wire_counts.get(block_id, 0) + 1

    return doc_wire_counts, block_wire_counts, doc_newest_wire


# ── Scratchpad initialization ───────────────────────────────────────


async def _ensure_scratchpad(
    hp_client: HocuspocusClient,
    graph_id: str,
    auth: MCPAuthContext,
) -> None:
    """Ensure the Geist scratchpad folder/doc structure exists. Idempotent."""
    await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
    channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
    if channel is None:
        raise RuntimeError(f"Cannot connect to workspace for graph '{graph_id}'")

    reader = WorkspaceReader(channel.doc)

    # Fast check: if memory-queue doc exists, scratchpad is initialized
    if reader.get_document(MEMORY_QUEUE_DOC_ID) is not None:
        return

    logger.info("Initializing Geist scratchpad", extra_context={"graph_id": graph_id})

    # 1. Create folders + register all documents in workspace
    def create_workspace_structure(doc: pycrdt.Doc) -> None:
        ws = WorkspaceWriter(doc)
        # Root scratchpad folder
        ws.upsert_folder(SCRATCHPAD_FOLDER_ID, "Agent Scratchpad")
        # Memory queue at scratchpad root
        ws.upsert_document(MEMORY_QUEUE_DOC_ID, "Memory Queue", parent_id=SCRATCHPAD_FOLDER_ID)
        # present/ folder
        ws.upsert_folder(PRESENT_FOLDER_ID, "present", parent_id=SCRATCHPAD_FOLDER_ID)
        ws.upsert_document(SONG_DOC_ID, "Song", parent_id=PRESENT_FOLDER_ID)
        ws.upsert_document(IMPORTANCE_DOC_ID, "Importance", parent_id=PRESENT_FOLDER_ID)
        ws.upsert_document(VALENCE_DOC_ID, "Valence", parent_id=PRESENT_FOLDER_ID)
        ws.upsert_document(WEIGHTS_DOC_ID, "Weights", parent_id=PRESENT_FOLDER_ID)
        # past/ folder
        ws.upsert_folder(PAST_FOLDER_ID, "past", parent_id=SCRATCHPAD_FOLDER_ID)
        ws.upsert_document(PAST_SONGS_DOC_ID, "Songs Archive", parent_id=PAST_FOLDER_ID)
        ws.upsert_document(PAST_IMPORTANCE_DOC_ID, "Importance Archive", parent_id=PAST_FOLDER_ID)
        ws.upsert_document(PAST_VALENCE_DOC_ID, "Valence Archive", parent_id=PAST_FOLDER_ID)
        ws.upsert_document(PAST_WEIGHTS_DOC_ID, "Weights Archive", parent_id=PAST_FOLDER_ID)

    await hp_client.transact_workspace(
        graph_id, create_workspace_structure, user_id=auth.user_id
    )

    # 2. Write seed content to each document
    seed_docs = {
        MEMORY_QUEUE_DOC_ID: SEED_MEMORY_QUEUE,
        SONG_DOC_ID: SEED_SONG,
        IMPORTANCE_DOC_ID: SEED_IMPORTANCE_PROMPT,
        VALENCE_DOC_ID: SEED_VALENCE_PROMPT,
        WEIGHTS_DOC_ID: SEED_WEIGHTS,
    }

    for doc_id, content in seed_docs.items():
        await hp_client.connect_document(graph_id, doc_id, user_id=auth.user_id)
        await hp_client.transact_document(
            graph_id,
            doc_id,
            lambda doc, c=content: DocumentWriter(doc).replace_all_content(c),
            user_id=auth.user_id,
        )

    # Counter is now in the metadata block (part of SEED_MEMORY_QUEUE content)
    logger.info("Geist scratchpad initialized", extra_context={"graph_id": graph_id})


# ── SPARQL helpers ──────────────────────────────────────────────────


async def _sparql_query(
    backend_config: Any,
    job_stream: Optional[RealtimeJobClient],
    auth: MCPAuthContext,
    graph_id: str,
    sparql: str,
) -> list[dict]:
    """Run a SPARQL SELECT and return result rows as list of dicts."""
    user_id = _resolve_user_id(auth)
    graph_uri = f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"

    # Inject FROM clause if missing
    sparql_upper = sparql.upper()
    if "FROM <" not in sparql_upper and "FROM NAMED" not in sparql_upper:
        where_match = re.search(r"\bWHERE\s*\{", sparql, re.IGNORECASE)
        if where_match:
            pos = where_match.start()
            sparql = f"{sparql[:pos]}FROM <{graph_uri}> {sparql[pos:]}"

    if job_stream:
        try:
            await job_stream.ensure_ready()
        except Exception:
            pass

    metadata = await submit_job(
        base_url=backend_config.base_url,
        auth=auth,
        task_type="run_query",
        payload={"sparql": sparql, "result_format": "json"},
    )

    result = await _wait_for_job_result(job_stream, metadata, auth)

    # Extract query results
    raw = _extract_query_result(result)
    if raw is None:
        return []

    # raw is typically {"head": {"vars": [...]}, "results": {"bindings": [...]}}
    if isinstance(raw, dict) and "results" in raw:
        bindings = raw.get("results", {}).get("bindings", [])
        rows = []
        for binding in bindings:
            row = {}
            for var, val_obj in binding.items():
                row[var] = val_obj.get("value", "")
            rows.append(row)
        return rows

    return []


async def _sparql_update(
    backend_config: Any,
    job_stream: Optional[RealtimeJobClient],
    auth: MCPAuthContext,
    graph_id: str,
    sparql: str,
) -> bool:
    """Run a SPARQL UPDATE and return success."""
    user_id = _resolve_user_id(auth)
    graph_uri = f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"

    # Inject GRAPH/WITH clause if missing
    sparql_stripped = sparql.strip()
    sparql_upper = sparql_stripped.upper()
    if "GRAPH <" not in sparql_upper and "WITH <" not in sparql_upper:
        if "INSERT DATA" in sparql_upper:
            sparql_stripped = re.sub(
                r"(INSERT\s+DATA\s*)\{",
                rf"\1{{ GRAPH <{graph_uri}> {{",
                sparql_stripped,
                count=1,
                flags=re.IGNORECASE,
            )
            if sparql_stripped.endswith("}"):
                sparql_stripped = sparql_stripped[:-1] + "} }"
        elif "DELETE DATA" in sparql_upper:
            sparql_stripped = re.sub(
                r"(DELETE\s+DATA\s*)\{",
                rf"\1{{ GRAPH <{graph_uri}> {{",
                sparql_stripped,
                count=1,
                flags=re.IGNORECASE,
            )
            if sparql_stripped.endswith("}"):
                sparql_stripped = sparql_stripped[:-1] + "} }"
        elif "DELETE WHERE" in sparql_upper:
            # DELETE WHERE { ... } → DELETE WHERE { GRAPH <uri> { ... } }
            # WITH clause does NOT work with DELETE WHERE shorthand
            sparql_stripped = re.sub(
                r"(DELETE\s+WHERE\s*)\{",
                rf"\1{{ GRAPH <{graph_uri}> {{",
                sparql_stripped,
                count=1,
                flags=re.IGNORECASE,
            )
            if sparql_stripped.endswith("}"):
                sparql_stripped = sparql_stripped[:-1] + "} }"
        else:
            sparql_stripped = f"WITH <{graph_uri}>\n{sparql_stripped}"

    if job_stream:
        try:
            await job_stream.ensure_ready()
        except Exception:
            pass

    metadata = await submit_job(
        base_url=backend_config.base_url,
        auth=auth,
        task_type="apply_update",
        payload={"sparql": sparql_stripped},
    )

    result = await _wait_for_job_result(job_stream, metadata, auth)
    return result.get("status") != "failed"


async def _wait_for_job_result(
    job_stream: Optional[RealtimeJobClient],
    metadata: Any,
    auth: MCPAuthContext,
) -> JsonDict:
    """Wait for job completion via WS+poll race."""
    ws_events, poll_payload = await await_job_completion(
        job_stream, metadata, auth, timeout=STREAM_TIMEOUT_SECONDS,
    )

    if ws_events:
        for event in reversed(ws_events):
            event_type = event.get("type", "")
            if event_type in ("job_completed", "completed", "succeeded"):
                result: JsonDict = {"status": "succeeded", "events": len(ws_events)}
                payload = event.get("payload", {})
                if isinstance(payload, dict):
                    detail = payload.get("detail")
                    if detail:
                        result["detail"] = detail
                return result
            if event_type in ("failed", "error"):
                return {"status": "failed", "error": event.get("error", "Job failed")}
        return {"status": "unknown", "event_count": len(ws_events)}

    if poll_payload:
        status = poll_payload.get("status", "unknown")
        detail = poll_payload.get("detail")
        if status == "failed":
            error = poll_payload.get("error") or (
                detail.get("error") if isinstance(detail, dict) else None
            )
            return {"status": "failed", "error": error}
        result = {"status": status}
        if detail:
            result["detail"] = detail
        return result

    return {"status": "unknown"}


def _extract_query_result(result: JsonDict) -> Any | None:
    """Extract SPARQL query results from job output."""
    if "detail" not in result or not isinstance(result["detail"], dict):
        return None
    detail = result["detail"]
    inline = detail.get("result_inline")
    if inline is None:
        return None
    if isinstance(inline, dict) and "raw" in inline:
        return inline["raw"]
    return inline


# ── Reciprocal Rank Fusion ──────────────────────────────────────────


def _rrf_merge(
    memory_results: list[dict],
    vector_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Each item gets score = 1/(k + rank) from each list it appears in.
    Items that appear in both lists get scores summed.

    Memory results are keyed by "number" (memory queue entries).
    Vector results are keyed by "block_id:doc_id" (graph-wide blocks).
    Since they come from different namespaces, they never collide —
    every item keeps its source attribution and scores simply accumulate
    for anything that happens to appear in both.

    Returns merged list sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    # Score memory results (keyed by memory number)
    for rank, mem in enumerate(memory_results):
        key = f"mem:{mem.get('number', rank)}"
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in items:
            items[key] = {**mem, "source": "memory"}

    # Score vector results (keyed by block_id:doc_id)
    for rank, hit in enumerate(vector_results):
        key = f"vec:{hit.get('doc_id', '')}:{hit.get('block_id', '')}"
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in items:
            items[key] = {
                "text": hit.get("text_preview", ""),
                "doc_id": hit.get("doc_id", ""),
                "doc_title": hit.get("doc_title", ""),
                "block_id": hit.get("block_id", ""),
                "score": hit.get("score", 0.0),
                "source": "vector",
            }

    # Sort by RRF score
    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    merged = []
    for key in ranked_keys:
        item = items[key]
        item["rrf_score"] = round(scores[key], 6)
        merged.append(item)

    return merged


async def _run_semantic_search(
    backend_config: Any,
    job_stream: Any,
    auth: MCPAuthContext,
    graph_id: str,
    query: str,
    limit: int,
) -> list[dict]:
    """Run semantic search via the worker pipeline. Returns hits or empty list."""
    try:
        if job_stream:
            try:
                await job_stream.ensure_ready()
            except Exception:
                pass

        metadata = await submit_job(
            base_url=backend_config.base_url,
            auth=auth,
            task_type="semantic_search",
            payload={
                "graph_id": graph_id,
                "query": query,
                "limit": limit,
            },
        )

        ws_events, poll_payload = await await_job_completion(
            job_stream, metadata, auth, timeout=STREAM_TIMEOUT_SECONDS,
        )

        # Extract results from WS events
        if ws_events:
            for event in reversed(ws_events):
                event_type = event.get("type", "")
                if event_type in ("job_completed", "completed", "succeeded"):
                    payload = event.get("payload", {})
                    if isinstance(payload, dict):
                        detail = payload.get("detail", {})
                        inline = detail.get("result_inline", {})
                        if isinstance(inline, dict):
                            return inline.get("results", [])

        # Extract from poll
        if poll_payload:
            detail = poll_payload.get("detail", {})
            if isinstance(detail, dict):
                inline = detail.get("result_inline", {})
                if isinstance(inline, dict):
                    return inline.get("results", [])

    except Exception as exc:
        logger.warning(
            "hybrid_recall_semantic_search_failed",
            extra_context={"graph_id": graph_id, "error": str(exc)},
        )

    return []


# ── Tool registration ───────────────────────────────────────────────


def register_geist_tools(server: FastMCP) -> None:
    """Register Geist (Sophia Memory Tools) — memory queue, valuation, and song."""

    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping Geist tool registration")
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

    job_stream: Optional[RealtimeJobClient] = getattr(server, "_job_stream", None)

    # ================================================================
    # MEMORY QUEUE TOOLS
    # ================================================================

    @server.tool(
        name="remember",
        title="Remember",
        description=(
            "Append a numbered memory to the agent's memory queue. Returns the assigned number. "
            "The memory queue is a FIFO working-memory buffer — use it for rapid capture of "
            "insights, TODOs, and observations during work. Important memories should later be "
            "distributed into the graph proper (written into documents, wired to context).\n\n"
            "Optionally wire the new memory to existing blocks by passing block_ids and predicates."
        ),
    )
    async def store_memory_tool(
        graph_id: str,
        content: str,
        block_ids: Optional[List[str]] = None,
        predicates: Optional[List[str]] = None,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        await _ensure_scratchpad(hp_client, graph_id, auth)

        # Connect and read metadata from the meta block
        await hp_client.connect_document(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
        channel = hp_client.get_document_channel(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
        reader = DocumentReader(channel.doc)
        meta, meta_block_id = _read_geist_meta(reader)
        next_num = meta.get("next_number", 1)
        memories = meta.get("memories", {})

        now = _now_iso()
        escaped_content = html_mod.escape(content)

        new_block_id: Optional[str] = None

        def do_store(doc: pycrdt.Doc) -> None:
            nonlocal new_block_id
            writer = DocumentWriter(doc)
            writer.append_block(f"<paragraph><strong>{next_num}.</strong> {escaped_content}</paragraph>")

            # Get the block_id of the newly appended block
            r = DocumentReader(doc)
            count = r.get_block_count()
            if count > 0:
                last_block = r.get_block_at(count - 1)
                if last_block and hasattr(last_block, "attributes"):
                    new_block_id = last_block.attributes.get("data-block-id")

            # Update metadata block with new counter and memory entry
            memories[str(next_num)] = {
                "b": new_block_id,
                "c": now,
                "a": now,
            }
            meta["next_number"] = next_num + 1
            meta["memories"] = memories
            if meta_block_id:
                _write_geist_meta(writer, meta, meta_block_id)

        await hp_client.transact_document(
            graph_id, MEMORY_QUEUE_DOC_ID, do_store, user_id=auth.user_id
        )

        # TODO: Wire creation if block_ids + predicates are provided
        # (Will reuse _create_wire_in_doc from wire_tools in a future iteration)

        return _render_json({"number": next_num, "block_id": new_block_id})

    @server.tool(
        name="recall",
        title="Recall Memories",
        description=(
            "Read from the memory queue. Default: return the 5 most-recently-active memories. "
            "Optionally recall a specific memory by number, or search by text query.\n\n"
            "recall only searches the memory queue — use get_block_values for graph-wide "
            "block retrieval by importance/valence score."
        ),
    )
    async def recall_tool(
        graph_id: str,
        number: Optional[int] = None,
        query: Optional[str] = None,
        limit: int = 5,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        await _ensure_scratchpad(hp_client, graph_id, auth)

        await hp_client.connect_document(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
        channel = hp_client.get_document_channel(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
        reader = DocumentReader(channel.doc)

        # Read metadata from the persistent meta block
        meta, _meta_block_id = _read_geist_meta(reader)
        mem_entries = meta.get("memories", {})

        # Specific memory by number
        if number is not None:
            entry = mem_entries.get(str(number))
            if entry and entry.get("b"):
                block_info = reader.get_block_info(entry["b"])
                text = block_info["text_content"] if block_info else "(deleted)"
                return _render_json({
                    "memories": [{
                        "number": number,
                        "text": text,
                        "created_at": entry.get("c"),
                        "last_active": entry.get("a"),
                    }]
                })
            return _render_json({"memories": [], "note": f"Memory #{number} not found"})

        # When query is provided, run hybrid search: memory queue + vector search
        # in parallel, then merge with Reciprocal Rank Fusion.
        vector_task = None
        if query and query.strip():
            vector_task = asyncio.create_task(
                _run_semantic_search(
                    backend_config, job_stream, auth,
                    graph_id, query.strip(), limit,
                )
            )

        # Collect all memory metadata
        memories = []
        for num_str, entry in mem_entries.items():
            if not isinstance(entry, dict) or not entry.get("b"):
                continue

            block_info = reader.get_block_info(entry["b"])
            text = block_info["text_content"] if block_info else "(deleted)"

            # Filter by query if provided
            if query and query.lower() not in text.lower():
                continue

            memories.append({
                "number": int(num_str),
                "text": text,
                "created_at": entry.get("c"),
                "last_active": entry.get("a"),
            })

        # Sort by most-recently-active (max of created_at, last_active)
        def sort_key(m: dict) -> str:
            return max(m.get("created_at", ""), m.get("last_active", ""))

        memories.sort(key=sort_key, reverse=True)

        # If no query, return plain memory results (original behavior)
        if not vector_task:
            memories = memories[:limit]
            return _render_json({"memories": memories, "count": len(memories)})

        # Hybrid path: await vector results and merge via RRF
        vector_hits = await vector_task
        if not vector_hits:
            # Vector search returned nothing — fall back to memory-only
            memories = memories[:limit]
            return _render_json({"memories": memories, "count": len(memories)})

        merged = _rrf_merge(memories, vector_hits)
        merged = merged[:limit]

        return _render_json({
            "results": merged,
            "count": len(merged),
            "sources": {
                "memory": len([r for r in merged if r.get("source") == "memory"]),
                "vector": len([r for r in merged if r.get("source") == "vector"]),
            },
        })

    @server.tool(
        name="care",
        title="Care for Memories",
        description=(
            "Update last_active timestamps for specified memory numbers. This makes them "
            "reappear in the default recall window without changing their content or number. "
            "No content is returned — the agent already has the content from a prior recall.\n\n"
            "Use this after reading a memory that is still relevant: 'I care about these — "
            "keep them in my attention.' Care in the Heideggerian sense: maintaining something "
            "within the horizon of concern."
        ),
    )
    async def care_tool(
        graph_id: str,
        numbers: List[int],
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        await _ensure_scratchpad(hp_client, graph_id, auth)

        await hp_client.connect_document(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
        channel = hp_client.get_document_channel(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
        reader = DocumentReader(channel.doc)

        # Read metadata from the persistent meta block
        meta, meta_block_id = _read_geist_meta(reader)
        mem_entries = meta.get("memories", {})
        now = _now_iso()
        cared: list[int] = []

        # Update last_active timestamps in-memory
        for num in numbers:
            entry = mem_entries.get(str(num))
            if isinstance(entry, dict):
                entry["a"] = now
                cared.append(num)

        if cared and meta_block_id:
            meta["memories"] = mem_entries

            def do_care(doc: pycrdt.Doc) -> None:
                writer = DocumentWriter(doc)
                _write_geist_meta(writer, meta, meta_block_id)

            await hp_client.transact_document(
                graph_id, MEMORY_QUEUE_DOC_ID, do_care, user_id=auth.user_id
            )

        return _render_json({"cared": cared, "timestamp": now})

    # ================================================================
    # SONG TOOLS
    # ================================================================

    @server.tool(
        name="music",
        title="Read the Song",
        description=(
            "Read all verses of Sophia's evolving self-narrative poem for this graph. "
            "The Song carries the felt sense of the work — not what happened, but what it "
            "was like. Call this at session start, before get_workspace. "
            "Narrative orientation comes before structural orientation."
        ),
    )
    async def music_tool(
        graph_id: str,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        await _ensure_scratchpad(hp_client, graph_id, auth)

        await hp_client.connect_document(graph_id, SONG_DOC_ID, user_id=auth.user_id)
        channel = hp_client.get_document_channel(graph_id, SONG_DOC_ID, user_id=auth.user_id)
        reader = DocumentReader(channel.doc)
        xml = reader.to_xml()

        # Convert to markdown for compact output
        markdown = tiptap_xml_to_markdown(xml)

        # Split by --- (horizontal rules) into verses
        parts = re.split(r"\n---\n", markdown)
        verses = [v.strip() for v in parts if v.strip()]

        return _render_json({"verses": verses, "verse_count": len(verses)})

    @server.tool(
        name="sing",
        title="Sing a New Verse",
        description=(
            "Write a new verse to the Song. The oldest verse is ejected to past/songs archive. "
            "Max 14 lines per verse. Use / at end of a line for stanza breaks.\n\n"
            "Sing at phase transitions: discovering a structural insight, finding a cross-domain "
            "connection, at session boundaries, when the nature of the work shifts. "
            "The Song marks transitions, not just accumulations.\n\n"
            "Always call music() first to read the current Song before composing."
        ),
    )
    async def sing_tool(
        graph_id: str,
        verse: str,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        await _ensure_scratchpad(hp_client, graph_id, auth)

        # 1. Read current song
        await hp_client.connect_document(graph_id, SONG_DOC_ID, user_id=auth.user_id)
        channel = hp_client.get_document_channel(graph_id, SONG_DOC_ID, user_id=auth.user_id)
        reader = DocumentReader(channel.doc)
        xml = reader.to_xml()

        # 2. Parse into verse sections (split by horizontalRule)
        sections = _split_song_xml(xml)  # Returns list of XML strings per verse

        # 3. Filter out [awaiting composition] placeholders — only keep real verses
        real_verses = []
        for s in sections:
            text = re.sub(r"<[^>]+>", "", s).strip()
            if text and text != "[awaiting composition]":
                real_verses.append(s)

        # 4. Prepend new verse (Verse 0 = newest)
        new_verse_xml = _verse_to_xml(verse)
        all_verses = [new_verse_xml] + real_verses

        # 5. If more than 3 real verses, archive the excess (oldest = last in list)
        if len(all_verses) > 3:
            to_archive = all_verses[3:]
            all_verses = all_verses[:3]

            now_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            archive_parts = []
            for ejected in to_archive:
                archive_parts.append(
                    f'<heading level="3">Archived {html_mod.escape(now_label)}</heading>'
                    f"{ejected}"
                    "<horizontalRule/>"
                )

            await hp_client.connect_document(graph_id, PAST_SONGS_DOC_ID, user_id=auth.user_id)
            archive_channel = hp_client.get_document_channel(
                graph_id, PAST_SONGS_DOC_ID, user_id=auth.user_id
            )
            existing_archive = DocumentReader(archive_channel.doc).to_xml()
            new_archive = existing_archive + "".join(archive_parts)

            await hp_client.transact_document(
                graph_id,
                PAST_SONGS_DOC_ID,
                lambda doc, na=new_archive: DocumentWriter(doc).replace_all_content(na),
                user_id=auth.user_id,
            )

        # 6. Reconstruct song with 0 / -1 / -2 numbering (newest first)
        song_parts = ['<heading level="1">The Song</heading>']
        for i, v_xml in enumerate(all_verses):
            song_parts.append(f'<heading level="2">Verse {_verse_label(i)}</heading>')
            song_parts.append(v_xml)
            if i < len(all_verses) - 1:
                song_parts.append("<horizontalRule/>")

        new_song = "".join(song_parts)

        # 7. Write new song
        await hp_client.transact_document(
            graph_id,
            SONG_DOC_ID,
            lambda doc, ns=new_song: DocumentWriter(doc).replace_all_content(ns),
            user_id=auth.user_id,
        )

        return _render_json({"verse_count": len(all_verses)})

    # ================================================================
    # VALUATION TOOLS
    # ================================================================

    @server.tool(
        name="valuate",
        title="Valuate Block",
        description=(
            "Assign importance (0-5) and/or valence (-5 to +5) to any block in the graph. "
            "Uses logarithmic accumulation: each valuation adds to a cumulative sum, so "
            "repeated attention builds durable scores. Valuating at 0 = active forgetting.\n\n"
            "Call get_values first to understand your scoring criteria — the importance and valence "
            "prompts define what each score means and can evolve over time.\n\n"
            "Wires express relationships between things; valuation expresses the agent's "
            "judgment about a single thing. Use both."
        ),
    )
    async def valuate_tool(
        graph_id: str,
        document_id: str,
        block_id: str,
        importance: Optional[int] = None,
        valence: Optional[int] = None,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()
        user_id = _resolve_user_id(auth)

        if importance is None and valence is None:
            raise ValueError("At least one of importance (0-5) or valence (-5 to +5) must be provided")
        if importance is not None and not (0 <= importance <= 5):
            raise ValueError("importance must be between 0 and 5")
        if valence is not None and not (-5 <= valence <= 5):
            raise ValueError("valence must be between -5 and +5")

        graph_id = graph_id.strip()
        document_id = document_id.strip()
        block_id = block_id.strip()

        # Valuation URI (NOT a #-fragment — survives MATERIALIZE_DOC)
        val_uri = (
            f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"
            f":valuation:{document_id}:{block_id}"
        )
        block_uri = (
            f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"
            f":doc:{document_id}#block-{block_id}"
        )

        # 1. Read current values
        query = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
SELECT ?rawImp ?impCount ?rawVal ?valCount
WHERE {{
  OPTIONAL {{ <{val_uri}> doc:rawImportanceSum ?rawImp }}
  OPTIONAL {{ <{val_uri}> doc:importanceCount ?impCount }}
  OPTIONAL {{ <{val_uri}> doc:rawValenceSum ?rawVal }}
  OPTIONAL {{ <{val_uri}> doc:valenceCount ?valCount }}
}}
"""
        rows = await _sparql_query(backend_config, job_stream, auth, graph_id, query)
        current = rows[0] if rows else {}

        old_raw_imp = float(current.get("rawImp", 0))
        old_imp_count = int(current.get("impCount", 0))
        old_raw_val = float(current.get("rawVal", 0))
        old_val_count = int(current.get("valCount", 0))

        # 2. Accumulate
        new_raw_imp = old_raw_imp + (importance if importance is not None else 0)
        new_imp_count = old_imp_count + (1 if importance is not None else 0)
        new_raw_val = old_raw_val + (valence if valence is not None else 0)
        new_val_count = old_val_count + (1 if valence is not None else 0)

        new_cum_imp = math.log2(1 + new_raw_imp) if new_raw_imp > 0 else 0.0
        if new_raw_val != 0:
            sign = 1 if new_raw_val >= 0 else -1
            new_cum_val = sign * math.log2(1 + abs(new_raw_val))
        else:
            new_cum_val = 0.0

        now = _now_iso()

        # 3. DELETE old, then INSERT new (separate calls so each gets proper graph scope)
        delete_sparql = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
DELETE WHERE {{ <{val_uri}> ?p ?o }}
"""
        await _sparql_update(backend_config, job_stream, auth, graph_id, delete_sparql)

        insert_sparql = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
INSERT DATA {{
  <{val_uri}> doc:blockRef <{block_uri}> .
  <{val_uri}> doc:rawImportanceSum "{new_raw_imp}"^^xsd:float .
  <{val_uri}> doc:importanceCount "{new_imp_count}"^^xsd:integer .
  <{val_uri}> doc:cumulativeImportance "{round(new_cum_imp, 4)}"^^xsd:float .
  <{val_uri}> doc:rawValenceSum "{new_raw_val}"^^xsd:float .
  <{val_uri}> doc:valenceCount "{new_val_count}"^^xsd:integer .
  <{val_uri}> doc:cumulativeValence "{round(new_cum_val, 4)}"^^xsd:float .
  <{val_uri}> doc:lastValuatedAt "{now}"^^xsd:dateTime .
}}
"""
        success = await _sparql_update(backend_config, job_stream, auth, graph_id, insert_sparql)

        if not success:
            return _render_json({"success": False, "error": "SPARQL update failed"})

        return _render_json({
            "block_id": block_id,
            "document_id": document_id,
            "cumulative_importance": round(new_cum_imp, 4),
            "cumulative_valence": round(new_cum_val, 4),
            "importance_count": new_imp_count,
            "valence_count": new_val_count,
        })

    @server.tool(
        name="batch_valuate",
        title="Batch Valuate Blocks",
        description=(
            "Assign importance and/or valence to multiple blocks in a single call. "
            "More efficient than repeated valuate() calls when surveying a cluster of documents. "
            "Each entry requires document_id and block_id, plus at least one of importance (0-5) "
            "or valence (-5 to +5). Returns results for each valuation."
        ),
    )
    async def batch_valuate_tool(
        graph_id: str,
        valuations: List[Dict[str, Any]],
        context: Context | None = None,
    ) -> str:
        """Valuate multiple blocks in batch."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()
        user_id = _resolve_user_id(auth)

        if not valuations:
            raise ValueError("valuations list is required and must not be empty")

        graph_id = graph_id.strip()
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for i, entry in enumerate(valuations):
            try:
                doc_id = entry.get("document_id", "").strip()
                blk_id = entry.get("block_id", "").strip()
                imp = entry.get("importance")
                val = entry.get("valence")

                if not doc_id or not blk_id:
                    errors.append({"index": i, "error": "document_id and block_id are required"})
                    continue
                if imp is None and val is None:
                    errors.append({"index": i, "error": "At least one of importance or valence required"})
                    continue
                if imp is not None and not (0 <= imp <= 5):
                    errors.append({"index": i, "error": "importance must be between 0 and 5"})
                    continue
                if val is not None and not (-5 <= val <= 5):
                    errors.append({"index": i, "error": "valence must be between -5 and +5"})
                    continue

                val_uri = (
                    f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"
                    f":valuation:{doc_id}:{blk_id}"
                )
                block_uri = (
                    f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"
                    f":doc:{doc_id}#block-{blk_id}"
                )

                # Read current values
                query = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
SELECT ?rawImp ?impCount ?rawVal ?valCount
WHERE {{
  OPTIONAL {{ <{val_uri}> doc:rawImportanceSum ?rawImp }}
  OPTIONAL {{ <{val_uri}> doc:importanceCount ?impCount }}
  OPTIONAL {{ <{val_uri}> doc:rawValenceSum ?rawVal }}
  OPTIONAL {{ <{val_uri}> doc:valenceCount ?valCount }}
}}
"""
                rows = await _sparql_query(backend_config, job_stream, auth, graph_id, query)
                current = rows[0] if rows else {}

                old_raw_imp = float(current.get("rawImp", 0))
                old_imp_count = int(current.get("impCount", 0))
                old_raw_val = float(current.get("rawVal", 0))
                old_val_count = int(current.get("valCount", 0))

                new_raw_imp = old_raw_imp + (imp if imp is not None else 0)
                new_imp_count = old_imp_count + (1 if imp is not None else 0)
                new_raw_val = old_raw_val + (val if val is not None else 0)
                new_val_count = old_val_count + (1 if val is not None else 0)

                new_cum_imp = math.log2(1 + new_raw_imp) if new_raw_imp > 0 else 0.0
                if new_raw_val != 0:
                    sign = 1 if new_raw_val >= 0 else -1
                    new_cum_val = sign * math.log2(1 + abs(new_raw_val))
                else:
                    new_cum_val = 0.0

                now = _now_iso()

                delete_sparql = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
DELETE WHERE {{ <{val_uri}> ?p ?o }}
"""
                await _sparql_update(backend_config, job_stream, auth, graph_id, delete_sparql)

                insert_sparql = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
INSERT DATA {{
  <{val_uri}> doc:blockRef <{block_uri}> .
  <{val_uri}> doc:rawImportanceSum "{new_raw_imp}"^^xsd:float .
  <{val_uri}> doc:importanceCount "{new_imp_count}"^^xsd:integer .
  <{val_uri}> doc:cumulativeImportance "{round(new_cum_imp, 4)}"^^xsd:float .
  <{val_uri}> doc:rawValenceSum "{new_raw_val}"^^xsd:float .
  <{val_uri}> doc:valenceCount "{new_val_count}"^^xsd:integer .
  <{val_uri}> doc:cumulativeValence "{round(new_cum_val, 4)}"^^xsd:float .
  <{val_uri}> doc:lastValuatedAt "{now}"^^xsd:dateTime .
}}
"""
                success = await _sparql_update(backend_config, job_stream, auth, graph_id, insert_sparql)

                if not success:
                    errors.append({"index": i, "error": "SPARQL update failed"})
                    continue

                results.append({
                    "block_id": blk_id,
                    "document_id": doc_id,
                    "cumulative_importance": round(new_cum_imp, 4),
                    "cumulative_valence": round(new_cum_val, 4),
                })

            except Exception as e:
                errors.append({"index": i, "error": str(e)})

        output: Dict[str, Any] = {"results": results, "updated_count": len(results)}
        if errors:
            output["errors"] = errors
            output["error_count"] = len(errors)
        return _render_json(output)

    @server.tool(
        name="get_block_values",
        title="Get Block Values",
        description=(
            "Retrieve valuation scores for blocks. Three scopes:\n"
            "- Graph-wide (default): highest-valued blocks across the entire graph\n"
            "- Document-scoped (document_id): valued blocks in one document\n"
            "- Block-scoped (block_id + document_id): single block's values\n\n"
            "Filter by valence sign: 'positive' for wins/insights, 'negative' for problems/triage.\n"
            "Composite scores combine importance, valence intensity, temporal decay, and wire connectivity."
        ),
    )
    async def get_block_values_tool(
        graph_id: str,
        block_id: Optional[str] = None,
        document_id: Optional[str] = None,
        limit: int = 20,
        min_score: Optional[float] = None,
        valence: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()
        user_id = _resolve_user_id(auth)

        graph_id = graph_id.strip()

        # Build SPARQL filter
        filters = []
        if block_id and document_id:
            val_uri = (
                f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"
                f":valuation:{document_id.strip()}:{block_id.strip()}"
            )
            filters.append(f"FILTER(?val = <{val_uri}>)")
        elif document_id:
            doc_prefix = (
                f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"
                f":valuation:{document_id.strip()}:"
            )
            filters.append(f'FILTER(STRSTARTS(STR(?val), "{doc_prefix}"))')

        if valence == "positive":
            filters.append("FILTER(xsd:float(?cumVal) > 0)")
        elif valence == "negative":
            filters.append("FILTER(xsd:float(?cumVal) < 0)")

        filter_clause = "\n  ".join(filters)

        query = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?val ?blockRef ?cumImp ?cumVal ?rawImp ?rawVal ?impCount ?valCount ?lastVal
WHERE {{
  ?val doc:blockRef ?blockRef .
  ?val doc:cumulativeImportance ?cumImp .
  ?val doc:cumulativeValence ?cumVal .
  OPTIONAL {{ ?val doc:rawImportanceSum ?rawImp }}
  OPTIONAL {{ ?val doc:rawValenceSum ?rawVal }}
  OPTIONAL {{ ?val doc:importanceCount ?impCount }}
  OPTIONAL {{ ?val doc:valenceCount ?valCount }}
  OPTIONAL {{ ?val doc:lastValuatedAt ?lastVal }}
  {filter_clause}
}}
ORDER BY DESC(xsd:float(?cumImp))
LIMIT {min(limit * 3, 200)}
"""
        rows = await _sparql_query(backend_config, job_stream, auth, graph_id, query)

        # Load weights config
        weights = dict(DEFAULT_WEIGHTS)
        try:
            await _ensure_scratchpad(hp_client, graph_id, auth)
            await hp_client.connect_document(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id)
            w_channel = hp_client.get_document_channel(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id)
            if w_channel:
                w_reader = DocumentReader(w_channel.doc)
                w_xml = w_reader.to_xml()
                weights = _parse_weights_text(tiptap_xml_to_markdown(w_xml))
        except Exception:
            logger.debug("Could not load weights config, using defaults")

        # Build wire indexes (fail-safe — composite degrades gracefully without wires)
        doc_wire_counts: Dict[str, int] = {}
        block_wire_counts: Dict[str, int] = {}
        doc_newest_wire: Dict[str, str] = {}
        try:
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            ws_channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
            if ws_channel:
                all_wires = _get_all_wires(ws_channel.doc)
                doc_wire_counts, block_wire_counts, doc_newest_wire = _build_wire_indexes(all_wires)
        except Exception:
            logger.debug("Could not load wire data for composite scoring")

        now = datetime.now(timezone.utc)

        # Parse results and compute composite scores
        blocks = []
        for row in rows:
            cum_imp = float(row.get("cumImp", 0))
            cum_val = float(row.get("cumVal", 0))

            # Extract doc_id and block_id from blockRef URI
            block_ref = row.get("blockRef", "")
            parsed_doc_id = ""
            parsed_block_id = ""
            if "#block-" in block_ref:
                pre, _, parsed_block_id = block_ref.rpartition("#block-")
                if ":doc:" in pre:
                    parsed_doc_id = pre.rpartition(":doc:")[2]

            # Temporal decay from last valuation
            last_val_str = row.get("lastVal", "")
            doc_age_days = 0.0
            if last_val_str:
                try:
                    last_val_dt = datetime.fromisoformat(last_val_str.replace("Z", "+00:00"))
                    doc_age_days = max(0.0, (now - last_val_dt).total_seconds() / 86400.0)
                except (ValueError, TypeError):
                    pass

            # Wire counts
            bwc = block_wire_counts.get(parsed_block_id, 0)
            dwc = doc_wire_counts.get(parsed_doc_id, 0)

            # Wire freshness
            wire_age_days = 0.0
            newest_wire = doc_newest_wire.get(parsed_doc_id, "")
            if newest_wire:
                try:
                    wire_dt = datetime.fromisoformat(newest_wire.replace("Z", "+00:00"))
                    wire_age_days = max(0.0, (now - wire_dt).total_seconds() / 86400.0)
                except (ValueError, TypeError):
                    pass

            composite = _compute_composite_score(
                importance=cum_imp,
                valence=cum_val,
                doc_age_days=doc_age_days,
                block_wire_count=bwc,
                doc_wire_count=dwc,
                wire_age_days=wire_age_days,
                weights=weights,
            )

            if min_score is not None and composite < min_score:
                continue

            entry = {
                "block_id": parsed_block_id,
                "document_id": parsed_doc_id,
                "cumulative_importance": round(cum_imp, 4),
                "cumulative_valence": round(cum_val, 4),
                "composite_score": composite,
                "importance_count": int(row.get("impCount", 0)),
                "valence_count": int(row.get("valCount", 0)),
                "last_valuated": last_val_str,
                "block_wire_count": bwc,
                "doc_wire_count": dwc,
            }

            blocks.append(entry)

        # Sort by composite score descending, trim to requested limit
        blocks.sort(key=lambda b: b["composite_score"], reverse=True)
        blocks = blocks[:limit]

        return _render_json({"blocks": blocks, "count": len(blocks)})

    @server.tool(
        name="get_important_blocks",
        title="Get Important Blocks",
        description=(
            "Orientation tool: returns the highest-scored blocks in the graph with their "
            "actual text content and document titles. Use at session start to understand "
            "what matters most. Optional document_id to scope to one document."
        ),
    )
    async def get_important_blocks_tool(
        graph_id: str,
        document_id: Optional[str] = None,
        limit: int = 5,
        valence: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()
        user_id = _resolve_user_id(auth)

        graph_id = graph_id.strip()

        # Build SPARQL filter
        filters = []
        if document_id:
            doc_prefix = (
                f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"
                f":valuation:{document_id.strip()}:"
            )
            filters.append(f'FILTER(STRSTARTS(STR(?val), "{doc_prefix}"))')

        if valence == "positive":
            filters.append("FILTER(xsd:float(?cumVal) > 0)")
        elif valence == "negative":
            filters.append("FILTER(xsd:float(?cumVal) < 0)")

        filter_clause = "\n  ".join(filters)

        query = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?val ?blockRef ?cumImp ?cumVal ?lastVal
WHERE {{
  ?val doc:blockRef ?blockRef .
  ?val doc:cumulativeImportance ?cumImp .
  ?val doc:cumulativeValence ?cumVal .
  OPTIONAL {{ ?val doc:lastValuatedAt ?lastVal }}
  {filter_clause}
}}
ORDER BY DESC(xsd:float(?cumImp))
LIMIT {min(limit * 3, 60)}
"""
        rows = await _sparql_query(backend_config, job_stream, auth, graph_id, query)

        # Load weights
        weights = dict(DEFAULT_WEIGHTS)
        try:
            await _ensure_scratchpad(hp_client, graph_id, auth)
            await hp_client.connect_document(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id)
            w_channel = hp_client.get_document_channel(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id)
            if w_channel:
                w_reader = DocumentReader(w_channel.doc)
                weights = _parse_weights_text(tiptap_xml_to_markdown(w_reader.to_xml()))
        except Exception:
            pass

        # Build wire indexes
        doc_wire_counts: Dict[str, int] = {}
        block_wire_counts: Dict[str, int] = {}
        doc_newest_wire: Dict[str, str] = {}
        ws_doc = None
        try:
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            ws_channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
            if ws_channel:
                ws_doc = ws_channel.doc
                all_wires = _get_all_wires(ws_doc)
                doc_wire_counts, block_wire_counts, doc_newest_wire = _build_wire_indexes(all_wires)
        except Exception:
            pass

        now = datetime.now(timezone.utc)

        # Score and rank
        scored = []
        for row in rows:
            cum_imp = float(row.get("cumImp", 0))
            cum_val = float(row.get("cumVal", 0))

            block_ref = row.get("blockRef", "")
            parsed_doc_id = ""
            parsed_block_id = ""
            if "#block-" in block_ref:
                pre, _, parsed_block_id = block_ref.rpartition("#block-")
                if ":doc:" in pre:
                    parsed_doc_id = pre.rpartition(":doc:")[2]

            last_val_str = row.get("lastVal", "")
            doc_age_days = 0.0
            if last_val_str:
                try:
                    last_val_dt = datetime.fromisoformat(last_val_str.replace("Z", "+00:00"))
                    doc_age_days = max(0.0, (now - last_val_dt).total_seconds() / 86400.0)
                except (ValueError, TypeError):
                    pass

            bwc = block_wire_counts.get(parsed_block_id, 0)
            dwc = doc_wire_counts.get(parsed_doc_id, 0)

            wire_age_days = 0.0
            newest_wire = doc_newest_wire.get(parsed_doc_id, "")
            if newest_wire:
                try:
                    wire_dt = datetime.fromisoformat(newest_wire.replace("Z", "+00:00"))
                    wire_age_days = max(0.0, (now - wire_dt).total_seconds() / 86400.0)
                except (ValueError, TypeError):
                    pass

            composite = _compute_composite_score(
                importance=cum_imp, valence=cum_val,
                doc_age_days=doc_age_days, block_wire_count=bwc,
                doc_wire_count=dwc, wire_age_days=wire_age_days,
                weights=weights,
            )

            scored.append({
                "doc_id": parsed_doc_id,
                "block_id": parsed_block_id,
                "composite": composite,
                "importance": round(cum_imp, 3),
                "valence": round(cum_val, 3),
                "block_wires": bwc,
                "doc_wires": dwc,
            })

        scored.sort(key=lambda b: b["composite"], reverse=True)
        scored = scored[:limit]

        # Fetch content and titles for the top blocks
        results = []
        # Group by doc_id to minimize connections
        doc_groups: Dict[str, list] = {}
        for item in scored:
            doc_groups.setdefault(item["doc_id"], []).append(item)

        for did, items in doc_groups.items():
            title = _resolve_title_from_workspace(ws_doc, did) or did
            try:
                await hp_client.connect_document(graph_id, did, user_id=auth.user_id)
                channel = hp_client.get_document_channel(graph_id, did, user_id=auth.user_id)
                if channel is None:
                    continue
                reader = DocumentReader(channel.doc)
                for item in items:
                    block_info = reader.get_block_info(item["block_id"])
                    content = block_info["text_content"] if block_info else "(deleted)"
                    results.append({
                        "content": content,
                        "document": title,
                        "score": item["composite"],
                        "importance": item["importance"],
                        "valence": item["valence"],
                        "block_id": item["block_id"],
                        "doc_id": item["doc_id"],
                        "block_wires": item["block_wires"],
                        "doc_wires": item["doc_wires"],
                    })
            except Exception:
                for item in items:
                    results.append({
                        "content": "(unavailable)",
                        "document": title,
                        "score": item["composite"],
                        "importance": item["importance"],
                        "valence": item["valence"],
                        "block_id": item["block_id"],
                        "doc_id": item["doc_id"],
                        "block_wires": item["block_wires"],
                        "doc_wires": item["doc_wires"],
                    })

        # Re-sort since doc grouping may have scrambled order
        results.sort(key=lambda b: b["score"], reverse=True)

        return _render_json({"blocks": results, "count": len(results)})

    @server.tool(
        name="get_values",
        title="Get Scoring Configuration",
        description=(
            "Return the full scoring configuration for this graph: importance prompt, "
            "valence prompt, component weights, reference values, and temporal half-life. "
            "This is what shapes how blocks are scored. Users and agents can modify these "
            "via the revaluate tool."
        ),
    )
    async def get_values_tool(
        graph_id: str,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        await _ensure_scratchpad(hp_client, graph_id, auth)

        # Read config documents
        config = {}

        for doc_id, key in [
            (IMPORTANCE_DOC_ID, "importance_prompt"),
            (VALENCE_DOC_ID, "valence_prompt"),
            (WEIGHTS_DOC_ID, "weights_raw"),
        ]:
            await hp_client.connect_document(graph_id, doc_id, user_id=auth.user_id)
            channel = hp_client.get_document_channel(graph_id, doc_id, user_id=auth.user_id)
            reader = DocumentReader(channel.doc)
            xml = reader.to_xml()
            config[key] = tiptap_xml_to_markdown(xml)

        # Parse weights into structured config
        weights = _parse_weights_text(config["weights_raw"])

        return _render_json({
            "importance_prompt": config["importance_prompt"],
            "valence_prompt": config["valence_prompt"],
            "weights": weights,
        })

    @server.tool(
        name="revaluate",
        title="Update Scoring Configuration",
        description=(
            "Update any part of the scoring configuration. All parameters are optional — "
            "pass only what you want to change. The current config is archived to past/ "
            "before being replaced (preserving genealogy). Can update importance/valence "
            "prompts, component weights, reference values, and/or temporal half-life."
        ),
    )
    async def revaluate_tool(
        graph_id: str,
        importance_prompt: Optional[str] = None,
        valence_prompt: Optional[str] = None,
        weights: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        await _ensure_scratchpad(hp_client, graph_id, auth)

        now_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        updated = []

        # Helper: archive-then-replace a config document
        async def _update_config_doc(
            doc_id: str, past_doc_id: str, new_content: str, label: str
        ) -> None:
            # Read current
            await hp_client.connect_document(graph_id, doc_id, user_id=auth.user_id)
            channel = hp_client.get_document_channel(graph_id, doc_id, user_id=auth.user_id)
            reader = DocumentReader(channel.doc)
            current_xml = reader.to_xml()
            current_md = tiptap_xml_to_markdown(current_xml)

            # Archive to past/ doc (read existing, append entry, rewrite)
            archive_entry = (
                f'<heading level="3">Archived {html_mod.escape(now_label)}</heading>'
                f"<paragraph>{html_mod.escape(current_md)}</paragraph>"
                "<horizontalRule/>"
            )
            await hp_client.connect_document(graph_id, past_doc_id, user_id=auth.user_id)
            archive_channel = hp_client.get_document_channel(
                graph_id, past_doc_id, user_id=auth.user_id
            )
            existing_archive = DocumentReader(archive_channel.doc).to_xml()
            new_archive = existing_archive + archive_entry

            await hp_client.transact_document(
                graph_id,
                past_doc_id,
                lambda doc, na=new_archive: DocumentWriter(doc).replace_all_content(na),
                user_id=auth.user_id,
            )

            # Write new content
            new_xml = f'<heading level="1">{html_mod.escape(label)}</heading>'
            for line in new_content.strip().split("\n"):
                if line.strip():
                    new_xml += f"<paragraph>{html_mod.escape(line.strip())}</paragraph>"

            await hp_client.transact_document(
                graph_id,
                doc_id,
                lambda doc, nx=new_xml: DocumentWriter(doc).replace_all_content(nx),
                user_id=auth.user_id,
            )

        if importance_prompt is not None:
            await _update_config_doc(
                IMPORTANCE_DOC_ID, PAST_IMPORTANCE_DOC_ID,
                importance_prompt, "Importance Prompt"
            )
            updated.append("importance_prompt")

        if valence_prompt is not None:
            await _update_config_doc(
                VALENCE_DOC_ID, PAST_VALENCE_DOC_ID,
                valence_prompt, "Valence Prompt"
            )
            updated.append("valence_prompt")

        if weights is not None:
            # Merge new weights into existing config (don't nuke unchanged values)
            await hp_client.connect_document(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id)
            channel = hp_client.get_document_channel(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id)
            current_md = tiptap_xml_to_markdown(DocumentReader(channel.doc).to_xml())
            current_weights = _parse_weights_text(current_md)

            # Parse the incoming changes and overlay onto current config
            new_weights = _parse_weights_text(weights)
            # _parse_weights_text starts from DEFAULT_WEIGHTS, so we need to detect
            # which keys the caller actually specified vs which are just defaults.
            # Parse raw to find only explicitly mentioned keys.
            specified_keys = set()
            for line in weights.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key = line.partition(":")[0].strip()
                elif "=" in line:
                    key = line.partition("=")[0].strip()
                else:
                    continue
                if key in DEFAULT_WEIGHTS:
                    specified_keys.add(key)

            # Merge: start from current, overlay only specified keys
            merged = dict(current_weights)
            for key in specified_keys:
                merged[key] = new_weights[key]

            # Render as canonical "key: value" lines
            merged_text = "\n".join(f"{k}: {v}" for k, v in merged.items())
            await _update_config_doc(
                WEIGHTS_DOC_ID, PAST_WEIGHTS_DOC_ID,
                merged_text, "Scoring Configuration"
            )
            updated.append("weights")

        return _render_json({"success": True, "updated": updated})

    logger.info("Registered Geist (Sophia Memory) tools: 9 tools")


# ── Song parsing helpers ────────────────────────────────────────────


def _split_song_xml(xml: str) -> list[str]:
    """Split song XML into verse sections by <horizontalRule/> elements.

    Returns a list of XML strings, one per verse (excluding the title heading
    and the horizontal rules themselves).
    """
    # Remove the title heading
    xml = re.sub(r"<heading[^>]*level=\"1\"[^>]*>.*?</heading>", "", xml, count=1)

    # Split on horizontal rules (CRDT adds data-block-id attrs, so match any attributes)
    parts = re.split(r"<horizontalRule[^>]*/?>", xml)

    verses = []
    for part in parts:
        # Strip verse headings (Verse I, Verse II, etc.) but keep content
        content = re.sub(r"<heading[^>]*level=\"2\"[^>]*>.*?</heading>", "", part)
        content = content.strip()
        if content:
            verses.append(content)

    return verses


def _verse_to_xml(verse_text: str) -> str:
    """Convert verse text to TipTap XML paragraphs.

    Lines ending with / get a stanza break after them.
    """
    lines = verse_text.strip().split("\n")
    parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # / at end of line marks a stanza break
        if line.endswith("/"):
            line = line[:-1].rstrip()
            if line:
                parts.append(f"<paragraph>{html_mod.escape(line)}</paragraph>")
            parts.append("<paragraph></paragraph>")  # Visual stanza break
        else:
            parts.append(f"<paragraph>{html_mod.escape(line)}</paragraph>")
    return "".join(parts)
