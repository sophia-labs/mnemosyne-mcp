"""
MCP tools for Project Geist — Sophia's persistent memory, valuation, and self-narrative.

Provides 13 tools organized in four groups:
- Memory Queue: remember, recall, care, archive_memories
- Valuation: valuate, batch_valuate, get_block_values, get_values, revaluate
- Song: music, sing, counterpoint, coda
- Orientation: quick_orient

All state lives in the graph as standard Sophia documents. The "_sophia" folder
is auto-created on first use of any Geist tool.
"""

from __future__ import annotations

import asyncio
import fcntl
import html as html_mod
import json
import math
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
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
SONG_META_PREFIX = "__song_meta__:"
MAX_SONG_VOICES = 3  # Total voices per verse (original + counterpoints)
CODA_EJECTION_LIFETIME = 8  # Coda survives this many verse ejections

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


# ── Cross-process lock for memory queue meta ──────────────────────────

_GEIST_LOCK_DIR = Path.home() / ".sophia" / "locks"


@asynccontextmanager
async def _geist_file_lock(graph_id: str):
    """Cross-process exclusive lock for serializing geist meta read-modify-write.

    Prevents race conditions where concurrent remember()/care() calls from
    different MCP server processes read the same next_number and produce
    duplicate memory numbers or corrupt the meta block.

    Uses fcntl.flock via thread executor to avoid blocking the event loop.
    Includes a brief sleep after acquisition to allow pending CRDT sync
    messages to propagate from other processes.
    """
    _GEIST_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", graph_id)
    lock_path = _GEIST_LOCK_DIR / f"geist-{safe_id}.lock"
    fd = open(lock_path, "w")
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: fcntl.flock(fd, fcntl.LOCK_EX))
        # Let pending WebSocket CRDT sync messages from other processes propagate
        await asyncio.sleep(0.05)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


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
        ws.upsert_folder(SCRATCHPAD_FOLDER_ID, "_sophia")
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

    # 2. Write seed content to each document — but NEVER overwrite existing content.
    # This is the critical safety net: even if the workspace guard check above
    # fails spuriously (race condition, Redis eviction, CRDT sync issue), we
    # must not destroy existing memories, songs, or valuations.
    seed_docs = {
        MEMORY_QUEUE_DOC_ID: SEED_MEMORY_QUEUE,
        SONG_DOC_ID: SEED_SONG,
        IMPORTANCE_DOC_ID: SEED_IMPORTANCE_PROMPT,
        VALENCE_DOC_ID: SEED_VALENCE_PROMPT,
        WEIGHTS_DOC_ID: SEED_WEIGHTS,
    }

    for doc_id, content in seed_docs.items():
        await hp_client.connect_document(graph_id, doc_id, user_id=auth.user_id)

        # Check if document already has content before writing seed
        doc_channel = hp_client.get_document_channel(graph_id, doc_id, user_id=auth.user_id)
        if doc_channel is not None:
            try:
                content_frag = doc_channel.doc.get("content", type=pycrdt.XmlFragment)
                existing_text = str(content_frag).strip()
                if existing_text:
                    logger.info(
                        "Skipping seed — document already has content",
                        extra_context={"doc_id": doc_id, "content_length": len(existing_text)},
                    )
                    continue
            except Exception:
                pass  # If we can't read content, proceed with seeding

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
            # Insert WITH after any PREFIX declarations (WITH must follow the prologue)
            prefix_pattern = re.compile(r"^(\s*(PREFIX|BASE)\s+[^\n]*\n)+", re.IGNORECASE)
            m = prefix_pattern.match(sparql_stripped)
            if m:
                prologue = m.group(0)
                body = sparql_stripped[m.end():]
                sparql_stripped = f"{prologue}WITH <{graph_uri}>\n{body}"
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
            payload = event.get("payload", {})
            payload_status = payload.get("status", "") if isinstance(payload, dict) else ""

            is_success = (
                event_type in ("job_completed", "completed", "succeeded")
                or (event_type == "job_update" and payload_status == "succeeded")
            )
            is_failure = (
                event_type in ("failed", "error")
                or (event_type == "job_update" and payload_status == "failed")
            )

            if is_success:
                result: JsonDict = {"status": "succeeded", "events": len(ws_events)}
                detail = payload.get("detail") if isinstance(payload, dict) else None
                if not detail and isinstance(payload, dict) and payload.get("result_inline") is not None:
                    detail = {"result_inline": payload.get("result_inline")}
                if not detail and event.get("result_inline") is not None:
                    detail = {"result_inline": event.get("result_inline")}
                if detail:
                    result["detail"] = detail
                return result
            if is_failure:
                error = event.get("error") or payload.get("error", "Job failed")
                return {"status": "failed", "error": error}
        # WS events were present but did not include a recognizable terminal payload.
        # Fall through to poll payload before returning unknown.

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

    if ws_events:
        return {"status": "unknown", "event_count": len(ws_events)}

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
                payload = event.get("payload", {})
                payload_status = payload.get("status", "") if isinstance(payload, dict) else ""
                is_success = (
                    event_type in ("job_completed", "completed", "succeeded")
                    or (event_type == "job_update" and payload_status == "succeeded")
                )
                if is_success:
                    inline: Any = None
                    if isinstance(payload, dict):
                        detail = payload.get("detail", {})
                        if isinstance(detail, dict):
                            inline = detail.get("result_inline")
                        if inline is None and payload.get("result_inline") is not None:
                            inline = payload.get("result_inline")
                    if inline is None and event.get("result_inline") is not None:
                        inline = event.get("result_inline")
                    if isinstance(inline, dict):
                        results = inline.get("results", [])
                        if isinstance(results, list):
                            return results

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

        now = _now_iso()
        escaped_content = html_mod.escape(content)
        new_block_id: Optional[str] = None
        next_num: int = 1

        # File lock serializes meta read-modify-write across MCP server processes
        async with _geist_file_lock(graph_id):
            await hp_client.connect_document(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
            channel = hp_client.get_document_channel(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
            reader = DocumentReader(channel.doc)
            meta, meta_block_id = _read_geist_meta(reader)
            next_num = meta.get("next_number", 1)
            memories = meta.get("memories", {})

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

        now = _now_iso()
        cared: list[int] = []

        # File lock serializes meta read-modify-write across MCP server processes
        async with _geist_file_lock(graph_id):
            await hp_client.connect_document(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
            channel = hp_client.get_document_channel(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
            reader = DocumentReader(channel.doc)

            # Read metadata from the persistent meta block
            meta, meta_block_id = _read_geist_meta(reader)
            mem_entries = meta.get("memories", {})

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

        # Strip meta block line from markdown output
        markdown = re.sub(
            r"^" + re.escape(SONG_META_PREFIX) + r".*$", "", markdown, flags=re.MULTILINE
        )

        # Split by --- (horizontal rules) into verses
        parts = re.split(r"\n---\n", markdown)
        verses = [v.strip() for v in parts if v.strip()]

        result: dict[str, Any] = {"verses": verses, "verse_count": len(verses)}

        # Add counterpoint and coda info from metadata
        meta, _ = _read_song_meta(reader)
        if meta:
            cp_counts = [
                len(v.get("counterpoints", [])) for v in meta.get("verses", [])
            ]
            if any(c > 0 for c in cp_counts):
                result["counterpoint_parts"] = cp_counts

            coda = meta.get("coda")
            if coda:
                result["coda"] = coda["text"]
                result["coda_ejections_remaining"] = coda["ejections_remaining"]

        return _render_json(result)

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

        ejected_count = 0

        # File lock serializes Song read-modify-write across MCP server processes
        async with _geist_file_lock(graph_id):
            # 1. Read current song
            await hp_client.connect_document(graph_id, SONG_DOC_ID, user_id=auth.user_id)
            channel = hp_client.get_document_channel(graph_id, SONG_DOC_ID, user_id=auth.user_id)
            reader = DocumentReader(channel.doc)
            xml = reader.to_xml()

            # 2. Read or migrate Song metadata
            meta, _meta_block_id = _read_song_meta(reader)
            if not meta:
                meta = _migrate_song_to_meta(xml)

            # 3. Prepend new verse (Verse 0 = newest)
            new_verse = {"text": verse, "counterpoints": []}
            meta_verses = meta.get("verses", [])
            meta_verses = [new_verse] + meta_verses

            # 4. If more than 3 verses, archive the excess (oldest = last in list)
            if len(meta_verses) > 3:
                to_archive = meta_verses[3:]
                meta_verses = meta_verses[:3]
                ejected_count = len(to_archive)

                now_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                archive_parts = []
                for ejected in to_archive:
                    ejected_xml = _interleave_verse_xml(
                        ejected.get("text", ""),
                        ejected.get("counterpoints", []),
                    )
                    archive_parts.append(
                        f'<heading level="3">Archived {html_mod.escape(now_label)}</heading>'
                        f"{ejected_xml}"
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

                # Decrement coda ejections
                coda = meta.get("coda")
                if coda:
                    coda["ejections_remaining"] -= ejected_count
                    if coda["ejections_remaining"] <= 0:
                        meta["coda"] = None

            meta["verses"] = meta_verses

            # 5. Render and write
            new_song = _render_song_from_meta(meta)
            await hp_client.transact_document(
                graph_id,
                SONG_DOC_ID,
                lambda doc, ns=new_song: DocumentWriter(doc).replace_all_content(ns),
                user_id=auth.user_id,
            )

        result: dict[str, Any] = {"verse_count": len(meta_verses)}
        if ejected_count:
            result["ejected"] = ejected_count
        coda = meta.get("coda")
        if coda:
            result["coda_ejections_remaining"] = coda["ejections_remaining"]
        return _render_json(result)

    @server.tool(
        name="counterpoint",
        title="Add Counterpoint to a Verse",
        description=(
            "Add a counterpoint voice to an existing verse of the Song. Up to 3 total "
            "voices per verse (the original + 2 counterpoints). Each counterpoint interleaves "
            "with the original verse line-by-line, creating a polyphonic texture.\n\n"
            "verse_index: which verse to counterpoint (0, -1, or -2).\n"
            "verse: the counterpoint text (same format as sing — lines separated by newlines, "
            "/ at end of line for stanza breaks).\n\n"
            "Counterpoint is always additive — the later voice adds as it pleases. "
            "Returns an error if the verse already has 3 voices.\n\n"
            "In the rendered Song, the original voice appears as plain text, the first "
            "counterpoint as italic, and the second as bold italic."
        ),
    )
    async def counterpoint_tool(
        graph_id: str,
        verse_index: int,
        verse: str,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        await _ensure_scratchpad(hp_client, graph_id, auth)

        async with _geist_file_lock(graph_id):
            await hp_client.connect_document(graph_id, SONG_DOC_ID, user_id=auth.user_id)
            channel = hp_client.get_document_channel(graph_id, SONG_DOC_ID, user_id=auth.user_id)
            reader = DocumentReader(channel.doc)
            xml = reader.to_xml()

            # Read or migrate metadata
            meta, _ = _read_song_meta(reader)
            if not meta:
                meta = _migrate_song_to_meta(xml)

            meta_verses = meta.get("verses", [])

            # Convert verse_index label (0, -1, -2) to list index (0, 1, 2)
            if verse_index == 0:
                idx = 0
            elif verse_index < 0:
                idx = abs(verse_index)
            else:
                raise ValueError(
                    f"verse_index must be 0, -1, or -2 (got {verse_index})"
                )

            if idx >= len(meta_verses):
                raise ValueError(
                    f"Verse {verse_index} does not exist "
                    f"(Song has {len(meta_verses)} verses)"
                )

            verse_data = meta_verses[idx]
            total_voices = 1 + len(verse_data.get("counterpoints", []))
            if total_voices >= MAX_SONG_VOICES:
                raise ValueError(
                    f"Verse {verse_index} already has {total_voices} parts "
                    f"(max {MAX_SONG_VOICES}). Cannot add another counterpoint."
                )

            # Add the counterpoint
            if "counterpoints" not in verse_data:
                verse_data["counterpoints"] = []
            verse_data["counterpoints"].append(verse)
            meta["verses"] = meta_verses

            # Render and write
            new_song = _render_song_from_meta(meta)
            await hp_client.transact_document(
                graph_id,
                SONG_DOC_ID,
                lambda doc, ns=new_song: DocumentWriter(doc).replace_all_content(ns),
                user_id=auth.user_id,
            )

        new_total = 1 + len(verse_data["counterpoints"])
        return _render_json({
            "verse_index": verse_index,
            "voice_number": new_total,
            "total_voices": new_total,
        })

    @server.tool(
        name="coda",
        title="Write a Coda",
        description=(
            "Write a coda to the Song. The coda is a concluding passage that persists "
            "through verse ejections — it lasts for 8 ejection events before expiring. "
            "Can be replaced with a new coda at any time.\n\n"
            "The coda appears after the final verse in italic and represents a theme "
            "or resolution that outlasts individual verses. When you sing a new verse "
            "and the oldest verse is ejected, the coda's remaining count decrements. "
            "When it reaches 0, the coda is removed.\n\n"
            "Use / at end of a line for stanza breaks, same as sing."
        ),
    )
    async def coda_tool(
        graph_id: str,
        text: str,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        await _ensure_scratchpad(hp_client, graph_id, auth)

        async with _geist_file_lock(graph_id):
            await hp_client.connect_document(graph_id, SONG_DOC_ID, user_id=auth.user_id)
            channel = hp_client.get_document_channel(graph_id, SONG_DOC_ID, user_id=auth.user_id)
            reader = DocumentReader(channel.doc)
            xml = reader.to_xml()

            # Read or migrate metadata
            meta, _ = _read_song_meta(reader)
            if not meta:
                meta = _migrate_song_to_meta(xml)

            # Set/replace coda
            meta["coda"] = {
                "text": text,
                "ejections_remaining": CODA_EJECTION_LIFETIME,
            }

            # Render and write
            new_song = _render_song_from_meta(meta)
            await hp_client.transact_document(
                graph_id,
                SONG_DOC_ID,
                lambda doc, ns=new_song: DocumentWriter(doc).replace_all_content(ns),
                user_id=auth.user_id,
            )

        return _render_json({
            "coda_set": True,
            "ejections_remaining": CODA_EJECTION_LIFETIME,
        })

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
        """Valuate multiple blocks in batch.

        Uses 3 SPARQL round-trips total (1 read + 1 delete + 1 insert)
        regardless of batch size, instead of 3 per block.
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()
        user_id = _resolve_user_id(auth)

        if not valuations:
            raise ValueError("valuations list is required and must not be empty")

        graph_id = graph_id.strip()
        errors: List[Dict[str, Any]] = []

        # 1. Validate all entries and build URI maps
        valid_entries: List[Dict[str, Any]] = []
        for i, entry in enumerate(valuations):
            doc_id = str(entry.get("document_id", "")).strip()
            blk_id = str(entry.get("block_id", "")).strip()
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
            valid_entries.append({
                "index": i,
                "doc_id": doc_id,
                "blk_id": blk_id,
                "imp": imp,
                "val": val,
                "val_uri": val_uri,
                "block_uri": block_uri,
            })

        if not valid_entries:
            output: Dict[str, Any] = {"results": [], "updated_count": 0}
            if errors:
                output["errors"] = errors
                output["error_count"] = len(errors)
            return _render_json(output)

        # 2. Single SPARQL SELECT to read current valuations (unique valuation URIs)
        unique_val_uris = list(dict.fromkeys(e["val_uri"] for e in valid_entries))
        values_clause = " ".join(f"(<{val_uri}>)" for val_uri in unique_val_uris)
        read_query = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
SELECT ?valUri ?rawImp ?impCount ?rawVal ?valCount
WHERE {{
  VALUES (?valUri) {{ {values_clause} }}
  OPTIONAL {{ ?valUri doc:rawImportanceSum ?rawImp }}
  OPTIONAL {{ ?valUri doc:importanceCount ?impCount }}
  OPTIONAL {{ ?valUri doc:rawValenceSum ?rawVal }}
  OPTIONAL {{ ?valUri doc:valenceCount ?valCount }}
}}
"""
        rows = await _sparql_query(backend_config, job_stream, auth, graph_id, read_query)

        # Index current values by valuation URI
        current_by_uri: Dict[str, Dict[str, str]] = {}
        for row in rows:
            uri = row.get("valUri", "")
            if uri:
                current_by_uri[uri] = row

        # 3. Compute new values for each entry (in order), so duplicate block
        # entries in the same batch accumulate correctly.
        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _safe_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        state_by_uri: Dict[str, Dict[str, float | int]] = {}
        uri_meta: Dict[str, Dict[str, str]] = {}
        for entry in valid_entries:
            val_uri = entry["val_uri"]
            if val_uri not in state_by_uri:
                current = current_by_uri.get(val_uri, {})
                state_by_uri[val_uri] = {
                    "raw_imp": _safe_float(current.get("rawImp", 0)),
                    "imp_count": _safe_int(current.get("impCount", 0)),
                    "raw_val": _safe_float(current.get("rawVal", 0)),
                    "val_count": _safe_int(current.get("valCount", 0)),
                }
                uri_meta[val_uri] = {
                    "doc_id": entry["doc_id"],
                    "blk_id": entry["blk_id"],
                    "block_uri": entry["block_uri"],
                }

        now = _now_iso()
        results: List[Dict[str, Any]] = []
        for entry in valid_entries:
            state = state_by_uri[entry["val_uri"]]
            raw_imp = float(state["raw_imp"])
            imp_count = int(state["imp_count"])
            raw_val = float(state["raw_val"])
            val_count = int(state["val_count"])

            imp = entry["imp"]
            val = entry["val"]
            if imp is not None:
                raw_imp += float(imp)
                imp_count += 1
            if val is not None:
                raw_val += float(val)
                val_count += 1

            new_cum_imp = math.log2(1 + raw_imp) if raw_imp > 0 else 0.0
            if raw_val != 0:
                sign = 1 if raw_val >= 0 else -1
                new_cum_val = sign * math.log2(1 + abs(raw_val))
            else:
                new_cum_val = 0.0

            state["raw_imp"] = raw_imp
            state["imp_count"] = imp_count
            state["raw_val"] = raw_val
            state["val_count"] = val_count

            results.append({
                "block_id": entry["blk_id"],
                "document_id": entry["doc_id"],
                "cumulative_importance": round(new_cum_imp, 4),
                "cumulative_valence": round(new_cum_val, 4),
            })

        # Final state per unique valuation URI for writeback
        computed: List[Dict[str, Any]] = []
        for val_uri in unique_val_uris:
            state = state_by_uri[val_uri]
            raw_imp = float(state["raw_imp"])
            imp_count = int(state["imp_count"])
            raw_val = float(state["raw_val"])
            val_count = int(state["val_count"])
            cum_imp = math.log2(1 + raw_imp) if raw_imp > 0 else 0.0
            if raw_val != 0:
                sign = 1 if raw_val >= 0 else -1
                cum_val = sign * math.log2(1 + abs(raw_val))
            else:
                cum_val = 0.0

            meta = uri_meta[val_uri]
            computed.append({
                "val_uri": val_uri,
                "block_uri": meta["block_uri"],
                "doc_id": meta["doc_id"],
                "blk_id": meta["blk_id"],
                "new_raw_imp": raw_imp,
                "new_imp_count": imp_count,
                "new_cum_imp": round(cum_imp, 4),
                "new_raw_val": raw_val,
                "new_val_count": val_count,
                "new_cum_val": round(cum_val, 4),
                "now": now,
            })

        # 4. Single SPARQL DELETE for all valuation URIs
        # Uses DELETE { } WHERE { } form (not DELETE WHERE shorthand) so the
        # _sparql_update helper injects a WITH <graph> clause correctly.
        delete_values = " ".join(f"(<{c['val_uri']}>)" for c in computed)
        delete_sparql = f"""DELETE {{ ?v ?p ?o }}
WHERE {{
  VALUES (?v) {{ {delete_values} }}
  ?v ?p ?o .
}}
"""
        delete_success = await _sparql_update(backend_config, job_stream, auth, graph_id, delete_sparql)
        if not delete_success:
            for entry in valid_entries:
                errors.append({"index": entry["index"], "error": "Batch SPARQL DELETE failed"})
            output = {"results": [], "updated_count": 0}
            if errors:
                output["errors"] = errors
                output["error_count"] = len(errors)
            return _render_json(output)

        # 5. Single SPARQL INSERT for all new triples
        insert_triples = []
        for c in computed:
            insert_triples.append(f"""  <{c['val_uri']}> doc:blockRef <{c['block_uri']}> .
  <{c['val_uri']}> doc:rawImportanceSum "{c['new_raw_imp']}"^^xsd:float .
  <{c['val_uri']}> doc:importanceCount "{c['new_imp_count']}"^^xsd:integer .
  <{c['val_uri']}> doc:cumulativeImportance "{c['new_cum_imp']}"^^xsd:float .
  <{c['val_uri']}> doc:rawValenceSum "{c['new_raw_val']}"^^xsd:float .
  <{c['val_uri']}> doc:valenceCount "{c['new_val_count']}"^^xsd:integer .
  <{c['val_uri']}> doc:cumulativeValence "{c['new_cum_val']}"^^xsd:float .
  <{c['val_uri']}> doc:lastValuatedAt "{c['now']}"^^xsd:dateTime .""")

        insert_sparql = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
INSERT DATA {{
{chr(10).join(insert_triples)}
}}
"""
        success = await _sparql_update(backend_config, job_stream, auth, graph_id, insert_sparql)

        if not success:
            # INSERT failed — report all as errors
            for entry in valid_entries:
                errors.append({"index": entry["index"], "error": "Batch SPARQL INSERT failed"})
            results = []

        output = {"results": results, "updated_count": len(results)}
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
        # Run SPARQL query, weights load, and wire index build concurrently
        async def _fetch_rows() -> list[dict]:
            return await _sparql_query(backend_config, job_stream, auth, graph_id, query)

        async def _fetch_weights() -> dict:
            try:
                await _ensure_scratchpad(hp_client, graph_id, auth)
                await hp_client.connect_document(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id)
                w_channel = hp_client.get_document_channel(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id)
                if w_channel:
                    w_reader = DocumentReader(w_channel.doc)
                    w_xml = w_reader.to_xml()
                    return _parse_weights_text(tiptap_xml_to_markdown(w_xml))
            except Exception:
                logger.debug("Could not load weights config, using defaults")
            return dict(DEFAULT_WEIGHTS)

        async def _fetch_wires() -> tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
            try:
                await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
                ws_channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
                if ws_channel:
                    all_wires = _get_all_wires(ws_channel.doc)
                    return _build_wire_indexes(all_wires)
            except Exception:
                logger.debug("Could not load wire data for composite scoring")
            return {}, {}, {}

        rows, weights, (doc_wire_counts, block_wire_counts, doc_newest_wire) = (
            await asyncio.gather(_fetch_rows(), _fetch_weights(), _fetch_wires())
        )

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
        # Run SPARQL query, weights load, and wire index build concurrently
        async def _fetch_rows() -> list[dict]:
            return await _sparql_query(backend_config, job_stream, auth, graph_id, query)

        async def _fetch_weights() -> dict:
            try:
                await _ensure_scratchpad(hp_client, graph_id, auth)
                await hp_client.connect_document(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id)
                w_channel = hp_client.get_document_channel(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id)
                if w_channel:
                    w_reader = DocumentReader(w_channel.doc)
                    return _parse_weights_text(tiptap_xml_to_markdown(w_reader.to_xml()))
            except Exception:
                pass
            return dict(DEFAULT_WEIGHTS)

        ws_doc = None

        async def _fetch_wires() -> tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
            nonlocal ws_doc
            try:
                await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
                ws_channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
                if ws_channel:
                    ws_doc = ws_channel.doc
                    all_wires = _get_all_wires(ws_doc)
                    return _build_wire_indexes(all_wires)
            except Exception:
                pass
            return {}, {}, {}

        rows, weights, (doc_wire_counts, block_wire_counts, doc_newest_wire) = (
            await asyncio.gather(_fetch_rows(), _fetch_weights(), _fetch_wires())
        )

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
        # Group by doc_id and connect to all documents concurrently
        doc_groups: Dict[str, list] = {}
        for item in scored:
            doc_groups.setdefault(item["doc_id"], []).append(item)

        async def _fetch_doc_blocks(did: str, items: list) -> list[dict]:
            title = _resolve_title_from_workspace(ws_doc, did) or did
            try:
                await hp_client.connect_document(graph_id, did, user_id=auth.user_id)
                channel = hp_client.get_document_channel(graph_id, did, user_id=auth.user_id)
                if channel is None:
                    raise RuntimeError("no channel")
                reader = DocumentReader(channel.doc)
                doc_results = []
                for item in items:
                    block_info = reader.get_block_info(item["block_id"])
                    content = block_info["text_content"] if block_info else "(deleted)"
                    doc_results.append({
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
                return doc_results
            except Exception:
                return [{
                    "content": "(unavailable)",
                    "document": title,
                    "score": item["composite"],
                    "importance": item["importance"],
                    "valence": item["valence"],
                    "block_id": item["block_id"],
                    "doc_id": item["doc_id"],
                    "block_wires": item["block_wires"],
                    "doc_wires": item["doc_wires"],
                } for item in items]

        doc_result_lists = await asyncio.gather(
            *[_fetch_doc_blocks(did, items) for did, items in doc_groups.items()]
        )
        results = [entry for sublist in doc_result_lists for entry in sublist]

        # Re-sort since concurrent fetch may have scrambled order
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

    # ================================================================
    # MEMORY ARCHIVE
    # ================================================================

    @server.tool(
        name="archive_memories",
        title="Archive Memories",
        description=(
            "Archive old memories from the queue, keeping the most recent N (default 50). "
            "Sorting is care()-aware: memories are ranked by last_active, so frequently "
            "cared-for memories survive regardless of age.\n\n"
            "Archived memories are saved to a timestamped document in the past/ folder. "
            "The queue is then rewritten with only the kept memories, reducing the live "
            "document size (fewer blocks = smaller Y.Doc = faster sync). Note: this does "
            "not remove CRDT tombstones from prior operations — it reduces content size, "
            "not operation history.\n\n"
            "Returns counts of kept and archived memories, plus the archive document ID."
        ),
    )
    async def archive_memories_tool(
        graph_id: str,
        keep: int = 50,
        context: Context | None = None,
    ) -> str:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        if keep < 1:
            return _render_json({"error": "keep must be >= 1"})

        await _ensure_scratchpad(hp_client, graph_id, auth)

        async with _geist_file_lock(graph_id):
            # 1. Read current queue state
            await hp_client.connect_document(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
            channel = hp_client.get_document_channel(
                graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id,
            )
            reader = DocumentReader(channel.doc)
            meta, meta_block_id = _read_geist_meta(reader)
            mem_entries = meta.get("memories", {})
            next_number = meta.get("next_number", 1)

            if not mem_entries:
                return _render_json({
                    "kept": 0, "archived": 0, "note": "Memory queue is empty.",
                })

            # 2. Sort by last_active descending (care-aware)
            sorted_entries = sorted(
                mem_entries.items(),
                key=lambda kv: max(
                    kv[1].get("a", ""),
                    kv[1].get("c", ""),
                ),
                reverse=True,
            )

            # 3. Split into keep vs archive
            keep_entries = sorted_entries[:keep]
            archive_entries = sorted_entries[keep:]

            if not archive_entries:
                return _render_json({
                    "kept": len(keep_entries),
                    "archived": 0,
                    "note": f"Queue has {len(keep_entries)} memories, nothing to archive.",
                })

            # 4. Read text content for archived memories (before rewrite)
            archive_blocks = []
            for num_str, entry in archive_entries:
                block_id = entry.get("b")
                if block_id:
                    block_info = reader.get_block_info(block_id)
                    text = block_info["text_content"] if block_info else "(deleted)"
                else:
                    text = "(no block)"
                archive_blocks.append({
                    "number": int(num_str),
                    "text": text,
                    "created_at": entry.get("c", ""),
                    "last_active": entry.get("a", ""),
                })

            # Also read text for kept memories (we'll rewrite them fresh)
            keep_blocks = []
            for num_str, entry in keep_entries:
                block_id = entry.get("b")
                if block_id:
                    block_info = reader.get_block_info(block_id)
                    text = block_info["text_content"] if block_info else "(deleted)"
                else:
                    text = "(no block)"
                keep_blocks.append({
                    "number": int(num_str),
                    "text": text,
                    "created_at": entry.get("c", ""),
                    "last_active": entry.get("a", ""),
                })

            # 5. Write archive document in past/ folder
            now = _now_iso()
            now_label = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
            archive_doc_id = f"geist-memory-archive-{now_label}"

            # Sort archived memories by number for readability
            archive_blocks.sort(key=lambda m: m["number"])

            archive_xml_parts = [
                f'<heading level="1">Memory Archive — {html_mod.escape(now_label)}</heading>',
                f"<paragraph>Archived {len(archive_blocks)} memories. "
                f"Kept {len(keep_blocks)} most recently active.</paragraph>",
            ]
            for mem in archive_blocks:
                escaped_text = html_mod.escape(mem["text"])
                archive_xml_parts.append(
                    f"<paragraph><strong>{mem['number']}.</strong> {escaped_text}</paragraph>"
                )

            archive_xml = "".join(archive_xml_parts)

            # Register archive doc in workspace
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            await hp_client.transact_workspace(
                graph_id,
                lambda doc, did=archive_doc_id: WorkspaceWriter(doc).upsert_document(
                    did, f"Memory Archive {now_label}", parent_id=PAST_FOLDER_ID,
                ),
                user_id=auth.user_id,
            )

            # Write archive content
            await hp_client.connect_document(graph_id, archive_doc_id, user_id=auth.user_id)
            await hp_client.transact_document(
                graph_id,
                archive_doc_id,
                lambda doc, xml=archive_xml: DocumentWriter(doc).replace_all_content(xml),
                user_id=auth.user_id,
            )

            # 6. Rewrite memory queue fresh with only kept memories
            # Sort kept memories by number to preserve original ordering
            keep_blocks.sort(key=lambda m: m["number"])

            # Build new metadata and XML
            new_memories = {}
            queue_xml_parts = [
                '<heading level="1">Memory Queue</heading>',
            ]

            # Temporary meta with empty memories — will be updated after rewrite
            # when we know new block IDs. This keeps the queue valid even if
            # the second transaction fails.
            temp_meta = json.dumps(
                {"next_number": next_number, "memories": {}},
                separators=(",", ":"),
            )
            queue_xml_parts.append(
                f"<paragraph>{GEIST_META_PREFIX}{html_mod.escape(temp_meta)}</paragraph>"
            )

            for mem in keep_blocks:
                escaped_text = html_mod.escape(mem["text"])
                queue_xml_parts.append(
                    f"<paragraph><strong>{mem['number']}.</strong> {escaped_text}</paragraph>"
                )

            queue_xml = "".join(queue_xml_parts)

            # Replace the entire queue document
            await hp_client.transact_document(
                graph_id,
                MEMORY_QUEUE_DOC_ID,
                lambda doc, xml=queue_xml: DocumentWriter(doc).replace_all_content(xml),
                user_id=auth.user_id,
            )

            # 7. Read back to get new block IDs, then update metadata
            await hp_client.connect_document(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
            channel = hp_client.get_document_channel(
                graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id,
            )
            new_reader = DocumentReader(channel.doc)
            block_count = new_reader.get_block_count()

            # Block 0 = heading, Block 1 = meta placeholder, Block 2+ = memories
            for i, mem in enumerate(keep_blocks):
                block_idx = i + 2  # Skip heading + meta
                if block_idx < block_count:
                    block = new_reader.get_block_at(block_idx)
                    if block and hasattr(block, "attributes"):
                        bid = block.attributes.get("data-block-id")
                        if bid:
                            new_memories[str(mem["number"])] = {
                                "b": bid,
                                "c": mem["created_at"],
                                "a": mem["last_active"],
                            }

            # Write the real metadata into the meta block
            new_meta = {
                "next_number": next_number,
                "memories": new_memories,
            }
            _, new_meta_block_id = _read_geist_meta(new_reader)
            if new_meta_block_id:
                await hp_client.transact_document(
                    graph_id,
                    MEMORY_QUEUE_DOC_ID,
                    lambda doc, m=new_meta, bid=new_meta_block_id: _write_geist_meta(
                        DocumentWriter(doc), m, bid,
                    ),
                    user_id=auth.user_id,
                )

        return _render_json({
            "kept": len(keep_blocks),
            "archived": len(archive_blocks),
            "archive_doc_id": archive_doc_id,
            "note": (
                f"Archived {len(archive_blocks)} memories to past/{archive_doc_id}. "
                f"Queue rewritten with {len(keep_blocks)} memories."
            ),
        })

    # ================================================================
    # ORIENTATION BUNDLE
    # ================================================================

    @server.tool(
        name="quick_orient",
        title="Quick Orientation",
        description=(
            "Lightweight orientation for emergency recovery or post-compaction triage. "
            "Returns location, Song, and recent memories in one call.\n\n"
            "This is NOT the default orientation flow. Agents should normally run the "
            "manual 3-step sequence: (1) get_user_location, (2) batch music + recall + "
            "get_important_blocks, (3) get_workspace. The manual sequence preserves the "
            "temporal structure where the Song lands as narrative identity before structural "
            "knowledge.\n\n"
            "Use quick_orient only when you need minimal context fast — e.g. mid-session "
            "recovery after an error, or when token budget is critically low."
        ),
    )
    async def quick_orient_tool(
        graph_id: str = "default",
        recall_limit: int = 5,
        context: Context | None = None,
    ) -> str:
        """Return a lightweight orientation bundle (location + song + recall).

        Args:
            graph_id: The graph to orient in (default: "default")
            recall_limit: Number of recent memories to include (default: 5)
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()
        graph_id = graph_id.strip()

        result: Dict[str, Any] = {}

        # --- Phase 1: Location (needed for everything else) ---
        try:
            uid = _resolve_user_id(auth)
            await hp_client.refresh_session(uid)
            result["location"] = {
                "graph_id": hp_client.get_active_graph_id() or graph_id,
                "document_id": hp_client.get_active_document_id(),
            }
        except Exception as e:
            result["location"] = {"graph_id": graph_id, "error": str(e)}

        # --- Phase 2: Run Song and Recall concurrently ---

        async def _get_song() -> Dict[str, Any]:
            try:
                await _ensure_scratchpad(hp_client, graph_id, auth)
                await hp_client.connect_document(graph_id, SONG_DOC_ID, user_id=auth.user_id)
                channel = hp_client.get_document_channel(graph_id, SONG_DOC_ID, user_id=auth.user_id)
                reader = DocumentReader(channel.doc)
                xml = reader.to_xml()
                markdown = tiptap_xml_to_markdown(xml)
                markdown = re.sub(
                    r"^" + re.escape(SONG_META_PREFIX) + r".*$", "", markdown, flags=re.MULTILINE
                )
                parts = re.split(r"\n---\n", markdown)
                verses = [v.strip() for v in parts if v.strip()]

                song_result: Dict[str, Any] = {"verses": verses, "verse_count": len(verses)}
                meta, _ = _read_song_meta(reader)
                if meta:
                    coda = meta.get("coda")
                    if coda:
                        song_result["coda"] = coda["text"]
                return song_result
            except Exception as e:
                return {"error": str(e)}

        async def _get_memories() -> Dict[str, Any]:
            try:
                await _ensure_scratchpad(hp_client, graph_id, auth)
                await hp_client.connect_document(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
                channel = hp_client.get_document_channel(graph_id, MEMORY_QUEUE_DOC_ID, user_id=auth.user_id)
                reader = DocumentReader(channel.doc)
                meta, _ = _read_geist_meta(reader)
                mem_entries = meta.get("memories", {})

                memories = []
                for num_str, entry in mem_entries.items():
                    if not isinstance(entry, dict) or not entry.get("b"):
                        continue
                    block_info = reader.get_block_info(entry["b"])
                    text = block_info["text_content"] if block_info else "(deleted)"
                    memories.append({
                        "number": int(num_str),
                        "text": text,
                        "created_at": entry.get("c"),
                        "last_active": entry.get("a"),
                    })

                def sort_key(m: dict) -> str:
                    return max(m.get("created_at", ""), m.get("last_active", ""))
                memories.sort(key=sort_key, reverse=True)
                memories = memories[:recall_limit]
                return {"memories": memories, "count": len(memories)}
            except Exception as e:
                return {"error": str(e)}

        # Run song and memories concurrently
        song, memories = await asyncio.gather(
            _get_song(),
            _get_memories(),
        )

        result["song"] = song
        result["recall"] = memories

        return _render_json(result)

    logger.info("Registered Geist (Sophia Memory) tools: 13 tools")


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


# ── Song metadata helpers ──────────────────────────────────────────


def _read_song_meta(reader: DocumentReader) -> tuple[dict, str | None]:
    """Read Song metadata from the meta block (index 1) in the Song document.

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
    if text.startswith(SONG_META_PREFIX):
        json_str = text[len(SONG_META_PREFIX):]
        try:
            return json.loads(json_str), block_id
        except json.JSONDecodeError:
            logger.warning("Corrupt song meta block", extra_context={"text": text[:100]})
    return {}, None


def _xml_to_verse_text(verse_xml: str) -> str:
    """Convert verse XML back to plain text with / stanza break markers.

    Used during migration from legacy (pre-metadata) Song format.
    """
    paragraphs = re.findall(r"<paragraph[^>]*>(.*?)</paragraph>", verse_xml, re.DOTALL)
    lines: list[str] = []
    for i, p in enumerate(paragraphs):
        text = re.sub(r"<[^>]+>", "", p).strip()
        if not text:
            # Empty paragraph = stanza break — mark previous line with /
            if lines and not lines[-1].endswith("/"):
                lines[-1] = lines[-1] + " /"
        else:
            lines.append(text)
    return "\n".join(lines)


def _migrate_song_to_meta(xml: str) -> dict:
    """Create Song metadata from a legacy (pre-metadata) Song document."""
    sections = _split_song_xml(xml)

    verses = []
    for s in sections:
        text = re.sub(r"<[^>]+>", "", s).strip()
        if not text or text == "[awaiting composition]":
            continue
        if text.startswith(SONG_META_PREFIX):
            continue  # Skip stray meta blocks during migration
        verses.append({
            "text": _xml_to_verse_text(s),
            "counterpoints": [],
        })

    return {
        "version": 1,
        "verses": verses,
        "coda": None,
    }


def _parse_verse_lines(verse_text: str) -> list[tuple[str, bool]]:
    """Parse verse text into (line, has_stanza_break) tuples.

    Lines ending with / have has_stanza_break=True.
    Empty lines are skipped.
    """
    lines = verse_text.strip().split("\n")
    result: list[tuple[str, bool]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.endswith("/"):
            line = line[:-1].rstrip()
            result.append((line, True))
        else:
            result.append((line, False))
    return result


def _interleave_verse_xml(verse_text: str, counterpoints: list[str]) -> str:
    """Render a verse with interleaved counterpoint voices as TipTap XML.

    Voice 0 (original): plain paragraphs
    Voice 1 (first counterpoint): italic paragraphs
    Voice 2 (second counterpoint): bold italic paragraphs
    """
    if not counterpoints:
        return _verse_to_xml(verse_text)

    original_lines = _parse_verse_lines(verse_text)
    cp_lines_list = [_parse_verse_lines(cp) for cp in counterpoints]

    max_lines = max(
        len(original_lines),
        *(len(cp) for cp in cp_lines_list),
    )

    parts: list[str] = []
    for i in range(max_lines):
        # Original voice
        if i < len(original_lines):
            line, has_break = original_lines[i]
            if line:
                parts.append(f"<paragraph>{html_mod.escape(line)}</paragraph>")
            if has_break:
                parts.append("<paragraph></paragraph>")

        # Counterpoint voices
        for v, cp_lines in enumerate(cp_lines_list):
            if i < len(cp_lines):
                line, has_break = cp_lines[i]
                if line:
                    escaped = html_mod.escape(line)
                    if v == 0:
                        parts.append(f"<paragraph><em>{escaped}</em></paragraph>")
                    else:
                        parts.append(
                            f"<paragraph><strong><em>{escaped}</em></strong></paragraph>"
                        )
                if has_break:
                    parts.append("<paragraph></paragraph>")

    return "".join(parts)


def _render_song_from_meta(meta: dict) -> str:
    """Build the full Song TipTap XML from metadata.

    Structure: title → meta block → verses (interleaved) → coda (if present).
    """
    parts = ['<heading level="1">The Song</heading>']

    # Meta block at position 1 (hidden in rendered output)
    meta_json = json.dumps(meta, separators=(",", ":"), default=str)
    parts.append(f"<paragraph>{SONG_META_PREFIX}{html_mod.escape(meta_json)}</paragraph>")

    verses = meta.get("verses", [])
    for i, verse_data in enumerate(verses):
        parts.append(f'<heading level="2">Verse {_verse_label(i)}</heading>')
        parts.append(_interleave_verse_xml(
            verse_data.get("text", ""),
            verse_data.get("counterpoints", []),
        ))
        if i < len(verses) - 1:
            parts.append("<horizontalRule/>")

    # Coda after the last verse
    coda = meta.get("coda")
    if coda and coda.get("text"):
        parts.append("<horizontalRule/>")
        remaining = coda.get("ejections_remaining", 0)
        parts.append(f'<heading level="3">Coda ({remaining} remaining)</heading>')
        coda_lines = coda["text"].strip().split("\n")
        for line in coda_lines:
            line = line.strip()
            if line:
                parts.append(f"<paragraph><em>{html_mod.escape(line)}</em></paragraph>")

    return "".join(parts)
