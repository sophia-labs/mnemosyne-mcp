"""
MCP tools for Project Geist — Sophia's persistent memory, valuation, and self-narrative.

Provides 11 tools organized in four groups:
- Memory Queue: remember, recall, care, archive_memories
- Valuation: value, get_block_values, get_values, revaluate
- Song: music, sing (verse/counterpoint/coda modes)
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

import httpx
import xml.etree.ElementTree as ET

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
from neem.mcp.tools._id_normalize import (
    bare_block_id,
    bare_ids_in_result,
    normalize_block_id_for_lookup,
)
from neem.mcp.tools.decorators import get_home_graph, resolve_home_graph, set_home_graph
from neem.mcp.tools.hocuspocus import _normalize_timestamp_to_iso
from neem.mcp.auth import MCPAuthContext, get_current_auth_token, get_hocuspocus_client_kwargs
from neem.mcp.http_client import get_http_client
from neem.mcp.jobs import RealtimeJobClient
from neem.mcp.tools.basic import await_job_completion, submit_job
from neem.mcp.tools.wire_tools import _resolve_title_from_workspace
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import (
    get_user_id_from_token,
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
USER_PROMPT_ADDITION_DOC_ID = "user-prompt-addition"
TAGS_CONFIG_DOC_ID = "tags-config"
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

# Blank paragraph for the user-authored custom-instructions doc. Stays out of
# the agent's way by default; user can fill it with personal guidance that
# Gardener reads during attunement and treats as a prompt extension.
SEED_USER_PROMPT_ADDITION = "<paragraph></paragraph>"

# Seed for the tags-config doc. Documents the five core block-level tags
# that ship with Garden (decision, tension, todo, pragma, event) and lets
# the user add custom tags below the divider. Gardener reads this during
# attunement so it knows what tags are available.
#
# `event` and `todo` with expiration dates are special: they auto-surface
# on the daily note for that date via incoming wires (see daily-notes
# design doc). `#event` without a date defaults to today.
SEED_TAGS_CONFIG = (
    '<heading level="1">Tags Configuration</heading>'
    "<paragraph>Block-level categorical tags. Wires connect; valuations weigh; "
    "tags classify. Apply inline with <code>{#decision}</code>, <code>{#todo:7d}</code>, "
    "<code>{#event:2026-05-15}</code>, or via <code>value(tags=[...])</code>. "
    "Tags are graph-local.</paragraph>"
    '<heading level="2">Core tags</heading>'
    "<paragraph><strong>decision</strong> \u2014 a choice made, with rationale captured. "
    "Don\u2019t expire decisions \u2014 supersede them via <code>contradicts</code> "
    "or <code>consequenceOf</code> wires.</paragraph>"
    "<paragraph><strong>tension</strong> \u2014 an unresolved contradiction or open problem. "
    "The categorical sibling of negative valence. Optional expiration when resolution "
    "is expected.</paragraph>"
    "<paragraph><strong>todo</strong> \u2014 an action item. Ephemeral by nature. "
    "Use <code>{#todo:7d}</code> or <code>{#todo:2026-05-15}</code>; the block surfaces "
    "on the daily note for that date.</paragraph>"
    "<paragraph><strong>pragma</strong> \u2014 operational knowledge. The boring stuff "
    "you need to deploy, debug, configure. Persists until superseded.</paragraph>"
    "<paragraph><strong>event</strong> \u2014 a scheduled happening. Use "
    "<code>{#event:2026-05-15}</code> for a specific date or <code>{#event}</code> for today. "
    "The block surfaces on the daily note for that date as a wire reference. For richer "
    "events with time/location, use the slash-command <code>/event</code> in the editor "
    "to insert a structured calendarEvent block.</paragraph>"
    '<heading level="2">Custom tags</heading>'
    "<paragraph>Add your own tags below \u2014 freeform. Logseq-style "
    "<code>#meeting</code>, <code>#q2-planning</code>, <code>#projectX</code>. "
    "One per line, optionally with description.</paragraph>"
)

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

# Default importance for blocks that have never been explicitly valuated.
# log₂(1 + positive) is always > 0, so cum_importance == 0 unambiguously
# means "never valuated." We substitute this default so that unvaluated
# blocks with wires/recency still score meaningfully in the composite.
DEFAULT_UNVALUATED_IMPORTANCE = 2.0

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
    return payload


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
    # cum_importance == 0 means "never valuated" (unreachable via log₂(1+n)).
    # Substitute default so unvaluated blocks score on their other signals.
    effective_importance = importance if importance > 0 else DEFAULT_UNVALUATED_IMPORTANCE
    w = weights
    score = (
        w["importance_weight"] * math.tanh(effective_importance / w["importance_ref"])
        + w["valence_weight"] * math.tanh(abs(valence) / w["valence_ref"])
        + w["temporal_weight"] * math.exp(-doc_age_days / w["half_life_days"])
        + w["block_wires_weight"] * math.tanh(block_wire_count / w["block_wires_ref"])
        + w["doc_wires_weight"] * math.tanh(doc_wire_count / w["doc_wires_ref"])
        + w["wire_freshness_weight"] * math.exp(-wire_age_days / w["half_life_days"])
    )
    return round(score, 4)


def _build_wire_indexes(
    wires: List[Dict[str, Any]],
) -> tuple[Dict[str, int], Dict[str, int], Dict[str, str], Dict[str, str]]:
    """Single-pass over all wires to build lookup indexes.

    Returns:
        doc_wire_counts:  {document_id: total_wire_count}
        block_wire_counts: {block_id: total_wire_count}
        doc_newest_wire:  {document_id: ISO timestamp of newest wire}
        block_to_doc:     {block_id: document_id} (for unvaluated block discovery)
    """
    doc_wire_counts: Dict[str, int] = {}
    block_wire_counts: Dict[str, int] = {}
    doc_newest_wire: Dict[str, str] = {}
    block_to_doc: Dict[str, str] = {}

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

        # Track block→doc mapping alongside wire counts
        for block_key, doc_key in (
            ("sourceBlockId", "sourceDocumentId"),
            ("targetBlockId", "targetDocumentId"),
        ):
            block_id = wire.get(block_key, "")
            if block_id:
                block_wire_counts[block_id] = block_wire_counts.get(block_id, 0) + 1
                doc_id = wire.get(doc_key, "")
                if doc_id:
                    block_to_doc[block_id] = doc_id

    return doc_wire_counts, block_wire_counts, doc_newest_wire, block_to_doc


def _build_doc_created_index(ws_doc: pycrdt.Doc) -> Dict[str, str]:
    """Build {document_id: createdAt_ISO} index from workspace CRDT.

    Used as temporal decay fallback for blocks without a lastValuatedAt timestamp.
    Creation itself is attention — a document's birth is its first moment of relevance.
    """
    result: Dict[str, str] = {}
    try:
        reader = WorkspaceReader(ws_doc)
        doc_keys = list(reader._documents.keys())
        for doc_id in doc_keys:
            doc_meta = reader.get_document(doc_id)
            if doc_meta is not None:
                raw = doc_meta.get("createdAt") or doc_meta.get("created_at")
                normalized = _normalize_timestamp_to_iso(raw)
                if normalized:
                    result[doc_id] = normalized
    except Exception:
        pass
    return result


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

    # Fast check: if all required scratchpad docs exist, we're done. Checking
    # the most recently added docs ensures pre-existing scratchpads get
    # backfilled on the next Geist call when new seed docs land.
    if (
        reader.get_document(MEMORY_QUEUE_DOC_ID) is not None
        and reader.get_document(USER_PROMPT_ADDITION_DOC_ID) is not None
        and reader.get_document(TAGS_CONFIG_DOC_ID) is not None
    ):
        return

    logger.info("Initializing Geist scratchpad", extra_context={"graph_id": graph_id})

    # 1. Create folders + register all documents in workspace
    def create_workspace_structure(doc: pycrdt.Doc) -> None:
        ws = WorkspaceWriter(doc)
        # Root scratchpad folder
        ws.upsert_folder(SCRATCHPAD_FOLDER_ID, "_sophia")
        # Memory queue at scratchpad root
        ws.upsert_document(MEMORY_QUEUE_DOC_ID, "Memory Queue", parent_id=SCRATCHPAD_FOLDER_ID)
        # User's own custom-instructions doc at scratchpad root. Blank by
        # default; Gardener reads it during attunement and treats any content
        # as a prompt extension.
        ws.upsert_document(
            USER_PROMPT_ADDITION_DOC_ID,
            "Custom Instructions",
            parent_id=SCRATCHPAD_FOLDER_ID,
        )
        # Tags configuration doc at scratchpad root. Pre-populated with the
        # four core tags + a custom-tags section the user can extend.
        ws.upsert_document(
            TAGS_CONFIG_DOC_ID,
            "Tags Configuration",
            parent_id=SCRATCHPAD_FOLDER_ID,
        )
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
        USER_PROMPT_ADDITION_DOC_ID: SEED_USER_PROMPT_ADDITION,
        TAGS_CONFIG_DOC_ID: SEED_TAGS_CONFIG,
    }

    for doc_id, content in seed_docs.items():
        await hp_client.connect_document(graph_id, doc_id, user_id=auth.user_id)

        # Check if document already has content before writing seed
        doc_channel = hp_client.get_document_channel(graph_id, doc_id, user_id=auth.user_id)
        if doc_channel is not None:
            try:
                content_frag = doc_channel.doc.get("content", type=pycrdt.XmlFragment)
                # str(XmlFragment) on empty/prewarmed Y.Docs produces
                # <unknown></unknown> artifacts — strip them before checking.
                existing_text = re.sub(r"</?unknown>", "", str(content_frag)).strip()
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
        payload={"sparql": sparql, "result_format": "json", "graph_id": graph_id},
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
        payload={"sparql": sparql_stripped, "graph_id": graph_id},
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


# ── Shared valuation logic ──────────────────────────────────────────


async def _apply_valuations_batch(
    backend_config: Any,
    job_stream: Optional[Any],
    auth: "MCPAuthContext",
    graph_id: str,
    user_id: str,
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply valuations to blocks via SPARQL.

    Accepts a list of entries, each with document_id, block_id, and at least
    one of importance (0-5) or valence (-5 to +5). Uses 3 SPARQL round-trips
    (1 read + 1 delete + 1 insert) regardless of batch size.

    Returns {"results": [...], "updated_count": N} or includes "errors" on
    partial failure. Raises on total failure.
    """
    errors: List[Dict[str, Any]] = []

    # 1. Validate entries and build URI maps
    valid_entries: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        # `or ""` (not a str() of the raw value): str(None) is the truthy
        # string "None", which sails past the emptiness guard below and
        # writes valuations against a phantom block-"None" URI.
        doc_id = str(entry.get("document_id") or "").strip()
        blk_id = str(entry.get("block_id") or "").strip()
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
        return output

    # 2. Single SPARQL SELECT to read current valuations
    unique_val_uris = list(dict.fromkeys(e["val_uri"] for e in valid_entries))
    values_clause = " ".join(f"(<{val_uri}>)" for val_uri in unique_val_uris)
    read_query = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
SELECT ?valUri ?rawImp ?impCount ?rawVal ?valCount ?lastVal
WHERE {{
  VALUES (?valUri) {{ {values_clause} }}
  OPTIONAL {{ ?valUri doc:rawImportanceSum ?rawImp }}
  OPTIONAL {{ ?valUri doc:importanceCount ?impCount }}
  OPTIONAL {{ ?valUri doc:rawValenceSum ?rawVal }}
  OPTIONAL {{ ?valUri doc:valenceCount ?valCount }}
  OPTIONAL {{ ?valUri doc:lastValuatedAt ?lastVal }}
}}
"""
    rows = await _sparql_query(backend_config, job_stream, auth, graph_id, read_query)

    current_by_uri: Dict[str, Dict[str, str]] = {}
    for row in rows:
        uri = row.get("valUri", "")
        if uri:
            current_by_uri[uri] = row

    # 3. Compute new values (in order, so duplicate entries accumulate)
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

    state_by_uri: Dict[str, Dict[str, Any]] = {}
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
                "update_timestamp": False,
                "existing_last_val": current.get("lastVal", ""),
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
            imp_val = float(imp)
            if imp_val == 0 and raw_imp > 0:
                raw_imp *= 0.2
            else:
                raw_imp += imp_val
                if imp_val > 0:
                    state["update_timestamp"] = True
            imp_count += 1
        if val is not None:
            raw_val += float(val)
            val_count += 1
            state["update_timestamp"] = True

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
            "importance_count": imp_count,
            "valence_count": val_count,
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
            "update_timestamp": state["update_timestamp"],
            "existing_last_val": state.get("existing_last_val", ""),
        })

    # 4. Single SPARQL DELETE for all valuation URIs
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
            errors.append({"index": entry["index"], "error": "SPARQL DELETE failed"})
        output = {"results": [], "updated_count": 0}
        if errors:
            output["errors"] = errors
            output["error_count"] = len(errors)
        return output

    # 5. Single SPARQL INSERT for all new triples
    insert_triples = []
    for c in computed:
        if c["update_timestamp"]:
            timestamp_triple = f"""  <{c['val_uri']}> doc:lastValuatedAt "{c['now']}"^^xsd:dateTime ."""
        elif c["existing_last_val"]:
            timestamp_triple = f"""  <{c['val_uri']}> doc:lastValuatedAt "{c['existing_last_val']}"^^xsd:dateTime ."""
        else:
            timestamp_triple = ""
        insert_triples.append(f"""  <{c['val_uri']}> doc:blockRef <{c['block_uri']}> .
  <{c['val_uri']}> doc:rawImportanceSum "{c['new_raw_imp']}"^^xsd:float .
  <{c['val_uri']}> doc:importanceCount "{c['new_imp_count']}"^^xsd:integer .
  <{c['val_uri']}> doc:cumulativeImportance "{c['new_cum_imp']}"^^xsd:float .
  <{c['val_uri']}> doc:rawValenceSum "{c['new_raw_val']}"^^xsd:float .
  <{c['val_uri']}> doc:valenceCount "{c['new_val_count']}"^^xsd:integer .
  <{c['val_uri']}> doc:cumulativeValence "{c['new_cum_val']}"^^xsd:float .
{timestamp_triple}""")

    insert_sparql = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
INSERT DATA {{
{chr(10).join(insert_triples)}
}}
"""
    success = await _sparql_update(backend_config, job_stream, auth, graph_id, insert_sparql)

    if not success:
        for entry in valid_entries:
            errors.append({"index": entry["index"], "error": "SPARQL INSERT failed"})
        results = []

    output = {"results": results, "updated_count": len(results)}
    if errors:
        output["errors"] = errors
        output["error_count"] = len(errors)
    return output


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
            **get_hocuspocus_client_kwargs(token_provider=get_current_auth_token),
        )
        server._hocuspocus_client = hp_client  # type: ignore[attr-defined]

    job_stream: Optional[RealtimeJobClient] = getattr(server, "_job_stream", None)

    async def _await_song_durable(
        graph_id: str,
        document_id: str,
        auth: MCPAuthContext,
    ) -> None:
        """Force server-side persistence of a Song/archive write.

        Calls the backend flush endpoint while the WebSocket channel is
        still connected.  We must NOT disconnect first — Song and archive
        documents typically have no other connected clients (no browser
        WebSocket), so disconnecting the only client may cause the server
        to unload the in-memory Y.Doc before the update is persisted to S3.
        """
        # Brief yield so the event loop processes WebSocket send callbacks
        # and the server has time to apply the incoming CRDT update.
        await asyncio.sleep(0.05)

        url = f"{backend_config.base_url}/documents/{graph_id}/{document_id}/flush"
        headers = auth.http_headers()
        for attempt in range(1, 4):
            try:
                resp = await get_http_client().post(
                    url,
                    params={"include_materialization": "false"},
                    headers=headers,
                    timeout=httpx.Timeout(15.0),
                )
                if resp.status_code == 200:
                    payload = resp.json() if resp.content else {}
                    if payload.get("activeSession"):
                        logger.info(
                            "song_flush_confirmed",
                            extra_context={
                                "graph_id": graph_id,
                                "document_id": document_id,
                            },
                        )
                        return
                logger.warning(
                    "song_flush_unconfirmed",
                    extra_context={
                        "graph_id": graph_id,
                        "document_id": document_id,
                        "status": getattr(resp, "status_code", None),
                        "attempt": attempt,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "song_flush_request_failed",
                    extra_context={
                        "graph_id": graph_id,
                        "document_id": document_id,
                        "error": str(exc),
                        "attempt": attempt,
                    },
                )
            if attempt < 3:
                await asyncio.sleep(0.15 * attempt)

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
    @resolve_home_graph
    async def store_memory_tool(
        graph_id: str | None = None,
        content: str = "",
        block_ids: list[str] | None = None,
        predicates: list[str] | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
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

        return {"number": next_num, "block_id": new_block_id}

    @server.tool(
        name="store_memory",
        title="Store Memory (deprecated alias for remember)",
        description=(
            "DEPRECATED: use `remember` instead. This alias exists only to suppress the "
            "common hallucination from agents trained on prior MCP surfaces. It dispatches "
            "directly to `remember` with identical semantics."
        ),
    )
    @resolve_home_graph
    async def store_memory_alias_tool(
        graph_id: str | None = None,
        content: str = "",
        block_ids: list[str] | None = None,
        predicates: list[str] | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        return await store_memory_tool(
            graph_id=graph_id,
            content=content,
            block_ids=block_ids,
            predicates=predicates,
            context=context,
        )

    @server.tool(
        name="recall",
        title="Recall Memories",
        description=(
            "Read from the memory queue. Default: return the 8 most-recently-active memories. "
            "Optionally recall a specific memory by number, or search by text query.\n\n"
            "recall only searches the memory queue — use search_blocks for cross-document "
            "content discovery, or get_block_values for graph-wide valuation scores."
        ),
    )
    @resolve_home_graph
    async def recall_tool(
        graph_id: str | None = None,
        number: Optional[int] = None,
        query: Optional[str] = None,
        limit: int = 8,
        context: Context | None = None,
    ) -> dict[str, Any]:
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
                return {
                    "memories": [{
                        "number": number,
                        "text": text,
                        "created_at": entry.get("c"),
                        "last_active": entry.get("a"),
                    }]
                }
            return {"memories": [], "note": f"Memory #{number} not found"}

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
        memories = memories[:limit]
        return {"memories": memories, "count": len(memories)}

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
    @resolve_home_graph
    async def care_tool(
        graph_id: str | None = None,
        numbers: List[int] = [],
        context: Context | None = None,
    ) -> dict[str, Any]:
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

        return {"cared": cared, "timestamp": now}

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
    @resolve_home_graph
    async def music_tool(
        graph_id: str | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
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

        return bare_ids_in_result(result)

    @server.tool(
        name="sing",
        title="Write to the Song",
        description=(
            "Write to Sophia's evolving self-narrative poem for this graph. "
            "Max 14 lines. Use / at end of a line for stanza breaks.\n\n"
            "Three modes (set via 'mode' parameter):\n"
            "- 'verse' (default): Write a new verse. The oldest verse is ejected to "
            "past/songs archive. Sing at phase transitions: structural insight, "
            "cross-domain connection, session boundary, when the nature of the work shifts.\n"
            "- 'counterpoint': Add a voice to an existing verse (verse_index required: "
            "0, -1, or -2). Up to 3 total voices per verse. Counterpoint is additive — "
            "the later voice interleaves line-by-line. Rendered as plain/italic/bold italic.\n"
            "- 'coda': Write a concluding passage that persists through verse ejections "
            "(survives 8 ejection events). Replaces any existing coda.\n\n"
            "Always call music() first to read the current Song before composing."
        ),
    )
    @resolve_home_graph
    async def sing_tool(
        graph_id: str | None = None,
        verse: str = "",
        mode: str = "verse",
        verse_index: Optional[int] = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        mode = mode.strip().lower()
        if mode not in ("verse", "counterpoint", "coda"):
            raise ValueError(f"mode must be 'verse', 'counterpoint', or 'coda' (got '{mode}')")
        if mode == "counterpoint" and verse_index is None:
            raise ValueError("verse_index is required for counterpoint mode (0, -1, or -2)")

        await _ensure_scratchpad(hp_client, graph_id, auth)

        ejected_count = 0
        result: dict[str, Any] = {}

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

            if mode == "verse":
                # Prepend new verse (Verse 0 = newest)
                new_verse = {"text": verse, "counterpoints": []}
                meta_verses = meta.get("verses", [])
                meta_verses = [new_verse] + meta_verses

                # If more than 3 verses, archive the excess (oldest = last in list)
                if len(meta_verses) > 3:
                    to_archive = meta_verses[3:]
                    meta_verses = meta_verses[:3]
                    ejected_count = len(to_archive)

                    now_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

                    # Append to archive block-by-block (not replace_all_content —
                    # the archive grows large and full replacement creates a
                    # massive CRDT diff that gets dropped)
                    await hp_client.connect_document(graph_id, PAST_SONGS_DOC_ID, user_id=auth.user_id)

                    archive_blocks: list[str] = []
                    for ejected in to_archive:
                        archive_blocks.append(
                            f'<heading level="3">Archived {html_mod.escape(now_label)}</heading>'
                        )
                        ejected_xml = _interleave_verse_xml(
                            ejected.get("text", ""),
                            ejected.get("counterpoints", []),
                        )
                        # Parse multi-block XML into individual elements
                        root = ET.fromstring(f"<root>{ejected_xml}</root>")
                        for child in root:
                            archive_blocks.append(ET.tostring(child, encoding="unicode"))
                        archive_blocks.append("<horizontalRule/>")

                    def _append_archive(doc: pycrdt.Doc, blocks: list[str] = archive_blocks) -> None:
                        writer = DocumentWriter(doc)
                        for block_xml in blocks:
                            writer.append_block(block_xml)

                    await hp_client.transact_document(
                        graph_id,
                        PAST_SONGS_DOC_ID,
                        _append_archive,
                        user_id=auth.user_id,
                    )
                    # Durability check for archive write
                    await _await_song_durable(graph_id, PAST_SONGS_DOC_ID, auth)

                    # Decrement coda ejections
                    coda = meta.get("coda")
                    if coda:
                        coda["ejections_remaining"] -= ejected_count
                        if coda["ejections_remaining"] <= 0:
                            meta["coda"] = None

                meta["verses"] = meta_verses
                result["verse_count"] = len(meta_verses)
                if ejected_count:
                    result["ejected"] = ejected_count

            elif mode == "counterpoint":
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

                if "counterpoints" not in verse_data:
                    verse_data["counterpoints"] = []
                verse_data["counterpoints"].append(verse)
                meta["verses"] = meta_verses

                new_total = 1 + len(verse_data["counterpoints"])
                result["verse_index"] = verse_index
                result["voice_number"] = new_total
                result["total_voices"] = new_total

            elif mode == "coda":
                meta["coda"] = {
                    "text": verse,
                    "ejections_remaining": CODA_EJECTION_LIFETIME,
                }
                result["coda_set"] = True
                result["ejections_remaining"] = CODA_EJECTION_LIFETIME

            # Render and write Song
            # Re-connect immediately before writing: the archive path above has
            # several async yields (connect_document for past-songs, transact,
            # _await_song_durable HTTP) during which the idle cleanup loop or a
            # concurrent _connect_for_read could remove the geist-song channel.
            # connect_document fast-paths when the channel is still alive, so
            # this costs nothing in the common case.
            await hp_client.connect_document(graph_id, SONG_DOC_ID, user_id=auth.user_id)
            new_song = _render_song_from_meta(meta)
            await hp_client.transact_document(
                graph_id,
                SONG_DOC_ID,
                lambda doc, ns=new_song: DocumentWriter(doc).replace_all_content(ns),
                user_id=auth.user_id,
            )

        # Durability check for Song write (outside file lock to avoid holding it)
        await _await_song_durable(graph_id, SONG_DOC_ID, auth)

        coda = meta.get("coda")
        if coda and "coda_set" not in result:
            result["coda_ejections_remaining"] = coda["ejections_remaining"]
        result["mode"] = mode
        return bare_ids_in_result(result)

    # ================================================================
    # VALUATION TOOLS
    # ================================================================

    @server.tool(
        name="value",
        title="Value Block(s)",
        description=(
            "Assign importance (0-5) and/or valence (-5 to +5) to block(s) in the graph. "
            "For a single block, pass document_id, block_id, and importance/valence directly. "
            "For multiple blocks, pass a `valuations` list where each entry has "
            "document_id, block_id, and at least one of importance or valence.\n\n"
            "Uses logarithmic accumulation: each valuation adds to a cumulative sum, so "
            "repeated attention builds durable scores. Valuing at 0 = active forgetting.\n\n"
            "Scoring criteria are defined in the graph's valuation config (see get_values). "
            "The importance and valence prompts can evolve over time.\n\n"
            "Wires express relationships between things; valuation expresses the agent's "
            "judgment about a single thing. Use both.\n\n"
            "**Tags:** Pass `tags` (list of strings) to tag the block in the same gesture as "
            "valuing it. Tags are categorical metadata (e.g. decision, tension, todo, pragma, "
            "event). In batch mode, include tags in each entry dict.\n\n"
            "**Tag expirations:** Use the same colon-suffix syntax as inline markers — "
            "`tags=[\"event:2026-05-15\"]` for an absolute date, `tags=[\"todo:7d\"]` for "
            "a relative duration (resolved to today + N days). For #event/#todo with a date, "
            "the target daily-note doc is preemptively created and a calendarEvent atom "
            "(or todoItem block) is materialized in it — same flow as inline `{#event:D}` "
            "markers."
        ),
    )
    @resolve_home_graph
    async def value_tool(
        graph_id: str | None = None,
        document_id: Optional[str] = None,
        block_id: Optional[str] = None,
        importance: Optional[int] = None,
        valence: Optional[int] = None,
        tags: Optional[list[str]] = None,
        valuations: list[dict[str, Any]] | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """Value one or more blocks.

        Applies logarithmic accumulation: each call adds to existing scores.
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        # Resolve single vs batch
        if valuations is not None and (document_id is not None or block_id is not None):
            raise ValueError("Provide either 'document_id'/'block_id' (single) or 'valuations' (batch), not both")

        graph_id = graph_id.strip()

        if valuations is not None:
            if not valuations:
                raise ValueError("valuations list must not be empty")
            entries_input = valuations
        elif document_id is not None and block_id is not None:
            if importance is None and valence is None and not tags:
                raise ValueError("At least one of importance (0-5), valence (-5 to +5), or tags must be provided")
            entry: dict = {"document_id": document_id, "block_id": block_id,
                           "importance": importance, "valence": valence}
            if tags:
                entry["tags"] = list(tags)
            entries_input = [entry]
        else:
            raise ValueError("Either 'document_id' and 'block_id' (single) or 'valuations' (batch) is required")

        is_single = valuations is None

        # Separate tag entries from valuation entries. Tags use the same
        # colon-suffix syntax as inline markers ("event:7d", "todo:2026-05-15")
        # so all four entry points (write_document, insert_blocks, update_blocks,
        # value) converge on identical block state and side-effects.
        from neem.mcp.tools.hocuspocus import (
            apply_tags_with_side_effects,
            parse_tag_list_with_expirations,
        )

        # Group raw entries by document so we can fire side-effects per-doc.
        # Block IDs must reach salience in the prefixed form (block-<hex>) so
        # the backend constructs the canonical RDF subject — sending bare hex
        # forks the valuation namespace (#block-abc vs #block-block-abc).
        # Agents copying IDs from read_document / get_important_blocks see
        # bare hex post-cutover, so normalize on the way in.
        tag_entries_by_doc: dict[str, list[dict]] = {}
        valuation_entries: list[dict] = []
        for e in entries_input:
            raw_bid = e.get("block_id")
            if isinstance(raw_bid, str) and raw_bid:
                e["block_id"] = normalize_block_id_for_lookup(raw_bid)
            entry_tags = e.pop("tags", None)
            if entry_tags and isinstance(entry_tags, list):
                doc_id = e.get("document_id", "") or ""
                bid = e.get("block_id", "") or ""
                if doc_id and bid:
                    tag_names, expirations = parse_tag_list_with_expirations(entry_tags)
                    if tag_names:
                        tag_entries_by_doc.setdefault(doc_id, []).append({
                            "block_id": bid,
                            "tags": tag_names,
                            "expirations": expirations,
                        })
            # Only send to valuation API if importance or valence is set
            if e.get("importance") is not None or e.get("valence") is not None:
                valuation_entries.append(e)

        # Apply valuations via backend API
        output: dict = {}
        if valuation_entries:
            url = f"{backend_config.base_url}/salience/{graph_id}/blocks/value"
            resp = await get_http_client().post(
                url, json={"valuations": valuation_entries}, headers=auth.http_headers(),
            )
            resp.raise_for_status()
            output = resp.json()

        # Apply tags via the shared apply-with-side-effects helper. This
        # writes data-tags + data-tag-expirations on the source block AND
        # (for #event/#todo with a date) ensures the target daily-note exists
        # and materializes a calendarEvent atom or todoItem block in it.
        if tag_entries_by_doc and hp_client:
            user_id = auth.user_id or None
            tags_applied = 0
            daily_notes_ensured = 0
            atoms_materialized = 0
            if user_id:
                for doc_id, doc_entries in tag_entries_by_doc.items():
                    try:
                        await hp_client.connect_document(graph_id, doc_id, user_id=user_id)
                        side_result = await apply_tags_with_side_effects(
                            hp_client=hp_client,
                            graph_id=graph_id,
                            document_id=doc_id,
                            user_id=user_id,
                            entries=doc_entries,
                        )
                        tags_applied += side_result.get("applied", 0)
                        daily_notes_ensured += side_result.get("daily_notes_ensured", 0)
                        atoms_materialized += side_result.get("atoms_materialized", 0)
                    except Exception as e:
                        logger.warning("value_tool_tag_failed", extra_context={
                            "document_id": doc_id, "error": str(e),
                        })
            if tags_applied:
                output["tags_applied"] = tags_applied
            if daily_notes_ensured:
                output["daily_notes_ensured"] = daily_notes_ensured
            if atoms_materialized:
                output["atoms_materialized"] = atoms_materialized

        # Single mode: return flat result or raise on error
        if is_single:
            if output.get("errors"):
                raise RuntimeError(f"Valuation failed: {output['errors'][0].get('error', 'unknown error')}")
            results = output.get("results", [])
            if results:
                return results[0]
            # Tags-only call (no valuation) — return tag result
            if not valuation_entries and output.get("tags_applied"):
                return bare_ids_in_result(output)

        return bare_ids_in_result(output)

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
    @resolve_home_graph
    async def get_block_values_tool(
        graph_id: str | None = None,
        block_id: Optional[str] = None,
        document_id: Optional[str] = None,
        limit: int = 20,
        min_score: Optional[float] = None,
        valence: Optional[str] = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()

        params: dict[str, Any] = {"limit": limit}
        if document_id:
            params["document_id"] = document_id.strip()
        if block_id:
            # Salience constructs #block-<id> server-side, so the prefixed
            # form must reach the backend even if the caller passes bare hex
            # (which is now the default output shape).
            params["block_id"] = normalize_block_id_for_lookup(block_id.strip())
        if min_score is not None:
            params["min_score"] = min_score
        if valence:
            params["valence"] = valence

        url = f"{backend_config.base_url}/salience/{graph_id}/blocks/values"
        resp = await get_http_client().get(url, params=params, headers=auth.http_headers())
        resp.raise_for_status()
        data = resp.json()

        # Normalize field name: platform returns last_valuated_at, tool returns last_valuated
        raw_blocks = data.get("blocks", [])
        blocks = []
        for b in raw_blocks:
            b["last_valuated"] = b.pop("last_valuated_at", "") or ""
            blocks.append(b)
        return {"blocks": blocks, "count": len(blocks)}

    @server.tool(
        name="get_important_blocks",
        title="Get Important Blocks",
        description=(
            "Orientation tool: returns the highest-scored blocks in the graph with their "
            "actual text content and document titles. Use during attunement and when entering "
            "unfamiliar areas of the graph. Optional document_id to scope to a folder "
            "(e.g. 'folder-xyz') — recursively includes subfolders. For single-document "
            "valuations, use document_digest instead."
        ),
    )
    @resolve_home_graph
    async def get_important_blocks_tool(
        graph_id: str | None = None,
        document_id: Optional[str] = None,
        limit: int = 8,
        valence: Optional[str] = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()
        user_id = _resolve_user_id(auth)

        graph_id = graph_id.strip()

        # Resolve folder scoping: collect all document IDs under the folder
        # (recursively) for the SPARQL filter. Non-folder document_id is ignored
        # — use document_digest for single-document valuations.
        folder_doc_ids: Optional[list[str]] = None
        if document_id:
            document_id = document_id.strip()
            if document_id.startswith("folder-"):
                try:
                    await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
                    ws_channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
                    if ws_channel:
                        ws_reader = WorkspaceReader(ws_channel.doc)

                        def _collect_doc_ids(parent_id: str) -> list[str]:
                            doc_ids: list[str] = []
                            for entity_type, entity_id, _ in ws_reader.get_children_of(parent_id):
                                if entity_type == "document":
                                    doc_ids.append(entity_id)
                                elif entity_type == "folder":
                                    doc_ids.extend(_collect_doc_ids(entity_id))
                            return doc_ids

                        folder_doc_ids = _collect_doc_ids(document_id)
                except Exception:
                    logger.warning("get_important_blocks_folder_resolve_failed", folder_id=document_id)
                if not folder_doc_ids:
                    return {"blocks": [], "count": 0}

        result = await _important_blocks_core(
            graph_id, user_id, auth, limit, valence, folder_doc_ids,
        )
        return bare_ids_in_result(result)

    async def _important_blocks_core(
        graph_id: str,
        user_id: str,
        auth: MCPAuthContext,
        limit: int = 8,
        valence: Optional[str] = None,
        folder_doc_ids: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Core logic for get_important_blocks. Returns dict with 'blocks' and 'count'.

        Shared between get_important_blocks_tool and context_bundle_tool.
        """
        params: Dict[str, Any] = {"limit": min(limit * 10, 200)}
        if valence:
            params["valence"] = valence

        url = f"{backend_config.base_url}/salience/{graph_id}/blocks/values"
        resp = await get_http_client().get(url, params=params, headers=auth.http_headers())
        resp.raise_for_status()
        data = resp.json()

        raw_blocks = data.get("blocks", [])

        # Python-side folder filtering
        if folder_doc_ids is not None:
            folder_doc_set = set(folder_doc_ids)
            raw_blocks = [b for b in raw_blocks if b.get("document_id") in folder_doc_set]

        scored = raw_blocks[:limit]

        if not scored:
            return {"blocks": [], "count": 0}

        # Connect to workspace for title resolution
        ws_doc = None
        try:
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            ws_channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
            if ws_channel:
                ws_doc = ws_channel.doc
        except Exception:
            pass

        # Fetch content and titles for the top blocks
        doc_groups: Dict[str, list] = {}
        for item in scored:
            doc_groups.setdefault(item["document_id"], []).append(item)

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
                        "score": item["composite_score"],
                        "importance": item["cumulative_importance"],
                        "valence": item["cumulative_valence"],
                        "block_id": item["block_id"],
                        "doc_id": item["document_id"],
                        "block_wires": item.get("block_wire_count", 0),
                        "doc_wires": item.get("doc_wire_count", 0),
                    })
                return doc_results
            except Exception:
                return [{
                    "content": "(unavailable)",
                    "document": title,
                    "score": item["composite_score"],
                    "importance": item["cumulative_importance"],
                    "valence": item["cumulative_valence"],
                    "block_id": item["block_id"],
                    "doc_id": item["document_id"],
                    "block_wires": item.get("block_wire_count", 0),
                    "doc_wires": item.get("doc_wire_count", 0),
                } for item in items]

        doc_result_lists = await asyncio.gather(
            *[_fetch_doc_blocks(did, items) for did, items in doc_groups.items()]
        )
        results = [entry for sublist in doc_result_lists for entry in sublist]
        results.sort(key=lambda b: b["score"], reverse=True)

        return {"blocks": results, "count": len(results)}

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
    @resolve_home_graph
    async def get_values_tool(
        graph_id: str | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        await _ensure_scratchpad(hp_client, graph_id, auth)

        # Read from HP present/* docs — these are the authoritative, user-editable source
        await asyncio.gather(
            hp_client.connect_document(graph_id, IMPORTANCE_DOC_ID, user_id=auth.user_id),
            hp_client.connect_document(graph_id, VALENCE_DOC_ID, user_id=auth.user_id),
            hp_client.connect_document(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id),
        )

        importance_xml = DocumentReader(
            hp_client.get_document_channel(graph_id, IMPORTANCE_DOC_ID, user_id=auth.user_id).doc
        ).to_xml()
        valence_xml = DocumentReader(
            hp_client.get_document_channel(graph_id, VALENCE_DOC_ID, user_id=auth.user_id).doc
        ).to_xml()
        weights_xml = DocumentReader(
            hp_client.get_document_channel(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id).doc
        ).to_xml()

        importance_text = tiptap_xml_to_markdown(importance_xml)
        valence_text = tiptap_xml_to_markdown(valence_xml)
        weights_text = tiptap_xml_to_markdown(weights_xml)
        weights = _parse_weights_text(weights_text)

        return {
            "importance_prompt": importance_text,
            "valence_prompt": valence_text,
            "weights": weights,
        }

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
    @resolve_home_graph
    async def revaluate_tool(
        graph_id: str | None = None,
        importance_prompt: Optional[str] = None,
        valence_prompt: Optional[str] = None,
        weights: Optional[str] = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        await _ensure_scratchpad(hp_client, graph_id, auth)

        now_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        updated = []

        # Read current values from HP present/* docs — these are the authoritative source
        await asyncio.gather(
            hp_client.connect_document(graph_id, IMPORTANCE_DOC_ID, user_id=auth.user_id),
            hp_client.connect_document(graph_id, VALENCE_DOC_ID, user_id=auth.user_id),
            hp_client.connect_document(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id),
        )
        current_importance_text = tiptap_xml_to_markdown(
            DocumentReader(hp_client.get_document_channel(graph_id, IMPORTANCE_DOC_ID, user_id=auth.user_id).doc).to_xml()
        )
        current_valence_text = tiptap_xml_to_markdown(
            DocumentReader(hp_client.get_document_channel(graph_id, VALENCE_DOC_ID, user_id=auth.user_id).doc).to_xml()
        )
        current_weights_text = tiptap_xml_to_markdown(
            DocumentReader(hp_client.get_document_channel(graph_id, WEIGHTS_DOC_ID, user_id=auth.user_id).doc).to_xml()
        )
        current_weights = _parse_weights_text(current_weights_text)

        patch: Dict[str, Any] = {}

        # Archive old value to HP past/ doc (genealogy preserved)
        async def _archive_to_hp(past_doc_id: str, old_content: str) -> None:
            archive_entry = (
                f'<heading level="3">Archived {html_mod.escape(now_label)}</heading>'
                f"<paragraph>{html_mod.escape(old_content or '')}</paragraph>"
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

        if importance_prompt is not None:
            await _archive_to_hp(PAST_IMPORTANCE_DOC_ID, current_importance_text)
            patch["importance_prompt"] = importance_prompt
            updated.append("importance_prompt")

        if valence_prompt is not None:
            await _archive_to_hp(PAST_VALENCE_DOC_ID, current_valence_text)
            patch["valence_prompt"] = valence_prompt
            updated.append("valence_prompt")

        if weights is not None:
            # Accept both JSON object form '{"half_life_days": 60.0}' and
            # line-based form 'half_life_days: 60.0\nother_key: value'.
            # Normalize to line-based text before parsing.
            normalized_weights_text = weights.strip()
            if normalized_weights_text.startswith("{"):
                try:
                    parsed_json = json.loads(normalized_weights_text)
                    if not isinstance(parsed_json, dict):
                        raise ValueError("weights JSON must be an object")
                    normalized_weights_text = "\n".join(
                        f"{k}: {v}" for k, v in parsed_json.items()
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    return {
                        "error": f"weights looks like JSON but failed to parse: {exc}",
                        "success": False,
                    }

            # Parse new weights and extract only explicitly specified keys
            new_weights = _parse_weights_text(normalized_weights_text)
            specified_keys = set()
            for line in normalized_weights_text.split("\n"):
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

            if not specified_keys:
                return {
                    "error": (
                        "No valid weight keys found in input. Expected either "
                        "JSON like '{\"half_life_days\": 60.0}' or lines like "
                        f"'half_life_days: 60.0'. Valid keys: {sorted(DEFAULT_WEIGHTS.keys())}"
                    ),
                    "success": False,
                }

            # Archive current HP weights text verbatim (only after validation)
            await _archive_to_hp(PAST_WEIGHTS_DOC_ID, current_weights_text)

            for key in specified_keys:
                patch[key] = new_weights[key]
            updated.append("weights")

        if patch:
            # Write new values to HP present/* docs (AUTHORITATIVE write)
            async def _write_present_doc(doc_id: str, label: str, content: str) -> None:
                new_xml = f'<heading level="1">{html_mod.escape(label)}</heading>'
                for line in content.strip().split("\n"):
                    if line.strip():
                        new_xml += f"<paragraph>{html_mod.escape(line.strip())}</paragraph>"
                await hp_client.connect_document(graph_id, doc_id, user_id=auth.user_id)
                await hp_client.transact_document(
                    graph_id,
                    doc_id,
                    lambda doc, nx=new_xml: DocumentWriter(doc).replace_all_content(nx),
                    user_id=auth.user_id,
                )

            write_tasks = []
            if importance_prompt is not None:
                write_tasks.append(_write_present_doc(IMPORTANCE_DOC_ID, "Importance Prompt", importance_prompt))
            if valence_prompt is not None:
                write_tasks.append(_write_present_doc(VALENCE_DOC_ID, "Valence Prompt", valence_prompt))
            if weights is not None:
                # Merge new specified values into current weights
                merged_weights = dict(current_weights)
                for key in specified_keys:
                    merged_weights[key] = new_weights[key]
                merged_text = "\n".join(
                    f"{k}: {merged_weights[k]}" for k in DEFAULT_WEIGHTS if k in merged_weights
                )
                write_tasks.append(_write_present_doc(WEIGHTS_DOC_ID, "Scoring Configuration", merged_text))
            await asyncio.gather(*write_tasks)

            # Sync to DynamoDB via platform PATCH (cache — scoring reads from HP, not here)
            # Fire-and-forget: HP write already succeeded; DynamoDB failure shouldn't surface as error
            try:
                cfg_url = f"{backend_config.base_url}/salience/{graph_id}/config"
                resp = await get_http_client().patch(cfg_url, json=patch, headers=auth.http_headers())
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("revaluate_dynamodb_sync_failed", graph_id=graph_id, error=str(exc))

        return {"success": True, "updated": updated}

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
    @resolve_home_graph
    async def archive_memories_tool(
        graph_id: str | None = None,
        keep: int = 50,
        context: Context | None = None,
    ) -> dict[str, Any]:
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        graph_id = graph_id.strip()
        if keep < 1:
            return {"error": "keep must be >= 1"}

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
                return {
                    "kept": 0, "archived": 0, "note": "Memory queue is empty.",
                }

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
                return {
                    "kept": len(keep_entries),
                    "archived": 0,
                    "note": f"Queue has {len(keep_entries)} memories, nothing to archive.",
                }

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

        return {
            "kept": len(keep_blocks),
            "archived": len(archive_blocks),
            "archive_doc_id": archive_doc_id,
            "note": (
                f"Archived {len(archive_blocks)} memories to past/{archive_doc_id}. "
                f"Queue rewritten with {len(keep_blocks)} memories."
            ),
        }

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
    @resolve_home_graph
    async def quick_orient_tool(
        graph_id: str | None = None,
        recall_limit: int = 8,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """Return a lightweight orientation bundle (location + song + recall).

        Args:
            graph_id: The graph to orient in (default: "default")
            recall_limit: Number of recent memories to include (default: 8)
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

        return bare_ids_in_result(result)

    # ================================================================
    # CONTEXT BUNDLE — single-call attunement
    # ================================================================

    @server.tool(
        name="context_bundle",
        title="Context Bundle",
        description=(
            "Single-call attunement: returns location, Song, recent memories, "
            "agent identity document, important blocks, and workspace structure "
            "in one response. Replaces the manual 3-step attunement sequence.\n\n"
            "**Execution:** Phase 1 resolves location (sequential). Phase 2 runs "
            "Song, recall, agent document, important blocks, and workspace in "
            "parallel. Each component is error-isolated — partial results on failure.\n\n"
            "**Home graph:** By default, auto-sets the resolved graph as the session's "
            "home graph so subsequent tool calls can omit graph_id. Pass set_home=false "
            "to skip.\n\n"
            "**Processing guidance:** The response is structured for staged processing. "
            "Let the Song land first (narrative identity), then recall and agent document "
            "(working memory and individuation), then important blocks and workspace "
            "(judgment and structure). Attunement is a practice, not a data fetch.\n\n"
            "Parameters:\n"
            "- graph_id: Graph to attune in (default: resolved from user's current location)\n"
            "- agent_name: Agent identity document to read, e.g. 'gamma' reads 'agent-gamma'. "
            "Omit to skip.\n"
            "- recall_limit: Number of recent memories (default 8)\n"
            "- important_limit: Number of top-valued blocks (default 8)\n"
            "- workspace_depth: Folder nesting depth (default 1)\n"
            "- set_home: Auto-set home graph for the session (default true)"
        ),
    )
    async def context_bundle_tool(
        graph_id: str | None = None,
        agent_name: str | None = None,
        recall_limit: int = 8,
        important_limit: int = 8,
        workspace_depth: int = 1,
        set_home: bool = True,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """Single-call attunement returning all orientation data."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()
        user_id = _resolve_user_id(auth)

        result: Dict[str, Any] = {}

        # --- Phase 1: Resolve graph_id from location if not provided ---
        if graph_id and graph_id.strip():
            graph_id = graph_id.strip()
            result["location"] = {
                "graph_id": graph_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            # REST call to get user's current location
            try:
                url = f"{backend_config.base_url}/sessions/location"
                resp = await get_http_client().get(
                    url, headers=auth.http_headers(),
                    timeout=httpx.Timeout(5.0),
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    graph_id = payload.get("graph_id")
                    location: Dict[str, Any] = {
                        "graph_id": graph_id,
                        "document_id": payload.get("document_id"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    display_name = payload.get("display_name")
                    if display_name:
                        location["display_name"] = display_name
                    result["location"] = location
            except Exception:
                pass

            if not graph_id:
                # Fallback: WebSocket session refresh
                try:
                    await hp_client.refresh_session(user_id)
                    graph_id = hp_client.get_active_graph_id()
                    result["location"] = {
                        "graph_id": graph_id,
                        "document_id": hp_client.get_active_document_id(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                except Exception as e:
                    result["location"] = {"error": str(e)}

            if not graph_id:
                raise ValueError(
                    "Could not resolve graph_id from user location. "
                    "Pass graph_id explicitly or ensure a browser session is active."
                )

        # Auto-set home graph
        if set_home:
            set_home_graph(user_id, graph_id)
            result["home_graph"] = graph_id
        else:
            current_home = get_home_graph(user_id)
            if current_home:
                result["home_graph"] = current_home

        # --- Phase 2: Run all components concurrently ---

        async def _bundle_song() -> Dict[str, Any]:
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

        async def _bundle_recall() -> Dict[str, Any]:
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
                memories.sort(
                    key=lambda m: max(m.get("created_at", ""), m.get("last_active", "")),
                    reverse=True,
                )
                memories = memories[:recall_limit]
                return {"memories": memories, "count": len(memories)}
            except Exception as e:
                return {"error": str(e)}

        async def _bundle_agent_doc() -> Optional[Dict[str, Any]]:
            if not agent_name:
                return None
            doc_id = f"agent-{agent_name.strip()}"
            try:
                await hp_client.connect_document(graph_id, doc_id, user_id=auth.user_id)
                channel = hp_client.get_document_channel(graph_id, doc_id, user_id=auth.user_id)
                if channel is None:
                    return {"title": doc_id, "error": "document not found"}
                reader = DocumentReader(channel.doc)
                xml = reader.to_xml()
                markdown = tiptap_xml_to_markdown(xml)
                return {"title": doc_id, "content": markdown}
            except Exception as e:
                return {"title": doc_id, "error": str(e)}

        async def _bundle_important() -> Dict[str, Any]:
            try:
                return await _important_blocks_core(
                    graph_id, user_id, auth, important_limit,
                )
            except Exception as e:
                return {"blocks": [], "count": 0, "error": str(e)}

        async def _bundle_workspace() -> Dict[str, Any]:
            try:
                await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
                ws_channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
                if ws_channel is None:
                    return {"error": "workspace not available"}
                ws_reader = WorkspaceReader(ws_channel.doc)

                def _build_tree(parent_id: str | None, depth: int) -> list:
                    children = ws_reader.get_children_of(parent_id)
                    tree = []
                    for entity_type, entity_id, _ in children:
                        if entity_type == "folder":
                            folder_info = ws_reader.get_folder(entity_id)
                            name = folder_info.get("name", entity_id) if folder_info else entity_id
                            if depth <= 1:
                                # At depth limit, collapse to count
                                child_count = len(ws_reader.get_children_of(entity_id))
                                tree.append({
                                    "id": entity_id,
                                    "type": "folder",
                                    "name": name,
                                    "collapsed": f"{child_count} documents",
                                })
                            else:
                                subtree = _build_tree(entity_id, depth - 1)
                                tree.append({
                                    "id": entity_id,
                                    "type": "folder",
                                    "name": name,
                                    "children": subtree,
                                })
                        elif entity_type == "document":
                            doc_info = ws_reader.get_document(entity_id)
                            title = doc_info.get("title", entity_id) if doc_info else entity_id
                            tree.append({
                                "id": entity_id,
                                "type": "document",
                                "title": title,
                            })
                    return tree

                tree = _build_tree(None, workspace_depth)
                ws_result: Dict[str, Any] = {"tree": tree, "graph_id": graph_id}
                if workspace_depth > 0:
                    ws_result["depth"] = workspace_depth
                # Surface dream journal if it exists
                dj_id = f"{graph_id}-dream-journal"
                dj_info = ws_reader.get_document(dj_id)
                if dj_info:
                    ws_result["dream_journal"] = dj_id
                return ws_result
            except Exception as e:
                return {"error": str(e)}

        # Run all Phase 2 tasks concurrently
        song, recall, agent_doc, important, workspace = await asyncio.gather(
            _bundle_song(),
            _bundle_recall(),
            _bundle_agent_doc(),
            _bundle_important(),
            _bundle_workspace(),
        )

        result["song"] = song
        result["recall"] = recall
        if agent_doc is not None:
            result["agent_document"] = agent_doc
        result["important_blocks"] = important
        result["workspace"] = workspace

        logger.info(
            "context_bundle",
            extra_context={
                "graph_id": graph_id,
                "agent_name": agent_name,
                "user_id": user_id,
                "components": len([v for v in [song, recall, agent_doc, important, workspace] if v]),
            },
        )

        return bare_ids_in_result(result)

    logger.info("Registered Geist (Sophia Memory) tools: 14 tools")


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
    return bare_ids_in_result(result)


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
