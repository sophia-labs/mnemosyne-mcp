"""
MCP tools that use Hocuspocus/Y.js for real-time document access.

These tools provide direct read/write access to Mnemosyne documents via Y.js
CRDT synchronization, bypassing the job queue for lower latency operations.
"""

from __future__ import annotations

import asyncio
import html as html_mod
import json
import math
import mimetypes
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import pycrdt
from mcp.server.fastmcp import Context, FastMCP

from neem.hocuspocus import HocuspocusClient, DocumentReader, DocumentWriter, WorkspaceWriter, WorkspaceReader
from neem.hocuspocus.converters import looks_like_markdown, markdown_to_tiptap_xml, tiptap_xml_to_html, tiptap_xml_to_markdown
from neem.hocuspocus.document import extract_title_from_xml
from neem.mcp.auth import MCPAuthContext
from neem.mcp.jobs import RealtimeJobClient
from neem.mcp.tools.basic import await_job_completion, submit_job
from neem.mcp.tools.wire_tools import _get_wires_for_document, _get_predicate_short_name
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_dev_user_id, get_internal_service_secret, get_user_id_from_token, validate_token_and_load

logger = LoggerFactory.get_logger("mcp.tools.hocuspocus")

JsonDict = Dict[str, Any]


def _ensure_xml(text: str) -> str:
    """Convert text to TipTap XML if needed.

    - If text starts with '<', it's returned as-is (assumed XML).
    - If text looks like markdown, it's parsed via markdown_to_tiptap_xml().
    - Otherwise, it's wrapped in a <paragraph> element as plain text.
    """
    content = text.strip()
    if not content:
        return content
    if content.startswith("<"):
        return content
    if looks_like_markdown(content):
        return markdown_to_tiptap_xml(content)
    return f"<paragraph>{html_mod.escape(content)}</paragraph>"


def _ensure_xml_multiblock(text: str) -> str:
    """Like _ensure_xml, but for write_document where input may contain
    multiple paragraphs or full markdown documents.

    - If text starts with '<', it's returned as-is (assumed XML).
    - If text looks like markdown, it's parsed via markdown_to_tiptap_xml().
    - Otherwise, plain text is split on double-newlines into <paragraph> blocks.
    """
    content = text.strip()
    if not content:
        return content
    if content.startswith("<"):
        return content
    if looks_like_markdown(content):
        return markdown_to_tiptap_xml(content)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return "".join(
        f"<paragraph>{html_mod.escape(p)}</paragraph>" for p in paragraphs
    )


def register_hocuspocus_tools(server: FastMCP) -> None:
    """Register document tools that use the Hocuspocus WebSocket client."""

    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping hocuspocus tool registration")
        return

    # Get or create the HocuspocusClient
    hp_client: Optional[HocuspocusClient] = getattr(server, "_hocuspocus_client", None)
    if hp_client is None:
        hp_client = HocuspocusClient(
            base_url=backend_config.base_url,
            token_provider=validate_token_and_load,
            dev_user_id=get_dev_user_id(),
            internal_service_secret=get_internal_service_secret(),
        )
        server._hocuspocus_client = hp_client  # type: ignore[attr-defined]
        logger.info(
            "Created HocuspocusClient for real-time document access",
            extra_context={"base_url": backend_config.base_url},
        )

    # Get job stream for SPARQL queries (used by scoring filter)
    job_stream: Optional[RealtimeJobClient] = getattr(server, "_job_stream", None)

    # ------------------------------------------------------------------
    # Helper: resolve user_id from auth context
    # ------------------------------------------------------------------
    def _resolve_user_id(auth: MCPAuthContext, token: str, user_id: Optional[str] = None) -> str:
        if not user_id:
            user_id = auth.user_id or (get_user_id_from_token(token) if token else None)
            if not user_id:
                raise RuntimeError(
                    "Could not determine user ID. Either provide it explicitly or "
                    "ensure your token contains a 'sub' claim."
                )
        return user_id

    # ------------------------------------------------------------------
    # Direct read path (Platform HTTP blob endpoint -> WebSocket fallback)
    # ------------------------------------------------------------------
    raw_read_path_mode = (os.getenv("MNEMOSYNE_READ_PATH", "websocket").strip().lower() or "websocket")
    mode_aliases = {
        "redis": "http_blob",  # deprecated alias
        "blob": "http_blob",
    }
    read_path_mode = mode_aliases.get(raw_read_path_mode, raw_read_path_mode)
    if read_path_mode not in {"websocket", "http_blob", "hybrid"}:
        logger.warning(
            "Invalid MNEMOSYNE_READ_PATH value; falling back to websocket",
            extra_context={"mode": raw_read_path_mode},
        )
        read_path_mode = "websocket"
    elif raw_read_path_mode in mode_aliases:
        logger.warning(
            "Deprecated MNEMOSYNE_READ_PATH alias in use",
            extra_context={"provided": raw_read_path_mode, "resolved": read_path_mode},
        )

    read_path_counters: Dict[str, int] = getattr(server, "_read_path_counters", None) or {}
    server._read_path_counters = read_path_counters  # type: ignore[attr-defined]

    def _bump_read_counter(name: str) -> None:
        read_path_counters[name] = read_path_counters.get(name, 0) + 1

    def _parse_float_env(name: str, default: float) -> float:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning(
                "Invalid float env var; using default",
                extra_context={"name": name, "value": raw, "default": default},
            )
            return default

    def _ymap_to_dict(ymap: pycrdt.Map) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in ymap.keys():
            value = ymap.get(key)
            if isinstance(value, pycrdt.Map):
                result[key] = _ymap_to_dict(value)
            elif isinstance(value, pycrdt.Array):
                result[key] = list(value)
            else:
                result[key] = value
        return result

    def _workspace_snapshot_from_doc(doc: pycrdt.Doc) -> Dict[str, Any]:
        folders_map: pycrdt.Map = doc.get("folders", type=pycrdt.Map)
        artifacts_map: pycrdt.Map = doc.get("artifacts", type=pycrdt.Map)
        documents_map: pycrdt.Map = doc.get("documents", type=pycrdt.Map)
        ui_map: pycrdt.Map = doc.get("ui", type=pycrdt.Map)
        return {
            "folders": _ymap_to_dict(folders_map),
            "artifacts": _ymap_to_dict(artifacts_map),
            "documents": _ymap_to_dict(documents_map),
            "ui": _ymap_to_dict(ui_map),
        }

    def _decode_ydoc(raw: bytes, *, label: str) -> pycrdt.Doc:
        try:
            doc = pycrdt.Doc()
            doc.apply_update(raw)
            return doc
        except Exception as exc:
            _bump_read_counter("decode_fail")
            raise RuntimeError(f"Failed to decode Y.Doc from {label}: {exc}") from exc

    _blob_timeout_seconds = _parse_float_env("MNEMOSYNE_READ_BLOB_TIMEOUT_SECONDS", 8.0)

    async def _http_get_blob(path: str, auth: MCPAuthContext) -> tuple[bytes | None, int, str | None]:
        url = f"{backend_config.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(_blob_timeout_seconds)) as client:
                response = await client.get(url, headers=auth.http_headers())
            if response.status_code == 200:
                blob_source = response.headers.get("X-Blob-Source")
                return response.content, 200, blob_source
            if response.status_code == 410:
                return None, 410, None
            _bump_read_counter("blob_http_error")
            logger.debug(
                "http_blob_read_non_200",
                extra_context={"url": path, "status": response.status_code},
            )
            return None, response.status_code, None
        except Exception as exc:
            logger.warning(
                "http_blob_read_failed",
                extra_context={"url": path, "error": str(exc)},
            )
            return None, 0, None

    def _record_read_source(tool_name: str, source: str) -> None:
        if source.startswith("http_blob"):
            _bump_read_counter("blob_hit")
            if source.endswith(":redis"):
                _bump_read_counter("redis_hit")
            elif source.endswith(":s3"):
                _bump_read_counter("s3_fallback")
        elif source == "websocket" and read_path_mode == "hybrid":
            _bump_read_counter("ws_fallback")
        logger.debug(
            "direct_read_source_selected",
            extra_context={
                "tool": tool_name,
                "source": source,
                "mode": read_path_mode,
                "counters": dict(read_path_counters),
            },
        )

    async def _load_workspace_doc_for_read(
        graph_id: str,
        user_id: str,
        *,
        auth: MCPAuthContext,
        tool_name: str,
    ) -> tuple[pycrdt.Doc, str]:
        if read_path_mode in {"http_blob", "hybrid"}:
            raw, status_code, blob_source = await _http_get_blob(
                f"/documents/{graph_id}/workspace/blob",
                auth,
            )
            if raw:
                doc = _decode_ydoc(raw, label=f"http blob workspace {graph_id}")
                source = f"http_blob:{blob_source}" if blob_source else "http_blob"
                _record_read_source(tool_name, source)
                return doc, source

            if read_path_mode == "http_blob":
                raise RuntimeError(
                    f"Workspace '{graph_id}' blob read failed (status={status_code}). "
                    "Set MNEMOSYNE_READ_PATH=hybrid or websocket to allow fallback."
                )

        await hp_client.connect_workspace(graph_id, user_id=user_id)
        channel = hp_client.get_workspace_channel(graph_id, user_id=user_id)
        if channel is None:
            raise RuntimeError(f"Could not connect to workspace for graph '{graph_id}'")
        _record_read_source(tool_name, "websocket")
        return channel.doc, "websocket"

    async def _load_document_doc_for_read(
        graph_id: str,
        document_id: str,
        user_id: str,
        *,
        auth: MCPAuthContext,
        tool_name: str,
    ) -> tuple[pycrdt.Doc, str]:
        if read_path_mode in {"http_blob", "hybrid"}:
            raw, status_code, blob_source = await _http_get_blob(
                f"/documents/{graph_id}/{document_id}/blob",
                auth,
            )
            if raw:
                doc = _decode_ydoc(raw, label=f"http blob document {document_id}")
                source = f"http_blob:{blob_source}" if blob_source else "http_blob"
                _record_read_source(tool_name, source)
                return doc, source

            if status_code == 410:
                raise RuntimeError(
                    f"Document '{document_id}' is tombstoned in graph '{graph_id}'."
                )

            if read_path_mode == "http_blob":
                raise RuntimeError(
                    f"Document '{document_id}' blob read failed (status={status_code}). "
                    "Set MNEMOSYNE_READ_PATH=hybrid or websocket to allow fallback."
                )

        await _connect_for_read(graph_id, document_id, user_id)
        channel = hp_client.get_document_channel(graph_id, document_id, user_id=user_id)
        if channel is None:
            raise RuntimeError(f"Document channel not found: {graph_id}/{document_id}")
        _record_read_source(tool_name, "websocket")
        return channel.doc, "websocket"

    logger.info(
        "direct_read_path_configured",
        extra_context={
            "mode": read_path_mode,
            "backend_base_url": backend_config.base_url,
            "blob_timeout_seconds": _blob_timeout_seconds,
        },
    )

    # ------------------------------------------------------------------
    # Helper: get document-level scores for workspace filtering
    # ------------------------------------------------------------------
    async def _get_excluded_docs_by_score(
        graph_id: str,
        min_score: float,
        auth: MCPAuthContext,
    ) -> set[str]:
        """Query block valuations and compute per-document scores.

        Returns (excluded_doc_ids, valued_doc_ids) where:
        - excluded_doc_ids: docs that have been valuated but score below min_score
        - valued_doc_ids: all docs that have any valuations (for enriched collapse counts)

        Documents with no valuations are NOT excluded (unscored docs always pass).
        Documents in _sophia are never excluded (infrastructure protection).

        Document score uses bicameral approach: avg importance + avg valence are
        normalized for doc size, but max importance ensures a single extraordinary
        block can carry a document.
        """
        user_id = auth.user_id or (get_user_id_from_token(auth.token) if auth.token else None)
        if not user_id:
            return set(), set()

        graph_uri = f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"

        sparql = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?docId
       (AVG(xsd:float(?cumImp)) AS ?avgImp)
       (MAX(xsd:float(?cumImp)) AS ?maxImp)
       (AVG(ABS(xsd:float(?cumVal))) AS ?avgAbsVal)
FROM <{graph_uri}>
WHERE {{
  ?val doc:blockRef ?blockRef .
  ?val doc:cumulativeImportance ?cumImp .
  ?val doc:cumulativeValence ?cumVal .
  BIND(STRBEFORE(STRAFTER(STR(?blockRef), ":doc:"), "#") AS ?docId)
}}
GROUP BY ?docId
"""
        try:
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

            ws_events, poll_payload = await await_job_completion(
                job_stream, metadata, auth, timeout=30.0,
            )

            # Extract results from WS events or poll
            raw = None
            if ws_events:
                for event in reversed(ws_events):
                    event_type = event.get("type", "")
                    payload = event.get("payload", {})
                    payload_status = payload.get("status", "") if isinstance(payload, dict) else ""
                    is_success = (
                        event_type in ("job_completed", "completed", "succeeded")
                        or (event_type == "job_update" and payload_status == "succeeded")
                    )
                    if not is_success:
                        continue
                    if isinstance(payload, dict):
                        detail = payload.get("detail")
                        if isinstance(detail, dict):
                            inline = detail.get("result_inline")
                            if isinstance(inline, dict) and "raw" in inline:
                                raw = inline["raw"]
                            elif inline is not None:
                                raw = inline
                        if raw is None and payload.get("result_inline") is not None:
                            inline = payload.get("result_inline")
                            if isinstance(inline, dict) and "raw" in inline:
                                raw = inline["raw"]
                            else:
                                raw = inline
                    if raw is None and event.get("result_inline") is not None:
                        inline = event.get("result_inline")
                        if isinstance(inline, dict) and "raw" in inline:
                            raw = inline["raw"]
                        else:
                            raw = inline
                    break
            if raw is None and poll_payload:
                detail = poll_payload.get("detail")
                if isinstance(detail, dict):
                    inline = detail.get("result_inline")
                    if isinstance(inline, dict) and "raw" in inline:
                        raw = inline["raw"]
                    elif inline is not None:
                        raw = inline

            if not raw or not isinstance(raw, dict):
                return set(), set()

            bindings = raw.get("results", {}).get("bindings", [])

            # Infrastructure doc IDs that should never be filtered
            protected_prefixes = ("geist-",)

            # Compute per-document scores and find those below threshold
            excluded = set()
            valued = set()
            for binding in bindings:
                doc_id = binding.get("docId", {}).get("value", "")
                if not doc_id:
                    continue

                valued.add(doc_id)

                # Never exclude infrastructure documents
                if any(doc_id.startswith(p) for p in protected_prefixes):
                    continue

                avg_imp = float(binding.get("avgImp", {}).get("value", 0))
                max_imp = float(binding.get("maxImp", {}).get("value", 0))
                avg_abs_val = float(binding.get("avgAbsVal", {}).get("value", 0))

                # Bicameral with max component: a single extraordinary block
                # can carry a document (max), but uniformly solid content also
                # surfaces (avg). Valence adds affective intensity.
                doc_score = max_imp * 0.4 + avg_imp * 0.3 + avg_abs_val * 0.3
                if doc_score < min_score:
                    excluded.add(doc_id)

            return excluded, valued

        except Exception as e:
            logger.warning(
                "Failed to query document scores for workspace filter, showing all docs",
                extra_context={"graph_id": graph_id, "error": str(e)},
            )
            return set(), set()  # Graceful degradation: show everything

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    async def _validate_document_in_workspace(
        graph_id: str,
        document_id: str,
        user_id: str,
        auth: Optional[MCPAuthContext] = None,
    ) -> None:
        """Verify a document exists in the workspace metadata.

        Raises RuntimeError with a helpful message listing available
        documents when the requested document is not found.
        """
        if read_path_mode != "websocket" and auth is not None:
            workspace_doc, _ = await _load_workspace_doc_for_read(
                graph_id,
                user_id,
                auth=auth,
                tool_name="validate_document",
            )
            snapshot = _workspace_snapshot_from_doc(workspace_doc)
            docs = snapshot.get("documents") or {}
            if document_id in docs:
                return

            available = []
            for doc_id, doc_info in docs.items():
                title = doc_info.get("title", doc_id) if isinstance(doc_info, dict) else doc_id
                available.append(f"  - {title} ({doc_id})")

            msg = (
                f"Document '{document_id}' not found in graph '{graph_id}'. "
                "It may have been deleted — if so, use a different document ID "
                "(deleted document IDs are tombstoned and cannot be reused)."
            )
            if available:
                shown = available[:15]
                msg += "\n\nAvailable documents:\n" + "\n".join(shown)
                if len(available) > 15:
                    msg += f"\n  ... and {len(available) - 15} more"
            else:
                msg += (
                    " The graph workspace is empty — this may mean the graph "
                    "doesn't exist or you're connected to the wrong backend."
                )
            msg += "\n\nUse get_workspace to see the full graph structure."
            raise RuntimeError(msg)

        def _exists_in_snapshot() -> tuple[bool, list[str]]:
            snapshot = hp_client.get_workspace_snapshot(graph_id, user_id=user_id)
            docs = snapshot.get("documents") or {}
            return (document_id in docs), list(docs.keys())

        await hp_client.connect_workspace(graph_id, user_id=user_id)
        ws_channel = hp_client.get_workspace_channel(graph_id, user_id=user_id)
        if ws_channel is None:
            raise RuntimeError(
                f"Could not connect to workspace for graph '{graph_id}'. "
                f"The graph may not exist or the backend may be unreachable. "
                f"Use list_graphs to see available graphs."
            )
        reader = WorkspaceReader(ws_channel.doc)
        if reader.get_document(document_id) is not None:
            return  # Document exists

        # Retry once with force-fresh workspace reconnect before failing.
        await hp_client.connect_workspace(
            graph_id,
            user_id=user_id,
            force_fresh=True,
            max_age=0,
        )
        ws_channel = hp_client.get_workspace_channel(graph_id, user_id=user_id)
        if ws_channel is not None:
            reader = WorkspaceReader(ws_channel.doc)
            if reader.get_document(document_id) is not None:
                return

        exists_in_snapshot, _ = _exists_in_snapshot()
        if exists_in_snapshot:
            # Snapshot says it exists, but reader path is inconsistent.
            # Surface explicit transient guidance instead of tombstone wording.
            raise RuntimeError(
                f"Document '{document_id}' appears in workspace snapshot for graph '{graph_id}', "
                "but is not visible through the current channel reader (transient sync divergence). "
                "Retry in a moment or reconnect MCP."
            )

        # Build a helpful error with available documents
        snapshot = hp_client.get_workspace_snapshot(graph_id, user_id=user_id)
        available: list[str] = []
        if snapshot:
            docs = snapshot.get("documents") or {}
            for doc_id, doc_info in docs.items():
                title = doc_info.get("title", doc_id) if isinstance(doc_info, dict) else doc_id
                available.append(f"  - {title} ({doc_id})")

        msg = (
            f"Document '{document_id}' not found in graph '{graph_id}'. "
            f"It may have been deleted — if so, use a different document ID "
            f"(deleted document IDs are tombstoned and cannot be reused)."
        )
        if available:
            shown = available[:15]
            msg += "\n\nAvailable documents:\n" + "\n".join(shown)
            if len(available) > 15:
                msg += f"\n  ... and {len(available) - 15} more"
        else:
            msg += (
                " The graph workspace is empty — this may mean the graph "
                "doesn't exist or you're connected to the wrong backend."
            )
        msg += "\n\nUse get_workspace to see the full graph structure."
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Helper: connect a document for reading (with TTL + retry)
    # ------------------------------------------------------------------
    _READ_MAX_AGE = 2.0  # seconds — reconnect if last sync is older than this

    async def _connect_for_read(
        graph_id: str, document_id: str, user_id: str,
        *,
        force_fresh: bool = False,
        disconnect_first: bool = False,
    ) -> None:
        """Connect to a document channel for a read operation.

        Uses max_age TTL so rapid sequential reads of the same doc reuse
        the cached channel, but stale channels from other agents' writes
        are refreshed.  Retries once on timeout (disconnect + fresh connect).
        """
        try:
            if disconnect_first:
                await hp_client.disconnect_document(graph_id, document_id, user_id=user_id)
            await hp_client.connect_document(
                graph_id, document_id, user_id=user_id,
                force_fresh=force_fresh,
                max_age=None if force_fresh else _READ_MAX_AGE,
            )
        except TimeoutError:
            logger.warning(
                "Document sync timed out on first attempt, retrying",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                },
            )
            # Tear down any partial state and try once more with force_fresh
            await hp_client.disconnect_document(graph_id, document_id, user_id=user_id)
            await hp_client.connect_document(
                graph_id, document_id, user_id=user_id,
                force_fresh=True,
            )

    async def _force_flush_document_via_api(
        auth: MCPAuthContext,
        graph_id: str,
        document_id: str,
        *,
        include_materialization: bool = False,
        attempts: int = 3,
    ) -> bool:
        """Call backend flush endpoint and return True if active session was flushed."""
        url = f"{backend_config.base_url}/documents/{graph_id}/{document_id}/flush"
        headers = auth.http_headers()
        params = {"include_materialization": "true" if include_materialization else "false"}

        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                    resp = await client.post(url, params=params, headers=headers)

                # Rolling deploy safety: endpoint may not exist yet on some pods.
                if resp.status_code == 404:
                    logger.debug(
                        "document_flush_endpoint_unavailable",
                        extra_context={"graph_id": graph_id, "document_id": document_id},
                    )
                    return False

                if resp.status_code != 200:
                    logger.warning(
                        "document_flush_http_error",
                        extra_context={
                            "graph_id": graph_id,
                            "document_id": document_id,
                            "status": resp.status_code,
                            "attempt": attempt,
                            "attempts": attempts,
                        },
                    )
                else:
                    try:
                        payload = resp.json()
                    except Exception:
                        payload = {}
                    active = bool(payload.get("activeSession"))
                    if active:
                        return True
            except Exception as exc:
                logger.warning(
                    "document_flush_request_failed",
                    extra_context={
                        "graph_id": graph_id,
                        "document_id": document_id,
                        "attempt": attempt,
                        "attempts": attempts,
                        "error": str(exc),
                    },
                )

            if attempt < attempts:
                await asyncio.sleep(0.12 * attempt)

        return False

    async def _get_location_via_rest(auth: MCPAuthContext, *, attempts: int = 2) -> Optional[dict]:
        """Fetch user's current location from the backend REST API.

        Returns dict with graph_id and document_id, or None if unavailable
        (endpoint not deployed yet — rolling deploy safety).
        Retries once on transient errors (timeout, 5xx).
        """
        url = f"{backend_config.base_url}/sessions/location"
        headers = auth.http_headers()
        last_error: Optional[str] = None
        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                    resp = await client.get(url, headers=headers)
                if resp.status_code == 404:
                    # Endpoint not yet deployed on this pod — fall back to WebSocket path
                    return None
                if resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                    if attempt < attempts:
                        logger.warning(
                            "session_location_retry",
                            extra_context={"status": resp.status_code, "attempt": attempt},
                        )
                        await asyncio.sleep(1.0)
                        continue
                    logger.warning(
                        "session_location_http_error",
                        extra_context={"status": resp.status_code, "attempts": attempts},
                    )
                    return None
                if resp.status_code != 200:
                    logger.warning(
                        "session_location_http_error",
                        extra_context={"status": resp.status_code},
                    )
                    return None
                payload = resp.json()
                graph_id = payload.get("graph_id")
                document_id = payload.get("document_id")
                source = payload.get("source", "unknown")
                if graph_id or document_id:
                    return {"graph_id": graph_id, "document_id": document_id}
                # REST succeeded but session has no location — return a sentinel
                # so the caller can report a clear error instead of falling through
                # to the slower WebSocket path.
                return {"graph_id": None, "document_id": None, "_empty": True, "_source": source}
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = str(exc)
                if attempt < attempts:
                    logger.warning(
                        "session_location_retry",
                        extra_context={"error": last_error, "attempt": attempt},
                    )
                    await asyncio.sleep(1.0)
                    continue
                logger.warning(
                    "session_location_request_failed",
                    extra_context={"error": last_error, "attempts": attempts},
                )
                return None
            except Exception as exc:
                logger.warning(
                    "session_location_request_failed",
                    extra_context={"error": str(exc)},
                )
                return None
        return None

    async def _await_document_durable(
        graph_id: str,
        document_id: str,
        user_id: str,
        *,
        attempts: int = 3,
        auth: Optional[MCPAuthContext] = None,
    ) -> bool:
        """Require a durable write checkpoint via a fresh document channel.

        This intentionally avoids same-channel verification:
        1. Tear down the current channel.
        2. Reconnect with force_fresh=True.
        3. Retry with short backoff on transient sync timeouts.

        If auth is provided, also calls backend flush endpoint for an
        immediate server-side persist checkpoint.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                # Give the event loop a moment to flush websocket send callbacks.
                await asyncio.sleep(0)
                await _connect_for_read(
                    graph_id,
                    document_id,
                    user_id,
                    force_fresh=True,
                    disconnect_first=True,
                )
                flush_confirmed = False
                if auth is not None:
                    flush_confirmed = await _force_flush_document_via_api(
                        auth,
                        graph_id,
                        document_id,
                        include_materialization=False,
                    )
                    if not flush_confirmed:
                        logger.warning(
                            "document_flush_unconfirmed",
                            extra_context={
                                "graph_id": graph_id,
                                "document_id": document_id,
                                "note": "fresh channel durable read succeeded, but flush endpoint did not confirm active session",
                            },
                        )
                return flush_confirmed
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    backoff = 0.15 * attempt
                    logger.warning(
                        "durable_read_retry",
                        extra_context={
                            "graph_id": graph_id,
                            "document_id": document_id,
                            "attempt": attempt,
                            "attempts": attempts,
                            "backoff_seconds": backoff,
                            "error": str(exc),
                        },
                    )
                    await asyncio.sleep(backoff)

        raise RuntimeError(
            f"Failed to establish fresh durability check channel for '{document_id}' "
            f"after {attempts} attempts: {last_error}"
        )

    @server.tool(
        name="get_user_location",
        title="Get Current Graph and Document",
        description=(
            "Returns the graph ID and document ID the user is currently viewing. "
            "This is the lightest-weight context tool — use it when you just need to know "
            "where the user is. Follow up with get_workspace to see the full graph structure, "
            "or read_document to see the document content."
        ),
    )
    async def get_user_location_tool(
        user_id: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Get the user's current graph and document IDs."""
        auth = MCPAuthContext.from_context(context)
        token = auth.require_auth()
        user_id = _resolve_user_id(auth, token, user_id)

        try:
            # Try REST endpoint first — reads from in-memory session on the server
            # pod, always fresh. Falls back to WebSocket reconnect if not deployed.
            location = await _get_location_via_rest(auth)
            if location is not None:
                if location.get("_empty"):
                    # REST succeeded but session has no active location
                    rest_source = location.get("_source", "unknown")
                    logger.warning(
                        "user_location_empty_from_rest",
                        extra_context={
                            "user_id": user_id,
                            "auth_source": auth.source,
                            "session_source": rest_source,
                        },
                    )
                    raise RuntimeError(
                        f"No active graph or document found in session for user '{user_id}' "
                        f"(session loaded from {rest_source}). "
                        f"The user may not have a browser tab open, or hasn't navigated "
                        f"to a document yet. Auth source: {auth.source}"
                    )
                return location

            # Fallback: reconnect WebSocket to get fresh session state
            # (the persistent WebSocket doesn't receive incremental Y.js updates
            # after initial sync, so reconnect gets latest from server memory/Redis)
            # Retry once on transient WebSocket/sync timeout.
            ws_error: Optional[Exception] = None
            for ws_attempt in range(1, 3):
                try:
                    await hp_client.refresh_session(user_id)
                    ws_error = None
                    break
                except Exception as e:
                    ws_error = e
                    if ws_attempt < 2:
                        logger.warning(
                            "session_refresh_retry",
                            extra_context={"error": str(e), "attempt": ws_attempt},
                        )
                        await asyncio.sleep(1.0)
            if ws_error is not None:
                raise ws_error

            graph_id = hp_client.get_active_graph_id()
            document_id = hp_client.get_active_document_id()

            if not graph_id and not document_id:
                logger.warning(
                    "user_location_empty",
                    extra_context={
                        "user_id": user_id,
                        "auth_source": auth.source,
                    },
                )
                raise RuntimeError(
                    f"No active session found for user '{user_id}'. "
                    f"The user may not have a browser tab open, or the session "
                    f"has not synced yet. Auth source: {auth.source}"
                )

            return {"graph_id": graph_id, "document_id": document_id}

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(
                "Failed to get user location",
                extra_context={"error": str(e), "user_id": user_id, "auth_source": auth.source},
            )
            raise RuntimeError(f"Failed to get user location: {e}")

    @server.tool(
        name="get_session_state",
        title="Get Full Session State",
        description=(
            "Returns the complete session state including all tabs, preferences, and UI settings. "
            "WARNING: This returns a large payload. Prefer get_user_location (for current graph/document) "
            "or get_workspace (for graph structure) unless you specifically need full session details."
        ),
    )
    async def get_session_state_tool(
        user_id: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Get the full session state (tabs, preferences, UI settings)."""
        auth = MCPAuthContext.from_context(context)
        token = auth.require_auth()
        user_id = _resolve_user_id(auth, token, user_id)

        try:
            await hp_client.refresh_session(user_id)

            active_graph = hp_client.get_active_graph_id()
            active_doc = hp_client.get_active_document_id()
            session_snapshot = hp_client.get_session_snapshot()

            result = {
                "active_graph_id": active_graph,
                "active_document_id": active_doc,
                "session": session_snapshot,
            }
            return result

        except Exception as e:
            logger.error(
                "Failed to get session state",
                extra_context={"error": str(e)},
            )
            raise RuntimeError(f"Failed to get session state: {e}")

    @server.tool(
        name="read_document",
        title="Read Document Content",
        description="""Reads document content with wire counts. Supports multiple output formats.

Formats (set via 'format' parameter):
- default (None): TipTap XML with full formatting and data-block-id attributes on every block. Use this when you need block IDs for surgical editing (edit_block_text, update_blocks, insert_block, delete_block) or block-level wire connections.
- 'markdown': Clean Markdown. Use this when you just need to read/understand a document's content without editing it. Much more compact than XML.
- 'ids_only': Returns just the ordered list of block IDs and count, no content. Use this when you already know the content but need block IDs for wiring or editing.

XML block types: paragraph, heading (level="1-3"), bulletList, orderedList, blockquote, codeBlock (language="..."), taskList (taskItem checked="true"), horizontalRule, image (src="...", alt="...")
XML marks (nestable): strong, em, strike, code, mark (highlight), a (href="..."), footnote (data-footnote-content="..."), commentMark (data-comment-id="...")

Also returns wire counts: document-level (outgoing, incoming, total) and block-level (which blocks have wires attached). Use get_wires for full wire details.

Works for all documents including uploaded files (which are documents with readOnly=true).

Always returns fresh content — automatically reconnects if the cached channel is older than 2 seconds, and retries once on sync timeout. Safe for multi-agent use where another agent may have written to this document.""",
    )
    async def read_document_tool(
        graph_id: str,
        document_id: str,
        format: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Read document content in the requested format, with wire counts.

        Args:
            graph_id: The graph containing the document
            document_id: The document to read
            format: Output format - None/"xml" (default, includes block IDs) or "markdown" (compact, readable)
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if format and format not in ("xml", "markdown", "ids_only"):
            raise ValueError("format must be 'xml', 'markdown', or 'ids_only'")

        try:
            # Validate document exists in workspace before connecting
            await _validate_document_in_workspace(graph_id, document_id, auth.user_id, auth=auth)

            document_doc, _ = await _load_document_doc_for_read(
                graph_id,
                document_id,
                auth.user_id,
                auth=auth,
                tool_name="read_document",
            )
            reader = DocumentReader(document_doc)
            xml_content = reader.to_xml()
            comments = reader.get_all_comments()

            # ids_only: return just the ordered list of block IDs, no content
            if format == "ids_only":
                fragment = reader.get_content_fragment()
                block_ids = []
                for child in fragment.children:
                    if hasattr(child, "attributes"):
                        bid = child.attributes.get("data-block-id")
                        if bid:
                            block_ids.append(bid)
                return {
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "format": "ids_only",
                    "block_ids": block_ids,
                    "block_count": len(block_ids),
                }

            # Convert to requested format
            if format == "markdown":
                content = tiptap_xml_to_markdown(xml_content)
            elif format == "html":
                title = extract_title_from_xml(xml_content) or document_id
                content = tiptap_xml_to_html(xml_content, title=title, themed=True)
            else:
                content = xml_content

            result = {
                "graph_id": graph_id,
                "document_id": document_id,
                "format": format or "xml",
                "content": content,
                "comments": comments,
            }

            # Add wire counts — overall totals + per-block breakdown
            try:
                workspace_doc, _ = await _load_workspace_doc_for_read(
                    graph_id,
                    auth.user_id,
                    auth=auth,
                    tool_name="read_document",
                )
                outgoing = _get_wires_for_document(workspace_doc, document_id, "outgoing")
                incoming = _get_wires_for_document(workspace_doc, document_id, "incoming")
                total = len(outgoing) + len(incoming)
                if total > 0:
                    block_counts: Dict[str, int] = {}
                    for wire in outgoing:
                        bid = wire.get("sourceBlockId")
                        if bid:
                            block_counts[bid] = block_counts.get(bid, 0) + 1
                    for wire in incoming:
                        bid = wire.get("targetBlockId")
                        if bid:
                            block_counts[bid] = block_counts.get(bid, 0) + 1
                    wires_info: Dict[str, Any] = {
                        "outgoing": len(outgoing),
                        "incoming": len(incoming),
                        "total": total,
                    }
                    if block_counts:
                        wires_info["by_block"] = block_counts
                    result["wires"] = wires_info
            except Exception as e:
                logger.debug(
                    "Failed to fetch wire counts for read_document (non-fatal)",
                    extra_context={"document_id": document_id, "error": str(e)},
                )

            return result

        except Exception as e:
            logger.error(
                "Failed to read document",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to read document: {e}")

    @server.tool(
        name="read_blocks",
        title="Read Blocks (Paginated)",
        description=(
            "Read a document's blocks sequentially with pagination. Returns blocks as "
            "a list with their content, type, block ID, and index. Use offset and limit "
            "to paginate through large documents.\n\n"
            "Pair with document_digest to understand document size before reading. "
            "For example: digest tells you a book has 500 blocks, then read_blocks "
            "with offset=0, limit=50 reads the first 50 blocks.\n\n"
            "Output format is controlled by the format parameter:\n"
            "- 'markdown' (default): Clean markdown rendering of each block\n"
            "- 'text': Plain text only (most compact)\n"
            "- 'xml': Full TipTap XML with attributes and block IDs\n\n"
            "Returns: blocks list, total_blocks count, has_more flag, and "
            "next_offset for easy pagination.\n\n"
            "Always returns fresh content — automatically reconnects if cached state "
            "is older than 2 seconds, and retries once on sync timeout."
        ),
    )
    async def read_blocks_tool(
        graph_id: str,
        document_id: str,
        offset: int = 0,
        limit: int = 50,
        format: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Read blocks sequentially from a document with pagination.

        Args:
            graph_id: The graph containing the document
            document_id: The document to read
            offset: Block index to start from (0-based, default 0)
            limit: Maximum number of blocks to return (default 50, max 200)
            format: Output format - 'markdown' (default), 'text', or 'xml'
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if limit > 200:
            limit = 200
        if offset < 0:
            offset = 0
        fmt = format or "markdown"
        if fmt not in ("markdown", "text", "xml"):
            raise ValueError("format must be 'markdown', 'text', or 'xml'")

        try:
            await _validate_document_in_workspace(graph_id, document_id, auth.user_id, auth=auth)

            document_doc, _ = await _load_document_doc_for_read(
                graph_id,
                document_id,
                auth.user_id,
                auth=auth,
                tool_name="read_blocks",
            )
            reader = DocumentReader(document_doc)
            fragment = reader.get_content_fragment()
            all_children = list(fragment.children)
            total_blocks = len(all_children)

            # Slice to requested range
            end = min(offset + limit, total_blocks)
            slice_children = all_children[offset:end]

            blocks = []
            for i, child in enumerate(slice_children):
                idx = offset + i
                if not hasattr(child, "attributes"):
                    continue

                attrs = dict(child.attributes)
                block_id = attrs.get("data-block-id", "")
                tag = child.tag if hasattr(child, "tag") else "unknown"

                # Get content in requested format
                xml_str = reader._serialize_element(child)

                if fmt == "xml":
                    content = xml_str
                elif fmt == "text":
                    plain = re.sub(r"<[^>]+>", "", xml_str)
                    content = plain.strip()
                else:  # markdown
                    content = tiptap_xml_to_markdown(xml_str).strip()

                block_entry: Dict[str, Any] = {
                    "index": idx,
                    "block_id": block_id,
                    "type": tag,
                    "content": content,
                }

                # Include heading level for headings
                if tag == "heading":
                    level = attrs.get("level", 1)
                    if isinstance(level, float):
                        level = int(level)
                    block_entry["level"] = level

                blocks.append(block_entry)

            has_more = end < total_blocks
            result: Dict[str, Any] = {
                "graph_id": graph_id,
                "document_id": document_id,
                "blocks": blocks,
                "total_blocks": total_blocks,
                "offset": offset,
                "limit": limit,
                "returned": len(blocks),
                "has_more": has_more,
            }
            if has_more:
                result["next_offset"] = end

            return result

        except Exception as e:
            logger.error(
                "Failed to read blocks",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "offset": offset,
                    "limit": limit,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to read blocks: {e}")

    @server.tool(
        name="document_digest",
        title="Document Digest",
        description=(
            "Returns a compact summary of a document for orientation and triage, "
            "without fetching full content. Includes:\n\n"
            "- **metadata**: title, folder path, readOnly status\n"
            "- **size**: block count, character count, word count\n"
            "- **freshness**: created_at timestamp\n"
            "- **headings**: section headings extracted from document structure\n"
            "- **wire_summary**: incoming/outgoing counts, predicate distribution, "
            "top connected documents\n"
            "- **valuation_summary**: top valued blocks with scores and excerpts\n\n"
            "Use this to decide whether a full read_document is needed. Much cheaper "
            "than reading the full document when you only need to understand what it "
            "contains and how it connects.\n\n"
            "Always returns fresh content — automatically reconnects if cached state "
            "is older than 2 seconds, and retries once on sync timeout."
        ),
    )
    async def document_digest_tool(
        graph_id: str,
        document_id: str,
        top_valued: int = 3,
        context: Context | None = None,
    ) -> dict:
        """Return a compact digest of a document.

        Args:
            graph_id: The graph containing the document
            document_id: The document to digest
            top_valued: Number of top-valued blocks to include (default 3)
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        try:
            # 1. Read workspace metadata (direct path with fallback)
            workspace_doc, _ = await _load_workspace_doc_for_read(
                graph_id,
                auth.user_id,
                auth=auth,
                tool_name="document_digest",
            )
            ws_reader = WorkspaceReader(workspace_doc)
            doc_meta = ws_reader.get_document(document_id)
            if doc_meta is None:
                raise RuntimeError(
                    f"Document '{document_id}' not found in graph '{graph_id}'. "
                    f"Use get_workspace to see available documents."
                )

            # Resolve folder path
            folder_path = None
            parent_id = doc_meta.get("parentId")
            if parent_id:
                path_parts = []
                current = parent_id
                seen = set()
                while current and current not in seen:
                    seen.add(current)
                    folder = ws_reader.get_folder(current)
                    if folder:
                        path_parts.append(folder.get("label", current))
                        current = folder.get("parentId")
                    else:
                        break
                path_parts.reverse()
                folder_path = "/".join(path_parts)

            metadata = {
                "title": doc_meta.get("title", document_id),
                "document_id": document_id,
                "folder_path": folder_path,
                "readOnly": doc_meta.get("readOnly", False),
            }
            if doc_meta.get("fileType"):
                metadata["fileType"] = doc_meta["fileType"]

            # 2. Read document and extract size + headings
            document_doc, _ = await _load_document_doc_for_read(
                graph_id,
                document_id,
                auth.user_id,
                auth=auth,
                tool_name="document_digest",
            )
            reader = DocumentReader(document_doc)
            fragment = reader.get_content_fragment()

            block_count = 0
            char_count = 0
            word_count = 0
            headings = []

            def _count_text(elem: Any) -> str:
                """Recursively extract plain text from a Y.js element."""
                if isinstance(elem, pycrdt.XmlText):
                    return str(elem)
                elif isinstance(elem, pycrdt.XmlElement):
                    parts = []
                    for child in elem.children:
                        parts.append(_count_text(child))
                    return "".join(parts)
                return ""

            for child in fragment.children:
                if isinstance(child, pycrdt.XmlElement):
                    block_count += 1
                    text = _count_text(child)
                    char_count += len(text)
                    word_count += len(text.split()) if text.strip() else 0

                    # Extract headings
                    tag = child.tag
                    if tag == "heading":
                        attrs = dict(child.attributes)
                        level = attrs.get("level", 1)
                        if isinstance(level, float):
                            level = int(level)
                        headings.append({
                            "level": level,
                            "text": text.strip()[:120],
                        })

            size = {
                "block_count": block_count,
                "char_count": char_count,
                "word_count": word_count,
            }

            # 3. Wire summary from workspace
            wire_summary = None
            try:
                outgoing = _get_wires_for_document(workspace_doc, document_id, "outgoing")
                incoming = _get_wires_for_document(workspace_doc, document_id, "incoming")
                total = len(outgoing) + len(incoming)
                if total > 0:
                    # Predicate distribution
                    pred_counts: Dict[str, int] = {}
                    for w in outgoing + incoming:
                        pred = _get_predicate_short_name(w.get("predicate", "unknown"))
                        pred_counts[pred] = pred_counts.get(pred, 0) + 1

                    # Top connected docs (by wire count)
                    neighbor_counts: Dict[str, int] = {}
                    neighbor_titles: Dict[str, str] = {}
                    for w in outgoing:
                        tid = w.get("targetDocumentId", "")
                        neighbor_counts[tid] = neighbor_counts.get(tid, 0) + 1
                        if w.get("targetTitle"):
                            neighbor_titles[tid] = w["targetTitle"]
                    for w in incoming:
                        sid = w.get("sourceDocumentId", "")
                        neighbor_counts[sid] = neighbor_counts.get(sid, 0) + 1
                        if w.get("sourceTitle"):
                            neighbor_titles[sid] = w["sourceTitle"]

                    # Top 5 neighbors by wire count
                    top_neighbors = sorted(
                        neighbor_counts.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    top_connected = []
                    for nid, count in top_neighbors:
                        title = neighbor_titles.get(nid)
                        if not title:
                            ndoc = ws_reader.get_document(nid)
                            title = ndoc.get("title", nid) if ndoc else nid
                        top_connected.append({"id": nid, "title": title, "wires": count})

                    wire_summary = {
                        "outgoing": len(outgoing),
                        "incoming": len(incoming),
                        "total": total,
                        "predicates": pred_counts,
                        "top_connected": top_connected,
                    }
            except Exception as e:
                logger.debug(
                    "Failed to fetch wire summary for digest (non-fatal)",
                    extra_context={"document_id": document_id, "error": str(e)},
                )

            # 4. Valuation summary (top valued blocks via SPARQL)
            valuation_summary = None
            if top_valued > 0:
                try:
                    user_id = auth.user_id or get_dev_user_id() or get_user_id_from_token(auth.token)
                    doc_prefix = (
                        f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"
                        f":valuation:{document_id}:"
                    )
                    query = f"""
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?val ?blockRef ?cumImp ?cumVal
WHERE {{
  ?val doc:blockRef ?blockRef .
  ?val doc:cumulativeImportance ?cumImp .
  ?val doc:cumulativeValence ?cumVal .
  FILTER(STRSTARTS(STR(?val), "{doc_prefix}"))
}}
ORDER BY DESC(xsd:float(?cumImp))
LIMIT {top_valued}
"""
                    from neem.mcp.tools.geist import _sparql_query as _geist_sparql_query
                    rows = await _geist_sparql_query(backend_config, job_stream, auth, graph_id, query)

                    if rows:
                        valued_blocks = []
                        for row in rows:
                            block_ref = row.get("blockRef", "")
                            block_id = block_ref.split("#block-")[-1] if "#block-" in block_ref else ""
                            importance = float(row.get("cumImp", 0))
                            valence_val = float(row.get("cumVal", 0))

                            # Get block text excerpt
                            excerpt = ""
                            if block_id:
                                try:
                                    for child in fragment.children:
                                        if isinstance(child, pycrdt.XmlElement):
                                            bid = dict(child.attributes).get("data-block-id")
                                            if bid == block_id:
                                                excerpt = _count_text(child).strip()[:150]
                                                break
                                except Exception:
                                    pass

                            valued_blocks.append({
                                "block_id": block_id,
                                "importance": round(importance, 2),
                                "valence": round(valence_val, 2),
                                "excerpt": excerpt,
                            })
                        valuation_summary = valued_blocks
                except Exception as e:
                    logger.debug(
                        "Failed to fetch valuations for digest (non-fatal)",
                        extra_context={"document_id": document_id, "error": str(e)},
                    )

            # 5. Build result
            result: Dict[str, Any] = {
                "metadata": metadata,
                "size": size,
            }
            if doc_meta.get("createdAt"):
                result["freshness"] = {"created_at": doc_meta["createdAt"]}
            if headings:
                result["headings"] = headings
            if wire_summary:
                result["wire_summary"] = wire_summary
            if valuation_summary:
                result["valuation_summary"] = valuation_summary

            return result

        except Exception as e:
            logger.error(
                "Failed to create document digest",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to create document digest: {e}")

    @server.tool(
        name="write_document",
        title="Write Document Content",
        description="""Replaces document content with TipTap XML. Syncs to UI in real-time.

WARNING: This REPLACES all content. For collaborative editing, prefer append_to_document.

Plain text is accepted: if the content doesn't start with '<', each paragraph (separated by blank lines) is auto-wrapped in <paragraph> tags. Use XML when you need formatting or specific block types.

Blocks: paragraph, heading (level="1-3"), bulletList, orderedList, blockquote, codeBlock (language="..."), taskList (taskItem checked="true"), horizontalRule, image (src="...", alt="...")
Marks (nestable): strong, em, strike, code, mark (highlight), a (href="..."), footnote (data-footnote-content="..."), commentMark (data-comment-id="...")
Example: <paragraph>Text with <mark>highlight</mark> and a note<footnote data-footnote-content="This is a footnote"/></paragraph>

IMPORTANT: Container blocks (blockquote, tableCell, tableHeader) require paragraph children — they cannot contain inline text directly. Auto-wrapping is applied as a fallback, but prefer explicit wrapping: <blockquote><paragraph>text</paragraph></blockquote>

Comments: Pass a dict mapping comment IDs to metadata. Comment IDs must match data-comment-id attributes in the content.
Example comments: {"comment-1": {"text": "Great point!", "author": "Claude"}}

Markdown is also accepted and auto-converted to TipTap XML.

Returns block_ids: an ordered list of all block IDs in the written document, enabling immediate block-level wiring without a separate read call.

`await_durable` (default true) forces post-write verification through a fresh document channel rather than the same cached channel.

NOT for: editing existing documents (use edit_block_text, update_blocks, or insert_block instead). Only use write_document for brand-new documents or when the user explicitly asks for a full rewrite.

Write tools use a persistent cached channel (no automatic reconnect like read tools). In multi-agent environments, always call read_document first to get current content before writing — this ensures your channel has the latest state from the server. CRDT merge prevents data corruption, but writing without reading first may silently overwrite another agent's recent changes.""",
    )
    async def write_document_tool(
        graph_id: str,
        document_id: str,
        content: str,
        comments: Optional[Dict[str, Any]] = None,
        await_durable: bool = True,
        context: Context | None = None,
    ) -> dict:
        """Write TipTap XML content to a document."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        try:
            # 1. Write document content and comments (with user context)
            await hp_client.connect_document(graph_id, document_id, user_id=auth.user_id)

            xml_content = _ensure_xml_multiblock(content)

            def write_content_and_comments(doc: Any) -> None:
                writer = DocumentWriter(doc)
                writer.replace_all_content(xml_content)
                # Write comments if provided
                if comments:
                    for comment_id, comment_data in comments.items():
                        writer.set_comment(
                            comment_id=comment_id,
                            text=comment_data.get("text", ""),
                            author=comment_data.get("author", "MCP Agent"),
                            author_id=comment_data.get("authorId", "mcp-agent"),
                            resolved=comment_data.get("resolved", False),
                            quoted_text=comment_data.get("quotedText"),
                        )

            await hp_client.transact_document(
                graph_id,
                document_id,
                write_content_and_comments,
                user_id=auth.user_id,
            )

            # Collect block IDs from the written document
            channel = hp_client.get_document_channel(graph_id, document_id, user_id=auth.user_id)
            block_ids = []
            if channel:
                reader = DocumentReader(channel.doc)
                fragment = reader.get_content_fragment()
                for child in fragment.children:
                    if hasattr(child, "attributes"):
                        bid = child.attributes.get("data-block-id")
                        if bid:
                            block_ids.append(bid)

            # 2. Verify content persisted (detect tombstoned document IDs)
            # Optionally force a fresh channel so verification does not read from
            # the same in-memory channel that just performed the write.
            expected_count = len(block_ids)
            if expected_count > 0:
                if await_durable:
                    await _await_document_durable(
                        graph_id,
                        document_id,
                        auth.user_id,
                        auth=auth,
                    )
                else:
                    await _connect_for_read(graph_id, document_id, auth.user_id)
                verify_channel = hp_client.get_document_channel(
                    graph_id, document_id, user_id=auth.user_id,
                )
                actual_count = 0
                if verify_channel:
                    verify_reader = DocumentReader(verify_channel.doc)
                    verify_frag = verify_reader.get_content_fragment()
                    actual_count = len([
                        c for c in verify_frag.children
                        if hasattr(c, "attributes") and c.attributes.get("data-block-id")
                    ])
                if actual_count < expected_count:
                    raise RuntimeError(
                        f"Write to '{document_id}' failed: wrote {expected_count} blocks "
                        f"but only {actual_count} persisted. This document ID may be "
                        f"tombstoned from a previous deletion — try a different document ID."
                    )

            # 3. Update workspace navigation so document appears in file tree
            # Extract title from first heading, fallback to document_id
            title = extract_title_from_xml(xml_content) or document_id
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            await hp_client.transact_workspace(
                graph_id,
                lambda doc: WorkspaceWriter(doc).upsert_document(document_id, title),
                user_id=auth.user_id,
            )

            return {
                "success": True,
                "graph_id": graph_id,
                "document_id": document_id,
                "title": title,
                "block_ids": block_ids,
                "durability_checked": await_durable,
            }

        except Exception as e:
            logger.error(
                "Failed to write document",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to write document: {e}")

    @server.tool(
        name="append_to_document",
        title="Append Block to Document",
        description=(
            "Appends one or more blocks to the end of a document. Accepts TipTap XML, markdown, "
            "or plain text. Use this for incremental additions without replacing existing content.\n\n"
            "Supports multiple blocks in a single call: pass markdown with multiple paragraphs, "
            "or XML with multiple top-level elements. Each block is appended in order within a "
            "single transaction. Plain text without XML tags is auto-wrapped in a <paragraph>.\n\n"
            "Container blocks (blockquote, tableCell, tableHeader) require paragraph children. "
            "Auto-wrapping is applied as a fallback, but prefer: <blockquote><paragraph>text</paragraph></blockquote>\n\n"
            "For appending to documents written by other agents, call read_document first to sync "
            "the channel — otherwise the append may conflict with content you haven't seen yet."
        ),
    )
    async def append_to_document_tool(
        graph_id: str,
        document_id: str,
        text: str,
        context: Context | None = None,
    ) -> dict:
        """Append one or more blocks to a document.

        Accepts markdown or TipTap XML with multiple blocks and automatically
        breaks them into individual append operations within a single transaction.

        Args:
            graph_id: The graph containing the document
            document_id: The document to append to
            text: Content to append. Can be:
                - Plain text (wrapped in <paragraph>)
                - Markdown (converted to TipTap XML)
                - TipTap XML (single or multiple blocks)
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not text:
            raise ValueError("text is required")

        try:
            # Validate document exists in workspace before appending
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id, auth=auth)

            # Connect to the document channel with user context
            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            # Convert to XML, handling multiple blocks
            content = _ensure_xml_multiblock(text)

            # Parse to extract individual block elements
            # Wrap in a root element to handle multiple top-level blocks
            blocks_xml: list[str] = []
            try:
                wrapped = f"<root>{content}</root>"
                root = ET.fromstring(wrapped)
                blocks = list(root)  # Extract all child elements
            except ET.ParseError:
                # If parsing fails, treat as single block (fallback to old behavior)
                blocks_xml = [content]
            else:
                # Convert each element back to XML string
                blocks_xml = [ET.tostring(block, encoding='unicode') for block in blocks]

            new_block_ids: list[str] = []

            def perform_append(doc: Any) -> None:
                nonlocal new_block_ids
                writer = DocumentWriter(doc)
                reader = DocumentReader(doc)

                for block_xml in blocks_xml:
                    writer.append_block(block_xml)
                    # Get the last block's ID after each append
                    count = reader.get_block_count()
                    if count > 0:
                        block = reader.get_block_at(count - 1)
                        if block and hasattr(block, "attributes"):
                            attrs = block.attributes
                            block_id = attrs.get("data-block-id") if "data-block-id" in attrs else ""
                            if block_id:
                                new_block_ids.append(block_id)

            await hp_client.transact_document(
                graph_id.strip(),
                document_id.strip(),
                perform_append,
                user_id=auth.user_id,
            )

            result = {
                "success": True,
                "graph_id": graph_id.strip(),
                "document_id": document_id.strip(),
                "new_block_id": new_block_ids[-1] if new_block_ids else "",
                "block_ids": new_block_ids,  # Return all appended block IDs
                "blocks_appended": len(new_block_ids),
            }
            return result

        except Exception as e:
            logger.error(
                "Failed to append to document",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to append to document: {e}")

    def _build_workspace_tree(
        snapshot: dict[str, Any],
        max_depth: int = 0,
        folder_id: str | None = None,
        excluded_doc_ids: set[str] | None = None,
        valued_doc_ids: set[str] | None = None,
        folders_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Transform flat workspace snapshot into a nested tree.

        Converts the flat {folders, documents} maps with parentId pointers
        into a pre-computed nested tree. Drops metadata the LLM doesn't
        need (timestamps, sort orders, storage paths, UI state).
        Uploaded files appear as documents with readOnly=true and fileType.

        Args:
            snapshot: Raw workspace snapshot with folders/documents maps.
            max_depth: Maximum nesting depth (0 = unlimited). At the limit,
                folders are collapsed to show child counts instead of contents.
            folder_id: If set, only return the subtree rooted at this folder.
            excluded_doc_ids: Document IDs to exclude (e.g. filtered by score).
            valued_doc_ids: Document IDs that have valuations (for enriched
                collapse counts showing "N documents, M valued").
            folders_only: If True, return only the folder hierarchy with
                document counts per folder. No individual documents listed.
        """
        folders = snapshot.get("folders", {})
        documents = snapshot.get("documents", {})

        # Build node for each entity
        nodes: dict[str, dict[str, Any]] = {}

        for fid, fdata in folders.items():
            node: dict[str, Any] = {"id": fid, "type": "folder", "name": fdata.get("name", "Untitled")}
            node["_parent"] = fdata.get("parentId")
            node["_order"] = fdata.get("order", 0)
            node["children"] = []
            nodes[fid] = node

        if not folders_only:
            for did, ddata in documents.items():
                if excluded_doc_ids and did in excluded_doc_ids:
                    continue
                node = {"id": did, "type": "document", "title": ddata.get("title", "Untitled")}
                if ddata.get("readOnly"):
                    node["readOnly"] = True
                sf_file_type = ddata.get("sf_fileType")
                if sf_file_type:
                    node["fileType"] = sf_file_type
                node["_parent"] = ddata.get("parentId")
                node["_order"] = ddata.get("order", 0)
                nodes[did] = node

        # Build tree by inserting children into parent folders
        root: list[dict[str, Any]] = []
        for nid, node in nodes.items():
            parent_id = node.pop("_parent", None)
            if parent_id and parent_id in nodes and "children" in nodes[parent_id]:
                nodes[parent_id]["children"].append(node)
            else:
                root.append(node)

        # In folders_only mode, annotate each folder with document counts
        # from the raw snapshot (documents weren't added as tree nodes).
        docs_per_folder: dict[str | None, int] = {}
        if folders_only:
            # Count direct documents per folder, and unfiled (root-level) docs
            for did, ddata in documents.items():
                if excluded_doc_ids and did in excluded_doc_ids:
                    continue
                pid = ddata.get("parentId")
                docs_per_folder[pid] = docs_per_folder.get(pid, 0) + 1

            def _annotate_doc_counts(items: list[dict[str, Any]]) -> None:
                for item in items:
                    fid = item.get("id")
                    direct = docs_per_folder.get(fid, 0)
                    if direct > 0:
                        item["documents"] = direct
                    if "children" in item:
                        _annotate_doc_counts(item["children"])

            _annotate_doc_counts(root)

            def _count_folder_docs(folder_node: dict[str, Any]) -> int:
                """Count documents under a folder using raw snapshot counts."""
                total = docs_per_folder.get(folder_node.get("id"), 0)
                for child in folder_node.get("children", []):
                    if child.get("type") == "folder":
                        total += _count_folder_docs(child)
                return total

        # Sort children by order, then strip internal _order keys
        # Apply depth truncation: at max_depth, collapse folders to counts
        def _sort_and_clean(items: list[dict[str, Any]], current_depth: int = 1) -> list[dict[str, Any]]:
            items.sort(key=lambda x: x.get("_order", 0))
            for item in items:
                item.pop("_order", None)
                if "children" in item:
                    if max_depth > 0 and current_depth >= max_depth:
                        # Collapse: count children recursively instead of listing them
                        if folders_only:
                            doc_count = _count_folder_docs(item)
                            val_count = 0
                        else:
                            doc_count, val_count = _count_descendants(item["children"])
                        del item["children"]
                        if doc_count > 0:
                            if valued_doc_ids and val_count > 0:
                                item["collapsed"] = f"{doc_count} documents, {val_count} valued"
                            else:
                                item["collapsed"] = f"{doc_count} documents"
                    else:
                        _sort_and_clean(item["children"], current_depth + 1)
                        if not item["children"]:
                            del item["children"]
            return items

        def _count_descendants(items: list[dict[str, Any]]) -> tuple[int, int]:
            """Count documents and valued documents in a subtree."""
            doc_count = 0
            val_count = 0
            for item in items:
                if item.get("type") == "document":
                    doc_count += 1
                    if valued_doc_ids and item.get("id") in valued_doc_ids:
                        val_count += 1
                if "children" in item:
                    sub_docs, sub_vals = _count_descendants(item["children"])
                    doc_count += sub_docs
                    val_count += sub_vals
            return doc_count, val_count

        tree = _sort_and_clean(root)

        # If folder_id is specified, extract just that subtree
        if folder_id:
            def _find_folder(items: list[dict[str, Any]], target_id: str) -> dict[str, Any] | None:
                for item in items:
                    if item.get("id") == target_id:
                        return item
                    if "children" in item:
                        found = _find_folder(item["children"], target_id)
                        if found:
                            return found
                return None

            folder_node = _find_folder(tree, folder_id)
            if folder_node and "children" in folder_node:
                tree = folder_node["children"]
            elif folder_node:
                tree = [folder_node]
            else:
                tree = []

        return tree

    @server.tool(
        name="get_workspace",
        title="Get Workspace Structure",
        description=(
            "Returns the folder and file structure of a graph's workspace, including all documents, "
            "folders, and their titles and organization. This is the primary tool for "
            "understanding what's in a graph. Use get_user_location first if you need to know which "
            "graph the user is in.\n\n"
            "**Parameters:**\n"
            "- depth (default 1): Maximum folder nesting depth. At the limit, folders collapse to "
            "show document counts instead of full listings. Use depth=0 for unlimited (full tree). "
            "Organize documents into folders for cleaner workspace views at default depth.\n"
            "- folder_id (optional): Return only the subtree under this folder. Useful for "
            "drilling into a specific area after seeing the top-level structure.\n"
            "- min_score (optional): Filter out documents with a document-level composite score below "
            "this threshold. Document scores are computed from block-level valuations (avg importance, "
            "avg valence). Only applies to documents that have been valuated; unscored documents "
            "are always shown.\n"
            "- folders_only (default false): Return only the folder hierarchy with document counts "
            "per folder. No individual documents listed. Useful for understanding graph organization "
            "without the full document list.\n\n"
            "This is always complete at the requested depth — prefer it over sparql_query for "
            "discovering what documents exist."
        ),
    )
    async def get_workspace_tool(
        graph_id: str,
        depth: int = 1,
        folder_id: Optional[str] = None,
        min_score: Optional[float] = None,
        folders_only: bool = False,
        context: Context | None = None,
    ) -> str:
        """Get workspace folder structure as a nested tree."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        try:
            workspace_doc, _ = await _load_workspace_doc_for_read(
                graph_id,
                auth.user_id,
                auth=auth,
                tool_name="get_workspace",
            )
            snapshot = _workspace_snapshot_from_doc(workspace_doc)

            has_docs = bool(snapshot.get("documents"))
            has_folders = bool(snapshot.get("folders"))
            if not has_docs and not has_folders:
                raise RuntimeError(
                    f"Graph '{graph_id}' not found — workspace is completely empty "
                    f"(no documents or folders). This usually means the "
                    f"graph ID is wrong or you're connected to the wrong backend. "
                    f"Use list_graphs to see available graphs."
                )

            # If min_score is set, query document-level scores and build exclusion set
            excluded_doc_ids: set[str] | None = None
            valued_doc_ids: set[str] | None = None
            if min_score is not None:
                excluded_doc_ids, valued_doc_ids = await _get_excluded_docs_by_score(
                    graph_id, min_score, auth
                )

            tree = _build_workspace_tree(
                snapshot,
                max_depth=depth,
                folder_id=folder_id,
                excluded_doc_ids=excluded_doc_ids,
                valued_doc_ids=valued_doc_ids,
                folders_only=folders_only,
            )
            result: dict[str, Any] = {"graph_id": graph_id, "tree": tree}
            if depth > 0:
                result["depth"] = depth
            if folder_id:
                result["folder_id"] = folder_id
            if folders_only:
                result["folders_only"] = True
                # Count unfiled (root-level) documents
                unfiled = sum(
                    1 for did, d in snapshot.get("documents", {}).items()
                    if not d.get("parentId")
                    and not (excluded_doc_ids and did in excluded_doc_ids)
                )
                if unfiled > 0:
                    result["unfiled_documents"] = unfiled
            return json.dumps(result)

        except Exception as e:
            logger.error(
                "Failed to get workspace",
                extra_context={"graph_id": graph_id, "error": str(e)},
            )
            raise RuntimeError(f"Failed to get workspace: {e}")

    # -------------------------------------------------------------------------
    # Folder Operations (Y.js-based, replacing HTTP job-based navigation.py)
    # -------------------------------------------------------------------------

    @server.tool(
        name="create_folder",
        title="Create Folder",
        description=(
            "Create a new folder in the workspace. "
            "Use parent_id to nest inside another folder (null for root level). "
            "The section parameter determines which sidebar section the folder appears in."
        ),
    )
    async def create_folder_tool(
        graph_id: str,
        folder_id: str,
        label: str,
        parent_id: Optional[str] = None,
        order: Optional[float] = None,
        section: str = "documents",
        context: Context | None = None,
    ) -> dict:
        """Create a new folder in the workspace via Y.js."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")
        if not label or not label.strip():
            raise ValueError("label is required and cannot be empty")
        if section not in ("documents", "artifacts"):
            raise ValueError("section must be 'documents' or 'artifacts'")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Create folder via Y.js transact
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).upsert_folder(
                    folder_id.strip(),
                    label.strip(),  # 'label' param → 'name' in Y.js
                    parent_id=parent_id.strip() if parent_id else None,
                    section=section,
                    order=order,
                ),
                user_id=auth.user_id,
            )

            return {
                "success": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "label": label.strip(),
                "parent_id": parent_id.strip() if parent_id else None,
                "section": section,
            }

        except Exception as e:
            logger.error(
                "Failed to create folder",
                extra_context={
                    "graph_id": graph_id,
                    "folder_id": folder_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to create folder: {e}")

    @server.tool(
        name="move_folder",
        title="Move Folder",
        description=(
            "Move a folder to a new parent folder. "
            "Set new_parent_id to null to move to root level. "
            "Optionally update the order for positioning among siblings."
        ),
    )
    async def move_folder_tool(
        graph_id: str,
        folder_id: str,
        new_parent_id: Optional[str] = None,
        new_order: Optional[float] = None,
        context: Context | None = None,
    ) -> dict:
        """Move a folder to a new parent via Y.js."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Read current folder state from Y.js
            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)
            current = reader.get_folder(folder_id.strip())

            if not current:
                raise RuntimeError(f"Folder '{folder_id}' not found in graph '{graph_id}'")

            # Validate target folder exists
            if new_parent_id and not reader.folder_exists(new_parent_id.strip()):
                raise ValueError(
                    f"Target folder '{new_parent_id}' does not exist in graph '{graph_id}'. "
                    f"Use get_workspace to see available folder IDs."
                )

            # Update folder with new parent/order via Y.js
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).update_folder(
                    folder_id.strip(),
                    parent_id=new_parent_id.strip() if new_parent_id else None,
                    order=new_order,
                ),
                user_id=auth.user_id,
            )

            return {
                "success": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "new_parent_id": new_parent_id.strip() if new_parent_id else None,
            }

        except Exception as e:
            logger.error(
                "Failed to move folder",
                extra_context={
                    "graph_id": graph_id,
                    "folder_id": folder_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to move folder: {e}")

    @server.tool(
        name="rename_folder",
        title="Rename Folder",
        description="Rename a folder's display label.",
    )
    async def rename_folder_tool(
        graph_id: str,
        folder_id: str,
        new_label: str,
        context: Context | None = None,
    ) -> dict:
        """Rename a folder via Y.js."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")
        if not new_label or not new_label.strip():
            raise ValueError("new_label is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Verify folder exists
            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)
            current = reader.get_folder(folder_id.strip())

            if not current:
                raise RuntimeError(f"Folder '{folder_id}' not found in graph '{graph_id}'")

            # Update folder name via Y.js
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).update_folder(
                    folder_id.strip(),
                    name=new_label.strip(),
                ),
                user_id=auth.user_id,
            )

            return {
                "success": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "new_label": new_label.strip(),
            }

        except Exception as e:
            logger.error(
                "Failed to rename folder",
                extra_context={
                    "graph_id": graph_id,
                    "folder_id": folder_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to rename folder: {e}")

    # ------------------------------------------------------------------
    # Helper: hard-delete a document via backend REST API
    # ------------------------------------------------------------------

    async def _hard_delete_document(auth: MCPAuthContext, graph_id: str, doc_id: str) -> None:
        """Submit a DOC_DELETE job to the backend for permanent deletion.

        This removes the document's RDF triples, S3 Y.Doc, and Redis keys.
        """
        url = f"{backend_config.base_url}/documents/{graph_id}/{doc_id}"
        headers = auth.http_headers()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(url, headers=headers)
            # 200 = completed inline, 202 = queued async — both are success
            if resp.status_code not in (200, 202):
                logger.warning(
                    "hard_delete_document_failed",
                    extra_context={
                        "graph_id": graph_id,
                        "doc_id": doc_id,
                        "status": resp.status_code,
                        "body": resp.text[:200] if resp.text else None,
                    },
                )

    def _collect_document_ids_recursive(reader: WorkspaceReader, folder_id: str) -> list[str]:
        """Recursively collect all document IDs under a folder."""
        doc_ids: list[str] = []
        for entity_type, entity_id, _ in reader.get_children_of(folder_id):
            if entity_type == "document":
                doc_ids.append(entity_id)
            elif entity_type == "folder":
                doc_ids.extend(_collect_document_ids_recursive(reader, entity_id))
        return doc_ids

    @server.tool(
        name="delete_folder",
        title="Delete Folder",
        description=(
            "Delete a folder from the workspace. "
            "Set cascade=true to delete all contents (subfolders, documents, artifacts). "
            "Without cascade, deletion fails if the folder has children. "
            "By default, permanently deletes document data (RDF, S3, Redis). "
            "Set hard=false to only remove from workspace navigation."
        ),
    )
    async def delete_folder_tool(
        graph_id: str,
        folder_id: str,
        cascade: bool = False,
        hard: bool = True,
        context: Context | None = None,
    ) -> dict:
        """Delete a folder via Y.js, optionally hard-deleting child documents."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            # Collect document IDs before cascade removes them from CRDT
            hard_deleted_docs: list[str] = []
            if hard and cascade:
                channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
                if channel:
                    reader = WorkspaceReader(channel.doc)
                    hard_deleted_docs = _collect_document_ids_recursive(reader, folder_id.strip())

            # Delete folder from workspace CRDT
            await hp_client.transact_workspace(
                graph_id.strip(),
                lambda doc: WorkspaceWriter(doc).delete_folder(folder_id.strip(), cascade=cascade),
                user_id=auth.user_id,
            )

            # Hard-delete child documents via backend
            if hard and hard_deleted_docs:
                for doc_id in hard_deleted_docs:
                    await _hard_delete_document(auth, graph_id.strip(), doc_id)

            return {
                "success": True,
                "deleted": True,
                "folder_id": folder_id.strip(),
                "graph_id": graph_id.strip(),
                "cascade": cascade,
                "hard": hard,
                "hard_deleted_documents": hard_deleted_docs if hard_deleted_docs else None,
            }

        except ValueError as ve:
            # Cascade error - folder has children
            raise RuntimeError(str(ve))
        except Exception as e:
            logger.error(
                "Failed to delete folder",
                extra_context={
                    "graph_id": graph_id,
                    "folder_id": folder_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to delete folder: {e}")

    # -------------------------------------------------------------------------
    # Artifact Operations
    # -------------------------------------------------------------------------

    @server.tool(
        name="read_artifact",
        title="Read Artifact Content",
        description=(
            "Read an artifact's metadata and, if the artifact has been ingested, its document content. "
            "Returns metadata (name, fileType, mimeType, status, ingestedDocId, etc.) always. "
            "If the artifact has been ingested into a document (ingestedDocId is set), also returns "
            "the ingested document's TipTap XML content and comments.\n\n"
            "If the artifact has not been ingested yet, the response will not include 'ingested_document'. "
            "Use ingest_artifact to convert the artifact into a readable document first."
        ),
    )
    async def read_artifact_tool(
        graph_id: str,
        artifact_id: str,
        context: Context | None = None,
    ) -> dict:
        """Read artifact metadata and document content.

        Artifacts are documents with readOnly=true and sf_* metadata.
        The artifact_id IS the document_id.
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not artifact_id or not artifact_id.strip():
            raise ValueError("artifact_id is required and cannot be empty")

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)

            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)
            # Artifacts are documents — look in documents map
            metadata = reader.get_document(artifact_id.strip())

            if not metadata:
                raise RuntimeError(f"Artifact '{artifact_id}' not found in graph '{graph_id}'")

            result: dict[str, Any] = {
                "graph_id": graph_id.strip(),
                "artifact_id": artifact_id.strip(),
                "metadata": metadata,
            }

            # Read the document content (artifact_id IS the document_id)
            try:
                await _connect_for_read(graph_id.strip(), artifact_id.strip(), auth.user_id)
                doc_channel = hp_client.get_document_channel(graph_id.strip(), artifact_id.strip(), user_id=auth.user_id)
                if doc_channel is not None:
                    doc_reader = DocumentReader(doc_channel.doc)
                    result["ingested_document"] = {
                        "document_id": artifact_id.strip(),
                        "content": doc_reader.to_xml(),
                        "comments": doc_reader.get_all_comments(),
                    }
            except Exception as doc_err:
                logger.warning(
                    "Could not read artifact document content",
                    extra_context={
                        "artifact_id": artifact_id,
                        "error": str(doc_err),
                    },
                )

            return result

        except Exception as e:
            logger.error(
                "Failed to read artifact",
                extra_context={
                    "graph_id": graph_id,
                    "artifact_id": artifact_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to read artifact: {e}")

    @server.tool(
        name="upload_artifact",
        title="Upload Artifact",
        description=(
            "Upload a local file as a new artifact in a graph. "
            "Reads the file from the local filesystem and uploads it to the platform.\n\n"
            "Supported formats: TXT, MD, HTML, and text-like files "
            "(.log, .csv, .json, .xml, .yaml, .py, .js, .ts, .sql, etc.)\n"
            "Maximum size: 50MB\n\n"
            "After uploading, use ingest_artifact to convert the artifact into a "
            "readable/editable document."
        ),
    )
    async def upload_artifact_tool(
        graph_id: str,
        file_path: str,
        parent_id: Optional[str] = None,
        label: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Upload a local file as a new artifact via the backend API."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not file_path or not file_path.strip():
            raise ValueError("file_path is required and cannot be empty")

        path = Path(file_path.strip())
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        file_size = path.stat().st_size
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            raise ValueError(
                f"File too large ({file_size / (1024 * 1024):.1f}MB). Maximum size: 50MB"
            )

        try:
            content = path.read_bytes()
            filename = label or path.name
            mime_type, _ = mimetypes.guess_type(path.name)
            if mime_type is None:
                mime_type = "application/octet-stream"

            url = f"{backend_config.base_url}/artifacts/{graph_id.strip()}/upload"

            data: dict[str, str] = {}
            if parent_id is not None and parent_id.strip():
                data["parent_id"] = parent_id.strip()

            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                resp = await client.post(
                    url,
                    files={"file": (filename, content, mime_type)},
                    data=data,
                    headers=auth.http_headers(),
                )
                if resp.status_code != 201:
                    raise RuntimeError(
                        f"Backend returned {resp.status_code}: {resp.text[:300]}"
                    )
                result = resp.json()

            # New upload endpoint returns documentId (auto-ingested).
            # Fall back to artifactId for backward compat with old backend.
            doc_id = result.get("documentId") or result.get("artifactId")
            return {
                "success": True,
                "document_id": doc_id,
                "file_type": result.get("fileType"),
                "title": result.get("title") or result.get("label"),
                "read_only": result.get("readOnly", True),
            }

        except Exception as e:
            logger.error(
                "Failed to upload artifact",
                extra_context={
                    "graph_id": graph_id,
                    "file_path": file_path,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to upload artifact: {e}")

    @server.tool(
        name="ingest_artifact",
        title="Ingest Artifact",
        description=(
            "Ingest an artifact into a readable document. Converts uploaded files (PDFs, etc.) "
            "into TipTap documents that can be read and searched.\n\n"
            "Modes:\n"
            "- 'ingest' (default): Creates a read-only e-reader document. Best for reference material.\n"
            "- 'import': Creates a fully editable document. Best when you need to modify the content.\n\n"
            "After ingestion, use read_artifact or read_document with the returned document_id "
            "to access the content.\n\n"
            "Note: New uploads are auto-ingested. This tool is only needed for legacy artifacts "
            "that were uploaded before auto-ingestion was enabled."
        ),
    )
    async def ingest_artifact_tool(
        graph_id: str,
        artifact_id: str,
        mode: str = "ingest",
        title: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Ingest an artifact into a document via the backend API."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not artifact_id or not artifact_id.strip():
            raise ValueError("artifact_id is required and cannot be empty")
        if mode not in ("ingest", "import"):
            raise ValueError("mode must be 'ingest' or 'import'")

        try:
            url = f"{backend_config.base_url}/artifacts/{graph_id.strip()}/{artifact_id.strip()}/import"
            body: dict[str, Any] = {
                "readOnly": mode == "ingest",
                "useYdocPath": True,
            }
            if title is not None:
                body["title"] = title

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                resp = await client.post(url, json=body, headers=auth.http_headers())
                if resp.status_code not in (200, 201):
                    raise RuntimeError(
                        f"Backend returned {resp.status_code}: {resp.text[:300]}"
                    )
                data = resp.json()

            return {
                "success": True,
                "graph_id": graph_id.strip(),
                "artifact_id": artifact_id.strip(),
                "document_id": data.get("documentId"),
                "title": data.get("title"),
                "mode": mode,
                "read_only": mode == "ingest",
            }

        except Exception as e:
            logger.error(
                "Failed to ingest artifact",
                extra_context={
                    "graph_id": graph_id,
                    "artifact_id": artifact_id,
                    "mode": mode,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to ingest artifact: {e}")

    # -------------------------------------------------------------------------
    # Document Navigation Operations
    # -------------------------------------------------------------------------

    async def _move_document_cross_graph(
        auth: MCPAuthContext,
        source_graph_id: str,
        target_graph_id: str,
        document_id: str,
        new_parent_id: Optional[str],
    ) -> dict:
        """Move a document between graphs by reading content, writing to target, and deleting source.

        Content and comments are preserved. Wires and block IDs are not — they are
        graph-scoped and cannot be transferred. The write happens before the delete
        so that a failure mid-operation leaves the source intact (at worst, duplicated).
        """
        # 1. Read source document content
        await _connect_for_read(source_graph_id, document_id, auth.user_id)
        channel = hp_client.get_document_channel(source_graph_id, document_id, user_id=auth.user_id)
        if channel is None:
            raise RuntimeError(f"Could not connect to document '{document_id}' in graph '{source_graph_id}'")

        reader = DocumentReader(channel.doc)
        xml_content = reader.to_xml()
        comments = reader.get_all_comments()

        # Get title from source workspace metadata
        await hp_client.connect_workspace(source_graph_id, user_id=auth.user_id)
        ws_channel = hp_client.get_workspace_channel(source_graph_id, user_id=auth.user_id)
        ws_reader = WorkspaceReader(ws_channel.doc)
        doc_meta = ws_reader.get_document(document_id)
        title = doc_meta.get("title", "Untitled") if doc_meta else "Untitled"

        # 2. Validate target graph and folder
        await hp_client.connect_workspace(target_graph_id, user_id=auth.user_id)
        if new_parent_id:
            target_ws = hp_client.get_workspace_channel(target_graph_id, user_id=auth.user_id)
            if target_ws is None:
                raise RuntimeError(f"Could not connect to workspace for graph '{target_graph_id}'")
            target_reader = WorkspaceReader(target_ws.doc)
            if not target_reader.folder_exists(new_parent_id.strip()):
                raise ValueError(
                    f"Target folder '{new_parent_id}' not found in graph '{target_graph_id}'. "
                    f"Use get_workspace to see available folder IDs."
                )

        # 3. Write content to target graph
        await hp_client.connect_document(target_graph_id, document_id, user_id=auth.user_id)

        def _write_content(doc):
            writer = DocumentWriter(doc)
            writer.replace_all_content(xml_content)
            if comments:
                for cid, cdata in comments.items():
                    writer.set_comment(
                        cid,
                        text=cdata.get("text", ""),
                        author=cdata.get("author", "Unknown"),
                        author_id=cdata.get("authorId", "unknown"),
                        resolved=cdata.get("resolved", False),
                        quoted_text=cdata.get("quotedText"),
                    )

        await hp_client.transact_document(
            target_graph_id, document_id, _write_content, user_id=auth.user_id,
        )

        # Force fresh-channel durability checkpoint before deleting source.
        await _await_document_durable(
            target_graph_id,
            document_id,
            auth.user_id,
            auth=auth,
        )

        # Register in target workspace
        parent = new_parent_id.strip() if new_parent_id else None
        await hp_client.transact_workspace(
            target_graph_id,
            lambda doc: WorkspaceWriter(doc).upsert_document(document_id, title, parent_id=parent),
            user_id=auth.user_id,
        )

        # 4. Delete from source (write succeeded — safe to delete)
        await hp_client.transact_workspace(
            source_graph_id,
            lambda doc: WorkspaceWriter(doc).delete_document(document_id),
            user_id=auth.user_id,
        )
        await _hard_delete_document(auth, source_graph_id, document_id)

        logger.info(
            "cross_graph_move_complete",
            extra_context={
                "document_id": document_id,
                "source_graph_id": source_graph_id,
                "target_graph_id": target_graph_id,
                "title": title,
            },
        )

        return {
            "success": True,
            "moved_cross_graph": True,
            "document_id": document_id,
            "source_graph_id": source_graph_id,
            "target_graph_id": target_graph_id,
            "new_parent_id": parent,
            "title": title,
            "warning": "Wires and block-level connections were not preserved in cross-graph move. Block IDs have changed.",
        }

    @server.tool(
        name="move_documents",
        title="Move Document(s)",
        description=(
            "Move document(s) to a folder. For a single document, pass document_id. "
            "For multiple documents, pass a document_ids list. All documents are moved "
            "to the same new_parent_id (null for root level). "
            "Set target_graph_id to move to a different graph "
            "(content and comments are preserved; wires and block IDs are not). "
            "Note: This updates the document's folder assignment in workspace navigation."
        ),
    )
    async def move_documents_tool(
        graph_id: str,
        document_id: Optional[str] = None,
        document_ids: list[str] | None = None,
        new_parent_id: str | None = None,
        target_graph_id: str | None = None,
        context: Context | None = None,
    ) -> dict:
        """Move one or more documents to a folder, or to a different graph."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")

        # Resolve single vs batch
        if document_ids is not None and document_id is not None:
            raise ValueError("Provide either 'document_id' (single) or 'document_ids' (batch), not both")

        if document_ids is not None:
            if not document_ids:
                raise ValueError("document_ids list must not be empty")
            ids_to_move = [d.strip() for d in document_ids]
        elif document_id is not None:
            ids_to_move = [document_id.strip()]
        else:
            raise ValueError("Either 'document_id' or 'document_ids' is required")

        is_single = document_id is not None
        is_cross_graph = target_graph_id and target_graph_id.strip() != graph_id.strip()

        if is_cross_graph:
            # Cross-graph move — iterate over each document
            results: List[Dict[str, Any]] = []
            errors: List[Dict[str, Any]] = []
            for did in ids_to_move:
                try:
                    result = await _move_document_cross_graph(
                        auth, graph_id.strip(), target_graph_id.strip(), did, new_parent_id,
                    )
                    results.append(result)
                except Exception as e:
                    errors.append({"document_id": did, "error": str(e)})

            if is_single:
                if errors:
                    raise RuntimeError(f"Failed to move document cross-graph: {errors[0]['error']}")
                return results[0]

            output: Dict[str, Any] = {"results": results, "moved_count": len(results)}
            if errors:
                output["errors"] = errors
                output["error_count"] = len(errors)
            return output

        # Same-graph move
        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)
            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)

            # Validate target folder exists
            if new_parent_id and not reader.folder_exists(new_parent_id.strip()):
                raise ValueError(
                    f"Target folder '{new_parent_id}' does not exist in graph '{graph_id}'. "
                    f"Use get_workspace to see available folder IDs."
                )

            results = []
            errors = []
            for did in ids_to_move:
                current = reader.get_document(did)
                if not current:
                    errors.append({"document_id": did, "error": f"Document '{did}' not found in graph '{graph_id}'"})
                    continue

                try:
                    await hp_client.transact_workspace(
                        graph_id.strip(),
                        lambda doc, _did=did: WorkspaceWriter(doc).update_document(
                            _did,
                            parent_id=new_parent_id.strip() if new_parent_id else None,
                        ),
                        user_id=auth.user_id,
                    )
                    results.append({
                        "success": True,
                        "document_id": did,
                        "graph_id": graph_id.strip(),
                        "new_parent_id": new_parent_id.strip() if new_parent_id else None,
                    })
                except Exception as e:
                    errors.append({"document_id": did, "error": str(e)})

            if is_single:
                if errors:
                    raise RuntimeError(f"Failed to move document: {errors[0]['error']}")
                return results[0]

            output = {"results": results, "moved_count": len(results)}
            if errors:
                output["errors"] = errors
                output["error_count"] = len(errors)
            return output

        except Exception as e:
            logger.error(
                "Failed to move documents",
                extra_context={"graph_id": graph_id, "error": str(e)},
            )
            raise RuntimeError(f"Failed to move documents: {e}")

    @server.tool(
        name="delete_documents",
        title="Delete Document(s)",
        description=(
            "Delete document(s). For a single document, pass document_id. "
            "For multiple documents, pass a document_ids list. "
            "By default, permanently deletes including content, RDF triples, and stored data. "
            "Set hard=false to only remove from workspace navigation (soft delete) "
            "— documents can then be recreated by writing to the same document_id."
        ),
    )
    async def delete_documents_tool(
        graph_id: str,
        document_id: Optional[str] = None,
        document_ids: list[str] | None = None,
        hard: bool = True,
        context: Context | None = None,
    ) -> dict:
        """Delete one or more documents, optionally with permanent data destruction."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")

        # Resolve single vs batch
        if document_ids is not None and document_id is not None:
            raise ValueError("Provide either 'document_id' (single) or 'document_ids' (batch), not both")

        if document_ids is not None:
            if not document_ids:
                raise ValueError("document_ids list must not be empty")
            ids_to_delete = [d.strip() for d in document_ids]
        elif document_id is not None:
            ids_to_delete = [document_id.strip()]
        else:
            raise ValueError("Either 'document_id' or 'document_ids' is required")

        is_single = document_id is not None

        try:
            await hp_client.connect_workspace(graph_id.strip(), user_id=auth.user_id)
            channel = hp_client.get_workspace_channel(graph_id.strip(), user_id=auth.user_id)
            if channel is None:
                raise RuntimeError(f"Workspace not connected: {graph_id}")

            reader = WorkspaceReader(channel.doc)
            results: List[Dict[str, Any]] = []
            errors: List[Dict[str, Any]] = []

            for did in ids_to_delete:
                current = reader.get_document(did)
                if not current:
                    errors.append({"document_id": did, "error": f"Document '{did}' not found in graph '{graph_id}'"})
                    continue

                try:
                    await hp_client.transact_workspace(
                        graph_id.strip(),
                        lambda doc, _did=did: WorkspaceWriter(doc).delete_document(_did),
                        user_id=auth.user_id,
                    )

                    if hard:
                        await _hard_delete_document(auth, graph_id.strip(), did)

                    results.append({
                        "success": True,
                        "deleted": True,
                        "hard": hard,
                        "document_id": did,
                        "graph_id": graph_id.strip(),
                        "title": current.get("title", "Untitled"),
                    })
                except Exception as e:
                    errors.append({"document_id": did, "error": str(e)})

            if is_single:
                if errors:
                    raise RuntimeError(f"Failed to delete document: {errors[0]['error']}")
                return results[0]

            output: Dict[str, Any] = {"results": results, "deleted_count": len(results)}
            if errors:
                output["errors"] = errors
                output["error_count"] = len(errors)
            return output

        except Exception as e:
            logger.error(
                "Failed to delete documents",
                extra_context={"graph_id": graph_id, "error": str(e)},
            )
            raise RuntimeError(f"Failed to delete documents: {e}")

    # -------------------------------------------------------------------------
    # Block-Level Document Operations
    # -------------------------------------------------------------------------

    @server.tool(
        name="get_block",
        title="Get Block by ID",
        description=(
            "Read a specific block by its data-block-id. Returns detailed info including "
            "the block's content and context (prev/next block IDs). "
            "Use this for targeted reads without fetching the entire document.\n\n"
            "Formats (set via 'format' parameter):\n"
            "- default (None): Full TipTap XML with attributes — shows exactly what the user sees. "
            "Use when you need raw markup or precise formatting details.\n"
            "- 'markdown': Compact markdown rendering of the block. Recommended for agent workflows "
            "that just need to read content.\n"
            "- 'text': Plain text only. Most compact — use when you only need the text, not structure.\n\n"
            "Always returns fresh content — automatically reconnects if cached state "
            "is older than 2 seconds, and retries once on sync timeout."
        ),
    )
    async def get_block_tool(
        graph_id: str,
        document_id: str,
        block_id: str,
        format: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Get detailed information about a block by its ID."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not block_id or not block_id.strip():
            raise ValueError("block_id is required")
        if format is not None and format not in ("markdown", "text"):
            raise ValueError("format must be None, 'markdown', or 'text'")

        try:
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id, auth=auth)

            document_doc, _ = await _load_document_doc_for_read(
                graph_id.strip(),
                document_id.strip(),
                auth.user_id,
                auth=auth,
                tool_name="get_block",
            )
            reader = DocumentReader(document_doc)
            block_info = reader.get_block_info(block_id.strip())

            if block_info is None:
                raise RuntimeError(f"Block '{block_id}' not found in document '{document_id}'.")

            # Format the block based on requested format
            if format == "markdown":
                block = {
                    "block_id": block_info["block_id"],
                    "index": block_info["index"],
                    "type": block_info["type"],
                    "markdown": tiptap_xml_to_markdown(block_info["xml"]),
                    "text_length": block_info["text_length"],
                    "context": block_info["context"],
                }
            elif format == "text":
                block = {
                    "block_id": block_info["block_id"],
                    "index": block_info["index"],
                    "type": block_info["type"],
                    "text_content": block_info["text_content"],
                    "text_length": block_info["text_length"],
                    "context": block_info["context"],
                }
            else:
                block = block_info

            return {"block": block}

        except Exception as e:
            logger.error(
                "Failed to get block",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "block_id": block_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to get block: {e}")

    @server.tool(
        name="query_blocks",
        title="Query Blocks",
        description=(
            "Search for blocks matching specific criteria. Filter by block type, indent level, "
            "list type, checked state, or text content. Returns a list of matching block summaries. "
            "Use this to find blocks without reading the entire document.\n\n"
            "Always returns fresh content — automatically reconnects if cached state "
            "is older than 2 seconds, and retries once on sync timeout.\n\n"
            "NOTE: This is a single-document structural filter — it queries one document's CRDT state "
            "directly, with no backend round-trip. Use it for structural navigation within a document "
            "(e.g., find all headings, find checked tasks, find blocks at indent level 2). "
            "For cross-document content discovery, use search_blocks instead (hybrid lexical+semantic)."
        ),
    )
    async def query_blocks_tool(
        graph_id: str,
        document_id: str,
        block_type: Optional[str] = None,
        heading_level: Optional[int] = None,
        indent: Optional[int] = None,
        indent_gte: Optional[int] = None,
        indent_lte: Optional[int] = None,
        list_type: Optional[str] = None,
        checked: Optional[bool] = None,
        text_contains: Optional[str] = None,
        limit: int = 50,
        context: Context | None = None,
    ) -> dict:
        """Query blocks matching specific criteria."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")

        try:
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id, auth=auth)

            document_doc, _ = await _load_document_doc_for_read(
                graph_id.strip(),
                document_id.strip(),
                auth.user_id,
                auth=auth,
                tool_name="query_blocks",
            )
            reader = DocumentReader(document_doc)
            matches = reader.query_blocks(
                block_type=block_type,
                heading_level=heading_level,
                indent=indent,
                indent_gte=indent_gte,
                indent_lte=indent_lte,
                list_type=list_type,
                checked=checked,
                text_contains=text_contains,
                limit=limit,
            )

            return {"count": len(matches), "blocks": matches}

        except Exception as e:
            logger.error(
                "Failed to query blocks",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to query blocks: {e}")

    @server.tool(
        name="update_blocks",
        title="Update Block(s)",
        description=(
            "Update block(s) by ID. For a single block, pass block_id with attributes "
            "and/or xml_content. For multiple blocks, pass an `updates` list where each "
            "entry has block_id and optionally attributes and/or xml_content.\n\n"
            "Can update attributes (indent, checked, listType) without changing content, "
            "or replace entire block content. Plain text in xml_content is auto-wrapped "
            "in a <paragraph>. Markdown is also accepted and auto-converted.\n\n"
            "Container blocks (blockquote, tableCell, tableHeader) require paragraph children — "
            "auto-wrapping applied as fallback, but prefer explicit: <blockquote><paragraph>text</paragraph></blockquote>\n\n"
            "Always read the document or block first (read_document or get_block) before updating — "
            "write tools use a cached channel and need a preceding read to sync latest state from the server."
        ),
    )
    async def update_blocks_tool(
        graph_id: str,
        document_id: str,
        block_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        xml_content: Optional[str] = None,
        updates: list[dict[str, Any]] | None = None,
        context: Context | None = None,
    ) -> dict:
        """Update one or more blocks' attributes or content."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")

        # Resolve single vs batch
        if updates is not None and block_id is not None:
            raise ValueError("Provide either 'block_id' (single) or 'updates' (batch), not both")

        if updates is not None:
            if not updates:
                raise ValueError("updates list must not be empty")
        elif block_id is not None:
            if attributes is None and xml_content is None:
                raise ValueError("Either attributes or xml_content must be provided")
            updates = [{"block_id": block_id, "attributes": attributes, "xml_content": xml_content}]
        else:
            raise ValueError("Either 'block_id' (single) or 'updates' (batch) is required")

        is_single = block_id is not None

        try:
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id, auth=auth)
            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            results: list[Dict[str, Any]] = []

            def perform_updates(doc: Any) -> None:
                writer = DocumentWriter(doc)
                for update in updates:
                    bid = update.get("block_id")
                    if not bid:
                        results.append({"error": "missing block_id"})
                        continue

                    try:
                        content = update.get("xml_content") or update.get("content")
                        attrs = update.get("attributes")

                        if content is None and attrs is None:
                            results.append({
                                "block_id": bid,
                                "error": "No xml_content or attributes provided — nothing to update",
                            })
                            continue

                        if content:
                            resolved = _ensure_xml(content)
                            writer.replace_block_by_id(bid.strip(), resolved)
                        if attrs:
                            writer.update_block_attributes(bid.strip(), attrs)
                        results.append({"block_id": bid, "success": True})
                    except Exception as e:
                        results.append({"block_id": bid, "error": str(e)})

            await hp_client.transact_document(
                graph_id.strip(),
                document_id.strip(),
                perform_updates,
                user_id=auth.user_id,
            )

            # Single mode: return flat result or raise on error
            if is_single and results:
                r = results[0]
                if "error" in r:
                    raise RuntimeError(f"Failed to update block {r.get('block_id', '?')}: {r['error']}")
                return r

            return {
                "success": all(r.get("success") for r in results),
                "results": results,
                "updated_count": sum(1 for r in results if r.get("success")),
                "error_count": sum(1 for r in results if "error" in r),
            }

        except Exception as e:
            logger.error(
                "Failed to update blocks",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to update blocks: {e}")

    @server.tool(
        name="edit_block_text",
        title="Edit Block Text",
        description=(
            "Insert or delete text at specific character offsets within a block, using "
            "CRDT-native operations that merge cleanly with concurrent browser edits. "
            "Unlike update_blocks (which replaces entire content), this enables true "
            "collaborative editing without data loss.\n\n"
            "Workflow: 1) Call get_block to read current text and length (this also syncs "
            "the channel to latest server state), "
            "2) Determine offset(s) for edits, "
            "3) Call edit_block_text with operations, "
            "4) Response includes updated text for verification.\n\n"
            "IMPORTANT: Always call get_block immediately before editing — the read syncs "
            "fresh state from the server, and the write reuses that channel. Skipping the "
            "read risks operating on stale offsets.\n\n"
            "Each operation has: type ('insert' or 'delete'), offset (0-indexed char position), "
            "text (for insert), length (for delete), attrs (optional formatting like {\"bold\": {}}), "
            "inherit_format (default true - inherit formatting from preceding character).\n\n"
            "Multiple operations are applied in a single transaction. "
            "Insert beyond text length appends at end. "
            "Delete beyond text length raises an error."
        ),
    )
    async def edit_block_text_tool(
        graph_id: str,
        document_id: str,
        block_id: str,
        operations: list[Dict[str, Any]],
        context: Context | None = None,
    ) -> dict:
        """Edit text within a block at specific character offsets.

        Args:
            graph_id: The graph containing the document
            document_id: The document containing the block
            block_id: The block to edit
            operations: List of insert/delete operations with offsets
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not block_id or not block_id.strip():
            raise ValueError("block_id is required")
        if not operations:
            raise ValueError("operations list is required and cannot be empty")

        try:
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id, auth=auth)

            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            updated_text_info: dict = {}

            def perform_edit(doc: Any) -> None:
                nonlocal updated_text_info
                writer = DocumentWriter(doc)
                updated_text_info = writer.edit_block_text(
                    block_id.strip(), operations
                )

            await hp_client.transact_document(
                graph_id.strip(),
                document_id.strip(),
                perform_edit,
                user_id=auth.user_id,
            )

            return {"success": True, "block": updated_text_info}

        except ValueError as ve:
            # Validation errors - return as-is for clear agent feedback
            raise RuntimeError(str(ve))
        except Exception as e:
            logger.error(
                "Failed to edit block text",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "block_id": block_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to edit block text: {e}")

    @server.tool(
        name="insert_block",
        title="Insert Block",
        description=(
            "Insert a new block relative to an existing block. Use position='after' or 'before' "
            "to specify where to insert. Returns the new block's generated ID. "
            "For appending to the end, use append_to_document instead. "
            "Plain text in xml_content is auto-wrapped in a <paragraph>. "
            "Container blocks (blockquote, tableCell, tableHeader) require paragraph children — "
            "auto-wrapping applied as fallback, but prefer explicit: <blockquote><paragraph>text</paragraph></blockquote> "
            "Markdown is also accepted and auto-converted.\n\n"
            "Always read the document first (read_document or get_block) before inserting — "
            "write tools use a cached channel and need a preceding read to sync latest state from the server."
        ),
    )
    async def insert_block_tool(
        graph_id: str,
        document_id: str,
        reference_block_id: str,
        xml_content: str,
        position: str = "after",
        context: Context | None = None,
    ) -> dict:
        """Insert a new block before or after a reference block.

        Args:
            graph_id: The graph containing the document
            document_id: The document to insert into
            reference_block_id: The block to insert relative to
            xml_content: TipTap XML for the new block
            position: 'after' or 'before' the reference block
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not reference_block_id or not reference_block_id.strip():
            raise ValueError("reference_block_id is required")
        if not xml_content or not xml_content.strip():
            raise ValueError("xml_content is required")
        if position not in ("after", "before"):
            raise ValueError("position must be 'after' or 'before'")

        # Auto-wrap plain text in a paragraph element
        resolved_xml = _ensure_xml(xml_content)

        try:
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id, auth=auth)

            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            new_block_id: str = ""

            def perform_insert(doc: Any) -> None:
                nonlocal new_block_id
                writer = DocumentWriter(doc)
                if position == "after":
                    new_block_id = writer.insert_block_after_id(
                        reference_block_id.strip(), resolved_xml
                    )
                else:
                    new_block_id = writer.insert_block_before_id(
                        reference_block_id.strip(), resolved_xml
                    )

            await hp_client.transact_document(
                graph_id.strip(),
                document_id.strip(),
                perform_insert,
                user_id=auth.user_id,
            )

            return {"success": True, "new_block_id": new_block_id}

        except Exception as e:
            logger.error(
                "Failed to insert block",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "reference_block_id": reference_block_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to insert block: {e}")

    @server.tool(
        name="delete_block",
        title="Delete Block",
        description=(
            "Delete a block by its ID. Use cascade=true to also delete all subsequent blocks "
            "with higher indent (indent-based children). Returns the list of deleted block IDs.\n\n"
            "Always read the document first (read_document) before deleting — "
            "write tools use a cached channel and need a preceding read to sync latest state from the server."
        ),
    )
    async def delete_block_tool(
        graph_id: str,
        document_id: str,
        block_id: str,
        cascade: bool = False,
        context: Context | None = None,
    ) -> dict:
        """Delete a block and optionally its indent-children.

        Args:
            graph_id: The graph containing the document
            document_id: The document containing the block
            block_id: The block to delete
            cascade: If True, also delete indent-children
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required")
        if not block_id or not block_id.strip():
            raise ValueError("block_id is required")

        try:
            # Validate document exists in workspace
            await _validate_document_in_workspace(graph_id.strip(), document_id.strip(), auth.user_id, auth=auth)

            await hp_client.connect_document(graph_id.strip(), document_id.strip(), user_id=auth.user_id)

            deleted_ids: list[str] = []

            def perform_delete(doc: Any) -> None:
                nonlocal deleted_ids
                writer = DocumentWriter(doc)
                deleted_ids = writer.delete_block_by_id(
                    block_id.strip(), cascade_children=cascade
                )

            await hp_client.transact_document(
                graph_id.strip(),
                document_id.strip(),
                perform_delete,
                user_id=auth.user_id,
            )

            # Tombstone (not delete) wires connected to deleted blocks.
            # Tombstoned wires are hidden from queries but survive for a
            # grace period so undo can restore them.  Permanent deletion
            # happens via sweep_tombstoned_wires on workspace flush.
            tombstoned_wires: list[str] = []
            if deleted_ids:
                def tombstone_wires(ws_doc: Any) -> None:
                    nonlocal tombstoned_wires
                    ws = WorkspaceWriter(ws_doc)
                    for bid in deleted_ids:
                        tombstoned_wires.extend(
                            ws.tombstone_wires_for_block(document_id.strip(), bid)
                        )

                await hp_client.transact_workspace(
                    graph_id.strip(), tombstone_wires, user_id=auth.user_id,
                )

            return {
                "success": True,
                "deleted_block_ids": deleted_ids,
                "tombstoned_wire_ids": tombstoned_wires,
            }

        except Exception as e:
            logger.error(
                "Failed to delete block",
                extra_context={
                    "graph_id": graph_id,
                    "document_id": document_id,
                    "block_id": block_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to delete block: {e}")

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    # ------------------------------------------------------------------
    # dump_chat — Save conversation content to a timestamped document
    # ------------------------------------------------------------------

    CHAT_LOGS_FOLDER_ID = "chat-logs"
    CHAT_LOGS_FOLDER_LABEL = "Chat Logs"

    @server.tool(
        name="dump_chat",
        title="Save Chat to Document",
        description=(
            "Saves conversation content to a timestamped document in a 'Chat Logs' folder. "
            "Pass the formatted conversation as markdown in the content parameter. "
            "Creates the Chat Logs folder automatically if it doesn't exist."
        ),
    )
    async def dump_chat_tool(
        graph_id: str,
        content: str,
        title: Optional[str] = None,
        context: Context | None = None,
    ) -> dict:
        """Save chat content to a new timestamped document in Chat Logs folder.

        Args:
            graph_id: The graph to save the chat log in
            content: Formatted conversation content (markdown or XML)
            title: Optional title; defaults to "Chat Log — <timestamp>"
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not content or not content.strip():
            raise ValueError("content is required")

        graph_id = graph_id.strip()
        now = datetime.now(timezone.utc)
        doc_id = f"chat-log-{now.strftime('%Y-%m-%d-%H%M%S')}"
        doc_title = title or f"Chat Log — {now.strftime('%b %d, %Y %I:%M %p')} UTC"

        try:
            # 1. Ensure Chat Logs folder exists
            await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
            ws_channel = hp_client.get_workspace_channel(graph_id, user_id=auth.user_id)
            if ws_channel is None:
                raise RuntimeError(f"Could not connect to workspace for graph '{graph_id}'.")

            reader = WorkspaceReader(ws_channel.doc)
            if not reader.folder_exists(CHAT_LOGS_FOLDER_ID):
                await hp_client.transact_workspace(
                    graph_id,
                    lambda doc: WorkspaceWriter(doc).upsert_folder(
                        CHAT_LOGS_FOLDER_ID,
                        CHAT_LOGS_FOLDER_LABEL,
                        section="documents",
                    ),
                    user_id=auth.user_id,
                )

            # 2. Write document content
            xml_content = _ensure_xml_multiblock(content)
            await hp_client.connect_document(graph_id, doc_id, user_id=auth.user_id)
            await hp_client.transact_document(
                graph_id,
                doc_id,
                lambda doc: DocumentWriter(doc).replace_all_content(xml_content),
                user_id=auth.user_id,
            )

            # Force fresh-channel durability checkpoint before exposing in navigation.
            await _await_document_durable(
                graph_id,
                doc_id,
                auth.user_id,
                auth=auth,
            )

            # 3. Register document in workspace under Chat Logs folder
            await hp_client.transact_workspace(
                graph_id,
                lambda doc: WorkspaceWriter(doc).upsert_document(
                    doc_id, doc_title, parent_id=CHAT_LOGS_FOLDER_ID,
                ),
                user_id=auth.user_id,
            )

            return {
                "success": True,
                "graph_id": graph_id,
                "document_id": doc_id,
                "title": doc_title,
                "folder_id": CHAT_LOGS_FOLDER_ID,
            }

        except Exception as e:
            logger.error(
                "Failed to dump chat",
                extra_context={
                    "graph_id": graph_id,
                    "doc_id": doc_id,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Failed to save chat log: {e}")

    logger.info("Registered hocuspocus tools (documents, navigation, and block operations)")
