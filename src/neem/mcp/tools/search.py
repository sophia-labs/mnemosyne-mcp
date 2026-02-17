"""MCP tools for document and content search.

Provides:
- search_documents: Fast title/path search against workspace Y.Doc
- search_blocks: Hybrid lexical (SPARQL) + semantic (Qdrant) content search
- reindex_graph: Admin tool for re-embedding all documents
"""

from __future__ import annotations

import asyncio
import json
import re as re_mod
from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context, FastMCP

from neem.hocuspocus import HocuspocusClient
from neem.mcp.auth import MCPAuthContext
from neem.mcp.jobs import RealtimeJobClient
from neem.mcp.tools.basic import await_job_completion, submit_job
from neem.mcp.trace import trace
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_user_id_from_token

logger = LoggerFactory.get_logger("mcp.tools.search")

STREAM_TIMEOUT_SECONDS = 60.0
JsonDict = Dict[str, Any]


def _render_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_folder_paths(folders: dict[str, dict]) -> dict[str, str]:
    """Build {folder_id: "Parent/Child/Grandchild"} path map from flat folder data."""
    paths: dict[str, str] = {}
    for fid in folders:
        parts: list[str] = []
        current: str | None = fid
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            fdata = folders.get(current)
            if not fdata:
                break
            parts.append(fdata.get("name") or "Untitled")
            current = fdata.get("parentId")
        paths[fid] = "/".join(reversed(parts))
    return paths


def _fuzzy_score(query_tokens: set[str], text_tokens: set[str]) -> float:
    """Simple token-overlap fuzzy score between 0 and 1."""
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / max(len(query_tokens), len(text_tokens))


def _escape_sparql_string(s: str) -> str:
    """Escape a string for use in SPARQL string literals."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _parse_block_uri(uri: str) -> tuple[str | None, str | None]:
    """Extract (doc_id, block_id) from RDF block URI.

    URI pattern: urn:mnemosyne:user:{uid}:graph:{gid}:doc:{doc_id}#block-{block_id}
    """
    if "#block-" in uri:
        doc_part, block_id = uri.rsplit("#block-", 1)
        doc_id = doc_part.rsplit(":", 1)[-1] if ":" in doc_part else None
        return doc_id, block_id
    # Fragment block (e.g., #frag) — skip these
    if "#" in uri:
        doc_part = uri.split("#")[0]
        doc_id = doc_part.rsplit(":", 1)[-1] if ":" in doc_part else None
        return doc_id, None
    return None, None


def _extract_inline_result(result: JsonDict) -> Any:
    """Extract result_inline from a job result dict."""
    detail = result.get("detail", {})
    if isinstance(detail, dict):
        inline = detail.get("result_inline")
        if inline is not None:
            # SPARQL results are wrapped in {"raw": actual_result}
            if isinstance(inline, dict) and "raw" in inline:
                return inline["raw"]
            return inline
    return None


# ---------------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------------


def register_search_tools(server: FastMCP) -> None:
    """Register search MCP tools."""

    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping search tool registration")
        return

    job_stream: Optional[RealtimeJobClient] = getattr(server, "_job_stream", None)
    hp_client: Optional[HocuspocusClient] = getattr(server, "_hocuspocus_client", None)

    # ==================================================================
    # search_documents — workspace title/path search
    # ==================================================================

    @server.tool(
        name="search_documents",
        title="Search Documents",
        description=(
            "Fast title and path search across all documents in a graph's workspace. "
            "Returns document IDs, titles, and folder paths. No SPARQL needed.\n\n"
            "**Modes:**\n"
            "- `auto` (default): Cascades through exact ID match → exact title → "
            "substring on title+path → fuzzy token overlap. Returns the first "
            "strategy that produces results.\n"
            "- `exact`: Case-insensitive exact title match only.\n"
            "- `substring`: Case-insensitive substring match on title and folder path.\n"
            "- `regex`: Regex pattern match on title and folder path.\n\n"
            "Use this instead of get_workspace when you know roughly what you're "
            "looking for. Much faster than scanning the full workspace tree."
        ),
    )
    async def search_documents_tool(
        graph_id: str,
        query: str,
        mode: str = "auto",
        limit: int = 20,
        folder_id: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        """Search documents by title and path."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not query or not query.strip():
            raise ValueError("query is required")
        if mode not in ("auto", "exact", "substring", "regex"):
            raise ValueError("mode must be one of: auto, exact, substring, regex")

        graph_id = graph_id.strip()
        query = query.strip()
        limit = min(max(limit, 1), 100)

        if hp_client is None:
            raise RuntimeError("Hocuspocus client not available for workspace search")

        # Connect and get snapshot
        await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
        snapshot = hp_client.get_workspace_snapshot(graph_id, user_id=auth.user_id)

        documents = snapshot.get("documents", {})
        folders = snapshot.get("folders", {})

        if not documents:
            return _render_json({"results": [], "total": 0, "query": query, "mode": mode})

        # Build folder path lookup
        folder_paths = _build_folder_paths(folders)

        # If folder_id is set, collect all descendant folder IDs
        allowed_folders: set[str] | None = None
        if folder_id:
            allowed_folders = {folder_id}
            # Walk folder tree to collect descendants
            changed = True
            while changed:
                changed = False
                for fid, fdata in folders.items():
                    if fid not in allowed_folders and fdata.get("parentId") in allowed_folders:
                        allowed_folders.add(fid)
                        changed = True

        # Build searchable entries
        entries: list[dict[str, Any]] = []
        for doc_id, ddata in documents.items():
            parent_id = ddata.get("parentId")
            # Filter by folder subtree if specified
            if allowed_folders is not None:
                if parent_id not in allowed_folders and (parent_id is not None or folder_id):
                    continue

            title = ddata.get("title") or "Untitled"
            fpath = folder_paths.get(parent_id, "") if parent_id else ""

            entries.append({
                "document_id": doc_id,
                "title": title,
                "folder_path": fpath,
                "folder_id": parent_id,
                "read_only": bool(ddata.get("readOnly")),
                "_title_lower": title.lower(),
                "_path_lower": fpath.lower(),
                "_searchable": f"{title} {fpath}".lower(),
            })

        # Search based on mode
        def _search_exact_id() -> list[dict]:
            return [e for e in entries if e["document_id"] == query]

        def _search_exact_title() -> list[dict]:
            q = query.lower()
            return [e for e in entries if e["_title_lower"] == q]

        def _search_substring() -> list[dict]:
            q = query.lower()
            results = []
            for e in entries:
                if q in e["_title_lower"]:
                    e = {**e, "match_type": "title_substring"}
                    results.append(e)
                elif q in e["_path_lower"]:
                    e = {**e, "match_type": "path_substring"}
                    results.append(e)
            return results

        def _search_regex() -> list[dict]:
            try:
                pattern = re_mod.compile(query, re_mod.IGNORECASE)
            except re_mod.error as exc:
                raise ValueError(f"Invalid regex pattern: {exc}")
            results = []
            for e in entries:
                if pattern.search(e["title"]):
                    e = {**e, "match_type": "regex_title"}
                    results.append(e)
                elif pattern.search(e["folder_path"]):
                    e = {**e, "match_type": "regex_path"}
                    results.append(e)
            return results

        def _search_fuzzy() -> list[dict]:
            q_tokens = set(query.lower().split())
            scored = []
            for e in entries:
                t_tokens = set(e["_searchable"].split())
                score = _fuzzy_score(q_tokens, t_tokens)
                if score >= 0.4:
                    scored.append(({**e, "match_type": "fuzzy"}, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in scored]

        results: list[dict] = []
        effective_mode = mode

        if mode == "auto":
            # Cascade: ID → exact title → substring → fuzzy
            results = _search_exact_id()
            if results:
                effective_mode = "id"
            else:
                results = _search_exact_title()
                if results:
                    effective_mode = "exact"
                    for r in results:
                        r["match_type"] = "exact_title"
                else:
                    results = _search_substring()
                    if results:
                        effective_mode = "substring"
                    else:
                        results = _search_fuzzy()
                        effective_mode = "fuzzy"
        elif mode == "exact":
            results = _search_exact_title()
            for r in results:
                r["match_type"] = "exact_title"
        elif mode == "substring":
            results = _search_substring()
        elif mode == "regex":
            results = _search_regex()

        # Clean internal keys and apply limit
        clean_results = []
        for r in results[:limit]:
            clean_results.append({
                "document_id": r["document_id"],
                "title": r["title"],
                "folder_path": r["folder_path"],
                "folder_id": r["folder_id"],
                "read_only": r["read_only"],
                "match_type": r.get("match_type", effective_mode),
            })

        return _render_json({
            "results": clean_results,
            "count": len(clean_results),
        })

    # ==================================================================
    # search_blocks — hybrid lexical + semantic content search
    # ==================================================================

    async def _run_semantic_search(
        graph_id: str,
        query: str,
        limit: int,
        doc_filter: Optional[str],
        auth: MCPAuthContext,
        ctx: Optional[Context],
    ) -> list[dict]:
        """Run semantic search via Qdrant job queue."""
        payload: JsonDict = {
            "graph_id": graph_id,
            "query": query,
            "limit": limit,
        }
        if doc_filter:
            payload["doc_filter"] = doc_filter

        if job_stream:
            try:
                await job_stream.ensure_ready()
            except Exception:
                pass

        metadata = await submit_job(
            base_url=backend_config.base_url,
            auth=auth,
            task_type="semantic_search",
            payload=payload,
        )

        result = await _wait_for_job_result(job_stream, metadata, ctx, auth)
        inline = _extract_inline_result(result)

        if inline and isinstance(inline, dict):
            if inline.get("error"):
                logger.warning("semantic_search_error", error=inline["error"])
                return []
            raw_results = inline.get("results", [])
            out = []
            for r in raw_results:
                out.append({
                    "block_id": r.get("block_id", ""),
                    "document_id": r.get("document_id", ""),
                    "document_title": r.get("document_title", ""),
                    "text": r.get("text", ""),
                    "similarity_score": r.get("similarity_score"),
                })
            return out

        if result.get("status") == "failed":
            logger.warning("semantic_search_failed", error=result.get("error"))
        return []

    async def _run_lexical_search(
        graph_id: str,
        query: str,
        limit: int,
        doc_filter: Optional[str],
        auth: MCPAuthContext,
        ctx: Optional[Context],
        doc_titles: dict[str, str],
        use_regex: bool = False,
    ) -> list[dict]:
        """Run lexical content search via SPARQL job queue."""
        user_id = auth.user_id or (get_user_id_from_token(auth.token) if auth.token else None)
        if not user_id:
            logger.warning("lexical_search_no_user_id")
            return []

        graph_uri = f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"

        # Build SPARQL FILTER
        if use_regex:
            escaped = _escape_sparql_string(query)
            filter_clause = f'FILTER(REGEX(?content, "{escaped}", "i"))'
        else:
            escaped = _escape_sparql_string(query.lower())
            filter_clause = f'FILTER(CONTAINS(LCASE(?content), "{escaped}"))'

        # Optional doc filter
        doc_filter_clause = ""
        if doc_filter:
            doc_uri_prefix = f"{graph_uri}:doc:{doc_filter}#"
            doc_filter_clause = f'\n    FILTER(STRSTARTS(STR(?blockUri), "{doc_uri_prefix}"))'

        sparql = (
            'PREFIX doc: <http://mnemosyne.dev/doc#>\n'
            'SELECT ?blockUri ?content\n'
            f'FROM <{graph_uri}>\n'
            'WHERE {\n'
            '    ?blockUri doc:content ?content .\n'
            f'    {filter_clause}{doc_filter_clause}\n'
            '}\n'
            f'LIMIT {limit}'
        )

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

        result = await _wait_for_job_result(job_stream, metadata, ctx, auth)
        inline = _extract_inline_result(result)

        if not inline:
            if result.get("status") == "failed":
                logger.warning("lexical_search_failed", error=result.get("error"))
            return []

        # Parse SPARQL JSON results
        bindings = []
        if isinstance(inline, dict):
            bindings = inline.get("results", {}).get("bindings", [])
        elif isinstance(inline, list):
            bindings = inline

        out = []
        for binding in bindings:
            block_uri = ""
            content = ""
            if isinstance(binding, dict):
                block_uri = binding.get("blockUri", {}).get("value", "") if isinstance(binding.get("blockUri"), dict) else str(binding.get("blockUri", ""))
                content = binding.get("content", {}).get("value", "") if isinstance(binding.get("content"), dict) else str(binding.get("content", ""))

            doc_id, block_id = _parse_block_uri(block_uri)
            if not doc_id or not block_id:
                continue

            out.append({
                "block_id": block_id,
                "document_id": doc_id,
                "document_title": doc_titles.get(doc_id, ""),
                "text": content[:500],  # Truncate for readability
            })

        return out

    @server.tool(
        name="search_blocks",
        title="Search Blocks",
        description=(
            "Cross-document content search combining lexical (exact/regex) and semantic "
            "(vector similarity) results. Returns matching blocks with document context.\n\n"
            "**Modes:**\n"
            "- `hybrid` (default): Runs both lexical and semantic search in parallel, "
            "merges results. Lexical finds exact matches; semantic surfaces meaning-based "
            "connections. Each result is tagged with match_source (lexical, semantic, or both).\n"
            "- `lexical`: Substring/regex content search only. Fast and deterministic. "
            "No vector embeddings needed.\n"
            "- `semantic`: Vector similarity search only. Finds content by meaning rather "
            "than exact text.\n\n"
            "Use this for finding content across documents. For finding documents by title, "
            "use search_documents instead."
        ),
    )
    async def search_blocks_tool(
        graph_id: str,
        query: str,
        mode: str = "hybrid",
        limit: int = 30,
        doc_filter: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        """Search for content blocks across documents."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")
        if not query or not query.strip():
            raise ValueError("query is required")
        if mode not in ("hybrid", "lexical", "semantic"):
            raise ValueError("mode must be one of: hybrid, lexical, semantic")

        graph_id = graph_id.strip()
        query = query.strip()
        limit = min(max(limit, 1), 100)
        doc_filter = doc_filter.strip() if doc_filter else None

        # Build doc title lookup from workspace for enriching lexical results
        doc_titles: dict[str, str] = {}
        if hp_client is not None:
            try:
                await hp_client.connect_workspace(graph_id, user_id=auth.user_id)
                snapshot = hp_client.get_workspace_snapshot(graph_id, user_id=auth.user_id)
                for did, ddata in snapshot.get("documents", {}).items():
                    doc_titles[did] = ddata.get("title") or "Untitled"
            except Exception as e:
                logger.warning("search_blocks_workspace_lookup_failed", error=str(e))

        if context:
            await context.report_progress(10, 100)

        # Detect if query looks like a regex (has special chars)
        use_regex = bool(re_mod.search(r'[.*+?^${}()|[\]\\]', query)) and mode != "semantic"

        lexical_results: list[dict] = []
        semantic_results: list[dict] = []
        lexical_count = 0
        semantic_count = 0

        if mode == "hybrid":
            # Fire both in parallel
            lex_task = _run_lexical_search(
                graph_id, query, limit, doc_filter, auth, None, doc_titles, use_regex,
            )
            sem_task = _run_semantic_search(
                graph_id, query, limit, doc_filter, auth, None,
            )
            raw = await asyncio.gather(lex_task, sem_task, return_exceptions=True)

            if isinstance(raw[0], list):
                lexical_results = raw[0]
            else:
                logger.warning("lexical_search_exception", error=str(raw[0]))

            if isinstance(raw[1], list):
                semantic_results = raw[1]
            else:
                logger.warning("semantic_search_exception", error=str(raw[1]))

        elif mode == "lexical":
            lexical_results = await _run_lexical_search(
                graph_id, query, limit, doc_filter, auth, context, doc_titles, use_regex,
            )

        elif mode == "semantic":
            semantic_results = await _run_semantic_search(
                graph_id, query, limit, doc_filter, auth, context,
            )

        # Merge results
        merged: dict[str, dict] = {}

        for r in lexical_results:
            bid = r.get("block_id", "")
            if bid:
                merged[bid] = {**r, "match_source": "lexical"}

        for r in semantic_results:
            bid = r.get("block_id", "")
            if not bid:
                continue
            if bid in merged:
                merged[bid]["match_source"] = "both"
                merged[bid]["similarity_score"] = r.get("similarity_score")
                # Prefer semantic's document_title if lexical's is empty
                if not merged[bid].get("document_title") and r.get("document_title"):
                    merged[bid]["document_title"] = r["document_title"]
            else:
                merged[bid] = {**r, "match_source": "semantic"}

        # Sort: both > lexical > semantic
        source_order = {"both": 0, "lexical": 1, "semantic": 2}
        sorted_results = sorted(
            merged.values(),
            key=lambda x: source_order.get(x.get("match_source", "semantic"), 3),
        )

        final_results = sorted_results[:limit]
        lexical_count = sum(1 for r in final_results if r.get("match_source") in ("lexical", "both"))
        semantic_count = sum(1 for r in final_results if r.get("match_source") in ("semantic", "both"))

        if context:
            await context.report_progress(100, 100)

        return _render_json({
            "results": final_results,
            "count": len(final_results),
            "lexical_count": lexical_count,
            "semantic_count": semantic_count,
        })

    # ==================================================================
    # reindex_graph — admin tool
    # ==================================================================

    @server.tool(
        name="reindex_graph",
        title="Reindex Graph",
        description=(
            "Re-embed all documents in a graph. Used for initial embedding of "
            "existing graphs, after model upgrades, or recovery after vector "
            "store data loss. Enqueues a COMPUTE_EMBEDDINGS job for each document.\n\n"
            "This is an admin/maintenance operation — normal document edits "
            "are automatically indexed via the MATERIALIZE_DOC pipeline."
        ),
    )
    async def reindex_graph_tool(
        graph_id: str,
        context: Context | None = None,
    ) -> str:
        """Re-embed all documents in a graph."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required")

        if job_stream:
            try:
                await job_stream.ensure_ready()
            except Exception:
                pass

        metadata = await submit_job(
            base_url=backend_config.base_url,
            auth=auth,
            task_type="reindex_graph",
            payload={"graph_id": graph_id.strip()},
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(job_stream, metadata, context, auth)
        inline = _extract_inline_result(result)

        if inline and isinstance(inline, dict):
            error = inline.get("error")
            if error:
                return _render_json({"error": error})
            return _render_json(inline)

        if result.get("status") == "failed":
            return _render_json({"error": result.get("error", "Reindex failed")})

        return _render_json({"error": "Failed to get reindex result"})

    logger.info("Registered search tools (search_documents, search_blocks, reindex_graph)")


async def _wait_for_job_result(
    job_stream: Optional[RealtimeJobClient],
    metadata: Any,
    context: Optional[Context],
    auth: MCPAuthContext,
) -> JsonDict:
    """Wait for job completion via WS+poll race."""
    trace("  _wait_for_job_result: racing WS + poll")
    ws_events, poll_payload = await await_job_completion(
        job_stream, metadata, auth, timeout=STREAM_TIMEOUT_SECONDS,
    )

    if ws_events:
        if context:
            await context.report_progress(80, 100)
        for event in reversed(ws_events):
            event_type = event.get("type", "")
            payload = event.get("payload", {})
            payload_status = payload.get("status", "") if isinstance(payload, dict) else ""

            # Match explicit completion types OR job_update with terminal status
            is_success = (
                event_type in ("job_completed", "completed", "succeeded")
                or (event_type == "job_update" and payload_status == "succeeded")
            )
            is_failure = (
                event_type in ("failed", "error")
                or (event_type == "job_update" and payload_status == "failed")
            )

            if is_success:
                if context:
                    await context.report_progress(100, 100)
                result: JsonDict = {"status": "succeeded", "events": len(ws_events)}
                if isinstance(payload, dict):
                    detail = payload.get("detail")
                    if detail:
                        result["detail"] = detail
                return result
            if is_failure:
                error = event.get("error") or payload.get("error", "Job failed")
                return {"status": "failed", "error": error}
        return {"status": "unknown", "event_count": len(ws_events)}

    if context:
        await context.report_progress(100, 100)

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
