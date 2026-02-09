"""
MCP tools for graph CRUD and SPARQL operations.

These tools use the job queue for create/delete operations and SPARQL query/update.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context, FastMCP

from neem.mcp.auth import MCPAuthContext
from neem.mcp.jobs import JobSubmitMetadata, RealtimeJobClient
from neem.mcp.tools.basic import await_job_completion, submit_job
from neem.mcp.trace import trace
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_user_id_from_token

logger = LoggerFactory.get_logger("mcp.tools.graph_ops")

STREAM_TIMEOUT_SECONDS = 60.0
JsonDict = Dict[str, Any]


def register_graph_ops_tools(server: FastMCP) -> None:
    """Register graph CRUD and SPARQL tools."""

    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping graph_ops tool registration")
        return

    job_stream: Optional[RealtimeJobClient] = getattr(server, "_job_stream", None)

    @server.tool(
        name="create_graph",
        title="Create Knowledge Graph",
        description=(
            "Creates a new knowledge graph with the given ID, title, and optional description. "
            "The graph_id should be a URL-safe identifier (e.g., 'my-project', 'research-notes')."
        ),
    )
    async def create_graph_tool(
        graph_id: str,
        title: str,
        description: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        """Create a new knowledge graph."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not title or not title.strip():
            raise ValueError("title is required and cannot be empty")

        payload = {
            "graph_id": graph_id.strip(),
            "title": title.strip(),
        }
        if description:
            payload["description"] = description.strip()

        if job_stream:
            try:
                await job_stream.ensure_ready()
            except Exception:
                pass

        metadata = await submit_job(
            base_url=backend_config.base_url,
            auth=auth,
            task_type="create_graph",
            payload=payload,
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(
            job_stream, metadata, context, auth
        )

        return _render_json({
            "success": True,
            "graph_id": graph_id.strip(),
            "title": title.strip(),
            "description": description.strip() if description else None,
            "job_id": metadata.job_id,
            **result,
        })

    @server.tool(
        name="delete_graph",
        title="Delete Knowledge Graph",
        description=(
            "Deletes a knowledge graph. By default, performs a soft delete (marks as deleted but retains data). "
            "Set hard=True to permanently delete the graph and all its contents. Hard delete cannot be undone."
        ),
    )
    async def delete_graph_tool(
        graph_id: str,
        hard: bool = False,
        context: Context | None = None,
    ) -> str:
        """Delete a knowledge graph.

        Args:
            graph_id: The ID of the graph to delete
            hard: If True, permanently delete the graph and all data.
                  If False (default), soft delete (mark as deleted but retain data).
        """
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")

        if job_stream:
            try:
                await job_stream.ensure_ready()
            except Exception:
                pass

        metadata = await submit_job(
            base_url=backend_config.base_url,
            auth=auth,
            task_type="delete_graph",
            payload={"graph_id": graph_id.strip(), "hard": hard},
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(
            job_stream, metadata, context, auth
        )

        return _render_json({
            "success": True,
            "graph_id": graph_id.strip(),
            "deleted": True,
            "hard_delete": hard,
            "job_id": metadata.job_id,
            **result,
        })

    @server.tool(
        name="sparql_query",
        title="Run SPARQL Query",
        description=(
            "Executes a read-only SPARQL SELECT or CONSTRUCT query against a specific graph. "
            "The graph_id parameter is required to scope the query to a named graph. "
            "Returns query results as JSON. Use this for searching and retrieving data from graphs.\n\n"
            "IMPORTANT - Use these exact namespace prefixes:\n"
            "  PREFIX doc: <http://mnemosyne.dev/doc#>\n"
            "  PREFIX dcterms: <http://purl.org/dc/terms/>\n"
            "  PREFIX nfo: <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>\n"
            "  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            "Do NOT use 'urn:mnemosyne:schema:doc:' - it will match nothing.\n\n"
            "The graph_id is automatically resolved to a graph URI and injected as a FROM clause. "
            "Common types: doc:TipTapDocument, doc:Folder, doc:Artifact. "
            "Common predicates: dcterms:title, nfo:fileName, nfo:belongsToContainer, doc:order, doc:section.\n\n"
            "NOT for: discovering what documents exist (use get_workspace instead — it's always complete). "
            "SPARQL is best for searching content across documents (doc:content triples), "
            "querying block structure, and counting entities. Document-level metadata may be incomplete in RDF."
        ),
    )
    async def sparql_query_tool(
        graph_id: str,
        sparql: str,
        result_format: str = "json",
        context: Context | None = None,
    ) -> str:
        """Execute a SPARQL SELECT/CONSTRUCT query against a specific graph."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required to scope the query")
        if not sparql or not sparql.strip():
            raise ValueError("sparql query is required and cannot be empty")

        # Extract user_id from auth context or token (same pattern as hocuspocus tools)
        user_id = auth.user_id or (get_user_id_from_token(auth.token) if auth.token else None)
        if not user_id:
            raise ValueError(
                "Could not determine user ID from authentication context. "
                "Ensure your token contains a 'sub' claim or set MNEMOSYNE_DEV_USER_ID."
            )

        # Build the graph URI and inject FROM clause if not already present
        graph_id = graph_id.strip()
        sparql = sparql.strip()
        graph_uri = f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"

        # Check if query already has FROM clause (case-insensitive)
        sparql_upper = sparql.upper()
        if "FROM <" not in sparql_upper and "FROM NAMED" not in sparql_upper:
            # Inject FROM clause before WHERE (SPARQL syntax: SELECT vars FROM <graph> WHERE)
            import re
            # Find WHERE clause position and insert FROM before it
            where_match = re.search(r"\bWHERE\s*\{", sparql, re.IGNORECASE)
            if where_match:
                insert_pos = where_match.start()
                sparql = f"{sparql[:insert_pos]}FROM <{graph_uri}> {sparql[insert_pos:]}"
            else:
                # Fallback: prepend FROM clause (may not work for all queries)
                logger.warning(
                    "sparql_query_from_injection_fallback",
                    extra_context={"graph_id": graph_id, "query_prefix": sparql[:50]},
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
            payload={
                "sparql": sparql,
                "result_format": result_format,
            },
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(
            job_stream, metadata, context, auth
        )

        # Extract query results from job output
        query_result = _extract_query_result(result)
        if query_result is not None:
            return _render_json({
                "success": True,
                "results": query_result,
                "job_id": metadata.job_id,
            })

        return _render_json({
            "success": True,
            "job_id": metadata.job_id,
            **result,
        })

    @server.tool(
        name="sparql_update",
        title="Run SPARQL Update",
        description=(
            "Executes a SPARQL INSERT, DELETE, or UPDATE operation to modify graph data. "
            "The graph_id parameter is required to scope the update to a specific graph. "
            "Use this for adding, modifying, or removing triples from graphs.\n\n"
            "IMPORTANT - Use these exact namespace prefixes:\n"
            "  PREFIX doc: <http://mnemosyne.dev/doc#>\n"
            "  PREFIX dcterms: <http://purl.org/dc/terms/>\n"
            "Do NOT use 'urn:mnemosyne:schema:doc:' - it will match nothing.\n\n"
            "The graph_id is automatically resolved and injected as a GRAPH or WITH clause. "
            "For INSERT DATA/DELETE DATA, the graph is wrapped automatically. "
            "For DELETE/INSERT WHERE patterns, a WITH clause is prepended."
        ),
    )
    async def sparql_update_tool(
        graph_id: str,
        sparql: str,
        context: Context | None = None,
    ) -> str:
        """Execute a SPARQL INSERT/DELETE/UPDATE operation against a specific graph."""
        auth = MCPAuthContext.from_context(context)
        auth.require_auth()

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required to scope the update")
        if not sparql or not sparql.strip():
            raise ValueError("sparql update is required and cannot be empty")

        # Extract user_id from auth context or token (same pattern as hocuspocus tools)
        user_id = auth.user_id or (get_user_id_from_token(auth.token) if auth.token else None)
        if not user_id:
            raise ValueError(
                "Could not determine user ID from authentication context. "
                "Ensure your token contains a 'sub' claim or set MNEMOSYNE_DEV_USER_ID."
            )

        # Build the graph URI for reference (updates typically use GRAPH clauses)
        graph_id = graph_id.strip()
        sparql = sparql.strip()
        graph_uri = f"urn:mnemosyne:user:{user_id}:graph:{graph_id}"

        # For updates, check if query references a graph - if not, wrap in GRAPH clause
        sparql_upper = sparql.upper()
        if "GRAPH <" not in sparql_upper and "WITH <" not in sparql_upper:
            # Check for INSERT DATA or DELETE DATA patterns
            if "INSERT DATA" in sparql_upper:
                # Replace INSERT DATA { with INSERT DATA { GRAPH <uri> {
                import re
                sparql = re.sub(
                    r"(INSERT\s+DATA\s*)\{",
                    rf"\1{{ GRAPH <{graph_uri}> {{",
                    sparql,
                    count=1,
                    flags=re.IGNORECASE,
                )
                # Add closing brace before final }
                sparql = sparql.rstrip()
                if sparql.endswith("}"):
                    sparql = sparql[:-1] + "} }"
            elif "DELETE DATA" in sparql_upper:
                import re
                sparql = re.sub(
                    r"(DELETE\s+DATA\s*)\{",
                    rf"\1{{ GRAPH <{graph_uri}> {{",
                    sparql,
                    count=1,
                    flags=re.IGNORECASE,
                )
                sparql = sparql.rstrip()
                if sparql.endswith("}"):
                    sparql = sparql[:-1] + "} }"
            else:
                # For other updates (INSERT/DELETE WHERE), prepend WITH clause
                sparql = f"WITH <{graph_uri}>\n{sparql}"

        if job_stream:
            try:
                await job_stream.ensure_ready()
            except Exception:
                pass

        metadata = await submit_job(
            base_url=backend_config.base_url,
            auth=auth,
            task_type="apply_update",
            payload={"sparql": sparql},
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(
            job_stream, metadata, context, auth
        )

        job_succeeded = result.get("status") != "failed"
        return _render_json({
            "success": job_succeeded,
            "job_id": metadata.job_id,
            **result,
        })

    logger.info("Registered graph operations tools (create, delete, query, update)")


async def _wait_for_job_result(
    job_stream: Optional[RealtimeJobClient],
    metadata: JobSubmitMetadata,
    context: Optional[Context],
    auth: MCPAuthContext,
) -> JsonDict:
    """Wait for job completion via WS+poll race, return result info including detail."""
    trace("  _wait_for_job_result: racing WS + poll")
    ws_events, poll_payload = await await_job_completion(
        job_stream, metadata, auth, timeout=STREAM_TIMEOUT_SECONDS,
    )

    # Try WS events first
    if ws_events:
        if context:
            await context.report_progress(80, 100)
        for event in reversed(ws_events):
            event_type = event.get("type", "")
            if event_type in ("job_completed", "completed", "succeeded"):
                if context:
                    await context.report_progress(100, 100)
                result: JsonDict = {"status": "succeeded", "events": len(ws_events)}
                payload = event.get("payload", {})
                if isinstance(payload, dict):
                    detail = payload.get("detail")
                    if detail:
                        result["detail"] = detail
                return result
            if event_type in ("failed", "error"):
                error = event.get("error", "Job failed")
                return {"status": "failed", "error": error}
        return {"status": "unknown", "event_count": len(ws_events)}

    # Try poll result
    if context:
        await context.report_progress(100, 100)

    if poll_payload:
        status = poll_payload.get("status", "unknown")
        detail = poll_payload.get("detail")
        if status == "failed":
            error = poll_payload.get("error") or (detail.get("error") if isinstance(detail, dict) else None)
            return {"status": "failed", "error": error}
        result: JsonDict = {"status": status}
        if detail:
            result["detail"] = detail
        return result

    return {"status": "unknown"}


def _extract_query_result(result: JsonDict) -> Optional[Any]:
    """Extract SPARQL query results from job output.

    The backend returns query results as:
    - detail.result_inline.raw for SPARQL SELECT/CONSTRUCT queries
    - detail.result_inline for other operations
    """
    if "detail" not in result or not isinstance(result["detail"], dict):
        return None

    detail = result["detail"]
    inline = detail.get("result_inline")
    if inline is None:
        return None

    # SPARQL query results are wrapped in {"raw": actual_result}
    if isinstance(inline, dict) and "raw" in inline:
        return inline["raw"]

    return inline


def _render_json(payload: JsonDict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
