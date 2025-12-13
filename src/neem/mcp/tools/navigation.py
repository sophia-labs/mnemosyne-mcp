"""
MCP tools for workspace navigation operations.

These tools provide file system-like operations for managing folders,
moving documents/artifacts, and organizing the workspace hierarchy.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import Context, FastMCP

from neem.mcp.jobs import JobSubmitMetadata, RealtimeJobClient
from neem.mcp.tools.basic import poll_job_until_terminal, stream_job, submit_job
from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import get_dev_user_id, validate_token_and_load

logger = LoggerFactory.get_logger("mcp.tools.navigation")

STREAM_TIMEOUT_SECONDS = 60.0
HTTP_TIMEOUT = 30.0
JsonDict = Dict[str, Any]


def register_navigation_tools(server: FastMCP) -> None:
    """Register workspace navigation tools for file system operations."""

    backend_config = getattr(server, "_backend_config", None)
    if backend_config is None:
        logger.warning("Backend config missing; skipping navigation tool registration")
        return

    job_stream: Optional[RealtimeJobClient] = getattr(server, "_job_stream", None)

    # -------------------------------------------------------------------------
    # Folder Operations
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
    ) -> str:
        """Create a new folder in the workspace."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")
        if not label or not label.strip():
            raise ValueError("label is required and cannot be empty")
        if section not in ("documents", "artifacts"):
            raise ValueError("section must be 'documents' or 'artifacts'")

        folder_data: JsonDict = {
            "label": label.strip(),
            "section": section,
        }
        if parent_id:
            folder_data["parentId"] = parent_id.strip()
        if order is not None:
            folder_data["order"] = order

        metadata = await submit_job(
            base_url=backend_config.base_url,
            token=token,
            task_type="folder_put",
            payload={
                "graph_id": graph_id.strip(),
                "folder_id": folder_id.strip(),
                "folder": folder_data,
            },
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(job_stream, metadata, context, token)

        return _render_json({
            "success": True,
            "folder_id": folder_id.strip(),
            "graph_id": graph_id.strip(),
            "label": label.strip(),
            "parent_id": parent_id.strip() if parent_id else None,
            "section": section,
            "job_id": metadata.job_id,
            **result,
        })

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
    ) -> str:
        """Move a folder to a new parent."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")

        # First, get the current folder to preserve its label and section
        current = await _get_entity(
            backend_config.base_url,
            token,
            graph_id.strip(),
            "folder",
            folder_id.strip(),
        )

        if not current:
            raise RuntimeError(f"Folder '{folder_id}' not found in graph '{graph_id}'")

        payload: JsonDict = {
            "label": current.get("label", folder_id),
            "section": current.get("section", "documents"),
        }
        if new_parent_id is not None:
            payload["parentId"] = new_parent_id.strip() if new_parent_id else None
        elif "parentId" in current:
            payload["parentId"] = current["parentId"]

        if new_order is not None:
            payload["order"] = new_order
        elif "order" in current:
            payload["order"] = current["order"]

        metadata = await submit_job(
            base_url=backend_config.base_url,
            token=token,
            task_type="folder_put",
            payload={
                "graph_id": graph_id.strip(),
                "folder_id": folder_id.strip(),
                "folder": payload,
            },
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(job_stream, metadata, context, token)

        return _render_json({
            "success": True,
            "folder_id": folder_id.strip(),
            "graph_id": graph_id.strip(),
            "new_parent_id": new_parent_id.strip() if new_parent_id else None,
            "job_id": metadata.job_id,
            **result,
        })

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
    ) -> str:
        """Rename a folder."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")
        if not new_label or not new_label.strip():
            raise ValueError("new_label is required and cannot be empty")

        # Get current folder to preserve other properties
        current = await _get_entity(
            backend_config.base_url,
            token,
            graph_id.strip(),
            "folder",
            folder_id.strip(),
        )

        if not current:
            raise RuntimeError(f"Folder '{folder_id}' not found in graph '{graph_id}'")

        payload: JsonDict = {
            "label": new_label.strip(),
            "section": current.get("section", "documents"),
        }
        if "parentId" in current and current["parentId"]:
            payload["parentId"] = current["parentId"]
        if "order" in current:
            payload["order"] = current["order"]

        metadata = await submit_job(
            base_url=backend_config.base_url,
            token=token,
            task_type="folder_put",
            payload={
                "graph_id": graph_id.strip(),
                "folder_id": folder_id.strip(),
                "folder": payload,
            },
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(job_stream, metadata, context, token)

        return _render_json({
            "success": True,
            "folder_id": folder_id.strip(),
            "graph_id": graph_id.strip(),
            "new_label": new_label.strip(),
            "job_id": metadata.job_id,
            **result,
        })

    @server.tool(
        name="delete_folder",
        title="Delete Folder",
        description=(
            "Delete a folder from the workspace. "
            "Set cascade=true to delete all contents (subfolders, documents, artifacts). "
            "Without cascade, deletion fails if the folder has children."
        ),
    )
    async def delete_folder_tool(
        graph_id: str,
        folder_id: str,
        cascade: bool = False,
        context: Context | None = None,
    ) -> str:
        """Delete a folder."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not folder_id or not folder_id.strip():
            raise ValueError("folder_id is required and cannot be empty")

        # Note: cascade is not yet supported by backend, but we keep the param
        # for future implementation
        metadata = await submit_job(
            base_url=backend_config.base_url,
            token=token,
            task_type="folder_delete",
            payload={
                "graph_id": graph_id.strip(),
                "folder_id": folder_id.strip(),
            },
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(job_stream, metadata, context, token)

        return _render_json({
            "success": True,
            "deleted": True,
            "folder_id": folder_id.strip(),
            "graph_id": graph_id.strip(),
            "job_id": metadata.job_id,
            **result,
        })

    # -------------------------------------------------------------------------
    # Artifact Operations
    # -------------------------------------------------------------------------

    @server.tool(
        name="move_artifact",
        title="Move Artifact",
        description=(
            "Move an artifact to a different folder. "
            "Set new_parent_id to null to move to root level."
        ),
    )
    async def move_artifact_tool(
        graph_id: str,
        artifact_id: str,
        new_parent_id: Optional[str] = None,
        new_order: Optional[float] = None,
        context: Context | None = None,
    ) -> str:
        """Move an artifact to a different folder."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not artifact_id or not artifact_id.strip():
            raise ValueError("artifact_id is required and cannot be empty")

        # Get current artifact to preserve other properties
        current = await _get_entity(
            backend_config.base_url,
            token,
            graph_id.strip(),
            "artifact",
            artifact_id.strip(),
        )

        if not current:
            raise RuntimeError(f"Artifact '{artifact_id}' not found in graph '{graph_id}'")

        # Build payload preserving all existing properties
        payload: JsonDict = {
            "label": current.get("label", artifact_id),
            "originalFilename": current.get("originalFilename", current.get("label", artifact_id)),
            "fileType": current.get("fileType", "unknown"),
            "status": current.get("status", "ready"),
        }

        # Update parent
        if new_parent_id is not None:
            payload["parentId"] = new_parent_id.strip() if new_parent_id else None
        elif "parentId" in current:
            payload["parentId"] = current["parentId"]

        # Update order
        if new_order is not None:
            payload["order"] = new_order
        elif "order" in current:
            payload["order"] = current["order"]

        # Preserve optional fields
        for field in ("storageKey", "mimeType", "sizeBytes", "errorMessage"):
            if field in current and current[field] is not None:
                payload[field] = current[field]

        metadata = await submit_job(
            base_url=backend_config.base_url,
            token=token,
            task_type="artifact_put",
            payload={
                "graph_id": graph_id.strip(),
                "artifact_id": artifact_id.strip(),
                "artifact": payload,
            },
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(job_stream, metadata, context, token)

        return _render_json({
            "success": True,
            "artifact_id": artifact_id.strip(),
            "graph_id": graph_id.strip(),
            "new_parent_id": new_parent_id.strip() if new_parent_id else None,
            "job_id": metadata.job_id,
            **result,
        })

    @server.tool(
        name="rename_artifact",
        title="Rename Artifact",
        description="Rename an artifact's display label.",
    )
    async def rename_artifact_tool(
        graph_id: str,
        artifact_id: str,
        new_label: str,
        context: Context | None = None,
    ) -> str:
        """Rename an artifact."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not artifact_id or not artifact_id.strip():
            raise ValueError("artifact_id is required and cannot be empty")
        if not new_label or not new_label.strip():
            raise ValueError("new_label is required and cannot be empty")

        # Get current artifact to preserve other properties
        current = await _get_entity(
            backend_config.base_url,
            token,
            graph_id.strip(),
            "artifact",
            artifact_id.strip(),
        )

        if not current:
            raise RuntimeError(f"Artifact '{artifact_id}' not found in graph '{graph_id}'")

        # Build payload with new label
        payload: JsonDict = {
            "label": new_label.strip(),
            "originalFilename": current.get("originalFilename", current.get("label", artifact_id)),
            "fileType": current.get("fileType", "unknown"),
            "status": current.get("status", "ready"),
        }

        # Preserve existing properties
        if "parentId" in current and current["parentId"]:
            payload["parentId"] = current["parentId"]
        if "order" in current:
            payload["order"] = current["order"]
        for field in ("storageKey", "mimeType", "sizeBytes", "errorMessage"):
            if field in current and current[field] is not None:
                payload[field] = current[field]

        metadata = await submit_job(
            base_url=backend_config.base_url,
            token=token,
            task_type="artifact_put",
            payload={
                "graph_id": graph_id.strip(),
                "artifact_id": artifact_id.strip(),
                "artifact": payload,
            },
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(job_stream, metadata, context, token)

        return _render_json({
            "success": True,
            "artifact_id": artifact_id.strip(),
            "graph_id": graph_id.strip(),
            "new_label": new_label.strip(),
            "job_id": metadata.job_id,
            **result,
        })

    # -------------------------------------------------------------------------
    # Document Operations
    # -------------------------------------------------------------------------

    @server.tool(
        name="move_document",
        title="Move Document",
        description=(
            "Move a document to a folder. "
            "Set new_parent_id to null to move to root level (unfiled). "
            "Note: This updates the document's folder assignment."
        ),
    )
    async def move_document_tool(
        graph_id: str,
        document_id: str,
        new_parent_id: Optional[str] = None,
        context: Context | None = None,
    ) -> str:
        """Move a document to a folder."""
        token = validate_token_and_load()
        if not token:
            raise RuntimeError("Not authenticated. Run `neem init` to refresh your token.")

        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id is required and cannot be empty")
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required and cannot be empty")

        # Get current document to preserve other properties
        current = await _get_entity(
            backend_config.base_url,
            token,
            graph_id.strip(),
            "document",
            document_id.strip(),
        )

        if not current:
            raise RuntimeError(f"Document '{document_id}' not found in graph '{graph_id}'")

        # Build payload with new parent
        payload: JsonDict = {
            "title": current.get("title", document_id),
            "blocks": current.get("blocks", []),
        }

        # Set the new parent
        if new_parent_id is not None:
            payload["parentId"] = new_parent_id.strip() if new_parent_id else None

        # Include expected revision for optimistic concurrency
        if "revision" in current:
            payload["expectedRevision"] = current["revision"]

        metadata = await submit_job(
            base_url=backend_config.base_url,
            token=token,
            task_type="doc_put",
            payload={
                "graph_id": graph_id.strip(),
                "doc_id": document_id.strip(),
                "document": payload,
            },
        )

        if context:
            await context.report_progress(10, 100)

        result = await _wait_for_job_result(job_stream, metadata, context, token)

        return _render_json({
            "success": True,
            "document_id": document_id.strip(),
            "graph_id": graph_id.strip(),
            "new_parent_id": new_parent_id.strip() if new_parent_id else None,
            "job_id": metadata.job_id,
            **result,
        })

    logger.info("Registered navigation tools (folder, artifact, document operations)")


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

async def _get_entity(
    base_url: str,
    token: str,
    graph_id: str,
    entity_type: str,
    entity_id: str,
) -> Optional[JsonDict]:
    """Fetch an entity (folder, artifact, document) by ID.

    Uses the navigation endpoints for folders/artifacts and documents endpoint
    for documents.
    """
    if entity_type == "document":
        url = f"{base_url.rstrip('/')}/documents/{graph_id}/{entity_id}"
    elif entity_type == "folder":
        url = f"{base_url.rstrip('/')}/navigation/{graph_id}/folders/{entity_id}"
    elif entity_type == "artifact":
        url = f"{base_url.rstrip('/')}/navigation/{graph_id}/artifacts/{entity_id}"
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")

    headers = {"Authorization": f"Bearer {token}"}
    dev_user = get_dev_user_id()
    if dev_user:
        headers["X-User-ID"] = dev_user

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()

            data = resp.json()

            # Handle async job response (202 Accepted)
            if resp.status_code == 202 and "job_id" in data:
                # Poll for result
                from neem.mcp.tools.basic import poll_job_until_terminal
                status_url = data.get("links", {}).get("status")
                if status_url:
                    result = await poll_job_until_terminal(status_url, token)
                    if result:
                        detail = result.get("detail", {})
                        return detail.get("result_inline", detail)
                return None

            return data
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        logger.error(
            f"Failed to get {entity_type}",
            extra_context={
                "graph_id": graph_id,
                "entity_id": entity_id,
                "status_code": e.response.status_code,
            },
        )
        raise


async def _wait_for_job_result(
    job_stream: Optional[RealtimeJobClient],
    metadata: JobSubmitMetadata,
    context: Optional[Context],
    token: str,
) -> JsonDict:
    """Wait for job completion via WebSocket or polling, return result info."""
    events = None
    if job_stream and metadata.links.websocket:
        events = await stream_job(job_stream, metadata, timeout=STREAM_TIMEOUT_SECONDS)

    if events:
        if context:
            await context.report_progress(80, 100)
        # Check for completion status in events
        for event in reversed(events):
            event_type = event.get("type", "")
            if event_type in ("job_completed", "completed", "succeeded"):
                if context:
                    await context.report_progress(100, 100)
                result: JsonDict = {"status": "succeeded", "events": len(events)}
                payload = event.get("payload", {})
                if isinstance(payload, dict):
                    detail = payload.get("detail")
                    if detail:
                        result["detail"] = detail
                return result
            if event_type in ("failed", "error"):
                error = event.get("error", "Job failed")
                return {"status": "failed", "error": error}
        return {"status": "unknown", "event_count": len(events)}

    # Fall back to polling
    status_payload = (
        await poll_job_until_terminal(metadata.links.status, token)
        if metadata.links.status
        else None
    )

    if context:
        await context.report_progress(100, 100)

    if status_payload:
        status = status_payload.get("status", "unknown")
        detail = status_payload.get("detail")
        if status == "failed":
            error = status_payload.get("error") or (
                detail.get("error") if isinstance(detail, dict) else None
            )
            return {"status": "failed", "error": error}
        result: JsonDict = {"status": status}
        if detail:
            result["detail"] = detail
        return result

    return {"status": "unknown"}


def _render_json(payload: JsonDict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
