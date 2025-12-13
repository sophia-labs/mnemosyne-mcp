"""Workspace navigation helpers for Y.js collaboration.

Provides high-level operations for reading and modifying workspace navigation
state stored as Y.Maps (folders, artifacts, documents, ui).

The workspace Y.Doc is separate from document content Y.Docs. It contains:
- folders: Map of folder entities (id -> {name, parentId, order, section})
- artifacts: Map of artifact entities (id -> {name, parentId, mimeType, status, order})
- documents: Map of document navigation entries (id -> {title, parentId, order, ...})
- ui: Map of UI state (expanded folders, selections, etc.)

This module provides WorkspaceWriter for creating/updating document entries
when documents are created via MCP.
"""

from __future__ import annotations

import time
from typing import Any

import pycrdt

from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("hocuspocus.workspace")


class WorkspaceWriter:
    """Write operations for workspace navigation Y.Doc.

    Used to create, update, and delete document entries in the workspace
    navigation. This ensures documents created via MCP appear in the
    browser's file tree.

    Usage:
        # Within a transact_workspace call:
        await client.transact_workspace(graph_id, lambda doc:
            WorkspaceWriter(doc).upsert_document(doc_id, "My Document")
        )
    """

    def __init__(self, doc: pycrdt.Doc) -> None:
        """Initialize the workspace writer.

        Args:
            doc: The workspace Y.Doc (contains folders, artifacts, documents, ui maps)
        """
        self._doc = doc
        self._documents: pycrdt.Map = doc.get("documents", type=pycrdt.Map)
        self._folders: pycrdt.Map = doc.get("folders", type=pycrdt.Map)

    def upsert_document(
        self,
        doc_id: str,
        title: str,
        *,
        parent_id: str | None = None,
        order: float | None = None,
    ) -> None:
        """Create or update a document entry in workspace navigation.

        If the document already exists, updates its title and optionally
        other fields. If it doesn't exist, creates a new entry.

        Args:
            doc_id: The document ID
            title: Display title for the document
            parent_id: Parent folder ID (None for root level)
            order: Sort order within parent (defaults to current timestamp)
        """
        now = time.time() * 1000  # Milliseconds for consistency with frontend

        existing = self._documents.get(doc_id)

        if isinstance(existing, pycrdt.Map):
            # Update existing document
            existing["title"] = title
            existing["updatedAt"] = now
            if parent_id is not None:
                existing["parentId"] = parent_id
            if order is not None:
                existing["order"] = order
            logger.debug(
                "Updated document in workspace",
                extra_context={"doc_id": doc_id, "title": title},
            )
        else:
            # Create new document entry
            doc_data: dict[str, Any] = {
                "title": title,
                "parentId": parent_id,
                "section": "documents",
                "order": order if order is not None else now,
                "createdAt": now,
                "updatedAt": now,
            }
            doc_map = pycrdt.Map(doc_data)
            self._documents[doc_id] = doc_map
            logger.debug(
                "Created document in workspace",
                extra_context={"doc_id": doc_id, "title": title},
            )

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from workspace navigation.

        Args:
            doc_id: The document ID to remove

        Returns:
            True if the document was removed, False if it didn't exist.
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            logger.debug(
                "Deleted document from workspace",
                extra_context={"doc_id": doc_id},
            )
            return True
        return False

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Get a document entry from workspace navigation.

        Args:
            doc_id: The document ID

        Returns:
            Document data dict or None if not found.
        """
        doc_map = self._documents.get(doc_id)
        if isinstance(doc_map, pycrdt.Map):
            return {key: doc_map.get(key) for key in doc_map.keys()}
        return None

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in workspace navigation.

        Args:
            doc_id: The document ID

        Returns:
            True if document exists in workspace.
        """
        return doc_id in self._documents


__all__ = ["WorkspaceWriter"]
