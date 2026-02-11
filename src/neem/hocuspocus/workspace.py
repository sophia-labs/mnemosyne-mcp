"""Workspace navigation helpers for Y.js collaboration.

Provides high-level operations for reading and modifying workspace navigation
state stored as Y.Maps (folders, documents, ui).

The workspace Y.Doc is separate from document content Y.Docs. It contains:
- folders: Map of folder entities (id -> {name, parentId, order, section})
- documents: Map of document navigation entries (id -> {title, parentId, order, readOnly, sf_*...})
- ui: Map of UI state (expanded folders, selections, etc.)

Uploaded files (artifacts) are documents with readOnly=true and sf_* metadata
(sf_fileType, sf_storageKey, sf_originalFilename, sf_mimeType, sf_sizeBytes).

This module provides:
- WorkspaceWriter: Create, update, and delete workspace entities
- WorkspaceReader: Read workspace entities and query children
"""

from __future__ import annotations

import time
from typing import Any

import pycrdt

from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("hocuspocus.workspace")

# Sentinel for "not provided" (distinct from None which means "set to null")
_UNSET = object()


class WorkspaceWriter:
    """Write operations for workspace navigation Y.Doc.

    Used to create, update, and delete entities (folders, documents)
    in the workspace navigation. This ensures changes made via MCP appear in
    the browser's file tree.

    Usage:
        # Within a transact_workspace call:
        await client.transact_workspace(graph_id, lambda doc:
            WorkspaceWriter(doc).upsert_document(doc_id, "My Document")
        )
        await client.transact_workspace(graph_id, lambda doc:
            WorkspaceWriter(doc).upsert_folder(folder_id, "My Folder")
        )
    """

    def __init__(self, doc: pycrdt.Doc) -> None:
        """Initialize the workspace writer.

        Args:
            doc: The workspace Y.Doc (contains folders, documents, ui maps)
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

    def update_document(
        self,
        doc_id: str,
        *,
        title: str | None = None,
        parent_id: Any = _UNSET,
        order: float | None = None,
    ) -> bool:
        """Update an existing document's properties.

        Args:
            doc_id: The document ID to update
            title: New title (if provided)
            parent_id: New parent folder ID (use None for root, _UNSET to not change)
            order: New sort order (if provided)

        Returns:
            True if document was updated, False if not found.
        """
        doc_map = self._documents.get(doc_id)
        if not isinstance(doc_map, pycrdt.Map):
            return False

        now = time.time() * 1000
        if title is not None:
            doc_map["title"] = title
        if parent_id is not _UNSET:
            doc_map["parentId"] = parent_id
        if order is not None:
            doc_map["order"] = order
        doc_map["updatedAt"] = now

        logger.debug(
            "Updated document in workspace",
            extra_context={"doc_id": doc_id},
        )
        return True

    # -------------------------------------------------------------------------
    # Folder Operations
    # -------------------------------------------------------------------------

    def get_folder(self, folder_id: str) -> dict[str, Any] | None:
        """Get a folder entry from workspace navigation.

        Args:
            folder_id: The folder ID

        Returns:
            Folder data dict or None if not found.
        """
        folder_map = self._folders.get(folder_id)
        if isinstance(folder_map, pycrdt.Map):
            return {key: folder_map.get(key) for key in folder_map.keys()}
        return None

    def folder_exists(self, folder_id: str) -> bool:
        """Check if a folder exists in workspace navigation.

        Args:
            folder_id: The folder ID

        Returns:
            True if folder exists in workspace.
        """
        return folder_id in self._folders

    def upsert_folder(
        self,
        folder_id: str,
        name: str,
        *,
        parent_id: str | None = None,
        section: str = "documents",
        order: float | None = None,
    ) -> None:
        """Create or update a folder entry in workspace navigation.

        If the folder already exists, updates its name and optionally other fields.
        If it doesn't exist, creates a new entry.

        Args:
            folder_id: The folder ID
            name: Display name for the folder
            parent_id: Parent folder ID (None for root level)
            section: Sidebar section ("documents" or "artifacts")
            order: Sort order within parent (defaults to current timestamp)
        """
        now = time.time() * 1000

        existing = self._folders.get(folder_id)

        if isinstance(existing, pycrdt.Map):
            # Update existing folder
            existing["name"] = name
            existing["updatedAt"] = now
            if parent_id is not None or existing.get("parentId") is None:
                existing["parentId"] = parent_id
            if section:
                existing["section"] = section
            if order is not None:
                existing["order"] = order
            logger.debug(
                "Updated folder in workspace",
                extra_context={"folder_id": folder_id, "name": name},
            )
        else:
            # Create new folder entry
            folder_data: dict[str, Any] = {
                "name": name,
                "parentId": parent_id,
                "section": section,
                "order": order if order is not None else now,
                "createdAt": now,
                "updatedAt": now,
            }
            folder_map = pycrdt.Map(folder_data)
            self._folders[folder_id] = folder_map
            logger.debug(
                "Created folder in workspace",
                extra_context={"folder_id": folder_id, "name": name},
            )

    def update_folder(
        self,
        folder_id: str,
        *,
        name: str | None = None,
        parent_id: Any = _UNSET,
        order: float | None = None,
    ) -> bool:
        """Update an existing folder's properties.

        Args:
            folder_id: The folder ID to update
            name: New name (if provided)
            parent_id: New parent folder ID (use None for root, _UNSET to not change)
            order: New sort order (if provided)

        Returns:
            True if folder was updated, False if not found.
        """
        folder_map = self._folders.get(folder_id)
        if not isinstance(folder_map, pycrdt.Map):
            return False

        now = time.time() * 1000
        if name is not None:
            folder_map["name"] = name
        if parent_id is not _UNSET:
            folder_map["parentId"] = parent_id
        if order is not None:
            folder_map["order"] = order
        folder_map["updatedAt"] = now

        logger.debug(
            "Updated folder in workspace",
            extra_context={"folder_id": folder_id},
        )
        return True

    def delete_folder(self, folder_id: str, cascade: bool = False) -> bool:
        """Delete a folder from workspace navigation.

        Args:
            folder_id: The folder ID to delete
            cascade: If True, delete all children (subfolders, artifacts, documents).
                     If False, raises ValueError if folder has children.

        Returns:
            True if folder was deleted, False if not found.

        Raises:
            ValueError: If cascade=False and folder has children.
        """
        if folder_id not in self._folders:
            return False

        children = self._get_children_of(folder_id)
        if children and not cascade:
            raise ValueError(
                f"Folder has {len(children)} children. Use cascade=True to delete."
            )

        if cascade:
            self._delete_children_recursive(folder_id)

        del self._folders[folder_id]
        logger.debug(
            "Deleted folder from workspace",
            extra_context={"folder_id": folder_id, "cascade": cascade},
        )
        return True

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_children_of(self, parent_id: str) -> list[tuple[str, str]]:
        """Get all entities that have this parent.

        Args:
            parent_id: The parent folder ID

        Returns:
            List of (entity_type, entity_id) tuples for all children.
        """
        children: list[tuple[str, str]] = []

        # Check folders
        for fid in self._folders.keys():
            folder = self._folders.get(fid)
            if isinstance(folder, pycrdt.Map) and folder.get("parentId") == parent_id:
                children.append(("folder", fid))

        # Check documents (includes uploaded files / artifacts)
        for did in self._documents.keys():
            doc = self._documents.get(did)
            if isinstance(doc, pycrdt.Map) and doc.get("parentId") == parent_id:
                children.append(("document", did))

        return children

    def _delete_children_recursive(self, parent_id: str) -> None:
        """Recursively delete all children of a folder.

        Args:
            parent_id: The parent folder ID whose children should be deleted.
        """
        children = self._get_children_of(parent_id)
        for entity_type, entity_id in children:
            if entity_type == "folder":
                # Recursively delete subfolder children first
                self._delete_children_recursive(entity_id)
                del self._folders[entity_id]
            elif entity_type == "document":
                del self._documents[entity_id]


class WorkspaceReader:
    """Read-only operations for workspace navigation Y.Doc.

    Used to query workspace state without modifications. Useful for checking
    current entity state before making changes.

    Usage:
        channel = hp_client.get_workspace_channel(graph_id)
        reader = WorkspaceReader(channel.doc)
        folder = reader.get_folder(folder_id)
    """

    def __init__(self, doc: pycrdt.Doc) -> None:
        """Initialize the workspace reader.

        Args:
            doc: The workspace Y.Doc (contains folders, documents, ui maps)
        """
        self._doc = doc
        self._documents: pycrdt.Map = doc.get("documents", type=pycrdt.Map)
        self._folders: pycrdt.Map = doc.get("folders", type=pycrdt.Map)

    def folder_exists(self, folder_id: str) -> bool:
        """Check if a folder exists in workspace navigation.

        Args:
            folder_id: The folder ID

        Returns:
            True if folder exists in workspace.
        """
        return folder_id in self._folders

    def get_folder(self, folder_id: str) -> dict[str, Any] | None:
        """Get a folder entry from workspace navigation.

        Args:
            folder_id: The folder ID

        Returns:
            Folder data dict or None if not found.
        """
        folder_map = self._folders.get(folder_id)
        if isinstance(folder_map, pycrdt.Map):
            return {key: folder_map.get(key) for key in folder_map.keys()}
        return None

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

    def get_children_of(self, parent_id: str | None) -> list[tuple[str, str, dict[str, Any]]]:
        """Get all entities that have this parent.

        Args:
            parent_id: The parent folder ID (None for root-level entities)

        Returns:
            List of (entity_type, entity_id, entity_data) tuples for all children.
        """
        children: list[tuple[str, str, dict[str, Any]]] = []

        # Check folders
        for fid in self._folders.keys():
            folder = self._folders.get(fid)
            if isinstance(folder, pycrdt.Map) and folder.get("parentId") == parent_id:
                data = {key: folder.get(key) for key in folder.keys()}
                children.append(("folder", fid, data))

        # Check documents (includes uploaded files / artifacts with readOnly=true)
        for did in self._documents.keys():
            doc = self._documents.get(did)
            if isinstance(doc, pycrdt.Map) and doc.get("parentId") == parent_id:
                data = {key: doc.get(key) for key in doc.keys()}
                children.append(("document", did, data))

        return children


__all__ = ["WorkspaceWriter", "WorkspaceReader"]
