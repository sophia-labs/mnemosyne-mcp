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
        self._wires: pycrdt.Map = doc.get("wires", type=pycrdt.Map)

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
            deleted_wires = self._delete_wires_for_document(doc_id)
            del self._documents[doc_id]
            logger.debug(
                "Deleted document from workspace",
                extra_context={"doc_id": doc_id, "deleted_wires": deleted_wires},
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

    def delete_wire(self, wire_id: str) -> list[str]:
        """Delete a wire by ID, including its inverse if bidirectional.

        Args:
            wire_id: The wire ID to delete (e.g. "w-a3f1b9c20d4e").

        Returns:
            List of wire IDs that were deleted (includes inverse if present).

        Raises:
            ValueError: If wire_id not found.
        """
        if wire_id not in self._wires:
            raise ValueError(f"Wire '{wire_id}' not found")

        deleted_ids: list[str] = []

        # If this IS an inverse wire, also delete the canonical
        if wire_id.endswith("-inv"):
            canonical_id = wire_id[:-4]
            if canonical_id in self._wires:
                del self._wires[canonical_id]
                deleted_ids.append(canonical_id)
        else:
            # Check if this wire has an inverse (bidirectional)
            inv_id = f"{wire_id}-inv"
            if inv_id in self._wires:
                del self._wires[inv_id]
                deleted_ids.append(inv_id)

        del self._wires[wire_id]
        deleted_ids.append(wire_id)

        return deleted_ids

    def delete_wires_matching(
        self, document_id: str, block_id: str | None = None
    ) -> list[str]:
        """Delete all wires connected to a document or block.

        Finds all wires where the document (and optionally block) appears as
        source or target, then deletes them along with their bidirectional
        inverses. Deduplicates so no wire is deleted twice.

        Args:
            document_id: Delete wires connected to this document.
            block_id: If provided, only delete wires where this block is the
                source or target block. Without this, all wires touching the
                document are deleted.

        Returns:
            List of all wire IDs that were deleted.
        """
        # Collect candidate wire IDs
        candidates: set[str] = set()
        for wire_id in list(self._wires.keys()):
            wire = self._wires.get(wire_id)
            if not isinstance(wire, pycrdt.Map):
                continue

            src_doc = wire.get("sourceDocumentId")
            tgt_doc = wire.get("targetDocumentId")

            if src_doc != document_id and tgt_doc != document_id:
                continue

            if block_id is not None:
                src_block = wire.get("sourceBlockId")
                tgt_block = wire.get("targetBlockId")
                # Wire must touch this block on the matching document side
                doc_matches_src = src_doc == document_id and src_block == block_id
                doc_matches_tgt = tgt_doc == document_id and tgt_block == block_id
                if not doc_matches_src and not doc_matches_tgt:
                    continue

            candidates.add(wire_id)

        # Expand to include bidirectional inverses, then delete
        to_delete: set[str] = set()
        for wire_id in candidates:
            to_delete.add(wire_id)
            if wire_id.endswith("-inv"):
                to_delete.add(wire_id[:-4])
            else:
                to_delete.add(f"{wire_id}-inv")

        deleted: list[str] = []
        for wire_id in to_delete:
            if wire_id in self._wires:
                del self._wires[wire_id]
                deleted.append(wire_id)

        return deleted

    def _delete_wires_for_document(self, doc_id: str) -> int:
        """Delete all workspace wires connected to a document.

        Args:
            doc_id: Document ID whose related wires should be removed.

        Returns:
            Number of wires removed.
        """
        wire_ids_to_delete: list[str] = []
        for wire_id in list(self._wires.keys()):
            wire = self._wires.get(wire_id)
            if not isinstance(wire, pycrdt.Map):
                continue

            if wire.get("sourceDocumentId") == doc_id or wire.get("targetDocumentId") == doc_id:
                wire_ids_to_delete.append(wire_id)

        for wire_id in wire_ids_to_delete:
            if wire_id in self._wires:
                del self._wires[wire_id]

        return len(wire_ids_to_delete)

    # ------------------------------------------------------------------
    # Wire tombstoning (undo-safe block-level wire cleanup)
    # ------------------------------------------------------------------

    def tombstone_wires_for_block(
        self, document_id: str, block_id: str, grace_seconds: float = 60.0,
    ) -> list[str]:
        """Mark wires connected to a block as tombstoned instead of deleting.

        Tombstoned wires have a ``_tombstonedAt`` timestamp. They are hidden
        from normal queries and will be permanently deleted by
        ``sweep_tombstoned_wires`` after the grace period expires.  If the
        block is restored (e.g. via undo), call ``restore_wires_for_block``
        to clear the tombstone and bring them back.

        Args:
            document_id: Document containing the deleted block.
            block_id: Block that was deleted.
            grace_seconds: Seconds before tombstoned wires become eligible
                for permanent deletion (default 60 s).

        Returns:
            List of wire IDs that were tombstoned.
        """
        now = time.time()
        tombstoned: list[str] = []

        for wire_id in list(self._wires.keys()):
            wire = self._wires.get(wire_id)
            if not isinstance(wire, pycrdt.Map):
                continue

            src_doc = wire.get("sourceDocumentId")
            tgt_doc = wire.get("targetDocumentId")
            src_block = wire.get("sourceBlockId")
            tgt_block = wire.get("targetBlockId")

            touches_block = (
                (src_doc == document_id and src_block == block_id) or
                (tgt_doc == document_id and tgt_block == block_id)
            )
            if not touches_block:
                continue

            # Already tombstoned — skip
            if wire.get("_tombstonedAt") is not None:
                continue

            wire["_tombstonedAt"] = now
            tombstoned.append(wire_id)

            # Also tombstone the inverse if bidirectional
            if wire_id.endswith("-inv"):
                canonical_id = wire_id[:-4]
            else:
                canonical_id = None
                inv_id = f"{wire_id}-inv"
                if inv_id in self._wires:
                    inv_wire = self._wires.get(inv_id)
                    if isinstance(inv_wire, pycrdt.Map) and inv_wire.get("_tombstonedAt") is None:
                        inv_wire["_tombstonedAt"] = now
                        tombstoned.append(inv_id)

            if canonical_id and canonical_id in self._wires:
                can_wire = self._wires.get(canonical_id)
                if isinstance(can_wire, pycrdt.Map) and can_wire.get("_tombstonedAt") is None:
                    can_wire["_tombstonedAt"] = now
                    tombstoned.append(canonical_id)

        if tombstoned:
            logger.debug(
                "Tombstoned wires for block",
                extra_context={
                    "document_id": document_id,
                    "block_id": block_id,
                    "tombstoned": tombstoned,
                },
            )

        return tombstoned

    def restore_wires_for_block(self, document_id: str, block_id: str) -> list[str]:
        """Clear tombstones on wires connected to a block (e.g. after undo).

        Args:
            document_id: Document containing the restored block.
            block_id: Block that was restored.

        Returns:
            List of wire IDs that were restored.
        """
        restored: list[str] = []

        for wire_id in list(self._wires.keys()):
            wire = self._wires.get(wire_id)
            if not isinstance(wire, pycrdt.Map):
                continue
            if wire.get("_tombstonedAt") is None:
                continue

            src_doc = wire.get("sourceDocumentId")
            tgt_doc = wire.get("targetDocumentId")
            src_block = wire.get("sourceBlockId")
            tgt_block = wire.get("targetBlockId")

            touches_block = (
                (src_doc == document_id and src_block == block_id) or
                (tgt_doc == document_id and tgt_block == block_id)
            )
            if not touches_block:
                continue

            del wire["_tombstonedAt"]
            restored.append(wire_id)

        if restored:
            logger.debug(
                "Restored tombstoned wires for block",
                extra_context={
                    "document_id": document_id,
                    "block_id": block_id,
                    "restored": restored,
                },
            )

        return restored

    def sweep_tombstoned_wires(self, grace_seconds: float = 60.0) -> list[str]:
        """Permanently delete wires whose tombstone has expired.

        Call this periodically (e.g. on workspace flush or session cleanup)
        to garbage-collect wires that were tombstoned and never restored.

        Args:
            grace_seconds: Only delete wires tombstoned longer than this.

        Returns:
            List of wire IDs permanently deleted.
        """
        now = time.time()
        to_delete: list[str] = []

        for wire_id in list(self._wires.keys()):
            wire = self._wires.get(wire_id)
            if not isinstance(wire, pycrdt.Map):
                continue
            ts = wire.get("_tombstonedAt")
            if ts is None:
                continue
            if now - ts >= grace_seconds:
                to_delete.append(wire_id)

        for wire_id in to_delete:
            if wire_id in self._wires:
                del self._wires[wire_id]

        if to_delete:
            logger.debug(
                "Swept tombstoned wires",
                extra_context={"deleted": to_delete, "grace_seconds": grace_seconds},
            )

        return to_delete

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
                self.delete_document(entity_id)


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
