"""Compatibility tests for workspace entry lookups."""

import pycrdt

from neem.hocuspocus.workspace import WorkspaceReader


def test_workspace_reader_get_document_accepts_plain_mapping_values() -> None:
    doc = pycrdt.Doc()
    doc.get("documents", type=pycrdt.Map)
    doc.get("folders", type=pycrdt.Map)
    reader = WorkspaceReader(doc)

    # Simulate a dict-like entry value in the workspace map.
    reader._documents = {  # type: ignore[assignment]
        "doc-1": {"title": "Transcript", "parentId": None, "readOnly": False},
    }

    entry = reader.get_document("doc-1")
    assert entry is not None
    assert entry["title"] == "Transcript"
    assert entry["readOnly"] is False


def test_workspace_reader_children_includes_plain_mapping_values() -> None:
    doc = pycrdt.Doc()
    doc.get("documents", type=pycrdt.Map)
    doc.get("folders", type=pycrdt.Map)
    reader = WorkspaceReader(doc)

    reader._folders = {  # type: ignore[assignment]
        "folder-1": {"label": "Notes", "parentId": None},
    }
    reader._documents = {  # type: ignore[assignment]
        "doc-1": {"title": "Transcript", "parentId": "folder-1"},
    }

    root_children = reader.get_children_of(None)
    assert ("folder", "folder-1", {"label": "Notes", "parentId": None}) in root_children

    folder_children = reader.get_children_of("folder-1")
    assert ("document", "doc-1", {"title": "Transcript", "parentId": "folder-1"}) in folder_children

