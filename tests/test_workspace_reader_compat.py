"""Compatibility tests for workspace entry lookups."""

import pycrdt

from neem.hocuspocus.workspace import WorkspaceReader, WorkspaceWriter, _resolve_document_key


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


# Regression coverage for the doc-<uuid> lookup bug (root-caused 2026-07-15).
#
# A document created via the web UI and keyed by the full `doc-<uuid>` form
# (the common case) used to read as "not found" for any MCP tool, because
# normalize_document_id_for_lookup stripped the prefix before any lookup
# happened. Fixed by making the actual lookup point (here) try both forms
# instead of guessing blindly at the tool-entry point.

_UUID = "511ddcc8-410f-4234-9374-7951fc600f15"


def _fresh_doc() -> pycrdt.Doc:
    doc = pycrdt.Doc()
    doc.get("documents", type=pycrdt.Map)
    doc.get("folders", type=pycrdt.Map)
    return doc


def test_resolve_document_key_matches_exact_form_first() -> None:
    docs = {f"doc-{_UUID}": {"title": "shrubbery bugs and annoyances"}}
    assert _resolve_document_key(docs, f"doc-{_UUID}") == f"doc-{_UUID}"


def test_resolve_document_key_falls_back_to_prefixed_form() -> None:
    # Entry keyed with the doc- prefix; caller queries with the bare UUID.
    docs = {f"doc-{_UUID}": {"title": "shrubbery bugs and annoyances"}}
    assert _resolve_document_key(docs, _UUID) == f"doc-{_UUID}"


def test_resolve_document_key_falls_back_to_bare_form() -> None:
    # Entry keyed bare (e.g. "poems"-style docs); caller queries prefixed.
    docs = {_UUID: {"title": "poems"}}
    assert _resolve_document_key(docs, f"doc-{_UUID}") == _UUID


def test_resolve_document_key_returns_none_when_absent() -> None:
    docs = {f"doc-{_UUID}": {"title": "shrubbery bugs and annoyances"}}
    assert _resolve_document_key(docs, "doc-00000000-0000-0000-0000-000000000000") is None


def test_resolve_document_key_does_not_touch_slugs() -> None:
    docs = {"garden-design-antinomies": {"title": "The Garden Design Antinomies"}}
    assert _resolve_document_key(docs, "garden-design-antinomies") == "garden-design-antinomies"
    assert _resolve_document_key(docs, "garden-design") is None


def test_workspace_reader_get_document_finds_prefixed_entry_via_bare_query() -> None:
    doc = _fresh_doc()
    reader = WorkspaceReader(doc)
    reader._documents = {  # type: ignore[assignment]
        f"doc-{_UUID}": {"title": "shrubbery bugs and annoyances"},
    }

    # This is the exact repro: get_workspace/search_documents return the
    # doc-prefixed form; a caller passing that form through unchanged must
    # resolve on the first try (the common, correct case).
    assert reader.get_document(f"doc-{_UUID}") is not None
    # And a caller who (for whatever reason) strips it first still resolves.
    assert reader.get_document(_UUID) is not None


def test_workspace_writer_document_exists_resolves_doc_prefixed_uuid() -> None:
    doc = _fresh_doc()
    writer = WorkspaceWriter(doc)
    writer.upsert_document(f"doc-{_UUID}", "shrubbery bugs and annoyances")

    assert writer.document_exists(f"doc-{_UUID}") is True
    assert writer.document_exists(_UUID) is True


def test_workspace_writer_upsert_updates_existing_doc_prefixed_entry_in_place() -> None:
    """upsert_document must not create a duplicate entry under the bare
    form when the canonical entry is doc-<uuid>-prefixed."""
    doc = _fresh_doc()
    writer = WorkspaceWriter(doc)
    reader = WorkspaceReader(doc)
    writer.upsert_document(f"doc-{_UUID}", "original title")

    writer.upsert_document(f"doc-{_UUID}", "updated title")

    documents_map = doc.get("documents", type=pycrdt.Map)
    assert len(list(documents_map.keys())) == 1
    assert reader.get_document(f"doc-{_UUID}")["title"] == "updated title"


def test_workspace_writer_delete_document_removes_doc_prefixed_entry_via_bare_id() -> None:
    doc = _fresh_doc()
    writer = WorkspaceWriter(doc)
    writer.upsert_document(f"doc-{_UUID}", "shrubbery bugs and annoyances")

    assert writer.delete_document(_UUID) is True
    assert writer.document_exists(f"doc-{_UUID}") is False

