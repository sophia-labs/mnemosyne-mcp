"""Tests for search_blocks post-filter logic.

Covers: deleted-doc exclusion, per-document capping, diagnostic counters,
and graceful degradation when workspace snapshot is unavailable.
"""

from __future__ import annotations

from neem.mcp.tools.search import _post_filter_results, MAX_RESULTS_PER_DOCUMENT


def _make_result(doc_id: str, block_id: str, source: str = "lexical") -> dict:
    return {
        "document_id": doc_id,
        "block_id": block_id,
        "text": f"text from {doc_id}/{block_id}",
        "match_source": source,
    }


class TestDeletedDocFiltering:
    def test_filters_out_deleted_docs(self):
        results = [
            _make_result("doc-a", "b1"),
            _make_result("doc-deleted", "b2"),
            _make_result("doc-a", "b3"),
        ]
        known = {"doc-a"}
        filtered, removed, capped = _post_filter_results(
            results, known_doc_ids=known, total_limit=30,
        )
        assert len(filtered) == 2
        assert removed == 1
        assert all(r["document_id"] == "doc-a" for r in filtered)

    def test_all_docs_deleted_returns_empty(self):
        results = [
            _make_result("doc-gone", "b1"),
            _make_result("doc-gone", "b2"),
        ]
        filtered, removed, capped = _post_filter_results(
            results, known_doc_ids=set(), total_limit=30,
        )
        assert filtered == []
        assert removed == 2

    def test_no_workspace_snapshot_skips_deletion_filter(self):
        results = [
            _make_result("doc-a", "b1"),
            _make_result("doc-maybe-deleted", "b2"),
        ]
        filtered, removed, capped = _post_filter_results(
            results, known_doc_ids=None, total_limit=30,
        )
        assert len(filtered) == 2
        assert removed == 0


class TestPerDocumentCap:
    def test_caps_at_max_per_document(self):
        results = [
            _make_result("doc-a", f"b{i}") for i in range(10)
        ]
        filtered, removed, capped = _post_filter_results(
            results, known_doc_ids={"doc-a"}, total_limit=30,
        )
        assert len(filtered) == MAX_RESULTS_PER_DOCUMENT
        assert capped == 10 - MAX_RESULTS_PER_DOCUMENT

    def test_cap_applies_per_document_not_globally(self):
        results = [
            _make_result("doc-a", "b1"),
            _make_result("doc-a", "b2"),
            _make_result("doc-a", "b3"),
            _make_result("doc-b", "b4"),
            _make_result("doc-b", "b5"),
            _make_result("doc-b", "b6"),
        ]
        known = {"doc-a", "doc-b"}
        filtered, removed, capped = _post_filter_results(
            results, known_doc_ids=known, total_limit=30,
        )
        assert len(filtered) == 6
        assert capped == 0

    def test_cap_plus_deletion_combined(self):
        results = [
            _make_result("doc-a", "b1"),
            _make_result("doc-a", "b2"),
            _make_result("doc-a", "b3"),
            _make_result("doc-a", "b4"),  # will be capped
            _make_result("doc-deleted", "b5"),  # will be filtered
            _make_result("doc-b", "b6"),
        ]
        known = {"doc-a", "doc-b"}
        filtered, removed, capped = _post_filter_results(
            results, known_doc_ids=known, total_limit=30,
        )
        assert len(filtered) == 4  # 3 from doc-a + 1 from doc-b
        assert removed == 1
        assert capped == 1

    def test_total_limit_respected(self):
        results = [
            _make_result(f"doc-{i}", "b1") for i in range(20)
        ]
        known = {f"doc-{i}" for i in range(20)}
        filtered, removed, capped = _post_filter_results(
            results, known_doc_ids=known, total_limit=5,
        )
        assert len(filtered) == 5


class TestDiagnostics:
    def test_zero_counters_when_nothing_filtered(self):
        results = [_make_result("doc-a", "b1")]
        filtered, removed, capped = _post_filter_results(
            results, known_doc_ids={"doc-a"}, total_limit=30,
        )
        assert removed == 0
        assert capped == 0

    def test_empty_input_returns_zero_counters(self):
        filtered, removed, capped = _post_filter_results(
            [], known_doc_ids={"doc-a"}, total_limit=30,
        )
        assert filtered == []
        assert removed == 0
        assert capped == 0
