"""
Response filtering utilities for token efficiency.

These helpers strip unnecessary fields from API responses, reducing token
consumption while preserving essential information for the AI model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

JsonDict = Dict[str, Any]


def filter_graph_list(graphs: List[JsonDict], *, include_stats: bool = True) -> List[JsonDict]:
    """
    Filter graph list to essential fields.

    Reduces from 10 fields to 5-6 essential fields (~60% token reduction).

    Args:
        graphs: Raw graph list from API
        include_stats: Include triple_count and timestamps

    Returns:
        Filtered graph list with only essential fields
    """
    essential_fields = {"graph_id", "title", "status"}

    if include_stats:
        essential_fields.update({"triple_count", "updated_at"})

    return [
        {k: v for k, v in graph.items() if k in essential_fields}
        for graph in graphs
    ]


def filter_graph_metadata(graph: JsonDict, *, verbose: bool = False) -> JsonDict:
    """
    Filter individual graph metadata.

    Args:
        graph: Raw graph metadata from API
        verbose: Include all fields (for detailed queries)

    Returns:
        Filtered graph metadata
    """
    if verbose:
        # Keep all fields for explicit graph queries
        return graph

    # Compact format for list operations
    essential = {
        "graph_id": graph.get("graph_id"),
        "title": graph.get("title"),
        "triple_count": graph.get("triple_count", 0),
        "status": graph.get("status", "unknown"),
    }

    # Add description only if non-empty
    if graph.get("description"):
        essential["description"] = graph["description"]

    # Add updated_at for recency context
    if graph.get("updated_at"):
        essential["updated_at"] = graph["updated_at"]

    return essential


def filter_job_status(status: JsonDict, *, include_debug: bool = False) -> JsonDict:
    """
    Filter job status to essential completion info.

    Args:
        status: Raw job status from API
        include_debug: Include trace_id and timing details

    Returns:
        Filtered job status
    """
    essential = {
        "job_id": status.get("job_id"),
        "status": status.get("status"),
    }

    if status.get("error"):
        essential["error"] = status["error"]

    if include_debug:
        if status.get("processing_time_ms") is not None:
            essential["duration_ms"] = status["processing_time_ms"]
        if status.get("trace_id"):
            essential["trace_id"] = status["trace_id"]

    return essential


def extract_result_from_job_detail(detail: JsonDict) -> Optional[Any]:
    """
    Extract inline result from job detail if available.

    Args:
        detail: Job detail object from status response

    Returns:
        Inline result data or None
    """
    if not detail:
        return None

    if detail.get("result_inline") is not None:
        return detail["result_inline"]

    return None


def filter_query_results(
    results: List[JsonDict],
    *,
    max_results: Optional[int] = None,
    exclude_fields: Optional[Set[str]] = None,
) -> List[JsonDict]:
    """
    Filter SPARQL query results for token efficiency.

    Args:
        results: Raw query results
        max_results: Limit number of results
        exclude_fields: Field names to exclude

    Returns:
        Filtered query results
    """
    filtered = results

    if max_results is not None:
        filtered = filtered[:max_results]

    if exclude_fields:
        filtered = [
            {k: v for k, v in row.items() if k not in exclude_fields}
            for row in filtered
        ]

    return filtered
