"""Tests for job metadata helpers."""

from neem.mcp.jobs.models import JobLinks, JobSubmitMetadata, WebSocketSubscriptionHint


def test_job_links_expand_and_hint_parsing():
    links_payload = {
        "status": "/jobs/abc/status",
        "result": "/jobs/abc/result",
        "websocket": {
            "description": "subscribe",
            "payload": {"action": "subscribe", "job_id": "abc"},
        },
    }

    links = JobLinks.from_api(links_payload).expand("http://api.local")

    assert links.status == "http://api.local/jobs/abc/status"
    assert links.result == "http://api.local/jobs/abc/result"
    assert isinstance(links.websocket, WebSocketSubscriptionHint)
    assert links.websocket.description == "subscribe"
    assert links.websocket.payload["action"] == "subscribe"


def test_job_submit_metadata_from_api():
    payload = {
        "job_id": "job-123",
        "status": "queued",
        "trace_id": "trace-1",
        "links": {
            "status": "/jobs/job-123",
            "result": None,
            "websocket": {
                "description": "listen",
                "payload": {"topic": "jobs", "job_id": "job-123"},
            },
        },
    }

    metadata = JobSubmitMetadata.from_api(payload, base_url="https://backend")

    assert metadata.job_id == "job-123"
    assert metadata.status == "queued"
    assert metadata.trace_id == "trace-1"
    assert metadata.links.status == "https://backend/jobs/job-123"
    assert metadata.links.websocket is not None
    assert metadata.links.websocket.payload["job_id"] == "job-123"


def test_job_links_from_top_level_poll_result_aliases():
    links_payload = {
        "poll_url": "/graphs/jobs/job-123",
        "result_url": "/graphs/jobs/job-123/result",
    }

    links = JobLinks.from_api(links_payload).expand("https://backend")

    assert links.status == "https://backend/graphs/jobs/job-123"
    assert links.result == "https://backend/graphs/jobs/job-123/result"


def test_job_submit_metadata_accepts_top_level_graph_job_urls():
    payload = {
        "job_id": "job-graph-1",
        "status": "queued",
        "trace_id": "trace-graph-1",
        "poll_url": "/graphs/jobs/job-graph-1",
        "result_url": "/graphs/jobs/job-graph-1/result",
        "websocket": {
            "description": "subscribe",
            "payload": {"type": "subscribe", "job_id": "job-graph-1"},
        },
    }

    metadata = JobSubmitMetadata.from_api(payload, base_url="https://backend")

    assert metadata.links.status == "https://backend/graphs/jobs/job-graph-1"
    assert metadata.links.result == "https://backend/graphs/jobs/job-graph-1/result"
    assert metadata.links.websocket is not None
    assert metadata.links.websocket.payload["job_id"] == "job-graph-1"
