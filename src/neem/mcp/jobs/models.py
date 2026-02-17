"""
Typed helpers for job submission/streaming metadata.

These dataclasses wrap the OpenAPI structures exposed by the FastAPI backend so
tools can work with strongly-typed navigation links and WebSocket hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urljoin

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class WebSocketSubscriptionHint:
    """Describes how to subscribe to a job via the /ws gateway."""

    description: str
    payload: JsonDict

    @classmethod
    def from_api(cls, data: Optional[JsonDict]) -> Optional["WebSocketSubscriptionHint"]:
        """Build hint from OpenAPI response payload."""
        if not data:
            return None

        description = data.get("description") or "Subscribe for realtime updates"
        payload = data.get("payload") or {}
        return cls(description=description, payload=payload)


@dataclass(frozen=True)
class JobLinks:
    """Navigation links and streaming hints attached to job responses."""

    status: str
    result: Optional[str]
    websocket: Optional[WebSocketSubscriptionHint]

    @classmethod
    def from_api(cls, data: JsonDict) -> "JobLinks":
        status_url = (
            data.get("status")
            or data.get("poll_url")
            or data.get("status_url")
            or ""
        )
        result_url = (
            data.get("result")
            or data.get("result_url")
        )
        return cls(
            status=status_url,
            result=result_url,
            websocket=WebSocketSubscriptionHint.from_api(data.get("websocket")),
        )

    def expand(self, base_url: str) -> "JobLinks":
        """Return a copy with absolute URLs derived from the backend base."""

        def _expand(url: Optional[str]) -> Optional[str]:
            if not url:
                return None
            return urljoin(f"{base_url.rstrip('/')}/", url)

        return JobLinks(
            status=_expand(self.status) or self.status,
            result=_expand(self.result),
            websocket=self.websocket,
        )


@dataclass(frozen=True)
class JobSubmitMetadata:
    """Normalized view over job submission responses."""

    job_id: str
    status: str
    trace_id: Optional[str]
    links: JobLinks

    @classmethod
    def from_api(cls, payload: JsonDict, base_url: str) -> "JobSubmitMetadata":
        raw_links = payload.get("links")
        links_payload = dict(raw_links) if isinstance(raw_links, dict) else {}

        # Back-compat: newer API variants can return top-level poll/result URLs
        # instead of nested links.status/links.result.
        if not links_payload.get("status"):
            links_payload["status"] = (
                payload.get("poll_url")
                or payload.get("status_url")
                or payload.get("status")
                or ""
            )
        if not links_payload.get("result"):
            links_payload["result"] = (
                payload.get("result_url")
                or payload.get("result")
            )
        if not links_payload.get("websocket") and isinstance(payload.get("websocket"), dict):
            links_payload["websocket"] = payload.get("websocket")

        links = JobLinks.from_api(links_payload).expand(base_url)
        return cls(
            job_id=payload["job_id"],
            status=payload.get("status", "queued"),
            trace_id=payload.get("trace_id"),
            links=links,
        )
