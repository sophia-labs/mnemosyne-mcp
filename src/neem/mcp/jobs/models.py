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
        return cls(
            status=data.get("status", ""),
            result=data.get("result"),
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
        links = JobLinks.from_api(payload.get("links", {})).expand(base_url)
        return cls(
            job_id=payload["job_id"],
            status=payload.get("status", "queued"),
            trace_id=payload.get("trace_id"),
            links=links,
        )
