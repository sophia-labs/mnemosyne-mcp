"""
MCP authentication context for multi-modal auth support.

Handles authentication from multiple sources:
1. HTTP request headers (sidecar mode - from OpenCode's mcpAuth)
2. Environment variables (dev mode)
3. Local token storage (CLI mode)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional

from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import (
    get_dev_user_id,
    get_internal_service_secret,
    validate_token_and_load,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

logger = LoggerFactory.get_logger("mcp.auth")


@dataclass
class MCPAuthContext:
    """Authentication context for MCP tool operations.

    Resolves auth from multiple sources in priority order:
    1. HTTP request headers (sidecar mode - from OpenCode's mcpAuth)
    2. Environment variables (dev mode - MNEMOSYNE_DEV_TOKEN/USER_ID)
    3. Local token storage (CLI mode - ~/.config/neem/token)

    The sidecar mode is critical for multi-tenant deployments where
    each user's session has a unique token passed via OpenCode's
    per-session MCP authentication.
    """

    token: Optional[str] = None
    user_id: Optional[str] = None
    internal_service_secret: Optional[str] = None
    source: str = "none"

    @classmethod
    def from_context(cls, ctx: Optional[Context]) -> MCPAuthContext:
        """Extract auth from MCP request context, with fallbacks.

        Priority:
        1. HTTP Authorization header (Bearer token from OpenCode mcpAuth)
        2. validate_token_and_load() for local/dev token
        3. Internal service auth with dev user
        """
        token: Optional[str] = None
        user_id: Optional[str] = None
        source = "none"

        # 1. Try HTTP request context (sidecar mode)
        # When OpenCode calls MCP via HTTP with mcpAuth, headers are available
        if ctx is not None:
            request_context = getattr(ctx, "request_context", None)
            if request_context is not None:
                request = getattr(request_context, "request", None)
                if request is not None:
                    headers = getattr(request, "headers", {})
                    # Log all headers for debugging (at INFO level so we can see in prod)
                    header_keys = list(headers.keys()) if hasattr(headers, "keys") else []
                    logger.info(
                        "mcp_request_headers",
                        extra_context={"header_keys": header_keys},
                    )
                    auth_header = headers.get("authorization", "") or headers.get("Authorization", "")
                    if auth_header.startswith("Bearer "):
                        token = auth_header[7:]
                        source = "http_header"
                        logger.info(
                            "auth_from_http_header",
                            extra_context={"token_prefix": token[:12] + "..." if token else None},
                        )
                    # OpenCode may also pass user context
                    user_id = headers.get("x-user-id") or headers.get("X-User-ID")
                    if user_id:
                        logger.info(
                            "user_id_from_header",
                            extra_context={"user_id": user_id},
                        )

        # 2. Fall back to local/env token (CLI mode)
        if not token:
            token = validate_token_and_load()
            if token:
                source = "local_storage" if source == "none" else source

        # 3. Dev mode user override
        if not user_id:
            user_id = get_dev_user_id()
            if user_id and source == "none":
                source = "dev_env"

        internal_secret = get_internal_service_secret()

        result = cls(
            token=token,
            user_id=user_id,
            internal_service_secret=internal_secret,
            source=source,
        )

        # Log final auth state
        logger.info(
            "auth_context_resolved",
            extra_context={
                "source": source,
                "has_token": bool(token),
                "has_user_id": bool(user_id),
                "has_internal_secret": bool(internal_secret),
            },
        )

        return result

    def require_auth(self) -> str:
        """Get token or raise if not authenticated.

        Returns the token if available. For internal service auth
        (sidecar with shared secret), returns empty string but allows
        the operation to proceed.
        """
        if self.token:
            return self.token

        if self.internal_service_secret and self.user_id:
            # Internal service auth is valid even without a token
            logger.debug(
                "auth_via_internal_service",
                extra_context={"user_id": self.user_id},
            )
            return ""

        raise RuntimeError(
            "Not authenticated. Either run `neem init` to get a token, "
            "or configure MNEMOSYNE_INTERNAL_SERVICE_SECRET and MNEMOSYNE_DEV_USER_ID."
        )

    def http_headers(self) -> Dict[str, str]:
        """Generate headers for backend API requests.

        Includes all auth headers needed for the mnemosyne-api:
        - Authorization: Bearer token (if available)
        - X-User-ID: User identifier (dev mode or from HTTP context)
        - X-Internal-Service: Shared secret for cluster-internal auth
        """
        headers: Dict[str, str] = {}

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        if self.user_id:
            headers["X-User-ID"] = self.user_id

        if self.internal_service_secret:
            headers["X-Internal-Service"] = self.internal_service_secret

        return headers

    def is_authenticated(self) -> bool:
        """Check if we have valid auth without raising."""
        return bool(self.token) or bool(self.internal_service_secret and self.user_id)


def create_context_aware_token_provider(
    ctx_holder: Callable[[], Optional[Context]],
) -> Callable[[], Optional[str]]:
    """Create a token provider that checks HTTP context first.

    This is useful for HocuspocusClient which takes a token_provider
    callable. The provider will check HTTP context (if available)
    before falling back to local storage.

    Args:
        ctx_holder: A callable that returns the current MCP Context,
                   or None if not in a request context.

    Returns:
        A token provider callable suitable for HocuspocusClient.
    """

    def provider() -> Optional[str]:
        ctx = ctx_holder()
        auth = MCPAuthContext.from_context(ctx)
        return auth.token

    return provider
