"""
MCP authentication context for multi-modal auth support.

Handles authentication from multiple sources:
1. HTTP request headers (sidecar mode - from OpenCode's mcpAuth)
2. Environment variables (dev mode)
3. Local token storage (CLI mode)
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Callable, Dict, Optional

from neem.utils.logging import LoggerFactory
from neem.utils.token_storage import (
    get_dev_user_id,
    get_internal_service_secret,
    get_user_id_from_token,
    validate_token_and_load,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

logger = LoggerFactory.get_logger("mcp.auth")
_CURRENT_AUTH_CONTEXT: contextvars.ContextVar[Optional["MCPAuthContext"]] = contextvars.ContextVar(
    "mcp_current_auth_context",
    default=None,
)
_INTERNAL_TRUST_AUTH_MODES = frozenset({"hosted", "sidecar"})
_DEMO_NOAUTH_AUTH_MODES = frozenset({"demo_noauth"})
_REQUEST_SCOPED_AUTH_MODES = _INTERNAL_TRUST_AUTH_MODES | frozenset({"public", "demo_noauth"})
_AGENT_SESSION_TOKEN_PREFIX = "agsa_"
_CHATGPT_DEMO_TOKEN_ENV = "MNEMOSYNE_CHATGPT_DEMO_TOKEN"
_CHATGPT_DEMO_USER_ID_ENV = "MNEMOSYNE_CHATGPT_DEMO_USER_ID"


def _auth_mode() -> str:
    return (os.getenv("MNEMOSYNE_MCP_AUTH_MODE", "").strip().lower() or "auto")


def _allow_local_fallbacks() -> bool:
    mode = _auth_mode()
    return mode not in _REQUEST_SCOPED_AUTH_MODES


def _allow_forwarded_user_id_header() -> bool:
    return _auth_mode() in _INTERNAL_TRUST_AUTH_MODES


def _allow_internal_service_auth() -> bool:
    return _auth_mode() in _INTERNAL_TRUST_AUTH_MODES


def _is_valid_public_bearer(token: str) -> bool:
    return token.startswith(_AGENT_SESSION_TOKEN_PREFIX)


def _is_demo_noauth_mode() -> bool:
    return _auth_mode() in _DEMO_NOAUTH_AUTH_MODES


def _get_demo_auth_token() -> Optional[str]:
    token = os.getenv(_CHATGPT_DEMO_TOKEN_ENV, "").strip()
    return token or None


def _get_demo_auth_user_id(token: Optional[str]) -> Optional[str]:
    configured = os.getenv(_CHATGPT_DEMO_USER_ID_ENV, "").strip()
    if configured:
        return configured
    if token:
        return get_user_id_from_token(token)
    return None


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

        Token priority:
        1. HTTP Authorization header (Bearer token from OpenCode mcpAuth)
        2. validate_token_and_load() for local/dev token

        User ID priority:
        1. HTTP X-User-ID header (sidecar mode)
        2. MNEMOSYNE_DEV_USER_ID or MNEMOSYNE_DEV_TOKEN env vars
        3. JWT token claims (sub, user_id, uid) - for local CLI users
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
                        if _is_demo_noauth_mode():
                            logger.info("ignoring_request_bearer_in_demo_noauth_mode")
                        else:
                            candidate = auth_header[7:]
                            if _auth_mode() == "public" and not _is_valid_public_bearer(candidate):
                                logger.warning(
                                    "rejecting_non_agent_public_bearer",
                                    extra_context={"auth_mode": _auth_mode()},
                                )
                            else:
                                token = candidate
                                source = "http_header"
                                logger.info(
                                    "auth_from_http_header",
                                    extra_context={"has_token": bool(token)},
                                )
                    # Sidecar deployments may also pass user context explicitly.
                    forwarded_user_id = headers.get("x-user-id") or headers.get("X-User-ID")
                    if forwarded_user_id and _allow_forwarded_user_id_header():
                        user_id = forwarded_user_id
                        logger.info(
                            "user_id_from_header",
                            extra_context={"user_id": user_id},
                        )
                    elif forwarded_user_id:
                        logger.warning(
                            "ignoring_forwarded_user_id_header",
                            extra_context={"auth_mode": _auth_mode()},
                        )

        # 2. ChatGPT demo mode uses a dedicated configured backend credential
        # while remaining noauth at the external MCP edge.
        if _is_demo_noauth_mode():
            demo_token = _get_demo_auth_token()
            if demo_token:
                token = demo_token
                source = "demo_env"
            demo_user_id = _get_demo_auth_user_id(demo_token)
            if demo_user_id:
                user_id = demo_user_id

        # 3. Fall back to local/env token (CLI mode).
        # In hosted/sidecar mode, avoid global local-token fallback because
        # requests must be scoped to the caller's forwarded auth context.
        if not token and _allow_local_fallbacks():
            token = validate_token_and_load()
            if token:
                source = "local_storage" if source == "none" else source

        # 4. Dev mode user override
        if not user_id and _allow_local_fallbacks():
            user_id = get_dev_user_id()
            if user_id and source == "none":
                source = "dev_env"

        # 5. Extract user_id from JWT token claims (for local CLI users)
        if not user_id and token:
            user_id = get_user_id_from_token(token)
            if user_id:
                logger.debug(
                    "user_id_from_token_claim",
                    extra_context={"user_id": user_id},
                )

        internal_secret = get_internal_service_secret() if _allow_internal_service_auth() else None

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

        _CURRENT_AUTH_CONTEXT.set(result)
        return result

    def require_auth(self) -> str:
        """Get token or raise if not authenticated.

        Returns the token if available. For internal service auth
        (sidecar with shared secret), returns empty string but allows
        the operation to proceed.
        """
        if self.token:
            return self.token

        if _allow_internal_service_auth() and self.internal_service_secret and self.user_id:
            # Internal service auth is valid even without a token
            logger.debug(
                "auth_via_internal_service",
                extra_context={"user_id": self.user_id},
            )
            return ""

        raise RuntimeError(
            f"{_CHATGPT_DEMO_TOKEN_ENV} is required in demo_noauth mode."
            if _is_demo_noauth_mode()
            else
            "Hosted session bearer token required."
            if _auth_mode() == "public"
            else "Not authenticated. Either run `neem init` to get a token, "
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

        if self.user_id and _allow_forwarded_user_id_header():
            headers["X-User-ID"] = self.user_id

        if self.internal_service_secret and _allow_internal_service_auth():
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


def get_hocuspocus_client_kwargs(
    *,
    token_provider: Callable[[], Optional[str]],
) -> Dict[str, object]:
    """Return HocuspocusClient auth kwargs appropriate for the active auth mode."""
    kwargs: Dict[str, object] = {"token_provider": token_provider}
    if _allow_internal_service_auth():
        kwargs["dev_user_id"] = get_dev_user_id()
        kwargs["internal_service_secret"] = get_internal_service_secret()
    else:
        kwargs["dev_user_id"] = None
        kwargs["internal_service_secret"] = None
    return kwargs


def get_current_auth_context() -> Optional[MCPAuthContext]:
    """Return the request-scoped auth context, if one is currently bound."""
    return _CURRENT_AUTH_CONTEXT.get()


def clear_current_auth_context() -> None:
    """Clear any request-scoped auth context from the current task context."""
    _CURRENT_AUTH_CONTEXT.set(None)


def get_current_auth_token() -> Optional[str]:
    """Return the active request token, falling back to local token when allowed."""
    current = _CURRENT_AUTH_CONTEXT.get()
    if current and current.token:
        return current.token
    if _allow_local_fallbacks():
        return validate_token_and_load()
    return None
