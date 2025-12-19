"""Decorators for MCP tool functions.

Provides reusable decorators for common patterns like authentication
and parameter validation.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

from neem.utils.token_storage import validate_token_and_load

# Type variables for generic decorator typing
F = TypeVar("F", bound=Callable[..., Any])


def require_auth(func: F) -> F:
    """Decorator that validates authentication before tool execution.

    Raises RuntimeError if no valid token is available.

    Usage:
        @require_auth
        async def my_tool(...) -> dict:
            # Token is guaranteed to be valid here
            ...
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = validate_token_and_load()
        if not token:
            raise RuntimeError(
                "Not authenticated. Run `neem init` to refresh your token."
            )
        return await func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def validate_required(*required_params: str) -> Callable[[F], F]:
    """Decorator that validates required string parameters are non-empty.

    Args:
        *required_params: Names of parameters that must be non-empty strings

    Usage:
        @validate_required("graph_id", "document_id")
        async def my_tool(graph_id: str, document_id: str, ...) -> dict:
            # graph_id and document_id are guaranteed non-empty
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for param in required_params:
                value = kwargs.get(param)
                if not value or not str(value).strip():
                    raise ValueError(f"{param} is required and cannot be empty")
            return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def strip_params(*params_to_strip: str) -> Callable[[F], F]:
    """Decorator that strips whitespace from specified string parameters.

    Args:
        *params_to_strip: Names of string parameters to strip

    Usage:
        @strip_params("graph_id", "document_id")
        async def my_tool(graph_id: str, document_id: str, ...) -> dict:
            # graph_id and document_id are already stripped
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for param in params_to_strip:
                if param in kwargs and isinstance(kwargs[param], str):
                    kwargs[param] = kwargs[param].strip()
            return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


__all__ = ["require_auth", "validate_required", "strip_params"]
