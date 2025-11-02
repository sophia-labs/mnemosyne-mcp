"""
MCP-specific error handling following existing MnemosyneError patterns.

Provides structured error handling for MCP operations while maintaining
consistency with the existing error hierarchy and logging patterns.
"""

from typing import Dict, Any, Optional
from neem.utils.errors import MnemosyneError, ErrorCategory, ErrorSeverity


class MCPError(MnemosyneError):
    """Base exception for MCP-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.RUNTIME,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        help_text: Optional[str] = None
    ):
        super().__init__(
            message=message,
            category=category,
            severity=severity,
            context=context or {},
            user_message=user_message,
            help_text=help_text
        )


class MCPAuthenticationError(MCPError):
    """MCP authentication/authorization errors."""
    
    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.WARNING,
            context=context,
            user_message=user_message or "Authentication required for MCP access"
        )


class MCPValidationError(MCPError):
    """MCP input validation errors."""
    
    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        error_context = context or {}
        if field:
            error_context["field"] = field
        if value is not None:
            error_context["invalid_value"] = str(value)
            
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            context=error_context,
            user_message=user_message or f"Invalid input: {message}"
        )


class MCPSessionError(MCPError):
    """MCP session management errors."""
    
    def __init__(
        self,
        message: str,
        *,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        error_context = context or {}
        if session_id:
            error_context["session_id"] = session_id
            
        super().__init__(
            message=message,
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.ERROR,
            context=error_context,
            user_message=user_message or "MCP session error"
        )


class MCPToolExecutionError(MCPError):
    """MCP tool execution errors."""
    
    def __init__(
        self,
        message: str,
        *,
        tool_name: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        error_context = context or {}
        if tool_name:
            error_context["tool_name"] = tool_name
        if user_id:
            error_context["user_id"] = user_id
            
        super().__init__(
            message=message,
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.ERROR,
            context=error_context,
            user_message=user_message or f"Tool execution failed: {tool_name or 'unknown'}"
        )


def handle_mcp_exception(exc: Exception, tool_name: Optional[str] = None) -> MCPError:
    """
    Convert general exceptions to MCP-specific errors.
    
    Args:
        exc: The original exception
        tool_name: Name of the tool that caused the error
        
    Returns:
        MCPError with appropriate categorization
    """
    if isinstance(exc, MCPError):
        return exc
    
    # Map common exceptions to MCP errors
    if isinstance(exc, PermissionError):
        return MCPAuthenticationError(
            message=str(exc),
            context={"original_exception": type(exc).__name__}
        )
    
    if isinstance(exc, ValueError):
        return MCPValidationError(
            message=str(exc),
            context={"original_exception": type(exc).__name__}
        )
    
    if isinstance(exc, TimeoutError):
        return MCPToolExecutionError(
            message=f"Tool execution timed out: {str(exc)}",
            tool_name=tool_name,
            context={"original_exception": type(exc).__name__}
        )
    
    # Generic MCP error for other exceptions
    return MCPToolExecutionError(
        message=f"Unexpected error: {str(exc)}",
        tool_name=tool_name,
        context={
            "original_exception": type(exc).__name__,
            "exception_message": str(exc)
        }
    )