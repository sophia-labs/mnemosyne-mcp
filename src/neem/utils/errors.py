"""Unified error handling for SOIL-2 with structured context."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, NoReturn, Tuple
from datetime import datetime
from enum import Enum
import traceback
import logging
import time

import typer


class ErrorSeverity(str, Enum):
    """Error severity levels for classification and handling."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for structured handling."""

    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    DATABASE = "database"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RUNTIME = "runtime"
    EXTERNAL = "external"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RESOURCE_LIMIT = "resource_limit"


class MnemosyneError(Exception):
    """
    Base exception for all Mnemosyne operations with structured context.

    Provides consistent error handling across all components with
    contextual information for debugging and user feedback.
    """

    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.RUNTIME,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        user_message: Optional[str] = None,
        help_text: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        """
        Initialize structured error.

        Args:
            message: Technical error message for developers
            context: Additional context data for debugging
            severity: Error severity level
            category: Error category for classification
            operation: Operation that failed (e.g., "query", "load_file")
            component: Component where error occurred (e.g., "cli", "server", "store")
            user_message: User-friendly error message
            help_text: Suggested resolution or help information
            error_code: Unique error code for documentation reference
        """
        super().__init__(message)

        self.message = message
        self.context = context or {}
        self.severity = severity
        self.category = category
        self.operation = operation
        self.component = component
        self.user_message = user_message or message
        self.help_text = help_text
        self.error_code = error_code
        self.exit_code: Optional[int] = None  # Set by CLI error handling
        self.timestamp = datetime.utcnow()
        self.traceback_info = traceback.format_stack()[:-1]  # Exclude current frame

        # Add automatic context
        self.context.update(
            {
                "timestamp": self.timestamp.isoformat(),
                "severity": self.severity.value,
                "category": self.category.value,
            }
        )

        if self.operation:
            self.context["operation"] = self.operation
        if self.component:
            self.context["component"] = self.component

    def __str__(self) -> str:
        """String representation for logging and debugging."""
        if self.operation and self.component:
            return f"[{self.component.upper()}] {self.operation} failed: {self.message}"
        elif self.operation:
            return f"Operation '{self.operation}' failed: {self.message}"
        return self.message

    def get_user_message(self) -> str:
        """Get user-friendly error message."""
        if self.user_message != self.message:
            return self.user_message

        # Generate user-friendly message from technical details
        if self.operation:
            return f"Failed to {self.operation}. {self.help_text if self.help_text else ''}"
        return self.message

    def get_context_for_logging(self) -> Dict[str, Any]:
        """Get context information for structured logging."""
        log_context = self.context.copy()
        log_context.update(
            {
                "error_type": self.__class__.__name__,
                "error_message": self.message,
                "user_message": self.get_user_message(),
            }
        )

        if self.error_code:
            log_context["error_code"] = self.error_code
        if self.help_text:
            log_context["help_text"] = self.help_text

        return log_context

    def with_context(self, **additional_context: Any) -> "MnemosyneError":
        """Add additional context to existing error."""
        self.context.update(additional_context)
        return self

    def is_user_error(self) -> bool:
        """Check if this is a user error vs system error."""
        return self.category in [
            ErrorCategory.VALIDATION,
            ErrorCategory.CONFIGURATION,
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.AUTHORIZATION,
        ]


# Component-specific exception classes
class DatabaseError(MnemosyneError):
    """Base class for database-related errors."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        database_path: Optional[str] = None,
        query: Optional[str] = None,
        component: Optional[str] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        **kwargs: Any,  # Justified: Parent MnemosyneError accepts arbitrary context fields
    ) -> None:
        super().__init__(
            message,
            operation=operation,
            component=component or "store",
            category=category or ErrorCategory.DATABASE,
            severity=severity or ErrorSeverity.ERROR,  # Provide default when None
            **kwargs,
        )

        if database_path:
            self.context["database_path"] = database_path
        if query:
            # Truncate long queries for logging
            self.context["query"] = query[:500] + "..." if len(query) > 500 else query


class QueryError(DatabaseError):
    """SPARQL query execution errors."""

    def __init__(
        self,
        message: str,
        *,
        query: str,
        query_type: Optional[str] = None,
        operation: Optional[str] = None,  # Allow operation override from caller
        user_message: Optional[str] = None,  # Allow user_message override
        help_text: Optional[str] = None,  # Allow help_text override
        **kwargs: Any,  # Justified: Parent DatabaseError accepts arbitrary context fields
    ) -> None:
        super().__init__(
            message,
            operation=operation or "execute_query",  # Use provided operation or default
            query=query,
            user_message=user_message or "Database query failed",
            help_text=help_text
            or "Check the SPARQL syntax and ensure the database is accessible",
            error_code="DB001",
            **kwargs,
        )

        if query_type:
            self.context["query_type"] = query_type


class UpdateError(DatabaseError):
    """SPARQL update operation errors."""

    def __init__(
        self,
        message: str,
        *,
        update: str,
        operation: Optional[str] = None,  # Allow operation override from caller
        **kwargs: Any,
    ) -> None:  # Justified: Parent DatabaseError accepts arbitrary context fields
        super().__init__(
            message,
            operation=operation
            or "execute_update",  # Use provided operation or default
            query=update,
            user_message="Database update failed",
            help_text="Check the SPARQL UPDATE syntax and ensure write permissions",
            error_code="DB002",
            **kwargs,
        )


class LoadError(DatabaseError):
    """Data loading errors."""

    def __init__(
        self,
        message: str,
        *,
        source: str,
        format: Optional[str] = None,
        **kwargs: Any,  # Justified: Parent DatabaseError accepts arbitrary context fields
    ) -> None:
        super().__init__(
            message,
            operation="load_data",
            user_message="Failed to load data into database",
            help_text="Check the data format and ensure the file is accessible",
            error_code="DB003",
            **kwargs,
        )

        self.context.update({"data_source": source, "data_format": format})


class ExportError(DatabaseError):
    """Data export errors."""

    def __init__(
        self,
        message: str,
        *,
        format: str,
        destination: Optional[str] = None,
        **kwargs: Any,  # Justified: Parent DatabaseError accepts arbitrary context fields
    ) -> None:
        super().__init__(
            message,
            operation="export_data",
            user_message="Failed to export database",
            help_text="Check the export format and ensure write permissions",
            error_code="DB004",
            **kwargs,
        )

        self.context.update({"export_format": format, "destination": destination})


class ConfigurationError(MnemosyneError):
    """Configuration-related errors."""

    def __init__(
        self,
        message: str,
        *,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        component: Optional[str] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        user_message: Optional[str] = None,
        help_text: Optional[str] = None,
        **kwargs: Any,  # Justified: kwargs can contain arbitrary structured data
    ) -> None:
        super().__init__(
            message,
            component=component or "config",
            category=category or ErrorCategory.CONFIGURATION,
            severity=severity or ErrorSeverity.ERROR,
            user_message=user_message or "Configuration error",
            help_text=help_text
            or "Check the configuration file and environment variables",
            error_code="CFG001",
            **kwargs,
        )

        if config_key:
            self.context["config_key"] = config_key
        if config_file:
            self.context["config_file"] = config_file


class ValidationError(ConfigurationError):
    """Configuration validation errors."""

    def __init__(
        self,
        message: str,
        *,
        field: str,
        value: Any,
        expected_type: Optional[
            Type[Any]
        ] = None,  # Justified: Type parameter needed for generic Type
        **kwargs: Any,  # Justified: kwargs can contain arbitrary structured data
    ) -> None:
        super().__init__(
            message,
            config_key=field,
            user_message=f"Invalid configuration value for '{field}'",
            help_text="Check the configuration documentation for valid values",
            error_code="CFG002",
            **kwargs,
        )

        self.context.update(
            {
                "field": field,
                "invalid_value": str(value),
                "expected_type": expected_type.__name__ if expected_type else None,
            }
        )


class SerializationError(MnemosyneError):
    """Data serialization/deserialization errors."""

    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        operation: str = "serialize",
        data_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            context=context,
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.ERROR,
            operation=operation,
            component="codec",
            user_message="Data serialization failed",
            help_text="Check data format and codec compatibility",
            error_code="SER001",
            **kwargs,
        )

        if data_type:
            self.context["data_type"] = data_type


class ServiceDiscoveryError(MnemosyneError):
    """
    Error during service discovery operations.

    Used for Docker service discovery, endpoint resolution, and health checks.
    """

    def __init__(
        self,
        message: str,
        *,
        service_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            component="service_discovery",
            category=ErrorCategory.NETWORK,
            user_message="Service discovery failed",
            help_text="Check Docker service availability and network connectivity",
            error_code="SVC001",
            **kwargs,
        )
        if service_name:
            self.context["service_name"] = service_name
        if endpoint:
            self.context["endpoint"] = endpoint


class CLIError(MnemosyneError):
    """CLI-specific errors."""

    def __init__(
        self,
        message: str,
        *,
        command: Optional[str] = None,
        exit_code: int = 1,
        user_message: Optional[str] = None,  # Allow user_message override
        **kwargs: Any,  # Justified: kwargs can contain arbitrary structured data
    ) -> None:
        super().__init__(
            message,
            component="cli",
            category=ErrorCategory.RUNTIME,
            user_message=user_message
            or message,  # CLI errors are already user-friendly
            **kwargs,
        )

        self.exit_code = exit_code
        if command:
            self.context["command"] = command


class FileSystemError(CLIError):
    """File system operation errors."""

    def __init__(
        self,
        message: str,
        *,
        path: str,
        operation: str,
        **kwargs: Any,  # Justified: kwargs can contain arbitrary structured data
    ) -> None:
        super().__init__(message, operation=operation, error_code="FS001", **kwargs)

        self.context.update({"file_path": path, "fs_operation": operation})


class ServerError(MnemosyneError):
    """Server and API errors."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int = 500,
        endpoint: Optional[str] = None,
        **kwargs: Any,  # Justified: kwargs can contain arbitrary structured data
    ) -> None:
        super().__init__(
            message,
            component="server",
            category=ErrorCategory.NETWORK,
            user_message="Server error occurred",
            **kwargs,
        )

        self.http_status = http_status
        self.context.update({"http_status": http_status, "endpoint": endpoint})


class APIError(ServerError):
    """API-specific errors."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int = 400,
        request_id: Optional[str] = None,
        **kwargs: Any,  # Justified: kwargs can contain arbitrary structured data
    ) -> None:
        super().__init__(
            message,
            http_status=http_status,
            user_message="API request failed",
            help_text="Check the API documentation for correct usage",
            error_code="API001",
            **kwargs,
        )

        if request_id:
            self.context["request_id"] = request_id


class RetryableError(MnemosyneError):
    """Errors that can be retried with backoff strategy."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        retry_count: int = 0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            operation=operation,
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.WARNING,
            user_message=f"Operation failed (attempt {retry_count + 1}/{max_retries + 1})",
            help_text="This operation will be retried automatically",
            error_code="RETRY001",
            **kwargs,
        )

        self.retry_count = retry_count
        self.max_retries = max_retries

        self.context.update(
            {
                "retry_count": retry_count,
                "max_retries": max_retries,
                "can_retry": self.can_retry(),
            }
        )

    def can_retry(self) -> bool:
        """Check if this error can be retried."""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> "RetryableError":
        """Create new RetryableError with incremented retry count."""
        return RetryableError(
            self.message,
            operation=self.operation or "unknown",
            retry_count=self.retry_count + 1,
            max_retries=self.max_retries,
            context=self.context.copy(),
        )


# Error Handler Protocols
class CLIErrorHandler:
    """
    Unified CLI error handling with proper NoReturn typing.

    Fixes MyPy missing return statement issues while maintaining
    consistent CLI error behavior.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def handle_error(self, error: Exception, operation: str = "operation") -> NoReturn:
        """
        Handle CLI errors with proper NoReturn typing.

        Args:
            error: Exception that occurred
            operation: Operation description for context

        Raises:
            typer.Exit: Always exits with appropriate code
        """
        exit_code = 1

        if isinstance(error, MnemosyneError):
            # Use structured error information
            user_message = error.get_user_message()
            exit_code = error.exit_code or 1  # Use 1 if exit_code is None

            # Log with structured context
            self.logger.error(
                f"CLI {operation} failed", extra=error.get_context_for_logging()
            )

            # Display user-friendly message
            typer.echo(f"❌ {user_message}", err=True)

            if error.help_text:
                typer.echo(f"💡 {error.help_text}", err=True)

        else:
            # Handle non-SOIL exceptions
            self.logger.error(f"CLI {operation} failed", exception=error)
            typer.echo(f"❌ Failed to {operation}: {error}", err=True)

        raise typer.Exit(code=exit_code)

    def handle_mnemosyne_error(self, error: MnemosyneError) -> NoReturn:
        """Handle SOIL-specific errors."""
        self.handle_error(error, error.operation or "operation")

    def validate_path_exists(self, path: Path, description: str = "Path") -> None:
        """
        Validate path exists with proper error handling.

        Args:
            path: Path to validate
            description: Description for error messages

        Raises:
            typer.Exit: If path doesn't exist
        """
        if not path.exists():
            error = FileSystemError(
                f"{description} not found: {path}",
                path=str(path),
                operation="validate_path",
                user_message=f"{description} does not exist",
                help_text=f"Ensure the {description.lower()} exists and is accessible",
            )
            self.handle_mnemosyne_error(error)


# Global CLI error handler
cli_error_handler = CLIErrorHandler()


# Legacy compatibility functions
def cli_error(message: str, exit_code: int = 1) -> NoReturn:
    """
    Display CLI error and exit (compatibility function).

    Args:
        message: Error message to display
        exit_code: Exit code (default: 1)

    Raises:
        typer.Exit: Always exits
    """
    error = CLIError(message, exit_code=exit_code)
    cli_error_handler.handle_mnemosyne_error(error)


def handle_exception(operation: str, error: Exception, exit_code: int = 1) -> NoReturn:
    """
    Handle exception with operation context (compatibility function).

    Args:
        operation: Operation description
        error: Exception that occurred
        exit_code: Exit code (default: 1)

    Raises:
        typer.Exit: Always exits
    """
    # Format error message for legacy compatibility
    error_message = f"Failed to {operation}: {error}"
    cli_error(error_message, exit_code)


def validate_path_exists(path: Path, description: str = "Path") -> None:
    """Validate that a path exists, error if not."""
    if not path.exists():
        cli_error(f"{description} not found: {path}")


def validate_path_not_exists(
    path: Path, description: str = "Path", allow_force: bool = False
) -> None:
    """Validate that a path doesn't exist, with optional force override."""
    if path.exists() and not allow_force:
        cli_error(f"{description} already exists at {path}. Use --force to overwrite.")


# Server Error Handling
class ServerErrorHandler:
    """Unified server error handling for FastAPI."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def handle_mnemosyne_error(
        self, request: Any, error: MnemosyneError  # Request object
    ) -> Tuple[Dict[str, Any], int]:  # Return tuple as server expects
        """
        Handle SOIL errors in FastAPI endpoints.

        Args:
            request: FastAPI request object
            error: SOIL error that occurred

        Returns:
            Tuple of (error_response_dict, status_code)
        """
        trace_id = getattr(getattr(request, "state", None), "trace_id", None)
        log_context = error.get_context_for_logging()
        if trace_id:
            log_context["trace_id"] = trace_id

        # Log with structured context
        self.logger.error(
            f"API error in {getattr(getattr(request, 'url', {}), 'path', 'unknown') if hasattr(request, 'url') else 'unknown'}",
            extra=log_context,
        )

        # Determine HTTP status code
        if isinstance(error, ServerError):
            status_code = error.http_status
        elif isinstance(error, ValidationError):
            status_code = 400
        elif isinstance(error, ConfigurationError):
            status_code = 503
        elif isinstance(error, DatabaseError):
            if isinstance(error, QueryError):
                status_code = 400  # Bad request for query syntax errors
            else:
                status_code = 503  # Service unavailable for database connection issues
        else:
            status_code = 500

        # Prepare error response
        error_response = {
            "error": {
                "type": error.__class__.__name__,
                "message": error.get_user_message(),
                "code": error.error_code,
                "category": error.category.value,
                "timestamp": error.timestamp.isoformat(),
            }
        }
        if trace_id:
            error_response["error"]["trace_id"] = trace_id

        # Add help text for user errors
        if error.is_user_error() and error.help_text:
            error_response["error"]["help"] = error.help_text

        # Add context for debugging (only in development)
        if (
            hasattr(request, "app")
            and hasattr(request.app, "state")
            and hasattr(request.app.state, "debug")
            and request.app.state.debug
        ):
            error_response["debug"] = {
                "operation": error.operation,
                "component": error.component,
                "context": error.context,  # type: ignore # Justified: context dict has Any values
            }

        from fastapi.responses import JSONResponse

        response = JSONResponse(status_code=status_code, content=error_response)
        if trace_id:
            response.headers["X-Trace-ID"] = trace_id
        return response

    async def handle_http_exception(
        self, request: Any, exc: Any  # HTTPException
    ) -> Any:  # JSONResponse
        """
        Handle FastAPI HTTP exceptions.

        Args:
            request: FastAPI request object
            exc: HTTPException that occurred

        Returns:
            JSONResponse with error details
        """
        from fastapi.responses import JSONResponse

        trace_id = getattr(getattr(request, "state", None), "trace_id", None)

        error_response = {
            "error": {
                "type": "HTTPException",
                "message": str(exc.detail),
                "status_code": exc.status_code,
                "timestamp": time.time(),
            }
        }
        if trace_id:
            error_response["error"]["trace_id"] = trace_id

        log_kwargs = {"extra": {"trace_id": trace_id}} if trace_id else {}
        self.logger.warning(
            f"HTTP exception: {exc.status_code} - {exc.detail}",
            **log_kwargs,
        )

        response = JSONResponse(status_code=exc.status_code, content=error_response)
        if trace_id:
            response.headers["X-Trace-ID"] = trace_id
        return response

    async def handle_generic_exception(
        self, request: Any, exc: Exception
    ) -> Any:  # JSONResponse
        """
        Handle unexpected exceptions.

        Args:
            request: FastAPI request object
            exc: Unexpected exception that occurred

        Returns:
            JSONResponse with error details
        """
        from fastapi.responses import JSONResponse
        import traceback

        trace_id = getattr(getattr(request, "state", None), "trace_id", None)

        error_response = {
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected error occurred",
                "timestamp": time.time(),
            }
        }
        if trace_id:
            error_response["error"]["trace_id"] = trace_id

        # Log the full traceback for debugging
        log_kwargs = {"extra": {"trace_id": trace_id}} if trace_id else {}
        self.logger.error(f"Unexpected exception: {exc}", exc_info=True, **log_kwargs)

        response = JSONResponse(status_code=500, content=error_response)
        if trace_id:
            response.headers["X-Trace-ID"] = trace_id
        return response

    def convert_to_http_exception(
        self, error: MnemosyneError
    ) -> Any:  # Justified: HTTPException type from external library
        """
        Convert SOIL error to HTTPException for compatibility.

        Args:
            error: SOIL error to convert

        Returns:
            HTTPException with appropriate status and details
        """
        from fastapi import HTTPException

        if isinstance(error, ServerError):
            status_code = error.http_status
        elif isinstance(error, ValidationError):
            status_code = 400
        elif isinstance(error, DatabaseError):
            status_code = 503 if error.severity == ErrorSeverity.CRITICAL else 500
        else:
            status_code = 500

        detail = error.get_user_message()
        if error.help_text and error.is_user_error():
            detail += f" {error.help_text}"

        return HTTPException(status_code=status_code, detail=detail)


# Store Error Handling
class StoreErrorHandler:
    """Unified store error handling with context preservation."""

    def __init__(self, component_name: str = "store"):
        self.component = component_name
        self.logger = logging.getLogger(__name__)

    def wrap_operation_error(
        self, operation: str, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> MnemosyneError:
        """
        Wrap operation errors with appropriate SOIL error type.

        Args:
            operation: Operation that failed
            error: Original exception
            context: Additional context information

        Returns:
            Appropriate SOIL error with context
        """
        if isinstance(error, MnemosyneError):
            # Already a SOIL error, just add context
            if context:
                error.with_context(**context)
            return error

        # Create appropriate SOIL error based on operation
        if operation == "query":
            return QueryError(
                str(error),
                query=context.get("query", "unknown") if context else "unknown",
            )
        elif operation == "update":
            return UpdateError(
                str(error),
                update=context.get("update", "unknown") if context else "unknown",
            )
        elif operation in ["load_file", "load_data"]:
            return LoadError(
                str(error),
                source=context.get("source", "unknown") if context else "unknown",
                format=context.get("format") if context else None,
            )
        elif operation == "export":
            return ExportError(
                str(error),
                format=context.get("format", "unknown") if context else "unknown",
                destination=context.get("destination") if context else None,
            )
        else:
            return DatabaseError(str(error), operation=operation)

    def handle_operation_error(
        self, operation: str, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Handle store operation error and re-raise as SOIL error.

        Args:
            operation: Operation that failed
            error: Original exception
            context: Additional context information

        Raises:
            MnemosyneError: Wrapped error with context
        """
        soil_error = self.wrap_operation_error(operation, error, context)

        # Log the error
        self.logger.error(
            f"Store {operation} failed", extra=soil_error.get_context_for_logging()
        )

        raise soil_error from error


# Multi-graph specific errors

class GraphNotFoundError(MnemosyneError):
    """Graph not found error."""

    def __init__(
        self,
        graph_id: str,
        *,
        available_graphs: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Graph '{graph_id}' not found",
            category=ErrorCategory.NOT_FOUND,
            operation="graph_access",
            component="multi_graph_store",
            user_message=f"The graph '{graph_id}' does not exist. Use GET /graphs to see available graphs.",
            help_text="Check the graph ID and ensure the graph has been created",
            error_code="MG001",
            **kwargs,
        )
        
        self.context.update({
            "graph_id": graph_id,
            "available_graphs": available_graphs or []
        })


class GraphAlreadyExistsError(MnemosyneError):
    """Graph already exists error."""

    def __init__(
        self,
        graph_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Graph '{graph_id}' already exists",
            category=ErrorCategory.CONFLICT,
            operation="create_graph",
            component="multi_graph_store",
            user_message=f"Graph '{graph_id}' already exists. Use a different name or delete the existing graph first.",
            help_text="Choose a different graph ID or delete the existing graph",
            error_code="MG002",
            **kwargs,
        )
        
        self.context.update({"graph_id": graph_id})


class ResourceLimitExceededError(MnemosyneError):
    """Resource limit exceeded error."""

    def __init__(
        self,
        message: str,
        *,
        limit_type: str = "resource",
        current_value: Optional[Any] = None,
        limit_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE_LIMIT,
            operation="resource_check",
            component="multi_graph_store",
            user_message=f"Resource limit exceeded: {message}",
            help_text="Delete unused graphs or increase resource limits",
            error_code="MG003",
            **kwargs,
        )
        
        self.context.update({
            "limit_type": limit_type,
            "current_value": current_value,
            "limit_value": limit_value,
        })


class InvalidGraphIdError(MnemosyneError):
    """Invalid graph ID error."""

    def __init__(
        self,
        graph_id: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Invalid graph ID '{graph_id}': {reason}",
            category=ErrorCategory.VALIDATION,
            operation="validate_graph_id",
            component="multi_graph_store",
            user_message=f"The graph ID '{graph_id}' is invalid: {reason}",
            help_text="Use alphanumeric characters, hyphens, and underscores only",
            error_code="MG004",
            **kwargs,
        )
        
        self.context.update({
            "graph_id": graph_id,
            "validation_reason": reason
        })


# File upload specific errors
class FileUploadError(MnemosyneError):
    """File upload specific errors."""
    
    def __init__(
        self,
        message: str,
        *,
        filename: str,
        file_size: Optional[int] = None,
        file_format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            operation="file_upload",
            component="api_v1",
            user_message=f"Failed to upload file '{filename}': {message}",
            help_text="Check file format and size limits",
            error_code="FU001",
            **kwargs,
        )
        
        self.context.update({
            "filename": filename,
            "file_size": file_size,
            "file_format": file_format,
        })


# Global error handlers
server_error_handler = ServerErrorHandler()
store_error_handler = StoreErrorHandler()
