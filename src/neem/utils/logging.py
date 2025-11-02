"""
Structured logging framework for SOIL-2.

This module provides comprehensive structured logging with context management,
performance monitoring, and integration with the dependency injection system.
"""

import logging
import logging.config
import structlog
from structlog import contextvars as structlog_contextvars
from typing import Any, Dict, Optional, Union, Callable, List, Protocol
from datetime import datetime
from enum import Enum
import json
import sys
import threading
import types
from pathlib import Path
from functools import wraps, lru_cache
from contextlib import contextmanager
import time


class LogLevel(str, Enum):
    """Log levels with string values."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    HAPPY = "HAPPY"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""

    STRUCTURED = "structured"
    SIMPLE = "simple"
    JSON = "json"
    CONSOLE = "console"


class ContextKeys:
    """Standard context keys for structured logging."""

    COMPONENT = "component"
    OPERATION = "operation"
    TRACE_ID = "trace_id"
    REQUEST_ID = "request_id"
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    DURATION_MS = "duration_ms"
    ERROR_TYPE = "error_type"
    DATABASE_PATH = "database_path"
    QUERY = "query"
    HTTP_STATUS = "http_status"
    ENDPOINT = "endpoint"
    CLI_COMMAND = "cli_command"


class SettingsProtocol(Protocol):
    """Protocol for settings objects that can configure logging."""

    def __getattr__(self, name: str) -> Any:
        """
        Allow access to dynamic settings attributes.

        NOTE: Any type justified here - settings objects have arbitrary dynamic attributes
        that cannot be statically typed.
        """
        ...


class StructuredLogger:
    """
    Type-safe structured logger with context management and performance optimization.

    Provides consistent logging across all SOIL-2 components with contextual
    information, lazy evaluation, and integration with configuration system.
    """

    def __init__(
        self,
        component: str,
        *,
        logger_name: Optional[str] = None,
        base_context: Optional[Dict[str, Any]] = None,
        enable_performance_tracking: bool = False,
    ):
        """
        Initialize structured logger for component.

        Args:
            component: Component name (e.g., "cli", "server", "store")
            logger_name: Optional logger name (defaults to component)
            base_context: Base context added to all log messages
            enable_performance_tracking: Enable automatic performance tracking
        """
        self.component = component
        self.logger_name = logger_name or f"mnemosyne.{component}"
        self.base_context = base_context or {}
        self.enable_performance_tracking = enable_performance_tracking

        # Thread-local storage for request context
        self._local = threading.local()

        # Initialize structlog logger
        self._logger = structlog.get_logger(self.logger_name)

        # Add component to base context
        self.base_context[ContextKeys.COMPONENT] = component

    def _get_current_context(self) -> Dict[str, Any]:
        """Get current logging context with thread-local data."""
        context = self.base_context.copy()

        # Add thread-local context if available
        if hasattr(self._local, "context"):
            context.update(self._local.context)

        return context

    def _log_with_context(
        self,
        level: str,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """
        Internal logging method with context management.

        Args:
            level: Log level
            message: Log message
            extra_context: Additional context for this log entry
            exception: Exception to include in log
            lazy_context: Function that returns context (called only if logging)
        """
        # Check if logging is enabled for this level using standard library
        # structlog filtering is handled by the wrapper_class, but we want to avoid
        # expensive lazy context evaluation if logging is disabled
        try:
            # Get the underlying logger's level
            if level.upper() == "HAPPY":
                log_level = 25  # Custom HAPPY level
            else:
                log_level = getattr(logging, level.upper())
            if logging.getLogger(self.logger_name).getEffectiveLevel() > log_level:
                return
        except (AttributeError, ValueError):
            # If level checking fails, just proceed with logging
            pass

        # Build context
        context = self._get_current_context()

        if extra_context:
            context.update(extra_context)

        # Add lazy context only if logging
        if lazy_context:
            try:
                context.update(lazy_context())
            except Exception as e:
                # Don't let context generation break logging
                context["context_error"] = str(e)

        # Add exception information
        if exception:
            context.update(
                {
                    ContextKeys.ERROR_TYPE: type(exception).__name__,
                    "error_message": str(exception),
                }
            )

        # Add timestamp
        context["timestamp"] = datetime.utcnow().isoformat()

        # Handle HAPPY level filtering
        if level.upper() == "HAPPY":
            # For HAPPY level, use info but mark it specially
            context["level"] = "HAPPY"
            context["_level_numeric"] = 25
            log_method = self._logger.info
        else:
            log_method = getattr(self._logger, level.lower())

        log_method(message, **context)

    def trace(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log trace message (very detailed debugging)."""
        self._log_with_context(
            "DEBUG",  # Map to DEBUG since TRACE isn't standard
            f"[TRACE] {message}",
            extra_context=extra_context,
            exception=exception,
            lazy_context=lazy_context,
        )

    def debug(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log debug message."""
        self._log_with_context(
            "DEBUG", message, extra_context=extra_context, lazy_context=lazy_context
        )

    def info(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log info message."""
        self._log_with_context(
            "INFO", message, extra_context=extra_context, lazy_context=lazy_context
        )

    def happy(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log happy milestone message (job progression, success events)."""
        self._log_with_context(
            "HAPPY", message, extra_context=extra_context, lazy_context=lazy_context
        )

    def warning(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log warning message."""
        self._log_with_context(
            "WARNING",
            message,
            extra_context=extra_context,
            exception=exception,
            lazy_context=lazy_context,
        )

    def error(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log error message."""
        self._log_with_context(
            "ERROR",
            message,
            extra_context=extra_context,
            exception=exception,
            lazy_context=lazy_context,
        )

    def critical(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log critical message."""
        self._log_with_context(
            "CRITICAL",
            message,
            extra_context=extra_context,
            exception=exception,
            lazy_context=lazy_context,
        )

    def with_context(self, **context: Any) -> "ContextualLogger":
        """
        Create contextual logger with additional context.

        Args:
            **context: Context to add to all subsequent log messages

        Returns:
            ContextualLogger instance with added context
        """
        return ContextualLogger(self, context)

    def set_request_context(self, **context: Any) -> None:
        """
        Set request-level context (thread-local).

        Args:
            **context: Context to add to thread-local storage
        """
        if not hasattr(self._local, "context"):
            self._local.context = {}
        if not hasattr(self._local, "context_keys"):
            self._local.context_keys = set()
        self._local.context.update(context)
        self._local.context_keys.update(context.keys())

        try:
            structlog_contextvars.bind_contextvars(**context)
        except Exception:  # pragma: no cover - defensive guard if structlog misconfigured
            # Fall back silently; structured logging still works with thread-local context
            pass

    def clear_request_context(self) -> None:
        """Clear request-level context."""
        if hasattr(self._local, "context"):
            keys_to_clear = list(
                getattr(self._local, "context_keys", set())
                or self._local.context.keys()
            )
            try:
                if keys_to_clear:
                    structlog_contextvars.unbind_contextvars(*keys_to_clear)
                else:
                    structlog_contextvars.clear_contextvars()
            except (LookupError, Exception):  # pragma: no cover - safety net
                # Ensure contextvars are reset even if specific keys missing
                try:
                    structlog_contextvars.clear_contextvars()
                except Exception:
                    pass
            self._local.context.clear()
            if hasattr(self._local, "context_keys"):
                self._local.context_keys.clear()

    def performance_timer(
        self, operation: str, *, threshold_ms: Optional[float] = None
    ) -> "PerformanceTimer":
        """
        Create performance timer for operation.

        Args:
            operation: Operation name for timing
            threshold_ms: Only log if duration exceeds threshold

        Returns:
            Performance timer context manager
        """
        return PerformanceTimer(self, operation, threshold_ms=threshold_ms)

    @contextmanager
    def operation_context(self, operation: str, **additional_context: Any) -> Any:
        """
        Context manager for operation with automatic timing and structured logging.

        Args:
            operation: Operation name
            **additional_context: Additional context for the operation

        Yields:
            ContextualLogger instance for the operation
        """
        start_time = time.perf_counter()

        # Create contextual logger for this operation
        context = {ContextKeys.OPERATION: operation}
        context.update(additional_context)
        contextual_logger = self.with_context(**context)

        contextual_logger.debug(f"Starting operation: {operation}")

        try:
            yield contextual_logger
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            contextual_logger.error(
                f"Operation '{operation}' failed",
                extra_context={ContextKeys.DURATION_MS: round(duration_ms, 2)},
                exception=e,
            )
            raise
        else:
            duration_ms = (time.perf_counter() - start_time) * 1000
            contextual_logger.debug(
                f"Operation '{operation}' completed successfully",
                extra_context={ContextKeys.DURATION_MS: round(duration_ms, 2)},
            )


class ContextualLogger:
    """
    Logger wrapper that adds context to all log messages.

    Provides convenient context management for related operations.
    """

    def __init__(self, base_logger: StructuredLogger, context: Dict[str, Any]):
        self.base_logger = base_logger
        self.context = context

    def _log_with_added_context(
        self,
        method_name: str,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log with additional context."""
        combined_context = self.context.copy()
        if extra_context:
            combined_context.update(extra_context)

        method = getattr(self.base_logger, method_name)

        # Build kwargs based on method signature with proper typing
        kwargs: Dict[str, Any] = {"extra_context": combined_context}
        if lazy_context is not None:
            kwargs["lazy_context"] = lazy_context
        if exception is not None and method_name in ("warning", "error", "critical"):
            kwargs["exception"] = exception

        method(message, **kwargs)

    def trace(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log trace with context."""
        self._log_with_added_context(
            "trace",
            message,
            extra_context=extra_context,
            exception=exception,
            lazy_context=lazy_context,
        )

    def debug(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log debug with context."""
        self._log_with_added_context(
            "debug", message, extra_context=extra_context, lazy_context=lazy_context
        )

    def info(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log info with context."""
        self._log_with_added_context(
            "info", message, extra_context=extra_context, lazy_context=lazy_context
        )

    def happy(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log happy milestone with context."""
        self._log_with_added_context(
            "happy", message, extra_context=extra_context, lazy_context=lazy_context
        )

    def warning(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log warning with context."""
        self._log_with_added_context(
            "warning",
            message,
            extra_context=extra_context,
            exception=exception,
            lazy_context=lazy_context,
        )

    def error(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log error with context."""
        self._log_with_added_context(
            "error",
            message,
            extra_context=extra_context,
            exception=exception,
            lazy_context=lazy_context,
        )

    def critical(
        self,
        message: str,
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        lazy_context: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Log critical with context."""
        self._log_with_added_context(
            "critical",
            message,
            extra_context=extra_context,
            exception=exception,
            lazy_context=lazy_context,
        )

    def with_context(self, **additional_context: Any) -> "ContextualLogger":
        """Create new contextual logger with additional context."""
        combined_context = self.context.copy()
        combined_context.update(additional_context)
        return ContextualLogger(self.base_logger, combined_context)

    def performance_timer(
        self, operation: str, *, threshold_ms: Optional[float] = None
    ) -> "PerformanceTimer":
        """Create performance timer with current context."""
        return PerformanceTimer(self.base_logger, operation, threshold_ms=threshold_ms)

    @contextmanager
    def operation_context(self, operation: str, **additional_context: Any) -> Any:
        """Context manager for operation with automatic timing and structured logging."""
        start_time = time.perf_counter()

        # Create contextual logger for this operation
        context = {ContextKeys.OPERATION: operation}
        context.update(additional_context)
        contextual_logger = self.with_context(**context)

        contextual_logger.debug(f"Starting operation: {operation}")

        try:
            yield contextual_logger
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            contextual_logger.error(
                f"Operation '{operation}' failed",
                extra_context={ContextKeys.DURATION_MS: round(duration_ms, 2)},
                exception=e,
            )
            raise
        else:
            duration_ms = (time.perf_counter() - start_time) * 1000
            contextual_logger.debug(
                f"Operation '{operation}' completed successfully",
                extra_context={ContextKeys.DURATION_MS: round(duration_ms, 2)},
            )


class PerformanceTimer:
    """
    Context manager for performance timing with automatic logging.

    Provides convenient timing for operations with threshold-based logging.
    """

    def __init__(
        self,
        logger: StructuredLogger,
        operation: str,
        *,
        threshold_ms: Optional[float] = None,
        log_level: str = "DEBUG",
    ):
        """
        Initialize performance timer.

        Args:
            logger: Logger to use for timing output
            operation: Operation being timed
            threshold_ms: Only log if duration exceeds threshold
            log_level: Log level for timing messages
        """
        self.logger = logger
        self.operation = operation
        self.threshold_ms = threshold_ms
        self.log_level = log_level
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> "PerformanceTimer":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[types.TracebackType],
    ) -> None:
        """End timing and log result."""
        self.end_time = time.perf_counter()

        if self.start_time is not None:
            duration_ms = (self.end_time - self.start_time) * 1000

            # Only log if above threshold
            if self.threshold_ms is None or duration_ms >= self.threshold_ms:
                log_method = getattr(self.logger, self.log_level.lower())
                log_method(
                    f"Operation '{self.operation}' completed",
                    extra_context={
                        ContextKeys.OPERATION: self.operation,
                        ContextKeys.DURATION_MS: round(duration_ms, 2),
                    },
                )

    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.start_time is not None and self.end_time is not None:
            return (self.end_time - self.start_time) * 1000
        return None


class LoggerFactory:
    """
    Centralized factory for creating component loggers.

    Manages logger lifecycle and configuration integration.
    """

    _loggers: Dict[str, StructuredLogger] = {}
    _configured: bool = False
    _happy_level_enabled: bool = False

    @classmethod
    def configure_logging(
        cls,
        level: str = "INFO",
        format_type: str = "structured",
        log_file: Optional[Path] = None,
        enable_console: bool = True,
        enable_performance_tracking: bool = False,
    ) -> None:
        """
        Configure global logging settings.

        Args:
            level: Minimum log level (TRACE, DEBUG, INFO, HAPPY, WARNING, ERROR, CRITICAL)
            format_type: Log format (structured, simple, json, console)
            log_file: Optional log file path
            enable_console: Enable console output
            enable_performance_tracking: Enable performance tracking globally
        """
        # Add custom HAPPY log level to Python logging
        HAPPY_LEVEL = 25  # Between INFO (20) and WARNING (30)
        logging.addLevelName(HAPPY_LEVEL, "HAPPY")

        def happy(self, message, *args, **kwargs):
            if self.isEnabledFor(HAPPY_LEVEL):
                self._log(HAPPY_LEVEL, message, args, **kwargs)

        # Add happy method to Logger class
        logging.Logger.happy = happy

        # Store configuration for filtering
        cls._happy_level_enabled = level.upper() == "HAPPY"
        # Configure structlog
        # NOTE: Any type justified for processors - structlog processor signatures are complex and dynamic
        processors: List[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
        ]

        # Add HAPPY level filtering and emoji processor if HAPPY level is enabled
        if level.upper() == "HAPPY":
            from .happy_filter import happy_level_filter, add_happy_emoji_prefix

            processors.extend(
                [
                    add_happy_emoji_prefix,
                    happy_level_filter,
                ]
            )

        if format_type == "json":
            processors.append(structlog.processors.JSONRenderer())
        elif format_type == "console":
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.processors.KeyValueRenderer())

        # For HAPPY level, use INFO level filtering but mark messages specially
        if level.upper() == "HAPPY":
            filter_level = logging.INFO
        else:
            filter_level = getattr(logging, level.upper())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(filter_level),
            logger_factory=structlog.WriteLoggerFactory(file=sys.stderr),
            cache_logger_on_first_use=True,
        )

        # Configure standard library logging
        logging_config: Dict[str, Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
                },
                "simple": {"format": "%(levelname)s: %(message)s"},
                "console": {
                    "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
                },
                "json": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processor": structlog.processors.JSONRenderer(),
                },
            },
            "handlers": {},
            "root": {
                "level": "INFO" if level.upper() == "HAPPY" else level,
                "handlers": [],
            },
        }

        # Add console handler
        if enable_console:
            logging_config["handlers"]["console"] = {
                "class": "logging.StreamHandler",
                "formatter": format_type if format_type != "json" else "structured",
                "level": "INFO" if level.upper() == "HAPPY" else level,
                "stream": "ext://sys.stderr",
            }
            logging_config["root"]["handlers"].append("console")

        # Add file handler
        if log_file:
            logging_config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": format_type if format_type != "json" else "structured",
                "level": level,
                "filename": str(log_file),
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
            }
            logging_config["root"]["handlers"].append("file")

        logging.config.dictConfig(logging_config)
        cls._configured = True

    @classmethod
    def get_logger(
        cls,
        component: str,
        *,
        logger_name: Optional[str] = None,
        base_context: Optional[Dict[str, Any]] = None,
        enable_performance_tracking: Optional[bool] = None,
    ) -> StructuredLogger:
        """
        Get or create logger for component.

        Args:
            component: Component name
            logger_name: Optional logger name
            base_context: Base context for all log messages
            enable_performance_tracking: Enable performance tracking

        Returns:
            StructuredLogger instance for component
        """
        cache_key = f"{component}:{logger_name or component}"

        if cache_key not in cls._loggers:
            cls._loggers[cache_key] = StructuredLogger(
                component,
                logger_name=logger_name,
                base_context=base_context,
                enable_performance_tracking=enable_performance_tracking or False,
            )

        return cls._loggers[cache_key]

    @classmethod
    def configure_from_settings(cls, settings: SettingsProtocol) -> None:
        """
        Configure logging from application settings.

        Args:
            settings: MnemosyneSettings instance with logging configuration
        """
        # Use settings to configure logging if available
        if hasattr(settings, "logging"):
            cls.configure_logging(
                level=settings.logging.level.value
                if hasattr(settings.logging, "level")
                else "INFO",
                format_type=getattr(settings.logging, "format", "structured"),
                log_file=getattr(settings.logging, "file", None),
                enable_console=True,
                enable_performance_tracking=getattr(settings.debug, "profiling", False)
                if hasattr(settings, "debug")
                else False,
            )
        else:
            # Fallback to default configuration
            cls.configure_logging()


# Convenience functions for component loggers
@lru_cache()
def get_cli_logger() -> StructuredLogger:
    """Get CLI component logger."""
    return LoggerFactory.get_logger("cli")


@lru_cache()
def get_server_logger() -> StructuredLogger:
    """Get server component logger."""
    return LoggerFactory.get_logger("server")


@lru_cache()
def get_store_logger() -> StructuredLogger:
    """Get store component logger."""
    return LoggerFactory.get_logger("store")


@lru_cache()
def get_config_logger() -> StructuredLogger:
    """Get configuration component logger."""
    return LoggerFactory.get_logger("config")


# Debug utilities
class DebugLogger:
    """
    Enhanced debug logging for development and troubleshooting.

    Provides additional context and verbosity in debug mode.
    """

    def __init__(self, base_logger: StructuredLogger, debug_enabled: bool = False):
        self.base_logger = base_logger
        self.debug_enabled = debug_enabled

    def trace_method_call(
        self, func_name: str, args: tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        """Trace method calls in debug mode."""
        if self.debug_enabled:
            self.base_logger.trace(
                f"Method call: {func_name}",
                extra_context={
                    "function": func_name,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )

    def dump_context(self, context_name: str, context_data: Dict[str, Any]) -> None:
        """Dump context information in debug mode."""
        if self.debug_enabled:
            self.base_logger.debug(
                f"Context dump: {context_name}",
                extra_context={"context_dump": context_data},
            )

    def performance_report(self, operation: str, timings: Dict[str, float]) -> None:
        """Report detailed performance information."""
        if self.debug_enabled:
            self.base_logger.info(
                f"Performance report: {operation}",
                extra_context={
                    "performance_timings": {k: f"{v:.2f}ms" for k, v in timings.items()}
                },
            )


def debug_trace(operation: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for debug tracing of operations."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # NOTE: Any types justified here - decorator must preserve arbitrary function signatures
            logger = get_store_logger()

            # Only trace if debug mode is enabled
            if hasattr(logger, "debug_enabled") and logger.debug_enabled:
                logger.trace(f"Starting {operation}")

                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration = (time.perf_counter() - start_time) * 1000
                    logger.trace(
                        f"Completed {operation}",
                        extra_context={ContextKeys.DURATION_MS: round(duration, 2)},
                    )
                    return result
                except Exception as e:
                    duration = (time.perf_counter() - start_time) * 1000
                    logger.trace(
                        f"Failed {operation}",
                        exception=e,
                        extra_context={ContextKeys.DURATION_MS: round(duration, 2)},
                    )
                    raise
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def log_cli_command(
    command_name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log CLI command execution."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # NOTE: Any types justified here - decorator must preserve arbitrary function signatures
            cli_logger = get_cli_logger()

            # Set command context
            cli_logger.set_request_context(
                cli_command=command_name,
                command_args=str(args),
                command_kwargs=str(kwargs),
            )

            with cli_logger.performance_timer(f"cli_command_{command_name}"):
                cli_logger.info(f"Starting CLI command: {command_name}")

                try:
                    result = func(*args, **kwargs)
                    cli_logger.info(f"CLI command completed: {command_name}")
                    return result
                except Exception as e:
                    cli_logger.error(f"CLI command failed: {command_name}", exception=e)
                    raise
                finally:
                    cli_logger.clear_request_context()

        return wrapper

    return decorator
