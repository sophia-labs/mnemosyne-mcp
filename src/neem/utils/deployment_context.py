"""
Deployment context detection and configuration for SOIL-2 multi-environment support.

Provides context-aware configuration that adapts service endpoints and settings
based on deployment environment (Development, Docker Local, Docker Swarm, Testing, Production).
"""

from __future__ import annotations

import os
import subprocess
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path

from neem.utils.logging import get_cli_logger
from neem.utils.errors import ConfigurationError


class DeploymentContext(Enum):
    """Deployment context enumeration defining supported environments."""

    DEVELOPMENT = "development"  # Local + Docker hybrid
    DOCKER_LOCAL = "docker_local"  # Full Docker Desktop stack
    DOCKER_SWARM = "docker_swarm"  # Docker Swarm cluster
    STANDALONE = "standalone"  # Single container all-in-one
    TESTING = "testing"  # Isolated test environment
    PRODUCTION = "production"  # Production deployment


class ContextDetector:
    """
    Automatic deployment context detection with manual override support.

    Detects the deployment environment based on system indicators and environment variables,
    providing the foundation for context-aware configuration.
    """

    def __init__(self):
        self.logger = get_cli_logger().with_context(component="context_detector")

    def detect_context(self) -> DeploymentContext:
        """
        Automatically detect deployment context based on environment.

        Returns:
            DeploymentContext: Detected deployment environment
        """
        self.logger.debug("Starting automatic context detection")

        # 1. Check explicit environment variable override
        env_context = os.getenv("SOIL_DEPLOYMENT_CONTEXT")
        if env_context:
            try:
                context = DeploymentContext(env_context.lower())
                self.logger.info(
                    "Using explicit context from environment variable",
                    extra_context={
                        "context": context.value,
                        "source": "SOIL_DEPLOYMENT_CONTEXT",
                    },
                )
                return context
            except ValueError:
                self.logger.warning(
                    "Invalid deployment context in environment variable",
                    extra_context={
                        "invalid_context": env_context,
                        "valid_contexts": [c.value for c in DeploymentContext],
                    },
                )

        # 2. Check for Docker environment indicators
        self.logger.info("🔍 Starting Docker environment detection")
        is_docker = self._is_running_in_docker()

        if is_docker:
            self.logger.info("✅ Docker environment detected")

            # Check for ECS environment (should be production context)
            is_ecs = self._is_ecs_environment()
            if is_ecs:
                context = DeploymentContext.PRODUCTION
                self.logger.info(
                    "🚀 ECS production context detected",
                    extra_context={"detection_method": "ecs_environment"},
                )
                return context

            is_swarm = self._is_docker_swarm_active()
            self.logger.info(
                "🔍 Checking Docker Swarm status",
                extra_context={"is_swarm_active": is_swarm},
            )

            if is_swarm:
                context = DeploymentContext.DOCKER_SWARM
                self.logger.info(
                    "🐳 Docker Swarm context detected",
                    extra_context={"detection_method": "swarm_active"},
                )
                return context
            else:
                context = DeploymentContext.DOCKER_LOCAL
                self.logger.info(
                    "🐳 Docker local context detected",
                    extra_context={"detection_method": "docker_env"},
                )
                return context
        else:
            self.logger.info("❌ No Docker environment detected")

        # 3. Check for testing environment
        if self._is_testing_environment():
            context = DeploymentContext.TESTING
            self.logger.info(
                "Testing context detected",
                extra_context={"detection_method": "test_indicators"},
            )
            return context

        # 4. Default to development
        context = DeploymentContext.DEVELOPMENT
        self.logger.info(
            "Development context selected as default",
            extra_context={"detection_method": "default"},
        )
        return context

    def _is_running_in_docker(self) -> bool:
        """
        Detect if running inside Docker container.

        Returns:
            bool: True if running in Docker container
        """
        # 🔍 ARCHAEOLOGICAL DEBUGGING: Docker Detection Analysis
        self.logger.info("🔍 Docker Container Detection Starting")

        # Check standard Docker indicators
        dockerenv_exists = os.path.exists("/.dockerenv")
        docker_container_env = os.getenv("DOCKER_CONTAINER") == "true"
        container_env = os.getenv("CONTAINER") is not None
        
        # Check ECS-specific environment variables
        ecs_metadata_uri = os.getenv("ECS_CONTAINER_METADATA_URI_V4") is not None
        aws_execution_env = os.getenv("AWS_EXECUTION_ENV") is not None
        ecs_container_name = os.getenv("ECS_CONTAINER_NAME") is not None

        docker_indicators = [
            dockerenv_exists,
            docker_container_env,
            container_env,
            ecs_metadata_uri,
            aws_execution_env,
            ecs_container_name,
        ]

        self.logger.info(
            "📋 Docker Environment Indicators",
            extra_context={
                "dockerenv_file_exists": dockerenv_exists,
                "docker_container_env": os.getenv("DOCKER_CONTAINER"),
                "container_env": os.getenv("CONTAINER"),
                "ecs_metadata_uri": ecs_metadata_uri,
                "aws_execution_env": os.getenv("AWS_EXECUTION_ENV"),
                "ecs_container_name": os.getenv("ECS_CONTAINER_NAME"),
                "any_standard_indicators": any(docker_indicators),
            },
        )

        if any(docker_indicators):
            self.logger.info("✅ Docker detected via standard indicators")
            return True

        # Check cgroup for docker/containerd/ECS patterns
        cgroup_docker_detected = False
        cgroup_content = None
        try:
            with open("/proc/1/cgroup", "r") as f:
                cgroup_content = f.read()
                patterns = ["docker", "containerd", "kubepods", "ecs"]
                matches = [pattern for pattern in patterns if pattern in cgroup_content]
                if matches:
                    cgroup_docker_detected = True
                    self.logger.info(
                        "✅ Docker detected via cgroup",
                        extra_context={
                            "matched_patterns": matches,
                            "cgroup_sample": cgroup_content[:200],
                        },
                    )
                else:
                    self.logger.info(
                        "❌ No Docker patterns in cgroup",
                        extra_context={"cgroup_sample": cgroup_content[:200]},
                    )
        except (FileNotFoundError, PermissionError) as e:
            self.logger.warning(
                "⚠️  Cannot read cgroup info",
                extra_context={"error": str(e), "error_type": type(e).__name__},
            )

        final_result = cgroup_docker_detected
        self.logger.info(
            "🎯 Docker Detection Result",
            extra_context={
                "is_docker": final_result,
                "detection_method": "cgroup" if cgroup_docker_detected else "none",
            },
        )

        return final_result

    def _is_docker_swarm_active(self) -> bool:
        """
        Detect if Docker Swarm is active.

        Returns:
            bool: True if Docker Swarm is active
        """
        try:
            # First check environment variable override
            if os.getenv("DOCKER_SWARM_MODE") == "true":
                return True

            # Check Docker info for swarm state
            result = subprocess.run(
                ["docker", "info", "--format", "{{.Swarm.LocalNodeState}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            swarm_state = result.stdout.strip()
            is_active = swarm_state == "active"

            self.logger.debug(
                "Docker Swarm state checked",
                extra_context={
                    "swarm_state": swarm_state,
                    "is_active": is_active,
                    "method": "docker_info",
                },
            )

            return is_active

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as e:
            self.logger.debug(
                "Docker Swarm detection failed",
                extra_context={"error": str(e), "method": "docker_info"},
            )
            return False

    def _is_testing_environment(self) -> bool:
        """
        Detect if running in testing environment.

        Returns:
            bool: True if in testing environment
        """
        testing_indicators = [
            # Environment variables
            os.getenv("SOIL_TESTING") == "true",
            os.getenv("PYTEST_CURRENT_TEST") is not None,
            os.getenv("CI") is not None,
            # Common testing framework patterns
            "pytest" in os.getenv("_", ""),
            "unittest" in os.getenv("_", ""),
            # Test data directory patterns
            any(marker in os.getcwd() for marker in ["test", "tests", "testing"]),
        ]

        return any(testing_indicators)

    def _is_ecs_environment(self) -> bool:
        """
        Detect if running in AWS ECS environment.

        Returns:
            bool: True if running in ECS environment
        """
        ecs_indicators = [
            # ECS-specific environment variables
            os.getenv("ECS_CONTAINER_METADATA_URI_V4") is not None,
            os.getenv("ECS_CONTAINER_METADATA_URI") is not None,
            os.getenv("AWS_EXECUTION_ENV") is not None,
            os.getenv("ECS_CONTAINER_NAME") is not None,
        ]

        # Check cgroup for ECS patterns
        ecs_cgroup_detected = False
        try:
            with open("/proc/1/cgroup", "r") as f:
                cgroup_content = f.read()
                ecs_cgroup_detected = "/ecs/" in cgroup_content
        except (FileNotFoundError, PermissionError):
            pass

        ecs_indicators.append(ecs_cgroup_detected)

        is_ecs = any(ecs_indicators)
        
        self.logger.info(
            "🔍 ECS Environment Detection",
            extra_context={
                "ecs_metadata_uri_v4": os.getenv("ECS_CONTAINER_METADATA_URI_V4") is not None,
                "aws_execution_env": os.getenv("AWS_EXECUTION_ENV"),
                "ecs_container_name": os.getenv("ECS_CONTAINER_NAME") is not None,
                "ecs_cgroup_detected": ecs_cgroup_detected,
                "is_ecs_environment": is_ecs,
            },
        )

        return is_ecs


class ContextAwareSettings:
    """
    Context-aware configuration provider that adapts base settings to deployment context.

    Integrates with existing MnemosyneSettings while providing deployment-specific overrides
    and service discovery integration.
    """

    def __init__(self, context: Optional[DeploymentContext] = None, base_settings=None):
        """
        Initialize context-aware settings.

        Args:
            context: Deployment context. If None, auto-detects.
            base_settings: Base MnemosyneSettings instance. If None, creates new instance.
        """
        self.detector = ContextDetector()
        self.context = context or self.detector.detect_context()
        self.logger = get_cli_logger().with_context(
            component="context_aware_settings", context=self.context.value
        )

        # Import MnemosyneSettings here to avoid circular imports
        try:
            from .settings import MnemosyneSettings

            self.base_settings = base_settings or MnemosyneSettings()
        except ImportError:
            self.logger.warning(
                "MnemosyneSettings not available - using minimal configuration"
            )
            self.base_settings = None

        self.logger.info(
            "Context-aware settings initialized",
            extra_context={
                "deployment_context": self.context.value,
                "has_base_settings": self.base_settings is not None,
            },
        )

        # Apply context-specific overrides
        self.effective_settings = self._apply_context_overrides()

    def _apply_context_overrides(self) -> Dict[str, Any]:
        """
        Apply deployment context-specific configuration overrides.

        Returns:
            Dict with effective configuration settings
        """
        self.logger.debug("Applying context-specific configuration overrides")

        # Start with base settings
        if self.base_settings:
            # Convert Pydantic model to dict for manipulation
            effective = (
                self.base_settings.model_dump()
                if hasattr(self.base_settings, "model_dump")
                else {}
            )
        else:
            effective = {}

        # Apply context-specific overrides
        context_overrides = self._get_context_overrides()
        effective = self._merge_nested_config(effective, context_overrides)

        # Apply environment variable overrides
        env_overrides = self._get_environment_overrides()
        effective = self._merge_nested_config(effective, env_overrides)

        self.logger.info(
            "Configuration overrides applied",
            extra_context={
                "context_overrides_count": len(context_overrides),
                "env_overrides_count": len(env_overrides),
                "total_settings": len(effective),
            },
        )

        return effective

    def _merge_nested_config(
        self, base: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.

        Args:
            base: Base configuration to merge into
            overrides: Override values that should take precedence

        Returns:
            Updated configuration dictionary
        """
        for key, value in overrides.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                base[key] = self._merge_nested_config(base[key], value)
            else:
                base[key] = value
        return base

    def _get_context_overrides(self) -> Dict[str, Any]:
        """
        Get deployment context-specific configuration overrides.

        Returns:
            Dict with context-specific settings
        """
        if self.context == DeploymentContext.DEVELOPMENT:
            return {
                # Development: Localhost first, Docker fallback
                "server": {"host": "localhost", "port": 8000},
                "database": {"path": "./dev-data/db"},
                "logging": {"level": "DEBUG", "enable_tracing": True, "format": "json"},
                "service_discovery": {"enabled": True, "fallback_enabled": True},
            }

        elif self.context == DeploymentContext.STANDALONE:
            return {
                # Standalone: Server with embedded worker
                "server": {"host": "0.0.0.0", "port": 8000},
                "database": {"path": "/data/db"},
                "logging": {"level": "INFO", "enable_tracing": True, "format": "json"},
                "service_discovery": {
                    "enabled": True,
                    "docker_network": "soil-network",
                },
                "worker": {
                    "embedded": True,
                    "include_server_api": False,  # API served through main server
                },
            }

        elif self.context == DeploymentContext.DOCKER_LOCAL:
            return {
                # Docker Local: Internal Docker networks with external access
                "server": {"host": "0.0.0.0", "port": 8000},
                "database": {"path": "/data/db"},
                "logging": {"level": "INFO", "enable_tracing": True, "format": "json"},
                "service_discovery": {
                    "enabled": True,
                    "docker_network": "soil-network",  # Match generated template
                },
                "port_mappings": {"worker_api": "8001:8001"},
                "worker": {
                    "embedded": False,
                    "include_server_api": True,  # Worker exposes its own API
                    "api_port": 8001,
                },
            }

        elif self.context == DeploymentContext.DOCKER_SWARM:
            return {
                # Docker Swarm: Pure internal service discovery
                "server": {"host": "0.0.0.0", "port": 8000},
                "database": {"path": "/data/db"},
                "logging": {"level": "INFO", "enable_tracing": False, "format": "json"},
                "service_discovery": {
                    "enabled": True,
                    "docker_network": "soil-network",  # Match generated template
                    "health_check_timeout": 30,
                },
                "worker": {
                    "embedded": False,
                    "include_server_api": False,  # No external worker API in swarm
                    "internal_health_check": True,  # Use internal health checking
                },
                "security": {"internal_only": True, "ssl_enabled": True},
            }

        elif self.context == DeploymentContext.TESTING:
            return {
                # Testing: Isolated test containers
                "database": {
                    "path": None,  # In-memory only
                    "create_if_missing": False,
                },
                "logging": {"level": "WARNING", "file": None, "format": "json"},
                "service_discovery": {"timeout": 5},
                "test_mode": {"enabled": True, "mock_external_services": True},
            }

        elif self.context == DeploymentContext.PRODUCTION:
            return {
                # Production: Security-focused configuration with AWS integration
                "server": {"host": "0.0.0.0", "port": 8000},
                "database": {"path": "/data/db"},
                "logging": {
                    "level": "INFO",
                    "enable_tracing": False,
                    "audit_enabled": True,
                    "format": "json",
                },
                "service_discovery": {"enabled": True, "health_check_timeout": 30},
                "security": {
                    "internal_only": True,
                    "ssl_enabled": True,
                    "authentication_required": True,
                },
            }

        return {}

    def _get_environment_overrides(self) -> Dict[str, Any]:
        """
        Get configuration overrides from environment variables.

        Returns:
            Dict with environment variable overrides
        """
        env_overrides = {}

        # Common environment variable patterns
        env_mappings = {
            "SOIL_SERVER_HOST": "server.host",
            "SOIL_SERVER_PORT": "server.port",
            "SOIL_DATABASE_PATH": "database.path",
            "SOIL_LOG_LEVEL": "logging.level",
            "SOIL_LOGGING__FORMAT": "logging.format",
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Handle nested configuration paths
                self._set_nested_value(env_overrides, config_path, value)

        return env_overrides

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: str) -> None:
        """
        Set nested configuration value using dot-separated path.

        Args:
            config: Configuration dictionary to update
            path: Dot-separated configuration path
            value: Value to set
        """
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Type conversion for common patterns
        final_key = keys[-1]
        if final_key == "port":
            try:
                value = int(value)
            except ValueError:
                pass
        elif value.lower() in ("true", "false"):
            value = value.lower() == "true"

        current[final_key] = value

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get configuration setting with default fallback.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        current = self.effective_settings

        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def get_server_config(self) -> Dict[str, Any]:
        """
        Get server configuration for current context.

        Returns:
            Server configuration dictionary
        """
        return self.get_setting("server", {"host": "localhost", "port": 8000})

    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration for current context.

        Returns:
            Database configuration dictionary
        """
        return self.get_setting("database", {"path": "./data/db"})

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration for current context.

        Returns:
            Logging configuration dictionary
        """
        return self.get_setting("logging", {"level": "INFO", "format": "json"})

    def get_service_discovery_config(self) -> Dict[str, Any]:
        """
        Get service discovery configuration for current context.

        Returns:
            Service discovery configuration dictionary
        """
        return self.get_setting("service_discovery", {"enabled": False})

    def is_development_context(self) -> bool:
        """Check if running in development context."""
        return self.context == DeploymentContext.DEVELOPMENT

    def is_docker_context(self) -> bool:
        """Check if running in any Docker context."""
        return self.context in (
            DeploymentContext.DOCKER_LOCAL,
            DeploymentContext.DOCKER_SWARM,
        )

    def is_production_context(self) -> bool:
        """Check if running in production context."""
        return self.context == DeploymentContext.PRODUCTION

    def requires_service_discovery(self) -> bool:
        """Check if context requires service discovery."""
        return self.get_setting("service_discovery.enabled", False)

    def get_worker_config(self) -> Dict[str, Any]:
        """
        Get worker configuration for current context.

        Returns:
            Worker configuration dictionary
        """
        return self.get_setting(
            "worker",
            {
                "embedded": False,
                "include_server_api": False,
                "internal_health_check": False,
            },
        )

    def is_worker_embedded(self) -> bool:
        """Check if worker runs embedded in server."""
        return self.get_setting("worker.embedded", False)

    def should_include_worker_api(self) -> bool:
        """Check if worker should expose its own API."""
        return self.get_setting("worker.include_server_api", False)

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive context summary for debugging and monitoring.

        Returns:
            Dictionary with context information
        """
        return {
            "deployment_context": self.context.value,
            "server_config": self.get_server_config(),
            "worker_config": self.get_worker_config(),
            "worker_embedded": self.is_worker_embedded(),
            "service_discovery_enabled": self.requires_service_discovery(),
            "is_docker_context": self.is_docker_context(),
            "is_production_context": self.is_production_context(),
            "effective_settings_count": len(self.effective_settings),
        }


def get_deployment_context() -> DeploymentContext:
    """
    Get current deployment context using cached detector.

    Returns:
        DeploymentContext: Current deployment environment
    """
    detector = ContextDetector()
    return detector.detect_context()


def get_context_aware_settings(
    context: Optional[DeploymentContext] = None,
) -> ContextAwareSettings:
    """
    Get context-aware settings instance.

    Args:
        context: Optional deployment context override

    Returns:
        ContextAwareSettings: Configured settings for current context
    """
    return ContextAwareSettings(context=context)
