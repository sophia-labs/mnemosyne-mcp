"""
Secure token storage for Mnemosyne MCP authentication.

Handles saving, loading, and validating JWT tokens for API access.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import os
import stat

import structlog
import jwt

DEV_TOKEN_ENV = "MNEMOSYNE_DEV_TOKEN"
DEV_USER_ENV = "MNEMOSYNE_DEV_USER_ID"
INTERNAL_SERVICE_SECRET_ENV = "MNEMOSYNE_INTERNAL_SERVICE_SECRET"

# Refresh token this many seconds before expiry to avoid edge cases
TOKEN_REFRESH_THRESHOLD_SECONDS = 300  # 5 minutes

logger = structlog.get_logger(__name__)


class TokenStorageError(Exception):
    """Base exception for token storage errors."""
    pass


class TokenConfig:
    """Token storage configuration."""

    # Default storage location
    DEFAULT_CONFIG_DIR = Path.home() / ".mnemosyne"
    DEFAULT_CONFIG_FILE = "config.json"

    # File permissions (user read/write only)
    SECURE_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR  # 0o600


def get_config_path() -> Path:
    """
    Get path to Mnemosyne config file.

    Respects MNEMOSYNE_CONFIG_DIR environment variable.

    Returns:
        Path to config.json
    """
    config_dir = os.getenv("MNEMOSYNE_CONFIG_DIR")

    if config_dir:
        base_dir = Path(config_dir)
    else:
        base_dir = TokenConfig.DEFAULT_CONFIG_DIR

    return base_dir / TokenConfig.DEFAULT_CONFIG_FILE


def get_dev_user_id() -> Optional[str]:
    """
    Return the dev-mode user identifier if explicitly configured.

    Preference order:
    1. MNEMOSYNE_DEV_USER_ID
    2. MNEMOSYNE_DEV_TOKEN (many local clusters treat the token as the user id)
    """
    user_id = os.getenv(DEV_USER_ENV)
    if user_id:
        return user_id.strip()

    dev_token = os.getenv(DEV_TOKEN_ENV)
    if dev_token:
        return dev_token.strip()

    return None


def get_internal_service_secret() -> Optional[str]:
    """
    Return the internal service secret for cluster-internal auth.

    When running as a sidecar in Kubernetes, this secret allows the MCP
    to authenticate with the API without needing a user JWT token.
    """
    secret = os.getenv(INTERNAL_SERVICE_SECRET_ENV)
    if secret:
        return secret.strip()
    return None


def ensure_config_dir() -> Path:
    """
    Ensure config directory exists with secure permissions.

    Returns:
        Path to config directory

    Raises:
        TokenStorageError: If directory cannot be created
    """
    config_dir = get_config_path().parent

    try:
        config_dir.mkdir(parents=True, exist_ok=True)

        # Set directory permissions (user only)
        config_dir.chmod(stat.S_IRWXU)  # 0o700

        logger.debug("Config directory ensured", path=str(config_dir))
        return config_dir

    except Exception as e:
        logger.error("Failed to create config directory", path=str(config_dir), error=str(e))
        raise TokenStorageError(f"Cannot create config directory: {config_dir}") from e


def save_token(
    token: str,
    user_info: Optional[Dict[str, Any]] = None,
    refresh_token: Optional[str] = None,
) -> Path:
    """
    Save authentication token to config file with secure permissions.

    Args:
        token: JWT token (id_token) to save
        user_info: Optional user information to save alongside token
        refresh_token: Optional OAuth refresh token for automatic token renewal

    Returns:
        Path to saved config file

    Raises:
        TokenStorageError: If token cannot be saved
    """
    config_path = get_config_path()
    ensure_config_dir()

    # Build config data
    config_data = {
        "token": token,
        "version": "1.0"
    }

    if refresh_token:
        config_data["refresh_token"] = refresh_token

    if user_info:
        config_data["user_info"] = user_info

    try:
        # Write config file
        config_path.write_text(json.dumps(config_data, indent=2))

        # Set secure permissions (user read/write only)
        config_path.chmod(TokenConfig.SECURE_PERMISSIONS)

        logger.info(
            "Token saved successfully",
            path=str(config_path),
            has_user_info=user_info is not None
        )

        return config_path

    except Exception as e:
        logger.error("Failed to save token", path=str(config_path), error=str(e))
        raise TokenStorageError(f"Cannot save token to {config_path}") from e


def load_token() -> Optional[str]:
    """
    Load authentication token from config file.

    Returns:
        JWT token if found and valid, None otherwise
    """
    config_path = get_config_path()

    if not config_path.exists():
        logger.debug("No token config file found", path=str(config_path))
        return None

    try:
        config_data = json.loads(config_path.read_text())
        token = config_data.get("token")

        if not token:
            logger.warning("Config file exists but contains no token", path=str(config_path))
            return None

        logger.debug("Token loaded successfully", path=str(config_path))
        return token

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in config file", path=str(config_path), error=str(e))
        return None

    except Exception as e:
        logger.error("Failed to load token", path=str(config_path), error=str(e))
        return None


def load_config() -> Optional[Dict[str, Any]]:
    """
    Load complete config including token and user info.

    Returns:
        Config dictionary if found, None otherwise
    """
    config_path = get_config_path()

    if not config_path.exists():
        return None

    try:
        return json.loads(config_path.read_text())
    except Exception as e:
        logger.error("Failed to load config", path=str(config_path), error=str(e))
        return None


def load_refresh_token() -> Optional[str]:
    """
    Load refresh token from config file.

    Returns:
        Refresh token if found, None otherwise
    """
    config = load_config()
    if not config:
        return None
    return config.get("refresh_token")


def delete_token() -> bool:
    """
    Delete saved token (logout).

    Returns:
        True if token was deleted, False if no token existed
    """
    config_path = get_config_path()

    if not config_path.exists():
        logger.debug("No token to delete", path=str(config_path))
        return False

    try:
        config_path.unlink()
        logger.info("Token deleted successfully", path=str(config_path))
        return True

    except Exception as e:
        logger.error("Failed to delete token", path=str(config_path), error=str(e))
        return False


def is_token_expired(token: str) -> bool:
    """
    Check if JWT token is expired (without validating signature).

    Args:
        token: JWT token to check

    Returns:
        True if expired or invalid, False if still valid
    """
    try:
        # Decode without verification (just to check expiration)
        payload = jwt.decode(token, options={"verify_signature": False})

        # Check exp claim
        exp = payload.get('exp')
        if not exp:
            logger.warning("Token has no expiration claim")
            return True

        # JWT exp is in seconds since epoch
        import time
        current_time = time.time()

        is_expired = current_time >= exp

        if is_expired:
            logger.debug("Token is expired", exp=exp, current_time=current_time)
        else:
            time_remaining = exp - current_time
            logger.debug("Token is valid", seconds_remaining=int(time_remaining))

        return is_expired

    except jwt.DecodeError as e:
        logger.error("Failed to decode token", error=str(e))
        return True

    except Exception as e:
        logger.error("Unexpected error checking token expiration", error=str(e))
        return True


def token_needs_refresh(token: str) -> bool:
    """
    Check if token is expired or will expire soon (within threshold).

    Args:
        token: JWT token to check

    Returns:
        True if token should be refreshed, False if still valid with buffer
    """
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        exp = payload.get('exp')
        if not exp:
            return True

        import time
        current_time = time.time()
        time_remaining = exp - current_time

        needs_refresh = time_remaining < TOKEN_REFRESH_THRESHOLD_SECONDS

        if needs_refresh:
            logger.debug(
                "Token needs refresh",
                seconds_remaining=int(time_remaining),
                threshold=TOKEN_REFRESH_THRESHOLD_SECONDS,
            )

        return needs_refresh

    except Exception:
        return True


def get_token_info(token: str) -> Optional[Dict[str, Any]]:
    """
    Extract information from JWT token without validation.

    Args:
        token: JWT token to decode

    Returns:
        Token payload if decodable, None otherwise
    """
    try:
        return jwt.decode(token, options={"verify_signature": False})
    except Exception as e:
        logger.error("Failed to decode token", error=str(e))
        return None


def get_user_id_from_token(token: Optional[str] = None) -> Optional[str]:
    """
    Extract user ID from a JWT token.

    First checks for dev mode user ID, then tries to extract from token.

    Args:
        token: Optional JWT token. If not provided, loads from storage.

    Returns:
        User ID if found, None otherwise
    """
    # Dev mode takes priority
    dev_user = get_dev_user_id()
    if dev_user:
        return dev_user

    # Try to extract from token
    if token is None:
        token = load_token()
    if not token:
        return None

    info = get_token_info(token)
    if not info:
        return None

    # Common JWT claims for user ID
    for claim in ("sub", "user_id", "uid"):
        if claim in info:
            return str(info[claim])

    return None


def _try_refresh_token() -> Optional[str]:
    """
    Attempt to refresh the token using stored refresh_token.

    Returns:
        New id_token if refresh succeeded, None otherwise
    """
    refresh_token = load_refresh_token()
    if not refresh_token:
        logger.debug("No refresh token available")
        return None

    try:
        import asyncio
        from .oauth import refresh_access_token

        # Run async refresh in sync context
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - this shouldn't happen often
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, refresh_access_token(refresh_token))
                tokens = future.result(timeout=35)
        except RuntimeError:
            # No running loop - normal case
            tokens = asyncio.run(refresh_access_token(refresh_token))

        if not tokens:
            logger.warning("Token refresh failed - refresh token may be expired")
            return None

        # Extract new tokens
        new_id_token = tokens.get("id_token")
        if not new_id_token:
            logger.warning("Refresh response missing id_token")
            return None

        # Cognito may or may not return a new refresh token
        new_refresh_token = tokens.get("refresh_token", refresh_token)

        # Save new tokens
        config = load_config() or {}
        user_info = config.get("user_info")
        save_token(new_id_token, user_info=user_info, refresh_token=new_refresh_token)

        logger.info("Token refreshed successfully")
        return new_id_token

    except Exception as e:
        logger.warning("Token refresh failed", error=str(e))
        return None


def validate_token_and_load() -> Optional[str]:
    """
    Load token and validate it's not expired. Auto-refreshes if needed.

    Returns:
        Valid token if found and not expired, None otherwise
    """
    dev_token = os.getenv(DEV_TOKEN_ENV)
    if dev_token:
        logger.warning(
            "Using development token override from environment",
            extra_context={"env_var": DEV_TOKEN_ENV},
        )
        return dev_token.strip()

    token = load_token()

    if not token:
        logger.debug("No token found")
        return None

    # Check if token needs refresh (expired or about to expire)
    if token_needs_refresh(token):
        logger.info("Token expired or expiring soon, attempting refresh")
        new_token = _try_refresh_token()
        if new_token:
            return new_token

        # Refresh failed - check if token is actually expired vs just close to expiry
        if is_token_expired(token):
            import sys
            print("⚠️  Authentication token has expired and refresh failed", file=sys.stderr)
            print("   Please run 'neem init' to re-authenticate", file=sys.stderr)
            return None
        else:
            # Token still valid, just couldn't refresh early - use it anyway
            logger.warning("Could not refresh token early, but token still valid")
            return token

    return token
