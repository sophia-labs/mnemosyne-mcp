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


def save_token(token: str, user_info: Optional[Dict[str, Any]] = None) -> Path:
    """
    Save authentication token to config file with secure permissions.

    Args:
        token: JWT token to save
        user_info: Optional user information to save alongside token

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


def validate_token_and_load() -> Optional[str]:
    """
    Load token and validate it's not expired.

    Returns:
        Valid token if found and not expired, None otherwise
    """
    token = load_token()

    if not token:
        logger.debug("No token found")
        return None

    if is_token_expired(token):
        logger.warning("Stored token is expired")
        import sys
        print("⚠️  Stored authentication token has expired", file=sys.stderr)
        print("   Please run 'neem init' to re-authenticate", file=sys.stderr)
        return None

    return token
