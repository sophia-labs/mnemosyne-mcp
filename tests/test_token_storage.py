"""Tests for token storage functionality."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import jwt

from neem.utils.token_storage import (
    get_config_path,
    ensure_config_dir,
    save_token,
    load_token,
    load_config,
    delete_token,
    is_token_expired,
    get_token_info,
    get_dev_user_id,
    validate_token_and_load,
    TokenConfig,
    TokenStorageError,
)


@pytest.fixture
def temp_config_dir(tmp_path, monkeypatch):
    """Fixture to use a temporary directory for config storage."""
    config_dir = tmp_path / ".mnemosyne"
    monkeypatch.setenv("MNEMOSYNE_CONFIG_DIR", str(config_dir))
    return config_dir


@pytest.fixture
def sample_jwt_token():
    """Create a sample JWT token with expiration."""
    payload = {
        "sub": "user123",
        "email": "test@example.com",
        "exp": int(time.time()) + 3600,  # Expires in 1 hour
    }
    # Create token without signature (we don't validate it)
    token = jwt.encode(payload, "secret", algorithm="HS256")
    return token


@pytest.fixture
def expired_jwt_token():
    """Create an expired JWT token."""
    payload = {
        "sub": "user123",
        "email": "test@example.com",
        "exp": int(time.time()) - 3600,  # Expired 1 hour ago
    }
    token = jwt.encode(payload, "secret", algorithm="HS256")
    return token


class TestConfigPath:
    """Tests for config path functions."""

    def test_get_config_path_default(self, monkeypatch):
        """Test getting default config path."""
        monkeypatch.delenv("MNEMOSYNE_CONFIG_DIR", raising=False)
        path = get_config_path()
        assert path == TokenConfig.DEFAULT_CONFIG_DIR / TokenConfig.DEFAULT_CONFIG_FILE

    def test_get_config_path_custom(self, tmp_path, monkeypatch):
        """Test getting custom config path from environment."""
        custom_dir = tmp_path / "custom"
        monkeypatch.setenv("MNEMOSYNE_CONFIG_DIR", str(custom_dir))
        path = get_config_path()
        assert path == custom_dir / TokenConfig.DEFAULT_CONFIG_FILE

    def test_ensure_config_dir_creates_directory(self, temp_config_dir):
        """Test that ensure_config_dir creates the directory."""
        assert not temp_config_dir.exists()
        result = ensure_config_dir()
        assert result.exists()
        assert result.is_dir()
        assert result == temp_config_dir

    def test_ensure_config_dir_idempotent(self, temp_config_dir):
        """Test that ensure_config_dir can be called multiple times."""
        ensure_config_dir()
        ensure_config_dir()
        assert temp_config_dir.exists()

    def test_ensure_config_dir_permissions(self, temp_config_dir):
        """Test that config directory has secure permissions."""
        result = ensure_config_dir()
        # Check that directory is readable and writable by owner
        assert result.stat().st_mode & 0o700


class TestTokenSave:
    """Tests for saving tokens."""

    def test_save_token_basic(self, temp_config_dir, sample_jwt_token):
        """Test saving a token without user info."""
        path = save_token(sample_jwt_token)
        assert path.exists()
        assert path == get_config_path()

        # Verify file contents
        data = json.loads(path.read_text())
        assert data["token"] == sample_jwt_token
        assert data["version"] == "1.0"
        assert "user_info" not in data

    def test_save_token_with_user_info(self, temp_config_dir, sample_jwt_token):
        """Test saving a token with user info."""
        user_info = {"email": "test@example.com", "name": "Test User"}
        path = save_token(sample_jwt_token, user_info)

        data = json.loads(path.read_text())
        assert data["token"] == sample_jwt_token
        assert data["user_info"] == user_info

    def test_save_token_overwrites_existing(self, temp_config_dir, sample_jwt_token):
        """Test that saving a token overwrites existing one."""
        # Save first token
        save_token("token1")

        # Save second token
        path = save_token(sample_jwt_token)

        # Verify only second token exists
        data = json.loads(path.read_text())
        assert data["token"] == sample_jwt_token

    def test_save_token_secure_permissions(self, temp_config_dir, sample_jwt_token):
        """Test that saved token has secure file permissions."""
        path = save_token(sample_jwt_token)
        # Check file is only readable/writable by owner (0o600)
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600


class TestTokenLoad:
    """Tests for loading tokens."""

    def test_load_token_success(self, temp_config_dir, sample_jwt_token):
        """Test loading a saved token."""
        save_token(sample_jwt_token)
        loaded_token = load_token()
        assert loaded_token == sample_jwt_token

    def test_load_token_not_found(self, temp_config_dir):
        """Test loading when no token exists."""
        token = load_token()
        assert token is None

    def test_load_token_invalid_json(self, temp_config_dir):
        """Test loading when config file has invalid JSON."""
        config_path = get_config_path()
        ensure_config_dir()
        config_path.write_text("invalid json {")

        token = load_token()
        assert token is None

    def test_load_token_missing_token_field(self, temp_config_dir):
        """Test loading when config file exists but has no token field."""
        config_path = get_config_path()
        ensure_config_dir()
        config_path.write_text(json.dumps({"version": "1.0"}))

        token = load_token()
        assert token is None

    def test_load_config_success(self, temp_config_dir, sample_jwt_token):
        """Test loading complete config."""
        user_info = {"email": "test@example.com"}
        save_token(sample_jwt_token, user_info)

        config = load_config()
        assert config is not None
        assert config["token"] == sample_jwt_token
        assert config["user_info"] == user_info
        assert config["version"] == "1.0"

    def test_load_config_not_found(self, temp_config_dir):
        """Test loading config when file doesn't exist."""
        config = load_config()
        assert config is None


class TestTokenDelete:
    """Tests for deleting tokens."""

    def test_delete_token_success(self, temp_config_dir, sample_jwt_token):
        """Test deleting an existing token."""
        save_token(sample_jwt_token)
        assert get_config_path().exists()

        result = delete_token()
        assert result is True
        assert not get_config_path().exists()

    def test_delete_token_not_exists(self, temp_config_dir):
        """Test deleting when no token exists."""
        result = delete_token()
        assert result is False


class TestTokenValidation:
    """Tests for token validation functions."""

    def test_is_token_expired_valid(self, sample_jwt_token):
        """Test checking expiration of a valid token."""
        assert is_token_expired(sample_jwt_token) is False

    def test_is_token_expired_expired(self, expired_jwt_token):
        """Test checking expiration of an expired token."""
        assert is_token_expired(expired_jwt_token) is True

    def test_is_token_expired_invalid_token(self):
        """Test checking expiration of an invalid token."""
        assert is_token_expired("not.a.valid.token") is True

    def test_is_token_expired_no_exp_claim(self):
        """Test checking token without exp claim."""
        payload = {"sub": "user123"}
        token = jwt.encode(payload, "secret", algorithm="HS256")
        assert is_token_expired(token) is True

    def test_get_token_info_success(self, sample_jwt_token):
        """Test extracting info from valid token."""
        info = get_token_info(sample_jwt_token)
        assert info is not None
        assert info["sub"] == "user123"
        assert info["email"] == "test@example.com"
        assert "exp" in info

    def test_get_token_info_invalid(self):
        """Test extracting info from invalid token."""
        info = get_token_info("invalid.token")
        assert info is None


class TestValidateTokenAndLoad:
    """Tests for combined validation and loading."""

    def test_validate_token_and_load_success(self, temp_config_dir, sample_jwt_token):
        """Test validating and loading a valid token."""
        save_token(sample_jwt_token)
        token = validate_token_and_load()
        assert token == sample_jwt_token

    def test_validate_token_and_load_no_token(self, temp_config_dir):
        """Test validating when no token exists."""
        token = validate_token_and_load()
        assert token is None

    def test_validate_token_and_load_expired(self, temp_config_dir, expired_jwt_token, capsys):
        """Test validating an expired token."""
        save_token(expired_jwt_token)
        token = validate_token_and_load()
        assert token is None

        # Check that user-friendly message was printed
        captured = capsys.readouterr()
        combined = f"{captured.out} {captured.err}".lower()
        assert "expired" in combined

    def test_validate_token_and_load_dev_override(self, temp_config_dir, monkeypatch):
        """Allow bypassing auth flow via MNEMOSYNE_DEV_TOKEN."""
        monkeypatch.setenv("MNEMOSYNE_DEV_TOKEN", "dev-token")
        token = validate_token_and_load()
        assert token == "dev-token"


class TestDevModeHelpers:
    """Tests for helper functions that support dev mode."""

    def test_get_dev_user_id_explicit_env(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DEV_USER_ID", "carol")
        monkeypatch.delenv("MNEMOSYNE_DEV_TOKEN", raising=False)
        assert get_dev_user_id() == "carol"

    def test_get_dev_user_id_falls_back_to_dev_token(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_DEV_USER_ID", raising=False)
        monkeypatch.setenv("MNEMOSYNE_DEV_TOKEN", "alice")
        assert get_dev_user_id() == "alice"

    def test_get_dev_user_id_none(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_DEV_USER_ID", raising=False)
        monkeypatch.delenv("MNEMOSYNE_DEV_TOKEN", raising=False)
        assert get_dev_user_id() is None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_save_token_creates_parent_directories(self, tmp_path, monkeypatch):
        """Test that save_token creates parent directories if they don't exist."""
        nested_dir = tmp_path / "a" / "b" / "c"
        monkeypatch.setenv("MNEMOSYNE_CONFIG_DIR", str(nested_dir))

        token = "test.token.here"
        path = save_token(token)
        assert path.exists()
        assert path.parent == nested_dir

    def test_config_path_with_spaces(self, tmp_path, monkeypatch):
        """Test config path with spaces in directory name."""
        config_dir = tmp_path / "my config dir"
        monkeypatch.setenv("MNEMOSYNE_CONFIG_DIR", str(config_dir))

        path = save_token("test.token")
        assert path.exists()
        assert "my config dir" in str(path)
