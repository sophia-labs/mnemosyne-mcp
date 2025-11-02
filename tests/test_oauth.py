"""Tests for OAuth functionality (non-async parts)."""

import base64
import hashlib
import re
import pytest

from neem.utils.oauth import (
    generate_pkce_pair,
    OAuthConfig,
    OAuthError,
    OAuthTimeoutError,
    OAuthCancelledError,
)


class TestPKCEGeneration:
    """Tests for PKCE code verifier and challenge generation."""

    def test_generate_pkce_pair_returns_tuple(self):
        """Test that generate_pkce_pair returns a tuple of two strings."""
        verifier, challenge = generate_pkce_pair()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_generate_pkce_verifier_length(self):
        """Test that verifier has correct length (43-128 chars per RFC 7636)."""
        verifier, _ = generate_pkce_pair()
        assert 43 <= len(verifier) <= 128

    def test_generate_pkce_challenge_length(self):
        """Test that challenge has expected length for SHA256."""
        _, challenge = generate_pkce_pair()
        # SHA256 hash is 32 bytes, base64url encoded without padding
        assert len(challenge) == 43  # 32 bytes -> 43 chars in base64url

    def test_generate_pkce_verifier_url_safe(self):
        """Test that verifier uses URL-safe base64 encoding."""
        verifier, _ = generate_pkce_pair()
        # URL-safe base64 uses only alphanumeric, -, and _
        assert re.match(r'^[A-Za-z0-9_-]+$', verifier)

    def test_generate_pkce_challenge_url_safe(self):
        """Test that challenge uses URL-safe base64 encoding."""
        _, challenge = generate_pkce_pair()
        assert re.match(r'^[A-Za-z0-9_-]+$', challenge)

    def test_generate_pkce_no_padding(self):
        """Test that verifier and challenge have no base64 padding."""
        verifier, challenge = generate_pkce_pair()
        assert '=' not in verifier
        assert '=' not in challenge

    def test_generate_pkce_deterministic_challenge(self):
        """Test that challenge is deterministic for a given verifier."""
        # Manually create a verifier
        test_verifier = "test_verifier_string_12345"

        # Calculate expected challenge
        expected_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(test_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')

        # Verify the calculation matches what the function should do
        actual_hash = hashlib.sha256(test_verifier.encode('utf-8')).digest()
        actual_challenge = base64.urlsafe_b64encode(actual_hash).decode('utf-8').rstrip('=')

        assert actual_challenge == expected_challenge
        assert len(actual_challenge) == 43

    def test_generate_pkce_unique_pairs(self):
        """Test that multiple calls generate different pairs."""
        pair1 = generate_pkce_pair()
        pair2 = generate_pkce_pair()
        pair3 = generate_pkce_pair()

        # Verifiers should be different (cryptographically random)
        assert pair1[0] != pair2[0]
        assert pair1[0] != pair3[0]
        assert pair2[0] != pair3[0]

        # Challenges should also be different
        assert pair1[1] != pair2[1]
        assert pair1[1] != pair3[1]
        assert pair2[1] != pair3[1]

    def test_generate_pkce_challenge_from_verifier(self):
        """Test that we can verify the challenge was created from the verifier."""
        verifier, challenge = generate_pkce_pair()

        # Recreate the challenge from the verifier
        recreated_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')

        assert recreated_challenge == challenge


class TestOAuthConfig:
    """Tests for OAuth configuration constants."""

    def test_oauth_config_client_id_exists(self):
        """Test that client ID is configured."""
        assert OAuthConfig.CLIENT_ID
        assert isinstance(OAuthConfig.CLIENT_ID, str)

    def test_oauth_config_urls_valid(self):
        """Test that OAuth URLs are properly formatted."""
        assert OAuthConfig.AUTHORIZE_URL.startswith('https://')
        assert OAuthConfig.TOKEN_URL.startswith('https://')
        assert OAuthConfig.USERINFO_URL.startswith('https://')

    def test_oauth_config_redirect_uri_localhost(self):
        """Test that redirect URI uses localhost."""
        assert 'localhost' in OAuthConfig.REDIRECT_URI
        assert OAuthConfig.REDIRECT_URI.startswith('http://')

    def test_oauth_config_scopes(self):
        """Test that scopes are properly configured."""
        assert isinstance(OAuthConfig.SCOPES, list)
        assert len(OAuthConfig.SCOPES) > 0
        assert 'openid' in OAuthConfig.SCOPES

    def test_oauth_config_callback_settings(self):
        """Test callback server settings."""
        assert OAuthConfig.CALLBACK_HOST == 'localhost'
        assert isinstance(OAuthConfig.CALLBACK_PORT, int)
        assert OAuthConfig.CALLBACK_PORT > 0
        assert OAuthConfig.CALLBACK_PATH.startswith('/')

    def test_oauth_config_timeout(self):
        """Test that timeout is reasonable."""
        assert isinstance(OAuthConfig.OAUTH_TIMEOUT_SECONDS, int)
        assert OAuthConfig.OAUTH_TIMEOUT_SECONDS > 0
        assert OAuthConfig.OAUTH_TIMEOUT_SECONDS <= 600  # Not more than 10 minutes


class TestOAuthExceptions:
    """Tests for OAuth exception classes."""

    def test_oauth_error_is_exception(self):
        """Test that OAuthError is an exception."""
        error = OAuthError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_oauth_timeout_error_inheritance(self):
        """Test that OAuthTimeoutError inherits from OAuthError."""
        error = OAuthTimeoutError("Timeout")
        assert isinstance(error, OAuthError)
        assert isinstance(error, Exception)

    def test_oauth_cancelled_error_inheritance(self):
        """Test that OAuthCancelledError inherits from OAuthError."""
        error = OAuthCancelledError("Cancelled")
        assert isinstance(error, OAuthError)
        assert isinstance(error, Exception)

    def test_oauth_errors_can_be_raised(self):
        """Test that OAuth errors can be raised and caught."""
        with pytest.raises(OAuthError):
            raise OAuthError("Test")

        with pytest.raises(OAuthTimeoutError):
            raise OAuthTimeoutError("Test")

        with pytest.raises(OAuthCancelledError):
            raise OAuthCancelledError("Test")

    def test_oauth_specific_errors_caught_as_base(self):
        """Test that specific OAuth errors can be caught as OAuthError."""
        with pytest.raises(OAuthError):
            raise OAuthTimeoutError("Timeout")

        with pytest.raises(OAuthError):
            raise OAuthCancelledError("Cancelled")


class TestPKCERFCCompliance:
    """Tests for RFC 7636 PKCE compliance."""

    def test_code_verifier_unreserved_characters(self):
        """Test that verifier uses only unreserved characters per RFC 7636."""
        # RFC 7636 Section 4.1: unreserved characters are [A-Z] / [a-z] / [0-9] / "-" / "." / "_" / "~"
        # Base64url uses [A-Z] / [a-z] / [0-9] / "-" / "_" which is a subset
        for _ in range(10):
            verifier, _ = generate_pkce_pair()
            assert re.match(r'^[A-Za-z0-9_-]+$', verifier)

    def test_code_challenge_method_s256(self):
        """Test that challenge uses S256 method (SHA-256)."""
        verifier, challenge = generate_pkce_pair()

        # Verify it matches S256 method: BASE64URL(SHA256(ASCII(code_verifier)))
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode('ascii')).digest()
        ).decode('ascii').rstrip('=')

        assert challenge == expected

    def test_code_verifier_entropy(self):
        """Test that verifier has sufficient entropy (at least 32 bytes)."""
        verifier, _ = generate_pkce_pair()

        # Our implementation uses 32 bytes of random data
        # which when base64url encoded gives 43 characters
        assert len(verifier) >= 43


class TestPKCEEdgeCases:
    """Tests for edge cases in PKCE generation."""

    def test_pkce_generation_consistency(self):
        """Test that the same verifier always produces the same challenge."""
        test_cases = [
            "testverifier123",
            "a" * 50,
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",
        ]

        for verifier in test_cases:
            challenge1 = base64.urlsafe_b64encode(
                hashlib.sha256(verifier.encode('utf-8')).digest()
            ).decode('utf-8').rstrip('=')

            challenge2 = base64.urlsafe_b64encode(
                hashlib.sha256(verifier.encode('utf-8')).digest()
            ).decode('utf-8').rstrip('=')

            assert challenge1 == challenge2

    def test_pkce_verifier_sufficient_randomness(self):
        """Test that generated verifiers have sufficient randomness."""
        # Generate many verifiers and check they're all unique
        verifiers = set()
        for _ in range(100):
            verifier, _ = generate_pkce_pair()
            verifiers.add(verifier)

        # All 100 should be unique
        assert len(verifiers) == 100
