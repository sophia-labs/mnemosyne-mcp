"""
OAuth 2.1 Authentication Flow with PKCE for Mnemosyne MCP CLI.

This module implements the Authorization Code flow with PKCE (Proof Key for Code Exchange)
for authenticating with AWS Cognito. Used by the MCP CLI to obtain JWT tokens for API access.
"""

import asyncio
import secrets
import hashlib
import base64
import json
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from urllib.parse import urlencode, parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
import threading

import httpx
import structlog

logger = structlog.get_logger(__name__)


class OAuthConfig:
    """AWS Cognito OAuth configuration for Mnemosyne."""

    CLIENT_ID = "46raltmjse1gjkkt6hvq30tsk7"
    AUTHORIZE_URL = "https://auth.sophia-labs.com/oauth2/authorize"
    TOKEN_URL = "https://auth.sophia-labs.com/oauth2/token"
    USERINFO_URL = "https://auth.sophia-labs.com/oauth2/userInfo"
    REDIRECT_URI = "http://localhost:8080/callback"
    SCOPES = ["openid", "email", "profile"]

    # Callback server settings
    CALLBACK_HOST = "localhost"
    CALLBACK_PORT = 8080
    CALLBACK_PATH = "/callback"

    # Timeout for waiting for user to complete OAuth flow
    OAUTH_TIMEOUT_SECONDS = 300  # 5 minutes


class OAuthError(Exception):
    """Base exception for OAuth flow errors."""
    pass


class OAuthTimeoutError(OAuthError):
    """User didn't complete OAuth flow in time."""
    pass


class OAuthCancelledError(OAuthError):
    """User cancelled the OAuth flow."""
    pass


def generate_pkce_pair() -> Tuple[str, str]:
    """
    Generate PKCE code verifier and challenge.

    Returns:
        Tuple of (verifier, challenge)
    """
    # Generate cryptographically random verifier (43-128 chars)
    verifier = base64.urlsafe_b64encode(
        secrets.token_bytes(32)
    ).decode('utf-8').rstrip('=')

    # Create SHA256 challenge from verifier
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode('utf-8')).digest()
    ).decode('utf-8').rstrip('=')

    logger.debug(
        "Generated PKCE pair",
        verifier_length=len(verifier),
        challenge_length=len(challenge)
    )

    return verifier, challenge


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    # Class variables to store the authorization code
    authorization_code: Optional[str] = None
    error: Optional[str] = None
    error_description: Optional[str] = None

    def do_GET(self) -> None:
        """Handle GET request to callback endpoint."""
        # Parse query parameters
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        # Check for authorization code
        if 'code' in params:
            CallbackHandler.authorization_code = params['code'][0]
            self.send_success_response()
            logger.info("Received authorization code from OAuth provider")

        # Check for error
        elif 'error' in params:
            CallbackHandler.error = params['error'][0]
            CallbackHandler.error_description = params.get('error_description', ['Unknown error'])[0]
            self.send_error_response()
            logger.warning(
                "OAuth error received",
                error=CallbackHandler.error,
                description=CallbackHandler.error_description
            )

        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid callback - missing code or error")

    def send_success_response(self) -> None:
        """Send success HTML page."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authentication Successful</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                .container {
                    background: white;
                    padding: 3rem;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    text-align: center;
                    max-width: 400px;
                }
                h1 { color: #667eea; margin-bottom: 1rem; }
                p { color: #555; line-height: 1.6; }
                .success-icon { font-size: 48px; margin-bottom: 1rem; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-icon">✓</div>
                <h1>Authentication Successful!</h1>
                <p>You've successfully authenticated with Mnemosyne.</p>
                <p>You can close this window and return to your terminal.</p>
            </div>
        </body>
        </html>
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def send_error_response(self) -> None:
        """Send error HTML page."""
        error_msg = CallbackHandler.error_description or CallbackHandler.error or "Unknown error"
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authentication Failed</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                }}
                .container {{
                    background: white;
                    padding: 3rem;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    text-align: center;
                    max-width: 400px;
                }}
                h1 {{ color: #f5576c; margin-bottom: 1rem; }}
                p {{ color: #555; line-height: 1.6; }}
                .error-icon {{ font-size: 48px; margin-bottom: 1rem; }}
                code {{ background: #f5f5f5; padding: 0.2rem 0.5rem; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="error-icon">✗</div>
                <h1>Authentication Failed</h1>
                <p>{error_msg}</p>
                <p>Please close this window and try again in your terminal.</p>
            </div>
        </body>
        </html>
        """
        self.send_response(400)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging (we use structlog)."""
        pass


async def start_callback_server(timeout_seconds: int = OAuthConfig.OAUTH_TIMEOUT_SECONDS) -> str:
    """
    Start local HTTP server to receive OAuth callback.

    Args:
        timeout_seconds: Maximum time to wait for callback

    Returns:
        Authorization code from OAuth provider

    Raises:
        OAuthTimeoutError: If user doesn't complete flow in time
        OAuthCancelledError: If user explicitly cancelled
        OAuthError: For other OAuth errors
    """
    # Reset class variables
    CallbackHandler.authorization_code = None
    CallbackHandler.error = None
    CallbackHandler.error_description = None

    # Create server
    server = HTTPServer(
        (OAuthConfig.CALLBACK_HOST, OAuthConfig.CALLBACK_PORT),
        CallbackHandler
    )

    logger.info(
        "Starting OAuth callback server",
        host=OAuthConfig.CALLBACK_HOST,
        port=OAuthConfig.CALLBACK_PORT
    )

    # Run server in background thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # Wait for callback (poll every 0.5 seconds)
        elapsed = 0
        poll_interval = 0.5

        while elapsed < timeout_seconds:
            # Check if we got a code or error
            if CallbackHandler.authorization_code:
                code = CallbackHandler.authorization_code
                logger.info("OAuth callback received successfully")
                return code

            if CallbackHandler.error:
                error = CallbackHandler.error
                description = CallbackHandler.error_description

                if error == "access_denied":
                    raise OAuthCancelledError(f"User cancelled authentication: {description}")
                else:
                    raise OAuthError(f"OAuth error: {error} - {description}")

            # Wait and increment timer
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout
        raise OAuthTimeoutError(
            f"Authentication timed out after {timeout_seconds} seconds. "
            "Please try again."
        )

    finally:
        # Shutdown server
        server.shutdown()
        logger.debug("OAuth callback server stopped")


async def exchange_code_for_token(code: str, code_verifier: str) -> Dict[str, Any]:
    """
    Exchange authorization code for JWT tokens.

    Args:
        code: Authorization code from OAuth callback
        code_verifier: PKCE code verifier

    Returns:
        Token response with id_token, access_token, etc.

    Raises:
        OAuthError: If token exchange fails
    """
    logger.info("Exchanging authorization code for tokens")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                OAuthConfig.TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": OAuthConfig.CLIENT_ID,
                    "code": code,
                    "redirect_uri": OAuthConfig.REDIRECT_URI,
                    "code_verifier": code_verifier
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                timeout=30.0
            )

            response.raise_for_status()
            tokens = response.json()

            logger.info(
                "Successfully obtained tokens",
                has_id_token='id_token' in tokens,
                has_access_token='access_token' in tokens,
                has_refresh_token='refresh_token' in tokens
            )

            return tokens

        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get('error_description', error_data.get('error', str(e)))
            except:
                error_detail = e.response.text

            logger.error(
                "Token exchange failed",
                status_code=e.response.status_code,
                error=error_detail
            )
            raise OAuthError(f"Failed to exchange code for token: {error_detail}") from e

        except Exception as e:
            logger.error("Unexpected error during token exchange", error=str(e))
            raise OAuthError(f"Token exchange failed: {str(e)}") from e


async def refresh_access_token(refresh_token: str) -> Optional[Dict[str, Any]]:
    """
    Use refresh token to obtain new access and ID tokens.

    Args:
        refresh_token: OAuth refresh token from previous authentication

    Returns:
        Token response with new id_token, access_token, and possibly new refresh_token.
        Returns None if refresh fails (e.g., refresh token expired).
    """
    logger.info("Attempting to refresh access token")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                OAuthConfig.TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": OAuthConfig.CLIENT_ID,
                    "refresh_token": refresh_token,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                timeout=30.0
            )

            if response.status_code == 400:
                # Refresh token likely expired or revoked
                error_data = response.json()
                error = error_data.get("error", "unknown")
                logger.warning(
                    "Refresh token rejected",
                    error=error,
                    description=error_data.get("error_description"),
                )
                return None

            response.raise_for_status()
            tokens = response.json()

            logger.info(
                "Successfully refreshed tokens",
                has_id_token="id_token" in tokens,
                has_access_token="access_token" in tokens,
                has_new_refresh_token="refresh_token" in tokens,
            )

            return tokens

        except httpx.HTTPStatusError as e:
            logger.warning(
                "Token refresh failed with HTTP error",
                status_code=e.response.status_code,
            )
            return None

        except Exception as e:
            logger.warning("Token refresh failed", error=str(e))
            return None


async def run_oauth_flow() -> Tuple[str, Optional[str]]:
    """
    Run complete OAuth Authorization Code + PKCE flow.

    Returns:
        Tuple of (id_token, refresh_token) for API authentication.
        refresh_token may be None if not provided by the IdP.

    Raises:
        OAuthError: If any step of the flow fails
    """
    logger.info("Starting OAuth authentication flow")

    try:
        # 1. Generate PKCE pair
        verifier, challenge = generate_pkce_pair()

        # 2. Build authorization URL
        params = {
            "client_id": OAuthConfig.CLIENT_ID,
            "response_type": "code",
            "redirect_uri": OAuthConfig.REDIRECT_URI,
            "scope": " ".join(OAuthConfig.SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256"
        }
        auth_url = f"{OAuthConfig.AUTHORIZE_URL}?{urlencode(params)}"

        logger.debug("Built authorization URL", url_length=len(auth_url))

        # 3. Open browser and start callback server concurrently
        import sys
        print(f"\n🔐 Opening browser for authentication...", file=sys.stderr)
        print(f"📍 If browser doesn't open, visit:\n   {auth_url}\n", file=sys.stderr)

        # Try to open browser
        try:
            webbrowser.open(auth_url)
            print("🌐 Browser opened", file=sys.stderr)
        except Exception as e:
            logger.warning("Failed to open browser", error=str(e))
            print(f"⚠️  Could not open browser automatically", file=sys.stderr)
            print(f"   Please open the URL manually", file=sys.stderr)

        print(f"⏳ Waiting for authentication (timeout: {OAuthConfig.OAUTH_TIMEOUT_SECONDS}s)...", file=sys.stderr)
        print(f"   Press Ctrl+C to cancel\n", file=sys.stderr)

        # 4. Wait for callback
        code = await start_callback_server()

        # 5. Exchange code for tokens
        print("✓ Authorization received", file=sys.stderr)
        print("🔄 Exchanging code for tokens...", file=sys.stderr)

        tokens = await exchange_code_for_token(code, verifier)

        # 6. Extract ID token (this is what we use for API auth)
        id_token = tokens.get('id_token')
        if not id_token:
            raise OAuthError("No ID token in response - unexpected token format")

        # Also extract refresh token for automatic renewal
        refresh_token = tokens.get('refresh_token')

        print("✓ Tokens received", file=sys.stderr)
        logger.info(
            "OAuth flow completed successfully",
            has_refresh_token=refresh_token is not None,
        )

        return id_token, refresh_token

    except (OAuthTimeoutError, OAuthCancelledError) as e:
        logger.warning("OAuth flow cancelled or timed out", reason=str(e))
        raise

    except KeyboardInterrupt:
        logger.info("OAuth flow interrupted by user")
        raise OAuthCancelledError("Authentication cancelled by user")

    except Exception as e:
        logger.error("OAuth flow failed", error=str(e), error_type=type(e).__name__)
        raise


async def get_user_info(access_token: str) -> Dict[str, Any]:
    """
    Fetch user information from Cognito userInfo endpoint.

    Args:
        access_token: OAuth access token

    Returns:
        User information (email, name, etc.)
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                OAuthConfig.USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning("Failed to fetch user info", error=str(e))
            return {}
