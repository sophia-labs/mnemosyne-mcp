"""
MCP initialization commands for Mnemosyne CLI.

Handles OAuth authentication for the Mnemosyne MCP server.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

from .utils.oauth import run_oauth_flow, get_user_info, OAuthError, OAuthTimeoutError, OAuthCancelledError
from .utils.token_storage import (
    save_token,
    load_token,
    delete_token,
    validate_token_and_load,
    get_token_info,
    get_config_path
)

import structlog

logger = structlog.get_logger(__name__)

# Typer app for MCP commands
mcp_app = typer.Typer(
    name="mcp",
    help="MCP (Model Context Protocol) authentication and configuration",
    no_args_is_help=True
)

console = Console()


@mcp_app.command("init")
def init_command(
    api_url: Optional[str] = typer.Option(
        None,
        "--api-url",
        help="Custom API URL (defaults to https://api.sophia-labs.com)"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-authentication even if already logged in"
    )
):
    """
    Initialize Mnemosyne MCP with OAuth authentication.

    This command will:
    1. Open your browser for authentication
    2. Save your authentication token securely

    After authentication, manually add the MCP server to your client (Claude Code, Goose, etc.).
    See the README for setup instructions.

    Example:
        neem init
        neem init --api-url http://localhost:8000  # For development
    """
    try:
        asyncio.run(_init_async(api_url, force))
    except KeyboardInterrupt:
        rprint("\n\n[yellow]❌ Authentication cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger.error("Initialization failed", error=str(e))
        rprint(f"\n[red]❌ Initialization failed: {e}[/red]")
        sys.exit(1)


async def _init_async(api_url: Optional[str], force: bool):
    """Async implementation of init command."""

    # Header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Mnemosyne MCP Initialization[/bold cyan]\n" +
        "Knowledge Graph AI Integration",
        border_style="cyan"
    ))
    console.print()

    # Check if already authenticated
    if not force:
        existing_token = validate_token_and_load()
        if existing_token:
            rprint("[green]✓[/green] You're already authenticated!")
            rprint(f"   Token stored at: [dim]{get_config_path()}[/dim]")

            # Show token info
            token_info = get_token_info(existing_token)
            if token_info:
                email = token_info.get('email', 'unknown')
                rprint(f"   Logged in as: [cyan]{email}[/cyan]")

            rprint("\n[dim]Use --force to re-authenticate[/dim]")

            console.print()
            _print_setup_tip()
            return

    # Run OAuth flow
    try:
        rprint("[bold]Step 1/2:[/bold] Authenticating with Mnemosyne...")
        console.print()

        id_token, refresh_token = await run_oauth_flow()

        console.print()
        rprint("[green]✓[/green] Authentication successful!")
        if refresh_token:
            rprint("[dim]   (with refresh token for automatic renewal)[/dim]")

    except OAuthTimeoutError as e:
        rprint(f"\n[yellow]⏱️  {e}[/yellow]")
        sys.exit(1)

    except OAuthCancelledError as e:
        rprint(f"\n[yellow]🚫 {e}[/yellow]")
        sys.exit(1)

    except OAuthError as e:
        rprint(f"\n[red]❌ Authentication failed: {e}[/red]")
        rprint("\n[dim]Please check:")
        rprint("  • Your internet connection")
        rprint("  • Cognito configuration (contact admin if this persists)")
        sys.exit(1)

    # Save token
    console.print()
    rprint("[bold]Step 2/2:[/bold] Saving authentication token...")

    try:
        # Get user info if possible
        token_info = get_token_info(id_token)
        user_info = None

        if token_info:
            user_info = {
                "email": token_info.get('email'),
                "sub": token_info.get('sub'),
                "name": token_info.get('name') or token_info.get('given_name')
            }

        config_path = save_token(id_token, user_info, refresh_token=refresh_token)

        rprint(f"[green]✓[/green] Token saved to: [cyan]{config_path}[/cyan]")

        if user_info and user_info.get('email'):
            rprint(f"   Logged in as: [cyan]{user_info['email']}[/cyan]")

    except Exception as e:
        rprint(f"[red]❌ Failed to save token: {e}[/red]")
        sys.exit(1)

    console.print()
    _print_setup_tip()

    # Success message
    console.print()
    console.print(Panel.fit(
        "[bold green]✓ Setup Complete![/bold green]\n\n" +
        "Mnemosyne authentication is ready to use.\n\n" +
        "[yellow]Next step:[/yellow] Follow the README to connect Claude Code (and other MCP clients).\n\n" +
        "[dim]To test: Ask Claude to query your knowledge graphs once the MCP server is configured.[/dim]",
        border_style="green"
    ))
    console.print()


def _print_setup_tip():
    """Provide guidance for adding the MCP server to any client."""
    rprint("[bold]Next step:[/bold] Add the Mnemosyne MCP server to your client.")
    rprint("   See the README for setup instructions for Claude Code, Goose, and Codex.")


@mcp_app.command("status")
def status_command():
    """
    Show current authentication status.

    Example:
        neem status
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Mnemosyne MCP Status[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    # Check token
    token = validate_token_and_load()

    if token:
        rprint("[green]✓[/green] Authentication: [bold green]Active[/bold green]")
        rprint(f"   Token location: [cyan]{get_config_path()}[/cyan]")

        # Show token info
        token_info = get_token_info(token)
        if token_info:
            email = token_info.get('email', 'unknown')
            exp = token_info.get('exp')

            rprint(f"   Logged in as: [cyan]{email}[/cyan]")

            if exp:
                import time
                remaining = exp - time.time()

                if remaining > 0:
                    hours = int(remaining / 3600)
                    rprint(f"   Token expires in: [cyan]{hours} hours[/cyan]")
                else:
                    rprint(f"   Token status: [red]Expired[/red]")
    else:
        rprint("[yellow]✗[/yellow] Authentication: [bold yellow]Not logged in[/bold yellow]")
        rprint("   Run [cyan]neem init[/cyan] to authenticate")

    console.print()
    rprint("[dim]To add the MCP server to your client, see the README for setup instructions.[/dim]")
    console.print()


@mcp_app.command("logout")
def logout_command(
    keep_config: bool = typer.Option(
        False,
        "--keep-config",
        help="(Deprecated) Claude Code configuration is managed manually and never modified."
    )
):
    """
    Log out and remove authentication token.

    Example:
        neem logout
        neem logout --keep-config  # Legacy flag; Claude settings stay untouched
    """
    console.print()

    # Delete token
    deleted = delete_token()

    if deleted:
        rprint("[green]✓[/green] Authentication token deleted")
        rprint(f"   Removed: [cyan]{get_config_path()}[/cyan]")
    else:
        rprint("[yellow]ℹ[/yellow] No authentication token found")

    # Claude Code configuration is manual now; keep flag for compatibility.
    if keep_config:
        rprint("[dim]Claude Code settings left untouched (flag retained for compatibility).[/dim]")
    else:
        rprint(
            "[dim]Claude Code settings are managed manually. "
            "Use `claude mcp remove mnemosyne-graph` if you want to remove the entry.[/dim]"
        )

    console.print()
    rprint("[dim]Run 'neem init' to log in again[/dim]")
    console.print()


@mcp_app.command("config")
def config_command(
    show_token: bool = typer.Option(
        False,
        "--show-token",
        help="Show authentication token (security risk!)"
    )
):
    """
    Show detailed configuration information.

    Example:
        neem config
        neem config --show-token  # Include token in output
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Mnemosyne MCP Configuration[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    # Token config
    rprint("[bold]Authentication Config:[/bold]")
    token = load_token()

    if token:
        rprint(f"  Location: [cyan]{get_config_path()}[/cyan]")

        if show_token:
            rprint(f"  Token: [yellow]{token[:20]}...{token[-20:]}[/yellow]")
        else:
            rprint(f"  Token: [dim](hidden, use --show-token to display)[/dim]")

        token_info = get_token_info(token)
        if token_info:
            rprint(f"  Email: [cyan]{token_info.get('email', 'N/A')}[/cyan]")
            rprint(f"  User ID: [cyan]{token_info.get('sub', 'N/A')}[/cyan]")
    else:
        rprint("  [yellow]Not authenticated[/yellow]")

    console.print()
    rprint("[dim]For MCP server setup instructions, see the README.[/dim]")
    console.print()


def main() -> None:
    """Entry point for CLI script."""
    mcp_app()
