"""
MCP (Model Context Protocol) server for Mnemosyne.

Provides stdio-based MCP server for Claude Code integration.
"""

from .server.standalone_server_stdio import run_stdio_mcp_server

__all__ = ["run_stdio_mcp_server"]
