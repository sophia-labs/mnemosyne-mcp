"""
Stdio-based MCP server for Claude Code integration.

This version uses stdin/stdout transport instead of HTTP/SSE for better
compatibility with Claude Code's MCP client.
"""

import os
import sys
from pathlib import Path
from .standalone_server import create_standalone_mcp_server
from neem.utils.logging import LoggerFactory

# Configure logging from environment before creating loggers
# This allows Codex CLI and other tools to control logging verbosity
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_file_env = os.getenv("LOG_FILE")
log_file_path = None

if log_file_env:
    log_file_path = Path(log_file_env).expanduser().resolve()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

LoggerFactory.configure_logging(
    level=log_level,
    enable_console=True,
    log_file=log_file_path
)

logger = LoggerFactory.get_logger("mcp.standalone_server_stdio")

def run_stdio_mcp_server():
    """Run the standalone MCP server with stdio transport."""
    # Only log startup message if not in quiet mode
    if log_level not in ["WARNING", "ERROR", "CRITICAL"]:
        logger.info("🚀 Starting MCP server with stdio transport")

    # Create the MCP server (same as HTTP version)
    mcp_server = create_standalone_mcp_server()

    try:
        if log_level not in ["WARNING", "ERROR", "CRITICAL"]:
            logger.info("🔌 Running MCP server with stdio transport")

        # Use stdio transport instead of HTTP
        mcp_server.run()

    except KeyboardInterrupt:
        if log_level not in ["WARNING", "ERROR", "CRITICAL"]:
            logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        if log_level not in ["WARNING", "ERROR", "CRITICAL"]:
            logger.info("✅ MCP server shutdown complete")


if __name__ == "__main__":
    run_stdio_mcp_server()
