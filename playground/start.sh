#!/bin/bash
#
# MCP Document Playground - Start Script
#
# This script starts the playground server and optionally opens the browser.
#
# Usage:
#   ./playground/start.sh           # Start server only
#   ./playground/start.sh --open    # Start server and open browser
#   ./playground/start.sh --test    # Start server and run test script
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=================================================="
echo "MCP Document Playground"
echo "=================================================="
echo ""

# Check if pycrdt is installed
if ! uv run python -c "import pycrdt" 2>/dev/null; then
    echo "Installing pycrdt..."
    uv add pycrdt
fi

# Check if uvicorn is installed
if ! uv run python -c "import uvicorn" 2>/dev/null; then
    echo "Installing uvicorn..."
    uv add uvicorn
fi

# Parse arguments
OPEN_BROWSER=false
RUN_TEST=false

for arg in "$@"; do
    case $arg in
        --open|-o)
            OPEN_BROWSER=true
            ;;
        --test|-t)
            RUN_TEST=true
            ;;
    esac
done

# Function to open browser
open_browser() {
    sleep 2  # Wait for server to start
    URL="http://localhost:8765"

    if command -v xdg-open &> /dev/null; then
        xdg-open "$URL" &
    elif command -v open &> /dev/null; then
        open "$URL" &
    elif command -v wslview &> /dev/null; then
        wslview "$URL" &
    else
        echo "Browser: $URL"
    fi
}

# Function to run test
run_test() {
    sleep 3  # Wait for server to start
    echo ""
    echo "Running test script..."
    echo ""
    uv run python playground/test_mcp.py
}

# Start server
if $OPEN_BROWSER; then
    open_browser &
fi

if $RUN_TEST; then
    (run_test) &
fi

echo "Starting playground server..."
echo ""
echo "Browser UI:     http://localhost:8765"
echo "Health check:   http://localhost:8765/health"
echo "Document API:   http://localhost:8765/api/documents"
echo ""
echo "To test with MCP, run in another terminal:"
echo "  uv run python playground/test_mcp.py"
echo ""
echo "Or interactive mode:"
echo "  uv run python playground/test_mcp.py --interactive"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================================="
echo ""

uv run python playground/server.py
