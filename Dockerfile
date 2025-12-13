# Mnemosyne MCP Server
# Provides knowledge graph tools via MCP HTTP transport for OpenCode integration

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy all project files needed for build
COPY pyproject.toml uv.lock LICENSE README.md ./
COPY src/ ./src/

# Install dependencies (includes local package)
RUN uv sync --frozen --no-dev

# Default environment variables
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8003
ENV LOG_LEVEL=INFO

# Expose MCP HTTP port
EXPOSE 8003

# Run the HTTP transport server
CMD ["uv", "run", "python", "-m", "neem.mcp.server.standalone_server"]
