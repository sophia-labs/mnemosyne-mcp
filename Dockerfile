# Mnemosyne MCP Server
# Provides knowledge graph tools via MCP HTTP transport for OpenCode integration

FROM python:3.12-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Create non-root user matching k8s securityContext (uid/gid 1000)
RUN useradd --create-home --uid 1000 --user-group --shell /usr/sbin/nologin appuser \
    && chown appuser:appuser /app

# Configure Python runtime behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH="/app/.venv/bin:${PATH}"

# Switch to non-root user before installing deps (so .venv is owned by appuser)
USER appuser

# Copy all project files needed for build
COPY --chown=appuser:appuser pyproject.toml uv.lock LICENSE README.md ./
COPY --chown=appuser:appuser src/ ./src/

# Install dependencies (includes local package)
RUN uv sync --frozen --no-dev

# Default environment variables
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8003
ENV LOG_LEVEL=INFO

# Expose MCP HTTP port
EXPOSE 8003

# Run the HTTP transport server (--frozen prevents runtime sync of dev deps)
CMD ["uv", "run", "--frozen", "python", "-m", "neem.mcp.server.standalone_server"]
