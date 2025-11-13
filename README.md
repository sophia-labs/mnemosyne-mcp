
# Mnemosyne MCP
## THIS IS A WORK IN PROGRESS AND THE DOCUMENTATION IS AI-GENERATED AND WILL BE REWRITTEN BY HUMAN BEFORE PEOPLE ARE WIDELY ENCOURAGED TO READ IT AND USE THIS CODE. THANK YOU FOR YOUR ATTENTION TO THIS MATTER XOXO VERA

**AI-powered knowledge graph integration for Claude Code, Goose & Codex**

The Mnemosyne MCP (`neem`) historically exposed a full suite of graph management tools. We are currently rebuilding those tools from scratch against a new FastAPI backend that runs inside our local kubectl context.

> **Status:** The stdio server now focuses solely on wiring Codex/Claude to the local FastAPI backend. All MCP tools have been removed intentionally while we redesign the interface.

**Current focus:**
- 🔌 Reliable connectivity to a local FastAPI backend (via env vars or kubectl port-forward)
- 🩺 Automatic startup health probe so you know whether the backend is reachable
- 🔐 Browser-based OAuth authentication (`neem init`) remains unchanged
- 🧱 Clean slate for the upcoming tool rearchitecture

## Installation

```bash
uv tool install -e . 
```

While developing locally with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
uv run neem --help
```

## Commands

```bash
neem init               # Authenticate (run browser-based OAuth)
neem status             # Show token status and Claude Code config
neem logout             # Remove saved token (optional: keep config)
neem config             # Inspect config details (with optional --show-token)
```

## Quick Start

### Step 1: Install and authenticate

```bash
# Install the package
uv tool install -e .

# Authenticate with Mnemosyne
neem init                     # Opens your browser to log in
```

> `neem init` handles authentication only—the next steps show how to connect each MCP client manually.

Before registering the MCP server, expose the FastAPI backend from your kubectl context (adjust service/namespace/ports as needed):

```bash
kubectl port-forward svc/mnemosyne-fastapi 8001:8000
```

### Step 2: Add MCP server to your agent

Skaffold’s default profile port-forwards `mnemosyne-api` on `8080` for HTTP and `mnemosyne-ws` on `8001` for WebSockets. Point `MNEMOSYNE_FASTAPI_URL` at the HTTP port and the MCP server will automatically assume the split WebSocket port on localhost (you can override it with `MNEMOSYNE_FASTAPI_WS_PORT` or `MNEMOSYNE_FASTAPI_WS_URL` if your layout differs).

#### Using Claude Code:

```bash
claude mcp add mnemosyne --scope user \
  --env MNEMOSYNE_FASTAPI_URL=http://127.0.0.1:8001 \
  --env LOG_LEVEL=ERROR \
  -- uv run neem-mcp-server
```

#### Using Codex
```bash
codex mcp add mnemosyne -- uv run neem-mcp-server \
  --env MNEMOSYNE_FASTAPI_URL=http://127.0.0.1:8001 \
  --env LOG_LEVEL=ERROR

> Dev-mode shortcut: append `--env MNEMOSYNE_DEV_TOKEN=<user>` and `--env MNEMOSYNE_DEV_USER_ID=<user>` to the commands above when the backend runs with `MNEMOSYNE_AUTH__MODE=dev_no_auth`. Both transports will impersonate that user without going through OAuth.
````

### Dev Mode (skip OAuth)

If the backend runs with `MNEMOSYNE_AUTH__MODE=dev_no_auth`, set both env vars before launching the MCP server to bypass the OAuth flow entirely:

```bash
export MNEMOSYNE_DEV_USER_ID=alice
export MNEMOSYNE_DEV_TOKEN=alice  # many clusters treat the token string as the user id
uv run neem-mcp-server
```

Both HTTP requests and the WebSocket handshake will send `X-User-ID: alice` plus `Sec-WebSocket-Protocol: Bearer.alice`, satisfying the backend’s dev-mode guards. Unset these envs when targeting production.

### Usage Examples

After registering the server, ask your MCP client to run `list_graphs`. It submits a job, streams realtime events over `/ws`, and falls back to HTTP polling when the backend does not advertise push hints.

## FastAPI Backend Configuration

The MCP server now assumes it should talk to the FastAPI backend that runs in your local kubectl context.

1. Point `kubectl` at the desired cluster (`kubectl config use-context ...`).
2. Port-forward the FastAPI service so it is reachable on your workstation (example: `kubectl port-forward svc/mnemosyne-fastapi 8001:8000`).
3. Start `neem-mcp-server` with one of the supported backend configuration options:
   - `MNEMOSYNE_FASTAPI_URL` (preferred) or the legacy `MNEMOSYNE_API_URL`.
   - `MNEMOSYNE_FASTAPI_HOST`, `MNEMOSYNE_FASTAPI_PORT`, and optional `MNEMOSYNE_FASTAPI_SCHEME` if you want to supply host/port separately (handy for kubectl port-forward scripts).
   - `MNEMOSYNE_FASTAPI_HEALTH_PATH` if the FastAPI app exposes a non-standard health endpoint (defaults to `/health`).

If none of these environment variables are set the server defaults to `http://127.0.0.1:8001`, which lines up with the sample port-forward above. On startup we issue a lightweight health probe so you immediately know whether the backend is reachable.

### Token Management

Tokens expire after a day. Re-run `neem init --force` whenever you need a fresh token, and restart Claude Code afterwards.

**Important**: Set `LOG_LEVEL=ERROR` for Codex CLI to avoid any stderr interference with the stdio protocol.

## Tooling Status

The first push-enabled tool is available:

- `list_graphs` – submits a `list_graphs` job, streams realtime updates via the backend’s `/ws` gateway when available, and falls back to HTTP polling otherwise. Ask Claude/Codex to “run list_graphs” after registering the MCP server to see a proof-of-concept end-to-end.

More graph/query tools will land once the FastAPI contract solidifies.

## Configuration

- Tokens are stored at `~/.mnemosyne/config.json` (override with
  `MNEMOSYNE_CONFIG_DIR`).

## Architecture

This package contains two main components:

### 1. CLI Tool (`neem`)
Located in `neem.cli`:
- OAuth PKCE authentication flow (`neem.utils.oauth`)
- Secure token storage (`neem.utils.token_storage`)
- Claude Code configuration management (`neem.utils.claude_config`)

### 2. MCP Server (`neem-mcp-server`)
Located in `neem.mcp.server`:
- **Stdio transport** – Communicates with Claude Code via stdin/stdout
- **Backend resolver** – Determines the FastAPI base URL from env vars or kubectl service hosts
- **Health probe** – Pings the FastAPI backend on startup so you know whether the port-forward/context is correct
- **Realtime job wiring** – `neem.mcp.jobs` ships a websocket-friendly client that tools can use to subscribe to job progress once the backend emits hints
- **Structured logging** – All logs go to stderr (stdio-safe) with optional file output

Key design principles for this reset:
- **Local-first loops** – Assume developers are targeting a FastAPI pod through kubectl
- **Minimal surface area** – Keep the server slim until the new tool contract is finalized
- **Explicit configuration** – Prefer environment variables over hidden defaults so CLI harnesses can inject settings

## Development

### Local Development

```bash
# Install in development mode
uv sync
uv pip install -e .

# Run the CLI
uv run neem init

# Test the MCP server
uv run neem-mcp-server
```

### Project Structure

```
src/neem/
├── cli.py                          # CLI commands
├── mcp/
│   ├── server/
│   │   ├── standalone_server.py    # FastAPI backend resolver + health probe (no tools yet)
│   │   └── standalone_server_stdio.py  # Stdio transport wrapper
│   ├── session.py                  # Session management
│   ├── errors.py                   # MCP-specific errors
│   └── response_objects.py         # Formatted MCP responses
└── utils/
    ├── oauth.py                    # OAuth PKCE flow
    ├── token_storage.py            # Token persistence
    ├── claude_config.py            # Claude Code config management
    ├── logging.py                  # Structured logging
    ├── deployment_context.py       # Environment configuration
    └── errors.py                   # Base error classes
```

### Environment Variables

- `MNEMOSYNE_FASTAPI_URL` – Preferred FastAPI base URL (defaults to `http://127.0.0.1:8001`). The legacy `MNEMOSYNE_API_URL` is still honored if set.
- `MNEMOSYNE_FASTAPI_HOST`, `MNEMOSYNE_FASTAPI_PORT`, `MNEMOSYNE_FASTAPI_SCHEME` – Specify host/port separately (handy for scripted kubectl port-forwards).
- `MNEMOSYNE_FASTAPI_HEALTH_PATH` – Alternate health-check path if your FastAPI app doesn't expose `/health`.
- `MNEMOSYNE_FASTAPI_WS_URL` – Override the WebSocket gateway directly (defaults to `ws(s)://<host>/ws` derived from the HTTP base).
- `MNEMOSYNE_FASTAPI_WS_PATH` – Custom path appended to the derived WebSocket URL when `MNEMOSYNE_FASTAPI_WS_URL` is unset.
- `MNEMOSYNE_FASTAPI_WS_PORT` – Override just the WebSocket port while keeping the same host/path (useful when HTTP and WS are forwarded on different local ports).
- `MNEMOSYNE_FASTAPI_WS_DISABLE` – Set to `true` to opt out of WebSocket streaming (falls back to HTTP polling).
- `MNEMOSYNE_CONFIG_DIR` – Token storage location (default: `~/.mnemosyne`)
- `MNEMOSYNE_DEV_TOKEN` – Optional dev-only override that skips the OAuth flow by injecting the provided bearer token directly (use only on trusted local stacks).
- `CLAUDE_CODE_SETTINGS_PATH` – Claude settings file (default: `~/.claude/settings.json`)
- `LOG_LEVEL` – Logging verbosity (default: `INFO`)
  - `DEBUG` – Verbose logging for troubleshooting
  - `INFO` – Normal operational logging (default)
  - `WARNING` – Quiet mode, only warnings and errors
  - `ERROR` – Silent mode, only errors (recommended for Codex CLI)
  - `CRITICAL` – Minimal logging, critical errors only

Sessions are stored in-memory by default; no external cache service is required.

## Troubleshooting

### MCP Server Not Loading

**For Claude Code:**

1. **Check configuration**: `cat ~/.claude.json | grep mnemosyne-graph`
2. **Test server directly**: `echo '{"jsonrpc": "2.0", "method": "initialize", "id": 1}' | neem-mcp-server`
3. **Check logs**: Look for stderr output when Claude Code starts
4. **Verify token**: `neem status` should show "Active" authentication
5. **Restart Claude Code**: Configuration changes require a complete restart

**For Goose CLI:**

1. **Check configuration**: `cat ~/.config/goose/config.yaml | grep mnemosyne-graph`
2. **Verify extension is enabled**: `enabled: true` in the config
3. **Check timeout**: Increase to 600 seconds if server is slow to start
4. **Test in session**: Start a new Goose session and ask it to list available tools
5. **Check environment**: Ensure `MNEMOSYNE_FASTAPI_URL` (or legacy `MNEMOSYNE_API_URL`) is set correctly

**For Codex CLI:**

1. **Enable debug logging**: Codex intentionally silences stderr, making debugging difficult
2. **Set LOG_LEVEL**: Use `LOG_LEVEL=ERROR` in the environment to prevent stderr interference
3. **Test manually**: Run `echo '{"jsonrpc":"2.0","method":"initialize","id":1}' | neem-mcp-server` to verify it works
4. **Check configuration**: Ensure `codex.json` or your config file has the correct command and environment variables

### Authentication Issues

1. **Token expired**: Run `neem init --force` to get a fresh token
2. **Check token**: `neem config` to see token details
3. **Verify API access**: The token should have access to the API endpoint

### Upload Issues

1. **File not found**: Ensure the file path is absolute or relative to working directory
2. **Format detection**: Explicitly specify format with `rdf_format` parameter if auto-detection fails
3. **Validation errors**: Try `validation_level="lenient"` for less strict parsing

### Schema/Proto Errors (Goose/Gemini)

If you see errors like "Unknown name 'type'" or "Proto field is not repeating, cannot start list":

1. **Cause**: This was caused by using JSON Schema reserved keywords (`format`, `type`, etc.) as parameter names
2. **Fixed in**: Latest version uses `result_format` and `rdf_format` instead of `format`
3. **Solution**: Update to latest version with `pip install --upgrade neem`

**Note**: Parameter names were changed to avoid conflicts with JSON Schema keywords:
- `format` → `result_format` (in `sparql_query`)
- `format` → `rdf_format` (in `upload_file_to_graph`)
- `validation` → `validation_level` (in `upload_file_to_graph`)

**Goose + Gemini users**: If you get proto errors even with the latest version, disable Goose's built-in extensions temporarily:

```yaml
# In ~/.config/goose/config.yaml
extensions:
  computercontroller:
    enabled: false  # Temporarily disable
  developer:
    enabled: false  # Temporarily disable
```

This is a known issue with Goose's built-in extensions and Gemini compatibility (not a Mnemosyne MCP issue).

See the `docs/` directory for end-user quick start and detailed guides that can
ship with the package or be published separately.
