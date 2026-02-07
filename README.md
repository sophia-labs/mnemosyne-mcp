
# Mnemosyne MCP
## THIS IS A WORK IN PROGRESS AND THE DOCUMENTATION IS AI-GENERATED AND WILL BE REWRITTEN BY HUMAN BEFORE PEOPLE ARE WIDELY ENCOURAGED TO READ IT AND USE THIS CODE. THANK YOU FOR YOUR ATTENTION TO THIS MATTER XOXO VERA

**AI-powered knowledge graph integration for Claude Code, Goose & Codex**

The Mnemosyne MCP (`neem`) historically exposed a full suite of graph management tools. We are currently rebuilding those tools from scratch against a new FastAPI backend that runs inside our local kubectl context.

> **Status:** The MCP server provides 23+ tools for knowledge graph management, SPARQL queries, real-time document editing, and workspace organization via Hocuspocus/Y.js.

**Features:**
- 🔌 Reliable connectivity to a local FastAPI backend (via env vars or kubectl port-forward)
- 🩺 Automatic startup health probe so you know whether the backend is reachable
- 🔐 Browser-based OAuth authentication (`neem init`)
- 📊 Full graph CRUD operations (create, list, delete)
- 🔍 SPARQL query and update support
- 📝 Real-time document editing via Y.js CRDT

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

Before registering the MCP server, expose the FastAPI backend from your kubectl context:

```bash
kubectl port-forward svc/mnemosyne-api 8080:80
```

The MCP server defaults to `http://127.0.0.1:8080` which matches this port-forward. Both HTTP and WebSocket connections go through port 8080.

### Step 2: Add MCP server to your agent

#### Using Claude Code (local backend):

```bash
claude mcp add mnemosyne --scope user \
  -- uv run neem-mcp-server
```

#### Using Claude Code (hosted API):

When connecting to the hosted API (e.g. `api.garden.sophia-labs.com`) rather than a local port-forward, disable WebSocket streaming. The WS endpoint is internal to the cluster and not exposed externally — without this flag, job-based tools like `list_graphs` will hang for up to 60 seconds before falling back to HTTP polling.

```bash
claude mcp add mnemosyne --scope user \
  --env MNEMOSYNE_FASTAPI_URL=https://api.garden.sophia-labs.com \
  -- uv run neem-mcp-server
```

Or add directly to `~/.claude.json`:

```json
"mcpServers": {
  "mnemosyne": {
    "type": "stdio",
    "command": "uv",
    "args": ["run", "neem-mcp-server"],
    "env": {
      "MNEMOSYNE_FASTAPI_URL": "https://api.garden.sophia-labs.com",
    }
  }
}
```

Authenticate first with `neem init`. Tokens are stored at `~/.mnemosyne/config.json` and auto-refresh for ~30 days.

#### Using Codex
```bash
codex mcp add mnemosyne \
  --env LOG_LEVEL=ERROR \
  -- uv run neem-mcp-server
```

> **Dev-mode shortcut:** Append `--env MNEMOSYNE_DEV_TOKEN=<user>` and `--env MNEMOSYNE_DEV_USER_ID=<user>` when the backend runs with `MNEMOSYNE_AUTH__MODE=dev_no_auth`. Both HTTP and WebSocket will impersonate that user without OAuth.
>
> **Custom port?** Add `--env MNEMOSYNE_FASTAPI_URL=http://127.0.0.1:XXXX` if your port-forward differs from the default 8080.

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

Tokens are automatically refreshed in the background using OAuth refresh tokens. After initial authentication, you'll stay logged in for approximately 30 days without any manual intervention.

When the refresh token eventually expires, simply run `neem init` to re-authenticate.

**Important**: Set `LOG_LEVEL=ERROR` for Codex CLI to avoid any stderr interference with the stdio protocol.

## Available MCP Tools

### Graph Management
- `list_graphs` – List all knowledge graphs owned by the authenticated user (use `include_deleted=true` to show soft-deleted graphs)
- `create_graph` – Create a new knowledge graph with ID, title, and optional description
- `delete_graph` – Delete a graph (soft delete by default, use `hard=true` to permanently delete)

### SPARQL Operations
- `sparql_query` – Execute read-only SPARQL SELECT/CONSTRUCT queries against your graphs
- `sparql_update` – Execute SPARQL INSERT/DELETE/UPDATE operations to modify graph data

> **Namespace Reference:** When writing SPARQL queries, use these exact prefixes:
> ```sparql
> PREFIX doc: <http://mnemosyne.dev/doc#>
> PREFIX dcterms: <http://purl.org/dc/terms/>
> PREFIX nfo: <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>
> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
> PREFIX nie: <http://www.semanticdesktop.org/ontologies/2007/01/19/nie#>
> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
> ```
> **WARNING:** Do NOT use `urn:mnemosyne:schema:doc:` as the doc namespace — it will match nothing.

### Orientation
- `get_user_location` – Get the graph and document the user is currently viewing (minimal tokens)
- `get_workspace` – Get the folder/file structure of a graph (primary exploration tool)
- `get_session_state` – Get full session state including tabs and preferences (large payload, rarely needed)

### Folder Operations
- `create_folder` – Create a new folder in the workspace
- `rename_folder` – Rename a folder
- `move_folder` – Move a folder to a different parent
- `delete_folder` – Delete a folder (with optional cascade to delete contents)

### Document Operations (via Hocuspocus/Y.js)
- `read_document` – Read document content as TipTap XML
- `write_document` – Replace document content with TipTap XML
- `append_to_document` – Add a block to the end of a document
- `move_document` – Move a document to a different folder
- `delete_document` – Remove a document from workspace navigation

### Block-Level Operations
- `get_block` – Read a specific block by its ID (includes text_length and formatting runs)
- `query_blocks` – Search for blocks matching specific criteria
- `update_block` – Update a block's attributes or replace entire content
- `edit_block_text` – Insert/delete text at character offsets within a block (CRDT-safe collaborative editing)
- `insert_block` – Insert a new block relative to an existing block
- `delete_block` – Delete a block (with optional cascade for children)
- `batch_update_blocks` – Update multiple blocks in a single transaction

### Artifact Operations
- `move_artifact` – Move an artifact to a different folder
- `rename_artifact` – Rename an artifact

### Wire Operations (Semantic Connections)
- `list_wire_predicates` – List available semantic predicates organized by category
- `create_wire` – Create a semantic connection between documents or blocks (syncs via Y.js CRDT)
- `get_wires` – Get all wires connected to a document (filter by direction: outgoing/incoming/both)
- `traverse_wires` – BFS traversal of the wire graph from a starting document (up to depth 10)

#### TipTap XML Format

Documents use TipTap's XML representation with full formatting support:

**Blocks:** `paragraph`, `heading` (level="1-3"), `bulletList`, `orderedList`, `blockquote`, `codeBlock` (language="..."), `taskList`, `taskItem` (checked="true"), `horizontalRule`

**Marks (nestable):** `strong`, `em`, `strike`, `code`, `mark` (highlight), `a` (href="...")

**Annotation Marks:** Special inline marks that reference external content:
- `footnote` – Self-contained annotation with `data-footnote-content` attribute
- `commentMark` – Reference annotation with `data-comment-id` attribute

Example:
```xml
<paragraph>Text with <mark>highlight</mark> and a note<footnote data-footnote-content="This is a footnote"/></paragraph>
```

Graph management and SPARQL tools submit jobs to the FastAPI backend, streaming realtime updates via WebSocket when available and falling back to HTTP polling otherwise. Document, block, folder, and wire operations use Y.js CRDT via Hocuspocus for real-time sync.

#### SPARQL Data Model

Documents, folders, and artifacts are materialized to RDF by two backend pipelines:
- **Workspace materializer** – Syncs metadata (titles, folder structure, order) from the workspace Y.Doc
- **Document materializer** – Syncs content (blocks, paragraphs, text nodes) from document Y.Docs

**Common RDF types:** `doc:TipTapDocument`, `doc:Folder`, `doc:Artifact`, `doc:XmlFragment`, `doc:Paragraph`, `doc:Heading`, `doc:TextNode`

**Common predicates:** `dcterms:title`, `nfo:fileName`, `nfo:belongsToContainer`, `doc:order`, `doc:section`, `doc:content`, `doc:childNode`, `doc:siblingOrder`, `doc:createdAt`, `doc:updatedAt`

**Entity URI pattern:** `urn:mnemosyne:user:{user_id}:graph:{graph_id}:{type}:{entity_id}`
Content fragments use `#` suffixes: `...doc:{id}#frag`, `...doc:{id}#block-{block_id}`, `...doc:{id}#node-{n}`

Example query to list all documents with titles:
```sparql
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX dcterms: <http://purl.org/dc/terms/>

SELECT ?doc ?title WHERE {
  ?doc a doc:TipTapDocument .
  ?doc dcterms:title ?title .
} ORDER BY ?title
```

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
├── cli.py                          # CLI commands (init, status, logout, config)
├── hocuspocus/                     # Y.js CRDT client layer
│   ├── client.py                   # WebSocket client for Hocuspocus
│   ├── document.py                 # Document Y.Doc operations
│   ├── protocol.py                 # Y.js sync protocol handler
│   └── workspace.py                # Workspace Y.Doc operations
├── mcp/
│   ├── server/
│   │   ├── standalone_server.py    # Backend resolver, health probe, tool registration
│   │   └── standalone_server_stdio.py  # Stdio transport wrapper
│   ├── tools/
│   │   ├── basic.py                # list_graphs + job helpers
│   │   ├── graph_ops.py            # create/delete graph, SPARQL query/update
│   │   ├── hocuspocus.py           # Document, block, folder, artifact operations
│   │   └── wire_tools.py           # Semantic connection tools
│   ├── jobs/                       # Job streaming (WebSocket + polling)
│   ├── session.py                  # Session management
│   ├── auth.py                     # MCP auth context
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

1. **Token not refreshing**: Tokens auto-refresh in the background. If you see auth errors, your refresh token may have expired (~30 days). Run `neem init` to re-authenticate.
2. **Check token**: `neem status` to see token details and expiry
3. **Force refresh**: `neem init --force` to get completely fresh tokens

### SPARQL Returns Empty Results

The most common cause is using the wrong namespace prefix. Use:
```sparql
PREFIX doc: <http://mnemosyne.dev/doc#>
```
Do NOT use `urn:mnemosyne:schema:doc:` — it will silently match nothing.

See the `docs/` directory for end-user quick start and detailed guides that can
ship with the package or be published separately.
