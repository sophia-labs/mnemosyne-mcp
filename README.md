
# Mnemosyne MCP
## THIS IS A WORK IN PROGRESS AND THE DOCUMENTATION IS AI-GENERATED AND WILL BE REWRITTEN BY HUMAN BEFORE PEOPLE ARE WIDELY ENCOURAGED TO READ IT AND USE THIS CODE. THANK YOU FOR YOUR ATTENTION TO THIS MATTER XOXO VERA

**AI-powered knowledge graph integration for Claude Code, Goose & Codex**

The Mnemosyne MCP (`neem`) provides seamless integration between AI coding agents (Claude Code, Goose CLI, OpenAI Codex CLI) and your Mnemosyne knowledge graphs through the Model Context Protocol (MCP). It handles OAuth authentication, provides a standard MCP stdio server, and enables AI agents to query, create, and manage your knowledge graphs directly.

**Features:**
- 🔐 Browser-based OAuth authentication
- 🤖 9 MCP tools for graph operations (query, create, upload, etc.)
- 📁 RDF file upload support (Turtle, RDF/XML, N-Triples, JSON-LD)
- 🔍 SPARQL query execution
- 📊 Graph schema analysis
- ✨ Optimized responses for LLM consumption 

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

### Step 2: Add MCP server to your agent

#### Using Claude Code:

```bash
claude mcp add mnemosyne --scope user \
  --env MNEMOSYNE_API_URL=https://api.sophia-labs.com \
  --env LOG_LEVEL=ERROR \
  -- uv run neem-mcp-server
```

#### Using Codex
```bash
codex mcp add mnemosyne -- uv run neem-mcp-server \
  --env MNEMOSYNE_API_URL=https://api.sophia-labs.com \
  --env LOG_LEVEL=ERROR
````
### Usage Examples

Once configured, you can ask your agent to:

- **List graphs**: "Show me all my knowledge graphs"
- **Query data**: "Run a SPARQL query to find all entities of type Person in my personal-knowledge graph"
- **Upload files**: "Upload the ontology.ttl file to my research-data graph"
- **Create graphs**: "Create a new graph called project-notes with description 'Notes from my projects'"
- **Get schema info**: "What types and properties exist in my personal-knowledge graph?"

### Token Management

Tokens expire after a day. Re-run `neem init --force` whenever you need a fresh token, and restart Claude Code afterwards.

**Important**: Set `LOG_LEVEL=ERROR` for Codex CLI to avoid any stderr interference with the stdio protocol.

## Available MCP Tools

The `neem-mcp-server` provides the following tools for any MCP client (Claude Code, Goose, etc.):

### Session Management
- **`create_session`** - Initialize a new MCP session with the authenticated user

### Graph Operations
- **`list_graphs`** - List all accessible knowledge graphs with optional stats and metadata
- **`get_graph_schema`** - Analyze graph schema (classes, properties, relationships)
- **`get_graph_info`** - Get comprehensive graph information and statistics
- **`create_graph`** - Create a new knowledge graph
- **`delete_graph`** - Delete an existing graph (requires confirmation)

### Data Operations
- **`sparql_query`** - Execute SPARQL queries against graphs
  - Parameters: `graph_id`, `query`, `result_format` (json/csv/xml), `timeout_seconds`
- **`upload_file_to_graph`** - Upload RDF files (Turtle, RDF/XML, N-Triples, JSON-LD) to graphs
  - Parameters: `graph_id`, `file_path`, `rdf_format` (optional), `validation_level` (strict/lenient/none), `namespace` (optional), `replace_existing` (bool)
  - Supports format auto-detection
  - Configurable validation levels
  - Progress tracking with job IDs

### System Operations
- **`get_system_health`** - Check system health and component status

All tools support:
- ✅ Automatic authentication via saved tokens
- ✅ Session caching for performance
- ✅ Detailed error messages with helpful suggestions
- ✅ Structured JSON responses optimized for LLM consumption

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
- **Stdio transport** - Communicates with Claude Code via stdin/stdout
- **HTTP API client** - Calls the Mnemosyne Graph API for all operations
- **Session management** - Redis-backed session caching (optional)
- **Structured logging** - All logs go to stderr (stdio-safe)

Key design principles:
- **Stateless MCP server** - All state lives in the API, MCP server is just a client
- **Token-based auth** - Tokens obtained via `neem init` are used for API authentication
- **Clean separation** - MCP protocol handling separate from business logic
- **Error handling** - Helpful error messages with suggestions for LLM consumption

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
│   │   ├── standalone_server.py    # Main MCP server with API client
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

- `MNEMOSYNE_API_URL` - API endpoint (default: `https://api.sophia-labs.com`)
- `MNEMOSYNE_CONFIG_DIR` - Token storage location (default: `~/.mnemosyne`)
- `CLAUDE_CODE_SETTINGS_PATH` - Claude settings file (default: `~/.claude/settings.json`)
- `LOG_LEVEL` - Logging verbosity (default: `INFO`)
  - `DEBUG` - Verbose logging for troubleshooting
  - `INFO` - Normal operational logging (default)
  - `WARNING` - Quiet mode, only warnings and errors
  - `ERROR` - Silent mode, only errors (recommended for Codex CLI)
  - `CRITICAL` - Minimal logging, critical errors only

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
5. **Check environment**: Ensure `MNEMOSYNE_API_URL` is set correctly

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
