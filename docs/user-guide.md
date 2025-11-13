# Neem MCP Authentication Guide

Quick guide to setting up OAuth authentication for Mnemosyne’s Claude Code integration.

> **Status:** The first FastAPI-backed tool (`list_graphs`) is available and streams job updates over WebSockets. Authentication still works the same, and more tools are coming soon.

---

## First-Time Setup

### Step 1: Authenticate with Mnemosyne

Open your terminal and run:

```bash
neem init
```

This will:
1. Open your browser to the Mnemosyne login page
2. Ask you to sign in with your credentials
3. Save your authentication token securely

**What you’ll see:**
- Browser opens to `auth.sophia-labs.com`
- Sign in with your email/password
- Browser shows “Authentication successful!”
- Terminal shows confirmation with your email and token expiry time

> `neem init` only saves your token. Add the Claude MCP entry manually in the next step.

### Step 2: Add the MCP server to Claude Code

Use the Claude CLI (preferred) or edit the settings file yourself. Before running the command, make sure the FastAPI backend is reachable (for example: `kubectl port-forward svc/mnemosyne-fastapi 8001:8000`).

```bash
claude mcp add mnemosyne-graph neem-mcp-server \
  --scope user \
  --env MNEMOSYNE_FASTAPI_URL=http://127.0.0.1:8001 \
  --env LOG_LEVEL=ERROR
```

Manual alternative (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "mnemosyne-graph": {
      "type": "stdio",
      "command": "neem-mcp-server",
      "env": {
        "MNEMOSYNE_FASTAPI_URL": "http://127.0.0.1:8001",
        "LOG_LEVEL": "ERROR"
      }
    }
  }
}
```

### Step 3: Restart Claude Code

After adding the MCP server, restart Claude Code so it picks up the new configuration.

### Step 4: Test It Works

1. Start `neem-mcp-server`.
2. Ask Claude/Codex: “List my knowledge graphs.”
3. The `list_graphs` tool will submit a job, stream realtime updates from `/ws`, and return the graph metadata once the backend finishes.

---

## Daily Use

Authentication continues to be **completely automatic**. Today you can:

- “List my knowledge graphs” → calls the `list_graphs` tool.

Coming soon (once new tools land):
- “Create a new knowledge graph called 'my-research' for storing research papers”
- “Show me what’s in my user-profiles graph”
- “Run this SPARQL query on my research-papers graph: SELECT * WHERE { ?s ?p ?o } LIMIT 10”

---

## Checking Authentication Status

To see if you’re authenticated and when your token expires:

```bash
neem status
```

**Example output:**
```
✓ Authentication: Active
Token location: /Users/you/.mnemosyne/config.json
Logged in as: your.email@example.com
Token expires in: 2 hours 15 minutes
```

---

## Token Expiration

Your authentication token expires after **about 4 hours**. When it expires:

1. Claude Code MCP tools will return authentication errors.
2. Re-authenticate: `neem init --force`
3. Restart Claude Code

**Tip:** Run `neem status` before long work sessions to check if you need to refresh.

---

## Logging Out

To remove your saved authentication:

```bash
neem logout
```

This deletes your token from `~/.mnemosyne/config.json`.

---

## Available MCP Tools

- `list_graphs` – Submits a `list_graphs` job to the FastAPI backend, streams realtime updates over `/ws`, and falls back to polling if the backend doesn’t advertise push hints. More tools will be added shortly.

---

## Troubleshooting

### “Cannot connect to host 127.0.0.1:8001” (or similar)

**Problem:** The FastAPI backend is not reachable from your workstation.

**Solution:**
1. Ensure `kubectl config current-context` points to the cluster that runs the FastAPI pod.
2. Port-forward the FastAPI service (example: `kubectl port-forward svc/mnemosyne-fastapi 8001:8000`).
3. Export `MNEMOSYNE_FASTAPI_URL=http://127.0.0.1:8001` (or use the host/port env vars) before launching `neem-mcp-server`.
4. If you are running inside the cluster, set `MNEMOSYNE_FASTAPI_HOST` / `PORT` to the service DNS name and port instead.

### “Authentication failed” or 401/403 errors

**Problem:** Token expired or invalid

**Solution:**
```bash
neem init --force
# Then restart Claude Code
```

### “Where did the tools go?”

**Problem:** Claude reports that no Mnemosyne tools exist.

**Solution:** Ensure you restarted Claude/Codex after adding the MCP server. If it still can’t see the tools, check the MCP server logs for initialization errors.

### MCP server never appears in Claude Code

**Problem:** Claude Code not configured or not restarted

**Solution:**
1. Run `claude mcp list` to confirm the server is registered (or inspect `~/.claude/settings.json`).
2. If it is missing, re-run the `claude mcp add …` command above.
3. Fully restart Claude Code (quit and reopen).

---

## Advanced Usage

### Custom FastAPI URL (Developers Only)

If you’re developing against a different FastAPI instance, set the environment variables before launching the MCP server:

```bash
export MNEMOSYNE_FASTAPI_URL=http://localhost:9000
uv run neem-mcp-server
```

You can also supply `MNEMOSYNE_FASTAPI_HOST`, `MNEMOSYNE_FASTAPI_PORT`, and `MNEMOSYNE_FASTAPI_SCHEME` separately if that fits better with your kubectl workflow.

### Viewing Your Configuration

```bash
neem config
```

Shows your current MCP configuration including API URL and settings file location.

### Token Location

Your authentication token is stored at:

```
~/.mnemosyne/config.json
```

**Permissions:** `0600` (only you can read/write)
## Dev Mode (skip OAuth)

When the backend runs with `MNEMOSYNE_AUTH__MODE=dev_no_auth`, you can bypass the browser flow by exporting a user id/token before starting the MCP server:

```bash
export MNEMOSYNE_DEV_USER_ID=alice
export MNEMOSYNE_DEV_TOKEN=alice
uv run neem-mcp-server
```

The server automatically attaches `X-User-ID: alice` and `Sec-WebSocket-Protocol: Bearer.alice` to every HTTP + WebSocket request so the backend accepts them. Remove these env vars when targeting real environments.
