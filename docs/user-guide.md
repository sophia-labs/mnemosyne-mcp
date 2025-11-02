# Neem MCP Authentication Guide

Quick guide to setting up OAuth authentication for Mnemosyne’s Claude Code integration.

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

Use the Claude CLI (preferred) or edit the settings file yourself:

```bash
claude mcp add mnemosyne-graph neem-mcp-server \
  --scope user \
  --env MNEMOSYNE_API_URL=https://api.sophia-labs.com \
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
        "MNEMOSYNE_API_URL": "https://api.sophia-labs.com",
        "LOG_LEVEL": "ERROR"
      }
    }
  }
}
```

### Step 3: Restart Claude Code

After adding the MCP server, restart Claude Code so it picks up the new configuration.

### Step 4: Test It Works

In Claude Code, ask Claude to use any Mnemosyne tool:

```
Hey Claude, can you list my knowledge graphs?
```

Claude will use the `mcp__mnemosyne-graph__list_graphs` tool and show you your graphs.

---

## Daily Use

Once set up, authentication is **completely automatic**. Just use Claude Code normally:

**Examples:**

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

Once authenticated, Claude has access to these Mnemosyne tools:

### Graph Management
- **list_graphs** – See all your knowledge graphs
- **create_graph** – Create a new graph
- **get_graph_info** – Get details about a specific graph
- **get_graph_schema** – View the structure/ontology of a graph
- **delete_graph** – Remove a graph (requires confirmation)

### Querying
- **sparql_query** – Run SPARQL queries on your graphs
- **get_system_health** – Check if the Mnemosyne API is working

### Sessions
- **create_session** – Create a new MCP session (usually automatic)

You don’t need to know the exact tool names — just ask Claude in natural language!

---

## Troubleshooting

### “Cannot connect to host localhost:8000”

**Problem:** MCP server trying to connect to the wrong API

**Solution:**
1. Check that you’re on the latest `neem` package version.
2. Restart Claude Code.
3. If it keeps failing, override the API URL:
   ```bash
   neem init --api-url https://api.sophia-labs.com
   ```

### “Authentication failed” or 401/403 errors

**Problem:** Token expired or invalid

**Solution:**
```bash
neem init --force
# Then restart Claude Code
```

### “Graph not found”

**Problem:** Trying to access a graph that doesn’t exist

**Solution:**
```bash
neem status  # Shows all configured graphs via Claude’s tooling
# Or ask Claude: "What graphs do I have?"
```

### MCP tools not showing up in Claude Code

**Problem:** Claude Code not configured or not restarted

**Solution:**
1. Run `claude mcp list` to confirm the server is registered (or inspect `~/.claude/settings.json`).
2. If it is missing, re-run the `claude mcp add …` command above.
3. Fully restart Claude Code (quit and reopen).

---

## Advanced Usage

### Custom API URL (Developers Only)

If you’re developing locally or using a different API server:

```bash
neem init --api-url http://localhost:8000
```

**Note:** Due to a Claude Code limitation, localhost URLs may be overridden to production. For local development, you might need to patch the code or use a tunnel.

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
