# Neem MCP – Quick Start

Get up and running with Mnemosyne’s Claude Code integration in 4 steps.

> **Status:** 23+ MCP tools are available for graph management, SPARQL queries, real-time document editing, and workspace organization.

---

## Setup (Do Once)

```bash
# 1. Authenticate (browser flow)
neem init

# 2. Wire up the FastAPI backend + register the MCP server
kubectl port-forward svc/mnemosyne-api 8080:80
claude mcp add mnemosyne --scope user \
  -- uv run neem-mcp-server

# Optional dev-mode shortcut (when backend runs with MNEMOSYNE_AUTH__MODE=dev_no_auth)
# claude mcp add mnemosyne --scope user \
#   --env MNEMOSYNE_DEV_USER_ID=alice \
#   --env MNEMOSYNE_DEV_TOKEN=alice \
#   -- uv run neem-mcp-server

# 3. Restart Claude Code
# (Quit and reopen the application)

# 4. Test it works
# Ask Claude: "List my knowledge graphs"
```

> The MCP server defaults to `http://127.0.0.1:8080` which matches `kubectl port-forward svc/mnemosyne-api 8080:80`. Override with `--env MNEMOSYNE_FASTAPI_URL=http://127.0.0.1:XXXX` if your port-forward differs.

**That’s it!** Authentication is automatic after `neem init`; MCP clients just need the one-time registration step.

---

## Common Commands

```bash
# Check authentication status
neem status

# Force re-authentication (tokens auto-refresh, but if needed)
neem init --force

# Log out
neem logout
```

> **Note:** Tokens automatically refresh in the background. You'll stay logged in for ~30 days without manual intervention.

---

## Using in Claude Code

**Graph Management:**
- "List my knowledge graphs" → `list_graphs`
- "Create a new graph called 'research' with title 'Research Notes'" → `create_graph`
- "Delete the test-graph" → `delete_graph` (soft delete by default, `hard=true` for permanent)

**SPARQL Operations:**
- "Run this SPARQL query: SELECT * WHERE { ?s ?p ?o } LIMIT 10" → `sparql_query`
- "Insert this triple into my graph..." → `sparql_update`

> **SPARQL Tip:** Always use `PREFIX doc: <http://mnemosyne.dev/doc#>` — never `urn:mnemosyne:schema:doc:`.

**Document Operations (real-time via Y.js):**
- "What document am I looking at in Mnemosyne?" → `get_active_context`
- "Show me the folder structure of my-graph" → `get_workspace`
- "Read the document at /notes/meeting.md" → `read_document`
- "Write this content to my document" → `write_document`
- "Add a paragraph to the end of my document" → `append_to_document`

### Dev Mode (skip OAuth)

If your local cluster runs with `MNEMOSYNE_AUTH__MODE=dev_no_auth`, set:

```bash
export MNEMOSYNE_DEV_USER_ID=alice
export MNEMOSYNE_DEV_TOKEN=alice
```

The MCP server will send `X-User-ID: alice` and `Sec-WebSocket-Protocol: Bearer.alice` automatically, so both HTTP and WebSocket calls are accepted without running `neem init`.

---

## Troubleshooting

**Auth errors?** → Tokens auto-refresh, but if your refresh token expired (~30 days), run `neem init`

**Tools not showing?** → Make sure you reinstalled after updates: `uv cache clean && uv tool install --no-cache .`

**Connection refused?** → Ensure the FastAPI backend is running and port-forwarded: `kubectl port-forward svc/mnemosyne-api 8080:80`

**SPARQL returns empty results?** → Check your namespace prefix. Use `PREFIX doc: <http://mnemosyne.dev/doc#>`, not `urn:mnemosyne:schema:doc:`

---

## Need More Help?

See the full guide: [`docs/user-guide.md`](user-guide.md)
