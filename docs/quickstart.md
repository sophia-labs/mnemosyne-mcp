# Neem MCP – Quick Start

Get up and running with Mnemosyne’s Claude Code integration in 4 steps.

> **Status:** The first FastAPI-backed tool (`list_graphs`) is live and streams results over WebSockets. More tools are on the way.

---

## Setup (Do Once)

```bash
# 1. Authenticate (browser flow)
neem init

# 2. Wire up the FastAPI backend + register the MCP server
kubectl port-forward svc/mnemosyne-fastapi 8001:8000
claude mcp add mnemosyne-graph neem-mcp-server \
  --scope user \
  --env MNEMOSYNE_FASTAPI_URL=http://127.0.0.1:8001 \
  --env LOG_LEVEL=ERROR

# Optional dev-mode shortcut (when backend runs with MNEMOSYNE_AUTH__MODE=dev_no_auth)
# export MNEMOSYNE_DEV_USER_ID=alice
# export MNEMOSYNE_DEV_TOKEN=alice

# 3. Restart Claude Code
# (Quit and reopen the application)

# 4. Test it works
# Ask Claude: "List my knowledge graphs"
```

**That’s it!** Authentication is automatic after `neem init`; MCP clients just need the one-time registration step.

---

## Common Commands

```bash
# Check authentication status
neem status

# Re-authenticate (when token expires after 4 hours)
neem init --force
# Then restart Claude Code

# Log out
neem logout
```

---

## Using in Claude Code

- “List my knowledge graphs” → Claude calls the `list_graphs` tool, which submits a job to FastAPI and streams realtime updates over `/ws`.
- More graph/query tools are coming soon.

### Dev Mode (skip OAuth)

If your local cluster runs with `MNEMOSYNE_AUTH__MODE=dev_no_auth`, set:

```bash
export MNEMOSYNE_DEV_USER_ID=alice
export MNEMOSYNE_DEV_TOKEN=alice
```

The MCP server will send `X-User-ID: alice` and `Sec-WebSocket-Protocol: Bearer.alice` automatically, so both HTTP and WebSocket calls are accepted without running `neem init`.

---

## Troubleshooting

**Token expired?** → `neem init --force` + restart Claude Code

**Tools not working?** → The new FastAPI-backed tools haven’t shipped yet 🙂 hold tight.

**Graph not found?** → The graph tools are under reconstruction, so Claude cannot answer yet.

---

## Need More Help?

See the full guide: [`docs/user-guide.md`](user-guide.md)
