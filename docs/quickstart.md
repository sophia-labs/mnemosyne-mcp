# Neem MCP – Quick Start

Get up and running with Mnemosyne’s Claude Code integration in 4 steps.

---

## Setup (Do Once)

```bash
# 1. Authenticate (browser flow)
neem init

# 2. Register the MCP server
claude mcp add mnemosyne-graph neem-mcp-server \
  --scope user \
  --env MNEMOSYNE_API_URL=https://api.sophia-labs.com \
  --env LOG_LEVEL=ERROR

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

Just ask Claude naturally:

- "Create a knowledge graph called 'my-project'"
- "Show me all my graphs"
- "Query the research-papers graph: SELECT * WHERE { ?s ?p ?o } LIMIT 10"
- "What's in my user-profiles graph?"

Claude will use the Mnemosyne MCP tools automatically!

---

## Troubleshooting

**Token expired?** → `neem init --force` + restart Claude Code

**Tools not working?** → Restart Claude Code

**Graph not found?** → Ask Claude: "What graphs do I have?"

---

## Need More Help?

See the full guide: [`docs/user-guide.md`](user-guide.md)
