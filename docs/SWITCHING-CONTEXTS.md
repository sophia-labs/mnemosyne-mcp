# Switching Between Local Dev and Cloud Contexts

See `mnemosyne-platform/DEV-SETUP.md` → "Switching MCP Between Local Dev and Cloud" for the full guide.

## Quick Reference

Edit `~/.claude.json` → `mcpServers.mnemosyne.env`:

**Local dev:** `MNEMOSYNE_FASTAPI_URL=http://127.0.0.1:8080` + `MNEMOSYNE_DEV_USER_ID=dev-user-001` + `MNEMOSYNE_DEV_TOKEN=dev-user-001`

**Cloud:** `MNEMOSYNE_FASTAPI_URL=https://api.garden.sophia-labs.com` (no DEV vars)

Then restart Claude Code (`/exit` and reopen).

## MCP-Specific Notes

- If you've made code changes to the MCP server, reinstall with `uv tool install -e . --force` before restarting Claude Code.
- `~/.claude.json` takes precedence over `~/.claude/settings.json` for `mcpServers`. Keep the config in one file only.
- The saved OAuth token at `~/.mnemosyne/config.json` is only used when `MNEMOSYNE_DEV_TOKEN` is not set.
