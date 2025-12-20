# MCP Server Session Summary - December 7, 2024

## What We Accomplished

### 1. Added 9 New MCP Tools (10 total)

The MCP server now exposes a full tool suite:

**Graph Management:**
- `list_graphs` - List all user's knowledge graphs
- `create_graph` - Create new graph with ID, title, description
- `delete_graph` - Permanently delete a graph

**SPARQL Operations:**
- `sparql_query` - Run SELECT/CONSTRUCT queries
- `sparql_update` - Run INSERT/DELETE/UPDATE operations

**Document Operations (via Hocuspocus/Y.js):**
- `get_active_context` - Get currently active graph/document from UI
- `get_workspace` - Get folder/file structure
- `read_document` - Read document as markdown
- `write_document` - Replace document content
- `append_to_document` - Add paragraph to document

### 2. Fixed Critical Issues

- **SPARQL query results not returned:** Fixed `_wait_for_job_result` to include `detail` field and `_extract_query_result` to unwrap `{"raw": result}` wrapper
- **uv caching issue:** Discovered `uv tool install --force` uses cached wheels; fix is `uv cache clean && uv tool install --no-cache .`

### 3. Updated Documentation

All docs now reflect the 10 available tools:
- `README.md` - Features section and Available MCP Tools
- `docs/quickstart.md` - Usage examples for all tools
- `docs/user-guide.md` - Daily use examples and tool reference

### 4. Commits Made

1. `eafcf0d` - Add graph CRUD, SPARQL, and Hocuspocus document tools to MCP server
2. `2ba3710` - Update documentation to reflect all 10 available MCP tools

---

## Architecture Overview

```
MCP Client (Claude/Codex)
        │
        ▼
┌─────────────────────────┐
│   neem-mcp-server       │  ← stdio transport
│   (FastMCP)             │
├─────────────────────────┤
│  Tools:                 │
│  - basic.py (list)      │
│  - graph_ops.py (CRUD)  │
│  - hocuspocus.py (docs) │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  FastAPI Backend        │  ← HTTP + WebSocket
│  /jobs/ endpoint        │
│  /ws gateway            │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Worker + Oxigraph      │
│  (RDF store per user)   │
└─────────────────────────┘
```

All tools use the job queue pattern:
1. Submit job via POST `/jobs/`
2. Stream results via WebSocket (or poll HTTP)
3. Return formatted response

---

## What's Next

### Entity-Level MCP Tools

The platform has a well-designed `EntityTypeHandler` protocol supporting:

| Entity | RDF Type | Y.js CRDT | Status |
|--------|----------|-----------|--------|
| Document | `doc:Document` | `Y.XmlFragment` | Supported via Hocuspocus |
| Folder | `doc:Folder` | `Y.Map` | Not yet in MCP |
| Artifact | `doc:Artifact` | `Y.Map` | Not yet in MCP |

**Potential new tools:**
- `list_entities(graph_id, entity_type)` - List folders/artifacts/documents
- `create_folder(graph_id, name, parent_id)` - Create folder
- `move_entity(graph_id, entity_id, new_parent_id)` - Reorganize
- `upload_artifact(graph_id, file)` - Upload files for processing

These would hit the `/entities/{graph_id}/{entity_type}/{entity_id}` REST API.

### Questions for Vera

1. Is the `/entities/` API stable enough to expose via MCP?
2. Should entity operations go through Hocuspocus (for collab) or REST (simpler)?
3. Any plans for custom entity types beyond Document/Folder/Artifact?
4. How should MCP handle artifact file uploads (presigned URLs)?

---

## Files Changed

```
src/neem/mcp/tools/
├── __init__.py          # Export registration functions
├── basic.py             # list_graphs tool + helpers
├── graph_ops.py         # NEW: create/delete graph, SPARQL query/update
└── hocuspocus.py        # Document tools (already existed)

docs/
├── quickstart.md        # Updated with all 10 tools
├── user-guide.md        # Updated with all 10 tools
└── dev-notes/
    └── 2024-12-07-session-summary.md  # This file

README.md                # Updated features and tool list
```

---

## Testing Notes

To test the MCP server locally:

```bash
# 1. Port-forward the backend
kubectl port-forward svc/mnemosyne-api 8001:80

# 2. Reinstall after changes (important: clear cache!)
uv cache clean && uv tool install --no-cache .

# 3. Test via Claude Code (requires new session after MCP changes)
# Ask: "List my knowledge graphs"
# Ask: "Run SPARQL query: SELECT * WHERE { ?s ?p ?o } LIMIT 5"
```
