# Mnemosyne MCP Testing Guide

This document provides a systematic test plan for verifying the Mnemosyne MCP server functionality. Run these tests in a fresh Claude Code session after configuring the MCP server.

## Prerequisites

1. Backend running with port-forwards:
   - API: `kubectl port-forward svc/mnemosyne-api 8080:80`
   - WebSocket: `kubectl port-forward svc/mnemosyne-ws 8001:8001`

2. Claude Code config (`~/.claude/settings.json`):
```json
{
  "mcpServers": {
    "mnemosyne": {
      "type": "stdio",
      "command": "/Users/eschaton/.local/bin/neem-mcp-server",
      "env": {
        "MNEMOSYNE_FASTAPI_URL": "http://127.0.0.1:8080",
        "MNEMOSYNE_DEV_USER_ID": "dev-user-001",
        "MNEMOSYNE_DEV_TOKEN": "dev-user-001"
      }
    }
  }
}
```

3. Start a **new Claude Code session** after updating config (MCP servers only load at session start).

---

## Test Suite

### 1. Basic Connectivity

**Test: List graphs**
```
Call mcp__mnemosyne__list_graphs
```
- Expected: Returns JSON with list of graphs (may be empty)
- Verifies: Backend connectivity, auth working

---

### 2. Graph Lifecycle

**Test: Create a test graph**
```
Call mcp__mnemosyne__create_graph with:
  graph_id: "test-graph-001"
  title: "MCP Test Graph"
  description: "Created by MCP testing"
```
- Expected: Returns success with job_id
- Note: status may show "failed" even when success=true (known issue with soft-deleted graphs)

**Test: List graphs again**
```
Call mcp__mnemosyne__list_graphs
```
- Expected: Should now include "test-graph-001"

**Test: Delete the test graph**
```
Call mcp__mnemosyne__delete_graph with:
  graph_id: "test-graph-001"
```
- Expected: Returns success

---

### 3. Workspace Operations

**Test: Get workspace structure**
```
Call mcp__mnemosyne__get_workspace with:
  graph_id: "dev-docs"
```
- Expected: Returns folder/file tree structure (may be empty)

**Test: Get active context**
```
Call mcp__mnemosyne__get_active_context
```
- Expected: Returns current active graph_id and document_id from UI session (may be null if UI not open)

---

### 4. Document Operations (via Y.js/Hocuspocus)

**Test: Read a document**
```
Call mcp__mnemosyne__read_document with:
  graph_id: "dev-docs"
  document_id: "test-doc-001"
```
- Expected: Returns markdown content (or creates empty doc if doesn't exist)

**Test: Write to a document**
```
Call mcp__mnemosyne__write_document with:
  graph_id: "dev-docs"
  document_id: "test-doc-001"
  content: "# Test Document\n\nThis was written by MCP.\n\n## Section 1\n\nHello from Claude!"
```
- Expected: Returns success
- Verify: If frontend is open at localhost:5173, content should appear in real-time

**Test: Append to document**
```
Call mcp__mnemosyne__append_to_document with:
  graph_id: "dev-docs"
  document_id: "test-doc-001"
  text: "This paragraph was appended by MCP."
```
- Expected: Returns success, content added to end of document

---

### 5. SPARQL Operations

**Test: SPARQL query**
```
Call mcp__mnemosyne__sparql_query with:
  sparql: "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
```
- Expected: Returns JSON with query results (may be empty)

**Test: SPARQL update** (use with caution)
```
Call mcp__mnemosyne__sparql_update with:
  sparql: "INSERT DATA { <http://example.org/test> <http://example.org/predicate> \"test value\" }"
```
- Expected: Returns success

---

## Known Issues

1. **Graph creation after deletion**: Creating a graph with same ID as a deleted graph may return `status: "failed"` but `success: true`. The graph may exist in soft-deleted state in the RDF store.

2. **MCP session caching**: Claude Code caches MCP connections. If you update the server code, you must start a new Claude Code session to pick up changes.

3. **WebSocket port**: The MCP server expects Hocuspocus on the same port as the API (8080). If you have a separate WS port (8001), document operations may fail unless configured.

---

## Cleanup

After testing, clean up test data:

```
Call mcp__mnemosyne__delete_graph with graph_id: "test-graph-001"
```

Or clear Redis:
```bash
kubectl exec deploy/redis -- redis-cli KEYS "mnemosyne:*test*" | xargs -I {} kubectl exec deploy/redis -- redis-cli DEL "{}"
```
