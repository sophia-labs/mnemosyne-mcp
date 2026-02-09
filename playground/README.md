# MCP Document Playground

A self-contained local development environment for rapid iteration on MCP document editing tools. Test the full Hocuspocus/Y.js CRDT stack without rebuilding or redeploying the cluster.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [The Playground Server](#the-playground-server)
- [The Browser Editor](#the-browser-editor)
- [Scripting MCP Operations](#scripting-mcp-operations)
- [Testing Workflows](#testing-workflows)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
cd /home/vera/dev/sophia/mnemosyne-mcp

# Terminal 1: Start the playground server
uv run python playground/server.py

# Terminal 2: Run the demo script
uv run python playground/mcp_script.py --demo

# Browser: Open http://localhost:8765 to see real-time changes
```

That's it. You now have a full document editing stack running locally.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR DEVELOPMENT LOOP                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │   Edit      │     │   Run       │     │   See       │          │
│   │   Code      │ ──▶ │   Script    │ ──▶ │   Result    │          │
│   │             │     │             │     │   in UI     │          │
│   └─────────────┘     └─────────────┘     └─────────────┘          │
│         │                   │                   ▲                   │
│         │                   │                   │                   │
│         ▼                   ▼                   │                   │
│   ┌─────────────────────────────────────────────────────┐          │
│   │              MCP Scripting Layer                     │          │
│   │         (playground/mcp_script.py)                   │          │
│   │                                                      │          │
│   │   MCP class → HocuspocusClient → DocumentWriter     │          │
│   └─────────────────────────────────────────────────────┘          │
│                            │                                        │
│                            │ WebSocket (Y.js sync protocol)         │
│                            ▼                                        │
│   ┌─────────────────────────────────────────────────────┐          │
│   │              Playground Server                       │          │
│   │         (playground/server.py)                       │          │
│   │                                                      │          │
│   │   FastAPI + In-Memory Y.Doc Store                   │          │
│   │   Port 8765                                          │          │
│   └─────────────────────────────────────────────────────┘          │
│                            │                                        │
│                            │ WebSocket (Y.js sync protocol)         │
│                            ▼                                        │
│   ┌─────────────────────────────────────────────────────┐          │
│   │              Browser Editor                          │          │
│   │         (playground/editor.html)                     │          │
│   │                                                      │          │
│   │   TipTap + Y.js + y-websocket                       │          │
│   │   http://localhost:8765                              │          │
│   └─────────────────────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### What This Replaces

| Production Stack | Playground Equivalent |
|------------------|----------------------|
| Kubernetes cluster | Single Python process |
| Redis | In-memory dict |
| PostgreSQL/RDF store | In-memory Y.Doc |
| Hocuspocus gateway | Minimal WebSocket handler |
| Full FastAPI app | ~300 lines of Python |

### What's Identical

- **Y.js protocol**: Same binary sync protocol
- **pycrdt operations**: Same CRDT library
- **DocumentReader/Writer**: Same code paths
- **HocuspocusClient**: Same client code
- **TipTap rendering**: Same editor behavior

---

## The Playground Server

### Starting the Server

```bash
cd /home/vera/dev/sophia/mnemosyne-mcp
uv run python playground/server.py
```

Output:
```
==============================================================
MCP Document Playground
==============================================================

Starting server on http://localhost:8765

Endpoints:
  - Browser UI:     http://localhost:8765
  - Health check:   http://localhost:8765/health
  - Document list:  http://localhost:8765/api/documents
  - WebSocket:      ws://localhost:8765/hocuspocus/docs/{graph}/{doc}

To test with MCP:
  export MNEMOSYNE_FASTAPI_URL=http://localhost:8765
  ...
==============================================================
```

### Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | TipTap editor UI |
| `/health` | GET | Health check (MCP probes this) |
| `/api/documents` | GET | List all documents in memory |
| `/api/document/{graph}/{doc}` | GET | Get document XML content |
| `/hocuspocus/docs/{graph}/{doc}` | WS | Y.js document sync |
| `/hocuspocus/workspace/{user}/{graph}` | WS | Y.js workspace sync |
| `/hocuspocus/session/{user}` | WS | Y.js session sync |

### How It Works

1. **Document Storage**: Documents are stored in a Python dict as `pycrdt.Doc` objects
2. **WebSocket Protocol**: Implements the Y.js sync protocol (sync_step1, sync_step2, sync_update)
3. **Multi-client Sync**: Broadcasts updates to all connected clients
4. **No Persistence**: Everything is in memory; restart clears all documents

---

## The Browser Editor

Open http://localhost:8765 after starting the server.

### Features

- **Full TipTap Editor**: Bold, italic, headings, lists, code blocks, etc.
- **Real-time Sync**: Changes from scripts appear instantly
- **Debug Panel**: Shows raw Y.js XML content (click to expand)
- **Document Selector**: Switch between graph/doc combinations

### Using the Editor

1. **Edit directly**: Type in the editor, formatting persists
2. **Watch script changes**: Run scripts in another terminal, see changes appear
3. **Debug XML**: Expand the debug panel to see the raw document structure
4. **Switch documents**: Change graph/doc IDs and click "Connect"

---

## Scripting MCP Operations

The `mcp_script.py` module provides a Pythonic interface to all MCP document operations.

### Basic Usage

```python
from playground.mcp_script import MCP
import asyncio

async def main():
    # Create client pointing to playground
    mcp = MCP("http://localhost:8765")

    # Connect to a document (sets as current context)
    await mcp.connect("test-graph", "test-doc")

    # Now all operations use this document by default
    content = await mcp.read_document()
    await mcp.append("Hello from script!")

    await mcp.close()

asyncio.run(main())
```

### Document Operations

```python
# Read document content as TipTap XML
content = await mcp.read_document()
content = await mcp.read_document("other-graph", "other-doc")

# Write document (REPLACES all content)
await mcp.write_document("""
<heading level="1">My Document</heading>
<paragraph>First paragraph.</paragraph>
<listItem listType="bullet">Item one</listItem>
<listItem listType="bullet">Item two</listItem>
""")

# Append a block to the end
block_id = await mcp.append("<paragraph>New paragraph</paragraph>")
block_id = await mcp.append("Plain text is auto-wrapped in paragraph")

# Clear all content
await mcp.clear()
```

### Block Queries

```python
# Get all blocks
blocks = await mcp.blocks()

# Get specific block types
paragraphs = await mcp.paragraphs()
headings = await mcp.headings()

# Get task items
all_tasks = await mcp.tasks()
incomplete = await mcp.tasks(completed=False)
done = await mcp.tasks(completed=True)

# Search by text
matches = await mcp.search("keyword")

# Advanced query
blocks = await mcp.query_blocks(
    block_type="listItem",      # paragraph, heading, listItem, codeBlock, etc.
    indent=1,                   # exact indent level
    indent_gte=1,               # indent >= 1
    indent_lte=3,               # indent <= 3
    list_type="task",           # bullet, ordered, task
    checked=False,              # for task items
    text_contains="TODO",       # text search
    limit=50,                   # max results
)
```

### Block Mutations

```python
# Get block info
info = await mcp.get_block("block-id")
print(info.type, info.text, info.attributes)

# Insert blocks
new_id = await mcp.insert_block_after("ref-id", "<paragraph>After</paragraph>")
new_id = await mcp.insert_block_before("ref-id", "<paragraph>Before</paragraph>")

# Update block content
await mcp.update_block("block-id", content="<paragraph>New content</paragraph>")

# Update block attributes
await mcp.update_block("block-id", indent=1)
await mcp.update_block("block-id", checked=True)
await mcp.update_block("block-id", list_type="task")
await mcp.update_block("block-id", indent=2, checked=False)

# Delete blocks
deleted_ids = await mcp.delete_block("block-id")
deleted_ids = await mcp.delete_block("block-id", cascade=True)  # delete children too
```

### Working with Multiple Documents

```python
# Option 1: Specify graph/doc each time
content1 = await mcp.read_document("graph1", "doc1")
content2 = await mcp.read_document("graph2", "doc2")

# Option 2: Switch context
await mcp.connect("graph1", "doc1")
await mcp.append("To doc1")

await mcp.connect("graph2", "doc2")
await mcp.append("To doc2")
```

### Quick Functions (No Class)

For simple scripts:

```python
from playground.mcp_script import read, write, append
import asyncio

async def quick():
    # These create a shared MCP instance automatically
    content = await read("test-graph", "test-doc")
    await append("test-graph", "test-doc", "Quick append!")

asyncio.run(quick())
```

### CLI Interface

```bash
# Run the built-in demo
uv run python playground/mcp_script.py --demo

# Read a document
uv run python playground/mcp_script.py --read test-graph test-doc

# Append text to a document
uv run python playground/mcp_script.py --append test-graph test-doc "Hello world!"
```

---

## Testing Workflows

### Workflow 1: Rapid Script Iteration

Best for: Testing document manipulation logic

```bash
# Terminal 1: Server
uv run python playground/server.py

# Terminal 2: Your script
uv run python my_test_script.py

# Browser: http://localhost:8765 (optional, for visual feedback)
```

### Workflow 2: Interactive Exploration

Best for: Exploring document structure, debugging

```bash
# Terminal 1: Server
uv run python playground/server.py

# Terminal 2: Interactive mode
uv run python playground/test_mcp.py --interactive
```

Commands in interactive mode:
- `read` - Show document content
- `append` - Add a paragraph (prompts for text)
- `blocks` - List all blocks
- `clear` - Clear document
- `quit` - Exit

### Workflow 3: Unit Testing Document Logic

Best for: Testing DocumentReader/DocumentWriter without network

```python
# No server needed!
import pycrdt
from neem.hocuspocus.document import DocumentReader, DocumentWriter

def test_append_block():
    doc = pycrdt.Doc()
    writer = DocumentWriter(doc)
    reader = DocumentReader(doc)

    writer.append_block("<paragraph>Test</paragraph>")

    assert reader.get_block_count() == 1
    blocks = reader.query_blocks(block_type="paragraph")
    assert len(blocks) == 1
    assert "Test" in blocks[0]["text_preview"]
```

Run with:
```bash
uv run pytest tests/test_block_operations.py -v
```

### Workflow 4: Testing with Claude Code

Best for: End-to-end MCP tool testing

```bash
# Terminal 1: Playground server
uv run python playground/server.py

# Register MCP server pointing to playground
claude mcp add mnemosyne-playground --scope user \
  --env MNEMOSYNE_FASTAPI_URL=http://localhost:8765 \
  --env MNEMOSYNE_DEV_USER_ID=test-user \
  --env MNEMOSYNE_DEV_TOKEN=test-token \
  -- uv run neem-mcp-server

# Restart Claude Code, then test with prompts like:
# "Read the document test-graph/test-doc"
# "Append a paragraph saying Hello World"
# "Find all incomplete tasks"
```

---

## API Reference

### MCP Class

```python
class MCP:
    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        user_id: str = "test-user",
        token: str = "test-token",
    ): ...

    # Connection
    async def connect(self, graph_id: str, doc_id: str) -> None
    async def close(self) -> None

    # Document operations
    async def read_document(self, graph_id=None, doc_id=None) -> str
    async def write_document(self, content: str, graph_id=None, doc_id=None) -> None
    async def append(self, content: str, graph_id=None, doc_id=None) -> str
    async def clear(self, graph_id=None, doc_id=None) -> None

    # Block queries
    async def get_block(self, block_id: str, ...) -> Optional[BlockInfo]
    async def query_blocks(self, block_type=None, indent=None, ...) -> List[Dict]
    async def blocks(self, ...) -> List[Dict]
    async def paragraphs(self, ...) -> List[Dict]
    async def headings(self, ...) -> List[Dict]
    async def tasks(self, completed=None, ...) -> List[Dict]
    async def search(self, text: str, ...) -> List[Dict]

    # Block mutations
    async def insert_block_after(self, block_id: str, content: str, ...) -> str
    async def insert_block_before(self, block_id: str, content: str, ...) -> str
    async def update_block(self, block_id: str, content=None, indent=None, ...) -> None
    async def delete_block(self, block_id: str, cascade=False, ...) -> List[str]
```

### BlockInfo Dataclass

```python
@dataclass
class BlockInfo:
    block_id: str           # Unique block identifier
    index: int              # Position in document
    type: str               # paragraph, heading, listItem, etc.
    text: str               # Plain text content
    xml: str                # Full XML representation
    attributes: Dict        # All block attributes
    prev_id: Optional[str]  # Previous block's ID
    next_id: Optional[str]  # Next block's ID
```

### TipTap XML Reference

**Block Types:**
```xml
<paragraph>Text content</paragraph>
<heading level="1">Heading 1</heading>
<heading level="2">Heading 2</heading>
<heading level="3">Heading 3</heading>
<listItem listType="bullet">Bullet item</listItem>
<listItem listType="ordered">Numbered item</listItem>
<listItem listType="task">Task item</listItem>
<listItem listType="task" checked="true">Done task</listItem>
<blockquote><paragraph>Quoted text</paragraph></blockquote>
<codeBlock language="python">code here</codeBlock>
<horizontalRule/>
```

**Inline Marks:**
```xml
<paragraph>
  <strong>Bold</strong>
  <em>Italic</em>
  <strike>Strikethrough</strike>
  <code>Inline code</code>
  <mark>Highlighted</mark>
  <a href="https://...">Link</a>
</paragraph>
```

**Block Attributes:**
```xml
<listItem
  listType="task"           <!-- bullet, ordered, task -->
  checked="true"            <!-- for tasks -->
  indent="1"                <!-- nesting level 0-5 -->
  collapsed="true"          <!-- for collapsible blocks -->
  data-block-id="abc123"    <!-- unique ID, auto-generated -->
>
```

---

## Troubleshooting

### Server won't start

```bash
# Check dependencies
uv sync

# Check port availability
lsof -i :8765
```

### Browser shows "Connecting..."

- Verify server is running on port 8765
- Check browser console for WebSocket errors
- Try refreshing the page

### Script can't connect

```python
# Verify the URL is correct
mcp = MCP("http://localhost:8765")  # Not https, not 127.0.0.1

# Check server health
curl http://localhost:8765/health
```

### Changes don't appear in browser

- The browser and script must use the same graph_id/doc_id
- Try clicking "Refresh Debug" in the browser
- Check the debug panel for the raw XML

### Import errors

```bash
# Make sure you're in the mnemosyne-mcp directory
cd /home/vera/dev/sophia/mnemosyne-mcp

# Sync dependencies
uv sync
```

---

## Files Reference

```
playground/
├── server.py       # Minimal Y.js WebSocket server (FastAPI)
├── editor.html     # TipTap browser editor with live sync
├── mcp_script.py   # Python scripting interface for MCP tools
├── test_mcp.py     # Test script with basic/block/interactive modes
├── start.sh        # Convenience launcher script
└── README.md       # This documentation
```

---

## Example: Complete Test Script

```python
#!/usr/bin/env python3
"""Example: Testing document editing workflow."""

import asyncio
from playground.mcp_script import MCP

async def test_document_workflow():
    mcp = MCP()

    try:
        # Setup
        await mcp.connect("test-graph", "test-doc")
        await mcp.clear()

        # Create structured document
        await mcp.write_document("""
<heading level="1">Project Tasks</heading>
<paragraph>Track your progress here.</paragraph>
<listItem listType="task">Write documentation</listItem>
<listItem listType="task">Add tests</listItem>
<listItem listType="task">Review code</listItem>
""")

        # Find incomplete tasks
        tasks = await mcp.tasks(completed=False)
        print(f"Found {len(tasks)} incomplete tasks")

        # Complete the first task
        if tasks:
            await mcp.update_block(tasks[0]["block_id"], checked=True)
            print(f"Completed: {tasks[0]['text_preview']}")

        # Add a new task
        await mcp.append('<listItem listType="task">Deploy to production</listItem>')

        # Show final state
        print("\nFinal document:")
        print(await mcp.read_document())

    finally:
        await mcp.close()

if __name__ == "__main__":
    asyncio.run(test_document_workflow())
```

Run with:
```bash
uv run python my_workflow_test.py
```
