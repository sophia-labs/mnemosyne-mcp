#!/usr/bin/env python3
"""
Minimal Hocuspocus-compatible Y.js server for MCP development.

This is a lightweight server that:
- Handles Y.js sync protocol over WebSocket
- Stores documents in memory (no Redis, no RDF)
- Serves a TipTap editor UI
- Provides the same endpoints the MCP expects

Usage:
    cd mnemosyne-mcp
    uv run python playground/server.py

Then open http://localhost:8765 in your browser to see the editor,
and point your MCP at http://localhost:8765
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pycrdt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neem.hocuspocus.protocol import (
    ProtocolDecodeError,
    ProtocolMessageType,
    decode_message,
    encode_sync_step1,
    encode_sync_step2,
    encode_sync_update,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("playground")

# Y.js sync protocol message types (for reference)
MSG_SYNC = 0
MSG_AWARENESS = 1


@dataclass
class Document:
    """In-memory Y.js document with connected clients."""
    doc: pycrdt.Doc = field(default_factory=pycrdt.Doc)
    connections: set[WebSocket] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# In-memory document store
documents: dict[str, Document] = {}


def get_or_create_document(graph_id: str, doc_id: str) -> Document:
    """Get or create a document by graph/doc ID."""
    key = f"{graph_id}:{doc_id}"
    if key not in documents:
        documents[key] = Document()
        logger.info(f"Created new document: {key}")
    return documents[key]


# FastAPI app
app = FastAPI(title="MCP Document Playground")


@app.get("/health")
async def health():
    """Health check endpoint (required by MCP)."""
    return {"status": "ok", "documents": len(documents)}


@app.websocket("/hocuspocus/docs/{graph_id}/{doc_id}")
async def document_websocket(
    websocket: WebSocket,
    graph_id: str,
    doc_id: str,
):
    """Y.js WebSocket endpoint for document collaboration."""
    await websocket.accept()

    document = get_or_create_document(graph_id, doc_id)
    document.connections.add(websocket)

    logger.info(f"Client connected to {graph_id}/{doc_id} (total: {len(document.connections)})")

    try:
        # Send our state vector as sync step 1
        state_vector = document.doc.get_state()
        await websocket.send_bytes(encode_sync_step1(state_vector))

        while True:
            data = await websocket.receive_bytes()

            if len(data) < 1:
                continue

            # Try to decode using the protocol module
            try:
                message = decode_message(data)

                if message.type == ProtocolMessageType.SYNC:
                    if message.subtype == "sync_step1":
                        # Client sent state vector, respond with our full state
                        update = document.doc.get_update()
                        await websocket.send_bytes(encode_sync_step2(update))
                        logger.info(f"Sent sync_step2 ({len(update)} bytes) to client")

                    elif message.subtype in ("sync_step2", "sync_update"):
                        # Client sent update, apply it
                        async with document.lock:
                            content_before = str(document.doc.get("content", type=pycrdt.XmlFragment))
                            document.doc.apply_update(message.payload)
                            content_after = str(document.doc.get("content", type=pycrdt.XmlFragment))
                            logger.info(f"Applied {message.subtype}: {content_before[:80]} -> {content_after[:80]}")

                        # Broadcast to other clients
                        broadcast_msg = encode_sync_update(message.payload)
                        for conn in document.connections:
                            if conn is not websocket:
                                try:
                                    await conn.send_bytes(broadcast_msg)
                                except Exception:
                                    pass

                elif message.type == ProtocolMessageType.AWARENESS:
                    # Broadcast awareness to other clients
                    for conn in document.connections:
                        if conn is not websocket:
                            try:
                                await conn.send_bytes(data)
                            except Exception:
                                pass

                elif message.type == ProtocolMessageType.PING:
                    # Respond with pong (just the message type byte)
                    await websocket.send_bytes(bytes([3]))  # PONG

            except ProtocolDecodeError:
                # Raw update - apply and broadcast
                logger.warning(f"Received raw update ({len(data)} bytes), applying directly")
                async with document.lock:
                    document.doc.apply_update(data)
                for conn in document.connections:
                    if conn is not websocket:
                        try:
                            await conn.send_bytes(data)
                        except Exception:
                            pass

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from {graph_id}/{doc_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        document.connections.discard(websocket)
        logger.info(f"Remaining connections to {graph_id}/{doc_id}: {len(document.connections)}")


@app.websocket("/hocuspocus/workspace/{graph_id}")
async def workspace_websocket(
    websocket: WebSocket,
    graph_id: str,
):
    """Y.js WebSocket endpoint for workspace (simplified)."""
    await websocket.accept()

    document = get_or_create_document(f"__workspace__{graph_id}", "__workspace__")
    document.connections.add(websocket)
    logger.info(f"Workspace client connected: {graph_id}")

    try:
        state_vector = document.doc.get_state()
        await websocket.send_bytes(encode_sync_step1(state_vector))

        while True:
            data = await websocket.receive_bytes()
            if len(data) < 1:
                continue

            try:
                message = decode_message(data)
                if message.type == ProtocolMessageType.SYNC:
                    if message.subtype == "sync_step1":
                        update = document.doc.get_update()
                        await websocket.send_bytes(encode_sync_step2(update))
                    elif message.subtype in ("sync_step2", "sync_update"):
                        async with document.lock:
                            document.doc.apply_update(message.payload)
                        broadcast_msg = encode_sync_update(message.payload)
                        for conn in document.connections:
                            if conn is not websocket:
                                try:
                                    await conn.send_bytes(broadcast_msg)
                                except Exception:
                                    pass
                elif message.type == ProtocolMessageType.AWARENESS:
                    for conn in document.connections:
                        if conn is not websocket:
                            try:
                                await conn.send_bytes(data)
                            except Exception:
                                pass
                elif message.type == ProtocolMessageType.PING:
                    await websocket.send_bytes(bytes([3]))
            except ProtocolDecodeError:
                async with document.lock:
                    document.doc.apply_update(data)

    except WebSocketDisconnect:
        pass
    finally:
        document.connections.discard(websocket)


@app.websocket("/hocuspocus/session/{user_id}")
async def session_websocket(
    websocket: WebSocket,
    user_id: str,
):
    """Y.js WebSocket endpoint for session state (simplified)."""
    await websocket.accept()

    document = get_or_create_document(f"__session__{user_id}", "__session__")
    document.connections.add(websocket)
    logger.info(f"Session client connected: {user_id}")

    try:
        state_vector = document.doc.get_state()
        await websocket.send_bytes(encode_sync_step1(state_vector))

        while True:
            data = await websocket.receive_bytes()
            if len(data) < 1:
                continue

            try:
                message = decode_message(data)
                if message.type == ProtocolMessageType.SYNC:
                    if message.subtype == "sync_step1":
                        update = document.doc.get_update()
                        await websocket.send_bytes(encode_sync_step2(update))
                    elif message.subtype in ("sync_step2", "sync_update"):
                        async with document.lock:
                            document.doc.apply_update(message.payload)
                        broadcast_msg = encode_sync_update(message.payload)
                        for conn in document.connections:
                            if conn is not websocket:
                                try:
                                    await conn.send_bytes(broadcast_msg)
                                except Exception:
                                    pass
                elif message.type == ProtocolMessageType.AWARENESS:
                    for conn in document.connections:
                        if conn is not websocket:
                            try:
                                await conn.send_bytes(data)
                            except Exception:
                                pass
                elif message.type == ProtocolMessageType.PING:
                    await websocket.send_bytes(bytes([3]))
            except ProtocolDecodeError:
                async with document.lock:
                    document.doc.apply_update(data)

    except WebSocketDisconnect:
        pass
    finally:
        document.connections.discard(websocket)


# Serve the editor UI
PLAYGROUND_DIR = Path(__file__).parent


@app.get("/")
async def index():
    """Serve the TipTap editor UI."""
    return FileResponse(PLAYGROUND_DIR / "editor.html")


@app.get("/api/documents")
async def list_documents():
    """List all documents (for debugging)."""
    result = []
    for key, doc in documents.items():
        if key.startswith("__"):
            continue
        content = str(doc.doc.get("content", type=pycrdt.XmlFragment))
        result.append({
            "key": key,
            "connections": len(doc.connections),
            "content_preview": content[:200] if content else "(empty)",
        })
    return {"documents": result}


@app.get("/api/document/{graph_id}/{doc_id}")
async def get_document_content(graph_id: str, doc_id: str):
    """Get document content as XML (for debugging)."""
    key = f"{graph_id}:{doc_id}"
    if key not in documents:
        return {"error": "Document not found"}

    doc = documents[key]
    content = str(doc.doc.get("content", type=pycrdt.XmlFragment))
    return {
        "graph_id": graph_id,
        "doc_id": doc_id,
        "content": content,
        "connections": len(doc.connections),
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MCP Document Playground")
    print("=" * 60)
    print("\nStarting server on http://localhost:8765")
    print("\nEndpoints:")
    print("  - Browser UI:     http://localhost:8765")
    print("  - Health check:   http://localhost:8765/health")
    print("  - Document list:  http://localhost:8765/api/documents")
    print("  - WebSocket:      ws://localhost:8765/hocuspocus/docs/{graph}/{doc}")
    print("\nTo test with MCP:")
    print("  export MNEMOSYNE_FASTAPI_URL=http://localhost:8765")
    print("  export MNEMOSYNE_DEV_USER_ID=test-user")
    print("  export MNEMOSYNE_DEV_TOKEN=test-token")
    print("  uv run python -c \"")
    print("    import asyncio")
    print("    from neem.hocuspocus import HocuspocusClient")
    print("    from neem.hocuspocus.document import DocumentWriter")
    print("    # ... test your edits")
    print("  \"")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
