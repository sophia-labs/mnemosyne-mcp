"""Hocuspocus/Y.js client for MCP integration.

Provides WebSocket connectivity to the Mnemosyne backend's Hocuspocus endpoints
for real-time collaborative state synchronization.
"""

from neem.hocuspocus.client import HocuspocusClient
from neem.hocuspocus.document import DocumentReader, DocumentWriter, extract_title_from_xml
from neem.hocuspocus.protocol import (
    ProtocolDecodeError,
    ProtocolMessage,
    ProtocolMessageType,
    decode_message,
    encode_sync_step1,
    encode_sync_step2,
    encode_sync_update,
)
from neem.hocuspocus.workspace import WorkspaceReader, WorkspaceWriter

__all__ = [
    "DocumentReader",
    "DocumentWriter",
    "HocuspocusClient",
    "ProtocolDecodeError",
    "ProtocolMessage",
    "ProtocolMessageType",
    "WorkspaceReader",
    "WorkspaceWriter",
    "decode_message",
    "encode_sync_step1",
    "encode_sync_step2",
    "encode_sync_update",
    "extract_title_from_xml",
]
