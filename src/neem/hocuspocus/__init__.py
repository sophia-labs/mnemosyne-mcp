"""Hocuspocus/Y.js client for MCP integration.

Provides WebSocket connectivity to the Mnemosyne backend's Hocuspocus endpoints
for real-time collaborative state synchronization.
"""

from neem.hocuspocus.client import HocuspocusClient
from neem.hocuspocus.document import Block, DocumentReader, DocumentWriter, TextSpan
from neem.hocuspocus.protocol import (
    ProtocolDecodeError,
    ProtocolMessage,
    ProtocolMessageType,
    decode_message,
    encode_sync_step1,
    encode_sync_step2,
    encode_sync_update,
)

__all__ = [
    "Block",
    "DocumentReader",
    "DocumentWriter",
    "HocuspocusClient",
    "ProtocolDecodeError",
    "ProtocolMessage",
    "ProtocolMessageType",
    "TextSpan",
    "decode_message",
    "encode_sync_step1",
    "encode_sync_step2",
    "encode_sync_update",
]
