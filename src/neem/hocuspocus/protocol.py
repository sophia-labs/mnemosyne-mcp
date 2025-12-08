"""Y.js WebSocket protocol encoding/decoding.

This module implements the y-protocols/sync framing using lib0-style varuint encoding.
It mirrors the backend's app/hocuspocus/protocol.py for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProtocolDecodeError(Exception):
    """Raised when a WebSocket frame cannot be parsed as a Y.js protocol message."""


class ProtocolMessageType(str, Enum):
    SYNC = "sync"
    AWARENESS = "awareness"
    PING = "ping"
    PONG = "pong"


@dataclass
class ProtocolMessage:
    type: ProtocolMessageType
    payload: bytes
    subtype: str | None = None


# Y.js protocol numeric codes (lib0 varuint encoded)
_MESSAGE_SYNC = 0
_MESSAGE_AWARENESS = 1
_MESSAGE_PING = 2
_MESSAGE_PONG = 3

# Sync subtypes
_SYNC_STEP1 = 0
_SYNC_STEP2 = 1
_SYNC_UPDATE = 2


def encode_sync_step1(state_vector: bytes) -> bytes:
    """Encode a sync step 1 message carrying the client state vector."""
    buffer: bytearray = bytearray()
    _write_var_uint(buffer, _MESSAGE_SYNC)
    _write_var_uint(buffer, _SYNC_STEP1)
    _write_var_uint8_array(buffer, state_vector)
    return bytes(buffer)


def encode_sync_step2(update: bytes) -> bytes:
    """Encode a sync step 2 message carrying a Y.js update."""
    buffer: bytearray = bytearray()
    _write_var_uint(buffer, _MESSAGE_SYNC)
    _write_var_uint(buffer, _SYNC_STEP2)
    _write_var_uint8_array(buffer, update)
    return bytes(buffer)


def encode_sync_update(update: bytes) -> bytes:
    """Encode a sync update message carrying a Y.js update."""
    buffer: bytearray = bytearray()
    _write_var_uint(buffer, _MESSAGE_SYNC)
    _write_var_uint(buffer, _SYNC_UPDATE)
    _write_var_uint8_array(buffer, update)
    return bytes(buffer)


def encode_ping() -> bytes:
    """Encode a ping frame."""
    buffer: bytearray = bytearray()
    _write_var_uint(buffer, _MESSAGE_PING)
    return bytes(buffer)


def decode_message(frame: bytes) -> ProtocolMessage:
    """Decode a protocol frame or raise ProtocolDecodeError."""
    try:
        mv = memoryview(frame)
        idx = 0

        msg_type, idx = _read_var_uint(mv, idx)
        if msg_type == _MESSAGE_SYNC:
            subtype, idx = _read_var_uint(mv, idx)
            if subtype == _SYNC_STEP1:
                payload, idx = _read_var_uint8_array(mv, idx)
                _ensure_consumed(idx, mv)
                return ProtocolMessage(ProtocolMessageType.SYNC, payload, subtype="sync_step1")
            if subtype == _SYNC_STEP2:
                payload, idx = _read_var_uint8_array(mv, idx)
                _ensure_consumed(idx, mv)
                return ProtocolMessage(ProtocolMessageType.SYNC, payload, subtype="sync_step2")
            if subtype == _SYNC_UPDATE:
                payload, idx = _read_var_uint8_array(mv, idx)
                _ensure_consumed(idx, mv)
                return ProtocolMessage(ProtocolMessageType.SYNC, payload, subtype="sync_update")
            raise ProtocolDecodeError(f"unknown sync subtype {subtype}")

        if msg_type == _MESSAGE_AWARENESS:
            payload, idx = _read_var_uint8_array(mv, idx)
            _ensure_consumed(idx, mv)
            return ProtocolMessage(ProtocolMessageType.AWARENESS, payload)

        if msg_type == _MESSAGE_PING:
            _ensure_consumed(idx, mv)
            return ProtocolMessage(ProtocolMessageType.PING, b"")

        if msg_type == _MESSAGE_PONG:
            _ensure_consumed(idx, mv)
            return ProtocolMessage(ProtocolMessageType.PONG, b"")

        raise ProtocolDecodeError(f"unknown message type {msg_type}")
    except ProtocolDecodeError:
        raise
    except Exception as exc:
        raise ProtocolDecodeError(str(exc)) from exc


def _read_var_uint(buffer: memoryview, idx: int) -> tuple[int, int]:
    """Read lib0 varuint starting at idx."""
    value = 0
    shift = 0
    while True:
        if idx >= len(buffer):
            raise ProtocolDecodeError("unexpected end of frame while reading varuint")
        byte_val = buffer[idx]
        idx += 1
        value |= (byte_val & 0x7F) << shift
        if byte_val < 0x80:
            break
        shift += 7
        if shift > 35:  # safety guard
            raise ProtocolDecodeError("varuint too large")
    return value, idx


def _read_var_uint8_array(buffer: memoryview, idx: int) -> tuple[bytes, int]:
    """Read a length-prefixed byte array."""
    length, idx = _read_var_uint(buffer, idx)
    end = idx + length
    if end > len(buffer):
        raise ProtocolDecodeError("length-prefixed payload exceeds frame size")
    return bytes(buffer[idx:end]), end


def _write_var_uint(buffer: bytearray, value: int) -> None:
    """Write lib0 varuint."""
    rest = value
    while rest >= 0x80:
        buffer.append((rest & 0x7F) | 0x80)
        rest >>= 7
    buffer.append(rest)


def _write_var_uint8_array(buffer: bytearray, value: bytes) -> None:
    """Write length-prefixed byte array."""
    _write_var_uint(buffer, len(value))
    buffer.extend(value)


def _ensure_consumed(idx: int, buffer: memoryview) -> None:
    """Ensure the full frame was consumed; otherwise treat as malformed."""
    if idx != len(buffer):
        raise ProtocolDecodeError("extra trailing bytes in frame")


__all__ = [
    "ProtocolDecodeError",
    "ProtocolMessage",
    "ProtocolMessageType",
    "decode_message",
    "encode_ping",
    "encode_sync_step1",
    "encode_sync_step2",
    "encode_sync_update",
]
