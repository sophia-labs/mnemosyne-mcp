"""Hocuspocus WebSocket client for Y.js synchronization.

Provides persistent connections to the Mnemosyne backend's Hocuspocus endpoints
for real-time state synchronization of sessions, workspaces, and documents.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import y_py as Y  # type: ignore[import-untyped]

from neem.hocuspocus.protocol import (
    ProtocolDecodeError,
    ProtocolMessageType,
    decode_message,
    encode_ping,
    encode_sync_step1,
    encode_sync_step2,
    encode_sync_update,
)
from neem.utils.logging import LoggerFactory

logger = LoggerFactory.get_logger("hocuspocus.client")

TokenProvider = Callable[[], Optional[str]]


@dataclass
class ChannelState:
    """State for a single Y.js channel (session, workspace, or document)."""

    doc: Y.YDoc = field(default_factory=Y.YDoc)
    ws: Optional[aiohttp.ClientWebSocketResponse] = None
    synced: asyncio.Event = field(default_factory=asyncio.Event)
    receiver_task: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class HocuspocusClient:
    """Client for connecting to Hocuspocus WebSocket endpoints.

    Manages persistent connections to:
    - Session channel: Cross-graph UI state (activeGraphId, activeDocumentId)
    - Workspace channels: Per-graph folder/artifact structure
    - Document channels: Per-document content

    Usage:
        client = HocuspocusClient(
            base_url="http://localhost:8001",
            token_provider=lambda: get_token(),
        )

        # Connect to session (auto-connects on first access)
        await client.ensure_session_connected(user_id)
        active_graph = client.get_active_graph_id()

        # Connect to a document
        await client.connect_document(graph_id, doc_id)
        blocks = client.get_document_blocks(graph_id, doc_id)

        # Cleanup
        await client.close()
    """

    def __init__(
        self,
        base_url: str,
        token_provider: TokenProvider,
        *,
        dev_user_id: Optional[str] = None,
        connect_timeout: float = 10.0,
        heartbeat_interval: float = 30.0,
    ) -> None:
        """Initialize the Hocuspocus client.

        Args:
            base_url: Base URL of the Mnemosyne API (e.g., http://localhost:8001)
            token_provider: Callable that returns the current auth token
            dev_user_id: Optional dev mode user ID (bypasses OAuth)
            connect_timeout: WebSocket connection timeout in seconds
            heartbeat_interval: Interval between ping messages
        """
        self._base_url = base_url.rstrip("/")
        self._token_provider = token_provider
        self._dev_user_id = dev_user_id
        self._connect_timeout = connect_timeout
        self._heartbeat_interval = heartbeat_interval

        # HTTP session for WebSocket connections
        self._session: Optional[aiohttp.ClientSession] = None

        # Channel state management
        self._session_channel: Optional[ChannelState] = None
        self._workspace_channels: Dict[str, ChannelState] = {}  # graph_id -> state
        self._document_channels: Dict[str, ChannelState] = {}  # graph_id:doc_id -> state

        # Shutdown flag
        self._closed = False

    @property
    def is_session_connected(self) -> bool:
        """Check if the session channel is connected and synced."""
        return (
            self._session_channel is not None
            and self._session_channel.ws is not None
            and not self._session_channel.ws.closed
            and self._session_channel.synced.is_set()
        )

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _build_ws_url(self, path: str) -> str:
        """Build WebSocket URL from HTTP base URL and path."""
        parsed = urlparse(self._base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        ws_base = f"{ws_scheme}://{parsed.netloc}"
        return urljoin(ws_base, path)

    def _build_auth_headers(self) -> Dict[str, str]:
        """Build authentication headers for WebSocket connection."""
        headers: Dict[str, str] = {}
        token = self._token_provider()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if self._dev_user_id:
            headers["X-User-ID"] = self._dev_user_id
        return headers

    def _build_ws_protocols(self) -> Optional[list[str]]:
        """Build WebSocket subprotocols for auth (used in dev mode)."""
        if self._dev_user_id:
            return [f"Bearer.{self._dev_user_id}"]
        return None

    # -------------------------------------------------------------------------
    # Session Channel (cross-graph UI state)
    # -------------------------------------------------------------------------

    async def ensure_session_connected(self, user_id: str) -> None:
        """Ensure connection to the user's session channel.

        Args:
            user_id: The user ID (used for room naming, but auth comes from token)
        """
        if self.is_session_connected:
            return

        if self._session_channel is None:
            self._session_channel = ChannelState()

        async with self._session_channel.lock:
            if self.is_session_connected:
                return  # Double-check after acquiring lock

            await self._connect_channel(
                self._session_channel,
                f"/hocuspocus/session/{user_id}",
                f"session:{user_id}",
            )

    async def _connect_channel(
        self,
        channel: ChannelState,
        path: str,
        channel_name: str,
    ) -> None:
        """Connect a channel to its WebSocket endpoint and perform initial sync."""
        ws_url = self._build_ws_url(path)
        headers = self._build_auth_headers()
        protocols = self._build_ws_protocols()

        logger.info(
            "Connecting to Hocuspocus channel",
            extra_context={"channel": channel_name, "url": ws_url},
        )

        try:
            session = await self._get_http_session()
            channel.ws = await session.ws_connect(
                ws_url,
                headers=headers,
                protocols=protocols or [],
                timeout=aiohttp.ClientTimeout(total=self._connect_timeout),
                heartbeat=self._heartbeat_interval,
            )

            # Start receiver task
            channel.receiver_task = asyncio.create_task(
                self._receiver_loop(channel, channel_name),
                name=f"hocuspocus-{channel_name}",
            )

            # Send our state vector to initiate sync
            state_vector = Y.encode_state_vector(channel.doc)
            await channel.ws.send_bytes(encode_sync_step1(state_vector))

            # Wait for sync to complete (server sends sync_step2)
            try:
                await asyncio.wait_for(channel.synced.wait(), timeout=10.0)
                logger.info(
                    "Hocuspocus channel synced",
                    extra_context={"channel": channel_name},
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Hocuspocus sync timeout, continuing anyway",
                    extra_context={"channel": channel_name},
                )

        except Exception as exc:
            logger.error(
                "Failed to connect to Hocuspocus channel",
                extra_context={"channel": channel_name, "error": str(exc)},
            )
            raise

    async def _receiver_loop(self, channel: ChannelState, channel_name: str) -> None:
        """Receive and process messages from the WebSocket."""
        if channel.ws is None:
            return

        try:
            async for msg in channel.ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    await self._handle_message(channel, msg.data, channel_name)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        "WebSocket error",
                        extra_context={"channel": channel_name, "error": channel.ws.exception()},
                    )
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                "Receiver loop error",
                extra_context={"channel": channel_name, "error": str(exc)},
            )
        finally:
            logger.debug(
                "Receiver loop ended",
                extra_context={"channel": channel_name},
            )

    async def _handle_message(
        self,
        channel: ChannelState,
        data: bytes,
        channel_name: str,
    ) -> None:
        """Handle an incoming WebSocket message."""
        try:
            message = decode_message(data)
        except ProtocolDecodeError:
            # Treat as raw Y.js update
            Y.apply_update(channel.doc, data)
            return

        if message.type == ProtocolMessageType.SYNC:
            if message.subtype == "sync_step1":
                # Server sent its state vector, respond with sync_step2
                update = Y.encode_state_as_update(channel.doc)
                if channel.ws and not channel.ws.closed:
                    await channel.ws.send_bytes(encode_sync_step2(update))

            elif message.subtype == "sync_step2":
                # Server sent us the full state diff - apply it
                Y.apply_update(channel.doc, message.payload)
                channel.synced.set()
                logger.debug(
                    "Received sync_step2, channel synced",
                    extra_context={"channel": channel_name},
                )

            elif message.subtype == "sync_update":
                # Incremental update from server
                Y.apply_update(channel.doc, message.payload)

        elif message.type == ProtocolMessageType.PING:
            # Respond to ping (though aiohttp heartbeat usually handles this)
            if channel.ws and not channel.ws.closed:
                from neem.hocuspocus.protocol import encode_ping
                # Actually we should encode_pong, but we don't have it - ping is fine
                pass

        elif message.type == ProtocolMessageType.AWARENESS:
            # Awareness updates - ignore for now (cursor positions, etc.)
            pass

    # -------------------------------------------------------------------------
    # Session State Accessors
    # -------------------------------------------------------------------------

    def get_active_graph_id(self) -> Optional[str]:
        """Get the currently active graph ID from the session."""
        if self._session_channel is None:
            return None
        navigation = self._session_channel.doc.get_map("navigation")
        return navigation.get("activeGraphId")

    def get_active_document_id(self) -> Optional[str]:
        """Get the currently active document ID from the session."""
        if self._session_channel is None:
            return None
        navigation = self._session_channel.doc.get_map("navigation")
        return navigation.get("activeDocumentId")

    def get_session_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the session state."""
        if self._session_channel is None:
            return {}

        doc = self._session_channel.doc

        def ymap_to_dict(ymap: Y.YMap) -> Dict[str, Any]:
            return {key: ymap.get(key) for key in ymap.keys()}

        return {
            "layout": ymap_to_dict(doc.get_map("layout")),
            "preferences": ymap_to_dict(doc.get_map("preferences")),
            "navigation": ymap_to_dict(doc.get_map("navigation")),
        }

    # -------------------------------------------------------------------------
    # Workspace Channel (per-graph filesystem)
    # -------------------------------------------------------------------------

    async def connect_workspace(self, graph_id: str) -> None:
        """Connect to a workspace channel for the given graph.

        Args:
            graph_id: The graph ID
        """
        if graph_id in self._workspace_channels:
            channel = self._workspace_channels[graph_id]
            if channel.ws and not channel.ws.closed and channel.synced.is_set():
                return  # Already connected and synced

        channel = ChannelState()
        self._workspace_channels[graph_id] = channel

        async with channel.lock:
            await self._connect_channel(
                channel,
                f"/hocuspocus/workspace/{graph_id}",
                f"workspace:{graph_id}",
            )

    def get_workspace_snapshot(self, graph_id: str) -> Dict[str, Any]:
        """Get a snapshot of the workspace state for a graph."""
        channel = self._workspace_channels.get(graph_id)
        if channel is None:
            return {}

        doc = channel.doc

        def ymap_to_dict(ymap: Y.YMap) -> Dict[str, Any]:
            result = {}
            for key in ymap.keys():
                value = ymap.get(key)
                if isinstance(value, Y.YMap):
                    result[key] = ymap_to_dict(value)
                elif isinstance(value, Y.YArray):
                    result[key] = list(value)
                else:
                    result[key] = value
            return result

        # The workspace structure varies - return the whole doc
        return {
            "filesystem": ymap_to_dict(doc.get_map("filesystem")),
        }

    # -------------------------------------------------------------------------
    # Document Channel (per-document content)
    # -------------------------------------------------------------------------

    async def connect_document(self, graph_id: str, doc_id: str) -> None:
        """Connect to a document channel.

        Args:
            graph_id: The graph ID
            doc_id: The document ID
        """
        key = f"{graph_id}:{doc_id}"

        if key in self._document_channels:
            channel = self._document_channels[key]
            if channel.ws and not channel.ws.closed and channel.synced.is_set():
                return  # Already connected and synced

        channel = ChannelState()
        self._document_channels[key] = channel

        async with channel.lock:
            await self._connect_channel(
                channel,
                f"/hocuspocus/docs/{graph_id}/{doc_id}",
                f"doc:{graph_id}:{doc_id}",
            )

    def get_document_channel(self, graph_id: str, doc_id: str) -> Optional[ChannelState]:
        """Get the channel state for a document."""
        key = f"{graph_id}:{doc_id}"
        return self._document_channels.get(key)

    async def apply_document_update(
        self,
        graph_id: str,
        doc_id: str,
        update: bytes,
    ) -> None:
        """Apply a Y.js update to a document and broadcast to server.

        Args:
            graph_id: The graph ID
            doc_id: The document ID
            update: The Y.js update bytes
        """
        key = f"{graph_id}:{doc_id}"
        channel = self._document_channels.get(key)
        if channel is None:
            raise ValueError(f"Document channel not connected: {key}")

        # Apply locally
        Y.apply_update(channel.doc, update)

        # Send to server
        if channel.ws and not channel.ws.closed:
            await channel.ws.send_bytes(encode_sync_update(update))

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    async def disconnect_document(self, graph_id: str, doc_id: str) -> None:
        """Disconnect from a document channel."""
        key = f"{graph_id}:{doc_id}"
        channel = self._document_channels.pop(key, None)
        if channel:
            await self._close_channel(channel)

    async def disconnect_workspace(self, graph_id: str) -> None:
        """Disconnect from a workspace channel."""
        channel = self._workspace_channels.pop(graph_id, None)
        if channel:
            await self._close_channel(channel)

    async def _close_channel(self, channel: ChannelState) -> None:
        """Close a channel's WebSocket and cleanup."""
        if channel.receiver_task:
            channel.receiver_task.cancel()
            try:
                await channel.receiver_task
            except asyncio.CancelledError:
                pass

        if channel.ws and not channel.ws.closed:
            await channel.ws.close()

    async def close(self) -> None:
        """Close all connections and cleanup."""
        self._closed = True

        # Close document channels
        for key in list(self._document_channels.keys()):
            channel = self._document_channels.pop(key)
            await self._close_channel(channel)

        # Close workspace channels
        for graph_id in list(self._workspace_channels.keys()):
            channel = self._workspace_channels.pop(graph_id)
            await self._close_channel(channel)

        # Close session channel
        if self._session_channel:
            await self._close_channel(self._session_channel)
            self._session_channel = None

        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        logger.info("Hocuspocus client closed")


__all__ = ["HocuspocusClient", "ChannelState"]
