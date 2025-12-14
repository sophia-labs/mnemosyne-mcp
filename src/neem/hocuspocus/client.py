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
import pycrdt

from neem.hocuspocus.protocol import (
    ProtocolDecodeError,
    ProtocolMessageType,
    decode_message,
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

    doc: pycrdt.Doc = field(default_factory=pycrdt.Doc)
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
            base_url="http://localhost:8080",
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
            base_url: Base URL of the Mnemosyne API (e.g., http://localhost:8080)
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
            state_vector = channel.doc.get_state()
            logger.info(
                "Sending sync_step1 with our state vector",
                extra_context={
                    "channel": channel_name,
                    "state_vector_size": len(state_vector),
                    "state_vector_hex": state_vector.hex() if state_vector else "",
                },
            )
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
        logger.debug(
            "Received WebSocket message",
            extra_context={
                "channel": channel_name,
                "data_size": len(data),
                "data_hex_preview": data[:30].hex() if data else "",
            },
        )

        try:
            message = decode_message(data)
        except ProtocolDecodeError as e:
            # Treat as raw Y.js update
            logger.warning(
                "Failed to decode as protocol message, treating as raw update",
                extra_context={
                    "channel": channel_name,
                    "error": str(e),
                    "data_size": len(data),
                },
            )
            channel.doc.apply_update(data)
            return

        logger.info(
            "Decoded protocol message",
            extra_context={
                "channel": channel_name,
                "msg_type": message.type.value,
                "msg_subtype": message.subtype,
                "payload_size": len(message.payload),
            },
        )

        if message.type == ProtocolMessageType.SYNC:
            if message.subtype == "sync_step1":
                # Server sent its state vector, respond with sync_step2
                update = channel.doc.get_update()
                logger.info(
                    "Responding to sync_step1 with sync_step2",
                    extra_context={
                        "channel": channel_name,
                        "our_update_size": len(update),
                        "our_update_hex_preview": update[:50].hex() if update else "",
                    },
                )
                if channel.ws and not channel.ws.closed:
                    await channel.ws.send_bytes(encode_sync_step2(update))

            elif message.subtype == "sync_step2":
                # Server sent us the full state diff - apply it
                # Log content BEFORE applying
                content_fragment = channel.doc.get("content", type=pycrdt.XmlFragment)
                content_before = str(content_fragment) if content_fragment else "(no content)"

                channel.doc.apply_update(message.payload)
                channel.synced.set()

                # Log content AFTER applying
                content_after = str(content_fragment) if content_fragment else "(no content)"
                logger.info(
                    "Applied sync_step2, channel synced",
                    extra_context={
                        "channel": channel_name,
                        "payload_size": len(message.payload),
                        "content_before": content_before[:500],
                        "content_after": content_after[:500],
                    },
                )

            elif message.subtype == "sync_update":
                # Incremental update from server
                content_fragment = channel.doc.get("content", type=pycrdt.XmlFragment)
                content_before = str(content_fragment) if content_fragment else "(no content)"

                channel.doc.apply_update(message.payload)

                content_after = str(content_fragment) if content_fragment else "(no content)"
                logger.info(
                    "Applied sync_update from server",
                    extra_context={
                        "channel": channel_name,
                        "payload_size": len(message.payload),
                        "content_before": content_before[:500],
                        "content_after": content_after[:500],
                    },
                )

        elif message.type == ProtocolMessageType.PING:
            # Respond to ping (though aiohttp heartbeat usually handles this)
            logger.debug("Received ping", extra_context={"channel": channel_name})

        elif message.type == ProtocolMessageType.AWARENESS:
            # Awareness updates - ignore for now (cursor positions, etc.)
            logger.debug(
                "Received awareness update (ignored)",
                extra_context={"channel": channel_name, "payload_size": len(message.payload)},
            )

    # -------------------------------------------------------------------------
    # Session State Accessors (Schema V2 - per-tab state)
    # -------------------------------------------------------------------------

    def get_all_tab_ids(self) -> list[str]:
        """Get all tab IDs from the session."""
        if self._session_channel is None:
            return []
        tabs_map: pycrdt.Map = self._session_channel.doc.get("tabs", type=pycrdt.Map)
        return list(tabs_map.keys())

    def get_most_recent_tab_id(self) -> Optional[str]:
        """Find the tab with the highest lastActiveAt timestamp.

        Returns:
            The most recently active tab ID, or None if no tabs exist.
        """
        if self._session_channel is None:
            return None

        tabs_map: pycrdt.Map = self._session_channel.doc.get("tabs", type=pycrdt.Map)
        most_recent_id: Optional[str] = None
        most_recent_time: int = -1

        for tab_id in tabs_map.keys():
            tab_map = tabs_map.get(tab_id)
            if tab_map is not None and isinstance(tab_map, pycrdt.Map):
                last_active = tab_map.get("lastActiveAt")
                if last_active is not None and last_active > most_recent_time:
                    most_recent_time = last_active
                    most_recent_id = tab_id

        return most_recent_id

    def get_tab_state(self, tab_id: str) -> Optional[Dict[str, Any]]:
        """Get state for a specific tab.

        Args:
            tab_id: The tab's unique identifier.

        Returns:
            Tab state dict or None if tab doesn't exist.
        """
        if self._session_channel is None:
            return None

        tabs_map: pycrdt.Map = self._session_channel.doc.get("tabs", type=pycrdt.Map)
        tab_map = tabs_map.get(tab_id)
        if tab_map is None or not isinstance(tab_map, pycrdt.Map):
            return None

        return {key: tab_map.get(key) for key in tab_map.keys()}

    def get_active_graph_id(self, tab_id: Optional[str] = None) -> Optional[str]:
        """Get the currently active graph ID from the session.

        Args:
            tab_id: Specific tab to query. If None, uses most recent tab.

        Returns:
            The active graph ID or None.
        """
        if self._session_channel is None:
            return None

        target_tab = tab_id or self.get_most_recent_tab_id()
        if not target_tab:
            return None

        tabs_map: pycrdt.Map = self._session_channel.doc.get("tabs", type=pycrdt.Map)
        tab_map = tabs_map.get(target_tab)
        if tab_map is None or not isinstance(tab_map, pycrdt.Map):
            return None

        return tab_map.get("activeGraphId")

    def get_active_document_id(self, tab_id: Optional[str] = None) -> Optional[str]:
        """Get the currently active document ID from the session.

        Args:
            tab_id: Specific tab to query. If None, uses most recent tab.

        Returns:
            The active document ID or None.
        """
        if self._session_channel is None:
            return None

        target_tab = tab_id or self.get_most_recent_tab_id()
        if not target_tab:
            return None

        tabs_map: pycrdt.Map = self._session_channel.doc.get("tabs", type=pycrdt.Map)
        tab_map = tabs_map.get(target_tab)
        if tab_map is None or not isinstance(tab_map, pycrdt.Map):
            return None

        return tab_map.get("activeDocumentId")

    def get_session_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the session state (Schema V2 with tabs)."""
        if self._session_channel is None:
            return {}

        doc = self._session_channel.doc

        def ymap_to_dict(ymap: pycrdt.Map) -> Dict[str, Any]:
            result = {}
            for key in ymap.keys():
                value = ymap.get(key)
                if isinstance(value, pycrdt.Map):
                    result[key] = ymap_to_dict(value)
                elif isinstance(value, pycrdt.Array):
                    result[key] = list(value)
                else:
                    result[key] = value
            return result

        # Build tabs snapshot (Schema V2)
        tabs_map: pycrdt.Map = doc.get("tabs", type=pycrdt.Map)
        tabs_snapshot = {}
        for tab_id in tabs_map.keys():
            tab_state = tabs_map.get(tab_id)
            if isinstance(tab_state, pycrdt.Map):
                tabs_snapshot[tab_id] = ymap_to_dict(tab_state)

        preferences_map: pycrdt.Map = doc.get("preferences", type=pycrdt.Map)

        return {
            "preferences": ymap_to_dict(preferences_map),
            "tabs": tabs_snapshot,
            "most_recent_tab_id": self.get_most_recent_tab_id(),
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

        def ymap_to_dict(ymap: pycrdt.Map) -> Dict[str, Any]:
            result = {}
            for key in ymap.keys():
                value = ymap.get(key)
                if isinstance(value, pycrdt.Map):
                    result[key] = ymap_to_dict(value)
                elif isinstance(value, pycrdt.Array):
                    result[key] = list(value)
                else:
                    result[key] = value
            return result

        # Workspace uses four separate YMaps: folders, artifacts, documents, ui
        folders_map: pycrdt.Map = doc.get("folders", type=pycrdt.Map)
        artifacts_map: pycrdt.Map = doc.get("artifacts", type=pycrdt.Map)
        documents_map: pycrdt.Map = doc.get("documents", type=pycrdt.Map)
        ui_map: pycrdt.Map = doc.get("ui", type=pycrdt.Map)

        return {
            "folders": ymap_to_dict(folders_map),
            "artifacts": ymap_to_dict(artifacts_map),
            "documents": ymap_to_dict(documents_map),
            "ui": ymap_to_dict(ui_map),
        }

    async def transact_workspace(
        self,
        graph_id: str,
        operation: Callable[[pycrdt.Doc], None],
    ) -> None:
        """Execute an operation on a workspace and broadcast the incremental update.

        This is the correct way to make collaborative workspace edits. It:
        1. Captures the state vector BEFORE changes
        2. Executes your operation (which modifies the doc)
        3. Computes the INCREMENTAL diff (not full state)
        4. Broadcasts only the diff to other clients

        Args:
            graph_id: The graph ID
            operation: Callable that modifies the doc (e.g., lambda doc: WorkspaceWriter(doc).upsert_document(...))

        Example:
            await client.transact_workspace(graph_id, lambda doc:
                WorkspaceWriter(doc).upsert_document(doc_id, "My Document")
            )
        """
        channel = self._workspace_channels.get(graph_id)
        if channel is None:
            raise ValueError(f"Workspace channel not connected: {graph_id}")

        # Capture state BEFORE changes
        old_state = channel.doc.get_state()

        # Execute the operation (modifies doc in place)
        operation(channel.doc)

        # Get INCREMENTAL update (diff from old state)
        incremental_update = channel.doc.get_update(old_state)

        # Only send if there are actual changes and connection is alive
        if incremental_update and channel.ws and not channel.ws.closed:
            await channel.ws.send_bytes(encode_sync_update(incremental_update))
            logger.debug(
                "Sent incremental workspace update",
                extra_context={
                    "graph_id": graph_id,
                    "update_size": len(incremental_update),
                },
            )

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

        DEPRECATED: This method has a bug - it double-applies updates when used
        with DocumentWriter methods that already modify the doc. Use
        transact_document() instead for proper incremental update handling.

        Args:
            graph_id: The graph ID
            doc_id: The document ID
            update: The Y.js update bytes
        """
        import warnings

        warnings.warn(
            "apply_document_update is deprecated. Use transact_document() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        key = f"{graph_id}:{doc_id}"
        channel = self._document_channels.get(key)
        if channel is None:
            raise ValueError(f"Document channel not connected: {key}")

        # Apply locally
        channel.doc.apply_update(update)

        # Send to server
        if channel.ws and not channel.ws.closed:
            await channel.ws.send_bytes(encode_sync_update(update))

    async def transact_document(
        self,
        graph_id: str,
        doc_id: str,
        operation: Callable[[pycrdt.Doc], None],
    ) -> None:
        """Execute an operation on a document and broadcast the incremental update.

        This is the correct way to make collaborative edits. It:
        1. Captures the state vector BEFORE changes
        2. Executes your operation (which modifies the doc)
        3. Computes the INCREMENTAL diff (not full state)
        4. Broadcasts only the diff to other clients

        Args:
            graph_id: The graph ID
            doc_id: The document ID
            operation: Callable that modifies the doc (e.g., lambda doc: writer.append_block(...))

        Example:
            await client.transact_document(graph_id, doc_id, lambda doc:
                DocumentWriter(doc).append_block("<paragraph>Hello</paragraph>")
            )
        """
        key = f"{graph_id}:{doc_id}"
        channel = self._document_channels.get(key)
        if channel is None:
            raise ValueError(f"Document channel not connected: {key}")

        # Capture state and content BEFORE changes
        old_state = channel.doc.get_state()
        content_fragment: pycrdt.XmlFragment = channel.doc.get("content", type=pycrdt.XmlFragment)
        content_before = str(content_fragment)
        block_count_before = len(list(content_fragment.children)) if content_fragment else 0

        logger.info(
            "transact_document: BEFORE operation",
            extra_context={
                "graph_id": graph_id,
                "doc_id": doc_id,
                "block_count": block_count_before,
                "state_vector_size": len(old_state),
                "content_preview": content_before[:500] if content_before else "(empty)",
            },
        )

        # Execute the operation (modifies doc in place)
        try:
            operation(channel.doc)
        except Exception as e:
            logger.error(
                "transact_document: operation FAILED",
                extra_context={
                    "graph_id": graph_id,
                    "doc_id": doc_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

        # Log content AFTER changes
        content_after = str(content_fragment)
        block_count_after = len(list(content_fragment.children)) if content_fragment else 0

        logger.info(
            "transact_document: AFTER operation",
            extra_context={
                "graph_id": graph_id,
                "doc_id": doc_id,
                "block_count_before": block_count_before,
                "block_count_after": block_count_after,
                "content_preview": content_after[:500] if content_after else "(empty)",
            },
        )

        # Get INCREMENTAL update (diff from old state)
        incremental_update = channel.doc.get_update(old_state)
        new_state = channel.doc.get_state()

        logger.info(
            "transact_document: computed incremental update",
            extra_context={
                "graph_id": graph_id,
                "doc_id": doc_id,
                "old_state_size": len(old_state),
                "new_state_size": len(new_state),
                "incremental_update_size": len(incremental_update) if incremental_update else 0,
            },
        )

        # Only send if there are actual changes and connection is alive
        if incremental_update and channel.ws and not channel.ws.closed:
            encoded_message = encode_sync_update(incremental_update)
            logger.info(
                "transact_document: sending encoded sync_update",
                extra_context={
                    "graph_id": graph_id,
                    "doc_id": doc_id,
                    "raw_update_size": len(incremental_update),
                    "encoded_message_size": len(encoded_message),
                    "encoded_hex_preview": encoded_message[:50].hex(),
                    "ws_closed": channel.ws.closed,
                },
            )
            await channel.ws.send_bytes(encoded_message)
            logger.info(
                "transact_document: SENT to server",
                extra_context={
                    "graph_id": graph_id,
                    "doc_id": doc_id,
                    "update_size": len(incremental_update),
                },
            )
        else:
            logger.warning(
                "transact_document: NOT sending update",
                extra_context={
                    "graph_id": graph_id,
                    "doc_id": doc_id,
                    "has_update": bool(incremental_update),
                    "update_size": len(incremental_update) if incremental_update else 0,
                    "has_ws": channel.ws is not None,
                    "ws_closed": channel.ws.closed if channel.ws else True,
                },
            )

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
