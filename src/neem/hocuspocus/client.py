"""Hocuspocus WebSocket client for Y.js synchronization.

Provides persistent connections to the Mnemosyne backend's Hocuspocus endpoints
for real-time state synchronization of sessions, workspaces, and documents.
"""

from __future__ import annotations

import asyncio
import os
import time
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
    synced_at: float = 0.0  # monotonic timestamp of last successful sync
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
        internal_service_secret: Optional[str] = None,
        connect_timeout: float = 10.0,
        sync_timeout: float = 20.0,
        connect_retries: int = 1,
        heartbeat_interval: float = 30.0,
    ) -> None:
        """Initialize the Hocuspocus client.

        Args:
            base_url: Base URL of the Mnemosyne API (e.g., http://localhost:8080)
            token_provider: Callable that returns the current auth token
            dev_user_id: Optional dev mode user ID (bypasses OAuth)
            internal_service_secret: Shared secret for cluster-internal auth
            connect_timeout: WebSocket connection timeout in seconds
            sync_timeout: Max seconds to wait for initial Y.js sync step 2
            connect_retries: Retries on sync timeout (per channel connect)
            heartbeat_interval: Interval between ping messages
        """
        self._base_url = base_url.rstrip("/")
        self._token_provider = token_provider
        self._dev_user_id = dev_user_id
        self._internal_service_secret = internal_service_secret
        self._connect_timeout = connect_timeout
        self._sync_timeout = max(
            1.0,
            float(os.getenv("MNEMOSYNE_HOCUSPOCUS_SYNC_TIMEOUT", str(sync_timeout))),
        )
        workspace_max_age_env = os.getenv("MNEMOSYNE_HOCUSPOCUS_WORKSPACE_MAX_AGE", "5.0")
        try:
            workspace_max_age = float(workspace_max_age_env)
        except ValueError:
            logger.warning(
                "Invalid MNEMOSYNE_HOCUSPOCUS_WORKSPACE_MAX_AGE, using default",
                extra_context={
                    "value": workspace_max_age_env,
                    "default": 5.0,
                },
            )
            workspace_max_age = 5.0
        self._workspace_max_age = max(
            0.0,
            workspace_max_age,
        )
        self._connect_retries = max(
            0,
            int(os.getenv("MNEMOSYNE_HOCUSPOCUS_CONNECT_RETRIES", str(connect_retries))),
        )
        self._heartbeat_interval = heartbeat_interval

        # HTTP session for WebSocket connections
        self._session: Optional[aiohttp.ClientSession] = None

        # Channel state management
        self._session_channel: Optional[ChannelState] = None
        self._workspace_channels: Dict[str, ChannelState] = {}  # user_id:graph_id -> state
        self._document_channels: Dict[str, ChannelState] = {}  # user_id:graph_id:doc_id -> state

        # Per-key connection locks to prevent concurrent channel creation races.
        # Without these, parallel callers can overwrite each other's ChannelState,
        # causing one caller to read from an unsynced channel.
        self._workspace_connect_locks: Dict[str, asyncio.Lock] = {}
        self._document_connect_locks: Dict[str, asyncio.Lock] = {}

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

    def _build_auth_headers(self, user_id: Optional[str] = None) -> Dict[str, str]:
        """Build authentication headers for WebSocket connection.

        Args:
            user_id: Optional per-connection user override. When provided,
                this must take precedence over the global dev fallback user so
                sidecars do not accidentally write to the wrong user namespace.
        """
        headers: Dict[str, str] = {}
        token = self._token_provider()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        effective_user_id = user_id or self._dev_user_id
        if effective_user_id:
            headers["X-User-ID"] = effective_user_id
        # Add internal service auth header for cluster-internal requests
        if self._internal_service_secret:
            headers["X-Internal-Service"] = self._internal_service_secret
        return headers

    def _build_ws_protocols(self, user_id: Optional[str] = None) -> Optional[list[str]]:
        """Build WebSocket subprotocols for auth.

        Supported formats:
        - internal.{user_id}.{secret} - internal service auth (preferred for sidecars)
        - Bearer.{token} - JWT token auth (required for cognito_jwt mode)
        - Bearer.{user_id} - dev mode fallback (dev_no_auth mode)

        Internal service auth via subprotocol is preferred because it survives
        proxies that may strip custom HTTP headers during WebSocket upgrade.

        Args:
            user_id: Override user_id for this connection. If not provided, uses _dev_user_id.
        """
        effective_user_id = user_id or self._dev_user_id
        # Prefer internal service auth via subprotocol (survives proxies)
        if self._internal_service_secret and effective_user_id:
            return [f"internal.{effective_user_id}.{self._internal_service_secret}"]
        # Use JWT token in subprotocol (required for cognito_jwt auth mode)
        token = self._token_provider()
        if token:
            return [f"Bearer.{token}"]
        # Fallback to user_id for dev_no_auth mode (server treats it as user identity)
        if effective_user_id:
            return [f"Bearer.{effective_user_id}"]
        return None

    # -------------------------------------------------------------------------
    # Session Channel (cross-graph UI state)
    # -------------------------------------------------------------------------

    async def ensure_session_connected(self, user_id: str) -> None:
        """Ensure connection to the user's session channel.

        Args:
            user_id: The user ID (used for room naming AND auth)
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
                user_id=user_id,
            )

    async def refresh_session(self, user_id: str) -> None:
        """Close and reconnect the session channel to get fresh state.

        The session WebSocket doesn't receive incremental Y.js updates
        after initial sync, so we reconnect to get the latest state.
        """
        if self._session_channel is not None:
            await self._close_channel(self._session_channel)
            self._session_channel = None
        await self.ensure_session_connected(user_id)

    async def _connect_channel(
        self,
        channel: ChannelState,
        path: str,
        channel_name: str,
        user_id: Optional[str] = None,
    ) -> None:
        """Connect a channel to its WebSocket endpoint and perform initial sync.

        Args:
            channel: The channel state to connect
            path: WebSocket endpoint path
            channel_name: Human-readable channel name for logging
            user_id: Override user_id for auth (uses _dev_user_id if not provided)
        """
        ws_url = self._build_ws_url(path)
        headers = self._build_auth_headers(user_id=user_id)
        protocols = self._build_ws_protocols(user_id=user_id)

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
                max_msg_size=16 * 1024 * 1024,  # 16MB — match uvicorn default
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
                await asyncio.wait_for(channel.synced.wait(), timeout=self._sync_timeout)
                channel.synced_at = time.monotonic()
                logger.info(
                    "Hocuspocus channel synced",
                    extra_context={"channel": channel_name},
                )
            except asyncio.TimeoutError:
                close_code = channel.ws.close_code if channel.ws is not None else None
                ws_exception = None
                try:
                    if channel.ws is not None:
                        ws_exception = channel.ws.exception()
                except Exception:
                    ws_exception = None
                logger.error(
                    "Hocuspocus sync timeout",
                    extra_context={
                        "channel": channel_name,
                        "timeout_seconds": self._sync_timeout,
                        "close_code": close_code,
                        "ws_exception": str(ws_exception) if ws_exception else None,
                    },
                )
                raise TimeoutError(
                    f"Hocuspocus sync timed out for channel: {channel_name} "
                    f"(timeout={self._sync_timeout}s close_code={close_code})"
                )

        except Exception as exc:
            await self._reset_channel(channel)
            logger.error(
                "Failed to connect to Hocuspocus channel",
                extra_context={"channel": channel_name, "error": str(exc)},
            )
            raise

    async def _reset_channel(self, channel: ChannelState) -> None:
        """Reset channel state after a failed connection attempt."""
        if channel.receiver_task:
            channel.receiver_task.cancel()
            try:
                await channel.receiver_task
            except asyncio.CancelledError:
                pass
            finally:
                channel.receiver_task = None

        if channel.ws and not channel.ws.closed:
            try:
                await channel.ws.close()
            except Exception:
                pass
        channel.ws = None
        channel.synced.clear()

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
            channel.synced_at = time.monotonic()
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
                channel.synced_at = time.monotonic()

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
                channel.synced_at = time.monotonic()

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

    async def connect_workspace(
        self,
        graph_id: str,
        user_id: Optional[str] = None,
        *,
        force_fresh: bool = False,
        max_age: Optional[float] = None,
    ) -> None:
        """Connect to a workspace channel for the given graph.

        Args:
            graph_id: The graph ID
            user_id: The user ID for auth (uses _dev_user_id if not provided)
            force_fresh: If True, always disconnect and reconnect.
            max_age: Max seconds since last sync before reconnecting.
                If not provided, uses MNEMOSYNE_HOCUSPOCUS_WORKSPACE_MAX_AGE
                (default 5s). Set to 0 to disable age-based reconnect.
        """
        effective_user_id = user_id or self._dev_user_id
        channel_key = f"{effective_user_id}:{graph_id}"
        effective_max_age = self._workspace_max_age if max_age is None else max_age

        if force_fresh and channel_key in self._workspace_channels:
            await self.disconnect_workspace(graph_id, user_id=effective_user_id)
        elif (
            effective_max_age is not None
            and effective_max_age > 0
            and channel_key in self._workspace_channels
        ):
            channel = self._workspace_channels[channel_key]
            if channel.synced_at > 0:
                age = time.monotonic() - channel.synced_at
                if age > effective_max_age:
                    logger.debug(
                        "Workspace channel stale; reconnecting",
                        extra_context={
                            "graph_id": graph_id,
                            "age_seconds": round(age, 3),
                            "max_age_seconds": effective_max_age,
                        },
                    )
                    await self.disconnect_workspace(graph_id, user_id=effective_user_id)

        # Fast path: already connected and synced (no lock needed)
        if channel_key in self._workspace_channels:
            channel = self._workspace_channels[channel_key]
            if channel.ws and not channel.ws.closed and channel.synced.is_set():
                return

        # Slow path: acquire per-key lock to prevent concurrent channel creation.
        # Without this, parallel callers (e.g. orientation batch: music + recall +
        # get_important_blocks) can overwrite each other's ChannelState, causing
        # one caller to read from an unsynced empty channel.
        if channel_key not in self._workspace_connect_locks:
            self._workspace_connect_locks[channel_key] = asyncio.Lock()

        async with self._workspace_connect_locks[channel_key]:
            # Re-check under lock (another caller may have connected while we waited)
            if channel_key in self._workspace_channels:
                channel = self._workspace_channels[channel_key]
                if channel.ws and not channel.ws.closed and channel.synced.is_set():
                    return

            for attempt in range(self._connect_retries + 1):
                channel = ChannelState()
                self._workspace_channels[channel_key] = channel

                try:
                    async with channel.lock:
                        await self._connect_channel(
                            channel,
                            f"/hocuspocus/workspace/{effective_user_id}/{graph_id}",
                            f"workspace:{graph_id}",
                            user_id=user_id,
                        )
                    return
                except TimeoutError:
                    # Remove broken/unsynced channel before retry.
                    self._workspace_channels.pop(channel_key, None)
                    if attempt >= self._connect_retries:
                        raise
                    retry_delay = 0.25 * (attempt + 1)
                    logger.warning(
                        "Workspace sync timed out; retrying connect",
                        extra_context={
                            "graph_id": graph_id,
                            "attempt": attempt + 1,
                            "max_retries": self._connect_retries,
                            "retry_delay_seconds": retry_delay,
                        },
                    )
                    await asyncio.sleep(retry_delay)
                except Exception:
                    self._workspace_channels.pop(channel_key, None)
                    raise

    def get_workspace_channel(
        self, graph_id: str, user_id: Optional[str] = None
    ) -> Optional[ChannelState]:
        """Get the channel state for a workspace."""
        effective_user_id = user_id or self._dev_user_id
        key = f"{effective_user_id}:{graph_id}"
        return self._workspace_channels.get(key)

    def get_workspace_snapshot(self, graph_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a snapshot of the workspace state for a graph."""
        channel_key = f"{user_id or self._dev_user_id}:{graph_id}"
        channel = self._workspace_channels.get(channel_key)
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
        user_id: Optional[str] = None,
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
            user_id: The user ID (uses _dev_user_id if not provided)

        Example:
            await client.transact_workspace(graph_id, lambda doc:
                WorkspaceWriter(doc).upsert_document(doc_id, "My Document")
            , user_id="user-123")
        """
        channel_key = f"{user_id or self._dev_user_id}:{graph_id}"
        channel = self._workspace_channels.get(channel_key)
        if channel is None:
            raise ValueError(f"Workspace channel not connected: {channel_key}")

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

    async def connect_document(
        self, graph_id: str, doc_id: str, user_id: Optional[str] = None,
        force_fresh: bool = False,
        max_age: Optional[float] = None,
    ) -> None:
        """Connect to a document channel.

        Args:
            graph_id: The graph ID
            doc_id: The document ID
            user_id: The user ID for auth (uses _dev_user_id if not provided)
            force_fresh: If True, always disconnect and reconnect.
            max_age: Max seconds since last sync before reconnecting.
                If set, a cached channel older than this is torn down and
                reconnected.  Preferred over force_fresh for read tools —
                it preserves the fast path for rapid sequential reads of
                the same document while still catching multi-agent staleness.
        """
        effective_user_id = user_id or self._dev_user_id
        key = f"{effective_user_id}:{graph_id}:{doc_id}"

        # Decide whether to tear down the cached channel
        need_teardown = False
        if force_fresh and key in self._document_channels:
            need_teardown = True
        elif max_age is not None and key in self._document_channels:
            channel = self._document_channels[key]
            if channel.synced_at > 0:
                age = time.monotonic() - channel.synced_at
                if age > max_age:
                    need_teardown = True

        if need_teardown:
            old_channel = self._document_channels.pop(key, None)
            if old_channel:
                await self._close_channel(old_channel)

        # Fast path: already connected and synced
        if key in self._document_channels:
            channel = self._document_channels[key]
            if channel.ws and not channel.ws.closed and channel.synced.is_set():
                return

        # Slow path: acquire per-key lock to prevent concurrent channel creation
        if key not in self._document_connect_locks:
            self._document_connect_locks[key] = asyncio.Lock()

        async with self._document_connect_locks[key]:
            # Re-check under lock (another coroutine may have connected while we waited)
            if key in self._document_channels:
                channel = self._document_channels[key]
                if channel.ws and not channel.ws.closed and channel.synced.is_set():
                    return

            for attempt in range(self._connect_retries + 1):
                channel = ChannelState()
                self._document_channels[key] = channel

                try:
                    async with channel.lock:
                        await self._connect_channel(
                            channel,
                            f"/hocuspocus/docs/{graph_id}/{doc_id}",
                            f"doc:{graph_id}:{doc_id}",
                            user_id=user_id,
                        )
                    return
                except TimeoutError:
                    # Clean up the broken entry so future calls don't see a zombie
                    self._document_channels.pop(key, None)
                    if attempt >= self._connect_retries:
                        raise
                    retry_delay = 0.25 * (attempt + 1)
                    logger.warning(
                        "Document sync timed out; retrying connect",
                        extra_context={
                            "graph_id": graph_id,
                            "doc_id": doc_id,
                            "attempt": attempt + 1,
                            "max_retries": self._connect_retries,
                            "retry_delay_seconds": retry_delay,
                        },
                    )
                    await asyncio.sleep(retry_delay)
                except Exception:
                    self._document_channels.pop(key, None)
                    raise

    def get_document_channel(
        self, graph_id: str, doc_id: str, user_id: Optional[str] = None
    ) -> Optional[ChannelState]:
        """Get the channel state for a document."""
        effective_user_id = user_id or self._dev_user_id
        key = f"{effective_user_id}:{graph_id}:{doc_id}"
        return self._document_channels.get(key)

    async def apply_document_update(
        self,
        graph_id: str,
        doc_id: str,
        update: bytes,
        user_id: Optional[str] = None,
    ) -> None:
        """Apply a Y.js update to a document and broadcast to server.

        DEPRECATED: This method has a bug - it double-applies updates when used
        with DocumentWriter methods that already modify the doc. Use
        transact_document() instead for proper incremental update handling.

        Args:
            graph_id: The graph ID
            doc_id: The document ID
            update: The Y.js update bytes
            user_id: The user ID (uses _dev_user_id if not provided)
        """
        import warnings

        warnings.warn(
            "apply_document_update is deprecated. Use transact_document() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        effective_user_id = user_id or self._dev_user_id
        key = f"{effective_user_id}:{graph_id}:{doc_id}"
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
        user_id: Optional[str] = None,
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
            user_id: The user ID (uses _dev_user_id if not provided)

        Example:
            await client.transact_document(graph_id, doc_id, lambda doc:
                DocumentWriter(doc).append_block("<paragraph>Hello</paragraph>")
            , user_id="user-123")
        """
        effective_user_id = user_id or self._dev_user_id
        key = f"{effective_user_id}:{graph_id}:{doc_id}"
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

        # Only send if there are actual changes
        if not incremental_update:
            logger.debug(
                "transact_document: no changes detected (empty diff)",
                extra_context={"graph_id": graph_id, "doc_id": doc_id},
            )
            return

        # Connection must be alive to persist changes
        if not channel.ws or channel.ws.closed:
            logger.error(
                "transact_document: WebSocket closed, cannot persist changes",
                extra_context={
                    "graph_id": graph_id,
                    "doc_id": doc_id,
                    "update_size": len(incremental_update),
                    "has_ws": channel.ws is not None,
                    "ws_closed": channel.ws.closed if channel.ws else True,
                },
            )
            raise RuntimeError(
                f"Cannot persist document changes: WebSocket connection to "
                f"'{doc_id}' is closed. The local document was modified but "
                f"the update was NOT sent to the server. Reconnect and retry."
            )

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

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    async def disconnect_document(
        self, graph_id: str, doc_id: str, user_id: Optional[str] = None
    ) -> None:
        """Disconnect from a document channel."""
        effective_user_id = user_id or self._dev_user_id
        key = f"{effective_user_id}:{graph_id}:{doc_id}"
        channel = self._document_channels.pop(key, None)
        if channel:
            await self._close_channel(channel)

    async def disconnect_workspace(self, graph_id: str, user_id: Optional[str] = None) -> None:
        """Disconnect from a workspace channel."""
        effective_user_id = user_id or self._dev_user_id
        key = f"{effective_user_id}:{graph_id}"
        channel = self._workspace_channels.pop(key, None)
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

    async def _flush_pending_updates(self, timeout: float = 0.5) -> None:
        """Wait briefly to ensure pending WebSocket sends complete.

        Y.js updates are sent asynchronously via WebSocket. This method
        gives pending sends time to complete before closing connections.
        """
        # Give the event loop time to process any pending sends
        # This is a simple approach; a more robust solution would track
        # pending operations explicitly
        await asyncio.sleep(min(timeout, 0.1))

        # Yield to allow any pending callbacks to run
        await asyncio.sleep(0)

    async def close(self) -> None:
        """Close all connections and cleanup.

        Waits briefly for pending updates to be sent before closing.
        """
        self._closed = True

        # Flush pending updates before closing
        await self._flush_pending_updates()

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
