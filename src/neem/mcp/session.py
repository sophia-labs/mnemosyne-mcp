"""
MCP session management using lightweight in-memory storage.

Provides user-scoped session management for MCP connections without requiring
an external Redis service. Sessions are tracked in-process with TTL handling.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

from neem.utils.logging import LoggerFactory
from .errors import MCPSessionError, MCPAuthenticationError

logger = LoggerFactory.get_logger("mcp.session")


class MCPSessionManager:
    """
    Manages MCP sessions using in-memory storage with TTL-based cleanup.

    Integrates with the existing user-scoped container system to provide
    isolated MCP sessions per user with proper authentication and cleanup,
    without relying on an external Redis dependency.
    """

    def __init__(self, session_ttl_seconds: int = 3600):
        """
        Initialize MCP session manager.

        Args:
            session_ttl_seconds: Session TTL in seconds (default: 1 hour)
        """
        self.session_ttl = session_ttl_seconds
        self.session_prefix = "mcp:session"
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._expirations: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    def _session_key(self, user_id: str, session_id: str) -> str:
        return f"{self.session_prefix}:{user_id}:{session_id}"

    async def _cleanup_expired_locked(self) -> int:
        """Remove expired sessions. Caller must hold the lock."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, expires_at in self._expirations.items() if expires_at <= now
        ]

        for key in expired_keys:
            self._sessions.pop(key, None)
            self._expirations.pop(key, None)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired MCP sessions")
        return len(expired_keys)

    def _copy_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy to avoid exposing internal state."""
        return dict(session_data)

    async def create_session(
        self,
        user_id: str,
        client_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new MCP session for user.
        
        Args:
            user_id: User identifier
            client_info: Optional client information
            
        Returns:
            Session ID
            
        Raises:
            MCPSessionError: If session creation fails
        """
        try:
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            session_key = f"{self.session_prefix}:{user_id}:{session_id}"

            # Note: In standalone mode, accessible_graphs will be populated by API queries
            # when needed, rather than storing them in session data
            accessible_graphs = []
            
            # Create session data
            session_data = {
                "user_id": user_id,
                "session_id": session_id,
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "accessible_graphs": accessible_graphs,
                "client_info": client_info or {},
                "status": "active"
            }
            
            expires_at = datetime.utcnow() + timedelta(seconds=self.session_ttl)

            async with self._lock:
                self._sessions[session_key] = session_data
                self._expirations[session_key] = expires_at
            
            logger.info(f"Created MCP session {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create MCP session for user {user_id}: {e}")
            raise MCPSessionError(
                f"Session creation failed: {str(e)}",
                context={"user_id": user_id}
            )
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        try:
            async with self._lock:
                await self._cleanup_expired_locked()

                suffix = f":{session_id}"
                for key, session_data in self._sessions.items():
                    if key.endswith(suffix):
                        return self._copy_session(session_data)

            return None
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def get_user_session(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data for specific user and session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        try:
            session_key = self._session_key(user_id, session_id)

            async with self._lock:
                await self._cleanup_expired_locked()
                session_data = self._sessions.get(session_key)
                if not session_data:
                    return None

                return self._copy_session(session_data)
            
        except Exception as e:
            logger.error(f"Failed to get user session {user_id}:{session_id}: {e}")
            return None
    
    async def update_session_activity(self, session_id: str) -> bool:
        """
        Update last activity timestamp for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if updated successfully
        """
        try:
            async with self._lock:
                await self._cleanup_expired_locked()

                suffix = f":{session_id}"
                for key, session_data in self._sessions.items():
                    if key.endswith(suffix):
                        session_data["last_activity"] = datetime.utcnow().isoformat()
                        self._expirations[key] = datetime.utcnow() + timedelta(
                            seconds=self.session_ttl
                        )
                        return True

            return False
            
        except Exception as e:
            logger.error(f"Failed to update session activity {session_id}: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete MCP session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            async with self._lock:
                await self._cleanup_expired_locked()

                suffix = f":{session_id}"
                for key in list(self._sessions.keys()):
                    if key.endswith(suffix):
                        self._sessions.pop(key, None)
                        self._expirations.pop(key, None)
                        logger.info(f"Deleted MCP session {session_id}")
                        return True

            return False
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions (Redis TTL should handle this automatically).
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            async with self._lock:
                cleaned_count = await self._cleanup_expired_locked()
                return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    async def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all active sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session data
        """
        try:
            async with self._lock:
                await self._cleanup_expired_locked()

                prefix = f"{self.session_prefix}:{user_id}:"
                sessions = [
                    self._copy_session(data)
                    for key, data in self._sessions.items()
                    if key.startswith(prefix)
                ]

                return sessions
            
        except Exception as e:
            logger.error(f"Failed to list sessions for user {user_id}: {e}")
            return []
    
    async def validate_session_for_user(self, user_id: str, session_id: str) -> bool:
        """
        Validate that session belongs to user and is active.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            True if session is valid for user
        """
        async with self._lock:
            await self._cleanup_expired_locked()

            session_key = self._session_key(user_id, session_id)
            session_data = self._sessions.get(session_key)

            if not session_data:
                return False

            # Verify user_id matches (defensive)
            if session_data.get("user_id") != user_id:
                return False

            # Verify session is active
            if session_data.get("status") != "active":
                return False

            # Update activity and extend TTL
            session_data["last_activity"] = datetime.utcnow().isoformat()
            self._expirations[session_key] = datetime.utcnow() + timedelta(
                seconds=self.session_ttl
            )

            return True
