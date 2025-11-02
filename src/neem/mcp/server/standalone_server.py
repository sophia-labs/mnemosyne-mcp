"""
Standalone MCP server that communicates with the API server via HTTP.

This approach provides clean service separation and leverages existing
API authentication patterns without complex ASGI sub-app mounting.

Key design principles:
- Pure MCP protocol handling (like API server is pure HTTP)
- HTTP client tools that call existing API endpoints
- Bearer token passthrough from MCP client to API
- Simple container architecture with service-to-service HTTP
"""

import os
import asyncio
import aiohttp
import uuid
import datetime
import traceback
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP, Context
import json

# Import utilities for logging and error handling
from neem.utils.logging import LoggerFactory
from neem.utils.errors import ServerError, APIError, ValidationError

# Import session manager and response objects
from ..session import MCPSessionManager
from ..response_objects import (
    McpSparqlResponse, McpSchemaResponse, McpGraphListResponse, McpErrorResponse,
    McpCreateGraphResponse, McpDeleteGraphResponse, McpGraphInfoResponse, McpSystemHealthResponse
)
from ..errors import MCPAuthenticationError

# Initialize structured logger
logger = LoggerFactory.get_logger("mcp.standalone_server")


class MCPAPIClient:
    """HTTP client for communicating with Mnemosyne API server."""
    
    def __init__(self, api_base_url: str = "http://mnemosyne-api:8001"):
        self.api_base_url = api_base_url.rstrip('/')
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        headers: Dict[str, str], 
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request to API server."""
        session = await self._get_session()
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            async with session.request(method, url, headers=headers, **kwargs) as response:
                if response.content_type == 'application/json':
                    data = await response.json()
                else:
                    text = await response.text()
                    data = {"response": text}
                
                if response.status >= 400:
                    error_msg = data.get('detail', f'HTTP {response.status}')
                    raise Exception(f"API request failed: {error_msg}")
                
                return data
        except aiohttp.ClientError as e:
            raise Exception(f"HTTP client error: {str(e)}")
    
    async def list_graphs(self, auth_token: str) -> Dict[str, Any]:
        """Call API /graphs endpoint."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        return await self._make_request("GET", "/graphs", headers)
    
    async def query_graph(
        self, 
        graph_id: str, 
        sparql: str, 
        auth_token: str,
        timeout_seconds: int = 10
    ) -> Dict[str, Any]:
        """Call API /graphs/{graph_id}/query endpoint."""
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
        payload = {"sparql": sparql}
        
        endpoint = f"/graphs/{graph_id}/query"
        
        # Add timeout to prevent hanging on invalid queries
        session = await self._get_session()
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with session.request("POST", url, headers=headers, json=payload, timeout=timeout) as response:
                if response.content_type == 'application/json':
                    data = await response.json()
                else:
                    text = await response.text()
                    data = {"response": text}
                
                if response.status >= 400:
                    error_msg = data.get('detail', f'HTTP {response.status}')
                    raise Exception(f"API request failed: {error_msg}")
                
                return data
        except asyncio.TimeoutError:
            raise Exception(f"Query timed out after {timeout_seconds} seconds - check SPARQL syntax and graph existence")
        except aiohttp.ClientError as e:
            raise Exception(f"HTTP client error: {str(e)}")
    
    async def get_graph_schema(self, graph_id: str, auth_token: str) -> Dict[str, Any]:
        """Call API /graphs/{graph_id}/schema endpoint."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        endpoint = f"/graphs/{graph_id}/schema"
        return await self._make_request("GET", endpoint, headers)


def create_session_manager() -> MCPSessionManager:
    """Create in-memory session manager."""
    session_manager = MCPSessionManager()
    logger.info("✅ In-memory session manager created")
    return session_manager


def extract_auth_token(ctx: Context) -> str:
    """Extract authentication token from MCP request context."""
    user_id = None

    # Extract from MCP request headers (HTTP/SSE transport)
    # For stdio transport, this will fail gracefully and fall back to MCP_DEFAULT_TOKEN
    try:
        if ctx and hasattr(ctx, 'request_context') and ctx.request_context:
            request_context = ctx.request_context
            if hasattr(request_context, 'request') and request_context.request:
                request = request_context.request

                # Try to access headers (may not exist in stdio transport)
                if hasattr(request, 'headers') and request.headers:
                    # Try X-User-ID header (development)
                    user_id = request.headers.get("X-User-ID")
                    if user_id:
                        logger.debug(f"Authenticated via X-User-ID header: {user_id}")
                        return user_id

                    # Try Authorization header
                    auth_header = request.headers.get("Authorization")
                    if auth_header and auth_header.startswith("Bearer "):
                        token = auth_header[7:]  # Remove "Bearer " prefix
                        logger.debug(f"Authenticated via Authorization header")
                        return token
    except (AttributeError, TypeError) as e:
        logger.debug(f"Could not extract auth from request context (expected for stdio transport): {e}")

    # Check for saved authentication token from `neem mcp init`
    try:
        from neem.utils.token_storage import load_token
        saved_token = load_token()
        if saved_token:
            logger.info(
                "Using saved authentication token from config file",
                extra_context={"source": "~/.mnemosyne/config.json"}
            )
            return saved_token
    except Exception as e:
        logger.debug(f"Could not load saved token (not an error): {e}")

    # Fallback for development: check environment variable
    fallback_token = os.getenv("MCP_DEFAULT_TOKEN")
    if fallback_token:
        logger.info(
            "Using MCP_DEFAULT_TOKEN environment variable for authentication",
            extra_context={"token": "***masked***"}
        )
        return fallback_token

    raise MCPAuthenticationError(
        "No authentication token provided. Please run 'neem mcp init' to authenticate.",
        context={"headers_present": bool(ctx and getattr(ctx, 'request_context', None))}
    )


def create_standalone_mcp_server() -> FastMCP:
    """
    Create standalone MCP server with HTTP client tools.
    
    This server handles MCP protocol and calls the API server via HTTP,
    following the same clean separation pattern as the existing architecture.
    """
    logger.info("Creating standalone MCP server with HTTP client tools")
    
    # Create FastMCP server in stateless mode to avoid session ID conflicts
    mcp_server = FastMCP(
        name="Mnemosyne Knowledge Graph (Standalone)",
        instructions="AI-native access to knowledge graph data via HTTP API calls",
        stateless_http=True  # Let us handle sessions manually
    )
    
    # Get API base URL from environment
    # Workaround: Claude Code doesn't properly pass env vars from settings,
    # so we override localhost with production API
    api_base_url = os.getenv("MNEMOSYNE_API_URL", "https://api.sophia-labs.com")
    if api_base_url.startswith("http://localhost") or api_base_url.startswith("http://127.0.0.1"):
        api_base_url = "https://api.sophia-labs.com"

    api_client = MCPAPIClient(api_base_url)
    
    # Initialize session manager (avoid naming conflict with FastMCP)
    session_manager = create_session_manager()
    
    async def get_session_context(ctx: Context) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Get session context from MCP request.

        Returns:
            Tuple of (auth_token, session_data) where session_data is None if no valid session
        """
        auth_token = extract_auth_token(ctx)
        user_id = auth_token  # In our system, token is user_id for development

        # Try to get session ID from headers (HTTP/SSE transport)
        # For stdio transport, this will be None
        session_id = None
        try:
            if ctx and hasattr(ctx, 'request_context') and ctx.request_context:
                request = ctx.request_context.request
                if hasattr(request, 'headers') and request.headers:
                    session_id = request.headers.get("X-Session-ID")
        except (AttributeError, TypeError) as e:
            logger.debug(f"Could not extract session ID from headers (expected for stdio): {e}")

        # If we have a session ID, try to validate and get session data
        if session_id:
            if await session_manager.validate_session_for_user(user_id, session_id):
                session_data = await session_manager.get_user_session(user_id, session_id)
                logger.debug(f"Using valid session {session_id} for user {user_id}")
                return auth_token, session_data
            else:
                logger.warning(f"Invalid session {session_id} for user {user_id}")

        # No valid session - return auth token only
        return auth_token, None
    
    @mcp_server.tool()
    async def create_session(
        ctx: Context,
        client_name: str = "Unknown Client"
    ) -> str:
        """
        Create new MCP session for the authenticated user.
        
        Args:
            client_name: Name/type of the MCP client
            
        Returns:
            JSON string with session information
        """
        try:
            # Extract user ID from auth token
            auth_token = extract_auth_token(ctx)
            user_id = auth_token  # In our system, token is user_id for development
            
            logger.info(f"Creating MCP session for user {user_id}")
            
            # Create session with client info
            user_agent = 'Unknown'
            try:
                if ctx and ctx.request_context and ctx.request_context.request and hasattr(ctx.request_context.request, 'headers'):
                    headers = ctx.request_context.request.headers
                    if headers:
                        user_agent = getattr(headers, 'user-agent', 'Unknown')
            except (AttributeError, TypeError):
                pass  # Use default 'Unknown'

            client_info = {
                "client_name": client_name,
                "user_agent": user_agent
            }
            
            session_id = await session_manager.create_session(user_id, client_info)
            
            # Get session data for response
            session_data = await session_manager.get_user_session(user_id, session_id)
            
            result = {
                "success": True,
                "session_id": session_id,
                "user_id": user_id,
                "accessible_graphs": session_data.get("accessible_graphs", []),
                "created_at": session_data.get("created_at"),
                "instructions": {
                    "next_step": "Include session ID in X-Session-ID header for subsequent requests",
                    "session_ttl": "1 hour"
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "error_type": "SESSION_CREATION_FAILED"
            }
            return json.dumps(error_result, indent=2)
    
    @mcp_server.tool()
    async def sparql_query(
        graph_id: str,
        query: str,
        ctx: Context,
        result_format: str = "json",
        timeout_seconds: int = 30
    ) -> str:
        """
        Execute SPARQL query against user's graph via API server.

        Args:
            graph_id: Graph ID to query
            query: SPARQL query to execute
            result_format: Result format (json, csv, xml)
            timeout_seconds: Query timeout in seconds

        Returns:
            JSON string with query results
        """
        try:
            # Get session context (cached or fresh)
            auth_token, session_data = await get_session_context(ctx)
            
            # Validate graph access if we have session data
            if session_data:
                accessible_graphs = session_data.get("accessible_graphs", [])
                if graph_id not in accessible_graphs:
                    error_result = {
                        "success": False,
                        "error": f"Graph '{graph_id}' not accessible to user",
                        "error_type": "GRAPH_ACCESS_DENIED",
                        "accessible_graphs": accessible_graphs
                    }
                    return json.dumps(error_result, indent=2)
                logger.debug(f"Graph '{graph_id}' validated via session cache")
            
            logger.info(f"Executing SPARQL query on graph '{graph_id}' via API")
            
            # Call API server
            api_response = await api_client.query_graph(
                graph_id, query, auth_token, timeout_seconds
            )
            
            # Transform to enhanced MCP response
            mcp_response = McpSparqlResponse.from_api_response(api_response, query, graph_id)
            
            return mcp_response.to_json()
            
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            
            # Determine error type and provide specific help
            error_msg = str(e)
            if "timed out" in error_msg.lower():
                error_type = "QUERY_TIMEOUT"
                help_info = {
                    "graph_id": graph_id,
                    "timeout_seconds": timeout_seconds,
                    "suggestions": [
                        "Check SPARQL syntax - invalid syntax can cause timeouts",
                        "Verify graph exists and is accessible",
                        "Try adding LIMIT clause to reduce result set",
                        "Simplify query by removing complex joins or filters"
                    ],
                    "common_timeout_causes": [
                        "Missing object in WHERE clause (e.g., '{ ?s ?p }' should be '{ ?s ?p ?o }')",
                        "Querying nonexistent graph",
                        "Invalid SPARQL syntax",
                        "Overly complex query without constraints"
                    ]
                }
            else:
                error_type = "SPARQL_QUERY_FAILED"
                help_info = {
                    "graph_id": graph_id,
                    "suggestions": [
                        "Check SPARQL syntax for missing brackets or semicolons",
                        "Verify graph access permissions",
                        "Try a simpler query to test connectivity"
                    ],
                    "common_fixes": [
                        "Ensure all variables start with ?",
                        "Check that PREFIX declarations are correct",
                        "Verify bracket matching: { }"
                    ]
                }
            
            error_response = McpErrorResponse.from_exception(e, error_type=error_type, help_info=help_info)
            return error_response.to_json()
    
    @mcp_server.tool()
    async def list_graphs(
        ctx: Context,
        include_stats: bool = True,
        include_metadata: bool = True
    ) -> str:
        """
        List all graphs accessible to the user via API server.
        
        Args:
            include_stats: Include graph statistics
            include_metadata: Include graph metadata
            
        Returns:
            JSON string with graph list
        """
        try:
            # Get session context (cached or fresh)
            auth_token, session_data = await get_session_context(ctx)
            
            # If we have valid session data with cached graphs, use it for basic list
            if session_data and not include_stats and not include_metadata:
                accessible_graphs = session_data.get("accessible_graphs", [])
                logger.info(f"Using cached graph list from session ({len(accessible_graphs)} graphs)")
                
                # Use cached response format
                mcp_response = McpGraphListResponse.from_cached_graphs(accessible_graphs)
                return mcp_response.to_json()
            
            # Need fresh data from API for stats/metadata or no session
            logger.info("Fetching fresh graph list via API")
            
            # Call API server
            api_response = await api_client.list_graphs(auth_token)
            
            # Transform to enhanced MCP format
            mcp_response = McpGraphListResponse.from_api_response(api_response, source="api_fresh")
            
            return mcp_response.to_json()
            
        except Exception as e:
            logger.error(f"List graphs failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            error_response = McpErrorResponse.from_exception(
                e,
                error_type="GRAPH_LIST_FAILED",
                help_info={
                    "suggestions": [
                        "Check API connectivity",
                        "Verify user authentication",
                        "Try again in a moment"
                    ]
                }
            )
            return error_response.to_json()
    
    @mcp_server.tool()
    async def get_graph_schema(
        graph_id: str,
        ctx: Context
    ) -> str:
        """
        Get schema information for a specific graph via API server.
        
        Args:
            graph_id: Graph ID to analyze
            
        Returns:
            JSON string with schema information
        """
        try:
            # Get session context (cached or fresh)
            auth_token, session_data = await get_session_context(ctx)
            
            # Validate graph access if we have session data
            if session_data:
                accessible_graphs = session_data.get("accessible_graphs", [])
                if graph_id not in accessible_graphs:
                    error_response = McpErrorResponse(
                        error_type="GRAPH_ACCESS_DENIED",
                        error_message=f"Graph '{graph_id}' not accessible to user",
                        help={
                            "accessible_graphs": accessible_graphs,
                            "suggestions": ["Use list_graphs to see available graphs"]
                        }
                    )
                    return error_response.to_json()
                logger.debug(f"Graph '{graph_id}' validated via session cache")
            
            logger.info(f"Getting schema for graph '{graph_id}' via API")
            
            # Call API server
            api_response = await api_client.get_graph_schema(graph_id, auth_token)
            
            # Transform to enhanced MCP format
            mcp_response = McpSchemaResponse.from_api_response(api_response, graph_id)
            
            return mcp_response.to_json()
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            error_response = McpErrorResponse.from_exception(
                e,
                error_type="SCHEMA_ANALYSIS_FAILED", 
                help_info={
                    "graph_id": graph_id,
                    "suggestions": [
                        "Check graph exists with list_graphs tool",
                        "Verify graph access permissions"
                    ]
                }
            )
            return error_response.to_json()
    
    # Phase 1: New MCP Tools - Graph Management and System Operations
    
    @mcp_server.tool()
    async def create_graph(
        graph_id: str,
        name: str,
        ctx: Context,
        description: str = "",
        is_persistent: bool = True,
        max_triples: int = 0
    ) -> str:
        """
        Create a new graph with specified configuration.
        
        Args:
            graph_id: Unique identifier for the graph
            name: Human-readable name for the graph
            description: Optional description of the graph's purpose
            is_persistent: Whether the graph persists across sessions
            max_triples: Optional limit on graph size
            ctx: MCP context for user authentication
        
        Returns:
            JSON response with graph creation details and next steps
        """
        operation_id = str(uuid.uuid4())
        
        with logger.operation_context("create_graph", operation_id=operation_id, graph_id=graph_id):
            try:
                # Get session context with structured logging
                auth_token, session_data = await get_session_context(ctx)
                user_id = auth_token  # In development, token is user_id
                
                logger.info(
                    "Creating new graph",
                    extra_context={
                        "user_id": user_id,
                        "graph_id": graph_id,
                        "name": name,
                        "is_persistent": is_persistent
                    }
                )
                
                # Prepare API request payload
                payload = {
                    "name": name,
                    "description": description,
                    "is_persistent": is_persistent
                }
                if max_triples > 0:
                    payload["max_triples"] = max_triples
                
                # Call API server
                api_response = await api_client._make_request(
                    "POST",
                    f"/graphs/{graph_id}",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    json=payload
                )
                
                logger.info(
                    "Graph created successfully",
                    extra_context={
                        "operation_id": operation_id,
                        "graph_id": graph_id,
                        "user_id": user_id
                    }
                )
                
                # Transform to enhanced MCP format
                mcp_response = McpCreateGraphResponse.from_api_response(
                    api_response, 
                    operation_id=operation_id,
                    user_id=user_id
                )
                
                return mcp_response.to_json()
                
            except Exception as e:
                logger.error(
                    "Graph creation failed",
                    extra_context={
                        "operation_id": operation_id,
                        "graph_id": graph_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                
                error_response = McpErrorResponse.from_exception(
                    e,
                    error_type="GRAPH_CREATION_FAILED",
                    help_info={
                        "operation_id": operation_id,
                        "graph_id": graph_id,
                        "suggestions": [
                            "Check that graph_id is unique and valid",
                            "Verify user permissions for graph creation",
                            "Ensure name is not empty"
                        ]
                    }
                )
                return error_response.to_json()
    
    @mcp_server.tool()
    async def delete_graph(
        graph_id: str,
        ctx: Context,
        confirm: bool = False,
        backup: bool = True
    ) -> str:
        """
        Delete a graph after confirmation.
        
        Args:
            graph_id: Graph to delete
            confirm: Confirmation flag (required for safety)
            backup: Whether to create backup before deletion
            ctx: MCP context for user authentication
        
        Returns:
            JSON response with deletion confirmation and backup info
        """
        operation_id = str(uuid.uuid4())
        
        with logger.operation_context("delete_graph", operation_id=operation_id, graph_id=graph_id):
            try:
                # Safety check - require explicit confirmation
                if not confirm:
                    error_response = McpErrorResponse(
                        error_type="CONFIRMATION_REQUIRED",
                        error_message="Graph deletion requires explicit confirmation",
                        help={
                            "required_parameter": "confirm=true",
                            "warning": "This operation is irreversible",
                            "suggestion": "Set confirm=true to proceed with deletion"
                        }
                    )
                    return error_response.to_json()
                
                # Get session context
                auth_token, session_data = await get_session_context(ctx)
                user_id = auth_token
                
                # Validate graph access
                if session_data:
                    accessible_graphs = session_data.get("accessible_graphs", [])
                    if graph_id not in accessible_graphs:
                        logger.warning(
                            "Graph access denied for deletion",
                            extra_context={
                                "user_id": user_id,
                                "graph_id": graph_id,
                                "accessible_graphs": accessible_graphs
                            }
                        )
                        error_response = McpErrorResponse(
                            error_type="GRAPH_ACCESS_DENIED",
                            error_message=f"Access denied for graph {graph_id}",
                            help={
                                "graph_id": graph_id,
                                "suggestion": "Use list_graphs to see accessible graphs"
                            }
                        )
                        return error_response.to_json()
                
                logger.info(
                    "Deleting graph",
                    extra_context={
                        "operation_id": operation_id,
                        "user_id": user_id,
                        "graph_id": graph_id,
                        "backup": backup
                    }
                )
                
                # Call API server
                api_response = await api_client._make_request(
                    "DELETE",
                    f"/graphs/{graph_id}",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    params={"backup": backup}
                )
                
                logger.info(
                    "Graph deleted successfully",
                    extra_context={
                        "operation_id": operation_id,
                        "graph_id": graph_id,
                        "user_id": user_id
                    }
                )
                
                # Transform to enhanced MCP format
                mcp_response = McpDeleteGraphResponse.from_api_response(
                    api_response,
                    operation_id=operation_id,
                    graph_id=graph_id,
                    user_id=user_id
                )
                
                return mcp_response.to_json()
                
            except Exception as e:
                logger.error(
                    "Graph deletion failed",
                    extra_context={
                        "operation_id": operation_id,
                        "graph_id": graph_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                
                error_response = McpErrorResponse.from_exception(
                    e,
                    error_type="GRAPH_DELETION_FAILED",
                    help_info={
                        "operation_id": operation_id,
                        "graph_id": graph_id,
                        "suggestions": [
                            "Check that graph exists and is accessible",
                            "Verify deletion permissions",
                            "Ensure graph is not currently in use"
                        ]
                    }
                )
                return error_response.to_json()
    
    @mcp_server.tool()
    async def get_graph_info(
        graph_id: str,
        ctx: Context,
        include_stats: bool = True,
        include_schema_preview: bool = False
    ) -> str:
        """
        Get comprehensive graph information.
        
        Args:
            graph_id: Graph to analyze
            include_stats: Include detailed statistics
            include_schema_preview: Include schema summary
            ctx: MCP context for user authentication
        
        Returns:
            JSON response with complete graph details
        """
        operation_id = str(uuid.uuid4())
        
        with logger.operation_context("get_graph_info", operation_id=operation_id, graph_id=graph_id):
            try:
                # Get session context
                auth_token, session_data = await get_session_context(ctx)
                user_id = auth_token
                
                # Validate graph access
                if session_data:
                    accessible_graphs = session_data.get("accessible_graphs", [])
                    if graph_id not in accessible_graphs:
                        logger.warning(
                            "Graph access denied for info request",
                            extra_context={
                                "user_id": user_id,
                                "graph_id": graph_id
                            }
                        )
                        error_response = McpErrorResponse(
                            error_type="GRAPH_ACCESS_DENIED",
                            error_message=f"Access denied for graph {graph_id}",
                            help={
                                "graph_id": graph_id,
                                "suggestion": "Use list_graphs to see accessible graphs"
                            }
                        )
                        return error_response.to_json()
                
                logger.info(
                    "Getting graph information",
                    extra_context={
                        "operation_id": operation_id,
                        "user_id": user_id,
                        "graph_id": graph_id,
                        "include_stats": include_stats,
                        "include_schema_preview": include_schema_preview
                    }
                )
                
                # Call API server
                params = {}
                if include_stats:
                    params["include_stats"] = "true"
                if include_schema_preview:
                    params["include_schema_preview"] = "true"
                
                api_response = await api_client._make_request(
                    "GET",
                    f"/graphs/{graph_id}",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    params=params
                )
                
                logger.info(
                    "Graph information retrieved successfully",
                    extra_context={
                        "operation_id": operation_id,
                        "graph_id": graph_id,
                        "user_id": user_id
                    }
                )
                
                # Transform to enhanced MCP format
                mcp_response = McpGraphInfoResponse.from_api_response(
                    api_response,
                    operation_id=operation_id,
                    graph_id=graph_id,
                    user_id=user_id
                )
                
                return mcp_response.to_json()
                
            except Exception as e:
                logger.error(
                    "Graph info retrieval failed",
                    extra_context={
                        "operation_id": operation_id,
                        "graph_id": graph_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                
                error_response = McpErrorResponse.from_exception(
                    e,
                    error_type="GRAPH_INFO_FAILED",
                    help_info={
                        "operation_id": operation_id,
                        "graph_id": graph_id,
                        "suggestions": [
                            "Check that graph exists with list_graphs tool",
                            "Verify graph access permissions",
                            "Try with include_stats=false for basic info"
                        ]
                    }
                )
                return error_response.to_json()

    @mcp_server.tool()
    async def upload_file_to_graph(
        graph_id: str,
        file_path: str,
        ctx: Context,
        rdf_format: str = "",
        validation_level: str = "strict",
        namespace: str = "",
        replace_existing: bool = False
    ) -> str:
        """
        Upload RDF file to graph with validation and progress tracking.

        Args:
            graph_id: Target graph ID
            file_path: Path to the RDF file to upload (Turtle, RDF/XML, N-Triples, JSON-LD)
            rdf_format: Optional RDF format override (turtle, rdfxml, ntriples, jsonld)
            validation_level: Validation level (strict, lenient, none)
            namespace: Optional namespace for the uploaded data
            replace_existing: Whether to replace existing data (default: append)
            ctx: MCP context for user authentication

        Returns:
            JSON response with upload job details and progress tracking URL
        """
        operation_id = str(uuid.uuid4())

        with logger.operation_context("upload_file_to_graph", operation_id=operation_id, graph_id=graph_id):
            try:
                # Get session context
                auth_token, session_data = await get_session_context(ctx)
                user_id = auth_token

                # Validate graph access if we have session data
                if session_data:
                    accessible_graphs = session_data.get("accessible_graphs", [])
                    if accessible_graphs and graph_id not in accessible_graphs:
                        logger.warning(
                            "Graph access denied for upload",
                            extra_context={
                                "user_id": user_id,
                                "graph_id": graph_id
                            }
                        )
                        error_response = McpErrorResponse(
                            error_type="GRAPH_ACCESS_DENIED",
                            error_message=f"Access denied for graph {graph_id}",
                            help={
                                "graph_id": graph_id,
                                "suggestion": "Use list_graphs to see accessible graphs"
                            }
                        )
                        return error_response.to_json()

                logger.info(
                    "Uploading file to graph",
                    extra_context={
                        "operation_id": operation_id,
                        "user_id": user_id,
                        "graph_id": graph_id,
                        "file_path": file_path,
                        "rdf_format": rdf_format,
                        "validation_level": validation_level
                    }
                )

                # Read file content
                import os
                if not os.path.exists(file_path):
                    error_response = McpErrorResponse(
                        error_type="FILE_NOT_FOUND",
                        error_message=f"File not found: {file_path}",
                        help={
                            "file_path": file_path,
                            "suggestion": "Verify the file path is correct and accessible"
                        }
                    )
                    return error_response.to_json()

                file_size = os.path.getsize(file_path)
                filename = os.path.basename(file_path)

                # Prepare multipart form data
                with open(file_path, 'rb') as f:
                    file_content = f.read()

                # Create form data
                form_data = aiohttp.FormData()
                form_data.add_field('file', file_content, filename=filename, content_type='application/octet-stream')

                if rdf_format and rdf_format.strip():
                    form_data.add_field('format', rdf_format)
                form_data.add_field('validation', validation_level)
                if namespace and namespace.strip():
                    form_data.add_field('namespace', namespace)
                form_data.add_field('replace_existing', str(replace_existing).lower())

                # Upload via API
                session = await api_client._get_session()
                url = f"{api_client.api_base_url}/graphs/{graph_id}/upload"
                headers = {"Authorization": f"Bearer {auth_token}"}

                async with session.post(url, data=form_data, headers=headers) as response:
                    if response.content_type == 'application/json':
                        data = await response.json()
                    else:
                        text = await response.text()
                        data = {"response": text}

                    if response.status >= 400:
                        error_msg = data.get('detail', f'HTTP {response.status}')
                        raise Exception(f"Upload failed: {error_msg}")

                    logger.info(
                        "File uploaded successfully",
                        extra_context={
                            "operation_id": operation_id,
                            "graph_id": graph_id,
                            "user_id": user_id,
                            "filename": filename,
                            "file_size": file_size
                        }
                    )

                    # Format response
                    upload_data = data.get("data", {})
                    result = {
                        "success": True,
                        "operation_id": operation_id,
                        "job_id": upload_data.get("job_id"),
                        "filename": upload_data.get("filename"),
                        "file_size_bytes": upload_data.get("file_size_bytes"),
                        "detected_format": upload_data.get("detected_format"),
                        "estimated_triples": upload_data.get("estimated_triples"),
                        "status": upload_data.get("status", "pending"),
                        "message": upload_data.get("message", "Upload initiated"),
                        "progress_url": upload_data.get("progress_url"),
                        "next_steps": {
                            "monitor": f"Use the job_id to check progress",
                            "job_id": upload_data.get("job_id")
                        }
                    }

                    return json.dumps(result, indent=2)

            except Exception as e:
                logger.error(
                    "File upload failed",
                    extra_context={
                        "operation_id": operation_id,
                        "graph_id": graph_id,
                        "file_path": file_path,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )

                error_response = McpErrorResponse.from_exception(
                    e,
                    error_type="FILE_UPLOAD_FAILED",
                    help_info={
                        "operation_id": operation_id,
                        "graph_id": graph_id,
                        "file_path": file_path,
                        "suggestions": [
                            "Verify the file is a valid RDF format (Turtle, RDF/XML, N-Triples, JSON-LD)",
                            "Check file permissions and path",
                            "Ensure graph exists and is accessible",
                            "Try with validation='lenient' for less strict parsing"
                        ]
                    }
                )
                return error_response.to_json()

    @mcp_server.resource(
        "resource://mnemosyne/overview",
        name="Mnemosyne MCP Overview",
        title="Mnemosyne MCP Quickstart",
        description="Entry point that explains how to work with the Mnemosyne knowledge graph tools.",
        mime_type="application/json"
    )
    def mnemosyne_overview_resource() -> dict:
        """Expose a small discoverable note for MCP clients."""
        return {
            "summary": "This MCP server connects Claude or compatible clients to Mnemosyne knowledge graphs.",
            "quickstart": [
                "Run the list_graphs tool to see accessible graphs and pick an ID.",
                "Call get_graph_schema <graph_id> to inspect classes and predicates.",
                "Use sparql_query <graph_id> <SPARQL> for data exploration; start with the sample queries from list_graphs."
            ],
            "authentication": "Authenticate via `neem mcp init` or set MCP_DEFAULT_TOKEN before starting the server.",
            "support": "Provide LOG_LEVEL=DEBUG to see detailed traces if requests fail."
        }

    # Store components for cleanup
    mcp_server._api_client = api_client
    mcp_server._session_manager = session_manager
    
    logger.info("✅ Standalone MCP server created with HTTP client tools and in-memory session management")
    logger.info(f"API base URL: {api_base_url}")
    logger.info("Available tools: create_session, sparql_query, list_graphs, get_graph_schema, create_graph, delete_graph, get_graph_info, get_system_health, upload_file_to_graph")
    
    return mcp_server




def run_standalone_mcp_server_sync():
    """Run the standalone MCP server (synchronous entry point)."""
    # Get configuration from environment
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8003"))
    
    logger.info(f"🚀 Starting standalone MCP server on {host}:{port}")
    
    # Create standalone MCP server
    mcp_server = create_standalone_mcp_server()
    
    try:
        logger.info("🔌 Starting MCP server with HTTP transport")
        
        # Use uvicorn with FastMCP's streamable HTTP app for container deployment
        import uvicorn
        http_app = mcp_server.streamable_http_app()
        uvicorn.run(http_app, host=host, port=port)
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        # Cleanup HTTP client if possible
        try:
            if hasattr(mcp_server, '_api_client'):
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(mcp_server._api_client.close())
                else:
                    asyncio.run(mcp_server._api_client.close())
        except:
            pass  # Best effort cleanup
        logger.info("✅ Standalone MCP server shutdown complete")




if __name__ == "__main__":
    # Run the standalone server synchronously
    run_standalone_mcp_server_sync()
