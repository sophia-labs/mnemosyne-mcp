"""
MCP Response Objects - Formal transforms of core Mnemosyne objects for MCP presentation.

This module provides structured response objects that transform raw API responses
into LLM-friendly, well-formatted MCP tool responses with enhanced metadata,
insights, and user guidance.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
import re


class McpResponseObject:
    """Base class for MCP-formatted responses"""
    
    def to_json(self) -> str:
        """Convert response object to formatted JSON string"""
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_api_response(cls, api_response: Dict[str, Any], **kwargs) -> 'McpResponseObject':
        """Transform raw API response to MCP format"""
        raise NotImplementedError


@dataclass 
class McpErrorResponse(McpResponseObject):
    """Structured error response with helpful guidance"""
    error_type: str
    error_message: str
    help: Optional[Dict[str, Any]] = None
    success: bool = False
    
    @classmethod
    def from_exception(cls, e: Exception, error_type: str = "UNKNOWN_ERROR", help_info: Optional[Dict[str, Any]] = None) -> 'McpErrorResponse':
        """Create error response from exception"""
        return cls(
            error_type=error_type,
            error_message=str(e),
            help=help_info or {}
        )


# Task 1.1: Enhanced SPARQL Query Results

@dataclass
class McpQuerySummary:
    """Summary information about the SPARQL query execution"""
    result_count: int
    query_time_ms: int
    graph_id: str
    result_type: str  # "tabular", "graph", "boolean", "empty"
    query_complexity: str  # "simple", "moderate", "complex"


@dataclass
class McpQueryResults:
    """Formatted query results for LLM consumption"""
    formatted_table: str
    raw_bindings: List[Dict[str, Any]]
    sample_results: List[Dict[str, Any]]


@dataclass
class McpQueryInsights:
    """Analytical insights about the query results"""
    result_type: str
    has_more_data: bool
    common_variables: List[str]
    data_patterns: List[str]
    optimization_hints: List[str]


@dataclass
class McpSparqlResponse(McpResponseObject):
    """Enhanced SPARQL query response with formatting and insights"""
    query_summary: McpQuerySummary
    results: McpQueryResults
    insights: McpQueryInsights
    success: bool = True
    
    @classmethod
    def from_api_response(cls, api_response: Dict[str, Any], query: str, graph_id: str) -> 'McpSparqlResponse':
        """Transform API SPARQL response to enhanced MCP format"""
        # API returns nested structure: {"success": true, "data": {"results": [...], "count": N}}
        data = api_response.get("data", {})
        raw_results = data.get("results", [])
        
        # Analyze query complexity
        query_complexity = _analyze_query_complexity(query)
        
        # Build response components
        query_summary = McpQuerySummary(
            result_count=len(raw_results),
            query_time_ms=data.get("query_time_ms", 0),
            graph_id=graph_id,
            result_type=_detect_result_type(raw_results),
            query_complexity=query_complexity
        )
        
        results = McpQueryResults(
            formatted_table=_format_as_markdown_table(raw_results),
            raw_bindings=raw_results,
            sample_results=raw_results[:5] if raw_results else []
        )
        
        insights = McpQueryInsights(
            result_type=query_summary.result_type,
            has_more_data=len(raw_results) >= 100,
            common_variables=_extract_common_variables(raw_results),
            data_patterns=_analyze_data_patterns(raw_results),
            optimization_hints=_generate_optimization_hints(query, raw_results)
        )
        
        return cls(
            query_summary=query_summary,
            results=results,
            insights=insights
        )


# Helper functions for SPARQL response transformation

def _detect_result_type(results: List[Dict[str, Any]]) -> str:
    """Detect the type of SPARQL results"""
    if not results:
        return "empty"
    
    if len(results) == 1 and len(results[0]) == 1:
        # Might be ASK query result
        first_value = list(results[0].values())[0]
        if isinstance(first_value, bool):
            return "boolean"
    
    # Check if results look like a table (multiple variables)
    if results and len(results[0]) > 1:
        return "tabular"
    
    return "graph"


def _format_as_markdown_table(results: List[Dict[str, Any]]) -> str:
    """Format SPARQL results as a markdown table"""
    if not results:
        return "**No results found**"
    
    if len(results) > 20:
        # Truncate for readability
        display_results = results[:20]
        truncated_note = f"\n\n*Showing first 20 of {len(results)} results*"
    else:
        display_results = results
        truncated_note = ""
    
    # Get headers from first result
    headers = list(display_results[0].keys())
    
    # Build markdown table
    table_lines = []
    
    # Header row
    header_row = "| " + " | ".join(headers) + " |"
    table_lines.append(header_row)
    
    # Separator row
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    table_lines.append(separator)
    
    # Data rows
    for result in display_results:
        row_values = []
        for header in headers:
            value = result.get(header, "")
            # Clean up value for table display
            cleaned_value = _clean_value_for_table(value)
            row_values.append(cleaned_value)
        
        row = "| " + " | ".join(row_values) + " |"
        table_lines.append(row)
    
    return "\n".join(table_lines) + truncated_note


def _clean_value_for_table(value: Any) -> str:
    """Clean and format a value for markdown table display"""
    if value is None:
        return ""
    
    str_value = str(value)
    
    # Truncate very long values
    if len(str_value) > 50:
        str_value = str_value[:47] + "..."
    
    # Escape pipe characters that would break markdown table
    str_value = str_value.replace("|", "\\|")
    
    # Clean up URIs to be more readable
    if str_value.startswith("http://") or str_value.startswith("https://"):
        # Extract local name from URI
        local_name = str_value.split("/")[-1].split("#")[-1]
        if local_name and local_name != str_value:
            str_value = f"{local_name} ({str_value})"
    
    return str_value


def _extract_common_variables(results: List[Dict[str, Any]]) -> List[str]:
    """Extract variables that appear in most results"""
    if not results:
        return []
    
    # Count variable frequency
    var_counts = {}
    total_results = len(results)
    
    for result in results:
        for var in result.keys():
            var_counts[var] = var_counts.get(var, 0) + 1
    
    # Return variables that appear in at least 80% of results
    common_vars = [
        var for var, count in var_counts.items() 
        if count >= (total_results * 0.8)
    ]
    
    return sorted(common_vars)


def _analyze_data_patterns(results: List[Dict[str, Any]]) -> List[str]:
    """Analyze patterns in the result data"""
    patterns = []
    
    if not results:
        return ["No data to analyze"]
    
    # Check for common data patterns
    first_result = results[0]
    
    # Pattern: All results have certain properties
    common_vars = _extract_common_variables(results)
    if common_vars:
        patterns.append(f"All results contain: {', '.join(common_vars)}")
    
    # Pattern: URI vs Literal analysis
    uri_vars = []
    literal_vars = []
    
    for var in first_result.keys():
        sample_values = [r.get(var) for r in results[:5] if r.get(var)]
        if sample_values:
            sample_str = str(sample_values[0])
            if sample_str.startswith("http://") or sample_str.startswith("https://"):
                uri_vars.append(var)
            else:
                literal_vars.append(var)
    
    if uri_vars:
        patterns.append(f"URI/Resource variables: {', '.join(uri_vars)}")
    if literal_vars:
        patterns.append(f"Literal/Value variables: {', '.join(literal_vars)}")
    
    # Pattern: Result diversity
    if len(results) > 1:
        patterns.append(f"Found {len(results)} distinct results")
    
    return patterns[:5]  # Limit to top 5 patterns


def _analyze_query_complexity(query: str) -> str:
    """Analyze SPARQL query complexity"""
    query_lower = query.lower()
    
    complexity_indicators = {
        'complex': ['union', 'optional', 'filter', 'group by', 'order by', 'having'],
        'moderate': ['limit', 'distinct', 'count', 'join']
    }
    
    # Count complex features
    complex_features = sum(1 for feature in complexity_indicators['complex'] if feature in query_lower)
    moderate_features = sum(1 for feature in complexity_indicators['moderate'] if feature in query_lower)
    
    if complex_features >= 2:
        return "complex"
    elif complex_features >= 1 or moderate_features >= 2:
        return "moderate"
    else:
        return "simple"


def _generate_optimization_hints(query: str, results: List[Dict[str, Any]]) -> List[str]:
    """Generate query optimization hints based on query and results"""
    hints = []
    query_lower = query.lower()
    
    # Large result set without LIMIT
    if len(results) >= 100 and 'limit' not in query_lower:
        hints.append("Consider adding LIMIT clause to reduce result set size")
    
    # No specific constraints
    if not any(keyword in query_lower for keyword in ['filter', 'limit', 'distinct']):
        hints.append("Add FILTER clauses to narrow down results")
    
    # Suggest DISTINCT if many duplicate-looking results
    if len(results) > 10:
        # Simple heuristic: if first few results have similar structure
        hints.append("Consider using DISTINCT if you're seeing duplicate results")
    
    # Complex query suggestions
    if 'union' in query_lower:
        hints.append("UNION queries can be slow - consider if separate queries might be more efficient")
    
    # Performance based on execution time (would need API response data)
    return hints[:3]  # Limit to top 3 hints


# Task 1.2: Rich Graph Schema Output

@dataclass
class McpSchemaSummary:
    """Core schema statistics"""
    total_classes: int
    total_properties: int
    primary_namespaces: List[str]


@dataclass
class McpSchemaClass:
    """Enhanced class information"""
    uri: str
    label: str
    description: Optional[str]
    instance_count: Optional[int]
    sample_query: str


@dataclass
class McpSchemaResponse(McpResponseObject):
    """Enhanced schema response"""
    graph_id: str
    schema_summary: McpSchemaSummary
    classes: List[McpSchemaClass]
    query_templates: Dict[str, str]
    success: bool = True
    
    @classmethod
    def from_api_response(cls, api_response: Dict[str, Any], graph_id: str) -> 'McpSchemaResponse':
        """Transform API schema response to enhanced MCP format"""
        schema_data = api_response.get("data", {})
        
        # Transform classes with simple enhancements
        classes = []
        for class_data in schema_data.get("classes", []):
            classes.append(McpSchemaClass(
                uri=class_data.get("uri", ""),
                label=class_data.get("label", "") or _extract_local_name(class_data.get("uri", "")),
                description=class_data.get("description"),
                instance_count=class_data.get("instance_count"),
                sample_query=_generate_sample_query(class_data.get("uri", ""))
            ))
        
        return cls(
            graph_id=graph_id,
            schema_summary=McpSchemaSummary(
                total_classes=len(classes),
                total_properties=len(schema_data.get("properties", [])),
                primary_namespaces=_extract_namespaces(schema_data)
            ),
            classes=classes,
            query_templates={
                "list_instances": "SELECT ?instance ?label WHERE { ?instance a <CLASS_URI> . OPTIONAL { ?instance rdfs:label ?label } } LIMIT 20",
                "explore_properties": "SELECT ?property ?value WHERE { <INSTANCE_URI> ?property ?value }",
                "count_by_type": "SELECT ?type (COUNT(?instance) as ?count) WHERE { ?instance a ?type } GROUP BY ?type ORDER BY DESC(?count)"
            }
        )


# Helper functions for schema response transformation

def _generate_sample_query(class_uri: str) -> str:
    """Generate one useful sample query"""
    if not class_uri:
        return "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
    
    local_name = _extract_local_name(class_uri)
    return f"SELECT ?{local_name.lower()} WHERE {{ ?{local_name.lower()} a <{class_uri}> }} LIMIT 10"


def _extract_local_name(uri: str) -> str:
    """Extract local name from URI"""
    if not uri:
        return "item"
    
    if "#" in uri:
        return uri.split("#")[-1]
    elif "/" in uri:
        return uri.split("/")[-1]
    else:
        return uri


def _extract_namespaces(schema_data: Dict[str, Any]) -> List[str]:
    """Extract primary namespaces from schema data"""
    namespaces = set()
    
    # From classes
    for class_data in schema_data.get("classes", []):
        uri = class_data.get("uri", "")
        if uri:
            if "#" in uri:
                namespaces.add(uri.split("#")[0] + "#")
            elif "/" in uri:
                parts = uri.split("/")
                if len(parts) > 1:
                    namespaces.add("/".join(parts[:-1]) + "/")
    
    # From properties  
    for prop_data in schema_data.get("properties", []):
        uri = prop_data.get("uri", "")
        if uri:
            if "#" in uri:
                namespaces.add(uri.split("#")[0] + "#")
            elif "/" in uri:
                parts = uri.split("/")
                if len(parts) > 1:
                    namespaces.add("/".join(parts[:-1]) + "/")
    
    return list(namespaces)[:5]  # Top 5 namespaces


# Task 1.3: Improved Graph List Display

@dataclass
class McpGraphStats:
    """Graph statistics"""
    triple_count: int
    size_mb: float
    last_updated: Optional[str]


@dataclass
class McpGraphInfo:
    """Enhanced graph information"""
    id: str
    name: str
    description: str
    stats: McpGraphStats
    sample_queries: List[str]


@dataclass
class McpGraphListResponse(McpResponseObject):
    """Enhanced graph list response"""
    graphs: List[McpGraphInfo]
    total_count: int
    source: str  # "session_cache" | "api_fresh"
    success: bool = True

    @classmethod
    def from_api_response(cls, api_response: Dict[str, Any], source: str = "api_fresh") -> 'McpGraphListResponse':
        """Transform API graph list response to enhanced MCP format"""
        graphs_data = api_response.get("data", [])

        graphs = []
        for graph_data in graphs_data:
            graph_id = graph_data.get("graph_id", graph_data.get("id", ""))

            graphs.append(McpGraphInfo(
                id=graph_id,
                name=graph_data.get("name", "Unnamed Graph"),
                description=graph_data.get("description", "No description available"),
                stats=McpGraphStats(
                    triple_count=graph_data.get("triple_count", 0),
                    size_mb=graph_data.get("size_mb", 0.0),
                    last_updated=graph_data.get("updated_at") or graph_data.get("last_modified")
                ),
                sample_queries=_generate_sample_queries_for_graph(graph_data, graph_id)
            ))

        return cls(
            graphs=graphs,
            total_count=len(graphs),
            source=source
        )

    @classmethod
    def from_cached_graphs(cls, graph_ids: List[str]) -> 'McpGraphListResponse':
        """Create simplified response from cached session data"""
        graphs = []
        for graph_id in graph_ids:
            graphs.append(McpGraphInfo(
                id=graph_id,
                name=f"Graph: {graph_id}",
                description="Cached graph info - use include_stats=true for full details",
                stats=McpGraphStats(
                    triple_count=0,
                    size_mb=0.0,
                    last_updated=None
                ),
                sample_queries=["# Use include_stats=true to get sample queries"]
            ))

        return cls(
            graphs=graphs,
            total_count=len(graphs),
            source="session_cache"
        )


# Phase 1: New Response Objects for Graph Management and System Operations

@dataclass
class McpOperationSummary:
    """Summary of an MCP operation"""
    operation_type: str
    operation_id: str
    timestamp: str
    duration_ms: Optional[int] = None
    user_id: Optional[str] = None


@dataclass
class McpCreateGraphResponse(McpResponseObject):
    """Response for graph creation operations"""
    operation_summary: McpOperationSummary
    graph_info: Dict[str, Any]
    next_steps: List[str]
    sample_queries: List[str]
    success: bool = True
    
    @classmethod
    def from_api_response(cls, api_response: Dict[str, Any], operation_id: str, user_id: Optional[str] = None) -> 'McpCreateGraphResponse':
        """Transform API graph creation response to MCP format"""
        import datetime
        
        graph_data = api_response.get("data", {})
        graph_id = graph_data.get("graph_id", graph_data.get("id", ""))
        
        return cls(
            operation_summary=McpOperationSummary(
                operation_type="create_graph",
                operation_id=operation_id,
                timestamp=datetime.datetime.now().isoformat(),
                user_id=user_id
            ),
            graph_info={
                "id": graph_id,
                "name": graph_data.get("name", ""),
                "description": graph_data.get("description", ""),
                "created_at": graph_data.get("created_at"),
                "is_persistent": graph_data.get("is_persistent", True),
                "max_triples": graph_data.get("max_triples")
            },
            next_steps=[
                f"Upload RDF data using upload_rdf_file tool",
                f"Query the graph using sparql_query tool with graph_id='{graph_id}'",
                f"Explore the schema using get_graph_schema tool"
            ],
            sample_queries=[
                f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT 10",
                f"SELECT (COUNT(*) as ?count) WHERE {{ ?s ?p ?o }}",
                f"SELECT DISTINCT ?type WHERE {{ ?s a ?type }}"
            ]
        )


@dataclass
class McpDeleteGraphResponse(McpResponseObject):
    """Response for graph deletion operations"""
    operation_summary: McpOperationSummary
    deletion_info: Dict[str, Any]
    backup_info: Optional[Dict[str, Any]]
    cleanup_results: Dict[str, Any]
    success: bool = True
    
    @classmethod
    def from_api_response(cls, api_response: Dict[str, Any], operation_id: str, graph_id: str, user_id: Optional[str] = None) -> 'McpDeleteGraphResponse':
        """Transform API graph deletion response to MCP format"""
        import datetime
        
        return cls(
            operation_summary=McpOperationSummary(
                operation_type="delete_graph",
                operation_id=operation_id,
                timestamp=datetime.datetime.now().isoformat(),
                user_id=user_id
            ),
            deletion_info={
                "graph_id": graph_id,
                "deleted_at": datetime.datetime.now().isoformat(),
                "confirmed": True
            },
            backup_info=api_response.get("backup_info"),
            cleanup_results={
                "files_removed": api_response.get("files_removed", 0),
                "space_freed_mb": api_response.get("space_freed_mb", 0),
                "cache_cleared": True
            }
        )


@dataclass
class McpGraphInfoResponse(McpResponseObject):
    """Response for detailed graph information"""
    operation_summary: McpOperationSummary
    graph_info: Dict[str, Any]
    statistics: Dict[str, Any]
    schema_preview: Optional[Dict[str, Any]]
    recommendations: List[str]
    success: bool = True
    
    @classmethod
    def from_api_response(cls, api_response: Dict[str, Any], operation_id: str, graph_id: str, user_id: Optional[str] = None) -> 'McpGraphInfoResponse':
        """Transform API graph info response to MCP format"""
        import datetime
        
        graph_data = api_response.get("data", {})
        
        # Generate recommendations based on graph state
        recommendations = []
        triple_count = graph_data.get("triple_count", 0)
        if triple_count == 0:
            recommendations.append("Graph is empty - consider uploading RDF data")
        elif triple_count < 100:
            recommendations.append("Small graph - suitable for experimentation and learning")
        elif triple_count > 10000:
            recommendations.append("Large graph - consider using LIMIT in queries for better performance")
        
        return cls(
            operation_summary=McpOperationSummary(
                operation_type="get_graph_info",
                operation_id=operation_id,
                timestamp=datetime.datetime.now().isoformat(),
                user_id=user_id
            ),
            graph_info={
                "id": graph_id,
                "name": graph_data.get("name", ""),
                "description": graph_data.get("description", ""),
                "created_at": graph_data.get("created_at"),
                "updated_at": graph_data.get("updated_at"),
                "is_persistent": graph_data.get("is_persistent", True)
            },
            statistics={
                "triple_count": triple_count,
                "size_mb": graph_data.get("size_mb", 0.0),
                "unique_subjects": graph_data.get("unique_subjects", 0),
                "unique_predicates": graph_data.get("unique_predicates", 0),
                "unique_objects": graph_data.get("unique_objects", 0)
            },
            schema_preview=graph_data.get("schema_preview"),
            recommendations=recommendations
        )


@dataclass
class McpSystemHealthResponse(McpResponseObject):
    """Response for system health check"""
    operation_summary: McpOperationSummary
    health_status: Dict[str, Any]
    components: List[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    recommendations: List[str]
    success: bool = True
    
    @classmethod
    def from_api_response(cls, api_response: Dict[str, Any], operation_id: str, user_id: Optional[str] = None) -> 'McpSystemHealthResponse':
        """Transform API health response to MCP format"""
        import datetime
        
        health_data = api_response.get("data", {})
        overall_status = health_data.get("status", "unknown")
        
        # Generate recommendations based on health status
        recommendations = []
        if overall_status != "healthy":
            recommendations.append("System health issues detected - check component status")
        if health_data.get("uptime_seconds", 0) < 300:  # Less than 5 minutes
            recommendations.append("System recently restarted - allow time for full initialization")
        
        return cls(
            operation_summary=McpOperationSummary(
                operation_type="get_system_health",
                operation_id=operation_id,
                timestamp=datetime.datetime.now().isoformat(),
                user_id=user_id
            ),
            health_status={
                "overall_status": overall_status,
                "uptime_seconds": health_data.get("uptime_seconds", 0),
                "version": health_data.get("version", "unknown"),
                "environment": health_data.get("environment", "unknown")
            },
            components=health_data.get("components", []),
            metrics=health_data.get("metrics"),
            recommendations=recommendations
        )


# Helper functions for graph list transformation

def _generate_sample_queries_for_graph(graph_data: Dict[str, Any], graph_id: str) -> List[str]:
    """Generate sample queries tailored to the graph"""
    queries = [
        "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10",
        "SELECT ?type (COUNT(?s) as ?count) WHERE { ?s a ?type } GROUP BY ?type ORDER BY DESC(?count)"
    ]
    
    # Add graph-specific queries based on available data
    triple_count = graph_data.get("triple_count", 0)
    
    if triple_count > 0:
        queries.append("SELECT DISTINCT ?p WHERE { ?s ?p ?o } LIMIT 20")
        queries.append("SELECT ?s WHERE { ?s a ?type } LIMIT 5")
    else:
        queries = ["# No data in graph yet - try adding some triples first"]
    
    return queries[:3]  # Limit to 3 sample queries