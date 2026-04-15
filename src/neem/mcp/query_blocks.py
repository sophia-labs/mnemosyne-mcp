"""Query block helpers shared by MCP tools.

These functions mirror the frontend query-block model closely enough that
agents can insert, inspect, and preview query blocks without hand-authoring
raw TipTap XML or duplicating result-shaping logic ad hoc.
"""

from __future__ import annotations

import html as html_mod
import json
import re
from typing import Any, Dict, Optional

JsonDict = Dict[str, Any]

DEFAULT_QUERY_BLOCK_MAX_ROWS = 100
MAX_QUERY_BLOCK_ROWS = 500
QUERY_BLOCK_VISUALIZATIONS = ("table", "stat", "triples", "vega", "network", "json")
QUERY_BLOCK_DISPLAY_MODES = ("auto", "manual", "agent")
UPDATE_KEYWORD_PATTERN = re.compile(
    r"\b(INSERT|DELETE|LOAD|CLEAR|CREATE|DROP|COPY|MOVE|ADD)\b",
    re.IGNORECASE,
)


def _escape_xml_attr(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\t", "&#9;")
        .replace("\n", "&#10;")
        .replace("\r", "&#13;")
    )


def clamp_max_rows(value: Any) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return DEFAULT_QUERY_BLOCK_MAX_ROWS
    return max(1, min(MAX_QUERY_BLOCK_ROWS, numeric))


def get_query_block_default_attrs() -> JsonDict:
    return {
        "comment": "",
        "query": "SELECT ?s ?p ?o WHERE {\n  ?s ?p ?o .\n}\nLIMIT 25",
        "displayMode": "auto",
        "visualization": "table",
        "maxRows": DEFAULT_QUERY_BLOCK_MAX_ROWS,
        "vegaLiteSpec": "",
        "collapsed": False,
    }


def normalize_query_block_attrs(attrs: Optional[Dict[str, Any]] = None) -> JsonDict:
    defaults = get_query_block_default_attrs()
    attrs = attrs or {}

    visualization = attrs.get("visualization", defaults["visualization"])
    if visualization == "bar":
        visualization = "vega"
    if visualization not in QUERY_BLOCK_VISUALIZATIONS:
        visualization = defaults["visualization"]

    display_mode = attrs.get("displayMode", defaults["displayMode"])
    if display_mode not in QUERY_BLOCK_DISPLAY_MODES:
        display_mode = defaults["displayMode"]

    return {
        "comment": attrs.get("comment") if isinstance(attrs.get("comment"), str) else defaults["comment"],
        "query": attrs.get("query") if isinstance(attrs.get("query"), str) else defaults["query"],
        "displayMode": display_mode,
        "visualization": visualization,
        "maxRows": clamp_max_rows(attrs.get("maxRows", defaults["maxRows"])),
        "vegaLiteSpec": attrs.get("vegaLiteSpec") if isinstance(attrs.get("vegaLiteSpec"), str) else defaults["vegaLiteSpec"],
        "collapsed": attrs.get("collapsed") is True or attrs.get("collapsed") == "true",
    }


def build_query_block_xml(attrs: Optional[Dict[str, Any]] = None, *, block_id: Optional[str] = None) -> str:
    normalized = normalize_query_block_attrs(attrs)
    parts = ["<queryBlock"]
    if block_id:
        parts.append(f' data-block-id="{_escape_xml_attr(block_id)}"')
    for key in ("comment", "query", "displayMode", "visualization", "maxRows", "vegaLiteSpec", "collapsed"):
        value = normalized[key]
        if isinstance(value, bool):
            value = "true" if value else "false"
        parts.append(f' {key}="{_escape_xml_attr(value)}"')
    parts.append("/>")
    return "".join(parts)


def strip_leading_sparql_declarations(sparql: str) -> str:
    remaining = sparql.lstrip()
    while remaining:
        if remaining.startswith("#"):
            newline_index = remaining.find("\n")
            remaining = remaining[newline_index + 1:].lstrip() if newline_index >= 0 else ""
            continue
        prefix_match = re.match(r"^PREFIX\s+[A-Za-z][\w-]*:\s*<[^>]+>\s*", remaining, re.IGNORECASE)
        if prefix_match:
            remaining = remaining[prefix_match.end():].lstrip()
            continue
        base_match = re.match(r"^BASE\s+<[^>]+>\s*", remaining, re.IGNORECASE)
        if base_match:
            remaining = remaining[base_match.end():].lstrip()
            continue
        break
    return remaining


def infer_query_block_query_kind(sparql: str) -> str:
    normalized = strip_leading_sparql_declarations(sparql)
    keyword_match = re.match(r"^(ASK|SELECT|CONSTRUCT|DESCRIBE)\b", normalized, re.IGNORECASE)
    keyword = keyword_match.group(1).lower() if keyword_match else None
    if not keyword or UPDATE_KEYWORD_PATTERN.search(normalized):
        raise ValueError("Only read-only ASK, SELECT, CONSTRUCT, and DESCRIBE queries are allowed")
    return keyword


def preferred_query_block_result_format(query_kind: str) -> str:
    return "nq" if query_kind in {"construct", "describe"} else "json"


def normalize_binding_term(term: Optional[Dict[str, Any]]) -> JsonDict:
    term = term or {}
    language = None
    if isinstance(term.get("xml:lang"), str):
        language = term["xml:lang"]
    elif isinstance(term.get("language"), str):
        language = term["language"]
    result: JsonDict = {
        "type": term["type"] if isinstance(term.get("type"), str) else "literal",
        "value": term["value"] if isinstance(term.get("value"), str) else "",
    }
    if isinstance(term.get("datatype"), str):
        result["datatype"] = term["datatype"]
    if language:
        result["language"] = language
    return result


def extract_sparql_json_payload(payload: Any) -> Optional[JsonDict]:
    if not isinstance(payload, dict):
        return None
    if any(key in payload for key in ("head", "results", "boolean")):
        return payload
    nested = payload.get("data")
    if isinstance(nested, dict) and any(key in nested for key in ("head", "results", "boolean")):
        return nested
    return None


def normalize_query_result(payload: Any, query_kind: str, duration_ms: int, media_type: str) -> JsonDict:
    if query_kind == "ask":
        data = extract_sparql_json_payload(payload)
        boolean_value = bool(data.get("boolean") if data else False)
        return {
            "queryKind": query_kind,
            "resultKind": "ask",
            "boolean": boolean_value,
            "columns": ["result"],
            "rows": [{"result": {"type": "literal", "value": str(boolean_value).lower()}}],
            "durationMs": duration_ms,
            "raw": payload,
        }

    if query_kind == "select":
        data = extract_sparql_json_payload(payload)
        raw_rows = []
        if isinstance(data, dict):
            results = data.get("results", {})
            if isinstance(results, dict) and isinstance(results.get("bindings"), list):
                raw_rows = results["bindings"]
        elif isinstance(payload, list):
            raw_rows = payload

        columns: list[str]
        if isinstance(data, dict):
            head = data.get("head", {})
            vars_value = head.get("vars") if isinstance(head, dict) else None
            if isinstance(vars_value, list):
                columns = [value for value in vars_value if isinstance(value, str)]
            else:
                columns = sorted({key for row in raw_rows if isinstance(row, dict) for key in row.keys()})
        else:
            columns = sorted({key for row in raw_rows if isinstance(row, dict) for key in row.keys()})

        rows = []
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            normalized_row = {
                key: normalize_binding_term(value if isinstance(value, dict) else None)
                for key, value in row.items()
            }
            rows.append(normalized_row)

        return {
            "queryKind": query_kind,
            "resultKind": "bindings",
            "columns": columns,
            "rows": rows,
            "durationMs": duration_ms,
            "raw": payload,
        }

    value = payload if isinstance(payload, str) else json.dumps(payload if payload is not None else {"value": ""}, indent=2)
    return {
        "queryKind": query_kind,
        "resultKind": "serialized",
        "mediaType": media_type,
        "value": value,
        "durationMs": duration_ms,
        "raw": payload,
    }


def infer_term_value_kind(term: Optional[Dict[str, Any]]) -> str:
    if not term or not term.get("value"):
        return "empty"
    if term.get("type") == "uri":
        return "uri"

    datatype = str(term.get("datatype", "")).lower()
    if any(marker in datatype for marker in ("#integer", "#decimal", "#double", "#float", "#int", "#long", "#short")):
        return "number"
    if "#boolean" in datatype:
        return "boolean"
    if "#date" in datatype or "#time" in datatype:
        return "date"

    value = str(term.get("value", "")).strip().lower()
    if value in {"true", "false"}:
        return "boolean"
    if re.match(r"^-?\d+(\.\d+)?$", value):
        return "number"
    if any(marker in value for marker in ("-", "/", ":", "t")):
        return "date" if _looks_like_date(value) else "literal"
    return "literal"


def _looks_like_date(value: str) -> bool:
    return bool(
        re.match(r"^\d{4}-\d{2}-\d{2}", value)
        or re.match(r"^\d{4}/\d{2}/\d{2}", value)
        or re.match(r"^\d{4}-\d{2}-\d{2}t", value)
    )


def build_column_profiles(result: JsonDict) -> list[JsonDict]:
    profiles: list[JsonDict] = []
    for name in result.get("columns", []):
        seen_kinds = set()
        seen_values = set()
        non_null_count = 0
        for row in result.get("rows", []):
            if not isinstance(row, dict):
                continue
            term = row.get(name)
            if not isinstance(term, dict) or not term.get("value"):
                continue
            non_null_count += 1
            seen_kinds.add(infer_term_value_kind(term))
            seen_values.add(f"{term.get('type')}:{term.get('value')}:{term.get('datatype', '')}:{term.get('language', '')}")

        value_kind = "empty"
        if len(seen_kinds) == 1:
            value_kind = next(iter(seen_kinds))
        elif len(seen_kinds) > 1:
            value_kind = "mixed"

        profiles.append({
            "name": name,
            "valueKind": value_kind,
            "nonNullCount": non_null_count,
            "distinctCount": len(seen_values),
        })
    return profiles


def ranked_candidates(*candidates: JsonDict) -> list[JsonDict]:
    return sorted(candidates, key=lambda item: item["confidence"], reverse=True)


def looks_like_triple_columns(columns: list[str]) -> bool:
    normalized = [column.lower() for column in columns]
    return any(
        all(column in normalized for column in candidate)
        for candidate in (("s", "p", "o"), ("subject", "predicate", "object"))
    )


def looks_like_edge_columns(columns: list[str]) -> bool:
    normalized = [column.lower() for column in columns]
    return any(
        all(column in normalized for column in candidate)
        for candidate in (("source", "target"), ("from", "to"), ("src", "dst"))
    )


def pick_bar_columns(profile: JsonDict) -> Optional[JsonDict]:
    columns = profile.get("columns", [])
    if len(columns) < 2:
        return None

    value_column = next((column for column in columns if column.get("valueKind") == "number"), None)
    if value_column is None:
        return None

    label_column = next(
        (
            column for column in columns
            if column.get("name") != value_column.get("name")
            and column.get("valueKind") in {"literal", "uri", "date", "mixed"}
        ),
        None,
    )
    if label_column is None:
        return None

    return {
        "labelColumn": label_column["name"],
        "valueColumn": value_column["name"],
    }


def profile_query_block_result(result: JsonDict) -> JsonDict:
    if result.get("resultKind") == "ask":
        return {
            "shape": "boolean",
            "rowCount": 1,
            "columns": [{"name": "result", "valueKind": "boolean", "nonNullCount": 1, "distinctCount": 1}],
            "candidates": ranked_candidates(
                {"kind": "stat", "confidence": 0.99, "reason": "Boolean results are best shown as a single answer."},
                {"kind": "table", "confidence": 0.6, "reason": "A one-row table is a reasonable fallback."},
                {"kind": "json", "confidence": 0.35, "reason": "Raw payload remains available for inspection."},
            ),
        }

    if result.get("resultKind") == "serialized":
        media_type = str(result.get("mediaType", "")).lower()
        is_triples = "n-quads" in media_type or media_type.endswith("/nq") or ".nq" in media_type
        value = result.get("value", "")
        row_count = len([line for line in str(value).splitlines() if line.strip()])
        return {
            "shape": "triples" if is_triples else "serialized",
            "rowCount": row_count,
            "columns": [],
            "candidates": ranked_candidates(
                {"kind": "triples", "confidence": 0.97, "reason": "Serialized RDF quads map naturally to a triples view."},
                {"kind": "json", "confidence": 0.7, "reason": "Raw structured output remains a useful fallback."},
            ) if is_triples else ranked_candidates(
                {"kind": "json", "confidence": 0.98, "reason": "Serialized outputs should be shown as raw structured data."},
            ),
        }

    columns = build_column_profiles(result)
    row_count = len(result.get("rows", []))
    if looks_like_triple_columns(result.get("columns", [])):
        return {
            "shape": "triples",
            "rowCount": row_count,
            "columns": columns,
            "candidates": ranked_candidates(
                {"kind": "triples", "confidence": 0.94, "reason": "Rows match subject/predicate/object columns."},
                {"kind": "network", "confidence": 0.78, "reason": "Triple-shaped rows can also be rendered as a graph."},
                {"kind": "table", "confidence": 0.82, "reason": "Tabular rendering still fits triple-like rows."},
                {"kind": "json", "confidence": 0.4, "reason": "Raw JSON is available for debugging."},
            ),
        }

    if row_count == 1 and len(result.get("columns", [])) == 1:
        return {
            "shape": "stat",
            "rowCount": row_count,
            "columns": columns,
            "candidates": ranked_candidates(
                {"kind": "stat", "confidence": 0.96, "reason": "A single binding is best shown as a single value."},
                {"kind": "table", "confidence": 0.7, "reason": "A single-row table is a reasonable fallback."},
                {"kind": "json", "confidence": 0.4, "reason": "Raw JSON is available for debugging."},
            ),
        }

    bar_columns = pick_bar_columns({"columns": columns})
    if bar_columns:
        return {
            "shape": "rows",
            "rowCount": row_count,
            "columns": columns,
            "candidates": ranked_candidates(
                {"kind": "vega", "confidence": 0.92, "reason": f"{bar_columns['labelColumn']} vs {bar_columns['valueColumn']} auto-charted as horizontal bar."},
                {"kind": "table", "confidence": 0.88, "reason": "Multi-row bindings can also be shown as a table."},
                {"kind": "json", "confidence": 0.42, "reason": "Raw JSON remains available for debugging."},
            ),
        }

    numeric_columns = [column for column in columns if column.get("valueKind") == "number"]
    if len(numeric_columns) >= 2:
        return {
            "shape": "rows",
            "rowCount": row_count,
            "columns": columns,
            "candidates": ranked_candidates(
                {"kind": "vega", "confidence": 0.90, "reason": f"{numeric_columns[0]['name']} vs {numeric_columns[1]['name']} auto-charted as scatter."},
                {"kind": "table", "confidence": 0.88, "reason": "Multi-row bindings can also be shown as a table."},
                {"kind": "json", "confidence": 0.42, "reason": "Raw JSON remains available for debugging."},
            ),
        }

    date_column = next((column for column in columns if column.get("valueKind") == "date"), None)
    numeric_column = numeric_columns[0] if numeric_columns else None
    if date_column and numeric_column:
        return {
            "shape": "rows",
            "rowCount": row_count,
            "columns": columns,
            "candidates": ranked_candidates(
                {"kind": "vega", "confidence": 0.91, "reason": f"{date_column['name']} vs {numeric_column['name']} auto-charted as line."},
                {"kind": "table", "confidence": 0.88, "reason": "Multi-row bindings can also be shown as a table."},
                {"kind": "json", "confidence": 0.42, "reason": "Raw JSON remains available for debugging."},
            ),
        }

    if looks_like_edge_columns(result.get("columns", [])):
        return {
            "shape": "rows",
            "rowCount": row_count,
            "columns": columns,
            "candidates": ranked_candidates(
                {"kind": "table", "confidence": 0.93, "reason": "Edge-like bindings still read well in a table."},
                {"kind": "network", "confidence": 0.72, "reason": "Source/target rows can be rendered as a network."},
                {"kind": "json", "confidence": 0.42, "reason": "Raw JSON remains available for debugging."},
            ),
        }

    return {
        "shape": "rows",
        "rowCount": row_count,
        "columns": columns,
        "candidates": ranked_candidates(
            {"kind": "table", "confidence": 0.95, "reason": "Multi-row bindings are best shown as a table."},
            {"kind": "json", "confidence": 0.42, "reason": "Raw JSON remains available for debugging."},
        ),
    }


def format_query_block_term(term: Optional[Dict[str, Any]]) -> str:
    if not term:
        return ""
    if term.get("type") == "uri":
        return str(term.get("value", ""))
    if term.get("type") == "bnode":
        return f"_:{term.get('value', '')}"
    if term.get("language"):
        return f"\"{term.get('value', '')}\"@{term['language']}"
    if term.get("datatype"):
        return f"\"{term.get('value', '')}\"^^{term['datatype']}"
    return str(term.get("value", ""))


def extract_query_block_stat_value(result: JsonDict, profile: Optional[JsonDict] = None) -> Optional[JsonDict]:
    profile = profile or profile_query_block_result(result)
    if result.get("resultKind") == "ask":
        return {
            "label": "Answer",
            "value": "True" if result.get("boolean") else "False",
            "supportingText": str(result.get("queryKind", "")).upper(),
        }

    if result.get("resultKind") != "bindings" or profile.get("shape") != "stat":
        return None

    columns = result.get("columns", [])
    rows = result.get("rows", [])
    if not columns or not rows:
        return None

    column = columns[0]
    value = format_query_block_term(rows[0].get(column) if isinstance(rows[0], dict) else None)
    if not value:
        return None
    return {
        "label": column,
        "value": value,
        "supportingText": str(result.get("queryKind", "")).upper(),
    }


def _format_serialized_triple_term(term: str) -> str:
    if term.startswith("<") and term.endswith(">"):
        return term[1:-1]
    return term


def extract_query_block_triples(result: JsonDict, profile: Optional[JsonDict] = None) -> list[JsonDict]:
    profile = profile or profile_query_block_result(result)
    if profile.get("shape") != "triples":
        return []

    if result.get("resultKind") == "bindings":
        columns = {column.lower(): column for column in result.get("columns", [])}
        subject_key = columns.get("subject") or columns.get("s")
        predicate_key = columns.get("predicate") or columns.get("p")
        object_key = columns.get("object") or columns.get("o")
        graph_key = columns.get("graph") or columns.get("g")
        if not subject_key or not predicate_key or not object_key:
            return []
        triples = []
        for row in result.get("rows", []):
            if not isinstance(row, dict):
                continue
            triple = {
                "subject": format_query_block_term(row.get(subject_key)),
                "predicate": format_query_block_term(row.get(predicate_key)),
                "object": format_query_block_term(row.get(object_key)),
            }
            if graph_key:
                triple["graph"] = format_query_block_term(row.get(graph_key))
            triples.append(triple)
        return triples

    if result.get("resultKind") != "serialized":
        return []

    triples = []
    term_pattern = re.compile(r'<[^>]*>|_:[^\s]+|"([^"\\]|\\.)*"(?:@[A-Za-z0-9-]+|\^\^<[^>]+>)?')
    for line in str(result.get("value", "")).splitlines():
        line = line.strip()
        if not line:
            continue
        matches = [match.group(0) for match in term_pattern.finditer(line)]
        if len(matches) < 3:
            continue
        triple = {
            "subject": _format_serialized_triple_term(matches[0]),
            "predicate": _format_serialized_triple_term(matches[1]),
            "object": _format_serialized_triple_term(matches[2]),
        }
        if len(matches) >= 4:
            triple["graph"] = _format_serialized_triple_term(matches[3])
        triples.append(triple)
    return triples


def extract_query_block_network_data(result: JsonDict, profile: Optional[JsonDict] = None) -> JsonDict:
    profile = profile or profile_query_block_result(result)
    node_map: dict[str, JsonDict] = {}
    edges: list[JsonDict] = []

    def ensure_node(node_id: str, label: str, kind: str) -> None:
        if node_id and node_id not in node_map:
            node_map[node_id] = {"id": node_id, "label": label, "kind": kind}

    if profile.get("shape") == "triples":
        for index, triple in enumerate(extract_query_block_triples(result, profile)):
            subject = triple["subject"]
            obj = triple["object"]
            ensure_node(subject, subject, "entity")
            ensure_node(obj, obj, "entity" if obj.startswith("http") else "literal")
            edges.append({
                "id": f"edge-{index}-{subject}-{triple['predicate']}-{obj}",
                "source": subject,
                "target": obj,
                "label": triple["predicate"],
            })
        return {"nodes": list(node_map.values()), "edges": edges}

    if result.get("resultKind") != "bindings":
        return {"nodes": [], "edges": []}

    columns = {column.lower(): column for column in result.get("columns", [])}
    source_key = columns.get("source") or columns.get("from") or columns.get("src")
    target_key = columns.get("target") or columns.get("to") or columns.get("dst")
    label_key = columns.get("label") or columns.get("predicate") or columns.get("edge")
    if not source_key or not target_key:
        return {"nodes": [], "edges": []}

    for index, row in enumerate(result.get("rows", [])):
        if not isinstance(row, dict):
            continue
        source = format_query_block_term(row.get(source_key))
        target = format_query_block_term(row.get(target_key))
        if not source or not target:
            continue
        ensure_node(source, source, "entity" if source.startswith("http") else "literal")
        ensure_node(target, target, "entity" if target.startswith("http") else "literal")
        edges.append({
            "id": f"edge-{index}-{source}-{target}",
            "source": source,
            "target": target,
            "label": format_query_block_term(row.get(label_key)) if label_key else "",
        })

    return {"nodes": list(node_map.values()), "edges": edges}


def supports_query_block_visualization(visualization: str, result: JsonDict, profile: Optional[JsonDict] = None) -> bool:
    profile = profile or profile_query_block_result(result)
    if visualization == "json":
        return True
    if visualization == "table":
        return result.get("resultKind") in {"ask", "bindings"}
    if visualization == "stat":
        return profile.get("shape") in {"boolean", "stat"}
    if visualization == "triples":
        return profile.get("shape") == "triples"
    if visualization == "vega":
        return result.get("resultKind") == "bindings"
    if visualization == "network":
        return len(extract_query_block_network_data(result, profile)["edges"]) > 0
    return False


def choose_auto_query_block_visualization(profile: JsonDict) -> str:
    candidates = profile.get("candidates", [])
    return candidates[0]["kind"] if candidates else "json"


def resolve_query_block_display(
    requested_visualization: str,
    result: JsonDict,
    *,
    display_mode: Optional[str] = None,
    profile: Optional[JsonDict] = None,
) -> JsonDict:
    profile = profile or profile_query_block_result(result)
    is_override = display_mode in {"manual", "agent"}
    requested = requested_visualization if is_override else choose_auto_query_block_visualization(profile)

    if supports_query_block_visualization(requested, result, profile):
        return {"kind": requested, "source": "override" if is_override else "auto"}

    fallback = choose_auto_query_block_visualization(profile)
    if supports_query_block_visualization(fallback, result, profile):
        return {
            "kind": fallback,
            "source": "fallback",
            "notice": (
                f"{requested.capitalize()} view is not available for "
                f"{str(result.get('queryKind', '')).upper()} results yet. "
                f"Showing {fallback.capitalize()} instead."
            ),
        }

    return {
        "kind": "json",
        "source": "fallback",
        "notice": f"Showing JSON because no richer renderer is available for {str(result.get('queryKind', '')).upper()} results.",
    }


def available_query_block_visualizations(result: JsonDict, profile: Optional[JsonDict] = None) -> list[str]:
    profile = profile or profile_query_block_result(result)
    return [
        visualization
        for visualization in QUERY_BLOCK_VISUALIZATIONS
        if supports_query_block_visualization(visualization, result, profile)
    ]


def generate_auto_vega_spec(result: JsonDict, profile: Optional[JsonDict] = None) -> Optional[str]:
    profile = profile or profile_query_block_result(result)
    if result.get("resultKind") != "bindings" or not result.get("rows"):
        return None

    bar_columns = pick_bar_columns(profile)
    if bar_columns:
        spec = {
            "mark": {"type": "bar", "cornerRadiusEnd": 4, "tooltip": True},
            "encoding": {
                "y": {
                    "field": bar_columns["labelColumn"],
                    "type": "nominal",
                    "sort": "-x",
                    "axis": {"title": bar_columns["labelColumn"], "labelLimit": 220},
                },
                "x": {
                    "field": bar_columns["valueColumn"],
                    "type": "quantitative",
                    "axis": {"title": bar_columns["valueColumn"]},
                },
                "color": {"value": "#39725a"},
            },
            "height": max(180, min(520, len(result.get("rows", [])) * 28)),
        }
        return json.dumps(spec, indent=2)

    numeric_columns = [column for column in profile.get("columns", []) if column.get("valueKind") == "number"]
    if len(numeric_columns) >= 2:
        label_column = next(
            (column for column in profile.get("columns", []) if column.get("valueKind") in {"literal", "uri"}),
            None,
        )
        spec: JsonDict = {
            "mark": {"type": "circle", "opacity": 0.8, "tooltip": True},
            "encoding": {
                "x": {"field": numeric_columns[0]["name"], "type": "quantitative"},
                "y": {"field": numeric_columns[1]["name"], "type": "quantitative"},
            },
            "height": 300,
        }
        if label_column:
            spec["encoding"]["color"] = {"field": label_column["name"], "type": "nominal"}
        return json.dumps(spec, indent=2)

    date_column = next((column for column in profile.get("columns", []) if column.get("valueKind") == "date"), None)
    numeric_column = numeric_columns[0] if numeric_columns else None
    if date_column and numeric_column:
        spec = {
            "mark": {"type": "line", "tooltip": True},
            "encoding": {
                "x": {"field": date_column["name"], "type": "temporal"},
                "y": {"field": numeric_column["name"], "type": "quantitative"},
            },
            "height": 300,
        }
        return json.dumps(spec, indent=2)

    return None

