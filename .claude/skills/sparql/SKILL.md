---
name: sparql
description: SPARQL reference for Mnemosyne knowledge graphs. Load when writing or debugging SPARQL queries against the RDF store — namespace prefixes, entity URI patterns, predicate catalog, common query patterns, and gotchas.
---

# Mnemosyne SPARQL Reference

## Namespace Prefixes

Two separate vocabularies. Getting these wrong is the #1 SPARQL mistake.

| Prefix | URI | Domain |
|--------|-----|--------|
| `doc:` | `http://mnemosyne.dev/doc#` | Document structure, blocks, valuations |
| `mnemo:` | `http://mnemosyne.ai/vocab#` | Graph metadata, wires |
| `dcterms:` | `http://purl.org/dc/terms/` | Titles, descriptions, timestamps |
| `nfo:` | `http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#` | File metadata (fileName, belongsToContainer) |
| `rdf:` | `http://www.w3.org/1999/02/22-rdf-syntax-ns#` | Type assertions |
| `rdfs:` | `http://www.w3.org/2000/01/rdf-schema#` | Labels, subclass |
| `nie:` | `http://www.semanticdesktop.org/ontologies/2007/01/19/nie#` | Content metadata (mimeType) |
| `xsd:` | `http://www.w3.org/2001/XMLSchema#` | Data types (dateTime, float, integer) |

**CRITICAL:** Never use `urn:mnemosyne:schema:doc:` — it matches nothing. The correct doc namespace is `http://mnemosyne.dev/doc#`.

Standard prefix block for most queries:
```sparql
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
```

For wire queries, add:
```sparql
PREFIX mnemo: <http://mnemosyne.ai/vocab#>
```

## Entity URI Patterns

```
Graph:      urn:mnemosyne:user:{user_id}:graph:{graph_id}
Document:   urn:mnemosyne:user:{user_id}:graph:{graph_id}:doc:{doc_id}
Folder:     urn:mnemosyne:user:{user_id}:graph:{graph_id}:folder:{folder_id}
Artifact:   urn:mnemosyne:user:{user_id}:graph:{graph_id}:artifact:{artifact_id}
Fragment:   ...doc:{doc_id}#frag
Block:      ...doc:{doc_id}#block-{block_id}
Valuation:  ...valuation:{doc_id}:{block_id}
```

Extracting doc_id from a block URI in SPARQL:
```sparql
BIND(STRBEFORE(STRAFTER(STR(?blockRef), ":doc:"), "#") AS ?docId)
```

## RDF Types

### Document Types
| Type | Description |
|------|-------------|
| `doc:TipTapDocument` | Document entity (NOT `doc:Document`) |
| `doc:XmlFragment` | Document content root (doc_uri + `#frag`) |
| `doc:Folder` | Folder container |
| `doc:Artifact` | Uploaded file |

### Block Types
| Type | Description |
|------|-------------|
| `doc:Paragraph` | Text paragraph |
| `doc:Heading` | Heading (has `doc:level`) |
| `doc:BulletItem` / `doc:BulletList` | Bullet list item / container |
| `doc:NumberedItem` / `doc:OrderedList` | Numbered list item / container |
| `doc:TodoItem` / `doc:TaskList` | Task list item / container |
| `doc:Quote` / `doc:Blockquote` | Quote block |
| `doc:CodeBlock` | Code block (has `doc:language`) |
| `doc:HorizontalRule` | Horizontal rule |
| `doc:TextNode` | Text content leaf node |
| `doc:Wikilink` | Internal cross-reference |

### Inline Mark Types
`doc:BoldSpan`, `doc:ItalicSpan`, `doc:StrikeSpan`, `doc:CodeSpan`, `doc:HighlightSpan`, `doc:LinkSpan`

### Wire/Graph Types
| Type | Description |
|------|-------------|
| `mnemo:Wire` | Semantic connection between documents/blocks |
| `mnemo:Graph` | User graph |
| `mnemo:DefaultGraph` | Default workspace graph |
| `mnemo:MetadataGraph` | User metadata container |

## Predicate Catalog

### Document Structure (`doc:`)
```
doc:content              - Text content of a node
doc:childNode            - Parent-to-child relationship (supports transitive: doc:childNode+)
doc:siblingOrder         - Order among siblings (float)
doc:nodeId               - Block ID for addressing
doc:hasBlock             - Legacy: document-to-block link
doc:blockId              - Legacy: block identifier
doc:order                - Sort order (float timestamp)
doc:textContent          - Legacy: block text content
doc:parentId             - Legacy: parent block reference
doc:level                - Heading level (1-3)
doc:codeText             - Code block text content
doc:language             - Code block language
doc:checked              - Task item checked state
doc:section              - Sidebar section ('documents' or 'artifacts')
doc:belongsToGraph       - Document-to-graph membership
doc:revision             - Document revision number
doc:updatedAt            - Last update timestamp
doc:updatedBy            - Last updater URI
doc:snippet              - Document text snippet
```

### Valuation Predicates (`doc:`)
```
doc:blockRef                 - URI reference to valued block
doc:rawImportanceSum         - Sum of raw importance inputs (xsd:float)
doc:importanceCount          - Count of importance valuations (xsd:integer)
doc:cumulativeImportance     - Logarithmic cumulative importance (xsd:float)
doc:rawValenceSum            - Sum of raw valence inputs (xsd:float)
doc:valenceCount             - Count of valence valuations (xsd:integer)
doc:cumulativeValence        - Logarithmic cumulative valence (xsd:float)
doc:lastValuatedAt           - Last valuation timestamp (xsd:dateTime)
```

### Wire Predicates (`mnemo:`)
```
mnemo:sourceDocument     - Wire source document URI
mnemo:targetDocument     - Wire target document URI
mnemo:sourceBlock        - Wire source block URI (optional)
mnemo:targetBlock        - Wire target block URI (optional)
mnemo:sourceGraph        - Source graph URI (for cross-graph wires)
mnemo:targetGraph        - Target graph URI (for cross-graph wires)
mnemo:predicate          - Semantic predicate URI
mnemo:bidirectional      - Boolean bidirectional flag
mnemo:sourceSnippet      - Cached source text preview
mnemo:targetSnippet      - Cached target text preview
mnemo:sourceTitle        - Cached source document title
mnemo:targetTitle        - Cached target document title
mnemo:snapshotAt         - Snapshot refresh timestamp
mnemo:createdAt          - Wire creation timestamp
```

### Graph Metadata Predicates (`mnemo:`)
```
mnemo:graphId            - Graph identifier string
mnemo:status             - Graph status ("active", "deleted")
mnemo:tripleCount        - Count of triples in graph
mnemo:lastQueryAt        - Last query timestamp
mnemo:lastUpdateAt       - Last update timestamp
```

### Dublin Core (`dcterms:`)
```
dcterms:title            - Title (documents, graphs, folders)
dcterms:description      - Description (graphs)
dcterms:created          - Creation timestamp
dcterms:modified         - Last modified timestamp
```

### File Metadata (`nfo:`)
```
nfo:fileName             - Display name for files/folders
nfo:belongsToContainer   - Parent folder relationship
```

## Common Query Patterns

### Find all documents in a graph
```sparql
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX dcterms: <http://purl.org/dc/terms/>

SELECT ?doc ?title
WHERE {
  ?doc a doc:TipTapDocument .
  OPTIONAL { ?doc dcterms:title ?title }
}
```

### Find valuated blocks (sorted by composite score)
```sparql
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?blockRef ?cumImp ?cumVal ?lastVal
WHERE {
  ?val doc:blockRef ?blockRef .
  ?val doc:cumulativeImportance ?cumImp .
  ?val doc:cumulativeValence ?cumVal .
  OPTIONAL { ?val doc:lastValuatedAt ?lastVal }
}
ORDER BY DESC(xsd:float(?cumImp) + ABS(xsd:float(?cumVal)))
LIMIT 20
```

### Find valuated blocks in a specific document
```sparql
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?blockRef ?cumImp ?cumVal
WHERE {
  ?val doc:blockRef ?blockRef .
  ?val doc:cumulativeImportance ?cumImp .
  ?val doc:cumulativeValence ?cumVal .
  FILTER(STRSTARTS(STR(?val), "urn:mnemosyne:user:USER_ID:graph:GRAPH_ID:valuation:DOC_ID"))
}
ORDER BY DESC(xsd:float(?cumImp))
```

### Search block content (lexical)
```sparql
PREFIX doc: <http://mnemosyne.dev/doc#>

SELECT ?block ?blockId ?text
WHERE {
  ?block doc:nodeId ?blockId .
  ?block doc:childNode+ ?textNode .
  ?textNode doc:content ?text .
  FILTER(CONTAINS(LCASE(?text), "search term"))
}
LIMIT 30
```

### Find wires for a document
```sparql
PREFIX mnemo: <http://mnemosyne.ai/vocab#>

SELECT ?wire ?targetDoc ?predicate ?bidirectional
WHERE {
  ?wire a mnemo:Wire ;
        mnemo:sourceDocument <DOC_URI> ;
        mnemo:targetDocument ?targetDoc ;
        mnemo:predicate ?predicate .
  OPTIONAL { ?wire mnemo:bidirectional ?bidirectional }
}
```

### Count predicates in use
```sparql
PREFIX mnemo: <http://mnemosyne.ai/vocab#>

SELECT ?predicate (COUNT(?wire) AS ?count)
WHERE {
  ?wire a mnemo:Wire ;
        mnemo:predicate ?predicate .
}
GROUP BY ?predicate
ORDER BY DESC(?count)
```

### Aggregate document-level valuation scores
```sparql
PREFIX doc: <http://mnemosyne.dev/doc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?docId
       (AVG(xsd:float(?cumImp)) AS ?avgImp)
       (MAX(xsd:float(?cumImp)) AS ?maxImp)
       (AVG(ABS(xsd:float(?cumVal))) AS ?avgAbsVal)
WHERE {
  ?val doc:blockRef ?blockRef .
  ?val doc:cumulativeImportance ?cumImp .
  ?val doc:cumulativeValence ?cumVal .
  BIND(STRBEFORE(STRAFTER(STR(?blockRef), ":doc:"), "#") AS ?docId)
}
GROUP BY ?docId
```

## Auto-Wrapping Behavior

The `sparql_query` and `sparql_update` MCP tools auto-inject graph context:

**SELECT queries:** A `FROM <graph_uri>` clause is injected before `WHERE` if no `FROM` or `FROM NAMED` is present. You don't need to specify the graph — it's derived from the `graph_id` parameter.

**UPDATE queries:** Wrapping depends on the query type:
| Pattern | Auto-wrapping |
|---------|---------------|
| `INSERT DATA { ... }` | `INSERT DATA { GRAPH <uri> { ... } }` |
| `DELETE DATA { ... }` | `DELETE DATA { GRAPH <uri> { ... } }` |
| `DELETE WHERE { ... }` | `DELETE WHERE { GRAPH <uri> { ... } }` |
| Other (INSERT/DELETE WHERE) | `WITH <uri>` prepended after PREFIXes |

**Important:** `DELETE WHERE` shorthand uses GRAPH wrapping, not WITH. The WITH clause is not compatible with DELETE WHERE shorthand in PyOxigraph.

## Gotchas

1. **Two vocabularies:** Document structure uses `doc:` (`http://mnemosyne.dev/doc#`). Wires and graph metadata use `mnemo:` (`http://mnemosyne.ai/vocab#`). Mixing them up silently returns empty results.

2. **Type name:** Document type is `doc:TipTapDocument`, not `doc:Document`.

3. **Type casting required:** Numeric comparisons and ORDER BY on valuation fields require `xsd:float()` casting. Without it, values compare as strings.

4. **Transitive closure:** Use `doc:childNode+` (one or more hops) to traverse from blocks to nested text nodes. A single `doc:childNode` won't reach text inside list items or containers.

5. **Dual materialization schemas:** Block queries may need UNION branches for legacy flat schema (`doc:hasBlock`, `doc:blockId`, `doc:textContent`) vs tree schema (`doc:childNode`, `doc:nodeId`, `doc:siblingOrder`). New documents use tree schema only.

6. **Composite ordering for list items:** Items inside list containers use `?fragOrder * 10000 + ?itemOrder + 1` to interleave correctly with surrounding blocks.

7. **Prefer MCP tools over raw SPARQL:** `search_documents` (title search), `search_blocks` (content search), `get_wires` (wire queries), `get_block_values` (valuations) handle these common patterns with less risk of error. Use SPARQL for queries these tools can't express — aggregations, custom filters, cross-entity joins.

8. **FILTER NOT EXISTS for cleanup:** Orphan detection uses `FILTER NOT EXISTS { ?child ?cp ?co . }` to find dangling references.

9. **VALUES clause for batch lookups:** More efficient than chaining FILTER ORs:
   ```sparql
   VALUES (?v) { (<uri1>) (<uri2>) (<uri3>) }
   ```

10. **String literal escaping:** Backslash, double-quote, newline, carriage return must be escaped in SPARQL string literals. The platform handles this automatically in query builders, but raw SPARQL via the MCP tool does not.
