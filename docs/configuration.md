# Configuration reference

Chonk has three configuration layers. They are applied in this order: hardcoded defaults in the library, then a TOML file (benchmark CLI only), then CLI flags. Each layer overrides only the keys it sets; the others fall through.

**Library API.** Pass arguments directly to constructors. No file on disk, no global state. This is the right layer for application code and tests.

**TOML config file.** Used exclusively by `demo/graphrag_bench.py`. A config file is loaded with `--config path/to/file.toml`, deep-merged over the hardcoded defaults, and then CLI flags override any key that was explicitly passed. A config file may declare `extends = "base.toml"` to inherit from a parent — chains up to depth 5.

**CLI flags.** Override any TOML value. Only explicitly-passed flags win; a flag that was not passed leaves the TOML value in place.

---

## Data sources

Chonk fetches documents through **Transport** and **Crawler** objects. Transports handle single URIs; crawlers discover and enumerate many URIs from a root. Both implement the same `fetch(uri) → FetchResult` protocol and are used by `DocumentLoader`.

### Available transports and crawlers

| Type | Class | URI pattern | Key constructor args |
|---|---|---|---|
| `directory` | `DirectoryCrawler` | local path / `file://` | `extensions`, `recursive`, `max_files`, `exclude_dirs` |
| `github` | `GitHubCrawler` | `https://github.com/org/repo` | `token`, `branch`, `extensions`, `max_files`, `repo_include`, `repo_exclude` |
| `web` | `WebCrawler` | `https://…` | `max_pages`, `max_depth`, `same_domain`, `exclude_patterns`, `include_pattern` |
| `db_schema` | `DatabaseSchemaCrawler` | SQLAlchemy URL | `connection_url`, `include_procs`, `include_views`, `include_triggers`, `schemas` |
| `sharepoint` | `SharePointCrawler` | SharePoint site URL | `site_url`, `auth_mode`, `tenant_id`, `client_id`, `client_secret`, `artifacts`, `max_items` |
| `gmail` | `GmailCrawler` | `gmail://…` | `client_id`, `client_secret`, `token_path`, `user_id` |
| `s3` | `S3Transport` | `s3://bucket/…` | boto3 env credentials |
| `http` | `HttpTransport` | `https://…` (single fetch) | — |
| `ftp` | `FtpTransport` | `ftp://…` | — |
| `sftp` | `SftpTransport` | `sftp://…` | — |
| `sql_query` | `SqlQueryTransport` | `sqlquery://<name>` | `connection` (SQLAlchemy), `sql` passed to `fetch()` |

`DatabaseSchemaCrawler` also populates chunks with `chunk_type` values (`db_table`, `db_column`, etc.) that `SchemaVocabBuilder.add_chunks()` uses to build schema-aware NER vocabulary. Set `[index.features] schema_vocab = true` when indexing database or API sources.

### TOML source declarations

Each `[[source]]` block in the config declares one data source. Multiple sources are processed in order and merged into a single index.

```toml
# Local filesystem
[[source]]
type       = "directory"
uri        = "/path/to/docs"
extensions = [".md", ".txt", ".pdf"]
recursive  = true
max_files  = 1000

# GitHub repository
[[source]]
type       = "github"
uri        = "https://github.com/myorg/myrepo"
branch     = "main"
extensions = [".py", ".md"]

# Database schema (also enables schema_vocab NER)
[[source]]
type             = "db_schema"
uri              = "postgresql://user:pass@host/db"
include_views    = true
schemas          = ["public"]

# SharePoint site
[[source]]
type       = "sharepoint"
uri        = "https://myorg.sharepoint.com/sites/mysite"
auth_mode  = "azure_ad"
artifacts  = ["documents", "lists", "pages"]

# Gmail inbox
[[source]]
type   = "gmail"
uri    = "gmail://inbox"
query  = "label:important after:2024/01/01"
limit  = 500

# Web crawl
[[source]]
type        = "web"
uri         = "https://docs.example.com"
max_pages   = 100
max_depth   = 3
same_domain = true

# SQL query result
[[source]]
type = "sql_query"
uri  = "sqlquery://my_report"
sql  = "SELECT title || ' ' || body AS content FROM articles"

# namespace is optional on any [[source]] block.
# Chunks from this source are tagged with the namespace value.
# Use --namespaces (CLI) or retrieval.namespaces (TOML) to restrict queries to
# specific namespaces at retrieval time.
[[source]]
type      = "directory"
uri       = "/path/to/internal-docs"
namespace = "internal"

[[source]]
type      = "github"
uri       = "https://github.com/myorg/public-wiki"
namespace = "public"

# namespace + domain registers the source in the normalized domains table.
# domain is a logical grouping within the namespace (e.g. "engineering", "support").
# Use --domain-ids (CLI) or retrieval.domain_ids (TOML) to restrict queries to
# specific (namespace, domain) pairs.  domain_id = "{namespace}:{domain}".
[[source]]
type      = "directory"
uri       = "/path/to/engineering-docs"
namespace = "acme"
domain    = "engineering"

[[source]]
type      = "sharepoint"
uri       = "https://acme.sharepoint.com/sites/support"
namespace = "acme"
domain    = "support"
```

### Library usage

```python
from chonk import DocumentLoader
from chonk.transports import DirectoryCrawler, DatabaseSchemaCrawler

loader = DocumentLoader()

# Crawl a directory
chunks = loader.load_crawl(DirectoryCrawler(extensions=[".md", ".txt"]), "/path/to/docs")

# Crawl a database schema (produces db_table / db_column chunks)
db_crawler = DatabaseSchemaCrawler("postgresql://user:pass@host/db")
schema_chunks = loader.load_crawl(db_crawler, "postgresql://user:pass@host/db")

# Single URL
chunks += loader.load("https://example.com/doc.pdf")
```

---

## Entity vocabulary

Entity vocabulary extends the NER pipeline with domain-specific names that spaCy would not recognize. These are added via `SchemaVocabBuilder` before NER runs.

There are two sources:

**Static list** — a fixed set of names provided verbatim. Matched case-insensitively at query time.

**DB query** — executes a SQL `SELECT` and adds each result row as an entity name. Useful for populating vocabularies from live databases (customer names, product SKUs, employee names, etc.).

A name may be declared under more than one `entity_type` in the same namespace — `"John Doe"` as both `customer` and `employee`. Each is its own entity (`customer:john_doe`, `employee:john_doe`), and one mention in a document matches both.

Data values are matched verbatim — no camelCase splitting is applied. Runs of whitespace and connector punctuation (`-`, `_`, `/`) are normalised to a single space on both sides of the comparison, so a vocabulary entry `"Acme Corp"` also matches `Acme-Corp`, `Acme  Corp`, and a name broken across a line. Sentence punctuation is left alone, so `"…Acme. Corp…"` does not match. Pass `normalize_separators=False` to `VocabularyMatcher`/`SchemaMatcher` for strict literal matching. `SchemaVocabBuilder.add_entities()` and `SchemaVocabBuilder.add_from_db()` both use `build_data_matcher()` to produce a `VocabularyMatcher` that runs alongside the schema matcher; `merge_matches` gives schema hits precedence over spaCy hits, and data-matcher hits supplement the combined result.

### TOML vocab declarations

```toml
# Static entity list
[[vocab.entities]]
type        = "static"
entity_type = "customer"
names       = ["Acme Corp", "Globex Inc", "Initech"]
namespace   = "finance"     # optional — omit for "global"

# DB-query populated vocabulary
[[vocab.entities]]
type        = "db_query"
entity_type = "employee"
connection  = "postgresql://user:pass@host/db"
sql         = "SELECT full_name FROM employees WHERE active = true"
namespace   = "hr"          # optional — omit for "global"

[[vocab.entities]]
type        = "db_query"
entity_type = "product"
connection  = "postgresql://user:pass@host/db"
sql         = "SELECT product_name FROM products"
```

`entity_type` is the label assigned to matched spans (e.g. `"customer"`, `"employee"`, `"product"`).

**`entity_type` is required for `static` and `db_query` entries, and must be omitted for `glossary`.** It is part of the entity id (`customer:acme_corp`), so a defaulted type would silently create a *different* entity than intended. A `glossary` entry always carries `term`, so setting `entity_type` there is rejected rather than ignored. Every malformed entry raises with the offending index — an entry that was skipped would leave the index looking successful with the vocabulary absent:

```
ValueError: vocab_entities[1]: unknown type 'lookup'. Expected one of db_query, glossary, static.
ValueError: vocab_entities[0]: entity_type is required for type='static'. It is part of
            the entity id, so defaulting it would create a different entity.
```

Also required: a non-empty `names` for `static`/`glossary`, and both `connection` and `sql` for `db_query`.

A third type, `glossary`, adds business/glossary terms:

```toml
[[vocab.entities]]
type      = "glossary"
names     = ["Customer Risk Score", "wire transfer"]
namespace = "risk"          # optional — omit for "global"
```

`glossary` differs from `static` in *how* the terms are matched, not just what they are labelled. Glossary terms run through `SchemaMatcher`: normalised, singularised, and variant-expanded, so `"Customer Risk Score"` also matches `customerRiskScore` and `customer_risk_score` in prose. `static` terms are matched verbatim, which is what you want for literal data values like company names. Glossary entries always carry entity_type `term`, so their IDs are `term:<slug>`.

`namespace` is supported on **every** vocab entry type and is optional; unset means `global`, the same rule `[[source]]` follows. It also applies to schema-derived vocabulary — `add_tables`, `add_sql`, `add_endpoints`, and `add_chunks` all take a `namespace`, so a table name contributed by one division is traceable to it. It records which namespace sourced the names — see [Entity aliases](#entity-aliases) for how that is stored and why a name shared by two namespaces stays a single entity.

### Library usage

```python
from chonk.ner import SchemaVocabBuilder

builder = SchemaVocabBuilder()

# Static names — namespace is optional; unset means "global"
builder.add_entities(["Acme Corp", "Globex Inc"], entity_type="customer",
                     namespace="finance")

# From a DB query
import duckdb
con = duckdb.connect("my.duckdb")
builder.add_from_db(con, {"employee": "SELECT full_name FROM employees WHERE active = true"},
                    namespace="hr")

# (entity_id, alias, namespace) for every term, ready to persist as entity_aliases
builder.namespaced_entities()

data_matcher = builder.build_data_matcher()

# Use alongside schema_matcher and spacy in NER loop
from chonk.ner import merge_matches
for chunk in loader_chunks:
    schema_hits = schema_matcher.match(chunk.content)
    data_hits   = data_matcher.match(chunk.content)
    spacy_hits  = spacy.match(chunk.content)
    combined    = merge_matches(schema_hits + data_hits, spacy_hits, source_text=chunk.content)
    entity_index.index_chunk(chunk_id, chunk.content, combined)
```

---

## Index features

These are built once against a DuckDB file and loaded at query time. None are required; each one enables an additional retrieval capability.

### NER (`build-ner`)

Runs `SpacyMatcher` over every chunk in the index, writes the results to a `chunk_entities` table, and persists them for later loading into an `EntityIndex`. The `EntityIndex` is then used by `EnhancedSearch` for entity-adjacency expansion and completeness gating.

Two optional flags extend what NER builds:

**`--with-embeddings`** — after NER, embeds all unique entity name strings and stores them in an `entity_embeddings` table. Required for `--ner-x` mode at query time, which uses ANN search over entity embeddings to find additional related entities when the literal-match expansion produces too few results.

**`--with-schema-vocab`** — before running spaCy, builds a `SchemaMatcher` from any chunks whose `chunk_type` is `db_table`, `db_column`, `api_endpoint`, `api_graphql_query`, `api_graphql_mutation`, or `api_graphql_type`. The schema matcher runs first; its hits suppress overlapping spaCy hits. Use this when your index contains schema or API chunks (from `loader.load_schema()` or `loader.load_api()`).

### SchemaMatcher and SchemaVocabBuilder

`SchemaVocabBuilder` extracts schema identifiers from several sources and compiles them into a `SchemaMatcher`. All identifiers go through `normalize_schema_term` before matching — `customerRiskScore`, `customer_risk_score`, and `CUSTOMER_RISK_SCORE` all produce the surface form `"customer risk score"` and match as the same entity.

`add_chunks()` inspects each chunk's `chunk_type` and `document_name`:

- `chunk_type in ("db_table", "db_column")` and `document_name` starts with `"schema:"` → extracts table and column names from the dotted path
- `chunk_type in ("api_endpoint", "api_graphql_query", "api_graphql_mutation", "api_graphql_type")` and `document_name` starts with `"api:"` → extracts the endpoint path
- `chunk_type == "api_field"` and `document_name` starts with `"api:"` → extracts the field name

Variants generated per term (all lowercase): normalized form, singular form (trailing `s` stripped), underscore form (if original contains `_`), and joined forms with spaces removed. First-registration wins on variant collision.

`build()` returns a `SchemaMatcher` for schema/API/column terms. `build_data_matcher()` returns a `VocabularyMatcher` for data-value terms (customer names, employee names, etc.) that were added via `add_entities()` or `add_from_db()`. Data values are matched verbatim — no camelCase splitting.

### Community index (`build-community`)

Embeds chunk breadcrumbs (heading vectors), computes a weighted average with content vectors (`alpha` controls heading weight, default `0.2`), builds a cosine similarity graph over chunks, and runs Louvain community detection. Community summaries are stored as `community_summary` chunks. Required for `mode="global"` retrieval.

### Entity-anchored SVO extraction (`build-svo`)

`cmd_build_svo` uses **entity-anchored extraction**: instead of generating free-form IDs, the LLM is given the entities that co-occur in each chunk (from the `chunk_entities` join table) and must constrain `subject_id` / `object_id` to those known entity IDs. This keeps all triples in the same ID space as the `entities` table.

Each LLM call returns a JSON object with three keys:
- `"triples"` — subject/verb/object triples from the allowed verb vocabulary
- `"descriptions"` — one-sentence description for each entity that lacks one (marked `✓` if already present)
- `"aliases"` — 1–3 alternate names or abbreviations per entity

After extraction:
1. Valid triples are written to `svo_triples` via `RelationshipIndex.save_to_db()`.
2. New descriptions are persisted to `entity_descriptions` with `source='llm'` (never overwrites `user` or `schema` source).
3. New aliases are persisted to `entity_aliases` with first-registration-wins semantics for `llm` source.
4. Entity embeddings are generated (see below) and upserted into the `embeddings` table.

Extraction runs concurrently via `ThreadPoolExecutor`. Chunks with fewer than 2 co-occurring entities are skipped.

At query time, `RelationshipIndex.load_from_db()` loads the table into memory. This is required for `mode="graph_first"`.

### Entity descriptions

The `entity_descriptions` table stores a human-readable description per `(entity_id, namespace)` pair. Three sources are supported, in descending priority:

| Source | Priority | When written |
|--------|----------|--------------|
| `user` | 0 (highest) | `store.update_entity_description()` — always overwrites |
| `schema` | 1 | Auto-populated from DB schema chunks at index time |
| `llm` | 2 (lowest) | Generated by `SVOExtractor.extract_entity_anchored()` |

Higher-priority descriptions are never overwritten by lower-priority ones.

```python
# User sets a description — always wins
store.update_entity_description("CustomerRiskScore", "Composite risk score per customer, range 0–100")

# LLM/schema source — respects priority
store.upsert_entity_description("CustomerRiskScore", "A risk score.", source="llm")
# ^ no-op: user description already present

# Batch upsert (returns count inserted)
n = store.upsert_entity_descriptions_batch(
    {"EntityA": "desc a", "EntityB": "desc b"},
    source="schema",
)

# Retrieve descriptions for a list of entity IDs
descs = store.get_entity_descriptions(["CustomerRiskScore", "FactTable"])
# → {"CustomerRiskScore": "Composite risk score...", "FactTable": "..."}

# Restrict to entities associated with a chunk in the given namespaces.
# None (the default) applies no restriction; [] matches nothing.
descs = store.get_entity_descriptions(["CustomerRiskScore"], namespaces=["finance"])
```

Chunk results are already namespace-filtered by `search(namespaces=[...])`, and
every entity/cluster expansion path is gated on that same set. The `namespaces`
argument above scopes the *entity metadata* joined into a prompt — including
entity IDs supplied by `query_entity_id_fn`, which bypass chunk filtering.
`EnhancedSearch.assemble_graph_context(..., namespaces=[...])` takes the same
argument for the Entities section of the MS-GraphRAG context block.

### Entity aliases

The `entity_aliases` table maps alternate names and abbreviations to canonical entity IDs. Primary key is `(alias, namespace, entity_id)` — entity_id is part of the key because one alias can name several entities in the same namespace. If John Doe both works at and shops at the same company, `john doe` maps to `employee:john_doe` *and* `customer:john_doe`; each is its own row and neither displaces the other. Use `store.resolve_entity_aliases(alias, namespace)` to get all of them (`resolve_entity_alias` returns the lowest-sorting one).

**Semantics:**
- `llm` and `schema` sources: first-registration wins *per mapping* — re-adding the same (alias, namespace, entity_id) is a no-op, but the same alias may still be added for a different entity.
- `user` source: refreshes the `source` column on an existing mapping.
- The namespace is optional everywhere it appears; unset means `global`.

**Entity identity.** `entities.id` is `"{entity_type}:{name_slug}"`, so
`customer:mercury` and `element:mercury` are distinct entities. The same holds
*within* one namespace: if John Doe both works at and shops at the same company
he is `employee:john_doe` and `customer:john_doe`, and a single mention of his
name in a document is recorded as evidence for both. A bare name slug
with no `:` is a pre-4 ID and means the index predates typed IDs — open it and
`SchemaVersionError` tells you to rebuild.

**Namespace as entity scope.** The ID does *not* include the namespace, so the
same name and type sourced from two namespaces is *one* entity row. Which namespaces
sourced it is recorded as one `entity_aliases` row per namespace — the `(alias,
namespace)` primary key keeps them distinct, so no source is lost:

```python
# "Acme Corp" sourced from two namespaces
store.get_entity_namespaces("acme_corp")   # → ["ns_a", "ns_b"]
```

Two writers produce these rows automatically:

| Source | `source` column | Namespace used |
| --- | --- | --- |
| Vocabulary entries (`vocab_entities`, `add_entities`, `add_from_db`) | `vocab_source` | The entry's `namespace`, or `global` |
| Identifier-suffix aliases (`customer_id` → `customer`) | `strip_suffix` | The namespace the chunk was crawled under, or `global` |
| SVO/description extraction | `llm` | `EntityGraphPipeline(namespace=...)` |

If `(alias, namespace)` is already registered for a *different* entity, tagging
raises rather than dropping the tag silently.

Aliases are generated automatically alongside SVO extraction. They can also be added manually:

```python
# Single alias
store.add_entity_alias("CRS", "CustomerRiskScore", source="user")

# Batch (dict maps alias → entity_id); returns count inserted
n = store.add_entity_aliases_batch(
    {"CRS": "CustomerRiskScore", "CLV": "CustomerLifetimeValue"},
    source="llm",
)

# Resolve an alias to its canonical entity_id
eid = store.resolve_entity_alias("CRS")   # → "CustomerRiskScore"

# All aliases for an entity
aliases = store.get_entity_aliases("CustomerRiskScore")  # → ["CRS", "Risk Score"]
```

### Filtering chunks by entity type

Every chunk row carries an `entity_types` array — the set of entity types the
chunk mentions, denormalized from `chunk_entities` by `build_ner`. It exists so
retrieval can filter on entity type without a join, which the external vector
backends cannot express:

```python
# Chunks that mention at least one customer
hits = store.search(query_vec, limit=10, entity_types=["customer"])

# Any-overlap semantics: chunks mentioning a customer OR an employee
hits = store.search(query_vec, limit=10, entity_types=["customer", "employee"])
```

Supported on all six backends — a native array column on DuckDB (`VARCHAR[]`)
and PostgreSQL (`TEXT[]`), and filterable payload/metadata on Qdrant, Pinecone,
and Weaviate, which `set_chunk_entity_types()` writes alongside the catalog row.

Indexed where the backend supports it: a GIN index on PostgreSQL (backing the
`&&` overlap operator), a keyword payload index on Qdrant, and a filterable
`TEXT_ARRAY` property on Weaviate. Pinecone indexes metadata automatically.
DuckDB has no list index — `list_has_any` is a scan over the candidate rows,
which is fine at the scale DuckDB is used for here.

Two things to know:

- The field is populated by `build_ner`, not at index time — NER runs after
  chunking. A chunk that has been indexed but not yet NER'd has `entity_types`
  `NULL` and matches no `entity_types` filter.
- Types come straight off the entity ID (`customer:john_doe` → `customer`), so
  there is no join and no second source of truth.

### Testing against non-DuckDB backends

Backend behaviour diverges, and every parity defect found so far — a surviving
`documents` registry row, `clear()` not cascading, DuckDB-only SQL on the NER
write path — was invisible to a DuckDB-only run. The parity suite
(`tests/unit/test_backend_registry_contract.py`) parametrizes over backends:

| Lane | Requirement | Runs by default |
| --- | --- | --- |
| `duckdb` | none | yes |
| `qdrant-local` | `qdrant-client` installed | yes — client-side engine, no server |
| `weaviate-embedded` | `weaviate-client` installed | yes — bundled binary, no server |
| `pg` | `CHONK_TEST_PG_DSN` | when set |
| `qdrant` / `pinecone` / `weaviate` | their service env var | when set |

`qdrant-local` uses qdrant-client's embedded engine via `Store(qdrant_url=":memory:")`
— pass `"file:/path/to/dir"` to persist instead. It supports every operation the
backend issues; payload indexes are accepted and ignored there, which costs speed,
not correctness.

`weaviate-embedded` runs weaviate-client's bundled binary. `Store(weaviate_url="embedded")`
starts one; an `http://` URL on a loopback host connects to a server already running
(`weaviate_grpc_port`, `weaviate_local_port`, and `weaviate_persistence_path` configure
both paths). The test fixture starts one server per session on ephemeral ports and gives
each test a fresh collection — startup takes seconds and occasionally times out, so a
per-test start would be slow and flaky. If it fails to start, the lane skips rather than
failing the run.

PostgreSQL needs a real server because `pgvector` supplies the `<=>` cosine
operator — a stock `postgres` image will not do, and an embedded PostgreSQL only
helps if the extension is compiled into it. CI runs a `pgvector/pgvector:pg16`
service container; locally:

```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg16
CHONK_TEST_PG_DSN=postgresql://postgres:pw@localhost:5432/postgres pytest tests/unit -q
```

`test_a_non_duckdb_backend_ran` fails if every non-DuckDB lane skipped, so a green
run cannot mean "nothing was exercised".

### Looking up an entity without its type

Entity IDs are type-qualified, but callers should not have to know the prefix:

`EntityLookup` and `NamespaceEvidence` are exported from both `chonk` and
`chonk.storage`, so callers can annotate against them without reaching into a
private module.

```python
store.resolve_entity_ids("Mercury")
# ["customer:mercury", "element:mercury"]   ← every type, caller disambiguates

store.resolve_entity_ids("Mercury", entity_type="customer")
# ["customer:mercury"]

store.resolve_entity_ids("customer:mercury")
# ["customer:mercury"]                      ← already qualified, passes through
```

Matching is on the name slug, so `"Acme Corp"`, `"acme corp"`, and `"acme_corp"`
are the same query. An unknown name returns `[]` rather than guessing.

**When the result is empty**, neither method raises — a miss is a normal outcome,
not an error. But `[]` has two causes, and `explain_entity_lookup()` separates
them:

```python
store.explain_entity_lookup("Mercury", entity_type="ghost")
# EntityLookup(ids=[], name_exists=True, available_types=["customer", "element"],
#              available_namespaces=["retail"], near_matches=["customer:mercury_systems"], ...)

store.explain_entity_lookup("Mercury", namespaces=["support"])
# EntityLookup(ids=[], name_exists=True, available_types=["customer", "element"],
#              available_namespaces=["retail"], ...)   ← wrong namespace, not a miss

store.explain_entity_lookup("Nobody")
# EntityLookup(ids=[], name_exists=False, available_types=[], available_namespaces=[], ...)
```

`name_exists=True` with empty `ids` means a *filter* excluded it — a caller
mistake with an obvious fix, and `available_types` / `available_namespaces` say
what to ask for instead. Everything empty is a genuine miss. The reported
alternatives deliberately ignore the filters that were passed.

Both methods take the same `entity_type` and `namespaces` filters, so they always
answer the same question. `namespaces` is an *evidence* filter — where the entity
actually appears, the same notion `get_entity_namespace_evidence()` reports — not
where its vocabulary was declared. `None` applies no restriction; `[]` matches
nothing, mirroring `search`. `near_matches` helps with typos and partial names, in both directions — an
under-specified query (`"mercury"` → `customer:mercury_systems`) and an
over-specified one (`"mercury systems corp"` → both). Comparison is on the name
slug only, so querying a type name like `"customer"` does not return every
`customer:*`. Ordered by closeness in length, capped at `NEAR_MATCH_LIMIT`, with
`near_matches_truncated` saying when more existed rather than silently trimming.

An empty result is only trustworthy because nothing upstream can manufacture one:
a malformed `vocab_entities` entry raises at build time instead of being dropped,
so "not found" cannot mean "your vocabulary was ignored".

### Which namespaces are relevant to an entity

Two different questions, two accessors — an entity is shared across namespaces,
so "where was it declared" and "where does it actually appear" diverge:

```python
# Declaration provenance — which namespace's vocabulary contributed the entity.
store.get_entity_namespaces("customer:acme_corp")
# ["global"]   ← a shared customer list declared once, centrally

# Evidence — which namespaces have documents that mention it, ranked.
store.get_entity_namespace_evidence("customer:acme_corp")
# [NamespaceEvidence(namespace="retail",  chunk_count=2, score=1.57, share=1.00),
#  NamespaceEvidence(namespace="support", chunk_count=1, score=0.79, share=0.25)]
```

Use evidence for routing — "which divisions is this customer relevant to". It
reads `chunk_entities`, which carries a namespace on every association.

`share` is `chunk_count` over that namespace's total chunk count. Raw counts are
biased by corpus size: a division with 3 of 5000 chunks mentioning a customer
outranks one with 2 of 2 on volume alone, while the small one is far more *about*
them. Both are returned; rank on `share`, size on `chunk_count`. Ordering is by
summed association score, then count, then namespace.

### Entity embeddings

After `build-svo` completes, `_upsert_entity_chunk_embeddings` generates a semantic embedding for each entity and stores it as a `chunk_type='entity'` row in the `embeddings` table.

**Text representation** per entity:
```
"{name}. {aliases_comma_separated}. {description}"
```
Parts that are absent are omitted.

Entity rows share the same HNSW index as document chunks. You can find semantically similar entities even without co-occurrence:

```python
# Search restricted to entity rows only
results = store.search(query_vec, k=10, chunk_types=["entity"])
```

Each entity's `document_name` is `__entity__{entity_id}`, making lookups unambiguous. The `chunk_id` is a deterministic hash of `(document_name, chunk_index, content)`.

Stale entity rows are purged (`DELETE FROM embeddings WHERE chunk_type = 'entity'`) before each rebuild, so re-running `build-svo` is idempotent.

### Entity normalizer

`chonk.ner.normalize_entity` and `chonk.ner.canonical_key` produce a canonical form for entity strings before deduplication or storage.

**Pipeline per entity string:**
1. Strip leading/trailing symbols (brackets, quotes, punctuation)
2. Collapse dotted acronyms: `U.S.A.` → `USA`
3. Collapse internal whitespace
4. Split on spaces / underscores / camelCase boundaries
5. Singularize the last token (head-noun rule) using `inflect`
6. Preserve acronym casing (all-uppercase strings like `IBM`, `FASB`, `API` are never singularized or lowercased)

Words in `SINGULAR_EXCEPTIONS` (`data`, `metadata`, `criteria`, `media`, etc.) are never singularized.

```python
from chonk.ner import normalize_entity, canonical_key, EntityNormalizer

normalize_entity("compliance policies")   # → "compliance policy"
normalize_entity("CustomerRiskScores")    # → "CustomerRiskScore"
normalize_entity("IBM")                   # → "IBM"
normalize_entity("U.S.A.")               # → "USA"
canonical_key("policies") == canonical_key("policy")   # True — dedup key

# Class form with per-instance exceptions
n = EntityNormalizer(extra_exceptions=frozenset({"scores"}))
n.normalize("risk scores")  # → "risk scores" (exception prevents singularization)
```

`canonical_key` returns the lowercase normalized form and is the deduplication key used when merging entity matches.

---

## Namespaces, Domains, and Sessions

Chonk organizes indexed content into a three-level hierarchy. Understanding this model matters when running multiple users, projects, or data sources from a single deployment.

### Data model

```
namespaces
  └── domains  (self-referential tree: a domain may have a parent_id)
        └── sources
              └── embeddings  (chunks; domain_id is a denormalized FK for fast filtering)
```

**`namespace`** — top-level owner or tenant. Examples: `"global"`, `"user:alice"`. Each namespace has a `namespace_id` (primary key), optional `description` and `owner`, and timestamps `created_at` / `updated_at`.

**`domain`** — logical grouping within a namespace. Examples: `"engineering"`, `"support"`. Primary key is `domain_id` (conventionally `"{namespace_id}:{domain_name}"`). A domain may have a `parent_id` pointing to another domain in the same namespace, forming a tree. Child domains are included automatically in any query that targets the parent.

**`source`** — one crawl target (a `[[source]]` TOML block or a `register_source` call). Each source belongs to a single namespace + domain.

**`embeddings`** — one row per chunk. Carries `namespace`, `domain_id`, `source_id`, and `session_fingerprint` as denormalized columns so a single WHERE clause filters without joins.

The `global` namespace is special. Every session folds in all global domains by default unless `include_global=False` is passed.

### Per-namespace DuckDB isolation

Each namespace gets its own `.duckdb` file. Call `attach_global()` to attach the global DB read-only to a user-namespace Store. Union views (`all_embeddings`, `all_chunk_entities`, `all_svo_triples`, `all_domains`, `all_sources`, `all_namespaces`) make reads transparent — queries against `all_embeddings` see both the user's own chunks and the global chunks.

```python
# Write-mode Store — held exclusively by the background Indexer
store = Store("user_alice.duckdb", embedding_dim=1024)
store.attach_global("global.duckdb")

# Read-only Store — for search sessions; any number can run concurrently
store = Store("user_alice.duckdb", embedding_dim=1024, read_only=True)
store.attach_global("global.duckdb")
```

DuckDB allows unlimited concurrent read-only connections to a file. The one-writer restriction applies only to write connections, which the background `Indexer` holds exclusively.

### Registering namespaces, domains, and sources

All three registration calls are upserts — safe to call repeatedly with the same arguments.

```python
store.register_namespace(
    "user:alice",
    description="Alice's personal workspace",
    owner="alice@example.com",
)

# Root domain. domain_id is the first positional argument.
store.register_domain(
    "user:alice:engineering",   # domain_id
    "user:alice",               # namespace_id
    "engineering",              # name
)

# Child domain. parent_id references the parent domain_id.
store.register_domain(
    "user:alice:engineering:backend",
    "user:alice",
    "backend",
    parent_id="user:alice:engineering",
)

store.register_source(
    "src-001",                          # source_id
    "user:alice:engineering",           # domain_id
    "directory",                        # type
    "/path/to/docs",                    # uri
)
```

### Sessions and `resolve_session`

A session covers one namespace plus a subset of its domains. Pass the resolved `domain_ids` list to `store.search()` to scope results to that session.

```python
# Read-only store for search
store = Store("user_alice.duckdb", read_only=True)
store.attach_global("global.duckdb")

# Resolve domain_ids for this session.
# active_domains are names within "user:alice"; global domains fold in automatically.
domain_ids = store.resolve_session(
    namespace_id="user:alice",
    active_domains=["engineering", "support"],
    include_global=True,   # default; folds in all domains from the global namespace
)

results = store.search(query_vec, limit=10, domain_ids=domain_ids)
```

`resolve_domain_ids` is the lower-level primitive. It accepts a list of `(namespace_id, domain_name)` pairs and uses a recursive CTE to include all child domains automatically.

```python
domain_ids = store.resolve_domain_ids(
    [("user:alice", "engineering"), ("global", "kb")],
    include_global=True,
)
```

### Community fingerprint cache

Community summaries are expensive to build. The store caches them by session composition.

`community_cache_valid` checks both that a fingerprint record exists and that the chunk count for those domains has not changed since the cache was written — so stale entries are detected without explicit invalidation.

```python
fingerprint = store.session_fingerprint(domain_ids)  # sha256(sorted(domain_ids))[:16]

if store.community_cache_valid(fingerprint, domain_ids):
    pass  # load cached community_summary chunks from the DB
else:
    # build community summaries, store chunks, then record the cache entry
    store.write_community_cache(fingerprint, domain_ids)
```

`invalidate_community_cache(domain_id)` deletes all cache entries that include the given `domain_id` and removes the corresponding `community_summary` chunks from `embeddings`. Call it after removing or re-indexing a domain.

### Deleting a domain

```python
rows_deleted = store.delete_domain("user:alice:old-project")
# Cascades to chunk_entities and svo_triples rows for the same chunks.
# Community cache entries that referenced this domain are automatically invalidated.
```

### TOML

```toml
[retrieval]
namespaces = ["user:alice", "global"]  # restrict to these namespaces
domain_ids = ["user:alice:engineering", "global:kb"]  # restrict to these domain_ids
```

CLI: `--namespaces user:alice global` and `--domain-ids user:alice:engineering global:kb`

---

## Background indexing

`Indexer` runs the full pipeline — crawl, chunk, embed, store — in a background thread. It is safe to abort mid-run. The abort flag is checked between embedding batches and before the store phase, so the DB is never left in a partial state.

### `Indexer` and `IndexHandle`

```python
from chonk import Indexer

def on_progress(phase: str, done: int, total: int):
    print(f"{phase}: {done}/{total}")

def on_complete(chunks: int):
    print(f"Indexed {chunks} chunks")

def on_error(phase: str, error: Exception):
    print(f"Error in {phase}: {error}")

def on_abort(chunks: int):
    print(f"Aborted after {chunks} chunks")

indexer = Indexer(
    store=store,               # write-mode Store
    embed_model="BAAI/bge-large-en-v1.5",
    on_progress=on_progress,
    on_complete=on_complete,
    on_error=on_error,
    on_abort=on_abort,
    embed_batch_size=256,
    min_chunk_size=400,
    max_chunk_size=1200,
)

source_config = {
    "type": "directory",
    "uri": "/path/to/docs",
    "namespace": "user:alice",
    "domain_id": "user:alice:engineering",
    "source_id": "src-001",
}

# Non-blocking
handle = indexer.index_source_async(source_config)

# Poll
while handle.running:
    time.sleep(1)

# Or wait with timeout
handle.join(timeout=300)

# Abort — finishes the current embedding batch, then stops cleanly
indexer.abort()
```

`index_source` is the blocking variant. It returns the number of chunks stored.

### Progress phases

`on_progress(phase, done, total)` fires at the end of each phase and after each embedding batch:

| `phase` | Meaning |
|---|---|
| `"crawl"` | Document discovery and chunking complete. `done == total == chunk_count`. |
| `"chunk"` | Same as crawl (chunking is integrated into crawl). |
| `"embed"` | After each embedding batch. `done` = batches completed, `total` = total batches. |
| `"store"` | All chunks written to DB. `done == total == chunk_count`. |

### Singleton registry

A process should have exactly one `Indexer` per namespace. Use `get_indexer` / `release_indexer` to enforce this:

```python
from chonk import get_indexer, release_indexer

# First call creates the Indexer; subsequent calls return the cached instance.
indexer = get_indexer(
    namespace_id="user:alice",
    store=write_store,
    embed_model="BAAI/bge-large-en-v1.5",
    on_progress=on_progress,
    on_complete=on_complete,
)

# Multiple sessions for the same user share one Indexer.
same = get_indexer("user:alice", write_store, "BAAI/bge-large-en-v1.5")
assert indexer is same  # True

# When a namespace is deleted or the process is shutting down:
release_indexer("user:alice")  # aborts any in-progress run and removes from registry
```

Search sessions always open their Store with `read_only=True`. They never need the Indexer and should never hold a write connection.

### On-the-fly source changes

Adding or removing a source does not require rebuilding the entire index.

**Add a source:**

```python
store.register_source("src-002", "user:alice:engineering", "directory", "/path/to/new-docs")
indexer = get_indexer("user:alice", write_store, embed_model)
handle = indexer.index_source_async({
    "type": "directory",
    "uri": "/path/to/new-docs",
    "namespace": "user:alice",
    "domain_id": "user:alice:engineering",
    "source_id": "src-002",
})
handle.join()
# Secondary indexes (NER, SVO, community summaries) for this namespace are now stale.
# Rebuild with: build-community --namespace user:alice  and  build-svo --namespace user:alice
```

**Remove a source:**

```python
store.delete_domain("user:alice:old-project")
# Cascades to chunk_entities and svo_triples.
# Community cache entries for this domain are automatically invalidated.
```

After adding or removing sources, secondary indexes for the affected namespace are stale and should be rebuilt. Community summaries are the most expensive; use the fingerprint cache to skip rebuilds when the session composition has not changed.

---

## Recommended index configuration

Derived from the ECDR-Bench leader (`fang_ner_ref_bc_laned60_community_k30_srr_bm25_mini`, score 0.516).

### Step 1 — Build

```python
import chonk

index = chonk.build("my_config.yaml")
```

That's it. One call runs: ingest → embed → FTS → NER → community. Each phase is skipped if already built (idempotent). SVO graph is opt-in via config.

Config file (`my_config.yaml`):

```yaml
store:
  path: my.duckdb
  embedding_dim: 1024

loader:
  min_chunk_size: 1100
  max_chunk_size: 2200
  enrich_context: true          # breadcrumb heading prefix in each chunk embedding

embed:
  model: BAAI/bge-large-en-v1.5
  batch_size: 256

index:
  ner: true                     # default true
  community: true               # default true
  svo: false                    # default false — set true to enable graph_first (calls LLM API)
  svo_model: gpt-4o-mini        # used when svo: true

search:
  k: 30
  entity_ref_expansion: true
  lane_entity_min_sim: 0.60

domain_dictionary:                   # optional — descriptions for domain names
  sales-analytics: "Sales and customer analytics data"
  sales-analytics/north-america: "North America regional sales"

sources:
  - name: my_docs
    type: glob                       # glob | json_array | sql
    path: ./docs
    pattern: "*.md"
    namespace: global                # optional
    domain: my_docs                  # optional — defaults to source name
```

Each source is registered as a domain (name defaults to the source `name`). `domain_dictionary` supplies descriptions; without it, domains are still registered — just without descriptions.

### Step 2 — Search

```python
results = index.search("What are the reporting obligations?")
```

Filter by namespace or domain name at call time:

```python
results = index.search("Q3 revenue?", domains=["sales-analytics"])
results = index.search("policy update?", namespaces=["global"])
results = index.search("north america trends?",
                       namespaces=["global"], domains=["sales-analytics/north-america"])
```

**Automated routing** — chonk automatically routes each query to the relevant namespaces and domains via an extra LLM call. This is on by default when `openai` is installed and `OPENAI_API_KEY` is set; otherwise it degrades to unscoped search with no error.

The routing model defaults to `gpt-4o-mini`. Override it with `index.routing_model` in the config, or set the instance attributes directly to swap in any LLM:

```python
index = chonk.build("config.yaml")

# Replace the default routing fn (applies to both namespace and domain routing)
def my_fn(prompt: str) -> str:
    ...  # call any LLM; return a string containing a JSON array

index.namespace_filter_llm_fn = my_fn
index.domain_filter_llm_fn = my_fn

# Disable routing entirely — search runs unscoped
index.namespace_filter_llm_fn = None
index.domain_filter_llm_fn = None

# Disable one dimension only
index.namespace_filter_llm_fn = None   # explicit namespaces only, domain routing still fires
```

The routing fn receives a fully-formed prompt and must return a string containing a JSON array. The default prompts look like this (shown as examples — replace the fn entirely if you want a different format or style):

**Namespace routing — example prompt:**
```
You are a retrieval routing assistant. Below are the available knowledge namespaces
and their descriptions. Select the namespaces most likely to contain evidence for
the query. Return ONLY a JSON array of namespace IDs, e.g. ["cyber", "financial"].
Include all namespaces that may be relevant; omit those that are clearly unrelated.

Namespaces:
- financial: SEC filings and earnings reports
- cyber: CVE advisories and threat intelligence

Query: What were Amazon's Q3 earnings?

Return ONLY a JSON array of namespace IDs.
```

**Domain routing — example prompt:**
```
You are a retrieval routing assistant. Below are the available knowledge domains
and their descriptions. Select the domains most likely to contain evidence for
the query. Return ONLY a JSON array of domain names exactly as listed,
e.g. ["sales/north-america", "finance/q1"].
Include all domains that may be relevant; omit those that are clearly unrelated.

Domains:
- sales/north-america (global): North America regional sales data
- sales/emea (global): EMEA regional sales data
- finance/q3 (global): Q3 financial reports

Query: What were north america sales in Q3?

Return ONLY a JSON array of domain names exactly as listed.
```

Routing is skipped (search runs unscoped for that dimension) when fewer than two namespaces/domains have descriptions registered, or when the explicit `namespaces=` / `domains=` argument is passed to `search()`.

Per-call overrides work the same way — pass `namespace_filter_llm_fn=` or `domain_filter_llm_fn=` directly to `search()` to override the instance default for a single call. Passing `namespaces=` or `domains=` explicitly bypasses the routing fn entirely for that dimension regardless of the instance attribute.

Inspect registered domains:

```python
index.domains()           # {"global": ["sales-analytics", "legal", ...], ...}
index.domains("global")   # {"global": ["sales-analytics", ...]}
```

### Step 3 — Live mutations

`Index` supports real-time mutations without restarting or rebuilding the whole store. All methods update the in-memory domain map immediately; secondary indexes (NER, community, SVO) are rebuilt asynchronously when `rebuild=True`.

#### Namespaces

```python
index.add_namespace("user:alice", description="Alice's workspace")
# No-op if already registered.

index.remove_namespace("user:alice")
# Deletes the namespace and all its domains, chunks, chunk_entities, and svo_triples rows.
```

#### Domains

```python
domain_id = index.add_domain("user:alice", "engineering/backend",
                              description="Backend services docs")
# Returns the stable domain_id. No-op if already registered.

n = index.remove_domain("user:alice", "engineering/backend")
# Returns count of chunks deleted. Cascades to chunk_entities and svo_triples.
```

#### Sources

```python
# Ingest a new source and trigger async secondary-index rebuild.
handle = index.add_source(
    {
        "type": "glob",
        "path": "./new-docs",
        "pattern": "*.md",
        "namespace": "user:alice",   # optional — defaults to "global"
        "domain": "engineering",     # optional — defaults to source name / type
        "enrich_context": True,      # optional — overrides loader default
    },
    rebuild=True,          # default; set False to skip secondary-index rebuild
    on_progress=None,      # (phase, done, total) -> None
    on_complete=None,      # (total_chunks) -> None
    on_error=None,         # (phase, exc) -> None
)
handle.join()   # block until NER + community rebuild finish

# Remove a source (by namespace + domain name).
handle = index.remove_source(
    "user:alice", "engineering",
    rebuild=True,
)
if handle:
    handle.join()
```

`add_source` accepts the same source dict schema as a `sources:` entry in the YAML config (`type`, `path`/`connection`/`array_field`, `pattern`, `namespace`, `domain`, `enrich_context`). Namespace and domain are registered automatically if they do not exist.

`add_source` returns `None` when the source produces no chunks or when `rebuild=False`.  
`remove_source` returns `None` when `rebuild=False` or the namespace has no remaining domains after deletion.

#### Explicit rebuild

```python
# Rebuild secondary indexes for one namespace (async by default).
handles = index.rebuild("user:alice")
for h in handles:
    h.join()

# Rebuild all namespaces, blocking.
index.rebuild(async_=False)

# Per-phase callbacks work the same as add_source / remove_source.
index.rebuild("global", on_progress=lambda phase, done, total: print(phase, done, total))
```

`rebuild` returns a list of `IndexHandle` objects — one per namespace rebuilt. Call `.join()` on each to wait for completion. Rebuild phases: NER → community detection (SVO is skipped unless originally configured).

Config `search:` keys set the defaults; call-site arguments override them:

```python
results = index.search("Who supplies components to Acme Corp?", k=10, mode="graph_first")
```

`mode="graph_first"` requires `index.svo: true` in the config. All other modes work with just the defaults.
| `query_text` | pass the question | Required when `query_embedding` is omitted; activates BM25 hybrid scoring |
| `mode` | `vector_first` | |

Apply a local reranker over the results before passing to the generator.

---

## Retrieval modes

Pass `mode=` to `EnhancedSearch.search()`.

### `vector_first` (default)

Seed pool = `k × seed_pool_multiplier` (default `3`). From the seed:

1. Structural expansion pulls the previous and next chunk from each seed chunk's document.
2. Entity expansion looks up all entity IDs present in the seed+structural pool, then fetches up to `entity_expansion_top_n` (default `3`) chunks per entity from `EntityIndex`. When `lane_entity_min_sim` is set, entity-linked chunks below that cosine similarity threshold are dropped before entering the pool.
3. Cluster expansion adds cluster-adjacent chunks, budget-limited to `cluster_budget` (default `2 × k`).
4. Greedy MMR selects the final top-k using a composite score: `rw × relevance + pw × priority + cw × coverage`, where `coverage = relevance - λ × max_sim_to_selected`.

After selection, if `entity_ref_expansion=True` and query entities are missing from the result text, an additional search loop fetches chunks for the missing entities.

### `graph_first`

Requires `build-svo` to have run (and a `RelationshipIndex` loaded). Requires either `query_entities` passed directly to `search()`, or a `query_ner_fn` set on the `EnhancedSearch` constructor.

Steps: run NER on query text → traverse `RelationshipIndex` 1-hop forward (`get_objects`) and backward (`get_subjects`) for each query entity → collect chunks linked to related entities via `EntityIndex` → augment with vector seeds → rerank with `_select_cohort`.

Falls back to `vector_first` silently when prerequisites are absent (no `relationship_index`, no `entity_index`, or NER produces no entity hits).

### `global`

Searches only chunks with `chunk_type = "community_summary"`. Requires `build-community` to have run. Returns the top-k community summary chunks ranked by vector similarity.

---

## Key retrieval parameters

These are constructor arguments on `EnhancedSearch` unless noted otherwise.

| Parameter | Default | Description |
|---|---|---|
| `seed_pool_multiplier` | `3` | Seed pool size = `k × multiplier` |
| `entity_expansion_top_n` | `3` | Max chunks fetched per entity in entity expansion |
| `cluster_budget` | `2 × k` (resolved at call time) | Max cluster-adjacent candidates |
| `lambda_diversity` | `0.3` | MMR redundancy penalty weight |
| `relevance_weight` | `0.5` | Composite score weight for relevance |
| `priority_weight` | `0.2` | Composite score weight for source priority |
| `coverage_weight` | `0.3` | Composite score weight for marginal coverage |
| `lane_entity_min_sim` | `None` | Drop entity-linked chunks below this cosine similarity |
| `entity_ref_expansion` | `False` | Enable post-selection gap-fill for missing entities |
| `entity_ref_expansion_k` | `20` | Total expansion pool size for entity-ref gap-fill |
| `entity_ref_expansion_per_k` | `None` | Chunks fetched per missing entity (overrides k÷n split) |
| `entity_ref_expansion_min_sim` | `None` | Drop expansion hits below this cosine similarity |
| `structural_expansion` | `True` | Enable prev/next chunk expansion |
| `entity_expansion` | `True` | Enable entity-adjacency expansion |
| `cluster_expansion` | `True` | Enable cluster-adjacency expansion |

`top_k` and `fetch_k` (in `graphrag_bench.py`) correspond to the `k` argument to `search()` and the upstream reranker pool size, not `EnhancedSearch` constructor args.

Source priority constants (used in composite scoring, not configurable):

| Provenance | Priority |
|---|---|
| `seed` | 1.0 |
| `structural` | 0.9 |
| `entity_adjacent` | 0.7 |
| `cluster_adjacent` | 0.5 |

---

## Generation variants

These are features of `graphrag_bench.py`, not the library itself.

### Plain generation

Default. Retrieves context, sends to LLM, returns the text answer.

### `--sr` (structured response)

Instructs the LLM to return JSON with three fields: `answer`, `key_claims`, and `evidence_used`. No additional retrieval. The structured output is useful as a chain-of-thought signal and produces parseable evidence citations.

### `--srr` (structured response with coverage check)

SR plus a coverage-check loop (up to 2 rounds). After the first structured response, the key entities in the query are embedded and compared against the `evidence_used` field using cosine similarity (threshold `0.35`). Entities not covered trigger `_srr_gap_fill`, which retrieves additional chunks for each uncovered entity and regenerates. The loop runs at most 2 rounds.

`--srr` requires the embedding model at query time (to compute entity–evidence coverage scores). A cheaper model can be specified for coverage checks only via `--srr-model` and `--srr-provider`.

---

## TOML config (benchmark CLI)

`demo/graphrag_bench.py` accepts `--config path/to/file.toml`. The canonical base config is at `work/configs/base.toml`; per-run configs are in `work/configs/runs/`. Both are tracked in git.

### Schema

```toml
# Extend a parent config — path relative to this file
extends = "base.toml"   # optional

# Data sources — zero or more; processed in order
[[source]]
type      = "directory"
uri       = "/path/to/docs"
namespace = "internal"   # optional
domain    = "engineering"  # optional; paired with namespace, registered in domains table

[[source]]
type      = "github"
uri       = "https://github.com/myorg/myrepo"
namespace = "public"     # optional
domain    = "wiki"         # optional

# Entity vocabulary — zero or more; extends spaCy NER
# namespace is optional on every entry type; unset means "global"
[[vocab.entities]]
type        = "static"
entity_type = "customer"
names       = ["Acme Corp", "Globex Inc"]
namespace   = "finance"

[[vocab.entities]]
type        = "db_query"
entity_type = "employee"
connection  = "postgresql://user:pass@host/db"
sql         = "SELECT full_name FROM employees WHERE active = true"
namespace   = "hr"

[index]
out_dir            = "work"
db_name            = "chonk_nobc_1100_2200.duckdb"
embed_model        = "BAAI/bge-large-en-v1.5"
spacy_model        = "en_core_web_sm"
min_chunk          = 1100
max_chunk          = 2200
embed_content_only = true

[index.features]
ner            = true
ner_embeddings = true
schema_vocab   = false
community      = true
svo            = false

[rerank]
enabled  = false
provider = "local"
model    = "BAAI/bge-reranker-large"

[retrieval]
top_k                = 5
fetch_k              = 50
search_mode          = "vector_first"
enhanced             = false
entity_ref_expansion = false
lane_entity_min_sim  = null
redundancy_threshold = null
cluster              = false
vanilla              = false
namespaces           = null   # optional list of namespace strings to restrict retrieval
domain_ids           = null   # optional list of domain_id strings ("{namespace}:{domain}") to restrict retrieval

[retrieval.community]
enabled       = false
min_coherence = 0.5

[gen]
provider    = "openai"
model       = "gpt-4o-mini"
temperature = 0.0

[sr]
enabled = false

[srr]
enabled  = false
provider = null
model    = null

[eval]
judge       = "gpt-4o-mini"
rpm         = 8000
batch_size  = 20
concurrency = 50
nan_limit   = 136
```

### `extends` chain

A config may declare `extends = "relative/path/to/parent.toml"`. The parent is loaded first, then the child is deep-merged over it. Chains resolve recursively up to depth 5. Keys absent from the child fall through from the parent unchanged; keys present in the child override the parent at any nesting level.

---

## Config layering

Config inheritance and namespace hierarchy are intentionally isomorphic: the parent config defines what is globally shared; child configs extend it and add user-specific content.

**`global.toml`** defines shared data sources using `namespace = "global"`:

```toml
# global.toml
[[source]]
type      = "directory"
uri       = "/shared/knowledge-base"
namespace = "global"
domain    = "kb"

[[source]]
type      = "github"
uri       = "https://github.com/myorg/docs"
namespace = "global"
domain    = "engineering"
```

**User configs** use `extends = "global.toml"` and add user-specific sources:

```toml
# alice.toml
extends = "global.toml"

[[source]]
type      = "gmail"
uri       = "gmail://inbox"
namespace = "user:alice"
domain    = "email"
```

**`store.resolve_session()`** always folds in all domains from the `global` namespace in addition to the explicitly requested pairs:

```python
domain_ids = store.resolve_session(
    namespace_id="user:alice",
    active_domains=["email"],
    # global namespace is folded in automatically (include_global=True)
)
```

**The `extends` chain and the namespace hierarchy are isomorphic**: the parent config corresponds to the `global` namespace; child configs correspond to user-specific namespaces. Sources without an explicit `namespace` field default to `global` when loaded through `_apply_config`.

### Resolution order

Hardcoded defaults (constants at the top of `graphrag_bench.py`) < TOML config file < CLI flags.

`_apply_config()` applies TOML values only when the corresponding CLI flag was not explicitly set. This means passing `--top-k 10` on the command line beats a `top_k = 5` in TOML, but omitting `--top-k` lets TOML win.

### Batch runs

```bash
python demo/graphrag_bench.py run-all --config-dir work/configs/
```

`run-all` discovers every `.toml` file in the directory and runs each one sequentially as a separate `run` invocation.

---

## Storage backends

`Store` supports three storage backends selected by constructor arguments. All three expose the same `VectorBackend` protocol, so retrieval code is identical regardless of which backend is active.

| Backend | When selected | Install extra |
|---|---|---|
| `DuckDBVectorBackend` | default (no `dsn`, `qdrant_url`, `pinecone_api_key`, or `weaviate_url`) | `storage` |
| `PgVectorBackend` | `dsn=` set | `pgvector` |
| `QdrantVectorBackend` | `qdrant_url=` set | `qdrant` |
| `PineconeVectorBackend` | `pinecone_api_key=` set | `pinecone` |
| `WeaviateVectorBackend` | `weaviate_url=` set | `weaviate` |

### DuckDB (default)

Embedded; zero external infrastructure. Vectors stored in a HNSW index alongside chunks. Suitable for single-process workloads up to ~10 M chunks.

```python
store = Store("my.duckdb", embedding_dim=1024)
store = Store(":memory:", embedding_dim=1024)        # ephemeral
store = Store("my.duckdb", read_only=True)           # concurrent readers
```

### PostgreSQL + pgvector

Requires a running PostgreSQL instance with `pgvector` installed. Supports hybrid BM25 + ANN search.

```bash
pip install "chonk-rag[pgvector]"
```

```python
store = Store(dsn="postgresql://user:pass@host:5432/mydb", embedding_dim=1024)
```

The `pgvector` extension is created automatically on first connect if the connecting user has `CREATE EXTENSION` privileges.

### Qdrant

Vectors stored in a Qdrant collection; chunk metadata stored in a DuckDB sidecar file. Suited for large-scale or multi-process deployments.

```bash
pip install "chonk-rag[qdrant]"
# Start Qdrant (Docker):
docker run -p 6333:6333 qdrant/qdrant
```

```python
store = Store(
    qdrant_url="http://localhost:6333",
    qdrant_collection="chonk",           # default
    qdrant_catalog_path=":memory:",       # use a file path for persistence
    embedding_dim=1024,
)
```

**Persistence.** Qdrant data is durable on the Qdrant server. The DuckDB sidecar holds the catalog (chunk metadata, documents table, namespace/domain registry). Pass a file path to `qdrant_catalog_path` to persist it across restarts:

```python
store = Store(
    qdrant_url="http://localhost:6333",
    qdrant_catalog_path="/data/chonk_catalog.duckdb",
)
```

**Qdrant Cloud.** Pass `qdrant_api_key` with your cloud API key:

```python
store = Store(
    qdrant_url="https://my-cluster.qdrant.io",
    qdrant_api_key="my-api-key",
)
```

**`compact()`.** When `Store.delete_domain()` removes entries from the catalog, the corresponding Qdrant points become orphaned (they are silently excluded from search results via a catalog join). Call `backend.compact()` to physically remove them from Qdrant and reclaim storage:

```python
store.delete_domain("user:alice:old-project")
store.vector.compact()   # remove orphaned Qdrant points
```

**Hybrid search.** When `query_text` is passed to `search()`, the Qdrant backend runs BM25 on the DuckDB catalog sidecar (which holds full chunk text) and merges the results with Qdrant ANN via Reciprocal Rank Fusion — the same hybrid strategy used by `DuckDBVectorBackend`. If the DuckDB `fts` extension is unavailable, the backend falls back silently to pure ANN.

### Pinecone

Vectors and chunk payload stored in a Pinecone serverless index; catalog metadata stored in a DuckDB sidecar file. Suited for fully managed, serverless vector search at any scale.

```bash
pip install "chonk-rag[pinecone]"
```

```python
import os
store = Store(
    pinecone_api_key=os.environ["PINECONE_API_KEY"],
    pinecone_index="chonk",                # default
    pinecone_catalog_path=":memory:",      # use a file path for persistence
    pinecone_cloud="aws",                  # default
    pinecone_region="us-east-1",           # default
    embedding_dim=1024,
)
```

**Persistence.** Pinecone stores vectors durably on Pinecone Cloud. The DuckDB sidecar holds catalog tables. Pass a file path to `pinecone_catalog_path` to persist the catalog across restarts:

```python
store = Store(
    pinecone_api_key=os.environ["PINECONE_API_KEY"],
    pinecone_catalog_path="/data/chonk_pinecone.duckdb",
)
```

**`compact()`.** When `Store.delete_domain()` removes catalog entries, the corresponding Pinecone vectors become orphaned (silently excluded from search via catalog join). Call `backend.compact()` to physically delete them:

```python
store.delete_domain("user:alice:old-project")
store.vector.compact()   # remove orphaned Pinecone vectors
```

**Hybrid search.** When `query_text` is passed to `search()`, the Pinecone backend runs BM25 on the DuckDB catalog sidecar and merges with Pinecone ANN via Reciprocal Rank Fusion.

### Weaviate

Vectors and chunk payload stored in Weaviate Cloud; catalog metadata stored in a DuckDB sidecar file. Supports native BM25 and ANN search inside Weaviate, merged via Reciprocal Rank Fusion.

```bash
pip install "chonk-rag[weaviate]"
```

```python
import os
store = Store(
    weaviate_url=os.environ["WEAVIATE_URL"],       # https://... cluster URL
    weaviate_api_key=os.environ["WEAVIATE_API_KEY"],
    weaviate_collection="Chonk",                   # default
    weaviate_catalog_path=":memory:",              # use a file path for persistence
    embedding_dim=1024,
)
```

The `weaviate_url` must use the `https://` prefix (e.g. `https://abc123.c0.us-east-1.aws.weaviate.cloud`). Strip any `grpc-` prefix if copied from the Weaviate Cloud console.

**Persistence.** Weaviate stores vectors durably on Weaviate Cloud. Pass a file path to `weaviate_catalog_path` to persist the DuckDB catalog across restarts:

```python
store = Store(
    weaviate_url=os.environ["WEAVIATE_URL"],
    weaviate_api_key=os.environ["WEAVIATE_API_KEY"],
    weaviate_catalog_path="/data/chonk_weaviate.duckdb",
)
```

**`compact()`.** When `Store.delete_domain()` removes catalog entries, the corresponding Weaviate objects become orphaned (excluded from search results via catalog join). Call `backend.compact()` to physically delete them:

```python
store.delete_domain("user:alice:old-project")
store.vector.compact()   # remove orphaned Weaviate objects
```

**Hybrid search.** When `query_text` is passed to `search()`, the Weaviate backend runs both Weaviate's native ANN and native BM25 in parallel, then merges the results via Reciprocal Rank Fusion. No DuckDB `fts` extension is required.

---

## Library usage example

```python
import numpy as np
from chonk import Store, EnhancedSearch
from chonk.ner import (
    EntityIndex,
    SchemaVocabBuilder,
    SpacyMatcher,
    merge_matches,
)

# ── Index phase (run once) ────────────────────────────────────────────────
loader_chunks = [...]   # DocumentChunk list from DocumentLoader
embeddings    = np.array([...], dtype=np.float32)  # shape (n, 1024)

with Store("my.duckdb", embedding_dim=1024) as store:
    store.add_document(loader_chunks, embeddings)

# ── Build NER index (run once, write to DB) ───────────────────────────────
schema_chunks = [c for c in loader_chunks
                 if c.chunk_type in ("db_table", "db_column")]
builder = SchemaVocabBuilder()
builder.add_chunks(schema_chunks)
schema_matcher = builder.build()

spacy = SpacyMatcher(model="en_core_web_sm", strip_numeric=True)
entity_index = EntityIndex()

for chunk in loader_chunks:
    schema_hits = schema_matcher.match(chunk.content)
    spacy_hits  = spacy.match(chunk.content)
    combined    = merge_matches(schema_hits, spacy_hits, source_text=chunk.content)
    # chunk_id must match the ID used when calling store.add_document()
    entity_index.index_chunk(chunk_id, chunk.content, combined)

entity_index.recompute_scores()

# ── Query phase ───────────────────────────────────────────────────────────
query_ner_fn = lambda text: [m.display_name for m in spacy.match(text)]

with Store("my.duckdb", embedding_dim=1024, read_only=True) as store:
    search = EnhancedSearch(
        store,
        entity_index=entity_index,
        query_ner_fn=query_ner_fn,
        lane_entity_min_sim=0.45,
        entity_ref_expansion=True,
        entity_ref_expansion_k=20,
    )
    results = search.search(
        query_vec,          # np.ndarray shape (1024,)
        k=10,
        query_text="...",
        mode="vector_first",
    )
    for sc in results:
        print(sc.score, sc.chunk.content[:80])
```

All `EnhancedSearch` constructor parameters and their defaults are listed in the [key retrieval parameters](#key-retrieval-parameters) table above.

`graph_first` requires `build-svo` to have been run. Everything else is auto-loaded.

**Index-time prerequisite** (run once, after `build-ner`):

```bash
chonk build-svo --db my.duckdb --namespace global \
    --model gpt-4o-mini \
    --batch-size 32
```

**Query-time** — same `EnhancedSearch` constructor as all other modes:

```python
with Store("my.duckdb", read_only=True) as store:
    search = EnhancedSearch(store, embed_fn=embed_fn)
    results = search.search(
        k=10,
        query_text="Who supplies components to Acme Corp?",
        mode="graph_first",
    )
```

`EnhancedSearch` auto-loads `EntityIndex`, `RelationshipIndex`, and `CommunityIndex` from the store's DuckDB on construction — no manual wiring needed. `graph_first` falls back to `vector_first` silently if `build-svo` has not been run.
