# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 19e1bf25-199b-4a20-b6ef-f7ad1e8e558d
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SchemaVocabBuilder — two-pass NER vocabulary extraction.

Builds a SchemaMatcher from structured schema metadata (TableMeta/ColumnMeta)
or raw SQL DDL text.  All identifiers are normalized via normalize_schema_term
before being added to the vocab, so camelCase, snake_case, and SCREAMING_SNAKE
all resolve to "first name" style surface forms.

Typical two-pass usage::

    from chonk.ner import SchemaVocabBuilder, SpacyMatcher, merge_matches

    # Pass 1: build vocab from DB metadata or DDL files
    builder = SchemaVocabBuilder()
    builder.add_tables(table_meta_list)          # from DB introspection
    builder.add_sql(ddl_text)                    # from .sql files
    builder.add_chunks(schema_chunks)            # from loader.load_schema()
    schema_matcher = builder.build()

    # Pass 2: combined NER on all document chunks
    spacy = SpacyMatcher()
    for chunk_id, chunk in indexed_chunks:
        schema_hits = schema_matcher.match(chunk.content)
        spacy_hits  = spacy.match(chunk.content)
        combined    = merge_matches(schema_hits, spacy_hits, source_text=chunk.content)
        entity_index.index_chunk(chunk_id, chunk.content, combined)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ._schema import SchemaMatcher
from ._vocabulary import VocabularyMatcher, _typed_id

# Namespace an unscoped vocabulary term is filed under. Matches the
# entity_aliases.namespace column default in chonk/storage/_schema.py.
DEFAULT_NAMESPACE = "global"

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# DB connection helpers
# ---------------------------------------------------------------------------


def _resolve_connection(connection: Any) -> Any:  # noqa: ANN401
    """Return a usable connection object from a URL string or engine/connection."""
    if isinstance(connection, str):
        try:
            import sqlalchemy as sa
        except ImportError as exc:
            raise ImportError(
                "sqlalchemy is required for add_from_db(). Install with: pip install sqlalchemy"
            ) from exc
        engine = sa.create_engine(connection)
        return engine.connect()
    # Engine: has connect() but not execute()
    if hasattr(connection, "connect") and not hasattr(connection, "execute"):
        return connection.connect()
    # Already a connection or duck-typed object with execute()
    if hasattr(connection, "execute"):
        return connection
    raise TypeError(
        f"add_from_db() connection must be a URL string, SQLAlchemy Engine, "
        f"or an object with .execute(). Got: {type(connection)}"
    )


def _execute(conn: Any, sql: str) -> list[str]:  # noqa: ANN401
    """Execute *sql*, validate single-column result, deduplicate, drop nulls.

    Raises:
        ValueError: if the query returns more than one column.
    """
    try:
        import sqlalchemy as sa

        result = conn.execute(sa.text(sql))
    except ImportError:
        result = conn.execute(sql)

    rows = list(result)
    if not rows:
        return []

    # Validate single column
    try:
        col_count = len(rows[0])
    except TypeError:
        col_count = 1  # scalar rows
    if col_count != 1:
        raise ValueError(
            f"add_from_db() queries must return exactly one column (got {col_count}). SQL: {sql!r}"
        )

    # Drop nulls, stringify, deduplicate preserving order
    seen: set[str] = set()
    values: list[str] = []
    for row in rows:
        val = row[0]
        if val is None:
            continue
        s = str(val).strip()
        if s and s not in seen:
            seen.add(s)
            values.append(s)
    return values


def _maybe_close(conn: Any, original: Any) -> None:  # noqa: ANN401
    """Close conn only if we created it (i.e. original was a URL string or Engine)."""
    if isinstance(original, str) or (
        hasattr(original, "connect") and not hasattr(original, "execute")
    ):
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SQL DDL parsing helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?"
    r"(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r'(?:"?\w+"?\.)?"?([a-zA-Z_]\w*)"?',
    re.IGNORECASE,
)

_ALTER_TABLE_RE = re.compile(
    r'ALTER\s+TABLE\s+(?:"?\w+"?\.)?"?([a-zA-Z_]\w*)"?',
    re.IGNORECASE,
)

# SQL primitive type keywords — column lines must start with an identifier
# followed by one of these to be recognised as a column definition.
_SQL_TYPES = (
    "INTEGER",
    "INT",
    "BIGINT",
    "SMALLINT",
    "TINYINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "NUMERIC",
    "REAL",
    "MONEY",
    "BOOLEAN",
    "BOOL",
    "TEXT",
    "VARCHAR",
    "CHAR",
    "NCHAR",
    "NVARCHAR",
    "CLOB",
    "BLOB",
    "BYTEA",
    "BINARY",
    "VARBINARY",
    "DATE",
    "TIME",
    "TIMESTAMP",
    "TIMESTAMPTZ",
    "INTERVAL",
    "DATETIME",
    "UUID",
    "JSON",
    "JSONB",
    "XML",
    "SERIAL",
    "BIGSERIAL",
    "SMALLSERIAL",
    "AUTOINCREMENT",
    "ARRAY",
    "HSTORE",
    "GEOMETRY",
    "GEOGRAPHY",
)

# Compiled: column-start line pattern
_COL_RE = re.compile(
    r'^\s*"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s+'
    r"(?:" + "|".join(_SQL_TYPES) + r")",
    re.IGNORECASE | re.MULTILINE,
)

# Lines beginning with these are DDL constraint/index clauses, not columns.
_SKIP_STARTS = re.compile(
    r"^\s*(?:PRIMARY|FOREIGN|UNIQUE|CHECK|INDEX|KEY|CONSTRAINT|REFERENCES|"
    r"COMMENT|ON\s|WITH\s|USING\s)",
    re.IGNORECASE,
)


def _extract_sql_terms(sql: str) -> tuple[list[str], list[str]]:
    """Return (table_names, column_names) extracted from SQL DDL text."""
    tables: list[str] = []
    columns: list[str] = []

    for m in _CREATE_TABLE_RE.finditer(sql):
        tables.append(m.group(1))
    for m in _ALTER_TABLE_RE.finditer(sql):
        name = m.group(1)
        if name not in tables:
            tables.append(name)

    for m in _COL_RE.finditer(sql):
        line = sql[m.start() : sql.find("\n", m.start())]
        if _SKIP_STARTS.match(line):
            continue
        columns.append(m.group(1))

    return tables, columns


# ---------------------------------------------------------------------------
# SchemaVocabBuilder
# ---------------------------------------------------------------------------


class SchemaVocabBuilder:
    """Accumulate schema identifiers and produce a SchemaMatcher.

    All identifiers are deduplicated (by raw value) before being passed to
    SchemaMatcher, which applies normalize_schema_term internally.

    Args:
        min_term_length: Raw identifier length below which terms are ignored
            (default 2 — filters out single-letter columns like ``i`` or ``n``).
    """

    def __init__(self, min_term_length: int = 2) -> None:
        self._min = min_term_length
        self._tables: set[str] = set()
        self._columns: set[str] = set()
        self._api_terms: set[str] = set()
        # Data-value vocab: list of (display_name, entity_type, namespace) tuples.
        # namespace is None when the term is not scoped to one.
        # Uses VocabularyMatcher (plain case-insensitive), not SchemaMatcher
        # (which normalises camelCase/snake_case — wrong for "Acme Corp").
        self._data_terms: list[tuple[str, str, str | None]] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_tables(self, tables: list[object]) -> SchemaVocabBuilder:
        """Add terms from a list of TableMeta objects (and their ColumnMeta).

        Accepts any object with a ``.name`` attribute and an optional
        ``.columns`` iterable of objects with a ``.name`` attribute.
        """
        for table in tables:
            name = getattr(table, "name", None)
            if name and len(name) >= self._min:
                self._tables.add(name)
            for col in getattr(table, "columns", None) or []:
                col_name = getattr(col, "name", None)
                if col_name and len(col_name) >= self._min:
                    self._columns.add(col_name)
        return self

    def add_endpoints(self, endpoints: list[object]) -> SchemaVocabBuilder:
        """Add terms from a list of EndpointMeta / FieldMeta objects."""
        for ep in endpoints:
            path = getattr(ep, "path", None)
            if path and len(path) >= self._min:
                self._api_terms.add(path)
            for field in getattr(ep, "fields", None) or []:
                name = getattr(field, "name", None)
                if name and len(name) >= self._min:
                    self._columns.add(name)
        return self

    def add_sql(self, ddl: str) -> SchemaVocabBuilder:
        """Extract and add table/column names from raw SQL DDL text."""
        tables, columns = _extract_sql_terms(ddl)
        for t in tables:
            if len(t) >= self._min:
                self._tables.add(t)
        for c in columns:
            if len(c) >= self._min:
                self._columns.add(c)
        return self

    def add_chunks(self, chunks: list[Any]) -> SchemaVocabBuilder:  # noqa: ANN401
        """Extract terms from DocumentChunk objects produced by load_schema() / load_api().

        Document names follow the pattern:
        - ``schema:<db>.<table>``       → table term
        - ``schema:<db>.<table>.<col>`` → column term
        - ``api:<api>.<path>.<field>``  → api / column term

        Chunks with other document_name patterns are silently skipped.
        """
        for chunk in chunks:
            doc_name: str = getattr(chunk, "document_name", "") or ""
            chunk_type: str = str(getattr(chunk, "chunk_type", "") or "")

            if chunk_type in ("db_table", "db_column") and doc_name.startswith("schema:"):
                parts = doc_name[len("schema:") :].split(".")
                # parts: [db, table] or [db, table, column]
                if len(parts) >= 2:
                    table = parts[1]
                    if len(table) >= self._min:
                        self._tables.add(table)
                if len(parts) >= 3:
                    col = parts[2]
                    if len(col) >= self._min:
                        self._columns.add(col)

            elif chunk_type in (
                "api_endpoint",
                "api_graphql_query",
                "api_graphql_mutation",
                "api_graphql_type",
            ):
                if doc_name.startswith("api:"):
                    parts = doc_name[len("api:") :].split(".", 1)
                    if len(parts) >= 2:
                        path = parts[1]
                        if len(path) >= self._min:
                            self._api_terms.add(path)

            elif chunk_type == "api_field" and doc_name.startswith("api:"):
                parts = doc_name[len("api:") :].split(".")
                if len(parts) >= 3:
                    field = parts[-1]
                    if len(field) >= self._min:
                        self._columns.add(field)

        return self

    def add_entities(
        self,
        names: list[str],
        entity_type: str = "term",
        namespace: str | None = None,
    ) -> SchemaVocabBuilder:
        """Add a plain list of entity names as data-value vocab.

        Unlike schema terms, these are matched verbatim (case-insensitive)
        — no camelCase or snake_case splitting.  Use for customer names,
        employee names, counterparty names, product names, etc.

        Args:
            names: Display-form strings (e.g. ``["Acme Corp", "John Smith"]``).
            entity_type: Label stored on matching ``EntityMatch`` objects
                (e.g. ``"customer"``, ``"employee"``).  Default ``"term"``.
            namespace: Namespace that sourced these names.  Recorded as an
                ``entity_aliases`` row per (entity, namespace) when the vocab is
                persisted.  ``None`` files them under ``DEFAULT_NAMESPACE``.
        """
        for name in names:
            name = name.strip()
            if len(name) >= self._min:
                self._data_terms.append((name, entity_type, namespace))
        return self

    def add_from_db(
        self,
        connection: Any,  # noqa: ANN401
        queries: dict[str, str] | list[str] | list[tuple[str, str]],
        entity_type: str = "term",
        row_limit: int | None = None,
        namespace: str | None = None,
    ) -> SchemaVocabBuilder:
        """Execute SQL queries and add result values as data-value vocab.

        Unlike schema terms (table/column names), these values are matched
        verbatim — no camelCase splitting.  Suitable for customer names,
        employee names, counterparty names, ticker symbols, etc.

        Args:
            connection: One of:
                - SQLAlchemy connection URL string
                - SQLAlchemy ``Engine`` (has ``.connect()``)
                - SQLAlchemy ``Connection`` or any object with ``.execute()``
            queries: One of:
                - ``dict[entity_type, sql]`` — e.g.
                  ``{"customer": "SELECT name FROM customers"}``
                - ``list[str]`` — SQL strings; all use ``entity_type`` arg
                - ``list[tuple[str, str]]`` — ``[(sql, entity_type), ...]``
            entity_type: Default entity type when ``queries`` is a list of strings.
            row_limit: Maximum rows fetched per query. ``None`` (the default) fetches
                every row. A vocabulary is a set, so a partial one silently weakens
                matching rather than failing — cap only when the caller means to.
            namespace: Namespace that sourced these values.  Recorded as an
                ``entity_aliases`` row per (entity, namespace) when the vocab is
                persisted.  ``None`` files them under ``DEFAULT_NAMESPACE``.

        Returns:
            ``self`` for chaining.
        """
        # Normalise queries to list of (sql, entity_type)
        if isinstance(queries, dict):
            pairs: list[tuple[str, str]] = [(sql, etype) for etype, sql in queries.items()]
        elif queries and isinstance(queries[0], tuple):
            pairs = list(queries)  # type: ignore[arg-type]
        else:
            pairs = [(sql, entity_type) for sql in queries]  # type: ignore[union-attr]

        conn = _resolve_connection(connection)
        try:
            for sql, etype in pairs:
                wrapped = f"SELECT * FROM ({sql}) _q"
                if row_limit is not None:
                    wrapped += f" LIMIT {row_limit}"
                # _execute validates single column, drops nulls, deduplicates
                values = _execute(conn, wrapped)
                for val in values:
                    if len(val) >= self._min:
                        self._data_terms.append((val, etype, namespace))
        finally:
            _maybe_close(conn, connection)

        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> SchemaMatcher:
        """Return a SchemaMatcher for schema/column/API identifier terms.

        SchemaMatcher applies normalize_schema_term internally, so
        ``firstName``, ``first_name``, and ``FIRST_NAME`` all produce the
        surface form "first name" and match as the same entity.
        """
        return SchemaMatcher(
            schema_terms=list(self._tables) + list(self._columns),
            api_terms=list(self._api_terms),
        )

    def build_data_matcher(self) -> VocabularyMatcher:
        """Return a VocabularyMatcher for data-value terms (plain matching).

        Data values (customer names, employee names, etc.) are matched
        verbatim and case-insensitively — no camelCase normalisation.
        """
        entities = [
            {
                "id": _typed_id(name, etype),
                "name": name.lower(),
                "display_name": name,
                "type": etype,
                "aliases": [],
            }
            for name, etype, _ns in self._data_terms
        ]
        return VocabularyMatcher(
            entities, match_mode="case_insensitive", min_entity_length=self._min
        )

    def namespaced_entities(self) -> list[tuple[str, str, str]]:
        """Return ``(entity_id, alias, namespace)`` for every data term.

        The alias is the lower-cased surface form — the same value stored in
        ``entities.name``.  An entity id is ``"{entity_type}:{name_slug}"``: the
        namespace is not part of it, so one name and type sourced from two
        namespaces yields one entity and two rows here, one per namespace; two
        types under one name yield two entities.  A term added without a namespace
        resolves to ``DEFAULT_NAMESPACE`` — the same default the ``entity_aliases``
        column carries.  Deduplicated, ordered by first insertion.
        """
        seen: set[tuple[str, str, str]] = set()
        out: list[tuple[str, str, str]] = []
        for name, _etype, ns in self._data_terms:
            row = (
                _typed_id(name, _etype),
                name.lower(),
                ns if ns is not None else DEFAULT_NAMESPACE,
            )
            if row not in seen:
                seen.add(row)
                out.append(row)
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def table_count(self) -> int:
        return len(self._tables)

    def column_count(self) -> int:
        return len(self._columns)

    def api_term_count(self) -> int:
        return len(self._api_terms)

    def data_term_count(self) -> int:
        return len(self._data_terms)
