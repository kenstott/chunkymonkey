# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 9be10bb0-7981-431f-b528-1253a4aeb26a
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""NerPipeline — unified two-pass NER with options.

Handles the common scenario (schema vocab + spaCy) behind one object::

    from chonk.ner import NerPipeline

    pipeline = NerPipeline(db_enrich=True, spacy_entities=True)

    # Feed schema sources (any combination)
    pipeline.add_tables(table_meta_list)       # from DB introspection
    pipeline.add_sql(open("schema.sql").read()) # from DDL files
    pipeline.add_chunks(schema_chunks)          # from loader.load_schema()

    # Run on document chunks (single call, everything merged internally)
    for chunk_id, chunk in indexed_chunks:
        matches = pipeline.match(chunk.content)
        entity_index.index_chunk(chunk_id, chunk.content, matches)

    # Or: run on all chunks at once and index in one shot
    pipeline.run_on_chunks(chunks, entity_index, chunk_ids)

Options
-------
db_enrich       Use schema/column/API terms as a custom vocabulary matcher.
                All identifiers are normalized (camelCase / snake_case /
                SCREAMING_SNAKE → "first name" style) so they match prose.
spacy_entities  Run spaCy NER.  Where both matchers fire on the same span,
                the schema vocab wins (schema entities suppress overlapping
                spaCy hits via merge_matches).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any  # noqa: F401

from ._merge import merge_matches
from ._schema_vocab import SchemaVocabBuilder
from ._vocabulary import EntityMatch, VocabularyMatcher

if TYPE_CHECKING:
    from ..models import DocumentChunk
    from ._index import EntityIndex
    from ._schema import SchemaMatcher
    from ._spacy import SpacyMatcher
    from ._spacy_labels import SpacyLabel


class NerPipeline:
    """Unified NER pipeline: schema vocab + data vocab + spaCy, all merged.

    Three matcher layers, applied in priority order (highest first):

    1. **Schema vocab** (``db_enrich=True``) — table/column/API identifier
       terms, normalised from camelCase/snake_case to prose form.
    2. **Data vocab** (``add_from_db()`` / ``add_entities()``) — actual values
       from database tables: customer names, employee names, counterparty names,
       ticker symbols, etc.  Matched verbatim (case-insensitive).
    3. **spaCy NER** (``spacy_entities=True``) — generic statistical NER.

    Schema and data vocab both suppress overlapping spaCy hits.

    Args:
        db_enrich: Match schema/column/API identifier terms.
            Feed via ``add_tables()``, ``add_sql()``, ``add_chunks()``.
        spacy_entities: Run spaCy NER.
        spacy_model: spaCy model name (default ``"en_core_web_sm"``).
        spacy_entity_types: Label whitelist for spaCy
            (e.g. ``["ORG", "PERSON", "GPE"]``). ``None`` keeps all 18 labels.
        min_schema_term_length: Drop schema identifiers shorter than this
            (default 2).
    """

    def __init__(
        self,
        db_enrich: bool = False,
        spacy_entities: bool = False,
        spacy_model: str = "en_core_web_sm",
        spacy_entity_types: list[str] | list[SpacyLabel] | None = None,
        min_schema_term_length: int = 2,
    ) -> None:
        self._db_enrich = db_enrich
        self._spacy_entities = spacy_entities
        self._spacy_model = spacy_model
        self._spacy_entity_types = spacy_entity_types
        self._builder = SchemaVocabBuilder(min_term_length=min_schema_term_length)
        self._schema_matcher: SchemaMatcher | None = None
        self._data_matcher: VocabularyMatcher | None = None
        self._spacy_matcher: SpacyMatcher | None = None
        self._schema_dirty = False
        self._data_dirty = False

    # ------------------------------------------------------------------
    # Schema identifier vocabulary
    # ------------------------------------------------------------------

    def add_tables(self, tables: list[object], namespace: str | None = None) -> NerPipeline:
        """Add table/column names from a list of TableMeta objects.

        Args:
            tables: TableMeta-like objects.
            namespace: Namespace that contributed the schema; unset means global.
        """
        self._builder.add_tables(tables, namespace=namespace)
        self._schema_dirty = True
        return self

    def add_endpoints(self, endpoints: list[object], namespace: str | None = None) -> NerPipeline:
        """Add path/field names from a list of EndpointMeta objects."""
        self._builder.add_endpoints(endpoints, namespace=namespace)
        self._schema_dirty = True
        return self

    def add_sql(self, ddl: str, namespace: str | None = None) -> NerPipeline:
        """Extract table and column names from raw SQL DDL text."""
        self._builder.add_sql(ddl, namespace=namespace)
        self._schema_dirty = True
        return self

    def add_chunks(self, chunks: list[object], namespace: str | None = None) -> NerPipeline:
        """Extract terms from DocumentChunk objects from load_schema()/load_api()."""
        self._builder.add_chunks(chunks, namespace=namespace)
        self._schema_dirty = True
        return self

    def add_business_terms(self, terms: list[str], namespace: str | None = None) -> NerPipeline:
        """Add glossary terms, matched as schema-shaped identifiers.

        Normalised, singularised, and variant-expanded like schema identifiers —
        ``"Customer Risk Score"`` matches ``customerRiskScore`` in prose. Use
        ``add_entities`` instead for literal data values. Entity type is ``"term"``.
        """
        self._builder.add_business_terms(terms, namespace=namespace)
        self._schema_dirty = True
        return self

    # ------------------------------------------------------------------
    # Data-value vocabulary
    # ------------------------------------------------------------------

    def add_entities(
        self,
        names: list[str],
        entity_type: str = "term",
        namespace: str | None = None,
    ) -> NerPipeline:
        """Add a plain list of known entity names (verbatim, case-insensitive).

        Use for any list you already have: customer names from a CRM export,
        ticker symbols from a spreadsheet, counterparty names from a config.

        Args:
            names: Display-form strings, e.g. ``["Acme Corp", "John Smith"]``.
            entity_type: Label on matching ``EntityMatch`` objects
                (e.g. ``"customer"``, ``"employee"``).
            namespace: Namespace that sourced these names. Persisted as an
                ``entity_aliases`` row per (entity, namespace); ``None`` files
                them under the ``global`` namespace.
        """
        self._builder.add_entities(names, entity_type=entity_type, namespace=namespace)
        self._data_dirty = True
        return self

    def add_from_db(
        self,
        connection: Any,  # noqa: ANN401
        queries: dict[str, str] | list[str] | list[tuple[str, str]],
        entity_type: str = "term",
        row_limit: int | None = None,
        namespace: str | None = None,
    ) -> NerPipeline:
        """Execute SQL queries against a live DB and add results as data vocab.

        Reuse an existing connection — pass the same engine or connection used
        to introspect the schema so no second connection is needed.

        Args:
            connection: SQLAlchemy URL string, Engine, Connection, or any
                object with ``.execute()``.
            queries: One of:

                - ``dict[entity_type, sql]`` —
                  ``{"customer": "SELECT name FROM customers"}``
                - ``list[str]`` — SQL strings; all use the ``entity_type`` arg
                - ``list[tuple[str, str]]`` —
                  ``[("SELECT name FROM customers", "customer")]``

            entity_type: Default label when ``queries`` is a plain list.
            row_limit: Maximum rows fetched per query. ``None`` (the default) fetches
                every row — a truncated vocabulary weakens matching silently.
            namespace: Namespace that sourced these values. Persisted as an
                ``entity_aliases`` row per (entity, namespace); ``None`` files
                them under the ``global`` namespace.
        """
        self._builder.add_from_db(
            connection,
            queries,
            entity_type=entity_type,
            row_limit=row_limit,
            namespace=namespace,
        )
        self._data_dirty = True
        return self

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def match(self, text: str) -> list[EntityMatch]:
        """Run all enabled matchers and return merged results.

        Priority: schema vocab = data vocab > spaCy.  Both vocab layers
        suppress overlapping spaCy hits; spaCy hits on non-overlapping spans
        survive.  Returns ``[]`` if no matchers are enabled.
        """
        vocab_hits: list[EntityMatch] = []

        if self._db_enrich:
            vocab_hits.extend(self._get_schema_matcher().match(text))

        if self._builder.data_term_count():
            vocab_hits.extend(self._get_data_matcher().match(text))

        spacy_hits: list[EntityMatch] = []
        if self._spacy_entities:
            spacy_hits = self._get_spacy_matcher().match(text)

        if vocab_hits and spacy_hits:
            return merge_matches(vocab_hits, spacy_hits, source_text=text)
        return vocab_hits or spacy_hits

    def run_on_chunks(
        self,
        chunks: list[DocumentChunk],
        entity_index: EntityIndex,
        chunk_ids: list[str] | None = None,
    ) -> dict[str, list[EntityMatch]]:
        """Run NER on every chunk and index results into *entity_index*.

        Args:
            chunks: List of DocumentChunk objects.
            entity_index: EntityIndex to record associations in.
            chunk_ids: Stable IDs parallel to *chunks*.  If ``None``,
                uses ``chunk.document_name + ":" + str(chunk.chunk_index)``.

        Returns:
            Mapping of ``chunk_id -> list[EntityMatch]`` for chunks with matches.
        """
        results: dict[str, list[EntityMatch]] = {}
        for i, chunk in enumerate(chunks):
            cid = (
                chunk_ids[i]
                if chunk_ids is not None
                else (f"{chunk.document_name}:{chunk.chunk_index}")
            )
            content = chunk.content or ""
            matches = self.match(content)
            if matches:
                entity_index.index_chunk(cid, content, matches)
                results[cid] = matches
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def term_counts(self) -> dict[str, int]:
        """Return counts of accumulated terms by category."""
        return {
            "tables": self._builder.table_count(),
            "columns": self._builder.column_count(),
            "api_terms": self._builder.api_term_count(),
            "data_terms": self._builder.data_term_count(),
        }

    # kept for backwards compatibility
    def schema_term_counts(self) -> dict[str, int]:
        return self.term_counts()

    # ------------------------------------------------------------------
    # Internal lazy initialisation
    # ------------------------------------------------------------------

    def _get_schema_matcher(self) -> SchemaMatcher:
        if self._schema_matcher is None or self._schema_dirty:
            self._schema_matcher = self._builder.build()
            self._schema_dirty = False
        return self._schema_matcher

    def _get_data_matcher(self) -> VocabularyMatcher:
        if self._data_matcher is None or self._data_dirty:
            self._data_matcher = self._builder.build_data_matcher()
            self._data_dirty = False
        return self._data_matcher

    def _get_spacy_matcher(self) -> SpacyMatcher:
        if self._spacy_matcher is None:
            from ._spacy import SpacyMatcher as _SpacyMatcher

            self._spacy_matcher = _SpacyMatcher(
                model=self._spacy_model,
                entity_types=self._spacy_entity_types,
            )
        return self._spacy_matcher
