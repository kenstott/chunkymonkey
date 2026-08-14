# Copyright (c) 2025 Kenneth Stott. MIT License.
"""Namespace tagging for entities sourced from a query or a static vocab list."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from chonk.ner._build import _build_vocab_matchers, _persist_associations
from chonk.ner._schema_vocab import SchemaVocabBuilder
from chonk.storage._schema import ENTITIES_DDL, ENTITY_ALIASES_DDL


def _make_source_db(tmp_path: Path) -> str:
    db_path = tmp_path / "source.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE customers (name VARCHAR)")
    con.execute("INSERT INTO customers VALUES ('Acme Corp'), ('Globex Inc')")
    con.close()
    return f"duckdb:///{db_path}"


class TestBuilderNamespaceTracking:
    def test_add_from_db_records_namespace(self, tmp_path):
        builder = SchemaVocabBuilder()
        builder.add_from_db(
            _make_source_db(tmp_path),
            {"customer": "SELECT name FROM customers"},
            namespace="ns_a",
        )
        assert builder.namespaced_entities() == [
            ("customer:acme_corp", "acme corp", "ns_a"),
            ("customer:globex_inc", "globex inc", "ns_a"),
        ]

    def test_no_namespace_defaults_to_global(self, tmp_path):
        builder = SchemaVocabBuilder()
        builder.add_from_db(_make_source_db(tmp_path), {"customer": "SELECT name FROM customers"})
        assert builder.namespaced_entities() == [
            ("customer:acme_corp", "acme corp", "global"),
            ("customer:globex_inc", "globex inc", "global"),
        ]
        assert builder.data_term_count() == 2

    def test_explicit_global_and_default_collapse_to_one_tag(self):
        builder = SchemaVocabBuilder()
        builder.add_entities(["Acme Corp"], entity_type="customer")
        builder.add_entities(["Acme Corp"], entity_type="customer", namespace="global")
        assert builder.namespaced_entities() == [("customer:acme_corp", "acme corp", "global")]

    def test_same_name_two_namespaces_yields_one_entity_two_tags(self):
        builder = SchemaVocabBuilder()
        builder.add_entities(["Acme Corp"], entity_type="customer", namespace="ns_a")
        builder.add_entities(["Acme Corp"], entity_type="customer", namespace="ns_b")

        tags = builder.namespaced_entities()
        assert tags == [
            ("customer:acme_corp", "acme corp", "ns_a"),
            ("customer:acme_corp", "acme corp", "ns_b"),
        ]
        assert len({entity_id for entity_id, _alias, _ns in tags}) == 1

    def test_duplicate_name_same_namespace_deduped(self):
        builder = SchemaVocabBuilder()
        builder.add_entities(["Acme Corp", "Acme Corp"], entity_type="customer", namespace="ns_a")
        assert builder.namespaced_entities() == [("customer:acme_corp", "acme corp", "ns_a")]

    def test_vocab_entries_pass_namespace_through(self, tmp_path):
        _schema, _data, tags = _build_vocab_matchers(
            [],
            use_schema_vocab=False,
            vocab_entities=[
                {
                    "type": "static",
                    "entity_type": "customer",
                    "names": ["Acme Corp"],
                    "namespace": "ns_a",
                },
                {
                    "type": "db_query",
                    "entity_type": "customer",
                    "connection": _make_source_db(tmp_path),
                    "sql": "SELECT name FROM customers",
                    "namespace": "ns_b",
                },
            ],
        )
        assert ("customer:acme_corp", "acme corp", "ns_a") in tags
        assert ("customer:acme_corp", "acme corp", "ns_b") in tags
        assert ("customer:globex_inc", "globex inc", "ns_b") in tags


class TestPersistNamespaceTags:
    @pytest.fixture()
    def con(self, tmp_path):
        con = duckdb.connect(str(tmp_path / "index.duckdb"))
        con.execute(ENTITIES_DDL)
        con.execute(ENTITY_ALIASES_DDL)
        con.execute(
            "CREATE TABLE chunk_entities (chunk_id TEXT, entity_id TEXT, "
            "frequency INTEGER, positions_json TEXT, score REAL, namespace TEXT, "
            "PRIMARY KEY (chunk_id, entity_id))"
        )
        con.execute("CREATE TABLE embeddings (chunk_id TEXT, namespace TEXT)")
        con.execute("INSERT INTO embeddings VALUES ('c1', 'ns_a')")
        yield con
        con.close()

    def _data(self, entity_id: str = "customer:acme_corp") -> dict[str, list[dict[str, object]]]:
        return {
            "associations": [
                {
                    "chunk_id": "c1",
                    "entity_id": entity_id,
                    "frequency": 1,
                    "positions": [],
                    "score": 1.0,
                }
            ]
        }

    def test_tag_written_for_matched_entity(self, con):
        _persist_associations(
            con,
            self._data(),
            {"customer:acme_corp": ("acme corp", "Acme Corp", "customer")},
            incremental=False,
            namespaced_entities=[("customer:acme_corp", "acme corp", "ns_a")],
        )
        rows = con.execute(
            "SELECT alias, entity_id, namespace, source FROM entity_aliases"
        ).fetchall()
        assert rows == [("acme corp", "customer:acme_corp", "ns_a", "vocab_source")]

    def test_same_name_two_namespaces_persists_both(self, con):
        _persist_associations(
            con,
            self._data(),
            {"customer:acme_corp": ("acme corp", "Acme Corp", "customer")},
            incremental=False,
            namespaced_entities=[
                ("customer:acme_corp", "acme corp", "ns_a"),
                ("customer:acme_corp", "acme corp", "ns_b"),
            ],
        )
        rows = con.execute(
            "SELECT namespace FROM entity_aliases "
            "WHERE entity_id = 'customer:acme_corp' ORDER BY namespace"
        ).fetchall()
        assert [r[0] for r in rows] == ["ns_a", "ns_b"]
        entities = con.execute("SELECT COUNT(*) FROM entities").fetchone()
        assert entities[0] == 1

    def test_unmatched_entity_is_not_tagged(self, con):
        _persist_associations(
            con,
            self._data(),
            {"customer:acme_corp": ("acme corp", "Acme Corp", "customer")},
            incremental=False,
            namespaced_entities=[("customer:globex_inc", "globex inc", "ns_a")],
        )
        rows = con.execute(
            "SELECT COUNT(*) FROM entity_aliases WHERE entity_id = 'customer:globex_inc'"
        ).fetchone()
        assert rows[0] == 0

    def test_alias_shared_by_two_entities_in_one_namespace(self, con):
        """One person can be both a customer and an employee — not a collision."""
        con.execute(
            "INSERT INTO entity_aliases(alias, entity_id, namespace, source) "
            "VALUES ('acme corp', 'employee:acme_corp', 'ns_a', 'llm')"
        )
        _persist_associations(
            con,
            self._data(),
            {"customer:acme_corp": ("acme corp", "Acme Corp", "customer")},
            incremental=False,
            namespaced_entities=[("customer:acme_corp", "acme corp", "ns_a")],
        )
        rows = con.execute(
            "SELECT entity_id FROM entity_aliases WHERE alias = 'acme corp' "
            "AND namespace = 'ns_a' ORDER BY entity_id"
        ).fetchall()
        assert [r[0] for r in rows] == ["customer:acme_corp", "employee:acme_corp"]

    def test_rerun_is_idempotent(self, con):
        for _ in range(2):
            _persist_associations(
                con,
                self._data(),
                {"customer:acme_corp": ("acme corp", "Acme Corp", "customer")},
                incremental=False,
                namespaced_entities=[("customer:acme_corp", "acme corp", "ns_a")],
            )
        count = con.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()
        assert count[0] == 1


class TestStoreEntityNamespaces:
    def test_reverse_lookup_returns_all_namespaces(self, tmp_path):
        from chonk.storage._store import Store

        store = Store(tmp_path / "idx.duckdb")
        try:
            store.add_entity_alias(
                "acme corp", "customer:acme_corp", source="vocab", namespace="ns_b"
            )
            store.add_entity_alias(
                "acme corp", "customer:acme_corp", source="vocab", namespace="ns_a"
            )
            assert store.get_entity_namespaces("customer:acme_corp") == ["ns_a", "ns_b"]
            assert store.get_entity_namespaces("unknown") == []
        finally:
            store.close()


class TestDocumentGeneratedAliasNamespace:
    """strip_suffix aliases inherit the namespace the chunk was crawled under."""

    @pytest.fixture()
    def con(self, tmp_path):
        con = duckdb.connect(str(tmp_path / "index.duckdb"))
        con.execute(ENTITIES_DDL)
        con.execute(ENTITY_ALIASES_DDL)
        con.execute(
            "CREATE TABLE chunk_entities (chunk_id TEXT, entity_id TEXT, "
            "frequency INTEGER, positions_json TEXT, score REAL, namespace TEXT, "
            "PRIMARY KEY (chunk_id, entity_id))"
        )
        con.execute("CREATE TABLE embeddings (chunk_id TEXT, namespace TEXT)")
        yield con
        con.close()

    def _assoc(self, chunk_id: str, entity_id: str) -> dict[str, object]:
        return {
            "chunk_id": chunk_id,
            "entity_id": entity_id,
            "frequency": 1,
            "positions": [],
            "score": 1.0,
        }

    def test_alias_uses_chunk_namespace(self, con):
        con.execute("INSERT INTO embeddings VALUES ('c1', 'ns_a')")
        _persist_associations(
            con,
            {"associations": [self._assoc("c1", "customer_id")]},
            {},
            incremental=False,
            namespaced_entities=[],
        )
        rows = con.execute(
            "SELECT alias, entity_id, namespace, source FROM entity_aliases"
        ).fetchall()
        assert rows == [("customer", "customer_id", "ns_a", "strip_suffix")]

    def test_alias_defaults_to_global_without_chunk_namespace(self, con):
        con.execute("INSERT INTO embeddings VALUES ('c1', NULL)")
        _persist_associations(
            con,
            {"associations": [self._assoc("c1", "customer_id")]},
            {},
            incremental=False,
            namespaced_entities=[],
        )
        rows = con.execute("SELECT namespace FROM entity_aliases").fetchall()
        assert [r[0] for r in rows] == ["global"]

    def test_same_alias_in_two_namespaces_kept_separately(self, con):
        con.execute("INSERT INTO embeddings VALUES ('c1', 'ns_a'), ('c2', 'ns_b')")
        _persist_associations(
            con,
            {
                "associations": [
                    self._assoc("c1", "customer_id"),
                    self._assoc("c2", "customer_id"),
                ]
            },
            {},
            incremental=False,
            namespaced_entities=[],
        )
        rows = con.execute(
            "SELECT namespace FROM entity_aliases WHERE alias = 'customer' ORDER BY namespace"
        ).fetchall()
        assert [r[0] for r in rows] == ["ns_a", "ns_b"]


class TestEntityMetadataNamespaceFilter:
    """Entity metadata lookups honour a namespace restriction."""

    @pytest.fixture()
    def store(self, tmp_path):
        from chonk.storage._store import Store

        store = Store(tmp_path / "idx.duckdb")
        conn = store.vector._conn
        conn.execute(
            "INSERT INTO entities(id, name, display_name, entity_type, description) VALUES "
            "('customer:acme_corp', 'acme corp', 'Acme Corp', 'customer', 'a customer'), "
            "('customer:globex_inc', 'globex inc', 'Globex Inc', 'customer', 'another customer')"
        )
        conn.execute(
            "INSERT INTO chunk_entities(chunk_id, entity_id, frequency, positions_json, "
            "score, namespace) VALUES "
            "('c1', 'customer:acme_corp', 1, '[]', 1.0, 'ns_a'), "
            "('c2', 'customer:globex_inc', 1, '[]', 1.0, 'ns_b')"
        )
        yield store
        store.close()

    def test_descriptions_unfiltered_returns_all(self, store):
        got = store.get_entity_descriptions(["customer:acme_corp", "customer:globex_inc"])
        assert set(got) == {"customer:acme_corp", "customer:globex_inc"}

    def test_descriptions_filtered_by_namespace(self, store):
        got = store.get_entity_descriptions(
            ["customer:acme_corp", "customer:globex_inc"], namespaces=["ns_a"]
        )
        assert got == {"customer:acme_corp": "a customer"}

    def test_descriptions_empty_namespace_list_matches_nothing(self, store):
        assert store.get_entity_descriptions(["customer:acme_corp"], namespaces=[]) == {}

    def test_null_namespace_treated_as_global(self, store):
        store.vector._conn.execute(
            "INSERT INTO chunk_entities(chunk_id, entity_id, frequency, positions_json, "
            "score, namespace) VALUES ('c3', 'customer:acme_corp', 1, '[]', 1.0, NULL)"
        )
        got = store.get_entity_descriptions(["customer:acme_corp"], namespaces=["global"])
        assert got == {"customer:acme_corp": "a customer"}

    def test_fetch_entity_records_filtered(self, store):
        from chonk.search._enhanced_graph import _GraphMixin

        mixin = _GraphMixin.__new__(_GraphMixin)
        mixin._store = store

        unfiltered = mixin._fetch_entity_records({"customer:acme_corp", "customer:globex_inc"})
        assert {r["id"] for r in unfiltered} == {"customer:acme_corp", "customer:globex_inc"}

        filtered = mixin._fetch_entity_records(
            {"customer:acme_corp", "customer:globex_inc"}, ["ns_b"]
        )
        assert [r["id"] for r in filtered] == ["customer:globex_inc"]

        assert mixin._fetch_entity_records({"customer:acme_corp"}, []) == []


class TestBuildNerEndToEnd:
    """Full build_ner pass: vocab namespace tags and doc-derived alias namespaces."""

    def test_vocab_and_document_aliases_are_namespaced(self, tmp_path):
        pytest.importorskip("spacy")
        pytest.importorskip("en_core_web_sm")

        import numpy as np

        from chonk.models import DocumentChunk
        from chonk.ner._build import build_ner
        from chonk.storage._store import Store

        store = Store(tmp_path / "idx.duckdb", embedding_dim=4)
        try:
            chunks = [
                DocumentChunk(
                    document_name="doc_a",
                    content="Acme Corp filed the report.",
                    chunk_index=0,
                    section=[],
                ),
                DocumentChunk(
                    document_name="doc_b",
                    content="Acme Corp also appears here.",
                    chunk_index=0,
                    section=[],
                ),
            ]
            embeddings = np.ones((2, 4), dtype="float32")
            store.add_document([chunks[0]], embeddings[:1], namespace="ns_a")
            store.add_document([chunks[1]], embeddings[1:], namespace="ns_b")

            build_ner(
                store,
                vocab_entities=[
                    {
                        "type": "static",
                        "entity_type": "customer",
                        "names": ["Acme Corp"],
                        "namespace": "ns_a",
                    }
                ],
                namespace="ns_a",
            )

            assert store.get_entity_namespaces("customer:acme_corp") == ["ns_a"]
            rows = store.vector._conn.execute(
                "SELECT source FROM entity_aliases WHERE entity_id = 'customer:acme_corp'"
            ).fetchall()
            assert [r[0] for r in rows] == ["vocab_source"]

            # The entity matched chunks in both namespaces, so a namespace-scoped
            # metadata lookup finds it under either.
            assert store.get_entity_descriptions(["customer:acme_corp"], namespaces=["ns_b"]) != {}
            assert store.get_entity_descriptions(["customer:acme_corp"], namespaces=["ns_c"]) == {}
        finally:
            store.close()


class TestTypedEntityIds:
    """Entity type is part of entity identity."""

    def test_same_name_different_types_are_distinct_entities(self):
        builder = SchemaVocabBuilder()
        builder.add_entities(["Mercury"], entity_type="customer", namespace="sales")
        builder.add_entities(["Mercury"], entity_type="element", namespace="chem")

        tags = builder.namespaced_entities()
        assert [t[0] for t in tags] == ["customer:mercury", "element:mercury"]

    def test_distinct_types_persist_as_two_entity_rows(self, tmp_path):
        import duckdb

        from chonk.storage._schema import ENTITIES_DDL, ENTITY_ALIASES_DDL

        con = duckdb.connect(str(tmp_path / "idx.duckdb"))
        con.execute(ENTITIES_DDL)
        con.execute(ENTITY_ALIASES_DDL)
        con.execute(
            "CREATE TABLE chunk_entities (chunk_id TEXT, entity_id TEXT, frequency INTEGER, "
            "positions_json TEXT, score REAL, namespace TEXT, PRIMARY KEY (chunk_id, entity_id))"
        )
        con.execute("CREATE TABLE embeddings (chunk_id TEXT, namespace TEXT)")
        con.execute("INSERT INTO embeddings VALUES ('c1', 'sales'), ('c2', 'chem')")
        try:
            _persist_associations(
                con,
                {
                    "associations": [
                        {
                            "chunk_id": "c1",
                            "entity_id": "customer:mercury",
                            "frequency": 1,
                            "positions": [],
                            "score": 1.0,
                        },
                        {
                            "chunk_id": "c2",
                            "entity_id": "element:mercury",
                            "frequency": 1,
                            "positions": [],
                            "score": 1.0,
                        },
                    ]
                },
                {
                    "customer:mercury": ("mercury", "Mercury", "customer"),
                    "element:mercury": ("mercury", "Mercury", "element"),
                },
                incremental=False,
                namespaced_entities=[
                    ("customer:mercury", "mercury", "sales"),
                    ("element:mercury", "mercury", "chem"),
                ],
            )
            rows = con.execute("SELECT id, entity_type FROM entities ORDER BY id").fetchall()
            assert rows == [
                ("customer:mercury", "customer"),
                ("element:mercury", "element"),
            ]
            # Same alias string, different namespaces → both survive.
            aliases = con.execute(
                "SELECT entity_id, namespace FROM entity_aliases WHERE alias = 'mercury' "
                "ORDER BY namespace"
            ).fetchall()
            assert aliases == [("element:mercury", "chem"), ("customer:mercury", "sales")]
        finally:
            con.close()

    def test_strip_id_alias_drops_type_prefix(self):
        from chonk.ner._build import _strip_id_alias

        assert _strip_id_alias("term:customer_id") == "customer"
        assert _strip_id_alias("org:acme_ref") == "acme"
        assert _strip_id_alias("org:acme") is None


class TestSameNameTwoTypesOneNamespace:
    """John Doe works at Walmart and shops there: customer AND employee."""

    def test_matcher_returns_both_entities_for_one_mention(self):
        builder = SchemaVocabBuilder()
        builder.add_entities(["John Doe"], entity_type="customer", namespace="walmart")
        builder.add_entities(["John Doe"], entity_type="employee", namespace="walmart")

        matches = builder.build_data_matcher().match("John Doe bought a drill.")
        assert sorted(m.entity_id for m in matches) == [
            "customer:john_doe",
            "employee:john_doe",
        ]
        # The same mention is evidence for both — identical spans, not split.
        assert {tuple(m.spans) for m in matches} == {((0, 8),)}

    def test_both_tagged_under_the_same_namespace(self):
        builder = SchemaVocabBuilder()
        builder.add_entities(["John Doe"], entity_type="customer", namespace="walmart")
        builder.add_entities(["John Doe"], entity_type="employee", namespace="walmart")
        assert builder.namespaced_entities() == [
            ("customer:john_doe", "john doe", "walmart"),
            ("employee:john_doe", "john doe", "walmart"),
        ]

    def test_store_round_trip_keeps_both_mappings(self, tmp_path):
        from chonk.storage._store import Store

        store = Store(tmp_path / "idx.duckdb")
        try:
            store.add_entity_alias("john doe", "customer:john_doe", namespace="walmart")
            store.add_entity_alias("john doe", "employee:john_doe", namespace="walmart")

            assert store.resolve_entity_aliases("john doe", namespace="walmart") == [
                "customer:john_doe",
                "employee:john_doe",
            ]
            # Singular accessor still answers, deterministically.
            assert store.resolve_entity_alias("john doe", "walmart") == "customer:john_doe"
            assert store.get_entity_namespaces("employee:john_doe") == ["walmart"]
        finally:
            store.close()

    def test_re_registering_same_mapping_is_idempotent(self, tmp_path):
        from chonk.storage._store import Store

        store = Store(tmp_path / "idx.duckdb")
        try:
            for _ in range(3):
                store.add_entity_alias("john doe", "customer:john_doe", namespace="walmart")
            assert store.resolve_entity_aliases("john doe", "walmart") == ["customer:john_doe"]
        finally:
            store.close()

    def test_batch_writes_one_alias_per_entity(self, tmp_path):
        from chonk.storage._store import Store

        store = Store(tmp_path / "idx.duckdb")
        try:
            assert (
                store.add_entity_aliases_batch(
                    {"john doe": "customer:john_doe"}, namespace="walmart"
                )
                == 1
            )
            assert (
                store.add_entity_aliases_batch(
                    {"john doe": "employee:john_doe"}, namespace="walmart"
                )
                == 1
            )
            # Re-running the first is skipped, not counted.
            assert (
                store.add_entity_aliases_batch(
                    {"john doe": "customer:john_doe"}, namespace="walmart"
                )
                == 0
            )
            assert len(store.resolve_entity_aliases("john doe", "walmart")) == 2
        finally:
            store.close()


class TestGlossaryAndSchemaTermNamespaces:
    """Glossary terms are reachable, and schema-shaped terms carry a namespace."""

    def test_glossary_terms_reach_the_matcher(self):
        builder = SchemaVocabBuilder()
        builder.add_business_terms(["Customer Risk Score"], namespace="risk")
        matches = builder.build().match("the customer risk score was high")
        assert [(m.entity_id, m.entity_type) for m in matches] == [
            ("term:customer_risk_score", "term")
        ]

    def test_glossary_is_normalised_unlike_static_vocab(self):
        """Glossary goes through SchemaMatcher: camelCase in prose still matches."""
        builder = SchemaVocabBuilder()
        builder.add_business_terms(["Customer Risk Score"])
        assert builder.build().match("the customerRiskScore field") != []

        verbatim = SchemaVocabBuilder()
        verbatim.add_entities(["Customer Risk Score"], entity_type="term")
        assert verbatim.build_data_matcher().match("the customerRiskScore field") == []

    def test_schema_terms_are_namespaced(self):
        builder = SchemaVocabBuilder()
        # Columns are matched at line start, so the DDL must be multi-line.
        builder.add_sql("CREATE TABLE customers (\n    full_name VARCHAR\n);", namespace="retail")
        tags = builder.namespaced_entities()
        assert ("schema:customer", "customer", "retail") in tags
        assert ("schema:full_name", "full name", "retail") in tags

    def test_glossary_terms_are_namespaced(self):
        builder = SchemaVocabBuilder()
        builder.add_business_terms(["Wire Transfer"], namespace="ops")
        assert builder.namespaced_entities() == [("term:wire_transfer", "wire transfer", "ops")]

    def test_unset_namespace_defaults_to_global(self):
        builder = SchemaVocabBuilder()
        builder.add_business_terms(["Wire Transfer"])
        assert builder.namespaced_entities() == [("term:wire_transfer", "wire transfer", "global")]

    def test_one_term_from_two_namespaces_yields_a_tag_each(self):
        builder = SchemaVocabBuilder()
        builder.add_business_terms(["Wire Transfer"], namespace="ops")
        builder.add_business_terms(["Wire Transfer"], namespace="risk")
        assert builder.namespaced_entities() == [
            ("term:wire_transfer", "wire transfer", "ops"),
            ("term:wire_transfer", "wire transfer", "risk"),
        ]

    def test_glossary_config_entry_is_wired(self):
        _schema, _data, tags = _build_vocab_matchers(
            [],
            use_schema_vocab=False,
            vocab_entities=[
                {"type": "glossary", "names": ["Wire Transfer"], "namespace": "ops"},
            ],
        )
        assert ("term:wire_transfer", "wire transfer", "ops") in tags
        assert _schema is not None
        assert _schema.match("a wire transfer was sent") != []


class TestVocabEntryValidation:
    """A dropped or mislabelled vocab entry yields a plausible-looking empty index."""

    def _build(self, entry):
        return _build_vocab_matchers([], use_schema_vocab=False, vocab_entities=[entry])

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unknown type 'lookup'"):
            self._build({"type": "lookup", "entity_type": "customer", "names": ["Acme"]})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="unknown type None"):
            self._build({"entity_type": "customer", "names": ["Acme"]})

    def test_error_names_the_offending_entry(self):
        with pytest.raises(ValueError, match=r"vocab_entities\[1\]"):
            _build_vocab_matchers(
                [],
                use_schema_vocab=False,
                vocab_entities=[
                    {"type": "static", "entity_type": "customer", "names": ["Acme"]},
                    {"type": "typo", "entity_type": "customer", "names": ["Globex"]},
                ],
            )

    def test_missing_entity_type_raises_rather_than_defaulting(self):
        """entity_type is part of the id, so a default is a different entity."""
        with pytest.raises(ValueError, match="entity_type is required"):
            self._build({"type": "static", "names": ["Acme"]})

    def test_glossary_rejects_an_entity_type_it_would_ignore(self):
        with pytest.raises(ValueError, match="glossary entries always carry"):
            self._build({"type": "glossary", "entity_type": "customer", "names": ["Acme"]})

    def test_missing_names_raises(self):
        with pytest.raises(ValueError, match="non-empty 'names'"):
            self._build({"type": "static", "entity_type": "customer"})

    def test_empty_names_raises(self):
        with pytest.raises(ValueError, match="non-empty 'names'"):
            self._build({"type": "static", "entity_type": "customer", "names": []})

    def test_db_query_requires_connection_and_sql(self):
        with pytest.raises(ValueError, match="requires a non-empty 'sql'"):
            self._build({"type": "db_query", "entity_type": "c", "connection": "duckdb://"})
        with pytest.raises(ValueError, match="requires a non-empty 'connection'"):
            self._build({"type": "db_query", "entity_type": "c", "sql": "SELECT 1"})

    def test_valid_entries_still_build(self):
        _schema, data, tags = _build_vocab_matchers(
            [],
            use_schema_vocab=False,
            vocab_entities=[
                {"type": "static", "entity_type": "customer", "names": ["Acme Corp"]},
                {"type": "glossary", "names": ["Wire Transfer"], "namespace": "ops"},
            ],
        )
        assert data is not None
        assert ("customer:acme_corp", "acme corp", "global") in tags
        assert ("term:wire_transfer", "wire transfer", "ops") in tags
