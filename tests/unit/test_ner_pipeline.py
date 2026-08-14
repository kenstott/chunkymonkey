# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 20988944-4fc2-4299-bd0a-487b5d812cd2
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holders.

"""Tests for SchemaVocabBuilder and NerPipeline."""

from __future__ import annotations

from chonk.ner import NerPipeline, SchemaVocabBuilder
from chonk.ner._schema_vocab import _extract_sql_terms

# ---------------------------------------------------------------------------
# normalize_schema_term via SchemaVocabBuilder + SchemaMatcher
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_snake_case_matches_prose(self):
        b = SchemaVocabBuilder()
        b._record(b._columns, "first_name", None)
        m = b.build()
        assert len(m.match("The first name field is required.")) == 1

    def test_camel_case_matches_prose(self):
        b = SchemaVocabBuilder()
        b._record(b._columns, "firstName", None)
        m = b.build()
        assert len(m.match("Enter the first name here.")) == 1

    def test_screaming_snake_matches_prose(self):
        b = SchemaVocabBuilder()
        b._record(b._columns, "FIRST_NAME", None)
        m = b.build()
        assert len(m.match("The first name is stored.")) == 1

    def test_pascal_case_matches_prose(self):
        b = SchemaVocabBuilder()
        b._record(b._columns, "EmployeeId", None)
        m = b.build()
        assert len(m.match("employee id is the primary key.")) == 1


# ---------------------------------------------------------------------------
# _extract_sql_terms
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS order_items (
    order_id      INTEGER NOT NULL,
    product_id    INTEGER NOT NULL,
    quantity      INTEGER DEFAULT 1,
    unit_price    NUMERIC(10,2),
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

CREATE TABLE customers (
    id            SERIAL PRIMARY KEY,
    firstName     VARCHAR(100),
    last_name     TEXT,
    emailAddress  TEXT UNIQUE
);

ALTER TABLE order_items ADD COLUMN discount NUMERIC;
"""


class TestExtractSqlTerms:
    def test_table_names_extracted(self):
        tables, _ = _extract_sql_terms(DDL)
        assert "order_items" in tables
        assert "customers" in tables

    def test_alter_table_name_extracted(self):
        tables, _ = _extract_sql_terms(DDL)
        assert "order_items" in tables

    def test_column_names_extracted(self):
        _, cols = _extract_sql_terms(DDL)
        assert "order_id" in cols
        assert "product_id" in cols
        assert "quantity" in cols
        assert "firstName" in cols
        assert "last_name" in cols

    def test_constraint_lines_not_extracted_as_columns(self):
        _, cols = _extract_sql_terms(DDL)
        assert "FOREIGN" not in cols
        assert "PRIMARY" not in cols


# ---------------------------------------------------------------------------
# SchemaVocabBuilder.add_sql
# ---------------------------------------------------------------------------


class TestSchemaVocabBuilderSql:
    def test_add_sql_populates_tables(self):
        b = SchemaVocabBuilder().add_sql(DDL)
        assert b.table_count() >= 2

    def test_add_sql_populates_columns(self):
        b = SchemaVocabBuilder().add_sql(DDL)
        assert b.column_count() > 0

    def test_matcher_matches_table_name_prose(self):
        b = SchemaVocabBuilder().add_sql(DDL)
        m = b.build()
        results = m.match("The order items are stored in the order_items table.")
        assert len(results) >= 1

    def test_matcher_matches_snake_column_in_prose(self):
        b = SchemaVocabBuilder().add_sql(DDL)
        m = b.build()
        results = m.match("The unit price is stored as a decimal.")
        assert any("unit" in r.name for r in results)


# ---------------------------------------------------------------------------
# SchemaVocabBuilder.add_tables  (using plain dicts as duck-type TableMeta)
# ---------------------------------------------------------------------------


class _FakeCol:
    def __init__(self, name):
        self.name = name


class _FakeTable:
    def __init__(self, name, cols):
        self.name = name
        self.columns = [_FakeCol(c) for c in cols]


class TestSchemaVocabBuilderTables:
    def test_table_name_counted(self):
        b = SchemaVocabBuilder()
        b.add_tables([_FakeTable("performance_reviews", [])])
        assert b.table_count() == 1

    def test_column_names_counted(self):
        b = SchemaVocabBuilder()
        b.add_tables([_FakeTable("t", ["employeeId", "reviewScore"])])
        assert b.column_count() == 2

    def test_table_and_column_match_prose(self):
        b = SchemaVocabBuilder()
        b.add_tables([_FakeTable("performance_reviews", ["employeeId"])])
        m = b.build()
        assert len(m.match("performance review scores per employee id")) >= 1

    def test_short_terms_filtered(self):
        b = SchemaVocabBuilder(min_term_length=3)
        b.add_tables([_FakeTable("id", ["a", "bb", "ccc"])])
        # "id" (len 2) and "a"/"bb" (len < 3) excluded; "ccc" (len 3) included
        assert b.table_count() == 0
        assert b.column_count() == 1

    def test_chaining(self):
        b = SchemaVocabBuilder()
        result = b.add_tables([_FakeTable("t", [])]).add_sql("CREATE TABLE t2 (col1 TEXT);")
        assert result is b


# ---------------------------------------------------------------------------
# SchemaVocabBuilder.add_chunks (duck-type DocumentChunk)
# ---------------------------------------------------------------------------


class _FakeChunk:
    def __init__(self, doc_name, chunk_type):
        self.document_name = doc_name
        self.chunk_type = chunk_type
        self.content = ""
        self.chunk_index = 0


class TestSchemaVocabBuilderChunks:
    def test_db_table_chunk_adds_table(self):
        chunk = _FakeChunk("schema:mydb.invoices", "db_table")
        b = SchemaVocabBuilder().add_chunks([chunk])
        assert b.table_count() == 1

    def test_db_column_chunk_adds_column(self):
        chunk = _FakeChunk("schema:mydb.invoices.invoice_date", "db_column")
        b = SchemaVocabBuilder().add_chunks([chunk])
        assert b.column_count() == 1

    def test_api_field_chunk_adds_column(self):
        chunk = _FakeChunk("api:myapi./users.emailAddress", "api_field")
        b = SchemaVocabBuilder().add_chunks([chunk])
        assert b.column_count() == 1

    def test_unknown_chunk_type_ignored(self):
        chunk = _FakeChunk("doc:something", "document")
        b = SchemaVocabBuilder().add_chunks([chunk])
        assert b.table_count() == 0
        assert b.column_count() == 0


# ---------------------------------------------------------------------------
# NerPipeline — db_enrich only
# ---------------------------------------------------------------------------


class TestNerPipelineDbEnrich:
    def test_no_options_returns_empty(self):
        p = NerPipeline()
        assert p.match("The invoice_id is the primary key.") == []

    def test_db_enrich_matches_schema_terms(self):
        p = NerPipeline(db_enrich=True)
        p.add_tables([_FakeTable("invoices", ["invoice_id", "customerName"])])
        results = p.match("The customer name is on the invoice.")
        assert len(results) >= 1

    def test_db_enrich_add_sql(self):
        p = NerPipeline(db_enrich=True)
        p.add_sql("CREATE TABLE orders (order_id INTEGER, total_amount NUMERIC);")
        results = p.match("The total amount for each order is tracked.")
        assert len(results) >= 1

    def test_db_enrich_add_chunks(self):
        chunk = _FakeChunk("schema:db.employees.first_name", "db_column")
        p = NerPipeline(db_enrich=True)
        p.add_chunks([chunk])
        results = p.match("Enter the first name.")
        assert len(results) >= 1

    def test_schema_term_counts(self):
        p = NerPipeline(db_enrich=True)
        p.add_tables([_FakeTable("orders", ["order_id", "amount"])])
        counts = p.schema_term_counts()
        assert counts["tables"] == 1
        assert counts["columns"] == 2

    def test_dirty_flag_rebuilds_matcher(self):
        p = NerPipeline(db_enrich=True)
        p.add_tables([_FakeTable("orders", [])])
        _ = p.match("orders are tracked")  # builds matcher
        p.add_tables([_FakeTable("invoices", [])])  # dirty
        results = p.match("invoices and orders")
        names = {r.name for r in results}
        # Both tables should be found after rebuild
        assert any("invoice" in n for n in names)


# ---------------------------------------------------------------------------
# NerPipeline — run_on_chunks
# ---------------------------------------------------------------------------


class TestNerPipelineRunOnChunks:
    def test_run_on_chunks_indexes_matches(self):
        from chonk.ner import EntityIndex

        p = NerPipeline(db_enrich=True)
        p.add_tables([_FakeTable("invoices", ["invoice_id"])])

        class _Chunk:
            document_name = "doc"
            chunk_index = 0
            content = "The invoice id is the primary key of invoices."

        idx = EntityIndex()
        results = p.run_on_chunks([_Chunk()], idx)
        assert len(results) == 1
        assert idx.total_chunks() == 1

    def test_run_on_chunks_custom_ids(self):
        from chonk.ner import EntityIndex

        p = NerPipeline(db_enrich=True)
        p.add_tables([_FakeTable("orders", [])])

        class _Chunk:
            document_name = "doc"
            chunk_index = 0
            content = "The orders are tracked."

        idx = EntityIndex()
        results = p.run_on_chunks([_Chunk()], idx, chunk_ids=["custom-id-1"])
        assert "custom-id-1" in results

    def test_run_on_chunks_no_match_not_indexed(self):
        from chonk.ner import EntityIndex

        p = NerPipeline(db_enrich=True)
        p.add_tables([_FakeTable("invoices", [])])

        class _Chunk:
            document_name = "doc"
            chunk_index = 0
            content = "Nothing relevant here at all."

        idx = EntityIndex()
        results = p.run_on_chunks([_Chunk()], idx)
        assert len(results) == 0
        assert idx.total_chunks() == 0
