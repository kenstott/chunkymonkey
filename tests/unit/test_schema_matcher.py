# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 62e8e7d7-5d5b-4582-8728-020d35c0ef96

"""Unit tests for SchemaMatcher and normalize_schema_term — Phase 1.4."""

from __future__ import annotations

from chonk import SchemaMatcher, normalize_schema_term
from chonk.ner import merge_matches

# ---------------------------------------------------------------------------
# normalize_schema_term
# ---------------------------------------------------------------------------


class TestNormalizeSchematerm:
    def test_underscore_to_spaces(self):
        assert normalize_schema_term("performance_reviews") == "performance reviews"

    def test_camel_case(self):
        assert normalize_schema_term("performanceReviews") == "performance reviews"

    def test_pascal_case(self):
        assert normalize_schema_term("PerformanceReviews") == "performance reviews"

    def test_all_caps_acronym(self):
        assert normalize_schema_term("HTMLParser") == "html parser"

    def test_kebab_case(self):
        assert normalize_schema_term("order-items") == "order items"

    def test_to_singular_strips_trailing_s(self):
        assert (
            normalize_schema_term("performance_reviews", to_singular=True) == "performance review"
        )

    def test_to_singular_no_s_unchanged(self):
        assert normalize_schema_term("employee", to_singular=True) == "employee"

    def test_single_word(self):
        assert normalize_schema_term("orders") == "orders"

    def test_single_word_singular(self):
        assert normalize_schema_term("orders", to_singular=True) == "order"

    def test_already_lowercase_spaces(self):
        assert normalize_schema_term("customer id") == "customer id"

    def test_mixed_underscores_and_camel(self):
        assert normalize_schema_term("customer_lastName") == "customer last name"


# ---------------------------------------------------------------------------
# SchemaMatcher — construction and entity_type assignment
# ---------------------------------------------------------------------------


class TestSchemaMatcherConstruction:
    def test_schema_terms_entity_type(self):
        m = SchemaMatcher(schema_terms=["performance_reviews"])
        results = m.match("The performance reviews are stored here.")
        assert len(results) == 1
        assert results[0].entity_type == "schema"

    def test_api_terms_entity_type(self):
        m = SchemaMatcher(api_terms=["getUser"])
        results = m.match("Call getUser to fetch the record.")
        assert len(results) == 1
        assert results[0].entity_type == "api"

    def test_business_terms_entity_type(self):
        m = SchemaMatcher(business_terms=["PII"])
        results = m.match("This field contains PII data.")
        assert len(results) == 1
        assert results[0].entity_type == "term"

    def test_all_three_term_types_together(self):
        m = SchemaMatcher(
            schema_terms=["employee_id"],
            api_terms=["getEmployee"],
            business_terms=["GDPR"],
        )
        text = "The employee_id field is GDPR-relevant; call getEmployee to retrieve it."
        results = m.match(text)
        types_found = {r.entity_type for r in results}
        assert "schema" in types_found
        assert "term" in types_found

    def test_empty_constructor_returns_no_matches(self):
        m = SchemaMatcher()
        assert m.match("anything") == []

    def test_none_args_treated_as_empty(self):
        m = SchemaMatcher(schema_terms=None, api_terms=None, business_terms=None)
        assert m.match("some text") == []


# ---------------------------------------------------------------------------
# SchemaMatcher — variant matching
# ---------------------------------------------------------------------------


class TestSchemaMatcherVariants:
    def test_matches_normalized_form(self):
        m = SchemaMatcher(schema_terms=["performance_reviews"])
        results = m.match("The performance reviews are stored.")
        assert len(results) == 1

    def test_matches_singular_form(self):
        m = SchemaMatcher(schema_terms=["performance_reviews"])
        results = m.match("Each performance review is logged.")
        assert len(results) == 1

    def test_matches_underscore_form(self):
        m = SchemaMatcher(schema_terms=["performance_reviews"])
        results = m.match("Column performance_reviews in the table.")
        assert len(results) == 1

    def test_matches_joined_form(self):
        m = SchemaMatcher(schema_terms=["performance_reviews"])
        results = m.match("The performancereviews table.")
        assert len(results) == 1

    def test_matches_joined_singular_form(self):
        m = SchemaMatcher(schema_terms=["performance_reviews"])
        results = m.match("A performancereview record.")
        assert len(results) == 1

    def test_case_insensitive(self):
        m = SchemaMatcher(schema_terms=["customer_id"])
        assert len(m.match("CUSTOMER_ID is the primary key")) == 1
        assert len(m.match("Customer_Id field")) == 1

    def test_no_match_on_substring(self):
        m = SchemaMatcher(schema_terms=["order"])
        # "orders" ends with "order" but "order" should not match inside "orders"
        # because "s" follows immediately (alphanumeric boundary check)
        results = m.match("the orderservice handles this")
        assert len(results) == 0

    def test_matches_multiple_occurrences(self):
        m = SchemaMatcher(schema_terms=["invoice"])
        results = m.match("invoice and invoice again")
        assert len(results) == 1
        assert results[0].frequency == 2
        assert len(results[0].spans) == 2

    def test_multiple_distinct_terms_all_matched(self):
        m = SchemaMatcher(schema_terms=["orders", "customers"])
        text = "Join orders with customers on customer_id."
        results = m.match(text)
        names = {r.name for r in results}
        assert "order" in names or "orders" in names
        assert "customer" in names or "customers" in names

    def test_entity_id_is_stable(self):
        m = SchemaMatcher(schema_terms=["employee_id"])
        r1 = m.match("employee_id is set")
        r2 = m.match("employee id is missing")
        assert r1[0].entity_id == r2[0].entity_id

    def test_positions_are_character_offsets(self):
        m = SchemaMatcher(schema_terms=["orders"])
        text = "The orders table"
        results = m.match(text)
        assert len(results) == 1
        pos = results[0].positions[0]
        assert text[pos : pos + len("orders")] == "orders"

    def test_spans_are_start_end_pairs(self):
        m = SchemaMatcher(schema_terms=["orders"])
        results = m.match("The orders table")
        span = results[0].spans[0]
        assert len(span) == 2
        assert span[1] > span[0]


# ---------------------------------------------------------------------------
# normalize_schema_term + SchemaMatcher — term without underscore
# ---------------------------------------------------------------------------


class TestNoUnderscoreTerm:
    def test_camel_matches_spaced(self):
        m = SchemaMatcher(schema_terms=["customerName"])
        results = m.match("The customer name field.")
        assert len(results) == 1

    def test_camel_matches_joined(self):
        m = SchemaMatcher(schema_terms=["customerName"])
        results = m.match("stored in customername")
        assert len(results) == 1

    def test_underscore_variant_still_not_generated(self):
        # _variants() only adds the literal underscore form when the source term
        # contains one — "customerName" contributes no "customer_name" key.
        from chonk.ner._schema import _variants

        assert "customer_name" not in _variants("customerName")

    def test_camel_matches_snake_case_in_text(self):
        # Separator normalisation makes the match symmetric: text "customer_name"
        # normalises to "customer name", which is the stored variant. This is the
        # equivalence normalize_schema_term() documents for underscores,
        # camelCase, PascalCase, and kebab-case.
        m = SchemaMatcher(schema_terms=["customerName"])
        results = m.match("field customer_name here")
        assert len(results) == 1
        # Span points at the original text, not the normalised copy.
        start, end = results[0].spans[0]
        assert "field customer_name here"[start:end] == "customer_name"

    def test_normalization_can_be_disabled(self):
        m = SchemaMatcher(schema_terms=["customerName"], normalize_separators=False)
        results = m.match("field customer_name here")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Integration: SchemaMatcher feeds merge_matches
# ---------------------------------------------------------------------------


class TestMergeIntegration:
    def test_schema_matches_survive_merge_when_no_overlap(self):
        from chonk.ner._vocabulary import EntityMatch

        schema_m = SchemaMatcher(schema_terms=["invoice_id"])
        text = "The invoice_id references London offices."
        schema_hits = schema_m.match(text)

        # Simulate a spaCy hit on "London" (no overlap with invoice_id)
        spacy_hit = EntityMatch(
            entity_id="ent_london",
            name="london",
            display_name="London",
            entity_type="GPE",
            frequency=1,
            positions=[text.index("London")],
            spans=[(text.index("London"), text.index("London") + 6)],
        )

        merged = merge_matches(schema_hits, [spacy_hit], source_text=text)
        entity_types = {m.entity_type for m in merged}
        assert "schema" in entity_types
        assert "GPE" in entity_types

    def test_schema_wins_on_overlap(self):
        from chonk.ner._vocabulary import EntityMatch

        schema_m = SchemaMatcher(schema_terms=["orders"])
        text = "The orders are tracked."
        schema_hits = schema_m.match(text)

        # Overlapping generic hit on the same "orders" span
        orders_start = text.index("orders")
        generic_hit = EntityMatch(
            entity_id="ent_orders_generic",
            name="orders",
            display_name="Orders",
            entity_type="ORG",
            frequency=1,
            positions=[orders_start],
            spans=[(orders_start, orders_start + 6)],
        )

        merged = merge_matches(schema_hits, [generic_hit], source_text=text)
        schema_results = [m for m in merged if m.entity_type == "schema"]
        org_results = [m for m in merged if m.entity_type == "ORG"]
        assert len(schema_results) == 1
        assert len(org_results) == 0
