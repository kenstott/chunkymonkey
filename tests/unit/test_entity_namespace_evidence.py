# Copyright (c) 2025 Kenneth Stott. MIT License.
"""get_entity_namespace_evidence — which namespaces actually mention an entity."""

from __future__ import annotations

import numpy as np
import pytest

from chonk.models import DocumentChunk
from chonk.storage._store import NamespaceEvidence, Store

DIM = 4
EID = "customer:acme_corp"


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "idx.duckdb", embedding_dim=DIM)
    yield store
    store.close()


def _seed(store, plan: list[tuple[str, str]]) -> list[str]:
    """plan is [(namespace, content), ...]; returns chunk_ids in insertion order."""
    for i, (ns, content) in enumerate(plan):
        store.add_document(
            [DocumentChunk(document_name=f"d{i}", content=content, chunk_index=0, section=[])],
            np.ones((1, DIM), dtype="float32"),
            namespace=ns,
        )
    rows = store.vector._conn.execute(
        "SELECT chunk_id FROM embeddings ORDER BY document_name"
    ).fetchall()
    return [r[0] for r in rows]


def _associate(store, pairs: list[tuple[str, str, float]]) -> None:
    """pairs is [(chunk_id, namespace, score), ...]."""
    for chunk_id, ns, score in pairs:
        store.vector._conn.execute(
            "INSERT INTO chunk_entities(chunk_id, entity_id, frequency, positions_json, "
            "score, namespace) VALUES (?, ?, 1, '[]', ?, ?)",
            [chunk_id, EID, score, ns],
        )


class TestEvidence:
    def test_ranks_namespaces_and_normalizes_by_corpus_size(self, store):
        # retail: 2 of 2 chunks mention it. support: 1 of 4.
        ids = _seed(
            store,
            [
                ("retail", "a"),
                ("retail", "b"),
                ("support", "c"),
                ("support", "d"),
                ("support", "e"),
                ("support", "f"),
            ],
        )
        _associate(
            store, [(ids[0], "retail", 0.8), (ids[1], "retail", 0.8), (ids[2], "support", 0.4)]
        )

        ev = store.get_entity_namespace_evidence(EID)
        assert [e.namespace for e in ev] == ["retail", "support"]
        assert [e.chunk_count for e in ev] == [2, 1]
        assert ev[0].share == pytest.approx(1.0)
        assert ev[1].share == pytest.approx(0.25)
        assert ev[0].score == pytest.approx(1.6)

    def test_share_beats_raw_count_for_a_small_focused_namespace(self, store):
        # boutique: 2 of 2 (all about it). bulk: 3 of 100 (barely mentions it).
        plan = [("boutique", "a"), ("boutique", "b")] + [("bulk", f"n{i}") for i in range(100)]
        _seed(store, plan)
        rows = store.vector._conn.execute("SELECT chunk_id, namespace FROM embeddings").fetchall()
        boutique = [c for c, n in rows if n == "boutique"]
        bulk = [c for c, n in rows if n == "bulk"]
        _associate(store, [(c, "boutique", 0.5) for c in boutique])
        _associate(store, [(c, "bulk", 0.5) for c in bulk[:3]])

        ev = {e.namespace: e for e in store.get_entity_namespace_evidence(EID)}
        # Raw count says bulk; share says boutique. Both are exposed.
        assert ev["bulk"].chunk_count > ev["boutique"].chunk_count
        assert ev["boutique"].share > ev["bulk"].share
        assert ev["boutique"].share == pytest.approx(1.0)
        assert ev["bulk"].share == pytest.approx(0.03)

    def test_unknown_entity_returns_empty(self, store):
        _seed(store, [("retail", "a")])
        assert store.get_entity_namespace_evidence("customer:nobody") == []

    def test_null_namespace_counts_as_global(self, store):
        ids = _seed(store, [(None, "a"), (None, "b")])
        _associate(store, [(ids[0], None, 0.5)])
        ev = store.get_entity_namespace_evidence(EID)
        assert [e.namespace for e in ev] == ["global"]
        assert ev[0].share == pytest.approx(0.5)

    def test_returns_namespace_evidence_objects(self, store):
        ids = _seed(store, [("retail", "a")])
        _associate(store, [(ids[0], "retail", 0.5)])
        (ev,) = store.get_entity_namespace_evidence(EID)
        assert isinstance(ev, NamespaceEvidence)
        assert (ev.namespace, ev.chunk_count) == ("retail", 1)

    def test_ordering_is_deterministic_on_ties(self, store):
        ids = _seed(store, [("bbb", "a"), ("aaa", "b")])
        _associate(store, [(ids[0], "bbb", 0.5), (ids[1], "aaa", 0.5)])
        assert [e.namespace for e in store.get_entity_namespace_evidence(EID)] == ["aaa", "bbb"]


class TestDeclarationVsEvidence:
    def test_declaration_and_evidence_answer_different_questions(self, store):
        ids = _seed(store, [("retail", "a"), ("support", "b")])
        _associate(store, [(ids[0], "retail", 0.5), (ids[1], "support", 0.5)])
        # Declared once, centrally.
        store.add_entity_alias("acme corp", EID, source="vocab_source", namespace="global")

        assert store.get_entity_namespaces(EID) == ["global"]
        assert [e.namespace for e in store.get_entity_namespace_evidence(EID)] == [
            "retail",
            "support",
        ]


class TestResolveEntityIds:
    """A caller should not have to know the type prefix."""

    @pytest.fixture()
    def populated(self, store):
        for eid, etype in [
            ("customer:mercury", "customer"),
            ("element:mercury", "element"),
            ("customer:acme_corp", "customer"),
        ]:
            store.vector._conn.execute(
                "INSERT INTO entities(id, name, display_name, entity_type) VALUES (?, ?, ?, ?)",
                [eid, eid.split(":")[1].replace("_", " "), eid, etype],
            )
        return store

    def test_unqualified_name_returns_every_type(self, populated):
        assert populated.resolve_entity_ids("Mercury") == [
            "customer:mercury",
            "element:mercury",
        ]

    def test_entity_type_narrows(self, populated):
        assert populated.resolve_entity_ids("Mercury", entity_type="customer") == [
            "customer:mercury"
        ]

    def test_qualified_id_passes_through(self, populated):
        assert populated.resolve_entity_ids("element:mercury") == ["element:mercury"]

    def test_qualified_id_that_does_not_exist_returns_empty(self, populated):
        assert populated.resolve_entity_ids("ghost:mercury") == []

    def test_surface_variants_normalise_to_the_same_slug(self, populated):
        for surface in ("Acme Corp", "acme corp", "acme_corp", "ACME  CORP"):
            assert populated.resolve_entity_ids(surface) == ["customer:acme_corp"], surface

    def test_unknown_name_returns_empty(self, populated):
        assert populated.resolve_entity_ids("nobody") == []

    def test_does_not_match_a_name_that_merely_ends_with_the_slug(self, populated):
        populated.vector._conn.execute(
            "INSERT INTO entities(id, name, display_name, entity_type) "
            "VALUES ('customer:big_mercury', 'big mercury', 'Big Mercury', 'customer')"
        )
        assert populated.resolve_entity_ids("Mercury") == [
            "customer:mercury",
            "element:mercury",
        ]


class TestExplainEntityLookup:
    """An empty result has two causes; the caller should be able to tell them apart."""

    @pytest.fixture()
    def populated(self, store):
        for eid, etype in [
            ("customer:mercury", "customer"),
            ("element:mercury", "element"),
            ("customer:mercury_systems", "customer"),
            ("customer:acme_corp", "customer"),
        ]:
            store.vector._conn.execute(
                "INSERT INTO entities(id, name, display_name, entity_type) VALUES (?, ?, ?, ?)",
                [eid, eid.split(":")[1].replace("_", " "), eid, etype],
            )
        return store

    def test_genuine_miss_reports_nothing_found(self, populated):
        result = populated.explain_entity_lookup("Nobody")
        assert result.ids == []
        assert result.name_exists is False
        assert result.available_types == []
        assert result.near_matches == []

    def test_filter_excluded_it_is_distinguishable_from_a_miss(self, populated):
        """The case an empty list cannot express: present, but not under that type."""
        result = populated.explain_entity_lookup("Mercury", entity_type="ghost")
        assert result.ids == []
        assert result.name_exists is True
        assert result.available_types == ["customer", "element"]

    def test_hit_agrees_with_resolve_entity_ids(self, populated):
        for name, etype in [("Mercury", None), ("Mercury", "customer"), ("Nobody", None)]:
            assert populated.explain_entity_lookup(
                name, entity_type=etype
            ).ids == populated.resolve_entity_ids(name, entity_type=etype), (name, etype)

    def test_near_matches_offer_partial_names(self, populated):
        result = populated.explain_entity_lookup("Mercury", entity_type="customer")
        assert result.ids == ["customer:mercury"]
        assert result.near_matches == ["customer:mercury_systems"]

    def test_near_matches_exclude_the_ids_returned(self, populated):
        result = populated.explain_entity_lookup("Mercury")
        assert set(result.ids).isdisjoint(result.near_matches)

    def test_truncation_is_reported_not_hidden(self, populated):
        from chonk.storage._store import NEAR_MATCH_LIMIT

        for i in range(NEAR_MATCH_LIMIT + 3):
            populated.vector._conn.execute(
                "INSERT INTO entities(id, name, display_name, entity_type) VALUES (?, ?, ?, ?)",
                [f"customer:mercury_v{i}", f"mercury v{i}", f"Mercury v{i}", "customer"],
            )
        result = populated.explain_entity_lookup("Mercury")
        assert len(result.near_matches) == NEAR_MATCH_LIMIT
        assert result.near_matches_truncated is True

    def test_no_truncation_flag_when_all_fit(self, populated):
        assert populated.explain_entity_lookup("Mercury").near_matches_truncated is False


class TestNamespaceFilteredLookup:
    """Namespace is an evidence filter — where the entity actually appears."""

    @pytest.fixture()
    def scoped(self, store):
        ids = _seed(store, [("retail", "a"), ("support", "b")])
        for eid, etype in [("customer:mercury", "customer"), ("element:mercury", "element")]:
            store.vector._conn.execute(
                "INSERT INTO entities(id, name, display_name, entity_type) VALUES (?, ?, ?, ?)",
                [eid, "mercury", eid, etype],
            )
            store.vector._conn.execute(
                "INSERT INTO chunk_entities(chunk_id, entity_id, frequency, positions_json, "
                "score, namespace) VALUES (?, ?, 1, '[]', 1.0, 'retail')",
                [ids[0], eid],
            )
        return store

    def test_matching_namespace_returns_the_entities(self, scoped):
        assert scoped.resolve_entity_ids("Mercury", namespaces=["retail"]) == [
            "customer:mercury",
            "element:mercury",
        ]

    def test_other_namespace_returns_nothing(self, scoped):
        assert scoped.resolve_entity_ids("Mercury", namespaces=["support"]) == []

    def test_none_applies_no_restriction(self, scoped):
        assert len(scoped.resolve_entity_ids("Mercury")) == 2

    def test_empty_namespace_list_matches_nothing(self, scoped):
        assert scoped.resolve_entity_ids("Mercury", namespaces=[]) == []

    def test_type_and_namespace_compose(self, scoped):
        assert scoped.resolve_entity_ids(
            "Mercury", entity_type="customer", namespaces=["retail"]
        ) == ["customer:mercury"]

    def test_explain_distinguishes_wrong_namespace_from_a_miss(self, scoped):
        wrong_ns = scoped.explain_entity_lookup("Mercury", namespaces=["support"])
        assert wrong_ns.ids == []
        assert wrong_ns.name_exists is True
        assert wrong_ns.available_namespaces == ["retail"]

        miss = scoped.explain_entity_lookup("Nobody", namespaces=["support"])
        assert miss.ids == []
        assert miss.name_exists is False
        assert miss.available_namespaces == []

    def test_explain_alternatives_ignore_the_filters(self, scoped):
        result = scoped.explain_entity_lookup(
            "Mercury", entity_type="customer", namespaces=["support"]
        )
        assert result.available_types == ["customer", "element"]
        assert result.available_namespaces == ["retail"]

    def test_explain_ids_agree_with_resolve_under_the_same_filters(self, scoped):
        for etype, ns in [(None, None), ("customer", ["retail"]), (None, ["support"]), (None, [])]:
            assert scoped.explain_entity_lookup(
                "Mercury", entity_type=etype, namespaces=ns
            ).ids == scoped.resolve_entity_ids("Mercury", entity_type=etype, namespaces=ns), (
                etype,
                ns,
            )


class TestNearMatches:
    """Both directions, slug only, closest first."""

    @pytest.fixture()
    def populated(self, store):
        for eid, etype in [
            ("customer:mercury", "customer"),
            ("element:mercury", "element"),
            ("customer:mercury_systems", "customer"),
            ("customer:acme_corp", "customer"),
        ]:
            store.vector._conn.execute(
                "INSERT INTO entities(id, name, display_name, entity_type) VALUES (?, ?, ?, ?)",
                [eid, eid.split(":")[1].replace("_", " "), eid, etype],
            )
        return store

    def test_query_shorter_than_the_stored_name(self, populated):
        assert populated.explain_entity_lookup("acme").near_matches == ["customer:acme_corp"]

    def test_query_longer_than_the_stored_name(self, populated):
        """Over-specification is the common typo case and used to return nothing."""
        result = populated.explain_entity_lookup("mercury systems corp")
        assert result.ids == []
        assert result.near_matches == [
            "customer:mercury_systems",
            "customer:mercury",
            "element:mercury",
        ]

    def test_type_prefix_is_not_matched(self, populated):
        """Querying a type name must not return every entity of that type."""
        assert populated.explain_entity_lookup("customer").near_matches == []

    def test_ordered_by_closeness_in_length(self, populated):
        result = populated.explain_entity_lookup("mercury systems corp")
        lengths = [len(i.split(":")[1]) for i in result.near_matches]
        assert lengths == sorted(lengths, key=lambda n: abs(n - len("mercury_systems_corp")))

    def test_very_short_stored_slugs_do_not_match_everything(self, populated):
        populated.vector._conn.execute(
            "INSERT INTO entities(id, name, display_name, entity_type) "
            "VALUES ('customer:ab', 'ab', 'AB', 'customer')"
        )
        assert "customer:ab" not in populated.explain_entity_lookup("acme corp").near_matches


class TestPublicExports:
    """Return types of public Store methods must be importable without a private path."""

    def test_importable_from_the_package_root(self):
        import chonk

        assert chonk.EntityLookup is not None
        assert chonk.NamespaceEvidence is not None
        assert "EntityLookup" in chonk.__all__
        assert "NamespaceEvidence" in chonk.__all__

    def test_importable_from_chonk_storage(self):
        import chonk.storage

        assert "EntityLookup" in chonk.storage.__all__
        assert "NamespaceEvidence" in chonk.storage.__all__

    def test_they_are_the_same_objects(self):
        import chonk
        import chonk.storage
        from chonk.storage._store import EntityLookup, NamespaceEvidence

        assert chonk.EntityLookup is EntityLookup is chonk.storage.EntityLookup
        assert chonk.NamespaceEvidence is NamespaceEvidence is chonk.storage.NamespaceEvidence

    def test_returned_values_are_the_exported_types(self, store):
        import chonk

        store.vector._conn.execute(
            "INSERT INTO entities(id, name, display_name, entity_type) "
            "VALUES ('customer:acme', 'acme', 'Acme', 'customer')"
        )
        assert isinstance(store.explain_entity_lookup("acme"), chonk.EntityLookup)


class TestLikeWildcardEscaping:
    """Name slugs contain '_', which is a LIKE single-character wildcard."""

    def test_underscore_does_not_match_an_arbitrary_character(self, store):
        for eid in ("customer:acme_corp", "customer:acmexcorp"):
            store.vector._conn.execute(
                "INSERT INTO entities(id, name, display_name, entity_type) "
                "VALUES (?, ?, ?, 'customer')",
                [eid, eid.split(":")[1], eid],
            )
        assert store.resolve_entity_ids("Acme Corp") == ["customer:acme_corp"]

    def test_escape_helper_covers_all_three_metacharacters(self):
        from chonk.storage._store import _like_escape

        assert _like_escape("acme_corp") == r"acme\_corp"
        assert _like_escape("100%") == r"100\%"
        assert _like_escape("a\\b") == r"a\\b"
