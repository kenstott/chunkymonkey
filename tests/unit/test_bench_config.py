# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 86556eeb-674d-464d-a494-abc97b7a2cad
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit and integration tests for graphrag_bench TOML config capabilities.

Covers:
  - _deep_merge
  - _load_config  (extends chain, depth limit)
  - _apply_config (all sections, CLI-wins guard)
  - [[vocab.entities]] static and db_query entries feeding
    _build_entity_index_from_store via SchemaVocabBuilder
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import duckdb
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import demo.graphrag_bench as _bench  # noqa: E402  # needs the sys.path shim above

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _write_toml(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_flat_override(self):
        result = _bench._deep_merge({"a": 1, "b": 2}, {"b": 99})
        assert result == {"a": 1, "b": 99}

    def test_nested_dict_merged_not_replaced(self):
        base = {"index": {"db_name": "old.duckdb", "embed_model": "bge"}}
        override = {"index": {"db_name": "new.duckdb"}}
        result = _bench._deep_merge(base, override)
        assert result["index"]["db_name"] == "new.duckdb"
        assert result["index"]["embed_model"] == "bge"

    def test_override_value_none_wins(self):
        result = _bench._deep_merge({"a": 1}, {"a": None})
        assert result["a"] is None

    def test_list_replaced_not_merged(self):
        base = {"sources": [1, 2, 3]}
        override = {"sources": [4, 5]}
        result = _bench._deep_merge(base, override)
        assert result["sources"] == [4, 5]

    def test_deep_nested_three_levels(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = _bench._deep_merge(base, override)
        assert result["a"]["b"]["c"] == 99
        assert result["a"]["b"]["d"] == 2

    def test_new_key_in_override_added(self):
        result = _bench._deep_merge({"a": 1}, {"z": 99})
        assert result["z"] == 99
        assert result["a"] == 1

    def test_base_not_mutated(self):
        base = {"a": {"x": 1}}
        _bench._deep_merge(base, {"a": {"y": 2}})
        assert "y" not in base["a"]


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_returns_empty_for_none(self):
        assert _bench._load_config(None) == {}

    def test_simple_toml_loaded(self, tmp_path):
        p = _write_toml(
            tmp_path,
            "cfg.toml",
            """\
            [index]
            db_name = "test.duckdb"
        """,
        )
        cfg = _bench._load_config(str(p))
        assert cfg["index"]["db_name"] == "test.duckdb"

    def test_extends_inherits_parent_keys(self, tmp_path):
        _write_toml(
            tmp_path,
            "base.toml",
            """\
            [index]
            embed_model = "bge-large"
            db_name = "base.duckdb"
        """,
        )
        child = _write_toml(
            tmp_path,
            "child.toml",
            """\
            extends = "base.toml"
            [index]
            db_name = "child.duckdb"
        """,
        )
        cfg = _bench._load_config(str(child))
        assert cfg["index"]["db_name"] == "child.duckdb"
        assert cfg["index"]["embed_model"] == "bge-large"

    def test_extends_chain_two_levels(self, tmp_path):
        _write_toml(
            tmp_path,
            "grand.toml",
            """\
            [gen]
            model = "gpt-4o-mini"
        """,
        )
        _write_toml(
            tmp_path,
            "parent.toml",
            """\
            extends = "grand.toml"
            [index]
            db_name = "parent.duckdb"
        """,
        )
        child = _write_toml(
            tmp_path,
            "child.toml",
            """\
            extends = "parent.toml"
            [retrieval]
            top_k = 10
        """,
        )
        cfg = _bench._load_config(str(child))
        assert cfg["gen"]["model"] == "gpt-4o-mini"
        assert cfg["index"]["db_name"] == "parent.duckdb"
        assert cfg["retrieval"]["top_k"] == 10

    def test_depth_limit_raises(self, tmp_path):
        # Build a chain longer than 5
        prev = _write_toml(tmp_path, "c0.toml", "[index]\ndb_name='c0.duckdb'\n")
        for i in range(1, 8):
            prev = _write_toml(
                tmp_path,
                f"c{i}.toml",
                f'extends = "c{i - 1}.toml"\n[index]\ndb_name="c{i}.duckdb"\n',
            )
        with pytest.raises(RuntimeError, match="max depth"):
            _bench._load_config(str(prev))

    def test_extends_key_removed_from_result(self, tmp_path):
        _write_toml(tmp_path, "base.toml", "[index]\ndb_name='base.duckdb'\n")
        child = _write_toml(
            tmp_path, "child.toml", 'extends = "base.toml"\n[retrieval]\ntop_k = 5\n'
        )
        cfg = _bench._load_config(str(child))
        assert "extends" not in cfg

    def test_source_array_loaded(self, tmp_path):
        p = _write_toml(
            tmp_path,
            "cfg.toml",
            """\
            [[source]]
            type = "directory"
            uri  = "/tmp/docs"
            [[source]]
            type = "github"
            uri  = "https://github.com/org/repo"
        """,
        )
        cfg = _bench._load_config(str(p))
        assert len(cfg["source"]) == 2
        assert cfg["source"][0]["type"] == "directory"
        assert cfg["source"][1]["type"] == "github"

    def test_vocab_entities_array_loaded(self, tmp_path):
        p = _write_toml(
            tmp_path,
            "cfg.toml",
            """\
            [[vocab.entities]]
            type        = "static"
            entity_type = "customer"
            names       = ["Acme Corp", "Globex Inc"]
        """,
        )
        cfg = _bench._load_config(str(p))
        entries = cfg["vocab"]["entities"]
        assert len(entries) == 1
        assert entries[0]["type"] == "static"
        assert entries[0]["entity_type"] == "customer"
        assert "Acme Corp" in entries[0]["names"]


# ---------------------------------------------------------------------------
# _apply_config
# ---------------------------------------------------------------------------


class TestApplyConfig:
    # ── index section ────────────────────────────────────────────────────────

    def test_db_name_applied_when_unset(self):
        args = _ns(db_name=None)
        _bench._apply_config({"index": {"db_name": "from_config.duckdb"}}, args)
        assert args.db_name == "from_config.duckdb"

    def test_db_name_not_overridden_when_set(self):
        args = _ns(db_name="cli.duckdb")
        _bench._apply_config({"index": {"db_name": "from_config.duckdb"}}, args)
        assert args.db_name == "cli.duckdb"

    # ── rerank ───────────────────────────────────────────────────────────────

    def test_rerank_enabled_by_config(self):
        args = _ns(rerank=False)
        _bench._apply_config({"rerank": {"enabled": True}}, args)
        assert args.rerank is True

    def test_rerank_not_disabled_by_config(self):
        args = _ns(rerank=True)
        _bench._apply_config({"rerank": {"enabled": False}}, args)
        assert args.rerank is True

    # ── retrieval ────────────────────────────────────────────────────────────

    def test_top_k_applied(self):
        args = _ns(top_k=None)
        _bench._apply_config({"retrieval": {"top_k": 15}}, args)
        assert args.top_k == 15

    def test_top_k_not_overridden(self):
        args = _ns(top_k=10)
        _bench._apply_config({"retrieval": {"top_k": 15}}, args)
        assert args.top_k == 10

    def test_lane_entity_min_sim_applied(self):
        args = _ns(lane_entity_min_sim=None)
        _bench._apply_config({"retrieval": {"lane_entity_min_sim": 0.45}}, args)
        assert args.lane_entity_min_sim == pytest.approx(0.45)

    def test_redundancy_threshold_applied(self):
        args = _ns(redundancy_threshold=None)
        _bench._apply_config({"retrieval": {"redundancy_threshold": 0.92}}, args)
        assert args.redundancy_threshold == pytest.approx(0.92)

    def test_search_mode_applied(self):
        args = _ns(search_mode="vector_first")
        _bench._apply_config({"retrieval": {"search_mode": "graph_first"}}, args)
        assert args.search_mode == "graph_first"

    def test_search_mode_not_overridden_when_non_default(self):
        args = _ns(search_mode="global")
        _bench._apply_config({"retrieval": {"search_mode": "graph_first"}}, args)
        assert args.search_mode == "global"

    def test_enhanced_applied(self):
        args = _ns(enhanced=False)
        _bench._apply_config({"retrieval": {"enhanced": True}}, args)
        assert args.enhanced is True

    def test_cluster_applied(self):
        args = _ns(cluster=False)
        _bench._apply_config({"retrieval": {"cluster": True}}, args)
        assert args.cluster is True

    def test_community_context_applied(self):
        args = _ns(community_context=False)
        _bench._apply_config({"retrieval": {"community": {"enabled": True}}}, args)
        assert args.community_context is True

    def test_community_min_coherence_applied(self):
        args = _ns(community_min_coherence=None)
        _bench._apply_config({"retrieval": {"community": {"min_coherence": 0.6}}}, args)
        assert args.community_min_coherence == pytest.approx(0.6)

    # ── gen ──────────────────────────────────────────────────────────────────

    def test_gen_model_applied(self):
        args = _ns(gen_model="gpt-4o-mini-2024-07-18", gen_provider="openai")
        _bench._apply_config({"gen": {"model": "gpt-4o"}}, args)
        assert args.gen_model == "gpt-4o"

    def test_gen_provider_applied(self):
        args = _ns(gen_provider="openai", gen_model="gpt-4o-mini-2024-07-18")
        _bench._apply_config({"gen": {"provider": "together"}}, args)
        assert args.gen_provider == "together"

    # ── sr / srr ─────────────────────────────────────────────────────────────

    def test_sr_enabled_by_config(self):
        args = _ns(sr=False)
        _bench._apply_config({"sr": {"enabled": True}}, args)
        assert args.sr is True

    def test_srr_enabled_by_config(self):
        args = _ns(srr=False)
        _bench._apply_config({"srr": {"enabled": True}}, args)
        assert args.srr is True

    def test_srr_model_applied(self):
        args = _ns(srr_model=None, srr_provider=None, srr=False)
        _bench._apply_config({"srr": {"model": "gpt-4o-mini", "provider": "openai"}}, args)
        assert args.srr_model == "gpt-4o-mini"
        assert args.srr_provider == "openai"

    # ── index.features ───────────────────────────────────────────────────────

    def test_svo_feature_sets_with_svo(self):
        args = _ns(with_svo=False)
        _bench._apply_config({"index": {"features": {"svo": True}}}, args)
        assert args.with_svo is True

    def test_svo_feature_not_overridden_when_already_set(self):
        args = _ns(with_svo=True)
        _bench._apply_config({"index": {"features": {"svo": False}}}, args)
        assert args.with_svo is True

    def test_community_feature_sets_with_community(self):
        args = _ns(with_community=False)
        _bench._apply_config({"index": {"features": {"community": True}}}, args)
        assert args.with_community is True

    def test_ner_feature_sets_with_ner(self):
        args = _ns(with_ner=False)
        _bench._apply_config({"index": {"features": {"ner": True}}}, args)
        assert args.with_ner is True

    def test_features_absent_leaves_flags_unchanged(self):
        args = _ns(with_svo=False, with_community=False, with_ner=False)
        _bench._apply_config({"index": {}}, args)
        assert args.with_svo is False
        assert args.with_community is False
        assert args.with_ner is False

    # ── vocab.entities ───────────────────────────────────────────────────────

    def test_vocab_entities_applied_from_config(self):
        args = _ns()
        entries = [{"type": "static", "entity_type": "customer", "names": ["Acme"]}]
        _bench._apply_config({"vocab": {"entities": entries}}, args)
        assert args.vocab_entities == [dict(entries[0], namespace="global")]

    def test_vocab_entity_namespace_defaults_to_global(self):
        args = _ns()
        entries = [
            {"type": "static", "entity_type": "customer", "names": ["Acme"]},
            {
                "type": "db_query",
                "entity_type": "employee",
                "connection": "duckdb:///x.duckdb",
                "sql": "SELECT n FROM e",
            },
        ]
        _bench._apply_config({"vocab": {"entities": entries}}, args)
        assert [e["namespace"] for e in args.vocab_entities] == ["global", "global"]

    def test_vocab_entity_namespace_preserved_when_set(self):
        args = _ns()
        entries = [
            {
                "type": "static",
                "entity_type": "customer",
                "names": ["Acme"],
                "namespace": "finance",
            },
            {
                "type": "db_query",
                "entity_type": "employee",
                "connection": "duckdb:///x.duckdb",
                "sql": "SELECT n FROM e",
                "namespace": "hr",
            },
        ]
        _bench._apply_config({"vocab": {"entities": entries}}, args)
        assert [e["namespace"] for e in args.vocab_entities] == ["finance", "hr"]

    def test_vocab_entities_not_overridden_when_already_set(self):
        existing = [{"type": "static", "entity_type": "product", "names": ["Widget"]}]
        args = _ns(vocab_entities=existing)
        cfg_entries = [{"type": "static", "entity_type": "customer", "names": ["Acme"]}]
        _bench._apply_config({"vocab": {"entities": cfg_entries}}, args)
        assert args.vocab_entities == existing

    def test_empty_vocab_entities_not_applied(self):
        args = _ns()
        _bench._apply_config({"vocab": {"entities": []}}, args)
        assert not getattr(args, "vocab_entities", None)

    def test_multiple_vocab_entries_preserved(self):
        args = _ns()
        entries = [
            {"type": "static", "entity_type": "customer", "names": ["Acme", "Globex"]},
            {
                "type": "db_query",
                "entity_type": "employee",
                "connection": "sqlite:///x.db",
                "sql": "SELECT name FROM emp",
            },
        ]
        _bench._apply_config({"vocab": {"entities": entries}}, args)
        assert len(args.vocab_entities) == 2
        assert args.vocab_entities[0]["type"] == "static"
        assert args.vocab_entities[1]["type"] == "db_query"

    # ── full round-trip from TOML file ────────────────────────────────────────

    def test_full_round_trip_from_toml(self, tmp_path):
        p = _write_toml(
            tmp_path,
            "run.toml",
            """\
            [retrieval]
            top_k = 10
            enhanced = true
            lane_entity_min_sim = 0.55

            [gen]
            model = "gpt-4o"

            [[vocab.entities]]
            type        = "static"
            entity_type = "customer"
            names       = ["Acme Corp"]
        """,
        )
        cfg = _bench._load_config(str(p))
        args = _ns(
            top_k=None,
            enhanced=False,
            lane_entity_min_sim=None,
            gen_model="gpt-4o-mini-2024-07-18",
            gen_provider="openai",
        )
        _bench._apply_config(cfg, args)
        assert args.top_k == 10
        assert args.enhanced is True
        assert args.lane_entity_min_sim == pytest.approx(0.55)
        assert args.gen_model == "gpt-4o"
        assert args.vocab_entities[0]["entity_type"] == "customer"


# ---------------------------------------------------------------------------
# SchemaVocabBuilder.add_entities / build_data_matcher
# ---------------------------------------------------------------------------


class TestSchemaVocabBuilderEntities:
    def test_static_names_matched(self):
        from chonk.ner import SchemaVocabBuilder

        builder = SchemaVocabBuilder()
        builder.add_entities(["Acme Corp", "Globex Inc"], entity_type="customer")
        matcher = builder.build_data_matcher()
        results = matcher.match("We work with Acme Corp on this project.")
        assert len(results) == 1
        assert results[0].entity_type == "customer"
        assert results[0].display_name == "Acme Corp"

    def test_entity_type_assigned_correctly(self):
        from chonk.ner import SchemaVocabBuilder

        builder = SchemaVocabBuilder()
        builder.add_entities(["John Smith"], entity_type="employee")
        matcher = builder.build_data_matcher()
        results = matcher.match("Contact John Smith for details.")
        assert results[0].entity_type == "employee"

    def test_case_insensitive_match(self):
        from chonk.ner import SchemaVocabBuilder

        builder = SchemaVocabBuilder()
        builder.add_entities(["Acme Corp"], entity_type="customer")
        matcher = builder.build_data_matcher()
        assert len(matcher.match("ACME CORP signed the deal.")) == 1

    def test_multiple_entity_types(self):
        from chonk.ner import SchemaVocabBuilder

        builder = SchemaVocabBuilder()
        builder.add_entities(["Acme Corp"], entity_type="customer")
        builder.add_entities(["Jane Doe"], entity_type="employee")
        matcher = builder.build_data_matcher()
        results = matcher.match("Jane Doe from Acme Corp attended.")
        types = {r.entity_type for r in results}
        assert "customer" in types
        assert "employee" in types

    def test_data_term_count(self):
        from chonk.ner import SchemaVocabBuilder

        builder = SchemaVocabBuilder()
        builder.add_entities(["Alpha", "Beta", "Gamma"], entity_type="product")
        assert builder.data_term_count() == 3

    def test_empty_names_list(self):
        from chonk.ner import SchemaVocabBuilder

        builder = SchemaVocabBuilder()
        builder.add_entities([], entity_type="customer")
        assert builder.data_term_count() == 0

    def test_chaining(self):
        from chonk.ner import SchemaVocabBuilder

        builder = SchemaVocabBuilder()
        result = builder.add_entities(["Acme"], entity_type="customer")
        assert result is builder

    def test_no_match_outside_vocab(self):
        from chonk.ner import SchemaVocabBuilder

        builder = SchemaVocabBuilder()
        builder.add_entities(["Acme Corp"], entity_type="customer")
        matcher = builder.build_data_matcher()
        assert matcher.match("Some other company handled this.") == []


# ---------------------------------------------------------------------------
# SchemaVocabBuilder.add_from_db
# ---------------------------------------------------------------------------


class TestSchemaVocabBuilderFromDb:
    def _make_db(self, tmp_path: Path) -> str:
        db_path = tmp_path / "test.duckdb"
        con = duckdb.connect(str(db_path))
        con.execute("CREATE TABLE customers (name VARCHAR)")
        con.execute("INSERT INTO customers VALUES ('Acme Corp'), ('Globex Inc'), ('Initech')")
        con.execute("CREATE TABLE employees (full_name VARCHAR)")
        con.execute("INSERT INTO employees VALUES ('Alice Smith'), ('Bob Jones')")
        con.close()
        return str(db_path)

    def test_names_loaded_from_db(self, tmp_path):
        from chonk.ner import SchemaVocabBuilder

        db_url = f"duckdb:///{self._make_db(tmp_path)}"
        builder = SchemaVocabBuilder()
        builder.add_from_db(db_url, {"customer": "SELECT name FROM customers"})
        assert builder.data_term_count() == 3

    def test_matched_after_db_load(self, tmp_path):
        from chonk.ner import SchemaVocabBuilder

        db_url = f"duckdb:///{self._make_db(tmp_path)}"
        builder = SchemaVocabBuilder()
        builder.add_from_db(db_url, {"customer": "SELECT name FROM customers"})
        matcher = builder.build_data_matcher()
        results = matcher.match("Acme Corp is our top client.")
        assert len(results) == 1
        assert results[0].entity_type == "customer"

    def test_multiple_queries_different_types(self, tmp_path):
        from chonk.ner import SchemaVocabBuilder

        db_url = f"duckdb:///{self._make_db(tmp_path)}"
        builder = SchemaVocabBuilder()
        builder.add_from_db(
            db_url,
            {
                "customer": "SELECT name FROM customers",
                "employee": "SELECT full_name FROM employees",
            },
        )
        assert builder.data_term_count() == 5
        matcher = builder.build_data_matcher()
        results = matcher.match("Alice Smith works at Globex Inc.")
        types = {r.entity_type for r in results}
        assert "customer" in types
        assert "employee" in types


# ---------------------------------------------------------------------------
# _build_entity_index_from_store — vocab_entities integration
# ---------------------------------------------------------------------------


class TestBuildEntityIndexVocabEntities:
    def _make_store(self, tmp_path: Path):
        import numpy as np

        from chonk import chunk_document
        from chonk.storage._store import Store

        db_path = tmp_path / "test.duckdb"
        text = (
            "Acme Corp signed a major contract with Globex Inc last quarter. "
            "The deal was brokered by Alice Smith from the sales team."
        )
        chunks = chunk_document("test_doc.txt", text, min_chunk_size=20, max_chunk_size=500)
        embeddings = np.random.rand(len(chunks), 1024).astype("float32")
        with Store(db_path, embedding_dim=1024) as store:
            store.add_document(chunks, embeddings)
        return db_path

    def test_static_vocab_entities_indexed(self, tmp_path):
        db_path = self._make_store(tmp_path)
        from chonk.storage._store import Store

        vocab_entries = [
            {"type": "static", "entity_type": "customer", "names": ["Acme Corp", "Globex Inc"]},
            {"type": "static", "entity_type": "employee", "names": ["Alice Smith"]},
        ]
        with Store(db_path, embedding_dim=1024) as store:
            idx = _bench._build_entity_index_from_store(
                store, use_schema_vocab=False, vocab_entities=vocab_entries
            )
        # entity_ids() returns chunk IDs; _entity_to_chunks keys are entity IDs
        assert len(idx._entity_to_chunks) > 0

    def test_no_vocab_entities_uses_spacy_only(self, tmp_path):
        db_path = self._make_store(tmp_path)
        from chonk.storage._store import Store

        with Store(db_path, embedding_dim=1024) as store:
            idx = _bench._build_entity_index_from_store(
                store, use_schema_vocab=False, vocab_entities=None
            )
        assert idx.total_chunks() > 0

    def test_empty_vocab_entities_list(self, tmp_path):
        db_path = self._make_store(tmp_path)
        from chonk.storage._store import Store

        with Store(db_path, embedding_dim=1024) as store:
            idx = _bench._build_entity_index_from_store(
                store, use_schema_vocab=False, vocab_entities=[]
            )
        assert idx.total_chunks() > 0
