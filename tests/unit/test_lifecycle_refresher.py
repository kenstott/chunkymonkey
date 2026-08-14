# Copyright (c) 2025 Kenneth Stott. MIT License.
"""Unit tests for NamespaceRefresher namespace enumeration and error reporting."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from chonk.lifecycle import NamespaceRefresher
from chonk.storage._store import Store


class TestConstruction:
    def test_requires_db_path_fn_or_dsn(self):
        with pytest.raises(ValueError, match="db_path_fn or dsn"):
            NamespaceRefresher(None, "model")

    def test_dsn_rejects_community_rebuild(self):
        with pytest.raises(NotImplementedError, match="run_community=False"):
            NamespaceRefresher(None, "model", dsn="postgresql://localhost/db")

    def test_dsn_with_community_disabled_constructs(self):
        refresher = NamespaceRefresher(
            None, "model", dsn="postgresql://localhost/db", run_community=False
        )
        assert refresher._dsn == "postgresql://localhost/db"


class TestCheckAll:
    def test_enumerates_through_store_not_duckdb(self):
        """Namespace enumeration goes through Store.list_namespaces(), not duckdb."""
        store = MagicMock()
        store.list_namespaces.return_value = ["global", "ns1"]
        store.namespace_cache_valid.return_value = True

        refresher = NamespaceRefresher(lambda ns: f"/tmp/{ns}.duckdb", "model")
        with patch("chonk.lifecycle.Store", return_value=store) as mock_store:
            refresher._check_all()

        store.list_namespaces.assert_called_once_with()
        assert store.namespace_cache_valid.call_args_list[0][0][0] == "global"
        assert mock_store.call_count == 3  # enumeration + one per namespace

    def test_pg_backend_opens_store_with_dsn(self):
        store = MagicMock()
        store.list_namespaces.return_value = ["ns1"]
        store.namespace_cache_valid.return_value = True

        refresher = NamespaceRefresher(
            None, "model", dsn="postgresql://localhost/db", run_community=False
        )
        with patch("chonk.lifecycle.Store", return_value=store) as mock_store:
            refresher._check_all()

        assert store.list_namespaces.call_count == 1
        for call in mock_store.call_args_list:
            assert call.kwargs == {"dsn": "postgresql://localhost/db"}

    def test_stale_namespace_triggers_rebuild(self):
        store = MagicMock()
        store.list_namespaces.return_value = ["ns1"]
        store.namespace_cache_valid.return_value = False
        rebuilt: list[str] = []

        refresher = NamespaceRefresher(
            lambda ns: f"/tmp/{ns}.duckdb", "model", on_rebuild=rebuilt.append
        )
        with (
            patch("chonk.lifecycle.Store", return_value=store),
            patch("chonk.lifecycle.build_namespace_async") as mock_build,
        ):
            refresher._check_all()

        assert rebuilt == ["ns1"]
        assert mock_build.call_args[0][:1] == ("ns1",)
        assert mock_build.call_args.kwargs["dsn"] is None

    def test_enumeration_failure_is_logged_not_swallowed(self, caplog):
        refresher = NamespaceRefresher(lambda ns: f"/tmp/{ns}.duckdb", "model")
        with (
            patch("chonk.lifecycle.Store", side_effect=RuntimeError("boom")),
            caplog.at_level(logging.ERROR, logger="chonk.lifecycle"),
        ):
            refresher._check_all()

        assert "namespace enumeration failed" in caplog.text
        assert "boom" in caplog.text

    def test_freshness_failure_is_logged_and_loop_continues(self, caplog):
        enum_store = MagicMock()
        enum_store.list_namespaces.return_value = ["ns1", "ns2"]
        good_store = MagicMock()
        good_store.namespace_cache_valid.return_value = True
        stores = [enum_store, RuntimeError("bad ns"), good_store]

        def _factory(*_args, **_kwargs):
            item = stores.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        refresher = NamespaceRefresher(lambda ns: f"/tmp/{ns}.duckdb", "model")
        with (
            patch("chonk.lifecycle.Store", side_effect=_factory),
            caplog.at_level(logging.ERROR, logger="chonk.lifecycle"),
        ):
            refresher._check_all()

        assert "freshness check failed for namespace 'ns1'" in caplog.text
        good_store.namespace_cache_valid.assert_called_once_with("ns2")


class TestStoreListNamespaces:
    def test_list_namespaces_duckdb(self, tmp_path):
        store = Store(tmp_path / "idx.duckdb")
        try:
            store.register_namespace("beta")
            store.register_namespace("alpha")
            assert store.list_namespaces() == ["alpha", "beta"]
        finally:
            store.close()
