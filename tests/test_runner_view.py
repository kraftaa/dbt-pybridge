import pytest

import dbt_pybridge.runner as runner_module
from dbt_pybridge.runner import LocalPythonModelRunner
from dbt_pybridge.session import TargetRelation

pd = pytest.importorskip("pandas")


def test_view_backing_relation_name():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    view_target = TargetRelation(database=None, schema="transform", identifier="bronze_beacons_python")
    backing = runner._view_backing_relation(view_target)
    assert backing.schema == "transform"
    assert backing.identifier.startswith("__dbt_pybridge_view_")
    assert len(backing.identifier.rsplit("_", 1)[-1]) == 8
    assert len(backing.identifier) <= 63


def test_runner_view_materialization_routes_through_swap(monkeypatch):
    """The view branch in run() must hand off to _materialize_view_via_swap
    with the right view target and the deterministic backing target."""

    class FakeSession:
        def __init__(self, credentials, limits, dataframe_backend, logger=None):
            self.conn = object()

        def close(self):
            return None

    captured = {}

    def fake_swap(self, conn, view_target, backing_target, write_backing):
        captured["view_target"] = view_target
        captured["backing_target"] = backing_target
        # Exercise the write_backing callback to confirm it's wired up correctly.
        captured["rows"] = write_backing(backing_target)
        return captured["rows"]

    def fake_write_model_result(
        conn, target, result, batch_size, materialized,
        incremental_strategy, unique_key, column_types,
        categorical_types, logger,
    ):
        captured["write_target"] = target
        captured["write_materialized"] = materialized
        return len(result)

    monkeypatch.setattr(runner_module, "LocalPostgresSession", FakeSession)
    monkeypatch.setattr(runner_module, "write_model_result", fake_write_model_result)
    monkeypatch.setattr(LocalPythonModelRunner, "_materialize_view_via_swap", fake_swap)

    compiled_code = """
import pandas as pd

class dbtObj:
    def __init__(self, load_df_function):
        self.ref = lambda *args, **kwargs: load_df_function('"public"."x"')
        self.source = lambda *args, **kwargs: load_df_function('"public"."x"')
        self.config = type("config", (), {"get": staticmethod(lambda k, d=None: d)})
        self.this = None
        self.is_incremental = False

def model(dbt, session):
    return pd.DataFrame({"id": [1, 2]})
"""
    parsed_model = {
        "database": "postgres",
        "schema": "transform",
        "alias": "bronze_beacons_python",
        "name": "bronze_beacons_python",
        "config": {"materialized": "view"},
    }
    runner = LocalPythonModelRunner(
        credentials=object(), parsed_model=parsed_model, compiled_code=compiled_code,
    )
    rows = runner.run()

    assert rows == 2
    assert captured["view_target"].identifier == "bronze_beacons_python"
    assert captured["backing_target"].identifier.startswith("__dbt_pybridge_view_")
    # write_backing(intermediate) inside the swap should call write_model_result
    # with materialized='table' and the intermediate identifier we passed.
    assert captured["write_materialized"] == "table"
    assert captured["write_target"] is captured["backing_target"]


def test_materialize_view_via_swap_rename_swap_order(monkeypatch):
    """The swap helper must (1) build new backing, (2) create intermediate
    view, (3) RENAME existing view → backup, (4) RENAME intermediate → target,
    (5) DROP backup, (6) DROP old backing, (7) RENAME intermediate backing →
    final. This ordering is what makes the rename-swap atomic-ish (matches
    dbt-core's Postgres convention)."""

    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            self.conn.sql.append(sql.strip())

    class FakeConn:
        def __init__(self):
            self.sql = []
            self.commits = 0

        def cursor(self):
            return FakeCursor(self)

        def commit(self):
            self.commits += 1

    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    conn = FakeConn()
    view_target = TargetRelation(database=None, schema="transform", identifier="my_view")
    backing_target = TargetRelation(
        database=None, schema="transform", identifier="__dbt_pybridge_view_my_view_aaaa1111",
    )

    def fake_relation_kind(self, conn, relation):
        # Only the live target slot starts as a view.
        if relation.identifier == "my_view":
            return "v"
        # Backup view and stable backing exist when cleanup runs after swap.
        return "v" if relation.identifier.endswith("__pybbkup") else (
            "r" if relation.identifier == backing_target.identifier else None
        )

    def fake_drop(self, conn, relation):
        # Track drops so the test can assert they happened.
        conn.sql.append(f"DROP {relation.identifier}")
        conn.commits += 1

    monkeypatch.setattr(LocalPythonModelRunner, "_relation_kind", fake_relation_kind)
    monkeypatch.setattr(LocalPythonModelRunner, "_drop_existing_relation", fake_drop)

    write_calls = []

    def write_backing(intermediate_backing):
        write_calls.append(intermediate_backing)
        return 7

    rows = runner._materialize_view_via_swap(
        conn,
        view_target=view_target,
        backing_target=backing_target,
        write_backing=write_backing,
    )

    assert rows == 7
    assert len(write_calls) == 1
    assert write_calls[0].identifier.endswith("__tmp")  # intermediate backing

    text = "\n".join(conn.sql)
    # New view at intermediate name, pointing at intermediate backing.
    assert "create view" in text.lower()
    assert "__pybtmp" in text
    # Rename existing → backup AND intermediate → target both present.
    assert "rename to \"my_view__pybbkup\"" in text
    assert "rename to \"my_view\"" in text
    # Final rename: intermediate backing → stable backing name.
    assert f'rename to "{backing_target.identifier}"' in text
    # Backup view dropped after swap.
    assert any("__pybbkup" in s for s in conn.sql if s.startswith("DROP "))


def test_create_or_replace_view_emits_create_view_only():
    class FakeCursor:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=None):
            self.conn.sql.append(query)

    class FakeConn:
        def __init__(self):
            self.sql = []
            self.commit_count = 0

        def cursor(self):
            return FakeCursor(self)

        def commit(self):
            self.commit_count += 1

    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    conn = FakeConn()
    view_target = TargetRelation(database=None, schema="transform", identifier="my_view")
    backing_target = TargetRelation(database=None, schema="transform", identifier="my_backing")

    runner._create_or_replace_view(conn, view_target, backing_target)

    assert any("create schema if not exists" in q.lower() for q in conn.sql)
    assert any("create view" in q.lower() for q in conn.sql)
    assert not any("drop " in q.lower() for q in conn.sql)
    assert conn.commit_count == 1
