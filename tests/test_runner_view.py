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
    assert len(backing.identifier) <= 63


def test_runner_view_materialization_uses_backing_table(monkeypatch):
    class FakeConn:
        pass

    class FakeSession:
        def __init__(self, credentials, limits, dataframe_backend):
            self.conn = FakeConn()

        def close(self):
            return None

    calls = {}

    def fake_write_model_result(conn, target, result, batch_size, materialized, incremental_strategy, unique_key):
        calls["write"] = {
            "conn": conn,
            "target": target,
            "materialized": materialized,
            "incremental_strategy": incremental_strategy,
            "unique_key": unique_key,
            "rows": len(result),
        }
        return len(result)

    def fake_create_or_replace_view(self, conn, view_target, backing_target):
        calls["view"] = {
            "conn": conn,
            "view_target": view_target,
            "backing_target": backing_target,
        }

    monkeypatch.setattr(runner_module, "LocalPostgresSession", FakeSession)
    monkeypatch.setattr(runner_module, "write_model_result", fake_write_model_result)
    monkeypatch.setattr(LocalPythonModelRunner, "_create_or_replace_view", fake_create_or_replace_view)

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

    runner = LocalPythonModelRunner(credentials=object(), parsed_model=parsed_model, compiled_code=compiled_code)
    rows = runner.run()

    assert rows == 2
    assert calls["write"]["materialized"] == "table"
    assert calls["write"]["target"].identifier.startswith("__dbt_pybridge_view_")
    assert calls["view"]["view_target"].identifier == "bronze_beacons_python"
    assert calls["view"]["backing_target"].identifier.startswith("__dbt_pybridge_view_")

