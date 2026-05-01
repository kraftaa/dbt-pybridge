import pytest

import dbt_pybridge.dataframe_io as dataframe_io
from dbt_pybridge.dataframe_io import _align_and_validate_columns, _temp_relation, _validate_unique_key, write_model_result
from dbt_pybridge.runner import LocalPythonModelRunner
from dbt_pybridge.session import TargetRelation

pd = pytest.importorskip("pandas")


def test_unique_key_normalization():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    assert runner._normalize_unique_key(None) is None
    assert runner._normalize_unique_key("id") == ["id"]
    assert runner._normalize_unique_key(["id", "site"]) == ["id", "site"]


def test_unique_key_normalization_invalid():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    with pytest.raises(RuntimeError):
        runner._normalize_unique_key({"id": 1})


def test_column_types_config_normalization():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    out = runner._column_types({"localpy_column_types": {"id": "bigint", 2: "text"}})
    assert out == {"id": "bigint", "2": "text"}


def test_column_types_config_normalization_pybridge_key():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    out = runner._column_types({"pybridge_column_types": {"id": "bigint", 2: "text"}})
    assert out == {"id": "bigint", "2": "text"}


def test_column_types_prefers_pybridge_key_when_both_set():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    out = runner._column_types(
        {
            "pybridge_column_types": {"id": "integer"},
            "localpy_column_types": {"id": "text"},
        }
    )
    assert out == {"id": "integer"}


def test_column_types_config_invalid():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    with pytest.raises(RuntimeError):
        runner._column_types({"localpy_column_types": ["id"]})


def test_categorical_types_config_normalization():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    out = runner._categorical_types({"localpy_categorical_types": {"status": "status_enum", 2: "x_enum"}})
    assert out == {"status": "status_enum", "2": "x_enum"}


def test_categorical_types_config_normalization_pybridge_key():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    out = runner._categorical_types({"pybridge_categorical_types": {"status": "status_enum", 2: "x_enum"}})
    assert out == {"status": "status_enum", "2": "x_enum"}


def test_categorical_types_config_invalid():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    with pytest.raises(RuntimeError):
        runner._categorical_types({"localpy_categorical_types": ["status"]})


def test_limits_parses_boolean_strings():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    limits = runner._limits(
        {
            "localpy_allow_large_tables": "false",
            "localpy_chunked_mode": "true",
        }
    )
    assert limits.allow_large_tables is False
    assert limits.chunked_mode is True


def test_limits_parses_boolean_strings_pybridge_keys():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    limits = runner._limits(
        {
            "pybridge_allow_large_tables": "false",
            "pybridge_chunked_mode": "true",
        }
    )
    assert limits.allow_large_tables is False
    assert limits.chunked_mode is True


def test_limits_rejects_invalid_boolean_strings():
    runner = LocalPythonModelRunner.__new__(LocalPythonModelRunner)
    with pytest.raises(RuntimeError):
        runner._limits({"pybridge_allow_large_tables": "maybe"})


def test_align_validate_columns_reorders():
    df = pd.DataFrame({"b": [1], "a": [2]})
    out = _align_and_validate_columns(df, ["a", "b"])
    assert list(out.columns) == ["a", "b"]


def test_align_validate_columns_reorders_numeric_labels():
    df = pd.DataFrame({1: [1], 0: [2]})
    out = _align_and_validate_columns(df, ["0", "1"])
    assert list(out.columns) == [0, 1]


def test_align_validate_columns_rejects_ambiguous_string_cast():
    df = pd.DataFrame([[1, 2]], columns=[1, "1"])
    with pytest.raises(RuntimeError):
        _align_and_validate_columns(df, ["1"])


def test_validate_unique_key_missing_column():
    with pytest.raises(RuntimeError):
        _validate_unique_key(["missing"], ["id", "name"])


def test_temp_relation_has_nonce():
    target = TargetRelation(database=None, schema="public", identifier="orders")
    a = _temp_relation(target).identifier
    b = _temp_relation(target).identifier
    assert a != b
    assert a.startswith("__dbt_pybridge_tmp_orders_")
    assert b.startswith("__dbt_pybridge_tmp_orders_")


def test_temp_relation_uses_no_schema_for_pg_temp():
    # Temp relations are created with CREATE TEMP TABLE, which puts them in
    # pg_temp regardless of any schema we'd specify. Returning schema=None
    # keeps the rendered SQL unqualified so it resolves via the search_path.
    target = TargetRelation(database=None, schema="transform", identifier="orders")
    temp = _temp_relation(target)
    assert temp.schema is None


def test_create_table_for_dataframe_emits_create_temp_table():
    class FakeCursor:
        def __init__(self):
            self.sql = []

        def execute(self, query, params=None):
            self.sql.append(query)

    cur = FakeCursor()
    target = TargetRelation(database=None, schema=None, identifier="t_xyz")
    df = pd.DataFrame({"id": [1]})
    dataframe_io._create_table_for_dataframe(cur, target, df, replace=False, temporary=True)
    assert any("create temp table" in q.lower() for q in cur.sql)
    # No "create schema" and no "drop table" should be emitted for a temp.
    assert not any("create schema" in q.lower() for q in cur.sql)
    assert not any("drop table" in q.lower() for q in cur.sql)


def test_create_table_for_dataframe_replace_uses_cascade():
    class FakeCursor:
        def __init__(self):
            self.sql = []

        def execute(self, query, params=None):
            self.sql.append(query)

    cur = FakeCursor()
    target = TargetRelation(database=None, schema=None, identifier="t")
    df = pd.DataFrame({"id": [1]})
    dataframe_io._create_table_for_dataframe(cur, target, df, replace=True)
    drop_stmts = [q for q in cur.sql if "drop table" in q.lower()]
    assert drop_stmts, "expected a drop table statement"
    assert all("cascade" in q.lower() for q in drop_stmts)


def test_write_model_result_rejects_unsupported_materialization():
    df = pd.DataFrame({"id": [1]})
    with pytest.raises(RuntimeError):
        write_model_result(
            conn=None,
            target=TargetRelation(database=None, schema="public", identifier="x"),
            result=df,
            materialized="view",
        )


def test_incremental_empty_first_run_creates_table(monkeypatch):
    df = pd.DataFrame({"id": pd.Series(dtype="int64"), "name": pd.Series(dtype="object")})
    target = TargetRelation(database=None, schema="public", identifier="x")

    calls = {"created": 0}

    def fake_table_columns_with_types(cur, relation):
        return []

    def fake_create(cur, relation, in_df, replace, column_types=None, categorical_types=None):
        calls["created"] += 1
        assert relation.identifier == "x"
        assert replace is True
        assert list(in_df.columns) == ["id", "name"]

    monkeypatch.setattr(dataframe_io, "_table_columns_with_types", fake_table_columns_with_types)
    monkeypatch.setattr(dataframe_io, "_create_table_for_dataframe", fake_create)

    rows = dataframe_io._apply_incremental_chunk(
        cur=object(),
        target=target,
        chunk_df=df,
        incremental_strategy="append",
        unique_key=None,
    )

    assert rows == 0
    assert calls["created"] == 1


def test_chunked_table_rejects_mismatched_columns(monkeypatch):
    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConn:
        def __init__(self):
            self.committed = False

        def cursor(self):
            return FakeCursor()

        def commit(self):
            self.committed = True

    monkeypatch.setattr(dataframe_io, "_create_table_for_dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(dataframe_io, "_copy_dataframe", lambda *args, **kwargs: 1)

    def _chunks():
        yield pd.DataFrame({"id": [1], "name": ["a"]})
        yield pd.DataFrame({"id": [2], "different_col": ["b"]})

    conn = FakeConn()
    with pytest.raises(RuntimeError):
        write_model_result(
            conn=conn,
            target=TargetRelation(database=None, schema="public", identifier="x"),
            result=_chunks(),
            materialized="table",
        )
    assert conn.committed is False


def test_create_table_uses_column_type_overrides():
    class FakeCursor:
        def __init__(self):
            self.sql = []

        def execute(self, query, params=None):
            self.sql.append(query)

    cur = FakeCursor()
    target = TargetRelation(database=None, schema=None, identifier="x")
    df = pd.DataFrame({"id": [1], "name": ["a"]})
    dataframe_io._create_table_for_dataframe(
        cur,
        target,
        df,
        replace=True,
        column_types={"id": "numeric(18,0)", "name": "text"},
    )
    assert any("numeric(18,0)" in query for query in cur.sql)


def test_create_table_rejects_unknown_column_type_override():
    class FakeCursor:
        def execute(self, query, params=None):
            return None

    target = TargetRelation(database=None, schema=None, identifier="x")
    df = pd.DataFrame({"id": [1]})
    with pytest.raises(RuntimeError):
        dataframe_io._create_table_for_dataframe(
            FakeCursor(),
            target,
            df,
            replace=True,
            column_types={"missing": "text"},
        )


def test_create_table_uses_categorical_type_overrides():
    class FakeCursor:
        def __init__(self):
            self.sql = []

        def execute(self, query, params=None):
            self.sql.append(query)

    cur = FakeCursor()
    target = TargetRelation(database=None, schema=None, identifier="x")
    df = pd.DataFrame({"status": pd.Series(["a", "b"], dtype="category"), "id": [1, 2]})
    dataframe_io._create_table_for_dataframe(
        cur,
        target,
        df,
        replace=True,
        categorical_types={"status": "status_enum"},
    )
    assert any("status_enum" in query for query in cur.sql)


def test_create_table_rejects_categorical_override_for_non_categorical_column():
    class FakeCursor:
        def execute(self, query, params=None):
            return None

    target = TargetRelation(database=None, schema=None, identifier="x")
    df = pd.DataFrame({"status": ["a", "b"]})
    with pytest.raises(RuntimeError):
        dataframe_io._create_table_for_dataframe(
            FakeCursor(),
            target,
            df,
            replace=True,
            categorical_types={"status": "status_enum"},
        )


def test_create_table_rejects_ambiguous_stringified_column_names():
    class FakeCursor:
        def execute(self, query, params=None):
            return None

    target = TargetRelation(database=None, schema=None, identifier="x")
    df = pd.DataFrame([[1, 2]], columns=[1, "1"])
    with pytest.raises(RuntimeError):
        dataframe_io._create_table_for_dataframe(
            FakeCursor(),
            target,
            df,
            replace=True,
        )
