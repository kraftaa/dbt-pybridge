import csv
from io import StringIO
import json

import pytest

from dbt_pybridge.dataframe_io import _copy_dataframe
from dbt_pybridge.session import TargetRelation

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


class FakeCursor:
    def __init__(self):
        self.copy_sql = None
        self.payload_text = None

    def copy_expert(self, sql, payload):
        self.copy_sql = sql
        self.payload_text = payload.getvalue()


def test_copy_dataframe_serializes_array_jsonb_and_bytea():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame(
        {
            "arr": [[1, 2], [3, 4]],
            "payload": [{"a": 1}, '{"b":2}'],
            "blob": [b"\x00\x01", b"\x02"],
        }
    )

    rows = _copy_dataframe(
        cur,
        target,
        df,
        column_sql_types={"arr": "bigint[]", "payload": "jsonb", "blob": "bytea"},
    )

    assert rows == 2
    parsed_rows = list(csv.reader(StringIO(cur.payload_text)))
    assert parsed_rows[0][0] == "{1,2}"
    assert parsed_rows[1][0] == "{3,4}"
    assert json.loads(parsed_rows[0][1]) == {"a": 1}
    assert json.loads(parsed_rows[1][1]) == {"b": 2}
    assert parsed_rows[0][2] == "\\x0001"
    assert parsed_rows[1][2] == "\\x02"


def test_copy_dataframe_nested_arrays_raise_clear_error():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame({"arr": [[[1, 2]]]})

    with pytest.raises(RuntimeError):
        _copy_dataframe(cur, target, df, column_sql_types={"arr": "bigint[]"})


def test_copy_dataframe_jsonb_non_json_string_is_encoded_as_json_string():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame({"payload": ["plain-text"]})

    _copy_dataframe(cur, target, df, column_sql_types={"payload": "jsonb"})
    parsed_rows = list(csv.reader(StringIO(cur.payload_text)))
    assert json.loads(parsed_rows[0][0]) == "plain-text"


def test_copy_dataframe_jsonb_scalar_like_string_stays_json_string():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame({"payload": ["true", "123", "null"]})

    _copy_dataframe(cur, target, df, column_sql_types={"payload": "jsonb"})
    parsed_rows = list(csv.reader(StringIO(cur.payload_text)))
    assert json.loads(parsed_rows[0][0]) == "true"
    assert json.loads(parsed_rows[1][0]) == "123"
    assert json.loads(parsed_rows[2][0]) == "null"


def test_copy_dataframe_jsonb_object_array_strings_pass_through():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame({"payload": ['{"a":1}', "[1,2,3]"]})

    _copy_dataframe(cur, target, df, column_sql_types={"payload": "jsonb"})
    parsed_rows = list(csv.reader(StringIO(cur.payload_text)))
    assert json.loads(parsed_rows[0][0]) == {"a": 1}
    assert json.loads(parsed_rows[1][0]) == [1, 2, 3]


def test_copy_dataframe_array_serializes_null_like_elements():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame({"arr": [[1, None], [2, pd.NA]]})

    _copy_dataframe(cur, target, df, column_sql_types={"arr": "bigint[]"})
    parsed_rows = list(csv.reader(StringIO(cur.payload_text)))
    assert parsed_rows[0][0] == "{1,NULL}"
    assert parsed_rows[1][0] == "{2,NULL}"


def test_copy_dataframe_array_rejects_invalid_boolean_literals():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame({"arr": [[True, "maybe"]]})

    with pytest.raises(RuntimeError):
        _copy_dataframe(cur, target, df, column_sql_types={"arr": "boolean[]"})


def test_copy_dataframe_boolean_scalar_allows_string_true_false_aliases():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame({"flag": ["yes", "off"]})

    _copy_dataframe(cur, target, df, column_sql_types={"flag": "boolean"})
    parsed_rows = list(csv.reader(StringIO(cur.payload_text)))
    assert parsed_rows[0][0] == "true"
    assert parsed_rows[1][0] == "false"


def test_copy_dataframe_boolean_scalar_accepts_numpy_bool():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame({"flag": [np.bool_(True), np.bool_(False)]}, dtype="object")

    _copy_dataframe(cur, target, df, column_sql_types={"flag": "boolean"})
    parsed_rows = list(csv.reader(StringIO(cur.payload_text)))
    assert parsed_rows[0][0] == "true"
    assert parsed_rows[1][0] == "false"


def test_copy_dataframe_rejects_ambiguous_stringified_column_names():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame([[1, 2]], columns=[1, "1"])

    with pytest.raises(RuntimeError):
        _copy_dataframe(cur, target, df, column_sql_types={"1": "bigint"})


def _extract_null_marker(copy_sql: str) -> str:
    # The COPY SQL is e.g. "... null '__dbt_pybridge_null_<hex>__')" — pull the
    # marker out so the test can compare what landed in the CSV against it.
    prefix = "null '"
    start = copy_sql.index(prefix) + len(prefix)
    end = copy_sql.index("'", start)
    return copy_sql[start:end]


def test_copy_dataframe_nullable_int64_writes_null_marker():
    # Regression: a nullable Int64 column with pd.NA used to land in the CSV
    # untouched (pandas' to_csv doesn't always honor na_rep for extension
    # dtypes), so Postgres tried to parse the marker as bigint and errored out.
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame({"id": pd.array([1, pd.NA, 3], dtype="Int64")})

    _copy_dataframe(cur, target, df, column_sql_types={"id": "bigint"})
    marker = _extract_null_marker(cur.copy_sql)
    parsed_rows = list(csv.reader(StringIO(cur.payload_text)))
    assert parsed_rows[0][0] == "1"
    assert parsed_rows[1][0] == marker
    assert parsed_rows[2][0] == "3"


def test_copy_dataframe_object_column_with_none_writes_null_marker():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame({"name": ["a", None, "c"]})

    _copy_dataframe(cur, target, df, column_sql_types={"name": "text"})
    marker = _extract_null_marker(cur.copy_sql)
    parsed_rows = list(csv.reader(StringIO(cur.payload_text)))
    assert parsed_rows[0][0] == "a"
    assert parsed_rows[1][0] == marker
    assert parsed_rows[2][0] == "c"


def test_copy_dataframe_datetime_with_nat_writes_null_marker():
    cur = FakeCursor()
    target = TargetRelation(database=None, schema="public", identifier="x")
    df = pd.DataFrame({"ts": pd.to_datetime(["2025-01-01", None, "2025-01-03"])})

    _copy_dataframe(cur, target, df, column_sql_types={"ts": "timestamp"})
    marker = _extract_null_marker(cur.copy_sql)
    parsed_rows = list(csv.reader(StringIO(cur.payload_text)))
    assert parsed_rows[1][0] == marker
