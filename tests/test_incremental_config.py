import pytest

from dbt_pybridge.dataframe_io import _align_and_validate_columns, _validate_unique_key, write_model_result
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


def test_align_validate_columns_reorders():
    df = pd.DataFrame({"b": [1], "a": [2]})
    out = _align_and_validate_columns(df, ["a", "b"])
    assert list(out.columns) == ["a", "b"]


def test_validate_unique_key_missing_column():
    with pytest.raises(RuntimeError):
        _validate_unique_key(["missing"], ["id", "name"])


def test_write_model_result_rejects_unsupported_materialization():
    df = pd.DataFrame({"id": [1]})
    with pytest.raises(RuntimeError):
        write_model_result(
            conn=None,
            target=TargetRelation(database=None, schema="public", identifier="x"),
            result=df,
            materialized="view",
        )

