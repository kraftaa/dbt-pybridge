import pytest

pd = pytest.importorskip("pandas")

from dbt_pybridge.dataframe_io import postgres_type_for_series


def test_dtype_mapping_int():
    assert postgres_type_for_series(pd.Series([1, 2, 3])) == "bigint"


def test_dtype_mapping_float():
    assert postgres_type_for_series(pd.Series([1.2, 2.3])) == "double precision"


def test_dtype_mapping_bool():
    assert postgres_type_for_series(pd.Series([True, False])) == "boolean"


def test_dtype_mapping_text_object():
    assert postgres_type_for_series(pd.Series(["a", "b"])) == "text"
