from datetime import date, datetime, time, timezone
from decimal import Decimal
import uuid

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from dbt_pybridge.dataframe_io import postgres_type_for_series


def test_dtype_mapping_int():
    assert postgres_type_for_series(pd.Series([1, 2, 3])) == "bigint"


def test_dtype_mapping_float():
    assert postgres_type_for_series(pd.Series([1.2, 2.3])) == "double precision"


def test_dtype_mapping_float32():
    assert postgres_type_for_series(pd.Series([1.2, 2.3], dtype="float32")) == "real"


def test_dtype_mapping_bool():
    assert postgres_type_for_series(pd.Series([True, False])) == "boolean"


def test_dtype_mapping_int16():
    assert postgres_type_for_series(pd.Series([1, 2], dtype="int16")) == "smallint"


def test_dtype_mapping_int32():
    assert postgres_type_for_series(pd.Series([1, 2], dtype="int32")) == "integer"


def test_dtype_mapping_uint32():
    assert postgres_type_for_series(pd.Series([1, 2], dtype="uint32")) == "bigint"


def test_dtype_mapping_uint64():
    assert postgres_type_for_series(pd.Series([1, 2], dtype="uint64")) == "numeric"


def test_dtype_mapping_text_object():
    assert postgres_type_for_series(pd.Series(["a", "b"])) == "text"


def test_dtype_mapping_timestamptz():
    s = pd.Series(pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True))
    assert postgres_type_for_series(s) == "timestamptz"


def test_dtype_mapping_date_object():
    assert postgres_type_for_series(pd.Series([date(2025, 1, 1)])) == "date"


def test_dtype_mapping_time_object():
    assert postgres_type_for_series(pd.Series([time(10, 30, 0)])) == "time"


def test_dtype_mapping_timetz_object():
    assert postgres_type_for_series(pd.Series([time(10, 30, 0, tzinfo=timezone.utc)])) == "timetz"


def test_dtype_mapping_interval():
    assert postgres_type_for_series(pd.Series(pd.to_timedelta([1, 2], unit="h"))) == "interval"


def test_dtype_mapping_decimal_object():
    assert postgres_type_for_series(pd.Series([Decimal("12.34")])) == "numeric(4,2)"


def test_dtype_mapping_uuid_object():
    assert postgres_type_for_series(pd.Series([uuid.uuid4()])) == "uuid"


def test_dtype_mapping_jsonb_object():
    assert postgres_type_for_series(pd.Series([{"k": "v"}])) == "jsonb"


def test_dtype_mapping_bytea_object():
    assert postgres_type_for_series(pd.Series([b"\x00\x01"])) == "bytea"


def test_dtype_mapping_datetime_object_with_timezone():
    assert postgres_type_for_series(pd.Series([datetime(2025, 1, 1, tzinfo=timezone.utc)])) == "timestamptz"


def test_dtype_mapping_int_array_object():
    assert postgres_type_for_series(pd.Series([[1, 2], [3]])) == "bigint[]"


def test_dtype_mapping_text_array_object():
    assert postgres_type_for_series(pd.Series([["a", "b"], ["c"]])) == "text[]"


def test_dtype_mapping_uuid_array_object():
    assert postgres_type_for_series(pd.Series([[uuid.uuid4(), uuid.uuid4()]])) == "uuid[]"


def test_dtype_mapping_mixed_array_falls_back_to_jsonb():
    assert postgres_type_for_series(pd.Series([[1, "x"], [2]])) == "jsonb"


def test_dtype_mapping_nested_array_falls_back_to_jsonb():
    assert postgres_type_for_series(pd.Series([[[1, 2]], [[3]]])) == "jsonb"


def test_dtype_mapping_mixed_object_falls_back_to_text():
    assert postgres_type_for_series(pd.Series([1, "x"])) == "text"


def test_dtype_mapping_mixed_datetime_tz_falls_back_to_text():
    assert (
        postgres_type_for_series(
            pd.Series([datetime(2025, 1, 1), datetime(2025, 1, 1, tzinfo=timezone.utc)])
        )
        == "text"
    )


def test_dtype_mapping_mixed_json_containers_prefers_jsonb():
    assert postgres_type_for_series(pd.Series([{"a": 1}, [1, 2, 3]])) == "jsonb"


def test_dtype_mapping_numpy_object_int64_is_bigint():
    assert postgres_type_for_series(pd.Series([np.int64(1)], dtype="object")) == "bigint"


def test_dtype_mapping_numpy_object_bool_is_boolean():
    assert postgres_type_for_series(pd.Series([np.bool_(True)], dtype="object")) == "boolean"
