from __future__ import annotations

from datetime import date as py_date
from datetime import datetime as py_datetime
from datetime import time as py_time
from decimal import Decimal
from io import StringIO
import json
import math
import re
import uuid
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from dbt_pybridge.session import TargetRelation, quote_ident


def is_pandas_df(value) -> bool:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return False
    return isinstance(value, pd.DataFrame)


def is_polars_df(value) -> bool:
    try:
        import polars as pl
    except ModuleNotFoundError:
        return False
    return isinstance(value, pl.DataFrame)


def to_pandas(df):
    import pandas as pd

    if is_pandas_df(df):
        return df
    if is_polars_df(df):
        return df.to_pandas()
    raise TypeError(f"Unsupported dataframe type: {type(df)!r}")


def _to_python_scalar(value):
    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        value_type = type(value)
        module = getattr(value_type, "__module__", "")
        if isinstance(module, str) and module.startswith("numpy"):
            try:
                return item_fn()
            except Exception:
                pass
    return value


def _is_null_scalar(value) -> bool:
    import pandas as pd

    value = _to_python_scalar(value)
    if value is None:
        return True
    result = pd.isna(value)
    if isinstance(result, bool):
        return result
    return False


def _iter_non_null_values(series, limit: int = 128):
    count = 0
    for value in series.values:
        if _is_null_scalar(value):
            continue
        yield value
        count += 1
        if count >= limit:
            break


def _infer_integer_type(dtype) -> str:
    dtype_name = str(dtype).lower()
    match = re.match(r"^(u?)int(\d+)$", dtype_name)
    if not match:
        return "bigint"

    is_unsigned = bool(match.group(1))
    bits = int(match.group(2))

    if not is_unsigned:
        if bits <= 16:
            return "smallint"
        if bits <= 32:
            return "integer"
        if bits <= 64:
            return "bigint"
        return "numeric"

    if bits <= 8:
        return "smallint"
    if bits <= 16:
        return "integer"
    if bits <= 32:
        return "bigint"
    return "numeric"


def _infer_float_type(dtype) -> str:
    dtype_name = str(dtype).lower()
    match = re.match(r"^float(\d+)$", dtype_name)
    if not match:
        return "double precision"
    bits = int(match.group(1))
    if bits <= 32:
        return "real"
    return "double precision"


def _infer_decimal_type(values) -> str:
    max_precision = 0
    max_scale = 0

    for value in values:
        if not isinstance(value, Decimal):
            return "numeric"
        if not value.is_finite():
            return "numeric"
        _, digits, exponent = value.as_tuple()
        if exponent >= 0:
            scale = 0
            precision = len(digits) + exponent
        else:
            scale = -exponent
            precision = max(len(digits), scale)
        max_precision = max(max_precision, precision)
        max_scale = max(max_scale, scale)

    if max_precision <= 0:
        return "numeric"
    if max_precision > 1000 or max_scale > 1000:
        return "numeric"
    return f"numeric({max_precision},{max_scale})"


def _infer_array_element_type(sample_values) -> Optional[str]:
    element_type = None
    has_values = False
    for value in sample_values:
        value = _to_python_scalar(value)
        if _is_null_scalar(value):
            continue
        has_values = True
        if isinstance(value, bool):
            current = "boolean"
        elif isinstance(value, int) and not isinstance(value, bool):
            current = "bigint"
        elif isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            current = "double precision"
        elif isinstance(value, str):
            current = "text"
        elif isinstance(value, uuid.UUID):
            current = "uuid"
        elif isinstance(value, py_datetime):
            current = "timestamptz" if value.tzinfo is not None else "timestamp"
        elif isinstance(value, py_date):
            current = "date"
        elif isinstance(value, py_time):
            current = "timetz" if value.tzinfo is not None else "time"
        elif isinstance(value, Decimal):
            current = "numeric"
        else:
            return None

        if element_type is None:
            element_type = current
        elif element_type != current:
            return None

    if not has_values or element_type is None:
        return None
    return f"{element_type}[]"


def _infer_object_type(series) -> str:
    sample_values = [_to_python_scalar(v) for v in _iter_non_null_values(series, limit=128)]
    if not sample_values:
        return "text"

    if all(isinstance(v, Decimal) for v in sample_values):
        return _infer_decimal_type(sample_values)
    if all(isinstance(v, uuid.UUID) for v in sample_values):
        return "uuid"
    if all(isinstance(v, dict) for v in sample_values):
        return "jsonb"
    if all(isinstance(v, (bytes, bytearray, memoryview)) for v in sample_values):
        return "bytea"
    if all(isinstance(v, py_datetime) for v in sample_values):
        tz_flags = {v.tzinfo is not None for v in sample_values}
        if len(tz_flags) > 1:
            return "text"
        first = sample_values[0]
        return "timestamptz" if first.tzinfo is not None else "timestamp"
    if all(isinstance(v, py_date) and not isinstance(v, py_datetime) for v in sample_values):
        return "date"
    if all(isinstance(v, py_time) for v in sample_values):
        tz_flags = {v.tzinfo is not None for v in sample_values}
        if len(tz_flags) > 1:
            return "text"
        first = sample_values[0]
        return "timetz" if first.tzinfo is not None else "time"
    if all(isinstance(v, (list, tuple)) for v in sample_values):
        element_samples = []
        for row_value in sample_values:
            if _is_null_scalar(row_value):
                continue
            element_samples.extend(row_value)
        array_type = _infer_array_element_type(element_samples)
        return array_type or "jsonb"
    if all(isinstance(v, str) for v in sample_values):
        return "text"
    if all(isinstance(v, bool) for v in sample_values):
        return "boolean"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in sample_values):
        return "bigint"
    if all(isinstance(v, float) for v in sample_values):
        return "double precision"
    has_container = any(isinstance(v, (dict, list, tuple)) for v in sample_values)
    if has_container:
        try:
            for value in sample_values:
                _to_json_text(value)
            return "jsonb"
        except TypeError:
            pass
    return "text"


def postgres_type_for_series(series) -> str:
    import pandas as pd

    from pandas.api.types import (
        is_bool_dtype,
        is_datetime64_any_dtype,
        is_float_dtype,
        is_integer_dtype,
        is_object_dtype,
        is_string_dtype,
        is_timedelta64_dtype,
    )

    if is_bool_dtype(series.dtype):
        return "boolean"
    if is_integer_dtype(series.dtype):
        return _infer_integer_type(series.dtype)
    if is_float_dtype(series.dtype):
        return _infer_float_type(series.dtype)
    if is_timedelta64_dtype(series.dtype):
        return "interval"
    if is_datetime64_any_dtype(series.dtype):
        try:
            if getattr(series.dt, "tz", None) is not None:
                return "timestamptz"
        except AttributeError:
            pass
        return "timestamp"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "text"
    if is_object_dtype(series.dtype):
        return _infer_object_type(series)
    if is_string_dtype(series.dtype):
        return "text"
    return "text"


_POSTGRES_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\s(),\[\]]*$")


def _validate_postgres_type_sql(column: str, postgres_type_sql: str) -> str:
    normalized = postgres_type_sql.strip()
    if not normalized:
        raise RuntimeError(f"Configured postgres type for column '{column}' cannot be empty")
    if not _POSTGRES_TYPE_RE.fullmatch(normalized):
        raise RuntimeError(
            "Configured postgres type contains unsupported characters for "
            f"column '{column}': {postgres_type_sql!r}"
        )
    return normalized


def _normalize_sql_type_name(type_sql: str) -> str:
    return " ".join(type_sql.strip().lower().split())


def _json_default_encoder(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (py_date, py_datetime, py_time)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return data.hex()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _to_json_text(value) -> str:
    if isinstance(value, str):
        stripped = value.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(value)
                return value
            except ValueError:
                pass
        return json.dumps(value)
    return json.dumps(value, default=_json_default_encoder)


def _to_bytea_text(value) -> str:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return "\\x" + value.hex()
    if isinstance(value, str):
        return value
    raise RuntimeError(f"Cannot serialize value of type {type(value)!r} to bytea")


_ARRAY_NUMERIC_BASE_TYPES = {
    "smallint",
    "integer",
    "bigint",
    "real",
    "double precision",
    "numeric",
}


_ARRAY_BOOLEAN_BASE_TYPES = {"boolean"}


_ARRAY_TEXTLIKE_BASE_TYPES = {
    "text",
    "character varying",
    "varchar",
    "character",
    "char",
    "citext",
}


def _quote_array_text_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


_PG_BOOL_TRUE_VALUES = {"t", "true", "y", "yes", "on", "1"}
_PG_BOOL_FALSE_VALUES = {"f", "false", "n", "no", "off", "0"}


def _to_pg_bool_text(value) -> str:
    value = _to_python_scalar(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        if value == 1:
            return "true"
        if value == 0:
            return "false"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _PG_BOOL_TRUE_VALUES:
            return "true"
        if lowered in _PG_BOOL_FALSE_VALUES:
            return "false"
    raise RuntimeError(
        f"Cannot serialize value {value!r} ({type(value)!r}) to Postgres boolean literal."
    )


def _array_scalar_text(value, base_type: str) -> str:
    value = _to_python_scalar(value)
    if _is_null_scalar(value):
        return "NULL"
    if isinstance(value, py_datetime):
        value = value.isoformat(sep=" ")
    elif isinstance(value, (py_date, py_time)):
        value = value.isoformat()
    elif isinstance(value, uuid.UUID):
        value = str(value)
    elif isinstance(value, Decimal):
        value = str(value)

    if base_type in _ARRAY_BOOLEAN_BASE_TYPES:
        return _to_pg_bool_text(value)

    if base_type in _ARRAY_NUMERIC_BASE_TYPES:
        if isinstance(value, bool):
            raise RuntimeError("Boolean values cannot be serialized into numeric array elements")
        return str(value)

    if base_type in {"json", "jsonb"}:
        return _quote_array_text_value(_to_json_text(value))

    if base_type in _ARRAY_TEXTLIKE_BASE_TYPES:
        return _quote_array_text_value(str(value))

    if base_type == "bytea":
        return _quote_array_text_value(_to_bytea_text(value))

    return _quote_array_text_value(str(value))


def _to_array_text(value, base_type: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(
            f"Cannot serialize value of type {type(value)!r} to array type {base_type}[]; "
            "expected list/tuple or a preformatted array literal string."
        )
    if any(isinstance(v, (list, tuple)) for v in value):
        raise RuntimeError("Nested Python lists are not supported for Postgres array COPY serialization")
    rendered = [_array_scalar_text(v, base_type) for v in value]
    return "{" + ",".join(rendered) + "}"


def _serialize_value_for_sql_type(value, sql_type: str):
    value = _to_python_scalar(value)
    if _is_null_scalar(value):
        return None

    normalized = _normalize_sql_type_name(sql_type)
    if normalized.endswith("[]"):
        base_type = normalized[:-2]
        return _to_array_text(value, base_type)
    if normalized in {"json", "jsonb"}:
        return _to_json_text(value)
    if normalized == "bytea":
        return _to_bytea_text(value)
    if normalized in _ARRAY_BOOLEAN_BASE_TYPES:
        return _to_pg_bool_text(value)
    if isinstance(value, py_datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, (py_date, py_time)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    return value


def _prepare_dataframe_for_copy(df, column_sql_types: Dict[str, str]):
    import pandas as pd

    # We rebuild each column from a Python list into an object-dtype Series so
    # pandas can't sneak nullable-extension behavior back in (Int64 → float64
    # via .where, pd.NA surviving .astype(object), to_csv ignoring na_rep on
    # extension dtypes, etc). Every null-like cell becomes Python None, which
    # to_csv replaces with na_rep reliably.
    prepared = df.copy()
    for col in prepared.columns:
        col_name = str(col)
        sql_type = column_sql_types.get(col_name)
        if sql_type:
            values = [_serialize_value_for_sql_type(v, sql_type) for v in prepared[col]]
        else:
            values = [None if _is_null_scalar(v) else _to_python_scalar(v) for v in prepared[col]]
        prepared[col] = pd.Series(values, index=prepared[col].index, dtype=object)
    return prepared


def _validate_unique_stringified_columns(df) -> None:
    labels = list(df.columns)
    stringified = [str(col) for col in labels]
    seen = set()
    duplicates = []
    for name in stringified:
        if name in seen:
            duplicates.append(name)
        else:
            seen.add(name)
    if duplicates:
        raise RuntimeError(
            "Python model dataframe has ambiguous column labels after string normalization. "
            f"Duplicate normalized names: {sorted(set(duplicates))}. "
            "Rename columns so each column has a unique string name."
        )


def _resolve_column_sql_types(
    df,
    column_types: Optional[Dict[str, str]] = None,
    categorical_types: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    import pandas as pd

    if df.columns.empty:
        raise RuntimeError("Python model returned a dataframe with zero columns")
    _validate_unique_stringified_columns(df)

    known_columns = {str(col) for col in df.columns}
    column_type_overrides: Dict[str, str] = {}
    if column_types:
        unknown = sorted(set(column_types) - known_columns)
        if unknown:
            raise RuntimeError(
                "pybridge_column_types (legacy localpy_column_types) has keys that are not in dataframe columns. "
                f"Unknown keys: {unknown}; dataframe columns: {sorted(known_columns)}"
            )
        column_type_overrides = {
            col: _validate_postgres_type_sql(col, pg_type)
            for col, pg_type in column_types.items()
        }

    categorical_type_overrides: Dict[str, str] = {}
    if categorical_types:
        unknown = sorted(set(categorical_types) - known_columns)
        if unknown:
            raise RuntimeError(
                "pybridge_categorical_types (legacy localpy_categorical_types) has keys that are not in dataframe columns. "
                f"Unknown keys: {unknown}; dataframe columns: {sorted(known_columns)}"
            )
        for col, pg_type in categorical_types.items():
            series = df[str(col)]
            if not isinstance(series.dtype, pd.CategoricalDtype):
                raise RuntimeError(
                    "pybridge_categorical_types (legacy localpy_categorical_types) can only be used for categorical columns. "
                    f"Column '{col}' has dtype {series.dtype!s}."
                )
            categorical_type_overrides[col] = _validate_postgres_type_sql(col, pg_type)

    resolved: Dict[str, str] = {}
    for col in df.columns:
        col_name = str(col)
        if col_name in column_type_overrides:
            resolved[col_name] = column_type_overrides[col_name]
        elif col_name in categorical_type_overrides:
            resolved[col_name] = categorical_type_overrides[col_name]
        else:
            resolved[col_name] = postgres_type_for_series(df[col_name])
    return resolved


def _create_table_for_dataframe(
    cur,
    target: TargetRelation,
    df,
    replace: bool,
    column_types: Optional[Dict[str, str]] = None,
    categorical_types: Optional[Dict[str, str]] = None,
    temporary: bool = False,
) -> None:
    if df.columns.empty:
        raise RuntimeError("Python model returned a dataframe with zero columns")
    _validate_unique_stringified_columns(df)

    if target.schema and not temporary:
        cur.execute(f"create schema if not exists {quote_ident(target.schema)}")

    target_sql = target.render()
    if replace and not temporary:
        # CASCADE matches dbt-core's drop_relation convention; downstream views
        # would otherwise block the rebuild and they'll be rebuilt on the next
        # run anyway. TEMP tables can't be replaced with this name (each call
        # uses a unique nonce), so the drop is skipped for them.
        cur.execute(f"drop table if exists {target_sql} cascade")

    resolved_types = _resolve_column_sql_types(
        df,
        column_types=column_types,
        categorical_types=categorical_types,
    )

    cols_sql = ", ".join(
        f"{quote_ident(str(col))} "
        f"{resolved_types[str(col)]}"
        for col in df.columns
    )
    create_keyword = "create temp table" if temporary else "create table"
    cur.execute(f"{create_keyword} {target_sql} ({cols_sql})")


def _copy_dataframe(cur, target: TargetRelation, df, column_sql_types: Optional[Dict[str, str]] = None) -> int:
    if df.empty:
        return 0
    _validate_unique_stringified_columns(df)

    copy_df = _prepare_dataframe_for_copy(df, column_sql_types or {})
    null_marker = f"__dbt_pybridge_null_{uuid.uuid4().hex}__"
    payload = StringIO()
    copy_df.to_csv(payload, index=False, header=False, na_rep=null_marker)
    payload.seek(0)

    columns_csv = ", ".join(quote_ident(str(col)) for col in df.columns)
    copy_sql = (
        f"copy {target.render()} ({columns_csv}) "
        f"from stdin with (format csv, null '{null_marker}', force_null ({columns_csv}))"
    )
    cur.copy_expert(copy_sql, payload)
    return len(df)


def _table_exists(cur, target: TargetRelation) -> bool:
    if target.schema:
        cur.execute(
            """
            select 1
            from pg_catalog.pg_class c
            join pg_catalog.pg_namespace n on n.oid = c.relnamespace
            where n.nspname = %s and c.relname = %s and c.relkind in ('r', 'p')
            limit 1
            """,
            (target.schema, target.identifier),
        )
    else:
        cur.execute(
            """
            select 1
            from pg_catalog.pg_class c
            where c.relname = %s and c.relkind in ('r', 'p') and pg_table_is_visible(c.oid)
            limit 1
            """,
            (target.identifier,),
        )
    return cur.fetchone() is not None


def _table_columns_with_types(cur, target: TargetRelation) -> List[Tuple[str, str]]:
    if not _table_exists(cur, target):
        return []
    if target.schema:
        cur.execute(
            """
            select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod)
            from pg_catalog.pg_attribute a
            join pg_catalog.pg_class c on c.oid = a.attrelid
            join pg_catalog.pg_namespace n on n.oid = c.relnamespace
            where n.nspname = %s
              and c.relname = %s
              and a.attnum > 0
              and not a.attisdropped
            order by a.attnum
            """,
            (target.schema, target.identifier),
        )
    else:
        cur.execute(
            """
            select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod)
            from pg_catalog.pg_attribute a
            join pg_catalog.pg_class c on c.oid = a.attrelid
            where c.relname = %s
              and pg_table_is_visible(c.oid)
              and a.attnum > 0
              and not a.attisdropped
            order by a.attnum
            """,
            (target.identifier,),
        )
    return [(row[0], row[1]) for row in cur.fetchall()]


def _align_and_validate_columns(df, target_columns: Sequence[str]):
    df_labels = list(df.columns)
    df_columns = [str(col) for col in df_labels]
    if set(df_columns) != set(target_columns):
        raise RuntimeError(
            "Incremental Python model columns must match target table columns exactly. "
            f"Target columns: {list(target_columns)}; model columns: {df_columns}"
        )
    if list(df_columns) == list(target_columns):
        return df

    labels_by_name = {}
    for label, name in zip(df_labels, df_columns):
        if name in labels_by_name:
            raise RuntimeError(
                "Python model columns become ambiguous when cast to string. "
                f"Column name '{name}' appears multiple times after normalization."
            )
        labels_by_name[name] = label

    ordered_labels = [labels_by_name[col] for col in target_columns]
    return df[ordered_labels]


def _temp_relation(target: TargetRelation) -> TargetRelation:
    # Schema is intentionally None — the table is created with CREATE TEMP TABLE
    # and lives in pg_temp for the session, so it auto-drops on disconnect/crash.
    normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in target.identifier.lower())
    suffix = normalized[:22] if normalized else "model"
    nonce = uuid.uuid4().hex[:8]
    identifier = f"__dbt_pybridge_tmp_{suffix}_{nonce}"
    return TargetRelation(database=None, schema=None, identifier=identifier)


def _validate_unique_key(unique_key: Optional[Sequence[str]], columns: Sequence[str]) -> List[str]:
    if not unique_key:
        raise RuntimeError(
            "Python incremental strategy 'merge' requires config(unique_key=...) "
            "as a string or list of strings."
        )
    keys = [str(k) for k in unique_key]
    missing = [k for k in keys if k not in columns]
    if missing:
        raise RuntimeError(
            f"Python incremental unique_key columns not found in model output: {missing}. "
            f"Available columns: {list(columns)}"
        )
    return keys


def _merge_from_temp(cur, target: TargetRelation, temp_sql: str, columns: Sequence[str], unique_key: Sequence[str]) -> None:
    target_sql = target.render()
    update_columns = [col for col in columns if col not in unique_key]
    on_clause = " and ".join(
        f"t.{quote_ident(col)} = s.{quote_ident(col)}" for col in unique_key
    )
    quoted_columns = ", ".join(quote_ident(col) for col in columns)
    update_set_sql = ", ".join(
        f"{quote_ident(col)} = s.{quote_ident(col)}" for col in update_columns
    )
    if update_set_sql:
        cur.execute(
            f"""
            update {target_sql} as t
            set {update_set_sql}
            from {temp_sql} as s
            where {on_clause}
            """
        )
    cur.execute(
        f"""
        insert into {target_sql} ({quoted_columns})
        select {quoted_columns}
        from {temp_sql} as s
        where not exists (
            select 1 from {target_sql} as t where {on_clause}
        )
        """
    )


def _delete_insert_from_temp(cur, target: TargetRelation, temp_sql: str, columns: Sequence[str], unique_key: Sequence[str]) -> None:
    target_sql = target.render()
    quoted_columns = ", ".join(quote_ident(col) for col in columns)
    join_pred = " and ".join(f"t.{quote_ident(col)} = s.{quote_ident(col)}" for col in unique_key)
    cur.execute(
        f"""
        delete from {target_sql} as t
        using {temp_sql} as s
        where {join_pred}
        """
    )
    cur.execute(
        f"""
        insert into {target_sql} ({quoted_columns})
        select {quoted_columns}
        from {temp_sql}
        """
    )


def _apply_incremental_chunk(
    cur,
    target: TargetRelation,
    chunk_df,
    incremental_strategy: str,
    unique_key: Optional[Sequence[str]],
    column_types: Optional[Dict[str, str]] = None,
    categorical_types: Optional[Dict[str, str]] = None,
) -> int:
    target_columns_with_types = _table_columns_with_types(cur, target)
    target_columns = [name for name, _ in target_columns_with_types]
    target_column_types = {name: col_type for name, col_type in target_columns_with_types}
    if not target_columns:
        first_run_types = _resolve_column_sql_types(
            chunk_df,
            column_types=column_types,
            categorical_types=categorical_types,
        )
        # First incremental run: create target table even if this chunk has zero rows,
        # so downstream index creation/grants can succeed.
        if chunk_df.empty:
            _create_table_for_dataframe(
                cur,
                target,
                chunk_df,
                replace=True,
                column_types=first_run_types,
            )
            return 0
        _create_table_for_dataframe(
            cur,
            target,
            chunk_df,
            replace=True,
            column_types=first_run_types,
        )
        return _copy_dataframe(cur, target, chunk_df, column_sql_types=first_run_types)

    if chunk_df.empty:
        return 0

    aligned = _align_and_validate_columns(chunk_df, target_columns)
    if incremental_strategy == "append":
        return _copy_dataframe(cur, target, aligned, column_sql_types=target_column_types)

    if incremental_strategy not in {"merge", "delete+insert"}:
        raise RuntimeError(
            "Unsupported Python incremental strategy. "
            f"Got '{incremental_strategy}', expected one of: append, merge, delete+insert"
        )

    keys = _validate_unique_key(unique_key, target_columns)
    temp = _temp_relation(target)
    temp_sql = temp.render()
    _create_table_for_dataframe(
        cur,
        temp,
        aligned,
        replace=False,
        column_types=target_column_types,
        temporary=True,
    )
    _copy_dataframe(cur, temp, aligned, column_sql_types=target_column_types)
    if incremental_strategy == "merge":
        _merge_from_temp(cur, target, temp_sql, target_columns, keys)
    else:
        _delete_insert_from_temp(cur, target, temp_sql, target_columns, keys)
    cur.execute(f"drop table if exists {temp_sql}")
    return len(aligned)


def write_model_result(
    conn,
    target: TargetRelation,
    result,
    batch_size: int = 100_000,
    materialized: str = "table",
    incremental_strategy: str = "append",
    unique_key: Optional[Sequence[str]] = None,
    column_types: Optional[Dict[str, str]] = None,
    categorical_types: Optional[Dict[str, str]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> int:
    if result is None:
        raise RuntimeError("Python model returned None; expected dataframe or iterable of dataframes")

    if materialized not in {"table", "incremental"}:
        raise RuntimeError(
            "Unsupported Python materialization for pybridge. "
            f"Expected 'table' or 'incremental', got '{materialized}'"
        )

    if is_pandas_df(result) or is_polars_df(result):
        df = to_pandas(result)
        if logger is not None:
            logger(f"Writing {len(df):,} rows to {target.render()} ({materialized})")
        with conn.cursor() as cur:
            if materialized == "table":
                resolved_types = _resolve_column_sql_types(
                    df,
                    column_types=column_types,
                    categorical_types=categorical_types,
                )
                _create_table_for_dataframe(
                    cur,
                    target,
                    df,
                    replace=True,
                    column_types=resolved_types,
                )
                rows_written = _copy_dataframe(cur, target, df, column_sql_types=resolved_types)
            else:
                rows_written = _apply_incremental_chunk(
                    cur,
                    target,
                    df,
                    incremental_strategy=incremental_strategy,
                    unique_key=unique_key,
                    column_types=column_types,
                    categorical_types=categorical_types,
                )
        conn.commit()
        return rows_written

    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, dict)):
        rows_written = 0
        created = False
        expected_columns = None
        expected_types = None
        with conn.cursor() as cur:
            for batch_idx, chunk in enumerate(result, start=1):
                if not (is_pandas_df(chunk) or is_polars_df(chunk)):
                    raise TypeError(
                        "Chunked Python model must yield pandas or polars dataframes; "
                        f"got {type(chunk)!r}"
                    )
                chunk_df = to_pandas(chunk)
                if logger is not None:
                    logger(f"Writing batch {batch_idx}, rows={len(chunk_df)}")
                if expected_columns is None:
                    expected_columns = [str(col) for col in chunk_df.columns]
                    if materialized == "table":
                        expected_types = _resolve_column_sql_types(
                            chunk_df,
                            column_types=column_types,
                            categorical_types=categorical_types,
                        )
                else:
                    chunk_df = _align_and_validate_columns(chunk_df, expected_columns)
                if materialized == "table":
                    if not created:
                        _create_table_for_dataframe(
                            cur,
                            target,
                            chunk_df,
                            replace=True,
                            column_types=expected_types,
                        )
                        created = True
                    rows_written += _copy_dataframe(cur, target, chunk_df, column_sql_types=expected_types)
                else:
                    rows_written += _apply_incremental_chunk(
                        cur,
                        target,
                        chunk_df,
                        incremental_strategy=incremental_strategy,
                        unique_key=unique_key,
                        column_types=column_types,
                        categorical_types=categorical_types,
                    )
                    created = True

            if not created:
                raise RuntimeError("Chunked Python model yielded no dataframes")

        conn.commit()
        return rows_written

    raise TypeError(
        "Unsupported Python model result type. Expected pandas/polars dataframe or iterable of dataframes; "
        f"got {type(result)!r}"
    )
