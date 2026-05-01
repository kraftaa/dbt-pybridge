from __future__ import annotations

from io import StringIO
from typing import Iterable

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


def postgres_type_for_series(series) -> str:
    from pandas.api.types import (
        is_bool_dtype,
        is_datetime64_any_dtype,
        is_float_dtype,
        is_integer_dtype,
    )

    if is_bool_dtype(series.dtype):
        return "boolean"
    if is_integer_dtype(series.dtype):
        return "bigint"
    if is_float_dtype(series.dtype):
        return "double precision"
    if is_datetime64_any_dtype(series.dtype):
        return "timestamp"
    return "text"


def _create_table_for_dataframe(cur, target: TargetRelation, df, replace: bool) -> None:
    if df.columns.empty:
        raise RuntimeError("Python model returned a dataframe with zero columns")

    if target.schema:
        cur.execute(f"create schema if not exists {quote_ident(target.schema)}")

    target_sql = target.render()
    if replace:
        cur.execute(f"drop table if exists {target_sql}")

    cols_sql = ", ".join(
        f"{quote_ident(str(col))} {postgres_type_for_series(df[str(col)])}"
        for col in df.columns
    )
    cur.execute(f"create table {target_sql} ({cols_sql})")


def _copy_dataframe(cur, target: TargetRelation, df) -> int:
    if df.empty:
        return 0

    payload = StringIO()
    df.to_csv(payload, index=False, header=False, na_rep="\\N")
    payload.seek(0)

    columns_csv = ", ".join(quote_ident(str(col)) for col in df.columns)
    copy_sql = (
        f"copy {target.render()} ({columns_csv}) "
        "from stdin with (format csv, null '\\N')"
    )
    cur.copy_expert(copy_sql, payload)
    return len(df)


def write_model_result(conn, target: TargetRelation, result, batch_size: int = 100_000) -> int:
    if result is None:
        raise RuntimeError("Python model returned None; expected dataframe or iterable of dataframes")

    if is_pandas_df(result) or is_polars_df(result):
        df = to_pandas(result)
        with conn.cursor() as cur:
            _create_table_for_dataframe(cur, target, df, replace=True)
            rows_written = _copy_dataframe(cur, target, df)
        conn.commit()
        return rows_written

    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, dict)):
        rows_written = 0
        created = False
        with conn.cursor() as cur:
            for chunk in result:
                if not (is_pandas_df(chunk) or is_polars_df(chunk)):
                    raise TypeError(
                        "Chunked Python model must yield pandas or polars dataframes; "
                        f"got {type(chunk)!r}"
                    )
                chunk_df = to_pandas(chunk)
                if not created:
                    _create_table_for_dataframe(cur, target, chunk_df, replace=True)
                    created = True
                rows_written += _copy_dataframe(cur, target, chunk_df)

            if not created:
                raise RuntimeError("Chunked Python model yielded no dataframes")

        conn.commit()
        return rows_written

    raise TypeError(
        "Unsupported Python model result type. Expected pandas/polars dataframe or iterable of dataframes; "
        f"got {type(result)!r}"
    )
