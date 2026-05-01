from __future__ import annotations

from io import StringIO
from typing import Iterable, List, Optional, Sequence

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


def _table_columns(cur, target: TargetRelation) -> List[str]:
    if not _table_exists(cur, target):
        return []
    if target.schema:
        cur.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = %s and table_name = %s
            order by ordinal_position
            """,
            (target.schema, target.identifier),
        )
    else:
        cur.execute(
            """
            select c.column_name
            from information_schema.columns c
            join pg_catalog.pg_class cls on cls.relname = c.table_name
            join pg_catalog.pg_namespace n on n.nspname = c.table_schema and n.oid = cls.relnamespace
            where c.table_name = %s and pg_table_is_visible(cls.oid)
            order by c.ordinal_position
            limit 10000
            """,
            (target.identifier,),
        )
    return [row[0] for row in cur.fetchall()]


def _align_and_validate_columns(df, target_columns: Sequence[str]):
    df_columns = [str(col) for col in df.columns]
    if set(df_columns) != set(target_columns):
        raise RuntimeError(
            "Incremental Python model columns must match target table columns exactly. "
            f"Target columns: {list(target_columns)}; model columns: {df_columns}"
        )
    if list(df_columns) == list(target_columns):
        return df
    return df[list(target_columns)]


def _temp_relation(target: TargetRelation) -> TargetRelation:
    schema = target.schema or "public"
    normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in target.identifier.lower())
    suffix = normalized[:30] if normalized else "model"
    identifier = f"__dbt_pybridge_tmp_{suffix}"
    return TargetRelation(database=None, schema=schema, identifier=identifier)


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
) -> int:
    if chunk_df.empty:
        return 0

    target_columns = _table_columns(cur, target)
    if not target_columns:
        _create_table_for_dataframe(cur, target, chunk_df, replace=True)
        return _copy_dataframe(cur, target, chunk_df)

    aligned = _align_and_validate_columns(chunk_df, target_columns)
    if incremental_strategy == "append":
        return _copy_dataframe(cur, target, aligned)

    if incremental_strategy not in {"merge", "delete+insert"}:
        raise RuntimeError(
            "Unsupported Python incremental strategy. "
            f"Got '{incremental_strategy}', expected one of: append, merge, delete+insert"
        )

    keys = _validate_unique_key(unique_key, target_columns)
    temp = _temp_relation(target)
    temp_sql = temp.render()
    cur.execute(f"drop table if exists {temp_sql}")
    _create_table_for_dataframe(cur, temp, aligned, replace=True)
    _copy_dataframe(cur, temp, aligned)
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
        with conn.cursor() as cur:
            if materialized == "table":
                _create_table_for_dataframe(cur, target, df, replace=True)
                rows_written = _copy_dataframe(cur, target, df)
            else:
                rows_written = _apply_incremental_chunk(
                    cur,
                    target,
                    df,
                    incremental_strategy=incremental_strategy,
                    unique_key=unique_key,
                )
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
                if materialized == "table":
                    if not created:
                        _create_table_for_dataframe(cur, target, chunk_df, replace=True)
                        created = True
                    rows_written += _copy_dataframe(cur, target, chunk_df)
                else:
                    rows_written += _apply_incremental_chunk(
                        cur,
                        target,
                        chunk_df,
                        incremental_strategy=incremental_strategy,
                        unique_key=unique_key,
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
