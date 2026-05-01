# dbt-pybridge

`dbt-pybridge` is a dbt adapter that enables Python models in a normal `dbt run` against Postgres.

It works by:
- compiling `.py` models through dbt
- executing Python locally (developer laptop or CI runner)
- loading `dbt.ref()` / `dbt.source()` data into pandas/polars
- writing the returned dataframe back into Postgres

## Status

MVP scope (table materialization only) is implemented.

- Supported: `materialized='table'`
- Supported DAG: `sql -> python -> sql`
- Supported return types: pandas DataFrame, polars DataFrame, or iterable/generator of dataframes

## Install

```bash
pip install -e .
```

Use a supported Python version (3.11/3.12 recommended).

## Profile

Set your profile `type` to `pybridge`:

```yaml
my_profile:
  target: dev
  outputs:
    dev:
      type: pybridge
      host: localhost
      user: postgres
      password: postgres
      port: 5432
      dbname: analytics
      schema: public
      threads: 1
```

## Example model

```python
def model(dbt, session):
    df = dbt.ref("stg_orders")
    df["double_amount"] = df["amount"] * 2
    return df
```

Chunked mode:

```python
def model(dbt, session):
    for batch in dbt.ref("stg_orders").iter_batches(batch_size=100_000):
        yield transform(batch)
```

## Runtime configs

Set model-level configs via `dbt.config(...)` in your python model:

- `localpy_dataframe_backend`: `pandas` (default) or `polars`
- `localpy_max_rows`: hard limit before failure (default `1_000_000`)
- `localpy_warn_rows`: warning threshold (default `200_000`)
- `localpy_allow_large_tables`: bypass hard row limit (default `false`)
- `localpy_chunked_mode`: allow oversized input only when using `iter_batches` (default `false`)
- `localpy_batch_size`: default batch size for `iter_batches` (default `100_000`)

## Honest limitations

- Not Snowpark
- Not Spark
- Python runs on local machine / CI runner
- Not intended for huge tables
- Best for small/medium transforms

## First milestone command

```bash
dbt run -s customer_features
```
