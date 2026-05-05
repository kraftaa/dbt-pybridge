# dbt-pybridge

Run dbt Python models on Postgres.

`dbt-pybridge` executes Python locally or in CI, reads inputs from Postgres via `dbt.ref()` / `dbt.source()`, then writes results back to Postgres.

It works by:
- compiling `.py` models through dbt
- executing Python locally (developer laptop or CI runner)
- loading `dbt.ref()` / `dbt.source()` data into pandas/polars
- writing the returned dataframe back into Postgres

This is useful when you want Python transforms in dbt without a warehouse-native Python runtime (for example Spark/Snowpark).

## Status

MVP scope for Python table + incremental + view materializations is implemented.

- Supported: `materialized='table'`
- Supported: `materialized='incremental'` (strategies: `append`, `merge`, `delete+insert`)
- Supported: `materialized='view'` (implemented as a managed backing table + SQL view)
- Supported DAG: `sql -> python -> sql`
- Supported return types: pandas DataFrame, polars DataFrame, or iterable/generator of dataframes

## Install

```bash
pip install dbt-pybridge
```

Requires:
- Python `>=3.11` (3.11/3.12 recommended)
- Postgres `>=10` (CASCADE drops, `force_null` with column lists, partitioned-table relkind support)
- `dbt-core >=1.10,<1.12`

Optional extras:
- Machine-learning examples (`customers_kmeans`, `orders_anomaly_isoforest`, `products_similarity_tfidf`) require `scikit-learn`:

```bash
pip install "dbt-pybridge[examples]"
```

## Trust model

`dbt-pybridge` runs your Python model code **in the same process** as the
adapter via `exec()`. The model has full access to the database connection
(read AND write), the local filesystem, environment variables, and any
network the adapter can reach.

Practical implications:

- Treat `.py` models the same as any other application code — only run models
  from sources you trust.
- The `.select(...)` / `.where(...)` / `.join(..., on=...)` helpers accept raw
  SQL fragments. The lightweight `;` and `"` guards are there to catch
  accidental typos, **not** to sandbox untrusted SQL. If you're in a context
  where authors of the python models can't be trusted, dbt-pybridge isn't the
  right tool.
- Your DB role's grants are the only real boundary. Run the dbt user with the
  least privileges needed for the models in scope.

## Docs

- [Scaling Python models](docs/scaling_python_models.md) — SQL vs Python decision rules, chunking/incremental patterns, and debugging checklist.

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

Safer projection pattern:

```python
def model(dbt, session):
    df = dbt.ref("stg_orders").select("order_id, amount, customer_id")
    return df
```

## SQL pushdown helpers

`dbt.ref(...)` returns a `RelationFrame` that you can chain SQL clauses onto
*before* the data is pulled into Python. Each helper wraps the previous SQL in
a subquery, so they compose. Use them when you only need a slice of upstream:

```python
def model(dbt, session):
    # Column projection — pushed into Postgres
    cols = dbt.ref("stg_orders").select("order_id, amount, customer_id")

    # Row filter — pushed into Postgres
    big = dbt.ref("stg_orders").where("amount > 100")

    # Join two refs in Postgres before loading
    enriched = (
        dbt.ref("stg_orders")
           .join(dbt.ref("stg_customers"), on="customer_id", how="left")
    )
    # Multi-key joins:
    pairs = dbt.ref("a").join(dbt.ref("b"), on=["customer_id", "site_id"])
    # Cross join:
    cartesian = dbt.ref("a").join(dbt.ref("b"), on=None, how="cross")

    return enriched.as_dataframe()
```

Notes:
- All three reject `;` and (for joins) `"` to keep accidental SQL injection
  out of model code; user code is already trusted (it runs via `exec`), but
  these guards catch typos.
- `.join()` uses Postgres' `USING (col, ...)` so the join column appears once
  in the output (matches pandas/polars merge semantics).
- The full DataFrame still loads into Python on `as_dataframe()`/`iter_batches`;
  the pushdowns just shrink what's pulled.

## Schema evolution for incremental models

If your incremental model adds/removes columns over time, set
`on_schema_change` so dbt-pybridge reconciles the target table automatically:

```python
def model(dbt, session):
    dbt.config(
        materialized="incremental",
        unique_key="order_id",
        on_schema_change="append_new_columns",  # default is "ignore"
    )
    ...
```

Supported values:

| value | behavior |
|---|---|
| `ignore` (default) | fail on column mismatch (preserves existing target schema) |
| `fail` | explicit fail with details on what drifted |
| `append_new_columns` | `ALTER TABLE ... ADD COLUMN` for any new dataframe column |
| `sync_all_columns` | append new columns AND drop columns missing from the model |

When `ignore` raises, the error message lists exactly which columns are
missing or new and points you at `append_new_columns` or `--full-refresh`.

By default `sync_all_columns` drops columns with plain `ALTER TABLE ... DROP
COLUMN` (PostgreSQL's default `RESTRICT` behavior). If a removed column has
dependents (views, materialized views, indexes, generated columns), Postgres
will refuse the drop and you'll get a clear error. To resolve, either:

- drop the dependents yourself and rerun,
- run `dbt run --full-refresh` for a clean rebuild, or
- opt into cascading drops with `pybridge_sync_drop_cascade=True`:

```python
dbt.config(
    materialized="incremental",
    on_schema_change="sync_all_columns",
    pybridge_sync_drop_cascade=True,  # ⚠️ silently drops dependents
)
```

> **⚠️ `pybridge_sync_drop_cascade=True` is destructive.** It uses
> `DROP COLUMN ... CASCADE`, which silently removes any dependent objects —
> including ones not managed by dbt. Only enable it when you're sure that's
> what you want.

## How to create Python models

1. Create `models/<name>_python.py`.
2. Define exactly one callable entrypoint: `def model(dbt, session): ...`.
3. Set materialization inside the function:
   - `dbt.config(materialized="table")`
4. Read upstream inputs using standalone ref/source assignments (important for dbt parser):
   - `orders = dbt.ref("stg_orders")`
   - `raw_orders = dbt.source("raw", "orders")`
5. Return one of:
   - pandas DataFrame
   - polars DataFrame
   - iterable/generator that yields pandas/polars DataFrames

Parser-safe pattern:

```python
def model(dbt, session):
    dbt.config(materialized="table")
    orders = dbt.ref("stg_orders")
    result = orders.copy()
    result["double_amount"] = result["amount"] * 2
    return result
```

Chunked mode:

```python
def model(dbt, session):
    for batch in dbt.ref("stg_orders").iter_batches(batch_size=100_000):
        yield transform(batch)
```

Runtime logging includes progress messages such as:

- `[pybridge] Loading "transform"."stg_orders" (2,300,000 rows, 120.0 MB)`
- `[pybridge] Processing batch 1, rows=100000`
- `[pybridge] Writing batch 1, rows=100000`

## Runtime configs

Set model-level configs via `dbt.config(...)` in your python model:

- `pybridge_dataframe_backend`: `pandas` (default) or `polars`
- `pybridge_max_rows`: hard limit before failure (default `1_000_000`)
- `pybridge_warn_rows`: warning threshold (default `200_000`)
- `pybridge_max_bytes`: hard estimated table-size limit before failure (default `536870912`, 512MB)
- `pybridge_warn_bytes`: warning estimated table-size threshold (default `134217728`, 128MB)
- `pybridge_allow_large_tables`: bypass hard row limit (default `false`)
- `pybridge_chunked_mode`: allow oversized input only when using `iter_batches` (default `false`)
- `pybridge_batch_size`: default batch size for `iter_batches` (default `100_000`)
- `pybridge_column_types`: optional explicit type map for created target tables, for example:
  - `{"id": "numeric(18,0)", "created_at": "timestamp", "payload": "jsonb"}`
- `pybridge_categorical_types`: optional categorical-column enum type map, for example:
  - `{"status": "status_enum", "tier": "tier_enum"}`
- `pybridge_sync_drop_cascade`: when `on_schema_change='sync_all_columns'`, controls
  whether dropped columns use `DROP COLUMN ... CASCADE` (default `false` — drops
  fail if dependents exist; opt in to silently drop dependents)

Legacy `localpy_*` keys are still accepted for backward compatibility.

## Type inference details

Default inferred target types now include:

- Numeric widths:
  - `smallint` / `integer` / `bigint` / `numeric` (for wide unsigned integers)
  - `real` / `double precision`
- Temporal:
  - `date`, `time`, `timetz`, `timestamp`, `timestamptz`, `interval`
- Structured / special:
  - `uuid`, `bytea`, `jsonb`
- Arrays (homogeneous scalar list/tuple object columns):
  - `boolean[]`, `bigint[]`, `double precision[]`, `text[]`, `uuid[]`, `date[]`, `time[]`, `timetz[]`, `timestamp[]`, `timestamptz[]`, `numeric[]`
  - mixed or nested list structures fall back to `jsonb`

Notes:

- `Decimal` object columns infer `numeric(precision,scale)` from sampled values.
- Empty or ambiguous object columns fall back to `text` (or `jsonb` for ambiguous list structures).
- You can always override with `pybridge_column_types`.

## Honest limitations

- Not Snowpark
- Not Spark
- Python runs on local machine / CI runner
- Not intended for huge tables
- Best for small/medium transforms
- Not a replacement for warehouse-scale computation
- For large tables, use filtering, incremental models, or chunked execution

## First milestone command

```bash
dbt run -s customer_features
```

## More examples

The `examples/mvp_project/` directory has runnable models for each major
feature:

- `customer_features.py` — minimal pandas table model
- `orders_polars.py` — polars backend (`pybridge_dataframe_backend='polars'`)
- `daily_revenue_incremental.py` — incremental + `merge` strategy with
  `unique_key`
- `orders_chunked_incremental.py` — chunked incremental pattern for bounded
  memory usage
- `orders_with_jsonb.py` — `pybridge_column_types` overrides for `jsonb`,
  `text[]`, and `numeric(18,4)`

```bash
cd examples/mvp_project
dbt run -s orders_polars
dbt run -s daily_revenue_incremental
dbt run -s daily_revenue_incremental                # second run exercises merge
dbt run -s daily_revenue_incremental --full-refresh # rebuild from scratch
dbt run -s orders_chunked_incremental
dbt run -s orders_with_jsonb
```
