# Scaling Python Models in dbt-pybridge

This guide explains when to use Python vs SQL and how to make larger or more complex Python models work reliably.

## Python vs SQL

Use SQL when:
- work is mostly joins, aggregations, filters, or set operations on large tables
- the warehouse can execute the transform directly and efficiently

Use Python when:
- logic is awkward in SQL but straightforward in Python
- you need Python libraries or code patterns that are hard to express in SQL
- you can keep the data volume bounded with filtering, incremental, or chunked execution

Good Python candidates:
- JSON / array cleanup and normalization
- text processing, regex-heavy transforms, custom scoring
- row-wise business rules that become hard to maintain in SQL

## Mental Model

`dbt-pybridge` runs Python outside the database:
1. dbt compiles the DAG.
2. Python loads relations from Postgres (`ref` / `source`).
3. Python transforms in-memory dataframes.
4. Results are written back to Postgres.

This means performance and memory are bounded by the local machine / CI runner, not by warehouse-scale compute.

## Patterns for Difficult Models

### 1. Push down first

Reduce width/rows before Python:
- use upstream SQL staging models for heavy joins and coarse filtering
- project only needed columns with `.select(...)`

Example:

```python
def model(dbt, session):
    df = dbt.ref("stg_orders").select("order_id, customer_id, amount")
    # Python logic on narrower dataframe
    return df
```

### 2. Use incremental materialization

For recurring pipelines, avoid full rebuilds:
- `materialized="incremental"`
- pick strategy:
  - `append` for immutable event-style data
  - `merge` for upserts (requires `unique_key`)
  - `delete+insert` for replacement-by-key behavior

### 3. Use chunked mode for larger inputs

Avoid loading all rows at once:
- enable `pybridge_chunked_mode=True`
- set `pybridge_batch_size`
- return/yield batch outputs

Example:

```python
def model(dbt, session):
    dbt.config(
        materialized="incremental",
        incremental_strategy="append",
        pybridge_chunked_mode=True,
        pybridge_batch_size=100_000,
    )

    for batch in dbt.ref("stg_orders").select("order_id, amount").iter_batches():
        out = batch.copy()
        out["amount_bucket"] = out["amount"].apply(
            lambda value: "high" if value >= 100 else "standard"
        )
        yield out
```

### 4. Control output types

For stable DDL and fewer runtime surprises:
- use `pybridge_column_types` to pin Postgres column types
- use `pybridge_categorical_types` for categorical/enums

Example:

```python
dbt.config(
    materialized="table",
    pybridge_column_types={
        "id": "bigint",
        "payload": "jsonb",
        "amount": "numeric(18,4)",
    },
)
```

### 5. Keep model boundaries intentional

Recommended split:
- SQL model(s): heavy relational work
- Python model: specialized logic
- SQL model(s): downstream joins/serving marts

This keeps Python focused and prevents local runtime bottlenecks.

## Runtime Guardrails

Relevant configs:
- `pybridge_max_rows`, `pybridge_warn_rows`
- `pybridge_max_bytes`, `pybridge_warn_bytes`
- `pybridge_allow_large_tables`
- `pybridge_chunked_mode`
- `pybridge_batch_size`

Default behavior is intentionally conservative for local/CI execution.

## Debugging Checklist

If a difficult model fails:
1. Confirm selected columns exist (`.select(...)` and downstream code match).
2. Print/log dtypes before merge/join in pandas/polars.
3. Normalize join-key dtypes on both sides before merge.
4. Pin output types with `pybridge_column_types` if inference is ambiguous.
5. Reduce batch size if memory pressure is high.
6. Move very heavy relational steps back into SQL staging.

## Do We Need This?

Yes. Without these patterns, users often apply Python to warehouse-scale transformations and hit memory/performance limits. With these patterns, Python models are reliable for the target use case: small/medium, specialized transforms inside a normal dbt DAG.
