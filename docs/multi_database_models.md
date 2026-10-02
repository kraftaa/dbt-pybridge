# Multi-database Python models

PyBridge can read `source()` inputs from independent Postgres connections,
transform them in the local/CI Python runtime, and write the result to the
active dbt target. `ref()` remains bound to the active dbt target so dbt's
build and read semantics stay aligned.

## Configure named connections

Keep credentials in `profiles.yml` and use environment variables for secrets.
All named connection settings are validated before model code runs; valid
connections are then opened lazily only when the model uses one of them. The
name `target` is reserved for the active dbt target.

```yaml
analytics_profile:
  target: dev
  outputs:
    dev:
      type: pybridge
      host: analytics.example.invalid
      user: transformer
      password: "{{ env_var('ANALYTICS_PASSWORD') }}"
      port: 5432
      dbname: analytics
      schema: transform
      threads: 1

      pybridge_connections:
        app_db:
          host: app.example.invalid
          user: reader
          password: "{{ env_var('APP_DB_PASSWORD') }}"
          port: 5432
          dbname: application
          sslmode: require

        billing_db:
          host: billing.example.invalid
          user: reader
          password: "{{ env_var('BILLING_DB_PASSWORD') }}"
          port: 5432
          dbname: billing
          sslmode: require
```

Supported named-connection settings are `host`, `user`, `password` (or
`pass`), `password_env`, `service`, `passfile`, `database` (or `dbname`),
`port`, `connect_timeout`, `search_path`, `keepalives_idle`, `sslmode`,
`sslcert`, `sslkey`, `sslrootcert`, and `application_name`. Configure only one
of `password`/`pass`, `password_env`, or `passfile`. A libpq `service` may
supply the host, user, database, and authentication settings. `password_env`
is resolved only when the named connection opens and is not retained on the
validated credential object.

## Route sources

Set the source's real Postgres database name and put the connection alias in
`config.meta.pybridge_connection`:

```yaml
version: 2

sources:
  - name: app
    database: application
    schema: public
    config:
      meta:
        pybridge_connection: app_db
    tables:
      - name: customers

  - name: billing
    database: billing
    schema: public
    config:
      meta:
        pybridge_connection: billing_db
    tables:
      - name: orders
```

The `database` value must match the `database`/`dbname` of its named
connection. This catches accidental routing to the wrong database before any
query is run.

Named routing is intentionally source-only. Setting `pybridge_connection` on a
model, seed, or snapshot and reading it with `ref()` raises an error. dbt builds
`ref()` dependencies through the active target, so allowing a separate read
route would make build and read semantics disagree. Build remote relations in
the appropriate dbt target/run and declare the resulting relation as a routed
source.

## Join data locally

Operations chained on one relation, and joins between relations on the same
connection, are pushed down to that Postgres database. Cross-connection joins
must be explicit:

```python
def model(dbt, session):
    dbt.config(
        materialized="table",
        pybridge_dataframe_backend="polars",
    )

    customers = (
        dbt.source("app", "customers")
        .select("customer_id, customer_name")
        .as_dataframe()
    )
    orders = (
        dbt.source("billing", "orders")
        .select("order_id, customer_id, amount")
        .as_dataframe()
    )

    return customers.join(orders, on="customer_id", how="left")
```

For pandas, use `customers.merge(orders, on="customer_id", how="left")`.

Calling `left_relation.join(right_relation, ...)` across two named
connections raises an error instead of silently downloading both tables.
Explicit loading makes the transfer visible and keeps the existing
`pybridge_max_rows` and `pybridge_max_bytes` guardrails in effect for each
input. PyBridge also enforces aggregate limits across every connection used by
the model:

```python
dbt.config(
    pybridge_max_total_rows=1_000_000,
    pybridge_warn_total_rows=200_000,
    pybridge_max_total_bytes=536_870_912,
    pybridge_warn_total_bytes=134_217_728,
)
```

For larger cross-connection joins, install the optional spill-backed engine:

```bash
pip install 'dbt-pybridge[federation]'
```

Then keep filters and projections on each source and request DuckDB explicitly.
Non-key column names must differ between the two sides; rename them in
`.select()`:

```python
def model(dbt, session):
    dbt.config(pybridge_chunked_mode=True)
    customers = dbt.source("app", "customers").select(
        "customer_id, customer_name"
    )
    orders = dbt.source("billing", "orders").select(
        "order_id, customer_id, amount"
    )
    return customers.join(
        orders,
        on="customer_id",
        how="left",
        engine="duckdb",
        memory_limit="512MB",
        threads=2,                        # optional; DuckDB defaults to all cores
        temp_dir="/mnt/scratch",          # optional; defaults to the system temp dir
        max_temp_directory_size="20GB",   # optional cap on spilled data
    )
```

Inputs are streamed in bounded batches into a temporary on-disk DuckDB
database, and DuckDB may spill join work to that directory. Temporary files are
removed after eager materialization or after the batch iterator closes. Use
`pybridge_chunked_mode=True` to opt into inputs above the normal eager limits.
An eagerly loaded join result is checked against `pybridge_max_rows` and
`pybridge_max_bytes`; return `joined.iter_batches()` when the joined result
itself may be large. Memory use is roughly `memory_limit` per running model, so
with dbt `threads: 4` budget four times that.

Staging tables are created from the Postgres column types, so values keep their
types across batches and `numeric` comes back as exact `Decimal`. Supported
types are booleans, integers, floats, `numeric`, text types, dates and times,
`interval`, `uuid`, `json`/`jsonb`, `bytea`, and integer/float/text arrays.
Cast anything else in `.select()` (for example `location::text`). Join keys
cannot be unconstrained `numeric` or JSON, which are staged as text.

When one side is small, pass `filter_by` to avoid downloading the whole other
side. `filter_by="left"` stages the left input first, then queries the right
source only for rows whose join key appears on the left (10,000 keys per
query, cast to the source column's type so its index can be used):

```python
    customers = dbt.source("app", "customers").where("region = 'EU'")
    orders = dbt.source("billing", "orders").select("order_id, customer_id, amount")
    return customers.join(
        orders, on="customer_id", how="left", engine="duckdb", filter_by="left"
    )
```

It needs a single join key and is accepted only where it cannot change the
result: `filter_by="left"` for inner and left joins, `filter_by="right"` for
inner and right joins.

The eager loader fetches at most `pybridge_max_rows + 1` rows before enforcing
the row limit, avoiding a separate `count(*)` scan and preventing changes
between a count query and the actual read from bypassing the guardrail.

## Align sources with a consistent cut

Separate servers cannot share one snapshot, and replicas or loaders may lag
each other. If each source has a column that only grows (`updated_at`,
`created_at`, or a sequence id), `consistent_cut` filters every input to the
same moment:

```python
from datetime import timedelta
from dbt_pybridge import consistent_cut


def model(dbt, session):
    customers = dbt.source("app", "customers")
    payments = dbt.source("billing", "payments")
    customers, payments = consistent_cut(
        (customers, "updated_at"),
        (payments, "paid_at"),
        lag=timedelta(minutes=5),
    )
    ...
```

Inside the snapshots pinned at model start, it takes each input's maximum,
uses the smallest as the watermark (minus `lag`), and keeps only rows at or
below it, so data newer than the slowest source is excluded everywhere. `lag`
covers transactions that stamp a value before they commit. The watermark is
logged. Rows that are updated or deleted in place, rather than appended, still
need change-data capture or replication into one database.

## Consistency and permissions

- The result is always materialized through the active target connection.
- Every named source uses a read-only `REPEATABLE READ` transaction, giving it
  a stable per-database snapshot for the duration of the model. Target reads
  also share one `REPEATABLE READ` snapshot by default; set
  `pybridge_target_isolation="read committed"` to opt out. Under repeatable
  read, an incremental merge into rows another session changes concurrently
  fails with a serialization error instead of overwriting them.
- Snapshots start at model start: PyBridge opens the target and every routed
  source the model declares and runs a first query on each back to back, then
  logs the measured gap (`Pinned source snapshots within N ms`). Separate
  servers still have independent transactions, so this is a few milliseconds
  of skew, not one atomic snapshot. For strict cross-system consistency, use
  `consistent_cut` (below).
- Use read-only, least-privileged roles for named source connections.
- Connection passwords are not embedded in compiled Python model code. Use
  `env_var()` in profiles so dbt can scrub secrets from its logs and artifacts.
- Load logs include the connection alias so source activity can be attributed
  without printing connection credentials.
