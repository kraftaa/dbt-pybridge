"""Real Postgres + dbt checks for typed DuckDB staging, key filters, and merges."""

import os
from decimal import Decimal
from pathlib import Path
import subprocess
import sys
import textwrap

import psycopg2
from psycopg2 import sql
import pytest


INTEGRATION_HOST = os.getenv("PYBRIDGE_INTEGRATION_HOST")
PORT = int(os.getenv("PYBRIDGE_INTEGRATION_PORT", "5432"))
USER = os.getenv("PYBRIDGE_INTEGRATION_USER", "postgres")
PASSWORD = os.getenv("PYBRIDGE_INTEGRATION_PASSWORD", "postgres")
TARGET_DB = "pybridge_it_cons_target"
APP_DB = "pybridge_it_cons_app"
BILLING_DB = "pybridge_it_cons_billing"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not INTEGRATION_HOST, reason="synthetic Postgres integration service not configured"
    ),
]


def _connect(database):
    return psycopg2.connect(
        host=INTEGRATION_HOST, port=PORT, user=USER, password=PASSWORD, dbname=database
    )


@pytest.fixture
def databases():
    admin = _connect("postgres")
    admin.autocommit = True
    names = [TARGET_DB, APP_DB, BILLING_DB]
    with admin.cursor() as cur:
        for name in names:
            cur.execute(sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(name)))
            cur.execute(sql.SQL("create database {}").format(sql.Identifier(name)))
    for name, ddl in (
        (APP_DB, [
            "create table public.customers (customer_id bigint, customer_name text)",
            "insert into public.customers values (1, 'Ada'), (2, 'Lin'), (3, 'Mo')",
        ]),
        (BILLING_DB, [
            "create table public.orders (order_id bigint, customer_id bigint, amount numeric(12,2))",
            # Whole numbers first: inferring DuckDB types from the first batch
            # used to stage DECIMAL(2,0) and round 8.25 to 8.
            "insert into public.orders values (10, 1, 12), (11, 2, 8.25), (12, 3, 1.55)",
        ]),
        (TARGET_DB, [
            "create schema transform",
            "create table public.events (event_id bigint)",
            "insert into public.events values (1), (2)",
        ]),
    ):
        conn = _connect(name)
        with conn, conn.cursor() as cur:
            for statement in ddl:
                cur.execute(statement)
        conn.close()
    try:
        yield
    finally:
        with admin.cursor() as cur:
            for name in names:
                cur.execute(sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(name)))
        admin.close()


def _run_dbt(tmp_path, models):
    project = tmp_path / "project"
    model_dir = project / "models"
    profiles = tmp_path / "profiles"
    model_dir.mkdir(parents=True)
    profiles.mkdir()
    (project / "dbt_project.yml").write_text(textwrap.dedent("""
        name: pybridge_consistency
        version: 1.0
        config-version: 2
        profile: pybridge_consistency
        model-paths: ["models"]
    """).strip() + "\n")
    (model_dir / "sources.yml").write_text(textwrap.dedent(f"""
        version: 2
        sources:
          - name: app
            database: {APP_DB}
            schema: public
            config:
              meta:
                pybridge_connection: app_db
            tables:
              - name: customers
          - name: billing
            database: {BILLING_DB}
            schema: public
            config:
              meta:
                pybridge_connection: billing_db
            tables:
              - name: orders
          - name: local
            database: {TARGET_DB}
            schema: public
            tables:
              - name: events
    """).strip() + "\n")
    for name, body in models.items():
        (model_dir / f"{name}.py").write_text(textwrap.dedent(body).strip() + "\n")

    def connection(database):
        return textwrap.indent(textwrap.dedent(f"""
            host: {INTEGRATION_HOST}
            port: {PORT}
            user: {USER}
            password: {PASSWORD}
            dbname: {database}
        """).strip(), " " * 10)

    (profiles / "profiles.yml").write_text(
        textwrap.dedent(f"""
            pybridge_consistency:
              target: test
              outputs:
                test:
                  type: pybridge
                  host: {INTEGRATION_HOST}
                  port: {PORT}
                  user: {USER}
                  password: {PASSWORD}
                  dbname: {TARGET_DB}
                  schema: transform
                  threads: 1
                  pybridge_connections:
                    app_db:
        """).rstrip() + "\n" + connection(APP_DB) + "\n"
        + "        billing_db:\n" + connection(BILLING_DB) + "\n"
    )
    dbt_executable = Path(sys.executable).parent / "dbt"
    return subprocess.run(
        [
            str(dbt_executable), "run",
            "--project-dir", str(project), "--profiles-dir", str(profiles),
        ],
        text=True,
        capture_output=True,
    )


def _rows(database, query):
    conn = _connect(database)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
    finally:
        conn.close()


def test_federated_join_keeps_numeric_precision(databases, tmp_path):
    completed = _run_dbt(tmp_path, {
        "customer_orders": """
def model(dbt, session):
    dbt.config(materialized="table", pybridge_batch_size=1)
    customers = dbt.source("app", "customers").select("customer_id, customer_name")
    orders = dbt.source("billing", "orders").select("order_id, customer_id, amount")
    return customers.join(orders, on="customer_id", how="inner", engine="duckdb")
""",
    })
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

    assert _rows(
        TARGET_DB,
        "select order_id, customer_name, amount from transform.customer_orders order by order_id",
    ) == [
        (10, "Ada", Decimal("12.00")),
        (11, "Lin", Decimal("8.25")),
        (12, "Mo", Decimal("1.55")),
    ]


def test_incremental_merge_rejects_duplicate_unique_keys(databases, tmp_path):
    model = """
def model(dbt, session):
    import pandas as pd

    dbt.config(materialized="incremental", unique_key="id", incremental_strategy="merge")
    if dbt.is_incremental:
        return pd.DataFrame({"id": [1, 1], "v": ["a", "b"]})
    return pd.DataFrame({"id": [1], "v": ["seed"]})
"""
    first = _run_dbt(tmp_path / "first", {"keyed": model})
    assert first.returncode == 0, first.stdout + "\n" + first.stderr

    second = _run_dbt(tmp_path / "second", {"keyed": model})
    assert second.returncode != 0
    assert "rows for unique_key" in second.stdout
    assert _rows(TARGET_DB, "select id, v from transform.keyed") == [(1, "seed")]


def test_duckdb_join_filter_by_fetches_only_matching_keys(databases, tmp_path):
    completed = _run_dbt(tmp_path, {
        "first_customer_orders": """
def model(dbt, session):
    dbt.config(materialized="table")
    customers = dbt.source("app", "customers").where("customer_id = 1").select(
        "customer_id, customer_name"
    )
    orders = dbt.source("billing", "orders").select("order_id, customer_id, amount")
    return customers.join(
        orders, on="customer_id", how="left", engine="duckdb", filter_by="left"
    )
""",
    })
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    # Only order 10 matches; the billing query was filtered to customer 1, so
    # the other orders were never loaded.
    assert "pybridge_key_filter" in completed.stdout
    assert "Loaded" in completed.stdout
    billing_loads = [
        line for line in completed.stdout.splitlines()
        if "[billing_db]" in line and "Loaded" in line
    ]
    assert billing_loads and all("(1 rows" in line for line in billing_loads), billing_loads
    assert _rows(
        TARGET_DB,
        "select order_id, customer_name, amount from transform.first_customer_orders",
    ) == [(10, "Ada", Decimal("12.00"))]


def test_first_incremental_run_rejects_duplicate_unique_keys(databases, tmp_path):
    completed = _run_dbt(tmp_path, {
        "keyed_first": """
def model(dbt, session):
    import pandas as pd

    dbt.config(materialized="incremental", unique_key="id", incremental_strategy="merge")
    return pd.DataFrame({"id": [1, 1], "v": ["a", "b"]})
""",
    })
    assert completed.returncode != 0
    assert "rows for unique_key" in completed.stdout
    assert _rows(TARGET_DB, "select to_regclass('transform.keyed_first')") == [(None,)]
