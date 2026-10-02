import os
from pathlib import Path
from decimal import Decimal
import subprocess
import sys
import textwrap

import psycopg2
from psycopg2 import sql
import pytest


INTEGRATION_HOST = os.getenv("PYBRIDGE_INTEGRATION_HOST")
pytestmark = pytest.mark.integration


def _connect(database):
    return psycopg2.connect(
        host=INTEGRATION_HOST,
        port=int(os.getenv("PYBRIDGE_INTEGRATION_PORT", "5432")),
        user=os.getenv("PYBRIDGE_INTEGRATION_USER", "postgres"),
        password=os.getenv("PYBRIDGE_INTEGRATION_PASSWORD", "postgres"),
        dbname=database,
    )


@pytest.mark.skipif(not INTEGRATION_HOST, reason="synthetic Postgres integration service not configured")
def test_real_dbt_run_reads_two_databases_and_writes_target(tmp_path):
    databases = ["pybridge_it_target", "pybridge_it_app", "pybridge_it_billing"]
    admin = _connect("postgres")
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            for database in databases:
                cur.execute(sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(database)))
                cur.execute(sql.SQL("create database {}").format(sql.Identifier(database)))

        app = _connect("pybridge_it_app")
        billing = _connect("pybridge_it_billing")
        try:
            with app, app.cursor() as cur:
                cur.execute("create table public.customers (customer_id bigint, customer_name text)")
                cur.execute("insert into public.customers values (1, 'Ada'), (2, 'Lin')")
            with billing, billing.cursor() as cur:
                cur.execute("create table public.orders (order_id bigint, customer_id bigint, amount numeric)")
                cur.execute("insert into public.orders values (10, 1, 12.50), (11, 2, 8.25)")
        finally:
            app.close()
            billing.close()

        project = tmp_path / "project"
        models = project / "models"
        profiles = tmp_path / "profiles"
        models.mkdir(parents=True)
        profiles.mkdir()

        (project / "dbt_project.yml").write_text(textwrap.dedent("""
            name: pybridge_integration
            version: 1.0
            config-version: 2
            profile: pybridge_integration
            model-paths: ["models"]
        """).strip() + "\n")
        (models / "sources.yml").write_text(textwrap.dedent("""
            version: 2
            sources:
              - name: app
                database: pybridge_it_app
                schema: public
                config:
                  meta:
                    pybridge_connection: app_db
                tables:
                  - name: customers
              - name: billing
                database: pybridge_it_billing
                schema: public
                config:
                  meta:
                    pybridge_connection: billing_db
                tables:
                  - name: orders
        """).strip() + "\n")
        (models / "customer_orders.py").write_text(textwrap.dedent("""
            def model(dbt, session):
                dbt.config(materialized="table")
                customers = dbt.source("app", "customers").select(
                    "customer_id, customer_name"
                )
                orders = dbt.source("billing", "orders").select(
                    "order_id, customer_id, amount"
                )
                return customers.join(
                    orders, on="customer_id", how="inner", engine="duckdb"
                )
        """).strip() + "\n")
        (profiles / "profiles.yml").write_text(textwrap.dedent(f"""
            pybridge_integration:
              target: test
              outputs:
                test:
                  type: pybridge
                  host: {INTEGRATION_HOST}
                  port: {int(os.getenv('PYBRIDGE_INTEGRATION_PORT', '5432'))}
                  user: {os.getenv('PYBRIDGE_INTEGRATION_USER', 'postgres')}
                  password: {os.getenv('PYBRIDGE_INTEGRATION_PASSWORD', 'postgres')}
                  dbname: pybridge_it_target
                  schema: transform
                  threads: 1
                  pybridge_connections:
                    app_db:
                      host: {INTEGRATION_HOST}
                      port: {int(os.getenv('PYBRIDGE_INTEGRATION_PORT', '5432'))}
                      user: {os.getenv('PYBRIDGE_INTEGRATION_USER', 'postgres')}
                      password: {os.getenv('PYBRIDGE_INTEGRATION_PASSWORD', 'postgres')}
                      dbname: pybridge_it_app
                    billing_db:
                      host: {INTEGRATION_HOST}
                      port: {int(os.getenv('PYBRIDGE_INTEGRATION_PORT', '5432'))}
                      user: {os.getenv('PYBRIDGE_INTEGRATION_USER', 'postgres')}
                      password: {os.getenv('PYBRIDGE_INTEGRATION_PASSWORD', 'postgres')}
                      dbname: pybridge_it_billing
        """).strip() + "\n")

        dbt_executable = Path(sys.executable).parent / "dbt"
        completed = subprocess.run(
            [
                str(dbt_executable), "run", "--select", "customer_orders",
                "--project-dir", str(project), "--profiles-dir", str(profiles),
            ],
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

        target = _connect("pybridge_it_target")
        try:
            with target.cursor() as cur:
                cur.execute(
                    "select order_id, customer_name, amount "
                    "from transform.customer_orders order by order_id"
                )
                assert cur.fetchall() == [
                    (10, "Ada", Decimal("12.5")),
                    (11, "Lin", Decimal("8.25")),
                ]
        finally:
            target.close()
    finally:
        with admin.cursor() as cur:
            for database in databases:
                cur.execute(sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(database)))
        admin.close()
