def model(dbt, session):
    """Incremental model with merge strategy.

    First `dbt run`: creates the table from the full upstream output.
    Subsequent runs: rows whose `order_id` already exists in the target are
    updated; new rows are inserted (this is the `merge` strategy).

    Try it:
        dbt run -s daily_revenue_incremental
        dbt run -s daily_revenue_incremental                       # incremental
        dbt run -s daily_revenue_incremental --full-refresh        # rebuild
    """
    dbt.config(
        materialized="incremental",
        unique_key="order_id",
        incremental_strategy="merge",
    )

    import pandas as pd

    orders = dbt.ref("stg_orders").as_dataframe()
    out = orders.copy()
    out["revenue"] = out["amount"] * 1.0
    out["loaded_at"] = pd.Timestamp.utcnow()
    return out
