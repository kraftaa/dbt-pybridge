def model(dbt, session):
    """Same transform as customer_features.py, but using the polars backend.

    Setting `localpy_dataframe_backend='polars'` makes `dbt.ref(...)` return a
    polars DataFrame instead of pandas. The output is converted back to pandas
    internally before COPY into Postgres, so you can return either type.
    """
    dbt.config(materialized="table", localpy_dataframe_backend="polars")

    import polars as pl

    orders = dbt.ref("stg_orders").as_dataframe()
    return orders.with_columns(double_amount=pl.col("amount") * 2)
