def model(dbt, session):
    """Same transform as customer_features.py, but using the polars backend.

    Setting `pybridge_dataframe_backend='polars'` makes `dbt.ref(...)` return a
    polars DataFrame instead of pandas. The output is converted back to pandas
    internally before COPY into Postgres, so you can return either type.

    The SQL-side `amount::double precision` cast is important for polars: stock
    polars can't ingest pandas dataframes that contain `decimal.Decimal` /
    nullable extension dtypes without `pyarrow` installed. Casting to a plain
    numpy-backed float column on the Postgres side keeps the example running
    on a minimal install.
    """
    dbt.config(materialized="table", pybridge_dataframe_backend="polars")

    import polars as pl

    orders = (
        dbt.ref("stg_orders")
           .select("order_id, amount::double precision as amount")
           .as_dataframe()
    )
    return orders.with_columns(double_amount=pl.col("amount") * 2)
