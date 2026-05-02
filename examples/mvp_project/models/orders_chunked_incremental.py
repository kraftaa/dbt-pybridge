def model(dbt, session):
    """Chunked incremental model for bounded memory usage.

    This pattern is for larger inputs where loading everything at once is
    unnecessary. It keeps memory bounded by yielding one processed batch at a
    time.
    """
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
