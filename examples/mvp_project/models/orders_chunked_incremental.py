def _process(src):
    # `yield` lives in a helper, not in `model()` itself. dbt's parser rejects
    # a `model()` that is itself a generator function ("model function should
    # return only one dataframe object"); having `model()` *return* a generator
    # works fine.
    for batch in src.iter_batches():
        out = batch.copy()
        out["amount_bucket"] = out["amount"].apply(
            lambda value: "high" if value >= 100 else "standard"
        )
        yield out


def model(dbt, session):
    """Chunked incremental model for bounded memory usage.

    This pattern is for larger inputs where loading everything at once is
    unnecessary — the helper yields one processed batch at a time and
    dbt-pybridge streams each batch into Postgres via COPY before the next
    batch is materialized.
    """
    dbt.config(
        materialized="incremental",
        incremental_strategy="append",
        pybridge_chunked_mode=True,
        pybridge_batch_size=100_000,
    )

    src = dbt.ref("stg_orders").select("order_id, amount")
    return _process(src)
