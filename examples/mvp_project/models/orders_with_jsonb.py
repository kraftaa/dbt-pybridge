def model(dbt, session):
    """Demonstrates jsonb + array Postgres types via `localpy_column_types`.

    Without the override, dbt-pybridge would infer types from the dataframe:
        - `tags`        list[str]   -> text[]
        - `attributes`  dict        -> jsonb
    The override is useful when you want a specific Postgres type that the
    inference can't reach (for example, `numeric(18,4)` for amounts, or `jsonb`
    instead of `json`).

    Nulls in any of these columns are written safely — dbt-pybridge converts
    them to a uuid-suffixed null sentinel that Postgres COPY treats as NULL.
    """
    dbt.config(
        materialized="table",
        localpy_column_types={
            "amount":     "numeric(18,4)",
            "tags":       "text[]",
            "attributes": "jsonb",
        },
    )

    import pandas as pd

    orders = dbt.ref("stg_orders").as_dataframe().copy()
    orders["tags"]       = [["new", "vip"], ["repeat", None]]
    orders["attributes"] = [
        {"channel": "web",    "promo": "SPRING25"},
        {"channel": "mobile", "promo": None},
    ]
    return orders
