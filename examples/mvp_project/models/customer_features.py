def model(dbt, session):
    df = dbt.ref("stg_orders")
    df["double_amount"] = df["amount"] * 2
    return df
