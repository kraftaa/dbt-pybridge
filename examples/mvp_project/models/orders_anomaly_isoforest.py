"""Multi-feature anomaly detection with IsolationForest.

Why this can't be SQL: Postgres has no equivalent — there's no learned
joint-density estimator in stock SQL. A row whose `amount`, `hour_of_day`,
and `day_of_week` are each individually normal but jointly improbable
(e.g. a large amount at 3 AM on Sunday) is exactly what Isolation Forest
catches and what window-function-based outlier rules cannot.

Synthetic data is generated inside the model so this example is fully
self-contained — no upstream staging table needed.

Run:
    dbt run -s orders_anomaly_isoforest

Inspect:
    SELECT * FROM orders_anomaly_isoforest WHERE is_anomaly ORDER BY anomaly_score LIMIT 20;
"""
import numpy as np
import pandas as pd


def _synthesize(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)

    # Normal traffic: lognormal amounts, business-hour-skewed timestamps
    base_ts = pd.Timestamp("2026-01-01")
    minutes_per_row = rng.integers(1, 60, n)
    timestamps = base_ts + pd.to_timedelta(np.cumsum(minutes_per_row), unit="min")
    amounts = np.exp(rng.normal(loc=4.5, scale=0.7, size=n))

    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "amount": amounts.round(2),
        "ordered_at": timestamps,
    })

    # Plant 5 honest outliers in non-boundary positions so the detector has
    # something real to find (rather than just edge effects on synthetic data).
    plants = pd.DataFrame({
        "order_id": [-1, -2, -3, -4, -5],
        "amount":   [50_000, 49_500, 75_000, 60_000, 55_000],
        "ordered_at": pd.to_datetime([
            "2026-02-15 03:14:00",  # 3 AM on a Sunday
            "2026-02-22 04:47:00",  # 4 AM on a Sunday
            "2026-03-08 02:30:00",  # 2 AM on a Sunday
            "2026-03-15 23:55:00",  # near midnight
            "2026-04-05 04:00:00",  # 4 AM on a Sunday
        ]),
    })
    return pd.concat([df, plants], ignore_index=True)


def _detect(df: pd.DataFrame, contamination: float = 0.005) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest

    df = df.copy()
    df["hour_of_day"] = df["ordered_at"].dt.hour.astype("int8")
    df["day_of_week"] = df["ordered_at"].dt.dayofweek.astype("int8")

    features = df[["amount", "hour_of_day", "day_of_week"]].to_numpy()
    forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(features)
    df["anomaly_score"] = forest.decision_function(features)
    df["is_anomaly"] = forest.predict(features) == -1

    return df[[
        "order_id", "amount", "ordered_at",
        "hour_of_day", "day_of_week",
        "anomaly_score", "is_anomaly",
    ]]


def model(dbt, session):
    dbt.config(
        materialized="table",
        pybridge_column_types={
            "anomaly_score": "double precision",
            "hour_of_day":   "smallint",
            "day_of_week":   "smallint",
        },
    )
    return _detect(_synthesize(n=5000), contamination=0.005)
