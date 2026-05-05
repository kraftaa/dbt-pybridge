"""Cluster customers by behavior using KMeans, store centroids as jsonb.

Why this can't be SQL: KMeans (and clustering generally) is not part of
stock Postgres. You can compute group statistics with GROUP BY, but you
can't *learn* groups from the data — sklearn does that in two lines.

Output is one row per customer with a cluster id and a `centroid` jsonb
column carrying the cluster's per-feature mean. The jsonb is queryable
later for ad-hoc "what does cluster 2 look like" questions.

Run:
    dbt run -s customers_kmeans

Inspect:
    SELECT cluster_id, count(*), avg(orders_last_90d), avg(avg_order_value)
    FROM customers_kmeans
    GROUP BY cluster_id ORDER BY cluster_id;
"""
import numpy as np
import pandas as pd


def _synthesize_customers(n: int = 1000) -> pd.DataFrame:
    """Three latent customer types — light/casual/power — with noise."""
    rng = np.random.default_rng(seed=7)
    type_idx = rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])

    profiles = np.array([
        # orders_last_90d, avg_order_value, days_since_signup
        [   2.0,  35.0,  400.0],   # light
        [  12.0, 120.0,  250.0],   # casual
        [  45.0, 250.0,  120.0],   # power
    ])
    means = profiles[type_idx]
    noise_scale = np.array([2.0, 30.0, 60.0])
    features = means + rng.normal(scale=noise_scale, size=means.shape)
    features = np.clip(features, a_min=0, a_max=None)

    return pd.DataFrame({
        "customer_id":      np.arange(1, n + 1),
        "orders_last_90d":  features[:, 0],
        "avg_order_value":  features[:, 1].round(2),
        "days_since_signup": features[:, 2].astype(int),
    })


def _cluster(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    feature_cols = ["orders_last_90d", "avg_order_value", "days_since_signup"]
    X = df[feature_cols].to_numpy(dtype=float)

    # Standardize so features with bigger natural ranges (e.g. days) don't
    # dominate the distance calculation.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)

    # Map centroids back to the original feature scale for human readability.
    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    out = df.copy()
    out["cluster_id"] = labels.astype(int)
    out["centroid"] = [
        {feat: float(round(val, 2)) for feat, val in zip(feature_cols, centroids_original[label])}
        for label in labels
    ]
    return out


def model(dbt, session):
    dbt.config(
        materialized="table",
        pybridge_column_types={
            "cluster_id":        "smallint",
            "centroid":          "jsonb",
            "orders_last_90d":   "double precision",
            "avg_order_value":   "numeric(12,2)",
            "days_since_signup": "integer",
        },
    )
    return _cluster(_synthesize_customers(n=1000), k=3)
