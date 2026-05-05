"""Find similar products via TF-IDF + cosine similarity.

Why this can't be SQL: Postgres has `pg_trgm` for trigram similarity (fine
for typo-correction on short strings), but no native TF-IDF over a corpus.
sklearn does the whole thing — vectorize, score, rank — in a few lines.

Output: one row per (source_product_id, similar_product_id, rank), with a
similarity score in [0, 1]. Useful for "people who looked at X also liked..."
recommendations, dedup detection, or related-item carousels.

Run:
    dbt run -s products_similarity_tfidf

Inspect:
    SELECT * FROM products_similarity_tfidf WHERE rank = 1 ORDER BY similarity DESC LIMIT 10;
"""
import numpy as np
import pandas as pd


_PRODUCTS = [
    ("Wireless Bluetooth Headphones",   "Over-ear wireless headphones with noise cancelling and 30-hour battery."),
    ("Wireless Earbuds",                "True wireless earbuds with active noise cancellation and charging case."),
    ("Bluetooth Speaker",               "Portable bluetooth speaker, waterproof, 12-hour battery, deep bass."),
    ("Noise-Cancelling Headphones",     "Premium noise cancelling headphones with hi-res audio."),
    ("USB-C Charging Cable",            "Braided USB-C to USB-C cable for fast charging laptops and phones."),
    ("Lightning Cable",                 "Apple Lightning cable for iPhone and iPad, 6 ft, MFi certified."),
    ("Laptop Stand",                    "Aluminum laptop stand, ergonomic, ventilated, holds up to 17-inch laptop."),
    ("Standing Desk Converter",         "Adjustable height standing desk converter, dual monitor support."),
    ("Coffee Beans Dark Roast",         "Single-origin Colombian coffee beans, dark roast, 1lb bag."),
    ("Espresso Beans Italian Roast",    "Italian roast espresso beans, rich and bold, 1lb bag."),
    ("Green Tea Sencha",                "Japanese sencha green tea, loose leaf, 100g tin."),
    ("Yoga Mat",                        "Non-slip yoga mat, 6mm thick, eco-friendly TPE, includes carrying strap."),
    ("Foam Roller",                     "High-density foam roller for muscle recovery and physical therapy."),
    ("Resistance Bands Set",            "Set of 5 resistance bands, looped, for home gym workouts."),
    ("Mechanical Keyboard",             "RGB backlit mechanical keyboard with cherry MX brown switches."),
    ("Ergonomic Mouse",                 "Vertical ergonomic mouse, wireless, reduces wrist strain."),
]


def _top_n_similar(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    df = df.copy()
    df["text"] = (df["name"] + " " + df["description"]).str.lower()

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=0.95)
    matrix = vectorizer.fit_transform(df["text"])
    sims = cosine_similarity(matrix)
    np.fill_diagonal(sims, -1.0)  # never recommend a product as similar to itself

    rows = []
    ids = df["product_id"].to_numpy()
    names = df["name"].to_numpy()
    for i, src_id in enumerate(ids):
        top_idx = np.argsort(sims[i])[::-1][:n]
        for rank, j in enumerate(top_idx, start=1):
            score = float(sims[i, j])
            if score <= 0:
                continue
            rows.append({
                "product_id":         int(src_id),
                "similar_product_id": int(ids[j]),
                "rank":               rank,
                "similarity":         score,
                "product_name":       names[i],
                "similar_product_name": names[j],
            })
    return pd.DataFrame(rows)


def model(dbt, session):
    dbt.config(
        materialized="table",
        pybridge_column_types={
            "similarity": "double precision",
            "rank":       "smallint",
        },
    )
    products = pd.DataFrame(
        [(i + 1, name, desc) for i, (name, desc) in enumerate(_PRODUCTS)],
        columns=["product_id", "name", "description"],
    )
    return _top_n_similar(products, n=3)
