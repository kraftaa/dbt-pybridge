from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional
import warnings

import psycopg2


@dataclass(frozen=True)
class ModelLimits:
    max_rows: int = 1_000_000
    warn_rows: int = 200_000
    batch_size: int = 100_000
    allow_large_tables: bool = False
    chunked_mode: bool = False


@dataclass(frozen=True)
class TargetRelation:
    database: Optional[str]
    schema: Optional[str]
    identifier: str

    def render(self) -> str:
        parts = []
        if self.database:
            parts.append(quote_ident(self.database))
        if self.schema:
            parts.append(quote_ident(self.schema))
        parts.append(quote_ident(self.identifier))
        return ".".join(parts)


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


class LocalPostgresSession:
    def __init__(self, credentials, limits: ModelLimits, dataframe_backend: str = "pandas") -> None:
        self.credentials = credentials
        self.limits = limits
        self.dataframe_backend = dataframe_backend
        self.conn = psycopg2.connect(
            dbname=credentials.database,
            user=credentials.user,
            host=credentials.host,
            password=credentials.password,
            port=credentials.port,
            connect_timeout=getattr(credentials, "connect_timeout", 10),
        )
        self.conn.autocommit = False

    def close(self) -> None:
        self.conn.close()

    def _query_to_pandas(self, query: str):
        import pandas as pd

        with self.conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=columns)

    def _iter_query_to_pandas(self, query: str, chunk_size: int):
        import pandas as pd

        # Named cursor streams results from Postgres in batches instead of loading all rows at once.
        cursor_name = f"pybridge_batch_{id(self)}"
        with self.conn.cursor(name=cursor_name) as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break
                yield pd.DataFrame(rows, columns=columns)

    def count_rows(self, relation_sql: str) -> int:
        query = f"select count(*) from {relation_sql}"
        with self.conn.cursor() as cur:
            cur.execute(query)
            return int(cur.fetchone()[0])

    def enforce_size_limits(self, relation_sql: str, for_chunking: bool = False) -> int:
        row_count = self.count_rows(relation_sql)
        if row_count > self.limits.warn_rows:
            warnings.warn(
                (
                    f"Loading relation {relation_sql} with {row_count:,} rows. "
                    "This plugin is intended for small/medium transforms."
                ),
                stacklevel=2,
            )

        can_bypass_limit = self.limits.allow_large_tables or (self.limits.chunked_mode and for_chunking)
        if row_count > self.limits.max_rows and not can_bypass_limit:
            raise RuntimeError(
                (
                    f"Relation {relation_sql} has {row_count:,} rows, above limit {self.limits.max_rows:,}. "
                    "Set localpy_allow_large_tables=true or localpy_chunked_mode=true to opt in."
                )
            )
        return row_count

    def load_relation(self, relation_sql: str):
        import polars as pl

        self.enforce_size_limits(relation_sql, for_chunking=False)
        query = f"select * from {relation_sql}"
        df = self._query_to_pandas(query)
        if self.dataframe_backend == "polars":
            return pl.from_pandas(df)
        return df

    def iter_relation_batches(self, relation_sql: str, batch_size: Optional[int] = None) -> Iterator:
        import polars as pl

        self.enforce_size_limits(relation_sql, for_chunking=True)
        query = f"select * from {relation_sql}"
        chunk_size = int(batch_size or self.limits.batch_size)
        if chunk_size <= 0:
            raise RuntimeError(f"Batch size must be > 0, got {chunk_size}")

        for chunk in self._iter_query_to_pandas(query, chunk_size):
            if self.dataframe_backend == "polars":
                yield pl.from_pandas(chunk)
            else:
                yield chunk
