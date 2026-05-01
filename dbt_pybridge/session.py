from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional
import uuid
import warnings

import psycopg2


@dataclass(frozen=True)
class ModelLimits:
    max_rows: int = 1_000_000
    warn_rows: int = 200_000
    max_bytes: int = 512 * 1024 * 1024
    warn_bytes: int = 128 * 1024 * 1024
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

    @staticmethod
    def _human_bytes(value: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(value)
        unit = units[0]
        for unit in units:
            if size < 1024 or unit == units[-1]:
                break
            size /= 1024.0
        if unit == "B":
            return f"{int(size)} {unit}"
        return f"{size:.1f} {unit}"

    @staticmethod
    def _split_relation_parts(relation_sql: str):
        text = relation_sql.strip()
        if not text:
            return None

        parts = []
        current = []
        in_quote = False
        i = 0
        while i < len(text):
            ch = text[i]
            if in_quote:
                current.append(ch)
                if ch == '"':
                    if i + 1 < len(text) and text[i + 1] == '"':
                        current.append(text[i + 1])
                        i += 1
                    else:
                        in_quote = False
            else:
                if ch == '"':
                    in_quote = True
                    current.append(ch)
                elif ch == ".":
                    part = "".join(current).strip()
                    if not part:
                        return None
                    parts.append(part)
                    current = []
                elif ch.isspace():
                    return None
                else:
                    current.append(ch)
            i += 1

        if in_quote:
            return None

        tail = "".join(current).strip()
        if not tail:
            return None
        parts.append(tail)
        return parts

    @staticmethod
    def _unquote_identifier(part: str) -> str:
        token = part.strip()
        if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
            return token[1:-1].replace('""', '"')
        return token

    def _normalize_relation_sql(self, relation_sql: str) -> str:
        parts = self._split_relation_parts(relation_sql)
        if not parts:
            return relation_sql
        if len(parts) != 3:
            return relation_sql

        db_part = self._unquote_identifier(parts[0])
        current_db = getattr(self.credentials, "database", None)
        if current_db and db_part.lower() != str(current_db).lower():
            raise RuntimeError(
                "dbt-pybridge does not support cross-database Postgres reads in Python models. "
                f"Got relation {relation_sql} while current database is '{current_db}'."
            )
        # Postgres cannot query cross-database via 3-part names; drop database qualifier.
        return ".".join(parts[1:])

    def _query_columns(self, query: str):
        with self.conn.cursor() as cur:
            cur.execute(f"{query} limit 0")
            if cur.description:
                return [desc[0] for desc in cur.description]
            return []

    def _query_to_pandas(self, query: str):
        import pandas as pd

        with self.conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            if cur.description:
                columns = [desc[0] for desc in cur.description]
            elif rows:
                columns = self._query_columns(query) or [f"column_{i+1}" for i in range(len(rows[0]))]
            else:
                columns = []
        return pd.DataFrame(rows, columns=columns)

    def _iter_query_to_pandas(self, query: str, chunk_size: int):
        import pandas as pd

        # Named cursor streams results from Postgres in batches instead of loading all rows at once.
        cursor_name = f"pybridge_batch_{uuid.uuid4().hex[:12]}"
        fallback_columns = self._query_columns(query)
        with self.conn.cursor(name=cursor_name) as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description] if cur.description else None
            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break
                if columns is None:
                    columns = fallback_columns or [f"column_{i+1}" for i in range(len(rows[0]))]
                yield pd.DataFrame(rows, columns=columns)

    def count_rows(self, relation_sql: str) -> int:
        relation_sql = self._normalize_relation_sql(relation_sql)
        query = f"select count(*) from {relation_sql}"
        with self.conn.cursor() as cur:
            cur.execute(query)
            return int(cur.fetchone()[0])

    def count_bytes(self, relation_sql: str) -> Optional[int]:
        relation_sql = self._normalize_relation_sql(relation_sql)
        with self.conn.cursor() as cur:
            cur.execute(
                "select pg_total_relation_size(to_regclass(%s))",
                (relation_sql,),
            )
            row = cur.fetchone()
            if not row:
                return None
            size = row[0]
            return int(size) if size is not None else None

    def enforce_size_limits(self, relation_sql: str, for_chunking: bool = False) -> int:
        relation_sql = self._normalize_relation_sql(relation_sql)
        row_count = self.count_rows(relation_sql)
        byte_count = self.count_bytes(relation_sql)
        if row_count > self.limits.warn_rows:
            warnings.warn(
                (
                    f"Loading relation {relation_sql} with {row_count:,} rows. "
                    "This plugin is intended for small/medium transforms."
                ),
                stacklevel=2,
            )
        if byte_count is not None and byte_count > self.limits.warn_bytes:
            warnings.warn(
                (
                    f"Loading relation {relation_sql} with estimated size {self._human_bytes(byte_count)}. "
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
        if byte_count is not None and byte_count > self.limits.max_bytes and not can_bypass_limit:
            raise RuntimeError(
                (
                    f"Relation {relation_sql} estimated size is {self._human_bytes(byte_count)}, "
                    f"above limit {self._human_bytes(self.limits.max_bytes)}. "
                    "Set localpy_allow_large_tables=true or localpy_chunked_mode=true to opt in."
                )
            )
        return row_count

    def load_relation(self, relation_sql: str):
        import polars as pl

        relation_sql = self._normalize_relation_sql(relation_sql)
        self.enforce_size_limits(relation_sql, for_chunking=False)
        query = f"select * from {relation_sql}"
        df = self._query_to_pandas(query)
        if self.dataframe_backend == "polars":
            return pl.from_pandas(df)
        return df

    def iter_relation_batches(self, relation_sql: str, batch_size: Optional[int] = None) -> Iterator:
        import polars as pl

        relation_sql = self._normalize_relation_sql(relation_sql)
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
