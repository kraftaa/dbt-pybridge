from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from contextlib import closing
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd

from dbt_pybridge.dataframe_io import write_model_result
from dbt_pybridge.session import (
    LocalPostgresSession,
    ModelLimits,
    PostgresSessionRegistry,
    TargetRelation,
    quote_ident,
)


class RelationFrame:
    """Lazy dataframe wrapper that supports both eager dataframe use and iter_batches()."""

    def __init__(self, session: LocalPostgresSession, relation_sql: str) -> None:
        self._session = session
        self._relation_sql = relation_sql
        self._df = None

    def _load(self):
        if self._df is None:
            self._df = self._session.load_relation(self._relation_sql)
        return self._df

    def iter_batches(self, batch_size: Optional[int] = None):
        return self._session.iter_relation_batches(self._relation_sql, batch_size=batch_size)

    def select(self, projection_sql: str):
        projection = str(projection_sql or "").strip()
        if not projection:
            raise RuntimeError("dbt.ref(...).select(...) requires a non-empty SQL projection string")
        if ";" in projection:
            raise RuntimeError("dbt.ref(...).select(...) does not allow semicolons")
        relation_sql = f"(select {projection} from {self._relation_sql}) as pybridge_select"
        return RelationFrame(self._session, relation_sql)

    def where(self, predicate_sql: str):
        predicate = str(predicate_sql or "").strip()
        if not predicate:
            raise RuntimeError("dbt.ref(...).where(...) requires a non-empty SQL predicate string")
        if ";" in predicate:
            raise RuntimeError("dbt.ref(...).where(...) does not allow semicolons")
        relation_sql = f"(select * from {self._relation_sql} where {predicate}) as pybridge_where"
        return RelationFrame(self._session, relation_sql)

    def join(
        self,
        other: "RelationFrame",
        on=None,
        how: str = "inner",
        engine: Optional[str] = None,
        memory_limit: str = "512MB",
        threads: Optional[int] = None,
        temp_dir: Optional[str] = None,
        max_temp_directory_size: Optional[str] = None,
        filter_by: Optional[str] = None,
    ):
        if not isinstance(other, RelationFrame):
            raise RuntimeError("dbt.ref(...).join(other, ...) requires another RelationFrame; pass dbt.ref('...')")
        if self._session is not other._session:
            if engine == "duckdb":
                return FederatedJoinFrame(
                    self,
                    other,
                    on=on,
                    how=how,
                    memory_limit=memory_limit,
                    threads=threads,
                    temp_dir=temp_dir,
                    max_temp_directory_size=max_temp_directory_size,
                    filter_by=filter_by,
                )
            left_name = getattr(self._session, "connection_name", "unknown")
            right_name = getattr(other._session, "connection_name", "unknown")
            raise RuntimeError(
                "Cannot push down a join across PyBridge connections "
                f"{left_name!r} and {right_name!r}. Load each side explicitly with "
                ".as_dataframe(), then join locally with Polars or pandas, or pass "
                "engine='duckdb' for an explicit spill-backed join."
            )
        if engine not in (None, "postgres"):
            raise RuntimeError(
                "The DuckDB join engine is only used for cross-connection joins"
            )
        how_normalized = str(how or "").strip().lower()
        valid_join_types = {"inner", "left", "right", "full", "full outer", "left outer", "right outer", "cross"}
        if how_normalized not in valid_join_types:
            raise RuntimeError(
                f"Invalid join type {how!r}. Expected one of: {sorted(valid_join_types)}"
            )
        # Wrap each side in `select * from (...) as <alias>`. Without this
        # wrap, chaining off a `.select()` / `.where()` (whose _relation_sql
        # already ends with `as pybridge_<op>`) would produce two consecutive
        # aliases (`as pybridge_select as pybridge_l`), which Postgres rejects.
        left = f"(select * from {self._relation_sql}) as pybridge_l"
        right = f"(select * from {other._relation_sql}) as pybridge_r"

        if how_normalized == "cross":
            if on:
                raise RuntimeError("Cross joins do not take an `on=` argument")
            relation_sql = (
                f"(select * from {left} cross join {right}) as pybridge_join"
            )
            return RelationFrame(self._session, relation_sql)

        if isinstance(on, str):
            keys = [on]
        elif on is None:
            keys = []
        else:
            try:
                keys = list(on)
            except TypeError:
                raise RuntimeError(
                    f"Invalid `on=` value {on!r}; expected a column name or list of names"
                ) from None
        keys = [str(k).strip() for k in keys if str(k).strip()]
        if not keys:
            raise RuntimeError("dbt.ref(...).join(...) requires `on=` to be a column name or list of names")
        for key in keys:
            if ";" in key or '"' in key:
                raise RuntimeError(f"Invalid join key {key!r}; column names must not contain ';' or '\"'")

        # USING (col) keeps a single deduplicated copy of the join column in
        # the output; if we used ON pybridge_l.k = pybridge_r.k with `select *`
        # we'd get two columns named k, which fails our duplicate-column check.
        using_columns = ", ".join(quote_ident(k) for k in keys)
        relation_sql = (
            f"(select * from {left} {how_normalized} join {right} using ({using_columns}))"
            f" as pybridge_join"
        )
        return RelationFrame(self._session, relation_sql)

    def __getattr__(self, item):
        return getattr(self._load(), item)

    def __getitem__(self, key):
        return self._load()[key]

    def __setitem__(self, key, value):
        self._load()[key] = value

    def __len__(self):
        return len(self._load())

    def __iter__(self):
        return iter(self._load())

    def __contains__(self, item):
        return item in self._load()

    def __repr__(self) -> str:
        return repr(self._load())

    def as_dataframe(self):
        return self._load()


_PLAIN_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def consistent_cut(*inputs, lag=None, log=print):
    """Filter every input to one shared watermark, for consistency across servers.

    Pass ``(relation_frame, column)`` pairs whose column only grows (an
    ``updated_at`` or ``created_at`` timestamp, or a sequence id). Inside the
    snapshots pinned at model start, the watermark is the smallest of the
    per-input maximums, minus ``lag``; each input keeps only rows with
    ``column <= watermark``. Rows past the slowest source are excluded
    everywhere, so the inputs describe the same moment. Use ``lag`` to cover
    transactions that stamp a value before they commit.

    Returns the filtered frames in input order.
    """
    if not inputs:
        raise RuntimeError("consistent_cut() needs at least one (relation, column) pair")
    pairs = []
    for item in inputs:
        try:
            frame, column = item
        except (TypeError, ValueError):
            raise RuntimeError(
                "consistent_cut() takes (relation, column) pairs, for example "
                "consistent_cut((dbt.source('app', 'customers'), 'updated_at'), ...)"
            ) from None
        if not isinstance(frame, RelationFrame) or isinstance(frame, FederatedJoinFrame):
            raise RuntimeError(
                "consistent_cut() needs relations from dbt.ref()/dbt.source(); "
                "apply it before a DuckDB join"
            )
        column = str(column)
        if not _PLAIN_COLUMN_RE.fullmatch(column):
            raise RuntimeError(f"consistent_cut() column must be a plain column name, got {column!r}")
        pairs.append((frame, column))

    maximums = []
    for frame, column in pairs:
        with frame._session.conn.cursor() as cur:
            cur.execute(f"select max({quote_ident(column)}) from {frame._relation_sql}")
            maximums.append(cur.fetchone()[0])
    present = [value for value in maximums if value is not None]
    if not present:
        log("[pybridge] consistent_cut: every input is empty; nothing to filter")
        return tuple(frame for frame, _ in pairs)
    watermark = min(present)
    if lag is not None:
        watermark = watermark - lag
    log(f"[pybridge] consistent_cut watermark {watermark!r} (per-input maximums {maximums!r})")

    filtered = []
    for frame, column in pairs:
        with frame._session.conn.cursor() as cur:
            literal = cur.mogrify("%s", (watermark,)).decode()
        filtered.append(
            RelationFrame(
                frame._session,
                f"(select * from {frame._relation_sql} "
                f"where {quote_ident(column)} <= {literal}) as pybridge_cut",
            )
        )
    return tuple(filtered)


# Postgres type OID -> DuckDB staging type. Staging tables are created from
# these up front: inferring types from the first batch silently rounds
# numerics (1.55 -> 2 after a batch of whole numbers) and rejects later
# values when the first batch is all NULL.
_DUCKDB_TYPES = {
    16: "BOOLEAN",
    17: "BLOB",
    19: "VARCHAR",
    20: "BIGINT",
    21: "SMALLINT",
    23: "INTEGER",
    25: "VARCHAR",
    700: "REAL",
    701: "DOUBLE",
    1007: "INTEGER[]",
    1009: "VARCHAR[]",
    1015: "VARCHAR[]",
    1016: "BIGINT[]",
    1021: "REAL[]",
    1022: "DOUBLE[]",
    1042: "VARCHAR",
    1043: "VARCHAR",
    1082: "DATE",
    1083: "TIME",
    1114: "TIMESTAMP",
    1184: "TIMESTAMPTZ",
    1186: "INTERVAL",
    2950: "UUID",
}
_PG_JSON_OIDS = {114, 3802}
_PG_NUMERIC_OID = 1700
_DUCKDB_MAX_DECIMAL_PRECISION = 38


def _encode_json(value):
    return json.dumps(value)


def _encode_bytes(value):
    return bytes(value)


def _decode_decimal(value):
    return Decimal(value)


class _StagedColumn:
    """How one Postgres column is staged in DuckDB and restored afterwards."""

    def __init__(self, name, type_oid, precision, scale):
        self.name = name
        self.encode = None
        self.decode = None
        self.text_passthrough = False
        self.read_as_text = False
        if type_oid == _PG_NUMERIC_OID:
            if precision and 0 < precision <= _DUCKDB_MAX_DECIMAL_PRECISION:
                # Join on real DECIMALs, but read results back as text:
                # DuckDB hands DECIMAL to pandas as float64.
                self.duckdb_type = f"DECIMAL({precision},{scale or 0})"
                self.decode = _decode_decimal
                self.read_as_text = True
            else:
                # Unconstrained numeric does not fit DuckDB's DECIMAL(38).
                # Carry exact text through the join and restore Decimals.
                self.duckdb_type = "VARCHAR"
                self.encode = str
                self.decode = _decode_decimal
                self.text_passthrough = True
        elif type_oid in _PG_JSON_OIDS:
            self.duckdb_type = "VARCHAR"
            self.encode = _encode_json
            self.decode = json.loads
            self.text_passthrough = True
        elif type_oid in _DUCKDB_TYPES:
            self.duckdb_type = _DUCKDB_TYPES[type_oid]
            if type_oid == 17:
                self.encode = _encode_bytes
        else:
            raise RuntimeError(
                f"Column {name!r} has Postgres type OID {type_oid}, which the DuckDB "
                "federated join does not stage. Cast it in .select(), for example "
                f"'{name}::text'."
            )


def _is_missing(value) -> bool:
    # DuckDB returns NULL text as NaN in pandas chunks.
    return value is None or (isinstance(value, float) and math.isnan(value))


def _map_non_null(series, fn):
    return series.map(lambda value: None if _is_missing(value) else fn(value)).astype(object)


class FederatedJoinFrame(RelationFrame):
    """Explicit cross-connection join staged in a temporary DuckDB database."""

    _SIZE_RE = re.compile(r"^[1-9][0-9]*(?:KB|MB|GB|TB)$", re.IGNORECASE)

    def __init__(
        self,
        left,
        right,
        on,
        how,
        memory_limit,
        threads=None,
        temp_dir=None,
        max_temp_directory_size=None,
        filter_by=None,
    ):
        if isinstance(on, str):
            keys = [on]
        elif on is None:
            keys = []
        else:
            try:
                keys = list(on)
            except TypeError:
                raise RuntimeError(
                    f"Invalid `on=` value {on!r}; expected a column name or list of names"
                ) from None
        self._keys = [str(key).strip() for key in keys if str(key).strip()]
        self._how = str(how or "").strip().lower()
        valid = {"inner", "left", "right", "full", "full outer", "left outer", "right outer", "cross"}
        if self._how not in valid:
            raise RuntimeError(
                f"Invalid join type {how!r}. Expected one of: {sorted(valid)}"
            )
        if self._how == "cross":
            if self._keys:
                raise RuntimeError("Cross joins do not take an `on=` argument")
        elif not self._keys:
            raise RuntimeError(
                "A DuckDB federated join requires `on=` to be a column name or list of names"
            )
        for key in self._keys:
            if ";" in key or '"' in key:
                raise RuntimeError(f"Invalid join key {key!r}")
        self._memory_limit = self._validated_size("memory_limit", memory_limit)
        self._max_temp_directory_size = (
            None
            if max_temp_directory_size is None
            else self._validated_size("max_temp_directory_size", max_temp_directory_size)
        )
        if threads is not None and (
            isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0
        ):
            raise RuntimeError(f"DuckDB threads must be a positive integer, got {threads!r}")
        self._threads = threads
        self._temp_dir = None if temp_dir is None else str(temp_dir)
        self._filter_by = self._validated_filter_by(filter_by)
        self._left = left
        self._right = right
        self._backend = getattr(left._session, "dataframe_backend", "pandas")
        self._limits = getattr(left._session, "limits", None) or ModelLimits()
        self._df = None

    # filter_by=<side> stages that side first and fetches only the other
    # side's rows whose key it contains. That is only correct when the other
    # side's unmatched rows would be dropped by the join anyway.
    _FILTER_SAFE_JOINS = {
        "left": {"inner", "left", "left outer"},
        "right": {"inner", "right", "right outer"},
    }
    _KEY_FILTER_CHUNK = 10_000

    def _validated_filter_by(self, filter_by):
        if filter_by is None:
            return None
        side = str(filter_by).strip().lower()
        if side not in self._FILTER_SAFE_JOINS:
            raise RuntimeError(f"filter_by must be 'left' or 'right', got {filter_by!r}")
        if self._how not in self._FILTER_SAFE_JOINS[side]:
            raise RuntimeError(
                f"filter_by={side!r} would drop rows a {self._how!r} join keeps; it works "
                f"with {sorted(self._FILTER_SAFE_JOINS[side])} joins"
            )
        if len(self._keys) != 1:
            raise RuntimeError("filter_by supports a single join key")
        return side

    @classmethod
    def _validated_size(cls, label, value):
        text = str(value).strip()
        if not cls._SIZE_RE.fullmatch(text):
            raise RuntimeError(
                f"DuckDB {label} must look like '512MB', '2GB', or another positive size"
            )
        return text.upper()

    @staticmethod
    def _duckdb_module():
        try:
            import duckdb
        except ImportError:
            raise RuntimeError(
                "DuckDB federation is optional. Install it with "
                "`pip install 'dbt-pybridge[federation]'`."
            ) from None
        return duckdb

    @staticmethod
    def _staged_columns(frame: RelationFrame):
        columns = frame._session.relation_columns(frame._relation_sql)
        return [_StagedColumn(*column) for column in columns]

    def _validate_columns(self, left_columns, right_columns):
        left_names = [column.name for column in left_columns]
        right_names = [column.name for column in right_columns]
        for side, names in (("left", left_names), ("right", right_names)):
            missing = [key for key in self._keys if key not in names]
            if missing:
                raise RuntimeError(
                    f"Join key(s) {missing} are missing from the {side} side; "
                    f"its columns are {names}"
                )
        overlap = sorted((set(left_names) & set(right_names)) - set(self._keys))
        if overlap:
            raise RuntimeError(
                f"Columns {overlap} exist on both sides of the DuckDB join. "
                "Rename or drop them in .select() before joining."
            )
        for column in left_columns + right_columns:
            if column.name in self._keys and column.text_passthrough:
                raise RuntimeError(
                    f"Join key {column.name!r} is unconstrained numeric or JSON, which is "
                    "staged as text and would compare by spelling. Cast it in .select(), "
                    f"for example '{column.name}::bigint'."
                )

    @staticmethod
    def _stage_frame(conn, table_name: str, frame: RelationFrame, columns, create=True) -> None:
        table = quote_ident(table_name)
        if create:
            column_ddl = ", ".join(
                f"{quote_ident(column.name)} {column.duckdb_type}" for column in columns
            )
            conn.execute(f"create table {table} ({column_ddl})")
        column_list = ", ".join(quote_ident(column.name) for column in columns)
        batches = frame._session.iter_relation_batches(frame._relation_sql, as_pandas=True)
        for batch in batches:
            batch = batch.copy()
            for column in columns:
                if column.encode is not None:
                    batch[column.name] = _map_non_null(batch[column.name], column.encode)
            conn.register("pybridge_batch", batch)
            try:
                # Explicit columns cast each batch into the declared types.
                conn.execute(
                    f"insert into {table} ({column_list}) "
                    f"select {column_list} from pybridge_batch"
                )
            finally:
                conn.unregister("pybridge_batch")

    def _stage_frame_by_keys(self, conn, table_name, frame, columns, key_table):
        """Stage `frame`, fetching only rows whose key exists in `key_table`."""
        column_ddl = ", ".join(
            f"{quote_ident(column.name)} {column.duckdb_type}" for column in columns
        )
        conn.execute(f"create table {quote_ident(table_name)} ({column_ddl})")
        key = quote_ident(self._keys[0])
        key_type = self._postgres_type_name(frame, self._keys[0])
        # Keys travel as text and are cast back to the source column's type
        # in Postgres, so the filter can still use an index on that column.
        keys_cursor = conn.cursor().execute(
            f"select distinct cast({key} as varchar) from {quote_ident(key_table)} "
            f"where {key} is not null"
        )
        while True:
            chunk = [row[0] for row in keys_cursor.fetchmany(self._KEY_FILTER_CHUNK)]
            if not chunk:
                break
            with frame._session.conn.cursor() as cur:
                keys_literal = cur.mogrify("%s::text[]", (chunk,)).decode()
            filtered = RelationFrame(
                frame._session,
                f"(select * from {frame._relation_sql} where {key} = "
                f"any(cast({keys_literal} as {key_type}[]))) as pybridge_key_filter",
            )
            self._stage_frame(conn, table_name, filtered, columns, create=False)

    @staticmethod
    def _postgres_type_name(frame, column):
        oid = next(
            type_oid
            for name, type_oid, _precision, _scale in frame._session.relation_columns(
                frame._relation_sql
            )
            if name == column
        )
        with frame._session.conn.cursor() as cur:
            cur.execute("select format_type(%s, null)", (oid,))
            return cur.fetchone()[0]

    def _open_query(self):
        duckdb = self._duckdb_module()
        left_columns = self._staged_columns(self._left)
        right_columns = self._staged_columns(self._right)
        self._validate_columns(left_columns, right_columns)
        tempdir = tempfile.TemporaryDirectory(
            prefix="dbt_pybridge_duckdb_", dir=self._temp_dir
        )
        database_path = os.path.join(tempdir.name, "federation.duckdb")
        conn = duckdb.connect(database_path)
        try:
            escaped_tempdir = tempdir.name.replace("'", "''")
            conn.execute(f"set temp_directory='{escaped_tempdir}'")
            conn.execute(f"set memory_limit='{self._memory_limit}'")
            conn.execute("set preserve_insertion_order=false")
            if self._threads is not None:
                conn.execute(f"set threads={int(self._threads)}")
            if self._max_temp_directory_size is not None:
                conn.execute(
                    f"set max_temp_directory_size='{self._max_temp_directory_size}'"
                )
            # Stage both sides only after their snapshots were pinned at
            # model start (see PostgresSessionRegistry.pin_snapshots).
            if self._filter_by == "left":
                self._stage_frame(conn, "pybridge_left", self._left, left_columns)
                self._stage_frame_by_keys(
                    conn, "pybridge_right", self._right, right_columns, "pybridge_left"
                )
            elif self._filter_by == "right":
                self._stage_frame(conn, "pybridge_right", self._right, right_columns)
                self._stage_frame_by_keys(
                    conn, "pybridge_left", self._left, left_columns, "pybridge_right"
                )
            else:
                self._stage_frame(conn, "pybridge_left", self._left, left_columns)
                self._stage_frame(conn, "pybridge_right", self._right, right_columns)
            left = quote_ident("pybridge_left")
            right = quote_ident("pybridge_right")
            by_name = {column.name: column for column in right_columns}
            by_name.update({column.name: column for column in left_columns})
            ordered = self._keys + [
                column.name
                for column in left_columns + right_columns
                if column.name not in self._keys
            ]
            select_list = ", ".join(
                f"cast({quote_ident(name)} as varchar) as {quote_ident(name)}"
                if by_name[name].read_as_text
                else quote_ident(name)
                for name in ordered
            )
            if self._how == "cross":
                query = f"select {select_list} from {left} cross join {right}"
            else:
                keys = ", ".join(quote_ident(key) for key in self._keys)
                query = (
                    f"select {select_list} from {left} {self._how} join {right} "
                    f"using ({keys})"
                )
            cursor = conn.execute(query)
            decoders = {
                column.name: column.decode
                for column in left_columns + right_columns
                if column.decode is not None
            }
            return tempdir, conn, cursor, decoders
        except BaseException:
            conn.close()
            tempdir.cleanup()
            raise

    def _iter_result_frames(self, chunk_size: int):
        tempdir, conn, cursor, decoders = self._open_query()
        try:
            self._result_columns = [column[0] for column in cursor.description]
            buffered = []
            buffered_rows = 0
            while True:
                # DuckDB hands out fixed-size vectors; regroup them into
                # chunk_size frames, concatenating each batch only once.
                chunk = cursor.fetch_df_chunk()
                exhausted = len(chunk) == 0
                if not exhausted:
                    buffered.append(chunk)
                    buffered_rows += len(chunk)
                while buffered_rows >= chunk_size or (exhausted and buffered_rows):
                    pending = pd.concat(buffered, ignore_index=True)
                    frame = pending.iloc[:chunk_size].reset_index(drop=True)
                    rest = pending.iloc[chunk_size:].reset_index(drop=True)
                    buffered = [rest] if len(rest) else []
                    buffered_rows = len(rest)
                    for name, decode in decoders.items():
                        frame[name] = _map_non_null(frame[name], decode)
                    yield frame
                if exhausted:
                    break
        finally:
            conn.close()
            tempdir.cleanup()

    def _to_backend(self, frame):
        if self._backend == "polars":
            import polars as pl
            return pl.from_pandas(frame)
        return frame

    def _load(self):
        if self._df is None:
            limits = self._limits
            frames = []
            rows = 0
            byte_count = 0
            result_frames = self._iter_result_frames(limits.batch_size)
            with closing(result_frames):
                for frame in result_frames:
                    rows += len(frame)
                    byte_count += int(frame.memory_usage(index=True, deep=True).sum())
                    if not limits.allow_large_tables and rows > limits.max_rows:
                        raise RuntimeError(
                            f"DuckDB federated join returned more than {limits.max_rows:,} rows. "
                            "Return joined.iter_batches() to stream it, or raise pybridge_max_rows."
                        )
                    if not limits.allow_large_tables and byte_count > limits.max_bytes:
                        raise RuntimeError(
                            f"DuckDB federated join result exceeded {limits.max_bytes:,} bytes. "
                            "Return joined.iter_batches() to stream it, or raise pybridge_max_bytes."
                        )
                    frames.append(frame)
            if frames:
                df = pd.concat(frames, ignore_index=True)
            else:
                df = pd.DataFrame(columns=self._result_columns)
            self._df = self._to_backend(df)
        return self._df

    def iter_batches(self, batch_size: Optional[int] = None):
        chunk_size = int(batch_size or self._limits.batch_size)
        if chunk_size <= 0:
            raise RuntimeError(f"Batch size must be > 0, got {chunk_size}")
        for frame in self._iter_result_frames(chunk_size):
            yield self._to_backend(frame)

    def select(self, projection_sql: str):
        raise RuntimeError("Apply select() before a DuckDB federated join")

    def where(self, predicate_sql: str):
        raise RuntimeError("Apply where() before a DuckDB federated join")

    def join(self, other, on=None, how="inner", engine=None, **_kwargs):
        raise RuntimeError("Chain additional joins after materializing the federated result")


class LocalPythonModelRunner:
    def __init__(self, credentials, parsed_model: Dict[str, Any], compiled_code: str) -> None:
        self.credentials = credentials
        self.parsed_model = parsed_model
        self.compiled_code = compiled_code

    def _model_config(self) -> Dict[str, Any]:
        raw_config = self.parsed_model.get("config", {})
        # dbt model config object behaves like mapping.
        return dict(raw_config)

    @staticmethod
    def _cfg_value(cfg: Dict[str, Any], key: str, legacy_key: Optional[str] = None, default: Any = None) -> Any:
        if key in cfg:
            return cfg.get(key)
        if legacy_key and legacy_key in cfg:
            return cfg.get(legacy_key)
        return default

    @staticmethod
    def _log(message: str) -> None:
        print(f"[pybridge] {message}")

    @staticmethod
    def _load_df_function(session_or_registry):
        # The callback dbt's compiled `dbtObj` calls for each `dbt.ref(...)` /
        # `dbt.source(...)`. We normalize the 3-part identifier dbt renders
        # ("db"."schema"."t") down to a 2-part one *here*, at the single entry
        # point, so every subsequent .select()/.where()/.join() wraps an
        # already-safe relation SQL and can't smuggle a cross-database
        # qualifier into the resulting subquery.
        def load(relation_sql: str, connection_name: Optional[str] = None) -> "RelationFrame":
            if isinstance(session_or_registry, PostgresSessionRegistry):
                session = session_or_registry.get(connection_name)
            else:
                # Backwards-compatible path for callers using this helper
                # directly with a single LocalPostgresSession.
                if connection_name:
                    raise RuntimeError(
                        "A named PyBridge connection requires a PostgresSessionRegistry"
                    )
                session = session_or_registry
            return RelationFrame(session, session._normalize_relation_sql(relation_sql))
        return load

    def _limits(self, cfg: Dict[str, Any]) -> ModelLimits:
        def _as_int(key: str, default: int) -> int:
            legacy_key = key.replace("pybridge_", "localpy_") if key.startswith("pybridge_") else None
            value = self._cfg_value(cfg, key, legacy_key, default)
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                raise RuntimeError(f"Invalid value for {key}: expected integer, got {value!r}") from None
            if parsed < 0:
                raise RuntimeError(f"Invalid value for {key}: expected >= 0, got {parsed}")
            return parsed

        def _as_bool(key: str, default: bool) -> bool:
            legacy_key = key.replace("pybridge_", "localpy_") if key.startswith("pybridge_") else None
            value = self._cfg_value(cfg, key, legacy_key, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, int) and value in (0, 1):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "t", "yes", "y", "on", "1"}:
                    return True
                if normalized in {"false", "f", "no", "n", "off", "0"}:
                    return False
            raise RuntimeError(
                f"Invalid value for {key}: expected boolean-like value, got {value!r}"
            )

        batch_size = _as_int("pybridge_batch_size", 100_000)
        if batch_size <= 0:
            raise RuntimeError(
                f"Invalid value for pybridge_batch_size: expected > 0, got {batch_size}"
            )

        return ModelLimits(
            max_rows=_as_int("pybridge_max_rows", 1_000_000),
            warn_rows=_as_int("pybridge_warn_rows", 200_000),
            max_bytes=_as_int("pybridge_max_bytes", 512 * 1024 * 1024),
            warn_bytes=_as_int("pybridge_warn_bytes", 128 * 1024 * 1024),
            max_total_rows=_as_int("pybridge_max_total_rows", 1_000_000),
            warn_total_rows=_as_int("pybridge_warn_total_rows", 200_000),
            max_total_bytes=_as_int("pybridge_max_total_bytes", 512 * 1024 * 1024),
            warn_total_bytes=_as_int("pybridge_warn_total_bytes", 128 * 1024 * 1024),
            batch_size=batch_size,
            allow_large_tables=_as_bool("pybridge_allow_large_tables", False),
            chunked_mode=_as_bool("pybridge_chunked_mode", False),
        )

    def _column_types(self, cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
        return self._type_mapping_config(cfg, "pybridge_column_types", legacy_key="localpy_column_types")

    def _categorical_types(self, cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
        return self._type_mapping_config(cfg, "pybridge_categorical_types", legacy_key="localpy_categorical_types")

    def _type_mapping_config(self, cfg: Dict[str, Any], key: str, legacy_key: Optional[str] = None) -> Optional[Dict[str, str]]:
        raw = self._cfg_value(cfg, key, legacy_key, None)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                f"Invalid {key} config: expected a dict of "
                "{column_name: postgres_type_sql}"
            )
        out: Dict[str, str] = {}
        for raw_col, raw_type in raw.items():
            col = str(raw_col).strip()
            pg_type = str(raw_type).strip()
            if not col or not pg_type:
                raise RuntimeError(
                    f"Invalid {key} config: column name and type must be non-empty strings."
                )
            out[col] = pg_type
        return out

    def _target_relation(self) -> TargetRelation:
        model_database = self.parsed_model.get("database")
        target_database = getattr(self.credentials, "database", None)
        if model_database and target_database and str(model_database) != str(target_database):
            raise RuntimeError(
                "Python model target database does not match the active PyBridge target: "
                f"model={model_database!r}, target={target_database!r}"
            )
        return TargetRelation(
            database=model_database,
            schema=self.parsed_model.get("schema"),
            identifier=self.parsed_model.get("alias") or self.parsed_model.get("name"),
        )

    def _normalize_unique_key(self, unique_key: Any) -> Optional[List[str]]:
        if unique_key is None:
            return None
        if isinstance(unique_key, str):
            return [unique_key]
        if isinstance(unique_key, (list, tuple)):
            keys = [str(v) for v in unique_key]
            if not keys:
                return None
            return keys
        raise RuntimeError(
            "Invalid unique_key for incremental Python model. "
            f"Expected string or list of strings, got {type(unique_key)!r}"
        )

    def _view_backing_relation(self, view_target: TargetRelation) -> TargetRelation:
        prefix = "__dbt_pybridge_view_"
        raw = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in view_target.identifier.lower())
        suffix = raw or "model"
        digest = hashlib.sha1(view_target.identifier.encode("utf-8")).hexdigest()[:8]
        max_suffix_len = max(1, 63 - len(prefix) - 1 - len(digest))
        identifier = f"{prefix}{suffix[:max_suffix_len]}_{digest}"
        return TargetRelation(
            database=view_target.database,
            schema=view_target.schema,
            identifier=identifier,
        )

    @staticmethod
    def _suffix_relation(relation: TargetRelation, suffix: str) -> TargetRelation:
        # Postgres identifiers are limited to 63 chars (NAMEDATALEN-1). Truncate
        # the base if the suffix would push us over so two long-but-different
        # base names don't collide after silent server-side truncation.
        max_base = max(1, 63 - len(suffix))
        return TargetRelation(
            database=relation.database,
            schema=relation.schema,
            identifier=relation.identifier[:max_base] + suffix,
        )

    def _view_intermediate_relation(self, view_target: TargetRelation) -> TargetRelation:
        return self._suffix_relation(view_target, "__pybtmp")

    def _view_backup_relation(self, view_target: TargetRelation) -> TargetRelation:
        return self._suffix_relation(view_target, "__pybbkup")

    def _backing_intermediate_relation(self, backing_target: TargetRelation) -> TargetRelation:
        return self._suffix_relation(backing_target, "__tmp")

    def _relation_kind(self, conn, relation: TargetRelation) -> Optional[str]:
        with conn.cursor() as cur:
            if relation.schema:
                cur.execute(
                    """
                    select c.relkind
                    from pg_catalog.pg_class c
                    join pg_catalog.pg_namespace n on n.oid = c.relnamespace
                    where n.nspname = %s and c.relname = %s
                    limit 1
                    """,
                    (relation.schema, relation.identifier),
                )
            else:
                cur.execute(
                    """
                    select c.relkind
                    from pg_catalog.pg_class c
                    where c.relname = %s and pg_table_is_visible(c.oid)
                    limit 1
                    """,
                    (relation.identifier,),
                )
            row = cur.fetchone()
            return row[0] if row else None

    def _drop_existing_relation(self, conn, relation: TargetRelation) -> None:
        relkind = self._relation_kind(conn, relation)
        if relkind is None:
            return

        relation_sql = relation.render()
        # CASCADE matches dbt-core's postgres__drop_relation convention: a
        # downstream view (or chain of views) shouldn't block a model rebuild,
        # since dbt rebuilds dependents on the next run anyway.
        drop_sql = {
            "r": f"drop table if exists {relation_sql} cascade",
            "p": f"drop table if exists {relation_sql} cascade",
            "v": f"drop view if exists {relation_sql} cascade",
            "m": f"drop materialized view if exists {relation_sql} cascade",
            "f": f"drop foreign table if exists {relation_sql} cascade",
        }.get(relkind, f"drop table if exists {relation_sql} cascade")

        with conn.cursor() as cur:
            cur.execute(drop_sql)
        conn.commit()

    def _create_view(self, conn, view_target: TargetRelation, backing_table: TargetRelation) -> None:
        """Plain `CREATE VIEW`. Caller must ensure no relation exists at view_target."""
        view_sql = view_target.render()
        backing_sql = backing_table.render()
        with conn.cursor() as cur:
            if view_target.schema:
                cur.execute(f"create schema if not exists {quote_ident(view_target.schema)}")
            cur.execute(f"create view {view_sql} as select * from {backing_sql}")
        conn.commit()

    # Kept under the old name so external callers (and the existing test) still
    # work. Internally just delegates to _create_view.
    def _create_or_replace_view(self, conn, view_target: TargetRelation, backing_table: TargetRelation) -> None:
        self._create_view(conn, view_target, backing_table)

    def _materialize_view_via_swap(
        self,
        conn,
        view_target: TargetRelation,
        backing_target: TargetRelation,
        write_backing,
    ) -> int:
        """Atomic-ish view materialization that mirrors dbt-core's Postgres pattern.

        Plain DROP+CREATE has a window where the view doesn't exist; on the
        Postgres path dbt-core uses a rename-swap so the user-facing name is
        always resolvable to *something*. We do the same here, with the extra
        wrinkle that we own the backing table too:

            1. Cleanup any leftovers from a prior failed run (intermediate
               view, backup view, intermediate backing).
            2. Build the new backing at an intermediate name (via
               `write_backing(intermediate_backing)`, which is just
               write_model_result with materialized='table').
            3. CREATE VIEW <intermediate_view> AS SELECT * FROM
               <intermediate_backing>.
            4. In a single transaction, rename existing view (if any) to
               backup, then rename intermediate to target. After commit,
               readers see the new view atomically.
            5. Drop the backup view (which still depended on the old backing).
            6. Drop the old backing under its stable name.
            7. Rename the intermediate backing into that stable name.
               PG view dependencies follow OIDs, not names, so step 7 doesn't
               break the (already-published) new view.
        """
        intermediate_view = self._view_intermediate_relation(view_target)
        backup_view = self._view_backup_relation(view_target)
        intermediate_backing = self._backing_intermediate_relation(backing_target)

        # 1. Cleanup leftovers from a prior failed run.
        for leftover in (intermediate_view, backup_view, intermediate_backing):
            self._drop_existing_relation(conn, leftover)

        # 2. Build the new backing at the intermediate name.
        rows_written = write_backing(intermediate_backing)

        # 3. Build the new view at the intermediate name, pointing at the new
        #    backing.
        self._create_view(conn, intermediate_view, intermediate_backing)

        # Whatever sits at the view target now: a view (rename to backup), a
        # non-view (drop with CASCADE — same behavior as dbt-core when an
        # incompatible relation occupies the target slot), or nothing.
        existing_kind = self._relation_kind(conn, view_target)
        existing_is_view = existing_kind == "v"
        if existing_kind is not None and not existing_is_view:
            self._drop_existing_relation(conn, view_target)
            existing_is_view = False

        # 4. Atomic swap (rename existing → backup, intermediate → target).
        with conn.cursor() as cur:
            if existing_is_view:
                cur.execute(
                    f"alter view {view_target.render()} rename to {quote_ident(backup_view.identifier)}"
                )
            cur.execute(
                f"alter view {intermediate_view.render()} rename to {quote_ident(view_target.identifier)}"
            )
        conn.commit()

        # 5. Drop the backup view (no-op if there was no prior view).
        if existing_is_view:
            self._drop_existing_relation(conn, backup_view)

        # 6. Drop the old backing under the stable name (no-op on first run).
        self._drop_existing_relation(conn, backing_target)

        # 7. Rename intermediate backing into the stable name. The new view's
        #    pg_rewrite rule references the backing's OID; the rename
        #    preserves OID, so the view continues to resolve correctly.
        with conn.cursor() as cur:
            cur.execute(
                f"alter table {intermediate_backing.render()} rename to {quote_ident(backing_target.identifier)}"
            )
        conn.commit()

        return rows_written

    def run(self) -> int:
        cfg = self._model_config()
        dataframe_backend = str(
            self._cfg_value(cfg, "pybridge_dataframe_backend", "localpy_dataframe_backend", "pandas")
        ).lower()
        limits = self._limits(cfg)
        materialized = str(cfg.get("materialized", "table")).lower()
        unique_key = self._normalize_unique_key(cfg.get("unique_key"))
        column_types = self._column_types(cfg)
        categorical_types = self._categorical_types(cfg)
        on_schema_change = str(cfg.get("on_schema_change", "ignore")).lower()
        sync_drop_cascade = self._cfg_value(cfg, "pybridge_sync_drop_cascade", None, False)
        if isinstance(sync_drop_cascade, str):
            sync_drop_cascade = sync_drop_cascade.strip().lower() in {"true", "t", "yes", "y", "on", "1"}
        else:
            sync_drop_cascade = bool(sync_drop_cascade)
        incremental_strategy = str(cfg.get("incremental_strategy", "default")).lower()
        if incremental_strategy == "default":
            incremental_strategy = "merge" if unique_key else "append"

        registry = PostgresSessionRegistry(
            target_credentials=self.credentials,
            named_connections=getattr(self.credentials, "pybridge_connections", None),
            limits=limits,
            dataframe_backend=dataframe_backend,
            logger=self._log,
            session_factory=LocalPostgresSession,
            target_isolation=self._cfg_value(
                cfg, "pybridge_target_isolation", None, "repeatable read"
            ),
        )
        session = registry.get()
        primary_error: Optional[BaseException] = None

        try:
            namespace: Dict[str, Any] = {}
            exec(self.compiled_code, namespace)
            registry.pin_snapshots(namespace.get("__pybridge_source_connections__") or [])

            model_fn = namespace.get("model")
            if model_fn is None:
                raise RuntimeError("Python model file must define model(dbt, session)")
            if not callable(model_fn):
                raise RuntimeError("Python model symbol 'model' must be callable")

            dbt_obj_cls = namespace.get("dbtObj")
            if dbt_obj_cls is None:
                raise RuntimeError("Compiled Python model is missing dbtObj from dbt py_script_postfix")

            dbt_obj = dbt_obj_cls(self._load_df_function(registry))
            model_result = model_fn(dbt_obj, session)
            if isinstance(model_result, RelationFrame):
                model_result = model_result.as_dataframe()
            target = self._target_relation()
            if materialized == "view":
                backing_target = self._view_backing_relation(target)

                def _write_backing(intermediate_backing):
                    return write_model_result(
                        conn=session.conn,
                        target=intermediate_backing,
                        result=model_result,
                        batch_size=limits.batch_size,
                        materialized="table",
                        incremental_strategy="append",
                        unique_key=None,
                        column_types=column_types,
                        categorical_types=categorical_types,
                        logger=self._log,
                    )

                return self._materialize_view_via_swap(
                    session.conn,
                    view_target=target,
                    backing_target=backing_target,
                    write_backing=_write_backing,
                )

            return write_model_result(
                conn=session.conn,
                target=target,
                result=model_result,
                batch_size=limits.batch_size,
                materialized=materialized,
                incremental_strategy=incremental_strategy,
                unique_key=unique_key,
                column_types=column_types,
                categorical_types=categorical_types,
                logger=self._log,
                on_schema_change=on_schema_change,
                cascade_drops=sync_drop_cascade,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                registry.close()
            except Exception as close_error:
                if primary_error is None:
                    raise
                self._log(
                    "Connection cleanup also failed after the model error: "
                    f"{type(close_error).__name__}: {close_error}"
                )
