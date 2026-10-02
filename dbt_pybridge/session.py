from __future__ import annotations

from dataclasses import dataclass, field
import os
import subprocess
from threading import Lock
import time
from typing import Any, Callable, Dict, Iterator, Mapping, Optional
import uuid
import warnings

import psycopg2


@dataclass(frozen=True)
class ModelLimits:
    max_rows: int = 1_000_000
    warn_rows: int = 200_000
    max_bytes: int = 512 * 1024 * 1024
    warn_bytes: int = 128 * 1024 * 1024
    max_total_rows: int = 1_000_000
    warn_total_rows: int = 200_000
    max_total_bytes: int = 512 * 1024 * 1024
    warn_total_bytes: int = 128 * 1024 * 1024
    batch_size: int = 100_000
    allow_large_tables: bool = False
    chunked_mode: bool = False


@dataclass(frozen=True)
class TargetRelation:
    database: Optional[str]
    schema: Optional[str]
    identifier: str

    def render(self) -> str:
        # PostgreSQL relation SQL is at most schema-qualified. `database` is
        # retained as metadata so the runner can validate it against the active
        # connection, but emitting database.schema.table would produce an
        # unsupported cross-database reference.
        parts = []
        if self.schema:
            parts.append(quote_ident(self.schema))
        parts.append(quote_ident(self.identifier))
        return ".".join(parts)


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


class ModelUsageTracker:
    """Tracks cumulative dataframe input across every connection in one model."""

    def __init__(self, limits: ModelLimits) -> None:
        self.limits = limits
        self.total_rows = 0
        self.total_bytes = 0
        self._warned_rows = False
        self._warned_bytes = False
        self._lock = Lock()

    def record(self, relation_sql: str, rows: int, byte_count: int, bypass_limits: bool) -> None:
        with self._lock:
            self.total_rows += rows
            self.total_bytes += byte_count
            if not self._warned_rows and self.total_rows > self.limits.warn_total_rows:
                warnings.warn(
                    (
                        f"Python model inputs have loaded {self.total_rows:,} rows in total. "
                        "This plugin is intended for small/medium transforms."
                    ),
                    stacklevel=3,
                )
                self._warned_rows = True
            if not self._warned_bytes and self.total_bytes > self.limits.warn_total_bytes:
                warnings.warn(
                    (
                        "Python model inputs have loaded an estimated "
                        f"{LocalPostgresSession._human_bytes(self.total_bytes)} in total. "
                        "This plugin is intended for small/medium transforms."
                    ),
                    stacklevel=3,
                )
                self._warned_bytes = True
            if bypass_limits:
                return
            if self.total_rows > self.limits.max_total_rows:
                raise RuntimeError(
                    f"Python model inputs exceeded pybridge_max_total_rows while loading "
                    f"{relation_sql}: {self.total_rows:,} > {self.limits.max_total_rows:,}"
                )
            if self.total_bytes > self.limits.max_total_bytes:
                raise RuntimeError(
                    "Python model inputs exceeded pybridge_max_total_bytes while loading "
                    f"{relation_sql}: {LocalPostgresSession._human_bytes(self.total_bytes)} > "
                    f"{LocalPostgresSession._human_bytes(self.limits.max_total_bytes)}"
                )


@dataclass(frozen=True)
class NamedPostgresCredentials:
    """Validated credentials for one non-target PyBridge connection."""

    database: Optional[str] = None
    host: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)
    password_env: Optional[str] = field(default=None, repr=False)
    password_command: Optional[tuple] = field(default=None, repr=False)
    service: Optional[str] = None
    passfile: Optional[str] = field(default=None, repr=False)
    port: Optional[int] = 5432
    connect_timeout: int = 10
    search_path: Optional[str] = None
    keepalives_idle: int = 0
    sslmode: Optional[str] = None
    sslcert: Optional[str] = None
    sslkey: Optional[str] = None
    sslrootcert: Optional[str] = None
    sslpassword: Optional[str] = field(default=None, repr=False)
    gssencmode: Optional[str] = None
    krbsrvname: Optional[str] = None
    application_name: Optional[str] = "dbt-pybridge"

    @classmethod
    def from_mapping(cls, name: str, raw: Mapping[str, Any]) -> "NamedPostgresCredentials":
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                f"Invalid pybridge_connections entry {name!r}: expected a mapping of Postgres settings"
            )

        supported = {
            "database", "dbname", "host", "user", "password", "pass", "port",
            "connect_timeout", "search_path", "keepalives_idle", "sslmode", "sslcert",
            "sslkey", "sslrootcert", "sslpassword", "gssencmode", "krbsrvname",
            "application_name", "password_env", "password_command", "service",
            "passfile",
        }
        unknown = sorted(set(raw) - supported)
        if unknown:
            raise RuntimeError(
                f"Invalid pybridge_connections entry {name!r}: unsupported keys {unknown}"
            )

        if "database" in raw and "dbname" in raw and raw["database"] != raw["dbname"]:
            raise RuntimeError(
                f"Invalid pybridge_connections entry {name!r}: database and dbname disagree"
            )
        if "password" in raw and "pass" in raw and raw["password"] != raw["pass"]:
            raise RuntimeError(
                f"Invalid pybridge_connections entry {name!r}: password and pass disagree"
            )

        database = raw.get("database", raw.get("dbname"))
        password = raw.get("password", raw.get("pass"))
        password_env = raw.get("password_env")
        service = raw.get("service")
        passfile = raw.get("passfile")
        password_command = raw.get("password_command")
        if password_command is not None:
            # A list, never a shell string: no quoting rules, no shell injection.
            if (
                isinstance(password_command, (str, bytes))
                or not isinstance(password_command, (list, tuple))
                or not password_command
                or not all(isinstance(part, str) and part for part in password_command)
            ):
                raise RuntimeError(
                    f"Invalid pybridge_connections entry {name!r}: password_command must be "
                    "a non-empty list of strings, for example "
                    "['aws', 'rds', 'generate-db-auth-token', '--hostname', '...']"
                )
            password_command = tuple(password_command)
        auth_methods = [
            password is not None,
            password_env is not None,
            passfile is not None,
            password_command is not None,
        ]
        if sum(auth_methods) > 1:
            raise RuntimeError(
                f"Invalid pybridge_connections entry {name!r}: configure only one of "
                "password/pass, password_env, password_command, or passfile"
            )
        if service is not None and not str(service).strip():
            raise RuntimeError(
                f"Invalid pybridge_connections entry {name!r}: service must be non-empty"
            )
        missing = [] if service else [
            key for key, value in (
                ("database/dbname", database),
                ("host", raw.get("host")),
                ("user", raw.get("user")),
            )
            if value is None or not str(value).strip()
        ]
        # No auth setting is required: libpq can authenticate with a client
        # certificate (sslcert/sslkey), Kerberos/GSSAPI, ~/.pgpass, or trust.
        if missing:
            raise RuntimeError(
                f"Invalid pybridge_connections entry {name!r}: missing required keys {missing}"
            )

        try:
            raw_port = raw.get("port", None if service else 5432)
            port = None if raw_port is None else int(raw_port)
            connect_timeout = int(raw.get("connect_timeout", 10))
            keepalives_idle = int(raw.get("keepalives_idle", 0))
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Invalid pybridge_connections entry {name!r}: port, connect_timeout, and "
                "keepalives_idle must be integers"
            ) from None
        if port is not None and not 0 < port <= 65535:
            raise RuntimeError(
                f"Invalid pybridge_connections entry {name!r}: port must be between 1 and 65535"
            )
        if connect_timeout < 0 or keepalives_idle < 0:
            raise RuntimeError(
                f"Invalid pybridge_connections entry {name!r}: timeout/keepalive values must be >= 0"
            )
        for key, value in (("password_env", password_env), ("passfile", passfile)):
            if value is not None and not str(value).strip():
                raise RuntimeError(
                    f"Invalid pybridge_connections entry {name!r}: {key} must be non-empty"
                )

        def optional_string(key: str, default=None):
            value = raw.get(key, default)
            return None if value is None else str(value)

        return cls(
            database=None if database is None else str(database),
            host=None if raw.get("host") is None else str(raw["host"]),
            user=None if raw.get("user") is None else str(raw["user"]),
            password=None if password is None else str(password),
            password_env=optional_string("password_env"),
            password_command=password_command,
            service=optional_string("service"),
            passfile=optional_string("passfile"),
            port=port,
            connect_timeout=connect_timeout,
            search_path=optional_string("search_path"),
            keepalives_idle=keepalives_idle,
            sslmode=optional_string("sslmode"),
            sslcert=optional_string("sslcert"),
            sslkey=optional_string("sslkey"),
            sslrootcert=optional_string("sslrootcert"),
            sslpassword=optional_string("sslpassword"),
            gssencmode=optional_string("gssencmode"),
            krbsrvname=optional_string("krbsrvname"),
            application_name=optional_string("application_name", "dbt-pybridge"),
        )


def _run_password_command(command, connection_name: str, timeout: int) -> str:
    """Fetch a short-lived token (RDS IAM, Entra ID, Vault) just before connecting.

    The token is passed straight to libpq and never logged or stored.
    """
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"password_command for PyBridge connection {connection_name!r} failed to run: "
            f"{type(exc).__name__}"
        ) from None
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"password_command for PyBridge connection {connection_name!r} exited with "
            f"status {completed.returncode}"
            + (f": {detail[-1][:200]}" if detail else "")
        )
    token = (completed.stdout or "").strip()
    if not token:
        raise RuntimeError(
            f"password_command for PyBridge connection {connection_name!r} printed no token"
        )
    return token


class LocalPostgresSession:
    def __init__(
        self,
        credentials,
        limits: ModelLimits,
        dataframe_backend: str = "pandas",
        logger: Optional[Callable[[str], None]] = None,
        connection_name: str = "target",
        usage_tracker: Optional[ModelUsageTracker] = None,
        target_isolation: str = "repeatable read",
    ) -> None:
        self.credentials = credentials
        self.limits = limits
        self.dataframe_backend = dataframe_backend
        self._logger = logger
        self.connection_name = connection_name
        self.usage_tracker = usage_tracker
        connect_kwargs: Dict[str, Any] = {}
        keepalives_idle = int(getattr(credentials, "keepalives_idle", 0) or 0)
        if keepalives_idle:
            connect_kwargs["keepalives_idle"] = keepalives_idle
        search_path = getattr(credentials, "search_path", None)
        if search_path:
            connect_kwargs["options"] = "-c search_path={}".format(
                str(search_path).replace(" ", "\\ ")
            )
        for key in (
            "sslmode", "sslcert", "sslkey", "sslrootcert", "sslpassword",
            "gssencmode", "krbsrvname", "application_name",
        ):
            value = getattr(credentials, key, None)
            if value:
                connect_kwargs[key] = value
        for key, value in (
            ("dbname", getattr(credentials, "database", None)),
            ("user", getattr(credentials, "user", None)),
            ("host", getattr(credentials, "host", None)),
            ("port", getattr(credentials, "port", None)),
            ("service", getattr(credentials, "service", None)),
            ("passfile", getattr(credentials, "passfile", None)),
        ):
            if value is not None:
                connect_kwargs[key] = value
        password = getattr(credentials, "password", None)
        password_env = getattr(credentials, "password_env", None)
        if password_env:
            password = os.environ.get(password_env)
            if password is None:
                raise RuntimeError(
                    f"Environment variable {password_env!r} configured for PyBridge "
                    f"connection {connection_name!r} is not set"
                )
        password_command = getattr(credentials, "password_command", None)
        if password_command:
            password = _run_password_command(
                password_command,
                connection_name,
                # connect_timeout 0 is libpq's "wait forever"; honor it here too.
                timeout=getattr(credentials, "connect_timeout", 10) or None,
            )
        if password is not None:
            connect_kwargs["password"] = password
        connect_kwargs["connect_timeout"] = getattr(credentials, "connect_timeout", 10)
        self.conn = psycopg2.connect(**connect_kwargs)
        if connection_name == "target":
            # Reads and writes share this transaction. REPEATABLE READ gives
            # every ref()/source() read in the model one snapshot; writes
            # still see their own changes. A concurrent update to rows this
            # model merges into fails loudly with a serialization error.
            self.conn.set_session(
                isolation_level=target_isolation.upper(), autocommit=False
            )
        else:
            # A stable snapshot per source avoids seeing changes midway through
            # a model. Separate servers still cannot share one global snapshot.
            self.conn.set_session(
                isolation_level="REPEATABLE READ", readonly=True, autocommit=False
            )
        dsn = self.conn.get_dsn_parameters()
        self.database = dsn.get("dbname") or getattr(credentials, "database", None)

    def close(self) -> None:
        self.conn.close()

    def pin_snapshot(self):
        """Run a first query so the REPEATABLE READ snapshot starts now."""
        with self.conn.cursor() as cur:
            cur.execute("select clock_timestamp()")
            return cur.fetchone()[0]

    def relation_columns(self, relation_sql: str):
        """Return (name, type_oid, precision, scale) for each relation column."""
        relation_sql = self._normalize_relation_sql(relation_sql)
        with self.conn.cursor() as cur:
            cur.execute(f"select * from {relation_sql} limit 0")
            return [
                (desc.name, desc.type_code, desc.precision, desc.scale)
                for desc in cur.description or []
            ]

    def _log(self, message: str) -> None:
        # getattr fallback: tests construct sessions via __new__, bypassing
        # __init__ where _logger is set. Don't fail noisily on those.
        logger = getattr(self, "_logger", None)
        if logger is not None:
            connection_name = getattr(self, "connection_name", "target")
            logger(f"[{connection_name}] {message}")

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
        current_db = getattr(self, "database", None) or getattr(
            self.credentials, "database", None
        )
        if current_db and db_part != str(current_db):
            raise RuntimeError(
                "dbt-pybridge does not support cross-database Postgres reads in Python models. "
                f"Got relation {relation_sql} while current database is '{current_db}'."
            )
        # Postgres cannot query cross-database via 3-part names; drop database qualifier.
        return ".".join(parts[1:])

    def _query_columns(self, query: str):
        with self.conn.cursor() as cur:
            cur.execute(f"select * from ({query}) as pybridge_columns limit 0")
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

    @staticmethod
    def _dataframe_bytes(df) -> int:
        memory_usage = df.memory_usage(index=True, deep=True)
        return int(memory_usage.sum())

    def _record_loaded_dataframe(
        self, relation_sql: str, df, for_chunking: bool, track_usage: bool = True
    ) -> None:
        row_count = len(df)
        byte_count = self._dataframe_bytes(df)
        usage_tracker = getattr(self, "usage_tracker", None)
        self._log(
            f"Loaded {relation_sql} ({row_count:,} rows, {self._human_bytes(byte_count)})"
        )
        if usage_tracker is None and row_count > self.limits.warn_rows:
            warnings.warn(
                f"Loaded relation {relation_sql} with {row_count:,} rows.",
                stacklevel=3,
            )
        if usage_tracker is None and byte_count > self.limits.warn_bytes:
            warnings.warn(
                f"Loaded relation {relation_sql} using {self._human_bytes(byte_count)}.",
                stacklevel=3,
            )
        bypass_limits = self.limits.allow_large_tables or (
            self.limits.chunked_mode and for_chunking
        )
        if not bypass_limits and row_count > self.limits.max_rows:
            raise RuntimeError(
                f"Relation {relation_sql} returned more than {self.limits.max_rows:,} rows"
            )
        if not bypass_limits and byte_count > self.limits.max_bytes:
            raise RuntimeError(
                f"Relation {relation_sql} loaded {self._human_bytes(byte_count)}, above limit "
                f"{self._human_bytes(self.limits.max_bytes)}"
            )
        if usage_tracker is not None and track_usage:
            usage_tracker.record(
                relation_sql, row_count, byte_count, bypass_limits=bypass_limits
            )

    def load_relation(self, relation_sql: str):
        import pandas as pd

        relation_sql = self._normalize_relation_sql(relation_sql)
        query = f"select * from {relation_sql}"
        if not self.limits.allow_large_tables:
            query += f" limit {self.limits.max_rows + 1}"
        chunks = []
        relation_rows = 0
        relation_bytes = 0
        usage_tracker = getattr(self, "usage_tracker", None)
        for chunk in self._iter_query_to_pandas(query, self.limits.batch_size):
            chunk_rows = len(chunk)
            chunk_bytes = self._dataframe_bytes(chunk)
            relation_rows += chunk_rows
            relation_bytes += chunk_bytes
            if not self.limits.allow_large_tables and relation_rows > self.limits.max_rows:
                raise RuntimeError(
                    f"Relation {relation_sql} returned more than {self.limits.max_rows:,} rows"
                )
            if not self.limits.allow_large_tables and relation_bytes > self.limits.max_bytes:
                raise RuntimeError(
                    f"Relation {relation_sql} loaded {self._human_bytes(relation_bytes)}, above limit "
                    f"{self._human_bytes(self.limits.max_bytes)}"
                )
            if usage_tracker is not None:
                usage_tracker.record(
                    relation_sql,
                    chunk_rows,
                    chunk_bytes,
                    bypass_limits=self.limits.allow_large_tables,
                )
            chunks.append(chunk)
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.DataFrame(columns=self._query_columns(query))
        self._record_loaded_dataframe(
            relation_sql, df, for_chunking=False, track_usage=False
        )
        if self.dataframe_backend == "polars":
            import polars as pl
            return pl.from_pandas(df)
        return df

    def iter_relation_batches(
        self, relation_sql: str, batch_size: Optional[int] = None, as_pandas: bool = False
    ) -> Iterator:
        relation_sql = self._normalize_relation_sql(relation_sql)
        query = f"select * from {relation_sql}"
        chunk_size = int(batch_size or self.limits.batch_size)
        if chunk_size <= 0:
            raise RuntimeError(f"Batch size must be > 0, got {chunk_size}")

        relation_rows = 0
        relation_bytes = 0
        bypass_limits = self.limits.allow_large_tables or self.limits.chunked_mode
        for batch_idx, chunk in enumerate(self._iter_query_to_pandas(query, chunk_size), start=1):
            relation_rows += len(chunk)
            relation_bytes += self._dataframe_bytes(chunk)
            if not bypass_limits and relation_rows > self.limits.max_rows:
                raise RuntimeError(
                    f"Relation {relation_sql} returned more than {self.limits.max_rows:,} rows"
                )
            if not bypass_limits and relation_bytes > self.limits.max_bytes:
                raise RuntimeError(
                    f"Relation {relation_sql} loaded {self._human_bytes(relation_bytes)}, above limit "
                    f"{self._human_bytes(self.limits.max_bytes)}"
                )
            self._record_loaded_dataframe(relation_sql, chunk, for_chunking=True)
            self._log(f"Processing batch {batch_idx}, rows={len(chunk)}")
            if self.dataframe_backend == "polars" and not as_pandas:
                import polars as pl
                yield pl.from_pandas(chunk)
            else:
                yield chunk


class PostgresSessionRegistry:
    """Lazily owns the target session and any explicitly named read sessions."""

    def __init__(
        self,
        target_credentials,
        named_connections: Optional[Mapping[str, Mapping[str, Any]]],
        limits: ModelLimits,
        dataframe_backend: str,
        logger: Optional[Callable[[str], None]] = None,
        session_factory=LocalPostgresSession,
        target_isolation: str = "repeatable read",
    ) -> None:
        normalized_isolation = " ".join(str(target_isolation).lower().split())
        if normalized_isolation not in {"repeatable read", "read committed"}:
            raise RuntimeError(
                "pybridge_target_isolation must be 'repeatable read' or 'read committed', "
                f"got {target_isolation!r}"
            )
        self._target_isolation = normalized_isolation
        self._target_credentials = target_credentials
        if named_connections is not None and not isinstance(named_connections, Mapping):
            raise RuntimeError("pybridge_connections must be a mapping of connection names to settings")

        self._named_credentials: Dict[str, NamedPostgresCredentials] = {}
        for raw_name, raw_settings in (named_connections or {}).items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise RuntimeError("Every pybridge_connections key must be a non-empty string")
            name = raw_name.strip()
            if name != raw_name:
                raise RuntimeError(
                    f"Invalid PyBridge connection name {raw_name!r}: leading/trailing whitespace is not allowed"
                )
            if name == "target":
                raise RuntimeError(
                    "PyBridge connection name 'target' is reserved for the active dbt target"
                )
            self._named_credentials[name] = NamedPostgresCredentials.from_mapping(
                name, raw_settings
            )
        self._limits = limits
        self._dataframe_backend = dataframe_backend
        self._logger = logger
        self._session_factory = session_factory
        self._sessions: Dict[str, LocalPostgresSession] = {}
        self._usage_tracker = ModelUsageTracker(limits)

    def get(self, connection_name: Optional[str] = None) -> LocalPostgresSession:
        if connection_name is None:
            name = "target"
        else:
            if not isinstance(connection_name, str):
                raise RuntimeError(
                    "pybridge_connection metadata must be a string connection name"
                )
            name = connection_name.strip()
            if not name or name != connection_name:
                raise RuntimeError(
                    f"Invalid PyBridge connection name {connection_name!r}"
                )
        if name in self._sessions:
            return self._sessions[name]

        if name == "target":
            credentials = self._target_credentials
        else:
            if name not in self._named_credentials:
                available = sorted(self._named_credentials)
                raise RuntimeError(
                    f"Unknown PyBridge connection {name!r}. Available named connections: {available}"
                )
            credentials = self._named_credentials[name]

        session = self._session_factory(
            credentials=credentials,
            limits=self._limits,
            dataframe_backend=self._dataframe_backend,
            logger=self._logger,
            connection_name=name,
            usage_tracker=self._usage_tracker,
            target_isolation=self._target_isolation,
        )
        self._sessions[name] = session
        return session

    def pin_snapshots(self, connection_names) -> None:
        """Start the target and every named source snapshot back to back.

        REPEATABLE READ snapshots begin at a transaction's first query, so
        lazily opened sources would otherwise start whenever the model first
        touches them (for a DuckDB join, after the other side has streamed).
        Separate servers still cannot share one snapshot; this bounds and
        logs the skew instead of leaving it unbounded.
        """
        names = ["target"] + sorted({str(n) for n in connection_names if n} - {"target"})
        if len(names) == 1:
            return
        sessions = [(name, self.get(name)) for name in names]
        # Measure locally: server clocks on different hosts may disagree.
        started = time.monotonic()
        pinned = [(name, session.pin_snapshot()) for name, session in sessions]
        skew_ms = (time.monotonic() - started) * 1000
        if self._logger is not None:
            listed = ", ".join(f"{name}@{ts}" for name, ts in pinned)
            self._logger(f"Pinned source snapshots within {skew_ms:.1f} ms ({listed})")

    def close(self) -> None:
        first_error = None
        for session in reversed(list(self._sessions.values())):
            try:
                session.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup path
                if first_error is None:
                    first_error = exc
        self._sessions.clear()
        if first_error is not None:
            raise first_error
