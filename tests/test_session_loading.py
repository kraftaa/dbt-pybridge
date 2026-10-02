import pytest

from dbt_pybridge.session import (
    LocalPostgresSession,
    ModelLimits,
    ModelUsageTracker,
    NamedPostgresCredentials,
    PostgresSessionRegistry,
    TargetRelation,
)


class FakeCursor:
    def __init__(self, rows, columns, with_description=True, executed_log=None):
        self._rows = list(rows)
        self._columns = list(columns)
        self.description = [(c,) for c in self._columns] if with_description else None
        self.executed = []
        self._executed_log = executed_log
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append(query)
        if self._executed_log is not None:
            self._executed_log.append(query)

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        if not self._rows:
            return None
        return self._rows[0]

    def fetchmany(self, size):
        start = self._offset
        end = start + size
        self._offset = end
        return self._rows[start:end]


class FakeConn:
    def __init__(self, rows, columns, with_description=True):
        self._rows = rows
        self._columns = columns
        self._with_description = with_description
        self.last_cursor_name = None
        self.cursor_names = []
        self.executed = []

    def cursor(self, name=None):
        self.last_cursor_name = name
        self.cursor_names.append(name)
        return FakeCursor(
            self._rows,
            self._columns,
            with_description=self._with_description,
            executed_log=self.executed,
        )


class FakeCredentials:
    def __init__(self, database="demo_db"):
        self.database = database


class FakeLiveConnection:
    def __init__(self):
        self.autocommit = None
        self.session_calls = []

    def set_session(self, **kwargs):
        self.session_calls.append(kwargs)

    def get_dsn_parameters(self):
        return {"dbname": "application"}

    def close(self):
        pass


def _make_session(rows, columns, backend="pandas", with_description=True):
    session = LocalPostgresSession.__new__(LocalPostgresSession)
    session.conn = FakeConn(rows=rows, columns=columns, with_description=with_description)
    session.credentials = FakeCredentials()
    session.limits = ModelLimits()
    session.dataframe_backend = backend
    return session


def test_load_relation_uses_cursor_dataframe():
    session = _make_session(rows=[(1, "a"), (2, "b")], columns=["id", "name"])

    df = session.load_relation('"public"."x"')

    assert list(df.columns) == ["id", "name"]
    assert len(df) == 2
    assert all("count(*)" not in query.lower() for query in session.conn.executed)
    assert any("limit 1000001" in query.lower() for query in session.conn.executed)


def test_load_relation_enforces_row_limit_from_bounded_fetch():
    session = _make_session(rows=[(1,), (2,), (3,)], columns=["id"])
    session.limits = ModelLimits(max_rows=2, warn_rows=100)

    with pytest.raises(RuntimeError) as exc:
        session.load_relation('"public"."x"')

    assert "more than 2 rows" in str(exc.value)
    assert any("limit 3" in query.lower() for query in session.conn.executed)


def test_iter_relation_batches_streams_batches():
    session = _make_session(rows=[(1,), (2,), (3,)], columns=["id"])

    chunks = list(session.iter_relation_batches('"public"."x"', batch_size=2))

    assert session.conn.last_cursor_name is not None
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 1


def test_iter_relation_batches_uses_unique_cursor_name_per_call():
    session = _make_session(rows=[(1,), (2,), (3,)], columns=["id"])

    list(session.iter_relation_batches('"public"."x"', batch_size=2))
    list(session.iter_relation_batches('"public"."x"', batch_size=2))

    named_cursors = [name for name in session.conn.cursor_names if name is not None]
    assert len(named_cursors) >= 2
    assert named_cursors[-1] != named_cursors[-2]


def test_iter_relation_batches_without_description_metadata():
    session = _make_session(rows=[(1, "a"), (2, "b")], columns=["id", "name"], with_description=False)

    chunks = list(session.iter_relation_batches('"public"."x"', batch_size=1))

    assert len(chunks) == 2
    assert list(chunks[0].columns) == ["column_1", "column_2"]


def test_normalize_relation_sql_drops_current_database():
    session = _make_session(rows=[], columns=[])
    relation = '"demo_db"."transform"."stg_big_orders"'
    assert session._normalize_relation_sql(relation) == '"transform"."stg_big_orders"'


def test_normalize_relation_sql_rejects_cross_database():
    session = _make_session(rows=[], columns=[])
    relation = '"other_db"."transform"."stg_big_orders"'
    try:
        session._normalize_relation_sql(relation)
    except RuntimeError as exc:
        assert "cross-database" in str(exc)
    else:
        raise AssertionError("Expected cross-database relation to raise RuntimeError")


def test_normalize_relation_sql_does_not_case_fold_quoted_database_names():
    session = _make_session(rows=[], columns=[])
    session.credentials = FakeCredentials(database="Demo_DB")

    with pytest.raises(RuntimeError):
        session._normalize_relation_sql('"demo_db"."transform"."orders"')


def test_target_relation_never_renders_postgres_database_qualifier():
    target = TargetRelation(database="analytics", schema="transform", identifier="orders")
    assert target.render() == '"transform"."orders"'


def test_named_credentials_accept_aliases_without_exposing_password():
    credentials = NamedPostgresCredentials.from_mapping(
        "app_db",
        {
            "host": "app.example.invalid",
            "user": "reader",
            "pass": "not-printed",
            "dbname": "application",
            "port": "5433",
            "sslmode": "require",
        },
    )

    assert credentials.database == "application"
    assert credentials.password == "not-printed"
    assert credentials.port == 5433
    assert credentials.sslmode == "require"
    assert "not-printed" not in repr(credentials)


def test_named_credentials_accept_just_in_time_password_environment():
    credentials = NamedPostgresCredentials.from_mapping(
        "app_db",
        {
            "host": "app.example.invalid",
            "user": "reader",
            "password_env": "SYNTHETIC_APP_PASSWORD",
            "dbname": "application",
        },
    )

    assert credentials.password is None
    assert credentials.password_env == "SYNTHETIC_APP_PASSWORD"
    assert "SYNTHETIC_APP_PASSWORD" not in repr(credentials)


def test_named_credentials_accept_libpq_service_without_inline_password():
    credentials = NamedPostgresCredentials.from_mapping(
        "app_db", {"service": "synthetic_app", "passfile": "/tmp/synthetic.pgpass"}
    )

    assert credentials.service == "synthetic_app"
    assert credentials.passfile == "/tmp/synthetic.pgpass"
    assert credentials.port is None


def test_named_credentials_reject_multiple_password_sources():
    with pytest.raises(RuntimeError) as exc:
        NamedPostgresCredentials.from_mapping(
            "app_db",
            {
                "host": "app.example.invalid",
                "user": "reader",
                "database": "application",
                "password": "secret",
                "password_env": "SYNTHETIC_APP_PASSWORD",
            },
        )

    assert "only one" in str(exc.value)


def test_named_session_resolves_password_at_connect_and_is_read_only(monkeypatch):
    captured = {}
    connection = FakeLiveConnection()

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return connection

    monkeypatch.setenv("SYNTHETIC_APP_PASSWORD", "runtime-secret")
    monkeypatch.setattr("dbt_pybridge.session.psycopg2.connect", fake_connect)
    credentials = NamedPostgresCredentials.from_mapping(
        "app_db",
        {
            "host": "app.example.invalid",
            "user": "reader",
            "database": "application",
            "password_env": "SYNTHETIC_APP_PASSWORD",
        },
    )

    session = LocalPostgresSession(
        credentials, ModelLimits(), connection_name="app_db"
    )

    assert captured["password"] == "runtime-secret"
    assert credentials.password is None
    assert session.database == "application"
    assert connection.session_calls == [
        {
            "isolation_level": "REPEATABLE READ",
            "readonly": True,
            "autocommit": False,
        }
    ]


def test_missing_just_in_time_password_fails_without_opening_connection(monkeypatch):
    monkeypatch.delenv("SYNTHETIC_APP_PASSWORD", raising=False)
    monkeypatch.setattr(
        "dbt_pybridge.session.psycopg2.connect",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not connect")),
    )
    credentials = NamedPostgresCredentials.from_mapping(
        "app_db",
        {
            "host": "app.example.invalid",
            "user": "reader",
            "database": "application",
            "password_env": "SYNTHETIC_APP_PASSWORD",
        },
    )

    with pytest.raises(RuntimeError) as exc:
        LocalPostgresSession(credentials, ModelLimits(), connection_name="app_db")

    assert "is not set" in str(exc.value)


def test_session_registry_is_lazy_and_caches_named_connections():
    created = []

    class RegistrySession:
        def __init__(self, credentials, limits, dataframe_backend, logger, connection_name, usage_tracker=None, target_isolation="repeatable read"):
            self.credentials = credentials
            self.connection_name = connection_name
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    target = FakeCredentials(database="analytics")
    registry = PostgresSessionRegistry(
        target_credentials=target,
        named_connections={
            "app_db": {
                "host": "app.example.invalid",
                "user": "reader",
                "password": "secret",
                "database": "application",
            }
        },
        limits=ModelLimits(),
        dataframe_backend="polars",
        session_factory=RegistrySession,
    )

    assert created == []
    target_session = registry.get()
    assert registry.get() is target_session
    app_session = registry.get("app_db")
    assert registry.get("app_db") is app_session
    assert app_session.credentials.database == "application"
    assert [session.connection_name for session in created] == ["target", "app_db"]

    registry.close()
    assert all(session.closed for session in created)


def test_session_registry_rejects_unknown_connection_without_showing_settings():
    registry = PostgresSessionRegistry(
        target_credentials=FakeCredentials(database="analytics"),
        named_connections={
            "app_db": {
                "host": "app.example.invalid",
                "user": "reader",
                "password": "secret",
                "database": "application",
            }
        },
        limits=ModelLimits(),
        dataframe_backend="pandas",
        session_factory=lambda **kwargs: None,
    )

    with pytest.raises(RuntimeError) as exc:
        registry.get("missing")

    assert "missing" in str(exc.value)
    assert "app_db" in str(exc.value)


def test_session_registry_validates_unused_connections_eagerly():
    with pytest.raises(RuntimeError) as exc:
        PostgresSessionRegistry(
            target_credentials=FakeCredentials(database="analytics"),
            named_connections={"unused_db": {"host": "example.invalid"}},
            limits=ModelLimits(),
            dataframe_backend="pandas",
        )

    assert "unused_db" in str(exc.value)
    assert "missing required keys" in str(exc.value)


def test_session_registry_rejects_reserved_target_name():
    with pytest.raises(RuntimeError) as exc:
        PostgresSessionRegistry(
            target_credentials=FakeCredentials(database="analytics"),
            named_connections={
                "target": {
                    "host": "example.invalid",
                    "user": "reader",
                    "password": "secret",
                    "database": "application",
                }
            },
            limits=ModelLimits(),
            dataframe_backend="pandas",
        )

    assert "reserved" in str(exc.value)


def test_session_registry_rejects_non_string_route_metadata():
    registry = PostgresSessionRegistry(
        target_credentials=FakeCredentials(database="analytics"),
        named_connections={},
        limits=ModelLimits(),
        dataframe_backend="pandas",
        session_factory=lambda **kwargs: None,
    )

    with pytest.raises(RuntimeError) as exc:
        registry.get(["app_db"])

    assert "must be a string" in str(exc.value)


def test_model_usage_tracker_enforces_aggregate_row_limit():
    tracker = ModelUsageTracker(
        ModelLimits(
            max_rows=10,
            warn_rows=10,
            max_bytes=10_000,
            warn_bytes=10_000,
            max_total_rows=3,
            warn_total_rows=100,
            max_total_bytes=10_000,
            warn_total_bytes=10_000,
        )
    )

    tracker.record('"public"."a"', rows=2, byte_count=100, bypass_limits=False)
    with pytest.raises(RuntimeError) as exc:
        tracker.record('"public"."b"', rows=2, byte_count=100, bypass_limits=False)

    assert "pybridge_max_total_rows" in str(exc.value)


def test_session_registry_pins_target_and_named_snapshots_together():
    import datetime as dt

    pinned = []
    messages = []

    class PinSession:
        def __init__(self, credentials, limits, dataframe_backend, logger, connection_name, usage_tracker=None, target_isolation="repeatable read"):
            self.connection_name = connection_name

        def pin_snapshot(self):
            pinned.append(self.connection_name)
            return dt.datetime(2024, 1, 1)

    settings = {"host": "h.example.invalid", "user": "reader", "database": "d"}
    registry = PostgresSessionRegistry(
        target_credentials=FakeCredentials(database="analytics"),
        named_connections={"app_db": settings, "billing_db": settings, "unused_db": settings},
        limits=ModelLimits(),
        dataframe_backend="pandas",
        logger=messages.append,
        session_factory=PinSession,
    )

    registry.pin_snapshots(["billing_db", "app_db", "billing_db"])

    # Only connections the model declares are opened; target goes first.
    assert pinned == ["target", "app_db", "billing_db"]
    assert "Pinned source snapshots within" in messages[-1]


def test_session_registry_skips_pinning_single_database_models():
    registry = PostgresSessionRegistry(
        target_credentials=FakeCredentials(database="analytics"),
        named_connections={},
        limits=ModelLimits(),
        dataframe_backend="pandas",
        session_factory=lambda **kwargs: pytest.fail("no session should open"),
    )

    registry.pin_snapshots([])


def test_session_registry_validates_target_isolation():
    with pytest.raises(RuntimeError, match="pybridge_target_isolation"):
        PostgresSessionRegistry(
            target_credentials=FakeCredentials(database="analytics"),
            named_connections={},
            limits=ModelLimits(),
            dataframe_backend="pandas",
            target_isolation="serializable-ish",
        )


@pytest.mark.parametrize(
    "auth",
    [
        {"sslmode": "verify-full", "sslcert": "/certs/client.crt", "sslkey": "/certs/client.key"},
        {"gssencmode": "prefer", "krbsrvname": "postgres"},
        {},
    ],
    ids=["client-certificate", "kerberos", "pgpass-or-trust"],
)
def test_named_credentials_allow_passwordless_auth(auth):
    from dbt_pybridge.session import NamedPostgresCredentials

    credentials = NamedPostgresCredentials.from_mapping(
        "app_db",
        {"host": "app.example.invalid", "user": "reader", "database": "application", **auth},
    )

    assert credentials.password is None
    for key, value in auth.items():
        assert getattr(credentials, key) == value


def test_named_credentials_hide_ssl_key_password():
    from dbt_pybridge.session import NamedPostgresCredentials

    credentials = NamedPostgresCredentials.from_mapping(
        "app_db",
        {
            "host": "app.example.invalid",
            "user": "reader",
            "database": "application",
            "sslkey": "/certs/client.key",
            "sslpassword": "key-secret",
        },
    )

    assert credentials.sslpassword == "key-secret"
    assert "key-secret" not in repr(credentials)


def _named_with_command(command):
    from dbt_pybridge.session import NamedPostgresCredentials

    return NamedPostgresCredentials.from_mapping(
        "app_db",
        {
            "host": "app.example.invalid",
            "user": "reader",
            "database": "application",
            "password_command": command,
        },
    )


def test_password_command_supplies_a_just_in_time_token(monkeypatch):
    import sys

    captured = {}

    class Connection:
        def set_session(self, **kwargs):
            pass

        def get_dsn_parameters(self):
            return {"dbname": "application"}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return Connection()

    monkeypatch.setattr("dbt_pybridge.session.psycopg2.connect", fake_connect)
    credentials = _named_with_command(
        [sys.executable, "-c", "print('short-lived-token')"]
    )

    LocalPostgresSession(credentials, ModelLimits(), connection_name="app_db")

    assert captured["password"] == "short-lived-token"
    assert "print" not in repr(credentials)


def test_password_command_failure_does_not_connect(monkeypatch):
    import sys

    monkeypatch.setattr(
        "dbt_pybridge.session.psycopg2.connect",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not connect")),
    )
    credentials = _named_with_command(
        [
            sys.executable,
            "-c",
            "import sys; print('stdout-secret'); "
            "sys.stderr.write('credential=example-secret\\n'); sys.exit(3)",
        ]
    )

    with pytest.raises(RuntimeError, match="exited with status 3") as exc:
        LocalPostgresSession(credentials, ModelLimits(), connection_name="app_db")

    # Credential helpers may print secrets; none of their output is surfaced.
    assert "example-secret" not in str(exc.value)
    assert "stdout-secret" not in str(exc.value)


@pytest.mark.parametrize("command", ["aws rds generate-db-auth-token", [], [""], [1]])
def test_password_command_must_be_a_list_of_strings(command):
    with pytest.raises(RuntimeError, match="password_command must be"):
        _named_with_command(command)


def test_password_command_is_exclusive_with_other_auth():
    from dbt_pybridge.session import NamedPostgresCredentials

    with pytest.raises(RuntimeError, match="configure only one of"):
        NamedPostgresCredentials.from_mapping(
            "app_db",
            {
                "host": "h",
                "user": "u",
                "database": "d",
                "password": "x",
                "password_command": ["true"],
            },
        )
