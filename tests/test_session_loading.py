import pytest

from dbt_pybridge.session import LocalPostgresSession, ModelLimits


class FakeCursor:
    def __init__(self, rows, columns, with_description=True):
        self._rows = list(rows)
        self._columns = list(columns)
        self.description = [(c,) for c in self._columns] if with_description else None
        self.executed = []
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append(query)

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

    def cursor(self, name=None):
        self.last_cursor_name = name
        self.cursor_names.append(name)
        return FakeCursor(self._rows, self._columns, with_description=self._with_description)


class FakeCredentials:
    def __init__(self, database="rx_development"):
        self.database = database


def _make_session(rows, columns, backend="pandas", with_description=True):
    session = LocalPostgresSession.__new__(LocalPostgresSession)
    session.conn = FakeConn(rows=rows, columns=columns, with_description=with_description)
    session.credentials = FakeCredentials()
    session.limits = ModelLimits()
    session.dataframe_backend = backend
    return session


def test_load_relation_uses_cursor_dataframe():
    session = _make_session(rows=[(1, "a"), (2, "b")], columns=["id", "name"])
    session.enforce_size_limits = lambda relation_sql, for_chunking=False: 2

    df = session.load_relation('"public"."x"')

    assert list(df.columns) == ["id", "name"]
    assert len(df) == 2


def test_iter_relation_batches_streams_batches():
    session = _make_session(rows=[(1,), (2,), (3,)], columns=["id"])
    session.enforce_size_limits = lambda relation_sql, for_chunking=False: 3

    chunks = list(session.iter_relation_batches('"public"."x"', batch_size=2))

    assert session.conn.last_cursor_name is not None
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 1


def test_iter_relation_batches_uses_unique_cursor_name_per_call():
    session = _make_session(rows=[(1,), (2,), (3,)], columns=["id"])
    session.enforce_size_limits = lambda relation_sql, for_chunking=False: 3

    list(session.iter_relation_batches('"public"."x"', batch_size=2))
    list(session.iter_relation_batches('"public"."x"', batch_size=2))

    named_cursors = [name for name in session.conn.cursor_names if name is not None]
    assert len(named_cursors) >= 2
    assert named_cursors[-1] != named_cursors[-2]


def test_iter_relation_batches_without_description_metadata():
    session = _make_session(rows=[(1, "a"), (2, "b")], columns=["id", "name"], with_description=False)
    session.enforce_size_limits = lambda relation_sql, for_chunking=False: 2

    chunks = list(session.iter_relation_batches('"public"."x"', batch_size=1))

    assert len(chunks) == 2
    assert list(chunks[0].columns) == ["column_1", "column_2"]


def test_normalize_relation_sql_drops_current_database():
    session = _make_session(rows=[], columns=[])
    relation = '"rx_development"."transform"."stg_big_orders"'
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


def test_enforce_size_limits_warns_on_large_bytes():
    session = _make_session(rows=[], columns=[])
    session.limits = ModelLimits(
        max_rows=1_000_000,
        warn_rows=1,
        max_bytes=1_000_000_000,
        warn_bytes=100,
        allow_large_tables=False,
        chunked_mode=False,
    )
    session.count_rows = lambda relation_sql: 10
    session.count_bytes = lambda relation_sql: 200

    with pytest.warns(UserWarning):
        session.enforce_size_limits('"public"."x"')


def test_enforce_size_limits_fails_on_byte_limit():
    session = _make_session(rows=[], columns=[])
    session.limits = ModelLimits(
        max_rows=1_000_000,
        warn_rows=1_000_000,
        max_bytes=100,
        warn_bytes=50,
        allow_large_tables=False,
        chunked_mode=False,
    )
    session.count_rows = lambda relation_sql: 10
    session.count_bytes = lambda relation_sql: 200

    with pytest.raises(RuntimeError) as exc:
        session.enforce_size_limits('"public"."x"')
    assert "above limit" in str(exc.value)
