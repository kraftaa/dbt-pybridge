from dbt_pybridge.session import LocalPostgresSession, ModelLimits


class FakeCursor:
    def __init__(self, rows, columns):
        self._rows = list(rows)
        self._columns = list(columns)
        self.description = [(c,) for c in self._columns]
        self.executed = []
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query):
        self.executed.append(query)

    def fetchall(self):
        return list(self._rows)

    def fetchmany(self, size):
        start = self._offset
        end = start + size
        self._offset = end
        return self._rows[start:end]


class FakeConn:
    def __init__(self, rows, columns):
        self._rows = rows
        self._columns = columns
        self.last_cursor_name = None

    def cursor(self, name=None):
        self.last_cursor_name = name
        return FakeCursor(self._rows, self._columns)


def _make_session(rows, columns, backend="pandas"):
    session = LocalPostgresSession.__new__(LocalPostgresSession)
    session.conn = FakeConn(rows=rows, columns=columns)
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

