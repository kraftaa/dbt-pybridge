import pytest

from dbt_pybridge.runner import RelationFrame


class DummySession:
    def __init__(self):
        self.loaded = False
        self.last_relation_sql = None

    def load_relation(self, relation_sql):
        self.loaded = True
        self.last_relation_sql = relation_sql
        return {"relation": relation_sql}

    def iter_relation_batches(self, relation_sql, batch_size=None):
        yield {"relation": relation_sql, "batch_size": batch_size}


def test_relation_frame_iter_batches_passthrough():
    session = DummySession()
    frame = RelationFrame(session, '"public"."orders"')

    batch = next(frame.iter_batches(batch_size=10))
    assert batch["batch_size"] == 10


def test_relation_frame_lazy_load():
    session = DummySession()
    frame = RelationFrame(session, '"public"."orders"')

    assert session.loaded is False
    repr(frame)
    assert session.loaded is True


def test_relation_frame_as_dataframe():
    session = DummySession()
    frame = RelationFrame(session, '"public"."orders"')
    assert frame.as_dataframe()["relation"] == '"public"."orders"'


def test_relation_frame_select_wraps_projection():
    session = DummySession()
    frame = RelationFrame(session, '"public"."orders"')

    selected = frame.select("id, amount")
    assert isinstance(selected, RelationFrame)
    selected.as_dataframe()
    assert session.last_relation_sql == '(select id, amount from "public"."orders") as pybridge_select'


def test_relation_frame_select_rejects_empty_projection():
    session = DummySession()
    frame = RelationFrame(session, '"public"."orders"')
    with pytest.raises(RuntimeError):
        frame.select("   ")


def test_relation_frame_select_rejects_semicolon():
    session = DummySession()
    frame = RelationFrame(session, '"public"."orders"')
    with pytest.raises(RuntimeError):
        frame.select("id; drop table x")


def test_relation_frame_where_wraps_predicate():
    session = DummySession()
    frame = RelationFrame(session, '"public"."orders"').where("amount > 100")
    frame.as_dataframe()
    assert (
        session.last_relation_sql
        == '(select * from "public"."orders" where amount > 100) as pybridge_where'
    )


def test_relation_frame_where_rejects_empty():
    session = DummySession()
    frame = RelationFrame(session, '"public"."orders"')
    with pytest.raises(RuntimeError):
        frame.where("")
    with pytest.raises(RuntimeError):
        frame.where("   ")


def test_relation_frame_where_rejects_semicolon():
    session = DummySession()
    frame = RelationFrame(session, '"public"."orders"')
    with pytest.raises(RuntimeError):
        frame.where("amount > 100; drop table orders")


def test_relation_frame_join_uses_using_clause():
    session = DummySession()
    a = RelationFrame(session, '"public"."orders"')
    b = RelationFrame(session, '"public"."customers"')
    joined = a.join(b, on="customer_id", how="left")
    joined.as_dataframe()
    assert (
        session.last_relation_sql
        == '(select * from (select * from "public"."orders") as pybridge_l '
           'left join (select * from "public"."customers") as pybridge_r using ("customer_id")'
           ') as pybridge_join'
    )


def test_relation_frame_join_after_select_does_not_double_alias():
    # Regression: chaining .select().join() used to emit
    # `... as pybridge_select as pybridge_l ...` which Postgres rejects.
    session = DummySession()
    a = RelationFrame(session, '"public"."orders"').select("id, customer_id")
    b = RelationFrame(session, '"public"."customers"')
    joined = a.join(b, on="customer_id", how="left")
    joined.as_dataframe()
    sql = session.last_relation_sql
    assert "as pybridge_select as pybridge_l" not in sql
    assert "as pybridge_select)" in sql  # the inner subquery alias survives
    assert "as pybridge_l" in sql


def test_relation_frame_cross_join_rejects_on_argument():
    session = DummySession()
    a = RelationFrame(session, '"public"."x"')
    b = RelationFrame(session, '"public"."y"')
    with pytest.raises(RuntimeError):
        a.join(b, on="id", how="cross")


def test_relation_frame_join_supports_multi_column_keys():
    session = DummySession()
    a = RelationFrame(session, '"public"."orders"')
    b = RelationFrame(session, '"public"."shipments"')
    joined = a.join(b, on=["customer_id", "order_id"], how="inner")
    joined.as_dataframe()
    assert 'using ("customer_id", "order_id")' in session.last_relation_sql
    assert 'inner join' in session.last_relation_sql


def test_relation_frame_join_cross_skips_using():
    session = DummySession()
    a = RelationFrame(session, '"public"."orders"')
    b = RelationFrame(session, '"public"."dim_dates"')
    joined = a.join(b, on=None, how="cross")
    joined.as_dataframe()
    assert 'cross join' in session.last_relation_sql
    assert 'using' not in session.last_relation_sql


def test_relation_frame_join_rejects_invalid_how():
    session = DummySession()
    a = RelationFrame(session, '"public"."x"')
    b = RelationFrame(session, '"public"."y"')
    with pytest.raises(RuntimeError):
        a.join(b, on="id", how="banana")


def test_relation_frame_join_rejects_non_relation_frame():
    session = DummySession()
    a = RelationFrame(session, '"public"."x"')
    with pytest.raises(RuntimeError):
        a.join({"foo": "bar"}, on="id")


def test_relation_frame_join_rejects_cross_connection_with_local_join_guidance():
    left_session = DummySession()
    left_session.connection_name = "app_db"
    right_session = DummySession()
    right_session.connection_name = "billing_db"

    left = RelationFrame(left_session, '"public"."customers"')
    right = RelationFrame(right_session, '"public"."orders"')

    with pytest.raises(RuntimeError) as exc:
        left.join(right, on="customer_id")

    message = str(exc.value)
    assert "app_db" in message
    assert "billing_db" in message
    assert ".as_dataframe()" in message
    assert "engine='duckdb'" in message


def test_relation_frame_cross_connection_duckdb_join_is_explicit():
    left_session = DummySession()
    left_session.connection_name = "app_db"
    left_session.dataframe_backend = "pandas"
    right_session = DummySession()
    right_session.connection_name = "billing_db"

    left = RelationFrame(left_session, '"public"."customers"')
    right = RelationFrame(right_session, '"public"."orders"')
    joined = left.join(
        right, on="customer_id", how="left", engine="duckdb", memory_limit="256MB"
    )

    from dbt_pybridge.runner import FederatedJoinFrame
    assert isinstance(joined, FederatedJoinFrame)


def test_relation_frame_duckdb_join_validates_memory_limit():
    left_session = DummySession()
    right_session = DummySession()
    left = RelationFrame(left_session, '"public"."a"')
    right = RelationFrame(right_session, '"public"."b"')

    with pytest.raises(RuntimeError):
        left.join(
            right,
            on="id",
            engine="duckdb",
            memory_limit="1GB; drop table x",
        )


def test_relation_frame_duckdb_join_executes_and_batches():
    pytest.importorskip("duckdb")
    import pandas as pd
    from dbt_pybridge.session import ModelLimits

    class FrameSession:
        def __init__(self, rows, connection_name):
            self.rows = rows
            self.connection_name = connection_name
            self.dataframe_backend = "pandas"
            self.limits = ModelLimits(batch_size=1)

        def iter_relation_batches(self, relation_sql, batch_size=None, as_pandas=False):
            size = batch_size or self.limits.batch_size
            for offset in range(0, len(self.rows), size):
                yield pd.DataFrame(self.rows[offset : offset + size])

        def relation_columns(self, relation_sql):
            return [
                (name, 25 if isinstance(value, str) else 20, None, None)
                for name, value in self.rows[0].items()
            ]

    customers = RelationFrame(
        FrameSession(
            [
                {"customer_id": 1, "customer_name": "Ada"},
                {"customer_id": 2, "customer_name": "Lin"},
            ],
            "app_db",
        ),
        '"public"."customers"',
    )
    orders = RelationFrame(
        FrameSession(
            [
                {"order_id": 10, "customer_id": 1},
                {"order_id": 11, "customer_id": 2},
            ],
            "billing_db",
        ),
        '"public"."orders"',
    )

    joined = customers.join(orders, on="customer_id", engine="duckdb")
    batches = list(joined.iter_batches(batch_size=1))

    assert sum(len(batch) for batch in batches) == 2
    assert set(batches[0].columns) == {"customer_id", "customer_name", "order_id"}


def test_relation_frame_join_rejects_dangerous_keys():
    session = DummySession()
    a = RelationFrame(session, '"public"."x"')
    b = RelationFrame(session, '"public"."y"')
    with pytest.raises(RuntimeError):
        a.join(b, on='id"; drop table x; --')


def test_load_df_function_normalizes_three_part_relation():
    # Regression: dbt may render a 3-part identifier ("db"."schema"."t").
    # Without normalization at the entry point, .select()/.where()/.join()
    # would wrap it in a subquery and Postgres would reject the cross-db
    # qualifier sitting inside.
    from dbt_pybridge.runner import LocalPythonModelRunner
    from dbt_pybridge.session import LocalPostgresSession, ModelLimits

    session = LocalPostgresSession.__new__(LocalPostgresSession)
    session.credentials = type("C", (), {"database": "demo_db"})()
    session.limits = ModelLimits()
    session.dataframe_backend = "pandas"

    load = LocalPythonModelRunner._load_df_function(session)
    rf = load('"demo_db"."transform"."orders"')
    assert rf._relation_sql == '"transform"."orders"'

    # The same load is the only place we need to normalize; downstream
    # .select() must NOT re-introduce the database qualifier.
    selected = rf.select("id, customer_id")
    assert '"demo_db"' not in selected._relation_sql


def test_load_df_function_routes_named_connection():
    from dbt_pybridge.runner import LocalPythonModelRunner
    from dbt_pybridge.session import LocalPostgresSession, ModelLimits, PostgresSessionRegistry

    target_session = LocalPostgresSession.__new__(LocalPostgresSession)
    target_session.credentials = type("C", (), {"database": "analytics"})()
    target_session.limits = ModelLimits()
    target_session.dataframe_backend = "pandas"
    target_session.connection_name = "target"

    app_session = LocalPostgresSession.__new__(LocalPostgresSession)
    app_session.credentials = type("C", (), {"database": "application"})()
    app_session.limits = ModelLimits()
    app_session.dataframe_backend = "pandas"
    app_session.connection_name = "app_db"

    registry = PostgresSessionRegistry.__new__(PostgresSessionRegistry)
    registry.get = lambda name=None: app_session if name == "app_db" else target_session

    load = LocalPythonModelRunner._load_df_function(registry)
    frame = load('"application"."public"."customers"', connection_name="app_db")

    assert frame._session is app_session
    assert frame._relation_sql == '"public"."customers"'


def test_target_relation_rejects_database_other_than_active_target():
    from dbt_pybridge.runner import LocalPythonModelRunner

    runner = LocalPythonModelRunner(
        credentials=type("C", (), {"database": "analytics"})(),
        parsed_model={
            "database": "other_database",
            "schema": "transform",
            "name": "orders",
        },
        compiled_code="",
    )

    with pytest.raises(RuntimeError) as exc:
        runner._target_relation()

    assert "does not match" in str(exc.value)


class TypedFrameSession:
    """Fake source session that reports Postgres column types like psycopg2."""

    def __init__(self, connection_name, columns, batches, limits=None):
        from dbt_pybridge.session import ModelLimits

        self.connection_name = connection_name
        self.dataframe_backend = "pandas"
        self.limits = limits or ModelLimits(batch_size=1)
        self._columns = columns
        self._batches = batches

    def relation_columns(self, relation_sql):
        return self._columns

    def iter_relation_batches(self, relation_sql, batch_size=None, as_pandas=False):
        import pandas as pd

        names = [column[0] for column in self._columns]
        for rows in self._batches:
            yield pd.DataFrame(rows, columns=names, dtype=object)


def _typed_join(left_columns, left_batches, right_columns, right_batches, limits=None, **kwargs):
    left = RelationFrame(
        TypedFrameSession("app_db", left_columns, left_batches, limits), '"public"."l"'
    )
    right = RelationFrame(
        TypedFrameSession("billing_db", right_columns, right_batches, limits), '"public"."r"'
    )
    return left.join(right, on="id", how="left", engine="duckdb", **kwargs)


def test_duckdb_join_keeps_declared_types_across_batches():
    pytest.importorskip("duckdb")
    import datetime as dt
    from decimal import Decimal

    when = dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
    joined = _typed_join(
        [("id", 20, None, None), ("amount", 1700, 12, 2), ("exact", 1700, 65535, 65535)],
        # The first batch has whole numbers and NULLs only; inferring types
        # from it used to round 1.55 to 2 and reject later values.
        [[(1, Decimal("1"), None)], [(2, Decimal("1.55"), Decimal("123456789012345678901234567890.123"))]],
        [("id", 20, None, None), ("payload", 3802, None, None), ("seen_at", 1184, None, None)],
        [[(1, None, None)], [(2, {"a": [1, 2]}, when)]],
    )

    df = joined.as_dataframe().sort_values("id").reset_index(drop=True)

    assert df.loc[1, "amount"] == Decimal("1.55")
    assert df.loc[1, "exact"] == Decimal("123456789012345678901234567890.123")
    assert df.loc[1, "payload"] == {"a": [1, 2]}
    assert df.loc[1, "seen_at"] == when
    assert df.loc[0, "payload"] is None


def test_duckdb_join_rejects_overlapping_non_key_columns():
    pytest.importorskip("duckdb")
    joined = _typed_join(
        [("id", 20, None, None), ("name", 25, None, None)],
        [[(1, "a")]],
        [("id", 20, None, None), ("name", 25, None, None)],
        [[(1, "x")]],
    )

    with pytest.raises(RuntimeError, match="exist on both sides"):
        joined.as_dataframe()


def test_duckdb_join_rejects_text_staged_join_key():
    pytest.importorskip("duckdb")
    joined = _typed_join(
        [("id", 1700, 65535, 65535)], [[(1,)]], [("id", 1700, 65535, 65535)], [[(1,)]]
    )

    with pytest.raises(RuntimeError, match="compare by spelling"):
        joined.as_dataframe()


def test_duckdb_join_rejects_unsupported_types():
    pytest.importorskip("duckdb")
    joined = _typed_join(
        [("id", 20, None, None), ("shape", 600, None, None)],
        [[(1, "(1,2)")]],
        [("id", 20, None, None)],
        [[(1,)]],
    )

    with pytest.raises(RuntimeError, match="shape::text"):
        joined.as_dataframe()


def test_duckdb_eager_join_result_obeys_row_limit():
    pytest.importorskip("duckdb")
    from dbt_pybridge.session import ModelLimits

    limits = ModelLimits(max_rows=3, batch_size=2)
    joined = _typed_join(
        [("id", 20, None, None)],
        [[(1,), (1,)]],
        [("id", 20, None, None), ("v", 20, None, None)],
        [[(1, 1), (1, 2)]],
        limits=limits,
    )

    with pytest.raises(RuntimeError, match="more than 3 rows"):
        joined.as_dataframe()

    # Streaming the same 4-row result is still allowed.
    assert sum(len(batch) for batch in joined.iter_batches()) == 4


def test_duckdb_join_validates_resource_settings():
    left = RelationFrame(DummySession(), '"public"."a"')
    right = RelationFrame(DummySession(), '"public"."b"')

    with pytest.raises(RuntimeError, match="threads"):
        left.join(right, on="id", engine="duckdb", threads=0)
    with pytest.raises(RuntimeError, match="max_temp_directory_size"):
        left.join(right, on="id", engine="duckdb", max_temp_directory_size="lots")


@pytest.mark.parametrize(
    "how, filter_by, message",
    [
        ("full", "left", "would drop rows"),
        ("right", "left", "would drop rows"),
        ("left", "right", "would drop rows"),
        ("inner", "middle", "must be 'left' or 'right'"),
    ],
)
def test_duckdb_join_filter_by_only_where_it_cannot_drop_rows(how, filter_by, message):
    left = RelationFrame(DummySession(), '"public"."a"')
    right = RelationFrame(DummySession(), '"public"."b"')

    with pytest.raises(RuntimeError, match=message):
        left.join(right, on="id", how=how, engine="duckdb", filter_by=filter_by)


def test_duckdb_join_filter_by_needs_a_single_key():
    left = RelationFrame(DummySession(), '"public"."a"')
    right = RelationFrame(DummySession(), '"public"."b"')

    with pytest.raises(RuntimeError, match="single join key"):
        left.join(right, on=["a", "b"], engine="duckdb", filter_by="left")


def test_duckdb_join_regroups_large_results_into_exact_batches():
    pytest.importorskip("duckdb")
    rows = 5_000  # spans several DuckDB vectors
    joined = _typed_join(
        [("id", 20, None, None)],
        [[(i,) for i in range(rows)]],
        [("id", 20, None, None), ("v", 20, None, None)],
        [[(i, i * 2) for i in range(rows)]],
    )

    sizes = [len(batch) for batch in joined.iter_batches(batch_size=1_500)]

    assert sizes == [1_500, 1_500, 1_500, 500]
