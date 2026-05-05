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
