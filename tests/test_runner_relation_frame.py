from dbt_pybridge.runner import RelationFrame


class DummySession:
    def __init__(self):
        self.loaded = False

    def load_relation(self, relation_sql):
        self.loaded = True
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
