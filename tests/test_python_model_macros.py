from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment
import pytest


MACRO_PATH = (
    Path(__file__).parents[1]
    / "dbt_pybridge"
    / "macros"
    / "python_model"
    / "python.sql"
)


class FakeRelation:
    def __init__(self, rendered):
        self._rendered = rendered

    def render(self):
        return self._rendered


def _macro_module(*, nodes=None, sources=None, relation_name):
    def raise_compiler_error(message):
        raise RuntimeError(message)

    environment = Environment(extensions=["jinja2.ext.do"])
    environment.globals.update(
        {
            "graph": SimpleNamespace(nodes=nodes or {}, sources=sources or {}),
            "ref": lambda *args, **kwargs: FakeRelation(relation_name),
            "source": lambda *args: FakeRelation(relation_name),
            "resolve_model_name": lambda value: str(value),
            "return": lambda value: value,
            "exceptions": SimpleNamespace(raise_compiler_error=raise_compiler_error),
        }
    )
    return environment.from_string(MACRO_PATH.read_text()).module


def test_build_source_function_compiles_connection_route():
    relation_name = '"application"."public"."customers"'
    source_node = SimpleNamespace(
        relation_name=relation_name,
        source_name="app",
        name="customers",
        config=SimpleNamespace(meta={"pybridge_connection": "app_db"}),
    )
    module = _macro_module(
        sources={"source.synthetic.app.customers": source_node},
        relation_name=relation_name,
    )

    code = module.build_source_function(
        {
            "name": "cross_customer_orders",
            "sources": [["app", "customers"]],
            "depends_on": SimpleNamespace(nodes=["source.synthetic.app.customers"]),
        }
    )
    namespace = {}
    exec(code, namespace)
    calls = []
    namespace["source"](
        "app",
        "customers",
        dbt_load_df_function=lambda relation_sql, connection_name=None: calls.append(
            (relation_sql, connection_name)
        ),
    )

    assert calls == [(relation_name, "app_db")]
    assert namespace["__pybridge_source_connections__"] == ["app_db"]


def test_build_ref_function_rejects_named_connection_route():
    relation_name = '"billing"."transform"."orders"'
    ref_node = SimpleNamespace(
        relation_name=relation_name,
        name="orders",
        package_name="synthetic",
        version=None,
        config=SimpleNamespace(meta={"pybridge_connection": "billing_db"}),
    )
    module = _macro_module(
        nodes={"model.synthetic.orders": ref_node},
        relation_name=relation_name,
    )

    with pytest.raises(RuntimeError) as exc:
        module.build_ref_function(
            {
                "name": "cross_customer_orders",
                "refs": [{"package": None, "name": "orders", "version": None}],
                "depends_on": SimpleNamespace(nodes=["model.synthetic.orders"]),
            }
        )

    assert "supported only on dbt sources" in str(exc.value)


def test_unrouted_ref_and_source_compile_to_valid_python():
    # Regression: an unrouted connection rendered as JSON `null`, so every
    # ref() and unrouted source() raised NameError under real dbt.
    relation_name = '"analytics"."transform"."orders"'
    ref_node = SimpleNamespace(
        relation_name=relation_name,
        name="orders",
        package_name="synthetic",
        version=None,
        config=SimpleNamespace(meta={}),
    )
    source_node = SimpleNamespace(
        relation_name=relation_name,
        source_name="raw",
        name="events",
        config=SimpleNamespace(meta={}),
    )
    module = _macro_module(
        nodes={"model.synthetic.orders": ref_node},
        sources={"source.synthetic.raw.events": source_node},
        relation_name=relation_name,
    )
    model = {
        "name": "downstream",
        "refs": [{"name": "orders"}],
        "sources": [["raw", "events"]],
        "depends_on": SimpleNamespace(
            nodes=["model.synthetic.orders", "source.synthetic.raw.events"]
        ),
    }
    namespace = {}
    exec(module.build_ref_function(model) + module.build_source_function(model), namespace)

    calls = []

    def load(relation_sql, connection_name=None):
        calls.append((relation_sql, connection_name))

    namespace["ref"]("orders", dbt_load_df_function=load)
    namespace["source"]("raw", "events", dbt_load_df_function=load)

    assert calls == [(relation_name, None), (relation_name, None)]
    assert namespace["__pybridge_source_connections__"] == []
