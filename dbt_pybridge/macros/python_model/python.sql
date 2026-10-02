{% macro build_ref_function(model) %}

    {%- set ref_dict = {} -%}
    {%- for _ref in model.refs -%}
        {% set _ref_args = [_ref.get('package'), _ref['name']] if _ref.get('package') else [_ref['name'],] %}
        {%- set resolved = ref(*_ref_args, v=_ref.get('version')) -%}

        {%- if resolved.render is defined and resolved.render is callable -%}
            {%- set resolved = resolved.render() -%}
        {%- endif -%}

        {%- set resolved_name = resolve_model_name(resolved) -%}
        {%- set route = namespace(connection=none) -%}
        {%- for _unique_id in model.depends_on.nodes -%}
            {%- set _node = graph.nodes.get(_unique_id) -%}
            {%- if _node is not none
                and _node.name == _ref['name']
                and (not _ref.get('package') or _node.package_name == _ref.get('package'))
                and (not _ref.get('version') or _node.version == _ref.get('version'))
                and _node.config is defined
                and _node.config.meta is defined
                and _node.config.meta -%}
                {%- set route.connection = _node.config.meta.get('pybridge_connection') -%}
            {%- endif -%}
        {%- endfor -%}
        {%- if route.connection is not none -%}
            {% do exceptions.raise_compiler_error(
                "pybridge_connection is supported only on dbt sources, not ref() nodes. "
                ~ "Build upstream models with the appropriate dbt target, then expose "
                ~ "the result as a routed source."
            ) %}
        {%- endif -%}

        {%- if _ref.get('version') -%}
            {% do _ref_args.extend(["v" ~ _ref['version']]) %}
        {%- endif -%}
        {#- tojson renders none as `null`, which is not Python; use "" for the target. -#}
        {%- do ref_dict.update({_ref_args | join('.'): {
            'relation': resolved_name,
            'connection': route.connection or ''
        }}) -%}
    {%- endfor -%}


def ref(*args, **kwargs):
    refs = {{ ref_dict | tojson }}
    key = '.'.join(args)
    version = kwargs.get("v") or kwargs.get("version")
    if version:
        key += f".v{version}"

    if key not in refs:
        available = sorted(refs.keys())
        raise RuntimeError(
            "Missing dbt.ref mapping for key "
            f"'{key}' in python model '{{ model['name'] }}'. "
            "dbt's static parser may miss chained ref expressions. "
            "Use a standalone assignment like: x = dbt.ref('model_name'). "
            f"Available refs in this model: {available}"
        )

    dbt_load_df_function = kwargs.get("dbt_load_df_function")
    return dbt_load_df_function(
        refs[key]["relation"],
        connection_name=refs[key]["connection"] or None,
    )

{% endmacro %}


{% macro build_source_function(model) %}

    {%- set source_dict = {} -%}
    {%- for _source in model.sources -%}
        {%- set resolved = source(*_source) -%}
        {%- if resolved.render is defined and resolved.render is callable -%}
            {%- set resolved = resolved.render() -%}
        {%- endif -%}
        {%- set resolved_name = resolve_model_name(resolved) -%}
        {%- set route = namespace(connection=none) -%}
        {%- for _unique_id in model.depends_on.nodes -%}
            {%- set _node = graph.sources.get(_unique_id) -%}
            {%- if _node is not none
                and _node.source_name == _source[0]
                and _node.name == _source[1]
                and _node.config is defined
                and _node.config.meta is defined
                and _node.config.meta -%}
                {%- set route.connection = _node.config.meta.get('pybridge_connection') -%}
            {%- endif -%}
        {%- endfor -%}
        {%- do source_dict.update({_source | join('.'): {
            'relation': resolved_name,
            'connection': route.connection or ''
        }}) -%}
    {%- endfor -%}


def source(*args, dbt_load_df_function):
    sources = {{ source_dict | tojson }}
    key = '.'.join(args)

    if key not in sources:
        available = sorted(sources.keys())
        raise RuntimeError(
            "Missing dbt.source mapping for key "
            f"'{key}' in python model '{{ model['name'] }}'. "
            f"Available sources in this model: {available}"
        )

    return dbt_load_df_function(
        sources[key]["relation"],
        connection_name=sources[key]["connection"] or None,
    )

{%- set source_connections = [] -%}
{%- for _route in source_dict.values() -%}
    {%- if _route['connection'] and _route['connection'] not in source_connections -%}
        {%- do source_connections.append(_route['connection']) -%}
    {%- endif -%}
{%- endfor %}

# Read by the PyBridge runner to start every source snapshot at model start.
__pybridge_source_connections__ = {{ source_connections | sort | tojson }}

{% endmacro %}
