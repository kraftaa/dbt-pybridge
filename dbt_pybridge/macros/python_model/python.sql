{% macro build_ref_function(model) %}

    {%- set ref_dict = {} -%}
    {%- for _ref in model.refs -%}
        {% set _ref_args = [_ref.get('package'), _ref['name']] if _ref.get('package') else [_ref['name'],] %}
        {%- set resolved = ref(*_ref_args, v=_ref.get('version')) -%}

        {%- if resolved.render is defined and resolved.render is callable -%}
            {%- set resolved = resolved.render() -%}
        {%- endif -%}

        {%- if _ref.get('version') -%}
            {% do _ref_args.extend(["v" ~ _ref['version']]) %}
        {%- endif -%}
        {%- do ref_dict.update({_ref_args | join('.'): resolve_model_name(resolved)}) -%}
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
    return dbt_load_df_function(refs[key])

{% endmacro %}


{% macro build_source_function(model) %}

    {%- set source_dict = {} -%}
    {%- for _source in model.sources -%}
        {%- set resolved = source(*_source) -%}
        {%- do source_dict.update({_source | join('.'): resolve_model_name(resolved)}) -%}
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

    return dbt_load_df_function(sources[key])

{% endmacro %}
