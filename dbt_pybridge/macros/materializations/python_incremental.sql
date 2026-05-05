{% materialization incremental, adapter='pybridge', supported_languages=['sql', 'python'] -%}

  {%- set existing_relation = load_cached_relation(this) -%}
  {%- set target_relation = this.incorporate(type='table') -%}
  {%- set temp_relation = make_temp_relation(target_relation)-%}
  {%- set intermediate_relation = make_intermediate_relation(target_relation)-%}
  {%- set backup_relation_type = 'table' if existing_relation is none else existing_relation.type -%}
  {%- set backup_relation = make_backup_relation(target_relation, backup_relation_type) -%}

  {%- set unique_key = config.get('unique_key') -%}
  {%- set full_refresh_mode = (should_full_refresh() or existing_relation.is_view) -%}
  {%- set on_schema_change = incremental_validate_on_schema_change(config.get('on_schema_change'), default='ignore') -%}
  {%- set is_python_model = model['language'] == 'python' -%}

  {%- set preexisting_intermediate_relation = load_cached_relation(intermediate_relation)-%}
  {%- set preexisting_backup_relation = load_cached_relation(backup_relation) -%}
  {% set grant_config = config.get('grants') %}
  {{ drop_relation_if_exists(preexisting_intermediate_relation) }}
  {{ drop_relation_if_exists(preexisting_backup_relation) }}

  {{ run_hooks(pre_hooks, inside_transaction=False) }}
  {% if not is_python_model %}
    {{ run_hooks(pre_hooks, inside_transaction=True) }}
  {% endif %}

  {% set to_drop = [] %}

  {% if is_python_model %}
    {%- set incremental_strategy = config.get('incremental_strategy') or 'default' -%}

    {% if full_refresh_mode and existing_relation is not none %}
      {{ drop_relation_if_exists(existing_relation) }}
      {% do adapter.commit() %}
      {% set existing_relation = none %}
    {% endif %}

    {% if existing_relation is not none and not existing_relation.is_table %}
      {{ drop_relation_if_exists(existing_relation) }}
      {% set existing_relation = none %}
    {% endif %}

    {% call statement('main', language='python') -%}
      {{ compiled_code }}
    {%- endcall %}

    {% if existing_relation is none or full_refresh_mode %}
      {% do create_indexes(target_relation) %}
    {% endif %}
  {% else %}
    {% set incremental_strategy = config.get('incremental_strategy') or 'default' %}
    {% set strategy_sql_macro_func = adapter.get_incremental_strategy_macro(context, incremental_strategy) %}

    {% if existing_relation is none %}
        {% set build_sql = get_create_table_as_sql(False, target_relation, sql) %}
        {% set relation_for_indexes = target_relation %}
    {% elif full_refresh_mode %}
        {% set build_sql = get_create_table_as_sql(False, intermediate_relation, sql) %}
        {% set relation_for_indexes = intermediate_relation %}
        {% set need_swap = true %}
    {% else %}
      {% do run_query(get_create_table_as_sql(True, temp_relation, sql)) %}
      {% set relation_for_indexes = temp_relation %}
      {% set contract_config = config.get('contract') %}
      {% if not contract_config or not contract_config.enforced %}
        {% do adapter.expand_target_column_types(
                 from_relation=temp_relation,
                 to_relation=target_relation) %}
      {% endif %}
      {% set dest_columns = process_schema_changes(on_schema_change, temp_relation, existing_relation) %}
      {% if not dest_columns %}
        {% set dest_columns = adapter.get_columns_in_relation(existing_relation) %}
      {% endif %}

      {% set incremental_predicates = config.get('predicates', none) or config.get('incremental_predicates', none) %}
      {% set strategy_arg_dict = ({'target_relation': target_relation, 'temp_relation': temp_relation, 'unique_key': unique_key, 'dest_columns': dest_columns, 'incremental_predicates': incremental_predicates }) %}
      {% set build_sql = strategy_sql_macro_func(strategy_arg_dict) %}

    {% endif %}

    {% call statement("main") %}
        {{ build_sql }}
    {% endcall %}

    {% if existing_relation is none or existing_relation.is_view or should_full_refresh() %}
      {% do create_indexes(relation_for_indexes) %}
    {% endif %}

    {% if need_swap %}
        {% do adapter.rename_relation(target_relation, backup_relation) %}
        {% do adapter.rename_relation(intermediate_relation, target_relation) %}
        {% do to_drop.append(backup_relation) %}
    {% endif %}
  {% endif %}

  {% set should_revoke = should_revoke(existing_relation, full_refresh_mode) %}
  {% do apply_grants(target_relation, grant_config, should_revoke=should_revoke) %}
  {% do persist_docs(target_relation, model) %}

  {% if not is_python_model %}
    {{ run_hooks(post_hooks, inside_transaction=True) }}
    {% do adapter.commit() %}
  {% endif %}

  {% for rel in to_drop %}
      {% do adapter.drop_relation(rel) %}
  {% endfor %}

  {{ run_hooks(post_hooks, inside_transaction=False) }}

  {{ return({'relations': [target_relation]}) }}

{%- endmaterialization %}
