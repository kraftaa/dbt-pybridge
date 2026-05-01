{%- materialization view, adapter='pybridge', supported_languages=['sql', 'python'] -%}

  {%- set existing_relation = load_cached_relation(this) -%}
  {%- set target_relation = this.incorporate(type='view') -%}
  {%- set is_python_model = model['language'] == 'python' -%}
  {%- set intermediate_relation =  make_intermediate_relation(target_relation) -%}
  {%- set preexisting_intermediate_relation = load_cached_relation(intermediate_relation) -%}
  {%- set backup_relation_type = 'view' if existing_relation is none else existing_relation.type -%}
  {%- set backup_relation = make_backup_relation(target_relation, backup_relation_type) -%}
  {%- set preexisting_backup_relation = load_cached_relation(backup_relation) -%}
  {% set grant_config = config.get('grants') %}

  {{ run_hooks(pre_hooks, inside_transaction=False) }}
  {{ drop_relation_if_exists(preexisting_intermediate_relation) }}
  {{ drop_relation_if_exists(preexisting_backup_relation) }}
  {% if not is_python_model %}
    {{ run_hooks(pre_hooks, inside_transaction=True) }}
  {% endif %}

  {% if is_python_model %}
    {% call statement('main', language='python') -%}
      {{ compiled_code }}
    {%- endcall %}
  {% else %}
    {% call statement('main') -%}
      {{ get_create_view_as_sql(intermediate_relation, sql) }}
    {%- endcall %}

    {% if existing_relation is not none %}
      {% set existing_relation = load_cached_relation(existing_relation) %}
      {% if existing_relation is not none %}
        {{ adapter.rename_relation(existing_relation, backup_relation) }}
      {% endif %}
    {% endif %}
    {{ adapter.rename_relation(intermediate_relation, target_relation) }}
  {% endif %}

  {% set should_revoke = should_revoke(existing_relation, full_refresh_mode=True) %}
  {% do apply_grants(target_relation, grant_config, should_revoke=should_revoke) %}
  {% do persist_docs(target_relation, model) %}

  {% if not is_python_model %}
    {{ run_hooks(post_hooks, inside_transaction=True) }}
    {{ adapter.commit() }}
    {{ drop_relation_if_exists(backup_relation) }}
  {% endif %}
  {{ run_hooks(post_hooks, inside_transaction=False) }}

  {{ return({'relations': [target_relation]}) }}
{%- endmaterialization -%}
