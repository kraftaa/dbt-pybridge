from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from dbt_pybridge.dataframe_io import write_model_result
from dbt_pybridge.session import LocalPostgresSession, ModelLimits, TargetRelation, quote_ident


class RelationFrame:
    """Lazy dataframe wrapper that supports both eager dataframe use and iter_batches()."""

    def __init__(self, session: LocalPostgresSession, relation_sql: str) -> None:
        self._session = session
        self._relation_sql = relation_sql
        self._df = None

    def _load(self):
        if self._df is None:
            self._df = self._session.load_relation(self._relation_sql)
        return self._df

    def iter_batches(self, batch_size: Optional[int] = None):
        return self._session.iter_relation_batches(self._relation_sql, batch_size=batch_size)

    def __getattr__(self, item):
        return getattr(self._load(), item)

    def __getitem__(self, key):
        return self._load()[key]

    def __setitem__(self, key, value):
        self._load()[key] = value

    def __len__(self):
        return len(self._load())

    def __iter__(self):
        return iter(self._load())

    def __contains__(self, item):
        return item in self._load()

    def __repr__(self) -> str:
        return repr(self._load())

    def as_dataframe(self):
        return self._load()


class LocalPythonModelRunner:
    def __init__(self, credentials, parsed_model: Dict[str, Any], compiled_code: str) -> None:
        self.credentials = credentials
        self.parsed_model = parsed_model
        self.compiled_code = compiled_code

    def _model_config(self) -> Dict[str, Any]:
        raw_config = self.parsed_model.get("config", {})
        # dbt model config object behaves like mapping.
        return dict(raw_config)

    def _limits(self, cfg: Dict[str, Any]) -> ModelLimits:
        def _as_int(key: str, default: int) -> int:
            value = cfg.get(key, default)
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                raise RuntimeError(f"Invalid value for {key}: expected integer, got {value!r}") from None
            if parsed < 0:
                raise RuntimeError(f"Invalid value for {key}: expected >= 0, got {parsed}")
            return parsed

        def _as_bool(key: str, default: bool) -> bool:
            value = cfg.get(key, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, int) and value in (0, 1):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "t", "yes", "y", "on", "1"}:
                    return True
                if normalized in {"false", "f", "no", "n", "off", "0"}:
                    return False
            raise RuntimeError(
                f"Invalid value for {key}: expected boolean-like value, got {value!r}"
            )

        return ModelLimits(
            max_rows=_as_int("localpy_max_rows", 1_000_000),
            warn_rows=_as_int("localpy_warn_rows", 200_000),
            max_bytes=_as_int("localpy_max_bytes", 512 * 1024 * 1024),
            warn_bytes=_as_int("localpy_warn_bytes", 128 * 1024 * 1024),
            batch_size=_as_int("localpy_batch_size", 100_000),
            allow_large_tables=_as_bool("localpy_allow_large_tables", False),
            chunked_mode=_as_bool("localpy_chunked_mode", False),
        )

    def _column_types(self, cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
        return self._type_mapping_config(cfg, "localpy_column_types")

    def _categorical_types(self, cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
        return self._type_mapping_config(cfg, "localpy_categorical_types")

    def _type_mapping_config(self, cfg: Dict[str, Any], key: str) -> Optional[Dict[str, str]]:
        raw = cfg.get(key)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                f"Invalid {key} config: expected a dict of "
                "{column_name: postgres_type_sql}"
            )
        out: Dict[str, str] = {}
        for raw_col, raw_type in raw.items():
            col = str(raw_col).strip()
            pg_type = str(raw_type).strip()
            if not col or not pg_type:
                raise RuntimeError(
                    f"Invalid {key} config: column name and type must be non-empty strings."
                )
            out[col] = pg_type
        return out

    def _target_relation(self) -> TargetRelation:
        return TargetRelation(
            database=self.parsed_model.get("database"),
            schema=self.parsed_model.get("schema"),
            identifier=self.parsed_model.get("alias") or self.parsed_model.get("name"),
        )

    def _normalize_unique_key(self, unique_key: Any) -> Optional[List[str]]:
        if unique_key is None:
            return None
        if isinstance(unique_key, str):
            return [unique_key]
        if isinstance(unique_key, (list, tuple)):
            keys = [str(v) for v in unique_key]
            if not keys:
                return None
            return keys
        raise RuntimeError(
            "Invalid unique_key for incremental Python model. "
            f"Expected string or list of strings, got {type(unique_key)!r}"
        )

    def _view_backing_relation(self, view_target: TargetRelation) -> TargetRelation:
        prefix = "__dbt_pybridge_view_"
        raw = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in view_target.identifier.lower())
        suffix = raw or "model"
        digest = hashlib.sha1(view_target.identifier.encode("utf-8")).hexdigest()[:8]
        max_suffix_len = max(1, 63 - len(prefix) - 1 - len(digest))
        identifier = f"{prefix}{suffix[:max_suffix_len]}_{digest}"
        return TargetRelation(
            database=view_target.database,
            schema=view_target.schema,
            identifier=identifier,
        )

    def _relation_kind(self, conn, relation: TargetRelation) -> Optional[str]:
        with conn.cursor() as cur:
            if relation.schema:
                cur.execute(
                    """
                    select c.relkind
                    from pg_catalog.pg_class c
                    join pg_catalog.pg_namespace n on n.oid = c.relnamespace
                    where n.nspname = %s and c.relname = %s
                    limit 1
                    """,
                    (relation.schema, relation.identifier),
                )
            else:
                cur.execute(
                    """
                    select c.relkind
                    from pg_catalog.pg_class c
                    where c.relname = %s and pg_table_is_visible(c.oid)
                    limit 1
                    """,
                    (relation.identifier,),
                )
            row = cur.fetchone()
            return row[0] if row else None

    def _drop_existing_relation(self, conn, relation: TargetRelation) -> None:
        relkind = self._relation_kind(conn, relation)
        if relkind is None:
            return

        relation_sql = relation.render()
        # CASCADE matches dbt-core's postgres__drop_relation convention: a
        # downstream view (or chain of views) shouldn't block a model rebuild,
        # since dbt rebuilds dependents on the next run anyway.
        drop_sql = {
            "r": f"drop table if exists {relation_sql} cascade",
            "p": f"drop table if exists {relation_sql} cascade",
            "v": f"drop view if exists {relation_sql} cascade",
            "m": f"drop materialized view if exists {relation_sql} cascade",
            "f": f"drop foreign table if exists {relation_sql} cascade",
        }.get(relkind, f"drop table if exists {relation_sql} cascade")

        with conn.cursor() as cur:
            cur.execute(drop_sql)
        conn.commit()

    def _create_or_replace_view(self, conn, view_target: TargetRelation, backing_table: TargetRelation) -> None:
        # Caller is responsible for dropping any pre-existing relation.
        view_sql = view_target.render()
        backing_sql = backing_table.render()
        with conn.cursor() as cur:
            if view_target.schema:
                cur.execute(f"create schema if not exists {quote_ident(view_target.schema)}")
            cur.execute(f"create view {view_sql} as select * from {backing_sql}")
        conn.commit()

    def run(self) -> int:
        cfg = self._model_config()
        dataframe_backend = str(cfg.get("localpy_dataframe_backend", "pandas")).lower()
        limits = self._limits(cfg)
        materialized = str(cfg.get("materialized", "table")).lower()
        unique_key = self._normalize_unique_key(cfg.get("unique_key"))
        column_types = self._column_types(cfg)
        categorical_types = self._categorical_types(cfg)
        incremental_strategy = str(cfg.get("incremental_strategy", "default")).lower()
        if incremental_strategy == "default":
            incremental_strategy = "merge" if unique_key else "append"

        session = LocalPostgresSession(
            credentials=self.credentials,
            limits=limits,
            dataframe_backend=dataframe_backend,
        )

        try:
            namespace: Dict[str, Any] = {}
            exec(self.compiled_code, namespace)

            model_fn = namespace.get("model")
            if model_fn is None:
                raise RuntimeError("Python model file must define model(dbt, session)")
            if not callable(model_fn):
                raise RuntimeError("Python model symbol 'model' must be callable")

            dbt_obj_cls = namespace.get("dbtObj")
            if dbt_obj_cls is None:
                raise RuntimeError("Compiled Python model is missing dbtObj from dbt py_script_postfix")

            def load_df_function(relation_sql: str):
                return RelationFrame(session, relation_sql)

            dbt_obj = dbt_obj_cls(load_df_function)
            model_result = model_fn(dbt_obj, session)
            if isinstance(model_result, RelationFrame):
                model_result = model_result.as_dataframe()
            target = self._target_relation()
            if materialized == "view":
                backing_target = self._view_backing_relation(target)
                # Drop the user-facing view first; otherwise the backing-table
                # replace below fails with a dependency error, and we'd also be
                # constrained by `CREATE OR REPLACE VIEW`'s no-rename rule on
                # the rebuild.
                self._drop_existing_relation(session.conn, target)
                rows_written = write_model_result(
                    conn=session.conn,
                    target=backing_target,
                    result=model_result,
                    batch_size=limits.batch_size,
                    materialized="table",
                    incremental_strategy="append",
                    unique_key=None,
                    column_types=column_types,
                    categorical_types=categorical_types,
                )
                self._create_or_replace_view(session.conn, target, backing_target)
                return rows_written

            return write_model_result(
                conn=session.conn,
                target=target,
                result=model_result,
                batch_size=limits.batch_size,
                materialized=materialized,
                incremental_strategy=incremental_strategy,
                unique_key=unique_key,
                column_types=column_types,
                categorical_types=categorical_types,
            )
        finally:
            session.close()
