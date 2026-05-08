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

    def select(self, projection_sql: str):
        projection = str(projection_sql or "").strip()
        if not projection:
            raise RuntimeError("dbt.ref(...).select(...) requires a non-empty SQL projection string")
        if ";" in projection:
            raise RuntimeError("dbt.ref(...).select(...) does not allow semicolons")
        relation_sql = f"(select {projection} from {self._relation_sql}) as pybridge_select"
        return RelationFrame(self._session, relation_sql)

    def where(self, predicate_sql: str):
        predicate = str(predicate_sql or "").strip()
        if not predicate:
            raise RuntimeError("dbt.ref(...).where(...) requires a non-empty SQL predicate string")
        if ";" in predicate:
            raise RuntimeError("dbt.ref(...).where(...) does not allow semicolons")
        relation_sql = f"(select * from {self._relation_sql} where {predicate}) as pybridge_where"
        return RelationFrame(self._session, relation_sql)

    def join(self, other: "RelationFrame", on=None, how: str = "inner"):
        if not isinstance(other, RelationFrame):
            raise RuntimeError("dbt.ref(...).join(other, ...) requires another RelationFrame; pass dbt.ref('...')")
        if self._session is not other._session:
            raise RuntimeError("dbt.ref(...).join(...) requires both refs to share a session")
        how_normalized = str(how or "").strip().lower()
        valid_join_types = {"inner", "left", "right", "full", "full outer", "left outer", "right outer", "cross"}
        if how_normalized not in valid_join_types:
            raise RuntimeError(
                f"Invalid join type {how!r}. Expected one of: {sorted(valid_join_types)}"
            )
        # Wrap each side in `select * from (...) as <alias>`. Without this
        # wrap, chaining off a `.select()` / `.where()` (whose _relation_sql
        # already ends with `as pybridge_<op>`) would produce two consecutive
        # aliases (`as pybridge_select as pybridge_l`), which Postgres rejects.
        left = f"(select * from {self._relation_sql}) as pybridge_l"
        right = f"(select * from {other._relation_sql}) as pybridge_r"

        if how_normalized == "cross":
            if on:
                raise RuntimeError("Cross joins do not take an `on=` argument")
            relation_sql = (
                f"(select * from {left} cross join {right}) as pybridge_join"
            )
            return RelationFrame(self._session, relation_sql)

        if isinstance(on, str):
            keys = [on]
        elif on is None:
            keys = []
        else:
            try:
                keys = list(on)
            except TypeError:
                raise RuntimeError(
                    f"Invalid `on=` value {on!r}; expected a column name or list of names"
                ) from None
        keys = [str(k).strip() for k in keys if str(k).strip()]
        if not keys:
            raise RuntimeError("dbt.ref(...).join(...) requires `on=` to be a column name or list of names")
        for key in keys:
            if ";" in key or '"' in key:
                raise RuntimeError(f"Invalid join key {key!r}; column names must not contain ';' or '\"'")

        # USING (col) keeps a single deduplicated copy of the join column in
        # the output; if we used ON pybridge_l.k = pybridge_r.k with `select *`
        # we'd get two columns named k, which fails our duplicate-column check.
        using_columns = ", ".join(quote_ident(k) for k in keys)
        relation_sql = (
            f"(select * from {left} {how_normalized} join {right} using ({using_columns}))"
            f" as pybridge_join"
        )
        return RelationFrame(self._session, relation_sql)

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

    @staticmethod
    def _cfg_value(cfg: Dict[str, Any], key: str, legacy_key: Optional[str] = None, default: Any = None) -> Any:
        if key in cfg:
            return cfg.get(key)
        if legacy_key and legacy_key in cfg:
            return cfg.get(legacy_key)
        return default

    @staticmethod
    def _log(message: str) -> None:
        print(f"[pybridge] {message}")

    @staticmethod
    def _load_df_function(session: LocalPostgresSession):
        # The callback dbt's compiled `dbtObj` calls for each `dbt.ref(...)` /
        # `dbt.source(...)`. We normalize the 3-part identifier dbt renders
        # ("db"."schema"."t") down to a 2-part one *here*, at the single entry
        # point, so every subsequent .select()/.where()/.join() wraps an
        # already-safe relation SQL and can't smuggle a cross-database
        # qualifier into the resulting subquery.
        def load(relation_sql: str) -> "RelationFrame":
            return RelationFrame(session, session._normalize_relation_sql(relation_sql))
        return load

    def _limits(self, cfg: Dict[str, Any]) -> ModelLimits:
        def _as_int(key: str, default: int) -> int:
            legacy_key = key.replace("pybridge_", "localpy_") if key.startswith("pybridge_") else None
            value = self._cfg_value(cfg, key, legacy_key, default)
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                raise RuntimeError(f"Invalid value for {key}: expected integer, got {value!r}") from None
            if parsed < 0:
                raise RuntimeError(f"Invalid value for {key}: expected >= 0, got {parsed}")
            return parsed

        def _as_bool(key: str, default: bool) -> bool:
            legacy_key = key.replace("pybridge_", "localpy_") if key.startswith("pybridge_") else None
            value = self._cfg_value(cfg, key, legacy_key, default)
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
            max_rows=_as_int("pybridge_max_rows", 1_000_000),
            warn_rows=_as_int("pybridge_warn_rows", 200_000),
            max_bytes=_as_int("pybridge_max_bytes", 512 * 1024 * 1024),
            warn_bytes=_as_int("pybridge_warn_bytes", 128 * 1024 * 1024),
            batch_size=_as_int("pybridge_batch_size", 100_000),
            allow_large_tables=_as_bool("pybridge_allow_large_tables", False),
            chunked_mode=_as_bool("pybridge_chunked_mode", False),
        )

    def _column_types(self, cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
        return self._type_mapping_config(cfg, "pybridge_column_types", legacy_key="localpy_column_types")

    def _categorical_types(self, cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
        return self._type_mapping_config(cfg, "pybridge_categorical_types", legacy_key="localpy_categorical_types")

    def _type_mapping_config(self, cfg: Dict[str, Any], key: str, legacy_key: Optional[str] = None) -> Optional[Dict[str, str]]:
        raw = self._cfg_value(cfg, key, legacy_key, None)
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

    @staticmethod
    def _suffix_relation(relation: TargetRelation, suffix: str) -> TargetRelation:
        # Postgres identifiers are limited to 63 chars (NAMEDATALEN-1). Truncate
        # the base if the suffix would push us over so two long-but-different
        # base names don't collide after silent server-side truncation.
        max_base = max(1, 63 - len(suffix))
        return TargetRelation(
            database=relation.database,
            schema=relation.schema,
            identifier=relation.identifier[:max_base] + suffix,
        )

    def _view_intermediate_relation(self, view_target: TargetRelation) -> TargetRelation:
        return self._suffix_relation(view_target, "__pybtmp")

    def _view_backup_relation(self, view_target: TargetRelation) -> TargetRelation:
        return self._suffix_relation(view_target, "__pybbkup")

    def _backing_intermediate_relation(self, backing_target: TargetRelation) -> TargetRelation:
        return self._suffix_relation(backing_target, "__tmp")

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

    def _create_view(self, conn, view_target: TargetRelation, backing_table: TargetRelation) -> None:
        """Plain `CREATE VIEW`. Caller must ensure no relation exists at view_target."""
        view_sql = view_target.render()
        backing_sql = backing_table.render()
        with conn.cursor() as cur:
            if view_target.schema:
                cur.execute(f"create schema if not exists {quote_ident(view_target.schema)}")
            cur.execute(f"create view {view_sql} as select * from {backing_sql}")
        conn.commit()

    # Kept under the old name so external callers (and the existing test) still
    # work. Internally just delegates to _create_view.
    def _create_or_replace_view(self, conn, view_target: TargetRelation, backing_table: TargetRelation) -> None:
        self._create_view(conn, view_target, backing_table)

    def _materialize_view_via_swap(
        self,
        conn,
        view_target: TargetRelation,
        backing_target: TargetRelation,
        write_backing,
    ) -> int:
        """Atomic-ish view materialization that mirrors dbt-core's Postgres pattern.

        Plain DROP+CREATE has a window where the view doesn't exist; on the
        Postgres path dbt-core uses a rename-swap so the user-facing name is
        always resolvable to *something*. We do the same here, with the extra
        wrinkle that we own the backing table too:

            1. Cleanup any leftovers from a prior failed run (intermediate
               view, backup view, intermediate backing).
            2. Build the new backing at an intermediate name (via
               `write_backing(intermediate_backing)`, which is just
               write_model_result with materialized='table').
            3. CREATE VIEW <intermediate_view> AS SELECT * FROM
               <intermediate_backing>.
            4. In a single transaction, rename existing view (if any) to
               backup, then rename intermediate to target. After commit,
               readers see the new view atomically.
            5. Drop the backup view (which still depended on the old backing).
            6. Drop the old backing under its stable name.
            7. Rename the intermediate backing into that stable name.
               PG view dependencies follow OIDs, not names, so step 7 doesn't
               break the (already-published) new view.
        """
        intermediate_view = self._view_intermediate_relation(view_target)
        backup_view = self._view_backup_relation(view_target)
        intermediate_backing = self._backing_intermediate_relation(backing_target)

        # 1. Cleanup leftovers from a prior failed run.
        for leftover in (intermediate_view, backup_view, intermediate_backing):
            self._drop_existing_relation(conn, leftover)

        # 2. Build the new backing at the intermediate name.
        rows_written = write_backing(intermediate_backing)

        # 3. Build the new view at the intermediate name, pointing at the new
        #    backing.
        self._create_view(conn, intermediate_view, intermediate_backing)

        # Whatever sits at the view target now: a view (rename to backup), a
        # non-view (drop with CASCADE — same behavior as dbt-core when an
        # incompatible relation occupies the target slot), or nothing.
        existing_kind = self._relation_kind(conn, view_target)
        existing_is_view = existing_kind == "v"
        if existing_kind is not None and not existing_is_view:
            self._drop_existing_relation(conn, view_target)
            existing_is_view = False

        # 4. Atomic swap (rename existing → backup, intermediate → target).
        with conn.cursor() as cur:
            if existing_is_view:
                cur.execute(
                    f"alter view {view_target.render()} rename to {quote_ident(backup_view.identifier)}"
                )
            cur.execute(
                f"alter view {intermediate_view.render()} rename to {quote_ident(view_target.identifier)}"
            )
        conn.commit()

        # 5. Drop the backup view (no-op if there was no prior view).
        if existing_is_view:
            self._drop_existing_relation(conn, backup_view)

        # 6. Drop the old backing under the stable name (no-op on first run).
        self._drop_existing_relation(conn, backing_target)

        # 7. Rename intermediate backing into the stable name. The new view's
        #    pg_rewrite rule references the backing's OID; the rename
        #    preserves OID, so the view continues to resolve correctly.
        with conn.cursor() as cur:
            cur.execute(
                f"alter table {intermediate_backing.render()} rename to {quote_ident(backing_target.identifier)}"
            )
        conn.commit()

        return rows_written

    def run(self) -> int:
        cfg = self._model_config()
        dataframe_backend = str(
            self._cfg_value(cfg, "pybridge_dataframe_backend", "localpy_dataframe_backend", "pandas")
        ).lower()
        limits = self._limits(cfg)
        materialized = str(cfg.get("materialized", "table")).lower()
        unique_key = self._normalize_unique_key(cfg.get("unique_key"))
        column_types = self._column_types(cfg)
        categorical_types = self._categorical_types(cfg)
        on_schema_change = str(cfg.get("on_schema_change", "ignore")).lower()
        sync_drop_cascade = self._cfg_value(cfg, "pybridge_sync_drop_cascade", None, False)
        if isinstance(sync_drop_cascade, str):
            sync_drop_cascade = sync_drop_cascade.strip().lower() in {"true", "t", "yes", "y", "on", "1"}
        else:
            sync_drop_cascade = bool(sync_drop_cascade)
        incremental_strategy = str(cfg.get("incremental_strategy", "default")).lower()
        if incremental_strategy == "default":
            incremental_strategy = "merge" if unique_key else "append"

        session = LocalPostgresSession(
            credentials=self.credentials,
            limits=limits,
            dataframe_backend=dataframe_backend,
            logger=self._log,
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

            dbt_obj = dbt_obj_cls(self._load_df_function(session))
            model_result = model_fn(dbt_obj, session)
            if isinstance(model_result, RelationFrame):
                model_result = model_result.as_dataframe()
            target = self._target_relation()
            if materialized == "view":
                backing_target = self._view_backing_relation(target)

                def _write_backing(intermediate_backing):
                    return write_model_result(
                        conn=session.conn,
                        target=intermediate_backing,
                        result=model_result,
                        batch_size=limits.batch_size,
                        materialized="table",
                        incremental_strategy="append",
                        unique_key=None,
                        column_types=column_types,
                        categorical_types=categorical_types,
                        logger=self._log,
                    )

                return self._materialize_view_via_swap(
                    session.conn,
                    view_target=target,
                    backing_target=backing_target,
                    write_backing=_write_backing,
                )

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
                logger=self._log,
                on_schema_change=on_schema_change,
                cascade_drops=sync_drop_cascade,
            )
        finally:
            session.close()
