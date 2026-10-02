# Changelog

## 0.3.0

- Add named Postgres connections for Python-model `source()` reads, with
  explicit rejection of ambiguous named routing on `ref()` dependencies.
- Keep same-connection joins pushed down and require explicit dataframe loading
  for cross-connection joins, or an explicit optional DuckDB spill-backed join.
- Use read-only repeatable-read transactions for named sources and support
  just-in-time environment, libpq service, and passfile authentication.
- Add per-model aggregate input limits and query-safe dataframe size checks.
- Validate named connection configuration before model execution.
- Preserve primary model errors when connection cleanup also fails.
- Add real multi-database Postgres integration coverage and a dbt/Python CI
  compatibility matrix.
- Fix `ref()` and unrouted `source()` calls compiling to invalid Python
  (`null`), which failed every such Python model under real dbt.
- Start the target and every routed source snapshot back to back at model
  start, and log the measured skew, instead of whenever each source is first read.
- Read the target in one `REPEATABLE READ` snapshot by default
  (`pybridge_target_isolation: read committed` restores the old behavior).
- DuckDB joins: stage tables with Postgres column types instead of inferring
  them from the first batch (which rounded numerics and rejected values after
  an all-NULL batch), return exact `Decimal` values, apply `pybridge_max_rows`
  and `pybridge_max_bytes` to eager join results, reject clashing non-key
  column names, and accept `threads`, `temp_dir`, and `max_temp_directory_size`.
- Reject incremental `merge`/`delete+insert` results that contain one
  `unique_key` more than once, within a batch or across yielded batches.
- Allow client-certificate, Kerberos/GSSAPI, `~/.pgpass`, and trust
  authentication for named connections; add `sslpassword`, `gssencmode`, and
  `krbsrvname`.
- Add `password_command` for short-lived tokens (RDS IAM, Entra ID, Vault),
  run without a shell each time a named connection opens.
- Add `consistent_cut()` to filter inputs from separate servers to one shared
  watermark.
- Add `filter_by=` to DuckDB joins to fetch only the other side's matching keys.
- Create unconstrained `numeric` for `Decimal` columns instead of a precision
  guessed from sampled values, which Postgres could silently round later rows
  into, and infer object column types from every value rather than the first 128.
