from dbt.adapters.contracts.connection import AdapterResponse
from dbt.adapters.postgres.impl import PostgresAdapter

from dbt.adapters.pybridge.connections import PybridgeConnectionManager
from dbt_pybridge.runner import LocalPythonModelRunner


class PybridgeAdapter(PostgresAdapter):
    ConnectionManager = PybridgeConnectionManager

    def submit_python_job(self, parsed_model: dict, compiled_code: str):
        credentials = self.connections.profile.credentials
        runner = LocalPythonModelRunner(credentials=credentials, parsed_model=parsed_model, compiled_code=compiled_code)
        rows_written = runner.run()
        return AdapterResponse(
            _message=f"OK ({rows_written} rows)",
            code="OK",
            rows_affected=rows_written,
        )
