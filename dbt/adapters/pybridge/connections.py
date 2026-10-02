from dataclasses import dataclass, field
from typing import Any, Dict

from dbt.adapters.postgres.connections import PostgresConnectionManager, PostgresCredentials


@dataclass
class PybridgeCredentials(PostgresCredentials):
    # Additional connections are opened only by Python models that explicitly
    # route a ref/source to one of these names. `repr=False` and omission from
    # `_connection_keys()` keep the mapping out of normal adapter connection
    # displays. Profiles should still use env_var() for password redaction.
    pybridge_connections: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    @property
    def type(self):
        return "pybridge"


class PybridgeConnectionManager(PostgresConnectionManager):
    TYPE = "pybridge"
