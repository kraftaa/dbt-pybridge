from dataclasses import dataclass

from dbt.adapters.postgres.connections import PostgresConnectionManager, PostgresCredentials


@dataclass
class PybridgeCredentials(PostgresCredentials):
    @property
    def type(self):
        return "pybridge"


class PybridgeConnectionManager(PostgresConnectionManager):
    TYPE = "pybridge"
