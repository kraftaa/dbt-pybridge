from dbt.adapters.pybridge.connections import PybridgeCredentials


def test_named_passwords_are_omitted_from_repr_and_connection_info():
    credentials = PybridgeCredentials.from_dict(
        {
            "database": "synthetic_analytics",
            "schema": "transform",
            "host": "analytics.example.invalid",
            "user": "transformer",
            "password": "target-secret",
            "port": 5432,
            "pybridge_connections": {
                "app_db": {
                    "host": "app.example.invalid",
                    "user": "reader",
                    "password": "nested-secret",
                    "dbname": "synthetic_application",
                }
            },
        }
    )

    assert "nested-secret" not in repr(credentials)
    connection_info = dict(credentials.connection_info())
    assert "pybridge_connections" not in connection_info
    assert "nested-secret" not in repr(connection_info)
