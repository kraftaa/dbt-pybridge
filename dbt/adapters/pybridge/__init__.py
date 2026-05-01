from dbt.adapters.base import AdapterPlugin

from dbt.adapters.pybridge.connections import PybridgeCredentials
from dbt.adapters.pybridge.impl import PybridgeAdapter
from dbt_pybridge import PACKAGE_PATH

Plugin = AdapterPlugin(
    adapter=PybridgeAdapter,
    credentials=PybridgeCredentials,
    include_path=PACKAGE_PATH,
    dependencies=["postgres"],
    project_name="dbt_pybridge",
)
