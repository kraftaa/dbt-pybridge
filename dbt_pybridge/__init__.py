import os

PACKAGE_PATH = os.path.dirname(__file__)

__all__ = ["PACKAGE_PATH", "consistent_cut"]


def __getattr__(name):
    # Lazy: the adapter imports PACKAGE_PATH before pandas/psycopg2 are needed.
    if name == "consistent_cut":
        from dbt_pybridge.runner import consistent_cut

        return consistent_cut
    raise AttributeError(f"module 'dbt_pybridge' has no attribute {name!r}")
