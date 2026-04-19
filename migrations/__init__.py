"""Schema migrations for the Hallmark QC database.

Migrations are applied in numeric order by :func:`run_migrations`.  Each
migration is idempotent: running it twice does nothing on the second pass
because the ``schema_migrations`` table records the applied version.
"""

from .runner import run_migrations

__all__ = ["run_migrations"]
