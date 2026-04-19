"""Migration runner for the Hallmark QC SQLite database."""

from __future__ import annotations

import sqlite3
from typing import Callable, List, Tuple

from . import m001_artifact_images


Migration = Tuple[int, str, Callable[[sqlite3.Connection], None]]


MIGRATIONS: List[Migration] = [
    (1, "artifact_images", m001_artifact_images.apply),
]


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def _applied_versions(conn: sqlite3.Connection) -> set[int]:
    cur = conn.execute("SELECT version FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def run_migrations(conn: sqlite3.Connection) -> List[int]:
    """Apply pending migrations to ``conn``.

    Returns the list of versions applied in this invocation.
    """
    _ensure_migrations_table(conn)
    applied = _applied_versions(conn)
    newly_applied: List[int] = []

    for version, name, func in MIGRATIONS:
        if version in applied:
            continue
        func(conn)
        conn.execute(
            "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
            (version, name),
        )
        conn.commit()
        newly_applied.append(version)

    return newly_applied
