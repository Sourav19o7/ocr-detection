"""Migration 002: Add fields for external API integration and upload queue.

New fields:
- batch_items: bis_job_no, branch, upload_status, rework_status, manak_huid
- item_images: device_source
- New table: api_keys for external API authentication
- New table: upload_queue for portal upload tracking
"""

from __future__ import annotations

import sqlite3


def apply(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Add new columns to batch_items
    # SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we check first
    cur.execute("PRAGMA table_info(batch_items)")
    existing_columns = {row[1] for row in cur.fetchall()}

    if "bis_job_no" not in existing_columns:
        cur.execute("ALTER TABLE batch_items ADD COLUMN bis_job_no TEXT")

    if "branch" not in existing_columns:
        cur.execute("ALTER TABLE batch_items ADD COLUMN branch TEXT DEFAULT 'default'")

    if "upload_status" not in existing_columns:
        cur.execute("ALTER TABLE batch_items ADD COLUMN upload_status TEXT DEFAULT 'pending'")

    if "rework_status" not in existing_columns:
        cur.execute("ALTER TABLE batch_items ADD COLUMN rework_status TEXT")

    if "manak_huid" not in existing_columns:
        cur.execute("ALTER TABLE batch_items ADD COLUMN manak_huid TEXT")

    # Add device_source to item_images
    cur.execute("PRAGMA table_info(item_images)")
    img_columns = {row[1] for row in cur.fetchall()}

    if "device_source" not in img_columns:
        cur.execute("ALTER TABLE item_images ADD COLUMN device_source TEXT")

    # Create api_keys table for external API authentication
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            branch TEXT,
            permissions TEXT DEFAULT '[]',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP
        )
        """
    )

    # Create upload_queue table for portal upload tracking
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS upload_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_id TEXT NOT NULL,
            bis_job_no TEXT NOT NULL,
            branch TEXT,
            image_type TEXT NOT NULL CHECK (image_type IN ('article', 'huid')),
            s3_key TEXT NOT NULL,
            status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'queued', 'uploading', 'uploaded', 'failed')),
            retry_count INTEGER DEFAULT 0,
            error_message TEXT,
            portal_reference TEXT,
            queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            uploaded_at TIMESTAMP,
            UNIQUE(tag_id, image_type)
        )
        """
    )

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_upload_queue_status ON upload_queue(status)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_upload_queue_bis_job ON upload_queue(bis_job_no)"
    )

    # Create manak_comparison table for Excel comparison results
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS manak_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison_id TEXT NOT NULL,
            bis_job_no TEXT NOT NULL,
            tag_id TEXT NOT NULL,
            manak_huid TEXT,
            manak_purity TEXT,
            ocr_huid TEXT,
            ocr_purity TEXT,
            huid_match INTEGER,
            purity_match INTEGER,
            status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'match', 'partial_match', 'mismatch', 'missing_ocr')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
        """
    )

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_manak_comparisons_status ON manak_comparisons(status)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_manak_comparisons_bis_job ON manak_comparisons(bis_job_no)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_manak_comparisons_tag ON manak_comparisons(tag_id)"
    )

    conn.commit()
