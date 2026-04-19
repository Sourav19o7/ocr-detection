"""Migration 001: introduce item_images to support HUID + artifact gallery.

A batch item may now have one HUID image (slot 0) plus up to three artifact
images (slots 1, 2, 3). The existing ``batch_items.image_path`` column keeps
its meaning for read compatibility; every existing row is backfilled into
``item_images`` so the new UI can treat all images uniformly.
"""

from __future__ import annotations

import sqlite3


def apply(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS item_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_item_id INTEGER NOT NULL REFERENCES batch_items(id) ON DELETE CASCADE,
            tag_id TEXT NOT NULL,
            image_type TEXT NOT NULL CHECK (image_type IN ('huid','artifact')),
            slot INTEGER NOT NULL,
            s3_key TEXT NOT NULL,
            s3_bucket TEXT NOT NULL DEFAULT '',
            content_type TEXT,
            size_bytes INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(tag_id, image_type, slot)
        )
        """
    )

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_item_images_tag_id ON item_images(tag_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_item_images_batch_item ON item_images(batch_item_id)"
    )

    # Backfill HUID images from batch_items.image_path. image_path may hold
    # either an S3 key or a local filesystem path — we keep it verbatim in
    # s3_key so existing URLs resolve unchanged.
    cur.execute(
        """
        INSERT OR IGNORE INTO item_images
            (batch_item_id, tag_id, image_type, slot, s3_key, s3_bucket)
        SELECT id, tag_id, 'huid', 0, image_path, ''
        FROM batch_items
        WHERE image_path IS NOT NULL AND image_path <> ''
        """
    )
