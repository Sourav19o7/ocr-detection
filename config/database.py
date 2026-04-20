"""
Database Models for Hallmark OCR System.

Tables:
- batches: Batch uploads (CSV/Excel with tag IDs and expected HUIDs)
- batch_items: Individual items within a batch
- ocr_results: OCR processing results for each item
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field
import json
import sqlite3
import os


class ProcessingStatus(str, Enum):
    """Status of an item in the processing pipeline."""
    PENDING = "pending"           # Uploaded, waiting for image
    PROCESSING = "processing"     # Image uploaded, OCR in progress
    COMPLETED = "completed"       # OCR completed
    FAILED = "failed"            # Processing failed
    MANUAL_REVIEW = "manual_review"  # Needs manual review


class QCDecision(str, Enum):
    """QC decision for an item."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"
    PENDING = "pending"


@dataclass
class Batch:
    """A batch upload containing tag IDs and expected HUIDs."""
    id: Optional[int] = None
    batch_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    total_items: int = 0
    processed_items: int = 0
    status: str = "pending"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "batch_name": self.batch_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class BatchItem:
    """An individual item within a batch."""
    id: Optional[int] = None
    batch_id: int = 0
    tag_id: str = ""
    expected_huid: str = ""
    status: ProcessingStatus = ProcessingStatus.PENDING
    image_path: Optional[str] = None       # S3 path or local path
    image_url: Optional[str] = None        # Public URL for viewing
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "batch_id": self.batch_id,
            "tag_id": self.tag_id,
            "expected_huid": self.expected_huid,
            "status": self.status.value if isinstance(self.status, ProcessingStatus) else self.status,
            "image_path": self.image_path,
            "image_url": self.image_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "metadata": self.metadata,
        }


@dataclass
class ItemImage:
    """An image associated with a batch item (HUID or artifact)."""
    id: Optional[int] = None
    batch_item_id: int = 0
    tag_id: str = ""
    image_type: str = "huid"  # 'huid' or 'artifact'
    slot: int = 0              # 0 for huid, 1..3 for artifact
    s3_key: str = ""
    s3_bucket: str = ""
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    uploaded_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "batch_item_id": self.batch_item_id,
            "tag_id": self.tag_id,
            "image_type": self.image_type,
            "slot": self.slot,
            "s3_key": self.s3_key,
            "s3_bucket": self.s3_bucket,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
        }


@dataclass
class OCRResult:
    """OCR processing result for a batch item."""
    id: Optional[int] = None
    batch_item_id: int = 0
    tag_id: str = ""
    expected_huid: str = ""
    actual_huid: Optional[str] = None
    huid_match: bool = False
    purity_code: Optional[str] = None
    karat: Optional[str] = None
    purity_percentage: Optional[float] = None
    confidence: float = 0.0
    decision: QCDecision = QCDecision.PENDING
    rejection_reasons: List[str] = field(default_factory=list)
    raw_ocr_text: str = ""
    processed_image_path: Optional[str] = None  # Path to annotated/processed image
    processed_image_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    processing_time_ms: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "batch_item_id": self.batch_item_id,
            "tag_id": self.tag_id,
            "expected_huid": self.expected_huid,
            "actual_huid": self.actual_huid,
            "huid_match": self.huid_match,
            "purity_code": self.purity_code,
            "karat": self.karat,
            "purity_percentage": self.purity_percentage,
            "confidence": self.confidence,
            "decision": self.decision.value if isinstance(self.decision, QCDecision) else self.decision,
            "rejection_reasons": self.rejection_reasons,
            "raw_ocr_text": self.raw_ocr_text,
            "processed_image_path": self.processed_image_path,
            "processed_image_url": self.processed_image_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }


class DatabaseManager:
    """SQLite database manager for the hallmark QC system."""

    def __init__(self, db_path: str = "hallmark_qc.db"):
        self.db_path = db_path
        self._init_database()
        self._run_migrations()

    def _run_migrations(self):
        """Apply any pending schema migrations."""
        # Local import avoids a circular import at module load time.
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from migrations import run_migrations

        conn = self._get_connection()
        try:
            run_migrations(conn)
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Batches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_items INTEGER DEFAULT 0,
                processed_items INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                metadata TEXT DEFAULT '{}'
            )
        """)

        # Batch items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                tag_id TEXT NOT NULL,
                expected_huid TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                image_path TEXT,
                image_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (batch_id) REFERENCES batches(id),
                UNIQUE(tag_id)
            )
        """)

        # OCR results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_item_id INTEGER NOT NULL,
                tag_id TEXT NOT NULL,
                expected_huid TEXT NOT NULL,
                actual_huid TEXT,
                huid_match INTEGER DEFAULT 0,
                purity_code TEXT,
                karat TEXT,
                purity_percentage REAL,
                confidence REAL DEFAULT 0.0,
                decision TEXT DEFAULT 'pending',
                rejection_reasons TEXT DEFAULT '[]',
                raw_ocr_text TEXT DEFAULT '',
                processed_image_path TEXT,
                processed_image_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_time_ms INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (batch_item_id) REFERENCES batch_items(id)
            )
        """)

        # Create indices for faster lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_items_tag_id ON batch_items(tag_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_items_batch_id ON batch_items(batch_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ocr_results_tag_id ON ocr_results(tag_id)")

        conn.commit()
        conn.close()

    # Batch operations
    def create_batch(self, batch: Batch) -> int:
        """Create a new batch and return its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO batches (batch_name, total_items, status, metadata)
               VALUES (?, ?, ?, ?)""",
            (batch.batch_name, batch.total_items, batch.status, json.dumps(batch.metadata))
        )
        batch_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return batch_id

    def get_batch(self, batch_id: int) -> Optional[Batch]:
        """Get a batch by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM batches WHERE id = ?", (batch_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Batch(
                id=row["id"],
                batch_name=row["batch_name"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                total_items=row["total_items"],
                processed_items=row["processed_items"],
                status=row["status"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        return None

    def get_all_batches(self) -> List[Batch]:
        """Get all batches."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM batches ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        return [
            Batch(
                id=row["id"],
                batch_name=row["batch_name"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                total_items=row["total_items"],
                processed_items=row["processed_items"],
                status=row["status"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def update_batch_progress(self, batch_id: int, processed_items: int):
        """Update batch processing progress."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE batches SET processed_items = ? WHERE id = ?",
            (processed_items, batch_id)
        )
        conn.commit()
        conn.close()

    def update_batch_status(self, batch_id: int, status: str):
        """Update batch status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE batches SET status = ? WHERE id = ?",
            (status, batch_id)
        )
        conn.commit()
        conn.close()

    # Batch item operations
    def create_batch_item(self, item: BatchItem) -> int:
        """Create a new batch item and return its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO batch_items
               (batch_id, tag_id, expected_huid, status, image_path, image_url, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                item.batch_id,
                item.tag_id,
                item.expected_huid,
                item.status.value if isinstance(item.status, ProcessingStatus) else item.status,
                item.image_path,
                item.image_url,
                json.dumps(item.metadata),
            )
        )
        item_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return item_id

    def create_batch_with_items(self, batch: Batch, items: List[BatchItem]) -> int:
        """Atomically create a batch and its items.

        Uses a single transaction so callers either get a fully-populated
        batch or no batch at all. Raises on any integrity error after
        rolling back.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN")
            cursor.execute(
                """INSERT INTO batches (batch_name, total_items, status, metadata)
                   VALUES (?, ?, ?, ?)""",
                (
                    batch.batch_name,
                    len(items),
                    batch.status,
                    json.dumps(batch.metadata),
                ),
            )
            batch_id = cursor.lastrowid
            for item in items:
                cursor.execute(
                    """INSERT OR REPLACE INTO batch_items
                       (batch_id, tag_id, expected_huid, status, image_path, image_url, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        batch_id,
                        item.tag_id,
                        item.expected_huid,
                        item.status.value if isinstance(item.status, ProcessingStatus) else item.status,
                        item.image_path,
                        item.image_url,
                        json.dumps(item.metadata),
                    ),
                )
            conn.commit()
            return batch_id
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_batch_item_by_tag(self, tag_id: str) -> Optional[BatchItem]:
        """Get a batch item by tag ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM batch_items WHERE tag_id = ?", (tag_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return BatchItem(
                id=row["id"],
                batch_id=row["batch_id"],
                tag_id=row["tag_id"],
                expected_huid=row["expected_huid"],
                status=ProcessingStatus(row["status"]) if row["status"] else ProcessingStatus.PENDING,
                image_path=row["image_path"],
                image_url=row["image_url"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                processed_at=datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else None,
                metadata=json.loads(row["metadata"] or "{}"),
            )
        return None

    def get_batch_items(self, batch_id: int) -> List[BatchItem]:
        """Get all items in a batch."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM batch_items WHERE batch_id = ? ORDER BY id", (batch_id,))
        rows = cursor.fetchall()
        conn.close()

        return [
            BatchItem(
                id=row["id"],
                batch_id=row["batch_id"],
                tag_id=row["tag_id"],
                expected_huid=row["expected_huid"],
                status=ProcessingStatus(row["status"]) if row["status"] else ProcessingStatus.PENDING,
                image_path=row["image_path"],
                image_url=row["image_url"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                processed_at=datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else None,
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def update_batch_item_image(self, tag_id: str, image_path: str, image_url: Optional[str] = None):
        """Update the image path for a batch item."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE batch_items
               SET image_path = ?, image_url = ?, status = ?
               WHERE tag_id = ?""",
            (image_path, image_url, ProcessingStatus.PROCESSING.value, tag_id)
        )
        conn.commit()
        conn.close()

    def update_batch_item_status(self, tag_id: str, status: ProcessingStatus):
        """Update the status of a batch item."""
        conn = self._get_connection()
        cursor = conn.cursor()
        processed_at = datetime.now().isoformat() if status == ProcessingStatus.COMPLETED else None
        cursor.execute(
            "UPDATE batch_items SET status = ?, processed_at = ? WHERE tag_id = ?",
            (status.value, processed_at, tag_id)
        )
        conn.commit()
        conn.close()

    # OCR result operations
    def create_ocr_result(self, result: OCRResult) -> int:
        """Create a new OCR result and return its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO ocr_results
               (batch_item_id, tag_id, expected_huid, actual_huid, huid_match,
                purity_code, karat, purity_percentage, confidence, decision,
                rejection_reasons, raw_ocr_text, processed_image_path,
                processed_image_url, processing_time_ms, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.batch_item_id,
                result.tag_id,
                result.expected_huid,
                result.actual_huid,
                1 if result.huid_match else 0,
                result.purity_code,
                result.karat,
                result.purity_percentage,
                result.confidence,
                result.decision.value if isinstance(result.decision, QCDecision) else result.decision,
                json.dumps(result.rejection_reasons),
                result.raw_ocr_text,
                result.processed_image_path,
                result.processed_image_url,
                result.processing_time_ms,
                json.dumps(result.metadata),
            )
        )
        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return result_id

    def get_ocr_result_by_tag(self, tag_id: str) -> Optional[OCRResult]:
        """Get OCR result by tag ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ocr_results WHERE tag_id = ? ORDER BY created_at DESC LIMIT 1", (tag_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return OCRResult(
                id=row["id"],
                batch_item_id=row["batch_item_id"],
                tag_id=row["tag_id"],
                expected_huid=row["expected_huid"],
                actual_huid=row["actual_huid"],
                huid_match=bool(row["huid_match"]),
                purity_code=row["purity_code"],
                karat=row["karat"],
                purity_percentage=row["purity_percentage"],
                confidence=row["confidence"],
                decision=QCDecision(row["decision"]) if row["decision"] else QCDecision.PENDING,
                rejection_reasons=json.loads(row["rejection_reasons"] or "[]"),
                raw_ocr_text=row["raw_ocr_text"],
                processed_image_path=row["processed_image_path"],
                processed_image_url=row["processed_image_url"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                processing_time_ms=row["processing_time_ms"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
        return None

    def get_results_by_batch(self, batch_id: int) -> List[dict]:
        """Get all results for a batch with item details."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT bi.tag_id, bi.expected_huid, bi.status as item_status,
                   bi.image_path, bi.image_url,
                   ocr.actual_huid, ocr.huid_match, ocr.purity_code, ocr.karat,
                   ocr.purity_percentage, ocr.confidence, ocr.decision,
                   ocr.rejection_reasons, ocr.raw_ocr_text,
                   ocr.processed_image_path, ocr.processed_image_url
            FROM batch_items bi
            LEFT JOIN ocr_results ocr
                   ON ocr.id = (SELECT id FROM ocr_results
                                 WHERE tag_id = bi.tag_id
                                 ORDER BY created_at DESC
                                 LIMIT 1)
            WHERE bi.batch_id = ?
            ORDER BY bi.id
        """, (batch_id,))
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "tag_id": row["tag_id"],
                "expected_huid": row["expected_huid"],
                "status": row["item_status"],
                "image_url": row["image_url"],
                "actual_huid": row["actual_huid"],
                "huid_match": bool(row["huid_match"]) if row["huid_match"] is not None else None,
                "purity_code": row["purity_code"],
                "karat": row["karat"],
                "purity_percentage": row["purity_percentage"],
                "confidence": row["confidence"],
                "decision": row["decision"],
                "rejection_reasons": json.loads(row["rejection_reasons"] or "[]"),
                "raw_ocr_text": row["raw_ocr_text"],
                "processed_image_url": row["processed_image_url"],
            })
        return results

    def get_full_result_by_tag(self, tag_id: str) -> Optional[dict]:
        """Get complete result data for a tag ID including batch info."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT bi.tag_id, bi.expected_huid, bi.status as item_status,
                   bi.image_path, bi.image_url, bi.batch_id,
                   b.batch_name,
                   ocr.actual_huid, ocr.huid_match, ocr.purity_code, ocr.karat,
                   ocr.purity_percentage, ocr.confidence, ocr.decision,
                   ocr.rejection_reasons, ocr.raw_ocr_text,
                   ocr.processed_image_path, ocr.processed_image_url,
                   ocr.processing_time_ms, ocr.created_at as processed_at
            FROM batch_items bi
            LEFT JOIN batches b ON bi.batch_id = b.id
            LEFT JOIN ocr_results ocr
                   ON ocr.id = (SELECT id FROM ocr_results
                                 WHERE tag_id = bi.tag_id
                                 ORDER BY created_at DESC
                                 LIMIT 1)
            WHERE bi.tag_id = ?
            LIMIT 1
        """, (tag_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "tag_id": row["tag_id"],
                "batch_id": row["batch_id"],
                "batch_name": row["batch_name"],
                "expected_huid": row["expected_huid"],
                "actual_huid": row["actual_huid"],
                "huid_match": bool(row["huid_match"]) if row["huid_match"] is not None else None,
                "status": row["item_status"],
                "image_url": row["image_url"],
                "purity_code": row["purity_code"],
                "karat": row["karat"],
                "purity_percentage": row["purity_percentage"],
                "confidence": row["confidence"],
                "decision": row["decision"],
                "rejection_reasons": json.loads(row["rejection_reasons"] or "[]"),
                "raw_ocr_text": row["raw_ocr_text"],
                "processed_image_url": row["processed_image_url"],
                "processing_time_ms": row["processing_time_ms"],
                "processed_at": row["processed_at"],
            }
        return None

    def get_batch_statistics(self, batch_id: int) -> dict:
        """Get statistics for a batch."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Total items
        cursor.execute("SELECT COUNT(*) as total FROM batch_items WHERE batch_id = ?", (batch_id,))
        total = cursor.fetchone()["total"]

        # Status counts
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM batch_items
            WHERE batch_id = ?
            GROUP BY status
        """, (batch_id,))
        status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

        # Decision counts
        cursor.execute("""
            SELECT decision, COUNT(*) as count
            FROM ocr_results
            WHERE batch_item_id IN (SELECT id FROM batch_items WHERE batch_id = ?)
            GROUP BY decision
        """, (batch_id,))
        decision_counts = {row["decision"]: row["count"] for row in cursor.fetchall()}

        # Match counts
        cursor.execute("""
            SELECT huid_match, COUNT(*) as count
            FROM ocr_results
            WHERE batch_item_id IN (SELECT id FROM batch_items WHERE batch_id = ?)
            GROUP BY huid_match
        """, (batch_id,))
        match_counts = {row["huid_match"]: row["count"] for row in cursor.fetchall()}

        # Average confidence
        cursor.execute("""
            SELECT AVG(confidence) as avg_conf
            FROM ocr_results
            WHERE batch_item_id IN (SELECT id FROM batch_items WHERE batch_id = ?)
        """, (batch_id,))
        avg_conf_row = cursor.fetchone()
        avg_confidence = avg_conf_row["avg_conf"] if avg_conf_row and avg_conf_row["avg_conf"] else 0.0

        conn.close()

        return {
            "total_items": total,
            "status_counts": status_counts,
            "decision_counts": decision_counts,
            "huid_matches": match_counts.get(1, 0),
            "huid_mismatches": match_counts.get(0, 0),
            "average_confidence": avg_confidence,
        }

    def update_batch_total(self, batch_id: int):
        """Update batch total_items count based on actual items."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE batches SET total_items = (
                SELECT COUNT(*) FROM batch_items WHERE batch_id = ?
            ) WHERE id = ?""",
            (batch_id, batch_id)
        )
        conn.commit()
        conn.close()

    # Item image operations
    def upsert_item_image(self, img: ItemImage) -> int:
        """Insert or replace an item image for (tag_id, image_type, slot)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO item_images
                   (batch_item_id, tag_id, image_type, slot, s3_key, s3_bucket,
                    content_type, size_bytes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(tag_id, image_type, slot) DO UPDATE SET
                   batch_item_id = excluded.batch_item_id,
                   s3_key        = excluded.s3_key,
                   s3_bucket     = excluded.s3_bucket,
                   content_type  = excluded.content_type,
                   size_bytes    = excluded.size_bytes,
                   uploaded_at   = CURRENT_TIMESTAMP""",
            (
                img.batch_item_id,
                img.tag_id,
                img.image_type,
                img.slot,
                img.s3_key,
                img.s3_bucket,
                img.content_type,
                img.size_bytes,
            ),
        )
        conn.commit()
        # On UPDATE lastrowid may be 0; fetch the authoritative id.
        cursor.execute(
            "SELECT id FROM item_images WHERE tag_id = ? AND image_type = ? AND slot = ?",
            (img.tag_id, img.image_type, img.slot),
        )
        row = cursor.fetchone()
        conn.close()
        return row["id"] if row else 0

    def get_item_images_for_tag(self, tag_id: str) -> List[ItemImage]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM item_images
               WHERE tag_id = ?
               ORDER BY CASE image_type WHEN 'huid' THEN 0 ELSE 1 END, slot""",
            (tag_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            ItemImage(
                id=row["id"],
                batch_item_id=row["batch_item_id"],
                tag_id=row["tag_id"],
                image_type=row["image_type"],
                slot=row["slot"],
                s3_key=row["s3_key"],
                s3_bucket=row["s3_bucket"] or "",
                content_type=row["content_type"],
                size_bytes=row["size_bytes"],
                uploaded_at=datetime.fromisoformat(row["uploaded_at"]) if row["uploaded_at"] else None,
            )
            for row in rows
        ]

    def get_item_image(self, tag_id: str, image_type: str, slot: int) -> Optional[ItemImage]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM item_images WHERE tag_id = ? AND image_type = ? AND slot = ?",
            (tag_id, image_type, slot),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return ItemImage(
            id=row["id"],
            batch_item_id=row["batch_item_id"],
            tag_id=row["tag_id"],
            image_type=row["image_type"],
            slot=row["slot"],
            s3_key=row["s3_key"],
            s3_bucket=row["s3_bucket"] or "",
            content_type=row["content_type"],
            size_bytes=row["size_bytes"],
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]) if row["uploaded_at"] else None,
        )

    def delete_item_image(self, tag_id: str, image_type: str, slot: int) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM item_images WHERE tag_id = ? AND image_type = ? AND slot = ?",
            (tag_id, image_type, slot),
        )
        changed = cursor.rowcount
        conn.commit()
        conn.close()
        return changed > 0

    def update_ocr_result_decision(
        self,
        tag_id: str,
        decision: 'QCDecision',
        rejection_reasons: List[str] = None,
        reviewer: Optional[str] = None
    ):
        """Update the decision for an OCR result (for manual review)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get current metadata
        cursor.execute(
            "SELECT metadata FROM ocr_results WHERE tag_id = ? ORDER BY created_at DESC LIMIT 1",
            (tag_id,)
        )
        row = cursor.fetchone()
        metadata = json.loads(row["metadata"] or "{}") if row else {}

        # Add reviewer info
        if reviewer:
            metadata["reviewer"] = reviewer
            metadata["reviewed_at"] = datetime.now().isoformat()

        cursor.execute(
            """UPDATE ocr_results
               SET decision = ?, rejection_reasons = ?, metadata = ?
               WHERE tag_id = ?""",
            (
                decision.value if hasattr(decision, 'value') else decision,
                json.dumps(rejection_reasons or []),
                json.dumps(metadata),
                tag_id
            )
        )
        conn.commit()
        conn.close()


# Global database instance
_db: Optional[DatabaseManager] = None


def get_database(db_path: str = "hallmark_qc.db") -> DatabaseManager:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = DatabaseManager(db_path)
    return _db
