"""
FastAPI OCR API - Hallmark QC Validation System.

Three-Stage Workflow:
1. Stage 1: Upload CSV/Excel with tag IDs and expected HUIDs
2. Stage 2: Upload images with tag ID for processing
3. Stage 3: Retrieve results by tag ID

Deploy on AWS or run locally with: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fix for PaddlePaddle PIR executor compatibility issue
# Must be set BEFORE importing any paddle modules
# These flags fix the "ConvertPirAttribute2RuntimeAttribute not support" error on Linux
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_enable_pir_with_pt_kernel"] = "0"
os.environ["FLAGS_pir_apply_inplace_pass"] = "0"

# Disable OneDNN/MKL-DNN which causes issues on some Linux systems (especially AWS)
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["MKLDNN_CACHE_CAPACITY"] = "0"
os.environ["FLAGS_tracer_mkldnn_ops_on"] = ""
os.environ["FLAGS_tracer_mkldnn_ops_off"] = "conv2d,batch_norm,pool2d"

# Additional Paddle stability flags
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.8"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logger.info(f"PIR flags set: FLAGS_enable_pir_api={os.environ.get('FLAGS_enable_pir_api')}, FLAGS_enable_pir_in_executor={os.environ.get('FLAGS_enable_pir_in_executor')}")

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from PIL import Image
from datetime import datetime
from starlette.middleware.sessions import SessionMiddleware
import io
import json
import time
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import secrets

# Load environment variables
load_dotenv()

# Import OCR engines
sys.path.insert(0, "src")
sys.path.insert(0, "config")

from ocr_model_v2 import OCREngineV2, OCRResultV2, HallmarkInfo, HallmarkType, CheckInfo
from database import (
    DatabaseManager, get_database, Batch, BatchItem, OCRResult as DBOCRResult,
    ItemImage, ProcessingStatus, QCDecision
)
from batch_parse import ParseError, parse_batch_file
from storage_service import StorageService, get_storage


app = FastAPI(
    title="Hallmark QC API",
    description="""
    AI-powered Hallmark Quality Control System with BIS compliance validation.

    ## Three-Stage Workflow

    ### Stage 1: Batch Upload
    Upload CSV/Excel files containing tag IDs and expected HUIDs.

    ### Stage 2: Image Processing
    Upload hallmark images with their corresponding tag IDs for OCR processing.

    ### Stage 3: Results Retrieval
    Retrieve OCR results, expected vs actual HUID comparison, and processed images.

    ## BIS Standards
    - Gold: IS 1417 (916, 750, 585, etc.)
    - Silver: IS 2112 (925, 999, etc.)
    - HUID: Mandatory 6-character alphanumeric code (since April 2023)
    """,
    version="4.0.0"
)

# Session middleware for authentication
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv('SESSION_SECRET', secrets.token_hex(32))
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Parse login users from environment variable (format: user1:pass1,user2:pass2)
login_users = {}
if os.getenv('LOGIN_USERS'):
    for user_pass in os.getenv('LOGIN_USERS').split(','):
        parts = user_pass.strip().split(':')
        if len(parts) == 2:
            login_users[parts[0].strip()] = parts[1].strip()

# Mount uploads directory for serving local images
uploads_dir = "./uploads"
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# Mount the dashboard SPA assets under /static (the HTML itself is served by
# the `/` route below so hash-based routing works on the root URL).
static_dir = "./static"
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize OCR engines at startup
ocr_engine_v2 = None
db: DatabaseManager = None
storage: StorageService = None


def compare_huids(actual_huid: str, expected_huid: str) -> bool:
    """
    Compare HUIDs flexibly - handles cases where expected_huid contains
    full hallmark text (e.g., '22K91677WAX9') and actual_huid is just
    the 6-character HUID (e.g., '77WAX9').
    """
    if not actual_huid or not expected_huid:
        return False

    actual = actual_huid.upper().strip()
    expected = expected_huid.upper().strip()

    # Exact match
    if actual == expected:
        return True

    # Check if actual HUID is contained at the end of expected (common case)
    # e.g., expected='22K91677WAX9', actual='77WAX9'
    if expected.endswith(actual):
        return True

    # Check if actual HUID is contained anywhere in expected
    if actual in expected:
        return True

    # Extract 6-char alphanumeric sequences from expected and compare
    import re
    expected_huids = re.findall(r'[A-Z0-9]{6}', expected)
    for exp_huid in expected_huids:
        # Skip pure numeric (likely purity codes like 916)
        if exp_huid.isdigit():
            continue
        if exp_huid == actual:
            return True

    return False


# S3 Client for presigned URLs
s3_client = None

def get_s3_client():
    """Get or create S3 client."""
    global s3_client
    if s3_client is None:
        s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION', 'ap-south-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    return s3_client


# Response Models
class RejectedRowResponse(BaseModel):
    row: int
    tag_id: str
    reason: str


class BatchUploadResponse(BaseModel):
    status: str
    batch_id: Optional[int] = None
    batch_name: Optional[str] = None
    total_rows: int
    accepted: int
    rejected: List[RejectedRowResponse] = []
    message: str


class ImageUploadResponse(BaseModel):
    status: str
    tag_id: str
    expected_huid: str
    actual_huid: Optional[str]
    huid_match: bool
    confidence: float
    decision: str
    message: str


class ResultResponse(BaseModel):
    status: str
    tag_id: str
    expected_huid: str
    actual_huid: Optional[str]
    huid_match: Optional[bool]
    purity_code: Optional[str]
    karat: Optional[str]
    purity_percentage: Optional[float]
    confidence: Optional[float]
    decision: Optional[str]
    rejection_reasons: List[str]
    raw_ocr_text: Optional[str]
    image_url: Optional[str]
    processed_image_url: Optional[str]
    processing_status: str


class BatchResultsResponse(BaseModel):
    status: str
    batch_id: int
    batch_name: str
    total_items: int
    processed_items: int
    statistics: Dict
    results: List[Dict]


class HallmarkData(BaseModel):
    purity_code: Optional[str] = None
    karat: Optional[str] = None
    purity_percentage: Optional[float] = None
    huid: Optional[str] = None
    bis_certified: bool = False


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global ocr_engine_v2, db, storage
    ocr_engine_v2 = OCREngineV2(enable_preprocessing=True)
    db = get_database()
    storage = get_storage()
    print(f"Storage type: {storage.storage_type}")


# =============================================================================
# Homepage & Health Endpoints
# =============================================================================

@app.get("/", response_class=FileResponse)
async def homepage():
    """Serve the dashboard SPA shell."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return RedirectResponse(url="/docs")


@app.get("/api/health")
async def api_health():
    """API health check endpoint."""
    return {
        "status": "ok",
        "message": "Hallmark QC API is running",
        "version": "4.0.0",
        "storage": storage.storage_type if storage else "not_initialized"
    }


@app.get("/health")
async def health():
    """Health check for deployment."""
    return {"status": "healthy"}


# =============================================================================
# STAGE 1: Batch Upload (CSV/Excel with tag IDs and expected HUIDs)
# =============================================================================

@app.post("/stage1/upload-batch", response_model=BatchUploadResponse)
async def upload_batch(
    file: UploadFile = File(...),
    batch_name: Optional[str] = Form(None),
    strict: bool = Query(False, description="When true, reject the whole batch if any row is invalid."),
):
    """Stage 1: Upload a CSV/Excel file of tag IDs + expected HUIDs.

    * Accepts ``.csv``, ``.xlsx``, ``.xls`` (415 otherwise).
    * Excel: auto-selects a sheet named ``HUID``/``PRINT`` (case-insensitive) or the first sheet.
    * Headers are normalized case/whitespace-insensitively; aliases like ``tag no``, ``HUID ID``, ``tag_number`` are mapped.
    * Validates each row. ``strict=true`` rejects the whole batch on any invalid row (422); otherwise valid rows are inserted and invalid rows returned.
    """
    contents = await file.read()

    try:
        parsed = parse_batch_file(contents, file.filename or "")
    except ParseError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.message, **e.extra},
        )

    if strict and parsed.rejected:
        return JSONResponse(
            status_code=422,
            content={
                "status": "rejected",
                "batch_id": None,
                "batch_name": None,
                "total_rows": parsed.total_rows,
                "accepted": 0,
                "rejected": [r.to_dict() for r in parsed.rejected],
                "message": "strict mode: batch rejected because some rows failed validation",
            },
        )

    if not parsed.accepted:
        return JSONResponse(
            status_code=422,
            content={
                "status": "rejected",
                "batch_id": None,
                "batch_name": None,
                "total_rows": parsed.total_rows,
                "accepted": 0,
                "rejected": [r.to_dict() for r in parsed.rejected],
                "message": "no valid rows found in the file",
            },
        )

    resolved_name = batch_name or (file.filename or f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    items = [
        BatchItem(
            batch_id=0,  # set by create_batch_with_items
            tag_id=row.tag_id,
            expected_huid=row.expected_huid,
            status=ProcessingStatus.PENDING,
        )
        for row in parsed.accepted
    ]
    batch = Batch(batch_name=resolved_name, total_items=len(items), status="pending")
    batch_id = db.create_batch_with_items(batch, items)

    return BatchUploadResponse(
        status="success",
        batch_id=batch_id,
        batch_name=resolved_name,
        total_rows=parsed.total_rows,
        accepted=len(parsed.accepted),
        rejected=[RejectedRowResponse(**r.to_dict()) for r in parsed.rejected],
        message=f"Uploaded {len(parsed.accepted)} of {parsed.total_rows} rows",
    )


@app.get("/stage1/batches")
async def list_batches():
    """List all uploaded batches."""
    batches = db.get_all_batches()
    return {
        "status": "success",
        "batches": [b.to_dict() for b in batches]
    }


@app.get("/stage1/batch/{batch_id}")
async def get_batch_details(batch_id: int):
    """Get details of a specific batch."""
    batch = db.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    items = db.get_batch_items(batch_id)
    stats = db.get_batch_statistics(batch_id)

    return {
        "status": "success",
        "batch": batch.to_dict(),
        "statistics": stats,
        "items": [item.to_dict() for item in items]
    }


# =============================================================================
# STAGE 2: Image Upload and Processing
# =============================================================================

def _persist_item_image(
    batch_item: BatchItem,
    file_bytes: bytes,
    filename: str,
    image_type: str,
    slot: int,
) -> ItemImage:
    """Write an image to structured storage and upsert the item_images row.

    Also updates batch_items.image_path/image_url when image_type='huid' so
    the legacy single-image fields stay in sync with the gallery.
    """
    s3_key, url, content_type = storage.upload_hallmark_image(
        file_bytes,
        batch_id=batch_item.batch_id,
        tag_id=batch_item.tag_id,
        image_type=image_type,
        slot=slot,
        filename=filename,
    )

    img = ItemImage(
        batch_item_id=batch_item.id,
        tag_id=batch_item.tag_id,
        image_type=image_type,
        slot=slot,
        s3_key=s3_key,
        s3_bucket=storage.config.s3_bucket if storage.use_s3 else "",
        content_type=content_type,
        size_bytes=len(file_bytes),
    )
    img.id = db.upsert_item_image(img)

    if image_type == "huid":
        db.update_batch_item_image(batch_item.tag_id, s3_key, url)

    return img


@app.post("/stage2/upload-image", response_model=ImageUploadResponse)
async def upload_and_process_image(
    file: UploadFile = File(...),
    tag_id: str = Form(...),
    image_type: str = Form("huid"),
    slot: int = Form(0),
):
    """
    Stage 2: Upload an image for a specific tag ID.

    When ``image_type='huid'`` (default) the image is OCR'd and compared
    against the expected HUID. When ``image_type='artifact'`` with
    ``slot in {1,2,3}`` the image is stored alongside the HUID without OCR
    — use it for secondary views of the same piece. HUID images always use
    slot 0.
    """
    # Validate inputs
    if image_type not in ("huid", "artifact"):
        raise HTTPException(status_code=400, detail="image_type must be 'huid' or 'artifact'")
    if image_type == "huid" and slot != 0:
        raise HTTPException(status_code=400, detail="HUID images must use slot=0")
    if image_type == "artifact" and slot not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="artifact slot must be 1, 2, or 3")

    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: PNG, JPG, JPEG, BMP, WEBP"
        )

    batch_item = db.get_batch_item_by_tag(tag_id)
    if not batch_item:
        raise HTTPException(
            status_code=404,
            detail=f"Tag ID '{tag_id}' not found. Please upload batch data first (Stage 1)."
        )

    try:
        start_time = time.time()

        contents = await file.read()

        # Store the file first (both HUID and artifact paths go here).
        _persist_item_image(batch_item, contents, file.filename, image_type, slot)

        # Artifacts are stored for reference only — no OCR.
        if image_type == "artifact":
            return ImageUploadResponse(
                status="success",
                tag_id=tag_id,
                expected_huid=batch_item.expected_huid,
                actual_huid=None,
                huid_match=False,
                confidence=0.0,
                decision="pending",
                message=f"Artifact image stored at slot {slot}",
            )

        # HUID path — run OCR + comparison + decision.
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        hallmark_info = ocr_engine_v2.extract_with_hallmark_info(image)

        # Extract actual HUID
        actual_huid = hallmark_info.huid
        expected_huid = batch_item.expected_huid

        # Compare HUIDs (flexible matching)
        huid_match = compare_huids(actual_huid, expected_huid)

        # Determine decision
        decision = QCDecision.PENDING
        rejection_reasons = []

        purity_valid = hallmark_info.purity_code is not None
        huid_valid = actual_huid is not None

        if not purity_valid:
            rejection_reasons.append("missing_purity_mark")
        if not huid_valid:
            rejection_reasons.append("missing_huid")
        if actual_huid and not huid_match:
            rejection_reasons.append("huid_mismatch")

        # Decision logic
        if purity_valid and huid_valid and huid_match:
            if hallmark_info.overall_confidence >= 0.85:
                decision = QCDecision.APPROVED
            elif hallmark_info.overall_confidence >= 0.50:
                decision = QCDecision.MANUAL_REVIEW
            else:
                decision = QCDecision.REJECTED
                rejection_reasons.append("low_confidence")
        elif purity_valid and huid_valid and not huid_match:
            decision = QCDecision.REJECTED
        else:
            if hallmark_info.overall_confidence < 0.50:
                decision = QCDecision.REJECTED
            else:
                decision = QCDecision.MANUAL_REVIEW

        processing_time = int((time.time() - start_time) * 1000)

        # Save processed image with annotations (optional)
        processed_image_path = None
        processed_image_url = None

        # Create OCR result record
        ocr_result = DBOCRResult(
            batch_item_id=batch_item.id,
            tag_id=tag_id,
            expected_huid=expected_huid,
            actual_huid=actual_huid,
            huid_match=huid_match,
            purity_code=hallmark_info.purity_code,
            karat=hallmark_info.karat,
            purity_percentage=hallmark_info.purity_percentage,
            confidence=hallmark_info.overall_confidence,
            decision=decision,
            rejection_reasons=rejection_reasons,
            raw_ocr_text=" ".join([r.text for r in hallmark_info.all_results]),
            processed_image_path=processed_image_path,
            processed_image_url=processed_image_url,
            processing_time_ms=processing_time,
        )
        db.create_ocr_result(ocr_result)

        # Update batch item status
        db.update_batch_item_status(tag_id, ProcessingStatus.COMPLETED)

        # Update batch progress
        batch = db.get_batch(batch_item.batch_id)
        if batch:
            db.update_batch_progress(batch.id, batch.processed_items + 1)

        return ImageUploadResponse(
            status="success",
            tag_id=tag_id,
            expected_huid=expected_huid,
            actual_huid=actual_huid,
            huid_match=huid_match,
            confidence=round(hallmark_info.overall_confidence, 3),
            decision=decision.value,
            message="Image processed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        # Update status to failed
        db.update_batch_item_status(tag_id, ProcessingStatus.FAILED)
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


@app.post("/stage2/upload-image-bulk")
async def upload_images_bulk(
    files: List[UploadFile] = File(...),
):
    """
    Stage 2 (Bulk): Upload multiple images at once.

    Image filenames should contain the tag_id (e.g., TAG001.jpg, TAG002.png).
    """
    results = []
    errors = []

    for file in files:
        # Extract tag_id from filename
        filename = os.path.splitext(file.filename)[0]
        tag_id = filename.strip()

        try:
            # Check if tag exists
            batch_item = db.get_batch_item_by_tag(tag_id)
            if not batch_item:
                errors.append({
                    "filename": file.filename,
                    "tag_id": tag_id,
                    "error": "Tag ID not found in batch data"
                })
                continue

            # Process the image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Store in the structured gallery layout (also updates batch_items).
            _persist_item_image(batch_item, contents, file.filename, "huid", 0)

            # Process OCR
            hallmark_info = ocr_engine_v2.extract_with_hallmark_info(image)
            actual_huid = hallmark_info.huid
            huid_match = compare_huids(actual_huid, batch_item.expected_huid)

            # Simple decision
            if hallmark_info.overall_confidence >= 0.85 and huid_match:
                decision = QCDecision.APPROVED
            elif hallmark_info.overall_confidence >= 0.50:
                decision = QCDecision.MANUAL_REVIEW
            else:
                decision = QCDecision.REJECTED

            # Save result
            ocr_result = DBOCRResult(
                batch_item_id=batch_item.id,
                tag_id=tag_id,
                expected_huid=batch_item.expected_huid,
                actual_huid=actual_huid,
                huid_match=huid_match,
                purity_code=hallmark_info.purity_code,
                confidence=hallmark_info.overall_confidence,
                decision=decision,
                raw_ocr_text=" ".join([r.text for r in hallmark_info.all_results]),
            )
            db.create_ocr_result(ocr_result)
            db.update_batch_item_status(tag_id, ProcessingStatus.COMPLETED)

            results.append({
                "tag_id": tag_id,
                "status": "success",
                "huid_match": huid_match,
                "decision": decision.value
            })

        except Exception as e:
            errors.append({
                "filename": file.filename,
                "tag_id": tag_id,
                "error": str(e)
            })

    return {
        "status": "completed",
        "processed": len(results),
        "errors": len(errors),
        "results": results,
        "error_details": errors
    }


@app.post("/stage2/upload-artifact")
async def upload_artifact(
    file: UploadFile = File(...),
    tag_id: str = Form(...),
    slot: int = Form(...),
):
    """Upload a non-HUID artifact image for a tag (slots 1, 2, or 3)."""
    if slot not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="artifact slot must be 1, 2, or 3")

    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: PNG, JPG, JPEG, BMP, WEBP",
        )

    batch_item = db.get_batch_item_by_tag(tag_id)
    if not batch_item:
        raise HTTPException(status_code=404, detail=f"Tag ID '{tag_id}' not found")

    contents = await file.read()
    img = _persist_item_image(batch_item, contents, file.filename, "artifact", slot)

    return {
        "status": "success",
        "tag_id": tag_id,
        "slot": slot,
        "s3_key": img.s3_key,
        "url": storage.presign_get(img.s3_key),
        "size_bytes": img.size_bytes,
    }


@app.delete("/stage2/artifact/{tag_id}/{slot}")
async def delete_artifact(tag_id: str, slot: int):
    """Delete an artifact image for a tag at the given slot (1, 2, or 3)."""
    if slot not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="artifact slot must be 1, 2, or 3")

    existing = db.get_item_image(tag_id, "artifact", slot)
    if not existing:
        raise HTTPException(status_code=404, detail="artifact not found")

    storage.delete_by_key(existing.s3_key)
    db.delete_item_image(tag_id, "artifact", slot)
    return {"status": "success", "tag_id": tag_id, "slot": slot}


# =============================================================================
# STAGE 3: Results Retrieval
# =============================================================================

def _resolve_image_urls(images: List[ItemImage]) -> Dict:
    """Turn a list of ItemImage rows into the shape used by the preview UI."""
    huid_entry = None
    artifacts = []
    for img in images:
        entry = {
            "s3_key": img.s3_key,
            "url": storage.presign_get(img.s3_key),
            "content_type": img.content_type,
            "size_bytes": img.size_bytes,
            "uploaded_at": img.uploaded_at.isoformat() if img.uploaded_at else None,
        }
        if img.image_type == "huid":
            huid_entry = entry
        else:
            artifacts.append({"slot": img.slot, **entry})
    artifacts.sort(key=lambda a: a["slot"])
    return {"huid": huid_entry, "artifacts": artifacts}


def _thumbnail_url_for_tag(tag_id: str) -> Optional[str]:
    huid = db.get_item_image(tag_id, "huid", 0)
    if not huid:
        return None
    return storage.presign_get(huid.s3_key)


@app.get("/stage3/item/{tag_id}")
async def get_item_detail(tag_id: str):
    """Full item payload — metadata + HUID + artifact images with signed URLs."""
    result = db.get_full_result_by_tag(tag_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Tag ID '{tag_id}' not found")

    images = _resolve_image_urls(db.get_item_images_for_tag(tag_id))

    return {
        "status": "success",
        "tag_id": result["tag_id"],
        "batch": {
            "id": result.get("batch_id"),
            "name": result.get("batch_name"),
        },
        "expected_huid": result["expected_huid"],
        "actual_huid": result.get("actual_huid"),
        "huid_match": result.get("huid_match"),
        "purity_code": result.get("purity_code"),
        "karat": result.get("karat"),
        "purity_percentage": result.get("purity_percentage"),
        "confidence": result.get("confidence"),
        "decision": result.get("decision"),
        "rejection_reasons": result.get("rejection_reasons", []),
        "raw_ocr_text": result.get("raw_ocr_text"),
        "processing_status": result.get("status", "pending"),
        "images": images,
        "timestamps": {
            "created_at": None,
            "processed_at": result.get("processed_at"),
            "processing_time_ms": result.get("processing_time_ms"),
        },
    }


@app.get("/stage3/result/{tag_id}", response_model=ResultResponse)
async def get_result_by_tag(tag_id: str):
    """
    Stage 3: Get complete result for a specific tag ID.

    Returns:
    - Expected HUID (from batch upload)
    - Actual HUID (from OCR)
    - Match status
    - OCR confidence and decision
    - Original and processed image URLs
    """
    result = db.get_full_result_by_tag(tag_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Tag ID '{tag_id}' not found"
        )

    return ResultResponse(
        status="success",
        tag_id=result["tag_id"],
        expected_huid=result["expected_huid"],
        actual_huid=result.get("actual_huid"),
        huid_match=result.get("huid_match"),
        purity_code=result.get("purity_code"),
        karat=result.get("karat"),
        purity_percentage=result.get("purity_percentage"),
        confidence=result.get("confidence"),
        decision=result.get("decision"),
        rejection_reasons=result.get("rejection_reasons", []),
        raw_ocr_text=result.get("raw_ocr_text"),
        image_url=result.get("image_url"),
        processed_image_url=result.get("processed_image_url"),
        processing_status=result.get("status", "pending")
    )


@app.get("/stage3/batch/{batch_id}/results", response_model=BatchResultsResponse)
async def get_batch_results(batch_id: int):
    """
    Stage 3: Get all results for a batch.

    Returns complete statistics and individual results for all items in the batch.
    """
    batch = db.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    results = db.get_results_by_batch(batch_id)
    for row in results:
        row["thumbnail_url"] = _thumbnail_url_for_tag(row["tag_id"])
    stats = db.get_batch_statistics(batch_id)

    return BatchResultsResponse(
        status="success",
        batch_id=batch_id,
        batch_name=batch.batch_name,
        total_items=batch.total_items,
        processed_items=stats.get("status_counts", {}).get("completed", 0),
        statistics=stats,
        results=results
    )


@app.get("/stage3/search")
async def search_results(
    tag_id: Optional[str] = Query(None, description="Search by tag ID"),
    huid: Optional[str] = Query(None, description="Search by HUID"),
    decision: Optional[str] = Query(None, description="Filter by decision"),
    batch_id: Optional[int] = Query(None, description="Filter by batch"),
):
    """Search and filter results."""
    if tag_id:
        result = db.get_full_result_by_tag(tag_id)
        if result:
            result["thumbnail_url"] = _thumbnail_url_for_tag(tag_id)
            return {"status": "success", "results": [result]}
        return {"status": "success", "results": []}

    if batch_id:
        results = db.get_results_by_batch(batch_id)
        if decision:
            results = [r for r in results if r.get("decision") == decision]
        for row in results:
            row["thumbnail_url"] = _thumbnail_url_for_tag(row["tag_id"])
        return {"status": "success", "results": results}

    return {"status": "success", "results": [], "message": "Please provide search criteria"}


# =============================================================================
# S3 Presigned URL Endpoints (for direct browser uploads)
# =============================================================================

class PresignedUrlRequest(BaseModel):
    fileName: str
    contentType: Optional[str] = "application/octet-stream"


class PresignedUrlResponse(BaseModel):
    uploadUrl: str
    key: str
    fileName: str


@app.post("/api/get-upload-url", response_model=PresignedUrlResponse)
async def get_upload_url(request: PresignedUrlRequest):
    """
    Get a presigned URL for direct S3 upload (authenticated uploads).

    This endpoint generates a presigned URL that allows direct upload
    to S3 from the browser without routing through the server.
    """
    try:
        client = get_s3_client()
        bucket_name = os.getenv('S3_BUCKET_NAME')
        folder_path = os.getenv('S3_FOLDER_PATH', 'uploads/')

        if not bucket_name:
            raise HTTPException(status_code=500, detail="S3 bucket not configured")

        timestamp = int(time.time() * 1000)
        unique_filename = f"{timestamp}-{request.fileName}"
        s3_key = f"{folder_path}{unique_filename}"

        presigned_url = client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': bucket_name,
                'Key': s3_key,
                'ContentType': request.contentType
            },
            ExpiresIn=3600  # URL valid for 1 hour
        )

        return PresignedUrlResponse(
            uploadUrl=presigned_url,
            key=s3_key,
            fileName=unique_filename
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {str(e)}")


@app.post("/api/get-ocr-upload-url", response_model=PresignedUrlResponse)
async def get_ocr_upload_url(request: PresignedUrlRequest):
    """
    Get a presigned URL for OCR image uploads (no auth required).

    This endpoint is specifically for uploading hallmark images for OCR processing.
    Images are stored in a dedicated OCR folder in S3.
    """
    try:
        client = get_s3_client()
        bucket_name = os.getenv('S3_BUCKET_NAME')
        folder_path = os.getenv('S3_OCR_FOLDER_PATH', 'ocr-images/')

        if not bucket_name:
            raise HTTPException(status_code=500, detail="S3 bucket not configured")

        timestamp = int(time.time() * 1000)
        unique_filename = f"{timestamp}-{request.fileName}"
        s3_key = f"{folder_path}{unique_filename}"

        # Default to image/jpeg for OCR uploads
        content_type = request.contentType if request.contentType != "application/octet-stream" else "image/jpeg"

        presigned_url = client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': bucket_name,
                'Key': s3_key,
                'ContentType': content_type
            },
            ExpiresIn=3600  # URL valid for 1 hour
        )

        return PresignedUrlResponse(
            uploadUrl=presigned_url,
            key=s3_key,
            fileName=unique_filename
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {str(e)}")


# =============================================================================
# Direct OCR Extraction
# =============================================================================

@app.post("/extract/v2")
async def extract_text_v2(file: UploadFile = File(...)):
    """Legacy: Extract text with hallmark detection."""
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        hallmark_info = ocr_engine_v2.extract_with_hallmark_info(image)

        # Build individual detection results
        detections = []
        for r in hallmark_info.all_results:
            detections.append({
                "text": r.text,
                "confidence": round(r.confidence, 4),
                "type": r.hallmark_type.value,
                "validated": r.validated,
            })

        return {
            "success": True,
            "full_text": " ".join([r.text for r in hallmark_info.all_results]),
            "average_confidence": hallmark_info.overall_confidence,
            "hallmark": {
                "purity_code": hallmark_info.purity_code,
                "karat": hallmark_info.karat,
                "purity_percentage": hallmark_info.purity_percentage,
                "huid": hallmark_info.huid,
                "bis_certified": hallmark_info.bis_certified
            },
            "detections": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.get("/qc/rules")
async def get_qc_rules():
    """Get current QC validation rules and BIS compliance standards."""
    return {
        "bis_standards": {
            "gold_grades": {
                "375": {"karat": "9K", "purity": 37.5},
                "585": {"karat": "14K", "purity": 58.5},
                "750": {"karat": "18K", "purity": 75.0},
                "875": {"karat": "21K", "purity": 87.5},
                "916": {"karat": "22K", "purity": 91.6},
                "958": {"karat": "23K", "purity": 95.8},
                "999": {"karat": "24K", "purity": 99.9},
            },
            "silver_grades": {
                "800": {"purity": 80.0, "grade": "Silver 800"},
                "925": {"purity": 92.5, "grade": "Sterling Silver"},
                "950": {"purity": 95.0, "grade": "Britannia Silver"},
                "999": {"purity": 99.9, "grade": "Fine Silver"},
            },
            "huid_required": True,
            "huid_format": "6 alphanumeric characters",
        },
        "validation_rules": {
            "auto_approve_confidence": 0.85,
            "auto_reject_confidence": 0.50,
            "manual_review_range": [0.50, 0.85],
        }
    }


@app.post("/validate/huid")
async def validate_huid(huid: str = Form(...)):
    """Validate HUID format."""
    import re

    cleaned = huid.upper().strip()
    cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)

    is_valid = (
        len(cleaned) == 6 and
        bool(re.match(r'^[A-Z0-9]{6}$', cleaned)) and
        bool(re.search(r'[A-Z]', cleaned))
    )

    return {
        "huid": huid,
        "cleaned": cleaned,
        "valid": is_valid,
        "errors": [] if is_valid else [
            "Invalid format" if len(cleaned) != 6 else None,
            "Must contain at least one letter" if not re.search(r'[A-Z]', cleaned) else None,
        ]
    }


# =============================================================================
# Downloads dashboard — folder-per-tag manifest + zip export
# =============================================================================

import io as _io
import zipfile as _zipfile
from fastapi.responses import StreamingResponse as _StreamingResponse


def _downloads_query(
    *,
    batch_id: Optional[int] = None,
    status_filter: Optional[List[str]] = None,
    decision_filter: Optional[List[str]] = None,
    match: Optional[str] = None,   # 'yes' | 'no' | None
    q: Optional[str] = None,
    limit_tags: Optional[List[str]] = None,
) -> List[dict]:
    """Return every tag that has at least one image, with its images + metadata.

    Filters are applied at the SQL level where cheap; tag-substring search
    and image attachment happen in Python so the joins stay simple.
    """
    conn = db._get_connection()
    try:
        params: list = []
        clauses: list[str] = ["EXISTS (SELECT 1 FROM item_images ii WHERE ii.tag_id = bi.tag_id)"]
        if batch_id is not None:
            clauses.append("bi.batch_id = ?"); params.append(batch_id)
        if status_filter:
            placeholders = ",".join("?" * len(status_filter))
            clauses.append(f"bi.status IN ({placeholders})")
            params.extend(status_filter)
        if q:
            clauses.append("bi.tag_id LIKE ?")
            params.append(f"%{q}%")
        if limit_tags:
            placeholders = ",".join("?" * len(limit_tags))
            clauses.append(f"bi.tag_id IN ({placeholders})")
            params.extend(limit_tags)

        rows = conn.execute(
            f"""
            SELECT bi.tag_id, bi.batch_id, bi.expected_huid, bi.status,
                   b.batch_name,
                   ocr.actual_huid, ocr.huid_match, ocr.decision,
                   ocr.rejection_reasons
              FROM batch_items bi
              LEFT JOIN batches b ON bi.batch_id = b.id
              LEFT JOIN ocr_results ocr
                ON ocr.id = (
                     SELECT id FROM ocr_results
                      WHERE tag_id = bi.tag_id
                      ORDER BY created_at DESC LIMIT 1
                   )
             WHERE {' AND '.join(clauses)}
             ORDER BY bi.tag_id
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    # Decision + match filters happen post-query to keep SQL simple.
    tags: list[dict] = []
    for row in rows:
        decision = row["decision"]
        huid_match = None if row["huid_match"] is None else bool(row["huid_match"])
        if decision_filter and decision not in decision_filter:
            continue
        if match == "yes" and huid_match is not True:
            continue
        if match == "no" and huid_match is not False:
            continue
        tags.append({
            "tag_id": row["tag_id"],
            "batch_id": row["batch_id"],
            "batch_name": row["batch_name"],
            "expected_huid": row["expected_huid"],
            "actual_huid": row["actual_huid"],
            "huid_match": huid_match,
            "status": row["status"],
            "decision": decision,
            "rejection_reasons": json.loads(row["rejection_reasons"] or "[]") if row["rejection_reasons"] else [],
        })

    # Attach images per tag and compute aggregates.
    for t in tags:
        images = db.get_item_images_for_tag(t["tag_id"])
        t["images"] = [
            {
                "image_type": img.image_type,
                "slot": img.slot,
                "s3_key": img.s3_key,
                "filename": os.path.basename(img.s3_key),
                "size_bytes": img.size_bytes,
                "content_type": img.content_type,
                "uploaded_at": img.uploaded_at.isoformat() if img.uploaded_at else None,
                "url": storage.presign_get(img.s3_key),
            }
            for img in images
        ]
        t["image_count"] = len(t["images"])
        t["total_size_bytes"] = sum(img.get("size_bytes") or 0 for img in t["images"])
    return tags


@app.get("/stage3/downloads/manifest")
async def downloads_manifest(
    batch_id: Optional[int] = Query(None),
    status: Optional[List[str]] = Query(None, description="Repeat for multiple values"),
    decision: Optional[List[str]] = Query(None, description="Repeat for multiple values"),
    match: Optional[str] = Query(None, pattern="^(yes|no)?$"),
    q: Optional[str] = Query(None, description="Tag ID substring search"),
):
    """Filtered list of tags (with images) for the downloads screen."""
    tags = _downloads_query(
        batch_id=batch_id,
        status_filter=status,
        decision_filter=decision,
        match=match if match else None,
        q=q.strip() if q else None,
    )
    total_images = sum(t["image_count"] for t in tags)
    total_bytes = sum(t["total_size_bytes"] for t in tags)
    return {
        "count": len(tags),
        "total_images": total_images,
        "total_size_bytes": total_bytes,
        "tags": tags,
    }


def _build_zip_response(tag_ids: List[str]) -> _StreamingResponse:
    """Build a zip of every image for the given tags (folder per tag).

    File names preserve their on-disk name ({uuid}{ext}) so they remain
    traceable back to the original S3 key. Tags that are missing on disk
    are silently skipped rather than failing the whole zip.
    """
    if not tag_ids:
        raise HTTPException(status_code=400, detail="tag_ids is empty")

    buf = _io.BytesIO()
    included = 0
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as zf:
        for tag_id in tag_ids:
            images = db.get_item_images_for_tag(tag_id)
            for img in images:
                data = storage.get_image(img.s3_key)
                if data is None:
                    continue
                basename = os.path.basename(img.s3_key)
                zf.writestr(f"{tag_id}/{basename}", data)
                included += 1

    if included == 0:
        raise HTTPException(status_code=404, detail="No images found for the requested tags")

    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"hallmark-images-{ts}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return _StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.get("/stage3/downloads/zip/{tag_id}")
async def download_tag_zip(tag_id: str):
    """Download all images for a single tag as a zip (tag_id/ folder inside)."""
    return _build_zip_response([tag_id])


class BulkDownloadRequest(BaseModel):
    tag_ids: List[str]


@app.post("/stage3/downloads/zip")
async def download_bulk_zip(body: BulkDownloadRequest):
    """Download zip with a folder per selected tag."""
    return _build_zip_response(body.tag_ids)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
