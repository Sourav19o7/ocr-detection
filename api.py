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

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Request, Depends, Body
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
from config.manak_parser import parse_manak_file, compare_with_ocr, ManakParseResult, extract_purity


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


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Force revalidation on static assets so stale CSS/JS can't bite users
    after an edit. ETag + Last-Modified still make 304 responses cheap."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path == "/":
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

# Initialize OCR engines at startup
ocr_engine_v2 = None
db: DatabaseManager = None
storage: StorageService = None


def compare_huids(actual_huid: str, expected_huid: str) -> bool:
    """
    Compare HUIDs flexibly - handles cases where expected_huid contains
    full hallmark text (e.g., '22K91677WAX9') and actual_huid is just
    the 6-character HUID (e.g., '77WAX9').

    Also handles spaces in HUIDs - e.g., expected='ABC 123' matches actual='ABC123'
    """
    if not actual_huid or not expected_huid:
        return False

    actual = actual_huid.upper().strip()
    expected = expected_huid.upper().strip()

    # Normalize: remove all spaces for comparison
    actual_normalized = actual.replace(" ", "")
    expected_normalized = expected.replace(" ", "")

    # Exact match (with spaces removed)
    if actual_normalized == expected_normalized:
        return True

    # Check if actual HUID is contained at the end of expected (common case)
    # e.g., expected='22K91677WAX9', actual='77WAX9'
    if expected_normalized.endswith(actual_normalized):
        return True

    # Check if actual HUID is contained anywhere in expected
    if actual_normalized in expected_normalized:
        return True

    # Extract 6-char alphanumeric sequences from expected and compare
    import re
    expected_huids = re.findall(r'[A-Z0-9]{6}', expected_normalized)
    for exp_huid in expected_huids:
        # Skip pure numeric (likely purity codes like 916)
        if exp_huid.isdigit():
            continue
        if exp_huid == actual_normalized:
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


class DecisionRequest(BaseModel):
    decision: str  # "approved" | "rejected" | "manual_review"
    rejection_reasons: Optional[List[str]] = None
    reviewer: Optional[str] = None


@app.post("/stage3/item/{tag_id}/decision")
async def set_item_decision(tag_id: str, body: DecisionRequest):
    """Operator action from the item preview — approve / reject / re-capture."""
    # Validate decision enum
    try:
        decision_enum = QCDecision(body.decision)
    except ValueError:
        allowed = [d.value for d in QCDecision]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision '{body.decision}'. Allowed: {allowed}",
        )

    # Confirm the tag exists and has an OCR result to update
    existing = db.get_ocr_result_by_tag(tag_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"No OCR result for tag '{tag_id}'. Upload a HUID image first.",
        )

    # Merge rejection_reasons: if reviewer passed any, use those; otherwise
    # keep what was already there.
    reasons = body.rejection_reasons
    if reasons is None:
        reasons = existing.rejection_reasons

    db.update_ocr_result_decision(
        tag_id=tag_id,
        decision=decision_enum,
        rejection_reasons=reasons,
        reviewer=body.reviewer,
    )

    return {
        "status": "success",
        "tag_id": tag_id,
        "decision": decision_enum.value,
        "rejection_reasons": reasons,
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


@app.get("/stage3/search/tags")
async def search_tag_ids(
    q: str = Query(..., min_length=1, description="Search query for tag ID or HUID"),
    limit: int = Query(10, ge=1, le=50, description="Max results to return"),
):
    """
    Search for items by tag ID or expected HUID with partial matching.
    Returns matching items with tag_id and expected_huid, sorted by relevance:
    - Exact tag_id match first
    - Exact HUID match
    - Tags with suffixes like _1, _2 (sorted numerically)
    - Other prefix/contains matches
    """
    conn = db._get_connection()
    cursor = conn.cursor()

    # Search for tags matching by tag_id OR expected_huid
    # Include decision, huid_match, and processed_at from ocr_results via LEFT JOIN
    cursor.execute(
        """SELECT bi.tag_id, bi.expected_huid,
                  ocr.decision, ocr.huid_match, ocr.confidence, ocr.created_at as processed_at
           FROM batch_items bi
           LEFT JOIN ocr_results ocr ON ocr.batch_item_id = bi.id
           WHERE bi.tag_id LIKE ? OR bi.tag_id LIKE ? OR bi.expected_huid LIKE ?
           ORDER BY
             CASE
               WHEN bi.tag_id = ? THEN 0
               WHEN bi.expected_huid = ? THEN 1
               WHEN bi.tag_id LIKE ? THEN 2
               WHEN bi.expected_huid LIKE ? THEN 3
               WHEN bi.tag_id LIKE ? THEN 4
               ELSE 5
             END,
             bi.tag_id
           LIMIT ?""",
        (f"%{q}%", f"{q}_%", f"%{q}%", q, q, f"{q}_%", f"{q}%", f"{q}%", limit)
    )
    rows = cursor.fetchall()
    conn.close()

    results = [{
        "tag_id": row["tag_id"],
        "expected_huid": row["expected_huid"],
        "decision": row["decision"] or "pending",
        "huid_match": bool(row["huid_match"]) if row["huid_match"] is not None else None,
        "confidence": row["confidence"],
        "processed_at": row["processed_at"]
    } for row in rows]

    # Sort tags with numeric suffixes properly (e.g., _1, _2, _10 instead of _1, _10, _2)
    def sort_key(item):
        tag = item["tag_id"]
        huid = item["expected_huid"] or ""

        # Exact tag match
        if tag.lower() == q.lower():
            return (0, 0, tag)
        # Exact HUID match
        if huid.lower() == q.lower():
            return (1, 0, tag)
        # Check if tag has a numeric suffix like _1, _2
        if tag.lower().startswith(q.lower()) and "_" in tag[len(q):]:
            suffix = tag[len(q):]
            if suffix.startswith("_"):
                try:
                    num = int(suffix[1:])
                    return (2, num, tag)
                except ValueError:
                    pass
        # Tag starts with query
        if tag.lower().startswith(q.lower()):
            return (3, 0, tag)
        # HUID starts with query
        if huid.lower().startswith(q.lower()):
            return (4, 0, tag)
        # Tag contains query
        if q.lower() in tag.lower():
            return (5, 0, tag)
        # HUID contains query
        return (6, 0, tag)

    results.sort(key=sort_key)

    return {
        "status": "success",
        "query": q,
        "results": results[:limit],
        # Keep backward compatibility
        "tags": [r["tag_id"] for r in results[:limit]]
    }


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


# =============================================================================
# EXTERNAL APIs for BAC Team / Camera Integration
# =============================================================================

# API Key authentication for external endpoints
EXTERNAL_API_KEYS = set()
if os.getenv('EXTERNAL_API_KEYS'):
    EXTERNAL_API_KEYS = set(k.strip() for k in os.getenv('EXTERNAL_API_KEYS').split(',') if k.strip())

# Also accept a single key for simplicity
if os.getenv('EXTERNAL_API_KEY'):
    EXTERNAL_API_KEYS.add(os.getenv('EXTERNAL_API_KEY').strip())


def verify_external_api_key(request: Request) -> bool:
    """Verify API key from request header."""
    api_key = request.headers.get('x-api-key')
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if EXTERNAL_API_KEYS and api_key not in EXTERNAL_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def parse_detachable_tag(tag_id: str) -> dict:
    """Parse tag ID to identify detachable items (_1, _2 suffix)."""
    import re
    match = re.match(r'^(.+)_(\d+)$', tag_id)
    if match:
        return {
            "base_tag": match.group(1),
            "suffix": f"_{match.group(2)}",
            "is_detachable": True,
            "part_number": int(match.group(2))
        }
    return {
        "base_tag": tag_id,
        "suffix": None,
        "is_detachable": False,
        "part_number": 0
    }


def generate_image_filename(tag_id: str, image_type: str, existing_count: int = 0) -> str:
    """Generate filename following the naming convention.

    Article: tagID.jpg, tagID(1).jpg, tagID(2).jpg
    HUID: tagID_HUID.jpg
    """
    if image_type == "huid":
        return f"{tag_id}_HUID.jpg"
    else:
        # Article photo
        if existing_count == 0:
            return f"{tag_id}.jpg"
        else:
            return f"{tag_id}({existing_count}).jpg"


class ExternalPhotoUploadRequest(BaseModel):
    tag_id: str
    job_no: str
    branch: str = "default"
    device_source: Optional[str] = None


class ExternalUploadUrlRequest(BaseModel):
    tag_id: str
    job_no: str
    branch: str = "default"
    image_type: str  # "article" or "huid"
    device_source: Optional[str] = None
    content_type: str = "image/jpeg"


class ExternalUploadUrlResponse(BaseModel):
    success: bool
    upload_url: str
    s3_key: str
    filename: str
    expires_in: int = 3600


class ExternalPhotoResponse(BaseModel):
    success: bool
    tag_id: str
    job_no: Optional[str] = None
    image_url: str
    is_detachable: bool
    detachable_suffix: Optional[str] = None
    uploaded_at: str


class ExternalHuidPhotoResponse(BaseModel):
    success: bool
    tag_id: str
    job_no: Optional[str] = None
    image_url: str
    uploaded_at: str
    ocr_result: Optional[dict] = None


@app.post("/api/external/get-upload-url", response_model=ExternalUploadUrlResponse)
async def external_get_upload_url(
    request: Request,
    body: ExternalUploadUrlRequest,
):
    """Get a presigned URL for direct S3 upload from external systems.

    This allows camera software to upload directly to S3 without routing through our server.
    After upload, call /api/external/confirm-upload to trigger OCR processing.
    """
    verify_external_api_key(request)

    if body.image_type not in ("article", "huid"):
        raise HTTPException(status_code=400, detail="image_type must be 'article' or 'huid'")

    # Generate filename
    filename = generate_image_filename(body.tag_id, body.image_type, 0)

    # Build S3 key
    folder = "article-photos" if body.image_type == "article" else "huid-photos"
    s3_key = f"{folder}/{body.branch}/{body.job_no}/{filename}"

    try:
        client = get_s3_client()
        bucket_name = os.getenv('S3_BUCKET_NAME')

        if not bucket_name:
            raise HTTPException(status_code=500, detail="S3 bucket not configured")

        presigned_url = client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': bucket_name,
                'Key': s3_key,
                'ContentType': body.content_type,
                'Metadata': {
                    'tag-id': body.tag_id,
                    'job-no': body.job_no,
                    'branch': body.branch,
                    'image-type': body.image_type,
                    'device-source': body.device_source or ''
                }
            },
            ExpiresIn=3600
        )

        return ExternalUploadUrlResponse(
            success=True,
            upload_url=presigned_url,
            s3_key=s3_key,
            filename=filename,
            expires_in=3600
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {str(e)}")


class ConfirmUploadRequest(BaseModel):
    tag_id: str
    job_no: str
    branch: str = "default"
    s3_key: str
    image_type: str
    device_source: Optional[str] = None


@app.post("/api/external/confirm-upload")
async def external_confirm_upload(
    request: Request,
    body: ConfirmUploadRequest,
):
    """Confirm that an image has been uploaded to S3 and trigger processing.

    For HUID images, this triggers OCR processing automatically.
    For Article images, this just records the upload.
    """
    verify_external_api_key(request)

    if body.image_type not in ("article", "huid"):
        raise HTTPException(status_code=400, detail="image_type must be 'article' or 'huid'")

    tag_info = parse_detachable_tag(body.tag_id)

    # Check if tag exists in batch_items, create if not
    batch_item = db.get_batch_item_by_tag(body.tag_id)

    if not batch_item:
        # Create a new batch for external uploads if needed
        conn = db._get_connection()
        try:
            cursor = conn.cursor()
            # Check if we have an external uploads batch for this job
            cursor.execute(
                "SELECT id FROM batches WHERE batch_name = ? LIMIT 1",
                (f"External_{body.job_no}",)
            )
            row = cursor.fetchone()
            if row:
                batch_id = row[0]
            else:
                # Create new batch
                cursor.execute(
                    "INSERT INTO batches (batch_name, status, metadata) VALUES (?, 'pending', ?)",
                    (f"External_{body.job_no}", json.dumps({"source": "external_api", "branch": body.branch}))
                )
                batch_id = cursor.lastrowid

            # Create batch item
            cursor.execute(
                """INSERT INTO batch_items (batch_id, tag_id, expected_huid, status, bis_job_no, branch)
                   VALUES (?, ?, '', 'pending', ?, ?)""",
                (batch_id, body.tag_id, body.job_no, body.branch)
            )
            conn.commit()
            batch_item = db.get_batch_item_by_tag(body.tag_id)
        finally:
            conn.close()

    # Get image URL
    image_url = storage.presign_get(body.s3_key)

    # Record the image
    slot = 0 if body.image_type == "huid" else 1
    img = ItemImage(
        batch_item_id=batch_item.id,
        tag_id=body.tag_id,
        image_type=body.image_type,
        slot=slot,
        s3_key=body.s3_key,
        s3_bucket=storage.config.s3_bucket if storage.use_s3 else "",
        content_type="image/jpeg",
        size_bytes=0,
    )
    db.upsert_item_image(img)

    # Update batch_item with bis_job_no if not set
    conn = db._get_connection()
    try:
        conn.execute(
            "UPDATE batch_items SET bis_job_no = ?, branch = ? WHERE tag_id = ?",
            (body.job_no, body.branch, body.tag_id)
        )
        conn.commit()
    finally:
        conn.close()

    # For HUID images, trigger OCR
    ocr_result = None
    if body.image_type == "huid":
        try:
            # Download image from S3 and run OCR
            image_bytes = storage.get_image(body.s3_key)
            if image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")

                hallmark_info = ocr_engine_v2.extract_with_hallmark_info(image)

                ocr_result = {
                    "huid_detected": hallmark_info.huid,
                    "purity_code": hallmark_info.purity_code,
                    "karat": hallmark_info.karat,
                    "purity_percentage": hallmark_info.purity_percentage,
                    "confidence": round(hallmark_info.overall_confidence, 3),
                    "raw_text": " ".join([r.text for r in hallmark_info.all_results]),
                    "decision": "approved" if hallmark_info.overall_confidence >= 0.85 else (
                        "manual_review" if hallmark_info.overall_confidence >= 0.50 else "rejected"
                    )
                }

                # Save OCR result
                db_result = DBOCRResult(
                    batch_item_id=batch_item.id,
                    tag_id=body.tag_id,
                    expected_huid=batch_item.expected_huid,
                    actual_huid=hallmark_info.huid,
                    huid_match=compare_huids(hallmark_info.huid, batch_item.expected_huid) if batch_item.expected_huid else None,
                    purity_code=hallmark_info.purity_code,
                    karat=hallmark_info.karat,
                    purity_percentage=hallmark_info.purity_percentage,
                    confidence=hallmark_info.overall_confidence,
                    decision=QCDecision(ocr_result["decision"]),
                    raw_ocr_text=ocr_result["raw_text"],
                )
                db.create_ocr_result(db_result)
                db.update_batch_item_status(body.tag_id, ProcessingStatus.COMPLETED)
        except Exception as e:
            logger.error(f"OCR processing failed for {body.tag_id}: {str(e)}")
            ocr_result = {"error": str(e)}

    response = {
        "success": True,
        "tag_id": body.tag_id,
        "job_no": body.job_no,
        "image_url": image_url,
        "image_type": body.image_type,
        "is_detachable": tag_info["is_detachable"],
        "detachable_suffix": tag_info["suffix"],
        "uploaded_at": datetime.now().isoformat()
    }

    if ocr_result:
        response["ocr_result"] = ocr_result

    return response


@app.post("/api/external/article-photo", response_model=ExternalPhotoResponse)
async def external_upload_article_photo(
    request: Request,
    file: UploadFile = File(...),
    tag_id: str = Form(...),
    job_no: str = Form(...),
    branch: str = Form("default"),
    device_source: Optional[str] = Form(None),
):
    """Upload an article photo from external camera system.

    This endpoint receives the image directly (multipart/form-data).
    The image is stored in S3 with naming convention: tagID.jpg
    """
    verify_external_api_key(request)

    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: PNG, JPG, JPEG, BMP, WEBP"
        )

    tag_info = parse_detachable_tag(tag_id)

    # Check/create batch item
    batch_item = db.get_batch_item_by_tag(tag_id)
    if not batch_item:
        # Create batch and item for external uploads
        conn = db._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM batches WHERE batch_name = ? LIMIT 1",
                (f"External_{job_no}",)
            )
            row = cursor.fetchone()
            if row:
                batch_id = row[0]
            else:
                cursor.execute(
                    "INSERT INTO batches (batch_name, status, metadata) VALUES (?, 'pending', ?)",
                    (f"External_{job_no}", json.dumps({"source": "external_api", "branch": branch}))
                )
                batch_id = cursor.lastrowid

            cursor.execute(
                """INSERT INTO batch_items (batch_id, tag_id, expected_huid, status, bis_job_no, branch)
                   VALUES (?, ?, '', 'pending', ?, ?)""",
                (batch_id, tag_id, job_no, branch)
            )
            conn.commit()
            batch_item = db.get_batch_item_by_tag(tag_id)
        finally:
            conn.close()

    # Read and store image
    contents = await file.read()

    # Generate filename with convention
    filename = generate_image_filename(tag_id, "article", 0)

    # Upload to S3
    s3_key = f"article-photos/{branch}/{job_no}/{filename}"

    if storage.use_s3:
        try:
            client = get_s3_client()
            bucket_name = os.getenv('S3_BUCKET_NAME')
            client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=contents,
                ContentType=file.content_type,
                Metadata={
                    'tag-id': tag_id,
                    'job-no': job_no,
                    'branch': branch,
                    'device-source': device_source or ''
                }
            )
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
    else:
        # Local storage fallback
        local_path = os.path.join(uploads_dir, s3_key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(contents)

    # Record in item_images
    img = ItemImage(
        batch_item_id=batch_item.id,
        tag_id=tag_id,
        image_type="article",
        slot=1,
        s3_key=s3_key,
        s3_bucket=storage.config.s3_bucket if storage.use_s3 else "",
        content_type=file.content_type,
        size_bytes=len(contents),
    )
    db.upsert_item_image(img)

    # Add device_source to image record
    if device_source:
        conn = db._get_connection()
        try:
            conn.execute(
                "UPDATE item_images SET device_source = ? WHERE tag_id = ? AND image_type = 'article'",
                (device_source, tag_id)
            )
            conn.commit()
        finally:
            conn.close()

    image_url = storage.presign_get(s3_key)

    return ExternalPhotoResponse(
        success=True,
        tag_id=tag_id,
        job_no=job_no,
        image_url=image_url,
        is_detachable=tag_info["is_detachable"],
        detachable_suffix=tag_info["suffix"],
        uploaded_at=datetime.now().isoformat()
    )


@app.post("/api/external/huid-photo", response_model=ExternalHuidPhotoResponse)
async def external_upload_huid_photo(
    request: Request,
    file: UploadFile = File(...),
    tag_id: str = Form(...),
    job_no: str = Form(...),
    branch: str = Form("default"),
    device_source: Optional[str] = Form(None),
):
    """Upload a HUID photo from microscopic camera and trigger OCR.

    This endpoint receives the image directly and automatically runs OCR.
    The image is stored in S3 with naming convention: tagID_HUID.jpg
    """
    verify_external_api_key(request)

    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: PNG, JPG, JPEG, BMP, WEBP"
        )

    # Check/create batch item
    batch_item = db.get_batch_item_by_tag(tag_id)
    if not batch_item:
        conn = db._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM batches WHERE batch_name = ? LIMIT 1",
                (f"External_{job_no}",)
            )
            row = cursor.fetchone()
            if row:
                batch_id = row[0]
            else:
                cursor.execute(
                    "INSERT INTO batches (batch_name, status, metadata) VALUES (?, 'pending', ?)",
                    (f"External_{job_no}", json.dumps({"source": "external_api", "branch": branch}))
                )
                batch_id = cursor.lastrowid

            cursor.execute(
                """INSERT INTO batch_items (batch_id, tag_id, expected_huid, status, bis_job_no, branch)
                   VALUES (?, ?, '', 'pending', ?, ?)""",
                (batch_id, tag_id, job_no, branch)
            )
            conn.commit()
            batch_item = db.get_batch_item_by_tag(tag_id)
        finally:
            conn.close()

    contents = await file.read()

    # Generate filename
    filename = generate_image_filename(tag_id, "huid", 0)
    s3_key = f"huid-photos/{branch}/{job_no}/{filename}"

    # Upload to S3
    if storage.use_s3:
        try:
            client = get_s3_client()
            bucket_name = os.getenv('S3_BUCKET_NAME')
            client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=contents,
                ContentType=file.content_type,
                Metadata={
                    'tag-id': tag_id,
                    'job-no': job_no,
                    'branch': branch,
                    'device-source': device_source or ''
                }
            )
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
    else:
        local_path = os.path.join(uploads_dir, s3_key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(contents)

    # Record in item_images
    img = ItemImage(
        batch_item_id=batch_item.id,
        tag_id=tag_id,
        image_type="huid",
        slot=0,
        s3_key=s3_key,
        s3_bucket=storage.config.s3_bucket if storage.use_s3 else "",
        content_type=file.content_type,
        size_bytes=len(contents),
    )
    db.upsert_item_image(img)

    # Add device_source
    if device_source:
        conn = db._get_connection()
        try:
            conn.execute(
                "UPDATE item_images SET device_source = ? WHERE tag_id = ? AND image_type = 'huid'",
                (device_source, tag_id)
            )
            conn.commit()
        finally:
            conn.close()

    # Run OCR
    ocr_result = None
    try:
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")

        start_time = time.time()
        hallmark_info = ocr_engine_v2.extract_with_hallmark_info(image)
        processing_time = int((time.time() - start_time) * 1000)

        huid_match = None
        if batch_item.expected_huid and hallmark_info.huid:
            huid_match = compare_huids(hallmark_info.huid, batch_item.expected_huid)

        decision = QCDecision.PENDING
        if hallmark_info.overall_confidence >= 0.85 and (huid_match is None or huid_match):
            decision = QCDecision.APPROVED
        elif hallmark_info.overall_confidence >= 0.50:
            decision = QCDecision.MANUAL_REVIEW
        else:
            decision = QCDecision.REJECTED

        ocr_result = {
            "huid_detected": hallmark_info.huid,
            "purity_code": hallmark_info.purity_code,
            "karat": hallmark_info.karat,
            "purity_percentage": hallmark_info.purity_percentage,
            "confidence": round(hallmark_info.overall_confidence, 3),
            "raw_text": " ".join([r.text for r in hallmark_info.all_results]),
            "decision": decision.value
        }

        # Save OCR result
        db_result = DBOCRResult(
            batch_item_id=batch_item.id,
            tag_id=tag_id,
            expected_huid=batch_item.expected_huid,
            actual_huid=hallmark_info.huid,
            huid_match=huid_match,
            purity_code=hallmark_info.purity_code,
            karat=hallmark_info.karat,
            purity_percentage=hallmark_info.purity_percentage,
            confidence=hallmark_info.overall_confidence,
            decision=decision,
            raw_ocr_text=ocr_result["raw_text"],
            processing_time_ms=processing_time,
        )
        db.create_ocr_result(db_result)
        db.update_batch_item_status(tag_id, ProcessingStatus.COMPLETED)

    except Exception as e:
        logger.error(f"OCR failed for {tag_id}: {str(e)}")
        ocr_result = {"error": str(e)}
        db.update_batch_item_status(tag_id, ProcessingStatus.FAILED)

    image_url = storage.presign_get(s3_key)

    return ExternalHuidPhotoResponse(
        success=True,
        tag_id=tag_id,
        job_no=job_no,
        image_url=image_url,
        uploaded_at=datetime.now().isoformat(),
        ocr_result=ocr_result
    )


@app.get("/api/external/photo-exists/{tag_id}")
async def external_check_photo_exists(
    request: Request,
    tag_id: str,
    image_type: Optional[str] = Query(None, description="'article', 'huid', or None for both"),
):
    """Check if photos exist for a tag ID."""
    verify_external_api_key(request)

    result = {
        "tag_id": tag_id,
        "article_photo": None,
        "huid_photo": None
    }

    if image_type in (None, "article"):
        article_img = db.get_item_image(tag_id, "article", 1)
        if article_img:
            result["article_photo"] = {
                "exists": True,
                "image_url": storage.presign_get(article_img.s3_key),
                "uploaded_at": article_img.uploaded_at.isoformat() if article_img.uploaded_at else None
            }
        else:
            result["article_photo"] = {"exists": False}

    if image_type in (None, "huid"):
        huid_img = db.get_item_image(tag_id, "huid", 0)
        if huid_img:
            ocr_result = db.get_ocr_result_by_tag(tag_id)
            result["huid_photo"] = {
                "exists": True,
                "image_url": storage.presign_get(huid_img.s3_key),
                "uploaded_at": huid_img.uploaded_at.isoformat() if huid_img.uploaded_at else None,
                "ocr_status": ocr_result.decision.value if ocr_result else None
            }
        else:
            result["huid_photo"] = {"exists": False}

    return result


@app.get("/api/external/ocr-result/{tag_id}")
async def external_get_ocr_result(request: Request, tag_id: str):
    """Get OCR result for a tag ID."""
    verify_external_api_key(request)

    ocr_result = db.get_ocr_result_by_tag(tag_id)
    if not ocr_result:
        raise HTTPException(status_code=404, detail=f"No OCR result for tag '{tag_id}'")

    return {
        "tag_id": tag_id,
        "expected_huid": ocr_result.expected_huid,
        "actual_huid": ocr_result.actual_huid,
        "huid_match": ocr_result.huid_match,
        "purity_code": ocr_result.purity_code,
        "karat": ocr_result.karat,
        "purity_percentage": ocr_result.purity_percentage,
        "confidence": ocr_result.confidence,
        "decision": ocr_result.decision.value if isinstance(ocr_result.decision, QCDecision) else ocr_result.decision,
        "processed_at": ocr_result.created_at.isoformat() if ocr_result.created_at else None
    }


# =============================================================================
# Manakonline Upload Queue APIs
# =============================================================================

@app.get("/api/manak/upload-queue")
async def get_manak_upload_queue(
    limit: int = Query(50, le=200),
    status: Optional[str] = Query(None),
):
    """Get items queued for Manakonline portal upload."""
    conn = db._get_connection()
    try:
        params = []
        where_clauses = []

        if status:
            where_clauses.append("status = ?")
            params.append(status)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        rows = conn.execute(
            f"""
            SELECT id, tag_id, bis_job_no, branch, image_type, s3_key,
                   status, retry_count, error_message, queued_at
            FROM upload_queue
            {where_sql}
            ORDER BY queued_at ASC
            LIMIT ?
            """,
            params + [limit]
        ).fetchall()

        items = []
        for row in rows:
            items.append({
                "id": row["id"],
                "tag_id": row["tag_id"],
                "bis_job_no": row["bis_job_no"],
                "branch": row["branch"],
                "image_type": row["image_type"],
                "image_url": storage.presign_get(row["s3_key"]),
                "status": row["status"],
                "retry_count": row["retry_count"],
                "error_message": row["error_message"],
                "queued_at": row["queued_at"]
            })

        # Get stats
        stats = conn.execute(
            """
            SELECT status, COUNT(*) as count
            FROM upload_queue
            GROUP BY status
            """
        ).fetchall()

        return {
            "items": items,
            "total_pending": sum(r["count"] for r in stats if r["status"] == "pending"),
            "stats": {r["status"]: r["count"] for r in stats}
        }
    finally:
        conn.close()


@app.post("/api/manak/queue-for-upload")
async def queue_for_manak_upload(
    tag_ids: List[str] = Form(...),
):
    """Queue tags for Manakonline portal upload."""
    conn = db._get_connection()
    try:
        queued = 0
        errors = []

        for tag_id in tag_ids:
            # Get batch item info
            batch_item = db.get_batch_item_by_tag(tag_id)
            if not batch_item:
                errors.append({"tag_id": tag_id, "error": "Tag not found"})
                continue

            # Get images
            images = db.get_item_images_for_tag(tag_id)

            for img in images:
                try:
                    # Get bis_job_no from batch_item
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT bis_job_no, branch FROM batch_items WHERE tag_id = ?",
                        (tag_id,)
                    )
                    row = cursor.fetchone()
                    bis_job_no = row["bis_job_no"] if row else ""
                    branch = row["branch"] if row else "default"

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO upload_queue
                        (tag_id, bis_job_no, branch, image_type, s3_key, status)
                        VALUES (?, ?, ?, ?, ?, 'pending')
                        """,
                        (tag_id, bis_job_no, branch, img.image_type, img.s3_key)
                    )
                    queued += 1
                except Exception as e:
                    errors.append({"tag_id": tag_id, "error": str(e)})

        conn.commit()

        return {
            "success": True,
            "queued_count": queued,
            "errors": errors
        }
    finally:
        conn.close()


@app.post("/api/manak/upload-result")
async def report_manak_upload_result(
    tag_id: str = Form(...),
    image_type: str = Form(...),
    status: str = Form(...),  # 'success' or 'failed'
    error_message: Optional[str] = Form(None),
    portal_reference: Optional[str] = Form(None),
):
    """Report result of a Manakonline portal upload (called by browser extension)."""
    conn = db._get_connection()
    try:
        if status == "success":
            conn.execute(
                """
                UPDATE upload_queue
                SET status = 'uploaded', portal_reference = ?, uploaded_at = CURRENT_TIMESTAMP
                WHERE tag_id = ? AND image_type = ?
                """,
                (portal_reference, tag_id, image_type)
            )
            # Also update batch_items upload_status
            conn.execute(
                "UPDATE batch_items SET upload_status = 'uploaded' WHERE tag_id = ?",
                (tag_id,)
            )
        else:
            conn.execute(
                """
                UPDATE upload_queue
                SET status = 'failed', error_message = ?, retry_count = retry_count + 1
                WHERE tag_id = ? AND image_type = ?
                """,
                (error_message, tag_id, image_type)
            )

        conn.commit()

        return {"success": True, "tag_id": tag_id, "status": status}
    finally:
        conn.close()


@app.get("/api/manak/upload-stats")
async def get_manak_upload_stats(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
):
    """Get upload statistics for the dashboard."""
    conn = db._get_connection()
    try:
        date_filter = ""
        params = []

        if date:
            date_filter = "WHERE DATE(queued_at) = ?"
            params.append(date)

        stats = conn.execute(
            f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'uploaded' THEN 1 ELSE 0 END) as uploaded,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'uploading' THEN 1 ELSE 0 END) as in_progress
            FROM upload_queue
            {date_filter}
            """,
            params
        ).fetchone()

        return {
            "date": date or "all",
            "total_queued": stats["total"] or 0,
            "uploaded": stats["uploaded"] or 0,
            "failed": stats["failed"] or 0,
            "pending": stats["pending"] or 0,
            "in_progress": stats["in_progress"] or 0
        }
    finally:
        conn.close()


# =============================================================================
# MANAKONLINE COMPARISON ENDPOINTS
# =============================================================================

@app.post("/api/manak/upload-comparison")
async def upload_manak_comparison_file(
    file: UploadFile = File(...),
    bis_job_no: Optional[str] = Form(None, description="BIS Job No (auto-extracted from filename if not provided)"),
):
    """
    Upload a Manakonline Excel file for comparison with OCR results.

    The file should contain:
    - AHC Tag (or Tag ID)
    - HUID
    - Declared Purity (optional)

    BIS Job No is extracted from filename if not provided.
    Supports .xlsx, .xls, .csv, and .numbers (Apple Numbers) formats.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read file content
    content = await file.read()

    # Parse the file
    result = parse_manak_file(content, file.filename)

    # Override BIS Job No if provided
    if bis_job_no:
        result = ManakParseResult(
            bis_job_no=bis_job_no,
            rows=result.rows,
            total_rows=result.total_rows,
            errors=result.errors
        )

    if not result.rows:
        return {
            "success": False,
            "bis_job_no": result.bis_job_no,
            "errors": result.errors or [{"error": "No valid rows found in file"}],
            "total_rows": 0
        }

    conn = db._get_connection()
    try:
        cur = conn.cursor()

        # Get OCR results for all tags in this job
        tag_ids = [row.ahc_tag for row in result.rows]
        placeholders = ','.join(['?' for _ in tag_ids])

        ocr_results = {}
        if tag_ids:
            rows = cur.execute(
                f"""
                SELECT bi.tag_id, ocr.actual_huid, ocr.purity_code,
                       ocr.huid_match, bi.upload_status
                FROM batch_items bi
                LEFT JOIN ocr_results ocr ON bi.id = ocr.batch_item_id
                WHERE bi.tag_id IN ({placeholders})
                """,
                tag_ids
            ).fetchall()

            for row in rows:
                ocr_results[row["tag_id"]] = dict(row)

        # Run comparison
        comparisons = compare_with_ocr(result.rows, ocr_results)

        # Store comparison results
        comparison_id = f"cmp_{result.bis_job_no}_{int(__import__('time').time())}"

        for comp in comparisons:
            cur.execute(
                """
                INSERT INTO manak_comparisons
                (comparison_id, bis_job_no, tag_id, manak_huid, manak_purity,
                 ocr_huid, ocr_purity, huid_match, purity_match, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    comparison_id,
                    result.bis_job_no,
                    comp['tag_id'],
                    comp['manak_huid'],
                    comp['manak_purity'],
                    comp['ocr_huid'],
                    comp['ocr_purity'],
                    1 if comp['huid_match'] else 0,
                    1 if comp['purity_match'] else 0,
                    comp['status']
                )
            )

            # Update batch_items with manak_huid for reference
            if comp['manak_huid']:
                cur.execute(
                    """
                    UPDATE batch_items SET manak_huid = ?, bis_job_no = ?
                    WHERE tag_id = ?
                    """,
                    (comp['manak_huid'], result.bis_job_no, comp['tag_id'])
                )

        conn.commit()

        # Calculate summary stats
        stats = {
            'total': len(comparisons),
            'matched': sum(1 for c in comparisons if c['status'] == 'match'),
            'partial_match': sum(1 for c in comparisons if c['status'] == 'partial_match'),
            'mismatch': sum(1 for c in comparisons if c['status'] == 'mismatch'),
            'missing_ocr': sum(1 for c in comparisons if c['status'] == 'missing_ocr')
        }

        return {
            "success": True,
            "comparison_id": comparison_id,
            "bis_job_no": result.bis_job_no,
            "total_rows": result.total_rows,
            "stats": stats,
            "comparisons": comparisons,
            "parse_errors": result.errors
        }

    finally:
        conn.close()


@app.get("/api/manak/comparison-results/{bis_job_no}")
async def get_comparison_results(
    bis_job_no: str,
    status: Optional[str] = Query(None, description="Filter by status: match, partial_match, mismatch, missing_ocr"),
):
    """Get comparison results for a BIS Job No."""
    conn = db._get_connection()
    try:
        query = """
            SELECT * FROM manak_comparisons
            WHERE bis_job_no = ?
        """
        params = [bis_job_no]

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"

        rows = conn.execute(query, params).fetchall()

        comparisons = [dict(row) for row in rows]

        # Calculate summary
        stats = {
            'total': len(comparisons),
            'matched': sum(1 for c in comparisons if c['status'] == 'match'),
            'partial_match': sum(1 for c in comparisons if c['status'] == 'partial_match'),
            'mismatch': sum(1 for c in comparisons if c['status'] == 'mismatch'),
            'missing_ocr': sum(1 for c in comparisons if c['status'] == 'missing_ocr')
        }

        return {
            "bis_job_no": bis_job_no,
            "stats": stats,
            "comparisons": comparisons
        }
    finally:
        conn.close()


@app.get("/api/manak/comparison-list")
async def list_comparisons(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all comparison batches."""
    conn = db._get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                bis_job_no,
                comparison_id,
                COUNT(*) as total_items,
                SUM(CASE WHEN status = 'match' THEN 1 ELSE 0 END) as matched,
                SUM(CASE WHEN status = 'mismatch' THEN 1 ELSE 0 END) as mismatched,
                SUM(CASE WHEN status = 'missing_ocr' THEN 1 ELSE 0 END) as missing_ocr,
                MIN(created_at) as created_at
            FROM manak_comparisons
            GROUP BY bis_job_no, comparison_id
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        ).fetchall()

        return {
            "comparisons": [dict(row) for row in rows],
            "limit": limit,
            "offset": offset
        }
    finally:
        conn.close()


@app.post("/api/manak/recompare/{tag_id}")
async def recompare_single_tag(tag_id: str):
    """
    Re-run comparison for a single tag after OCR has been updated.
    """
    conn = db._get_connection()
    try:
        cur = conn.cursor()

        # Get OCR result for this tag
        ocr_row = cur.execute(
            """
            SELECT bi.tag_id, ocr.actual_huid, ocr.purity_code,
                   bi.manak_huid, bi.bis_job_no
            FROM batch_items bi
            LEFT JOIN ocr_results ocr ON bi.id = ocr.batch_item_id
            WHERE bi.tag_id = ?
            """,
            (tag_id,)
        ).fetchone()

        if not ocr_row:
            raise HTTPException(status_code=404, detail=f"Tag {tag_id} not found")

        manak_huid = ocr_row["manak_huid"]
        if not manak_huid:
            raise HTTPException(status_code=400, detail=f"No Manakonline HUID found for tag {tag_id}")

        # Get the latest comparison entry
        comp_row = cur.execute(
            """
            SELECT * FROM manak_comparisons
            WHERE tag_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (tag_id,)
        ).fetchone()

        # Calculate new comparison
        huid_match = False
        purity_match = False

        if manak_huid and ocr_row["actual_huid"]:
            huid_match = (
                manak_huid.upper() == ocr_row["actual_huid"].upper() or
                manak_huid.upper() in ocr_row["actual_huid"].upper() or
                ocr_row["actual_huid"].upper() in manak_huid.upper()
            )

        manak_purity = comp_row["manak_purity"] if comp_row else None
        if manak_purity and ocr_row["purity_code"]:
            extracted_purity, _ = extract_purity(manak_purity)
            purity_match = extracted_purity == ocr_row["purity_code"] if extracted_purity else False

        # Determine status
        if huid_match and purity_match:
            status = 'match'
        elif huid_match or purity_match:
            status = 'partial_match'
        else:
            status = 'mismatch'

        # Update comparison record
        if comp_row:
            cur.execute(
                """
                UPDATE manak_comparisons
                SET ocr_huid = ?, ocr_purity = ?, huid_match = ?, purity_match = ?, status = ?,
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (
                    ocr_row["actual_huid"],
                    ocr_row["purity_code"],
                    1 if huid_match else 0,
                    1 if purity_match else 0,
                    status,
                    comp_row["id"]
                )
            )

        conn.commit()

        return {
            "success": True,
            "tag_id": tag_id,
            "manak_huid": manak_huid,
            "ocr_huid": ocr_row["actual_huid"],
            "huid_match": huid_match,
            "purity_match": purity_match,
            "status": status
        }

    finally:
        conn.close()


@app.post("/api/manak/mark-for-rework")
async def mark_items_for_rework(
    tag_ids: List[str] = Body(..., description="List of tag IDs to mark for rework"),
    reason: Optional[str] = Body(None, description="Reason for rework"),
):
    """Mark items for rework (re-capture/re-OCR)."""
    conn = db._get_connection()
    try:
        cur = conn.cursor()

        updated = []
        for tag_id in tag_ids:
            cur.execute(
                """
                UPDATE batch_items
                SET rework_status = 'pending', upload_status = 'pending'
                WHERE tag_id = ?
                """,
                (tag_id,)
            )
            if cur.rowcount > 0:
                updated.append(tag_id)

        conn.commit()

        return {
            "success": True,
            "marked_count": len(updated),
            "tag_ids": updated,
            "reason": reason
        }
    finally:
        conn.close()


@app.get("/api/manak/rework-queue")
async def get_rework_queue(
    bis_job_no: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """Get items pending rework."""
    conn = db._get_connection()
    try:
        query = """
            SELECT bi.*, mc.manak_huid, mc.manak_purity, mc.status as comparison_status
            FROM batch_items bi
            LEFT JOIN manak_comparisons mc ON bi.tag_id = mc.tag_id
            WHERE bi.rework_status = 'pending'
        """
        params = []

        if bis_job_no:
            query += " AND bi.bis_job_no = ?"
            params.append(bis_job_no)

        query += " ORDER BY bi.created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()

        return {
            "rework_items": [dict(row) for row in rows],
            "total": len(rows)
        }
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
