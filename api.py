"""
FastAPI OCR API - Hallmark QC Validation System.

Three-Stage Workflow:
1. Stage 1: Upload CSV/Excel with tag IDs and expected HUIDs
2. Stage 2: Upload images with tag ID for processing
3. Stage 3: Retrieve results by tag ID

Deploy on AWS or run locally with: uvicorn api:app --host 0.0.0.0 --port 8000
"""

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
import sys
import os
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

from ocr_model import OCREngine, OCRResult
from ocr_model_v2 import OCREngineV2, OCRResultV2, HallmarkInfo, HallmarkType, CheckInfo
from database import (
    DatabaseManager, get_database, Batch, BatchItem, OCRResult as DBOCRResult,
    ProcessingStatus, QCDecision
)
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

# Initialize OCR engines at startup
ocr_engine = None
ocr_engine_v2 = None
db: DatabaseManager = None
storage: StorageService = None

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
class BatchUploadResponse(BaseModel):
    status: str
    batch_id: int
    batch_name: str
    total_items: int
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
    global ocr_engine, ocr_engine_v2, db, storage
    ocr_engine = OCREngine()
    ocr_engine_v2 = OCREngineV2(enable_preprocessing=True)
    db = get_database()
    storage = get_storage()
    print(f"Storage type: {storage.storage_type}")


# =============================================================================
# Homepage & Health Endpoints
# =============================================================================

@app.get("/", response_class=RedirectResponse)
async def homepage():
    """Redirect to API documentation."""
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
    batch_name: Optional[str] = Form(None)
):
    """
    Stage 1: Upload CSV/Excel file with tag IDs and expected HUIDs.

    The file should have at least two columns:
    - tag_id: Unique identifier for each item
    - expected_huid: The expected HUID to validate against

    Supported formats: CSV, XLSX, XLS
    """
    # Validate file type
    allowed_types = [
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ]
    filename_lower = file.filename.lower()

    if file.content_type not in allowed_types and not (
        filename_lower.endswith(".csv") or
        filename_lower.endswith(".xlsx") or
        filename_lower.endswith(".xls")
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: CSV, XLSX, XLS"
        )

    try:
        contents = await file.read()

        # Parse file based on type
        if filename_lower.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        # Validate required columns
        required_columns = ["tag_id", "expected_huid"]
        # Also check for common variations
        column_mappings = {
            "tag_id": ["tag_id", "tagid", "tag", "id", "item_id"],
            "expected_huid": ["expected_huid", "huid", "expected_id", "expectedhuid"]
        }

        for req_col, variations in column_mappings.items():
            found = False
            for var in variations:
                if var in df.columns:
                    if var != req_col:
                        df = df.rename(columns={var: req_col})
                    found = True
                    break
            if not found:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required column: {req_col}. Found columns: {list(df.columns)}"
                )

        # Remove duplicates and empty rows
        df = df.dropna(subset=["tag_id", "expected_huid"])
        df = df.drop_duplicates(subset=["tag_id"])

        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid rows found in the file"
            )

        # Create batch
        batch_name = batch_name or f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch = Batch(
            batch_name=batch_name,
            total_items=len(df),
            status="pending"
        )
        batch_id = db.create_batch(batch)

        # Create batch items
        for _, row in df.iterrows():
            item = BatchItem(
                batch_id=batch_id,
                tag_id=str(row["tag_id"]).strip(),
                expected_huid=str(row["expected_huid"]).strip().upper(),
                status=ProcessingStatus.PENDING
            )
            db.create_batch_item(item)

        return BatchUploadResponse(
            status="success",
            batch_id=batch_id,
            batch_name=batch_name,
            total_items=len(df),
            message=f"Successfully uploaded {len(df)} items"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


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

@app.post("/stage2/upload-image", response_model=ImageUploadResponse)
async def upload_and_process_image(
    file: UploadFile = File(...),
    tag_id: str = Form(...)
):
    """
    Stage 2: Upload an image for a specific tag ID and process it.

    The image will be:
    1. Stored in S3 (or local storage)
    2. Processed through OCR
    3. Compared against the expected HUID from Stage 1
    """
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: PNG, JPG, JPEG, BMP, WEBP"
        )

    # Check if tag_id exists in database
    batch_item = db.get_batch_item_by_tag(tag_id)
    if not batch_item:
        raise HTTPException(
            status_code=404,
            detail=f"Tag ID '{tag_id}' not found. Please upload batch data first (Stage 1)."
        )

    try:
        start_time = time.time()

        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Upload original image to storage
        image_path, image_url = storage.upload_image(
            contents,
            tag_id,
            file.filename,
            prefix="originals"
        )

        # Update batch item with image path
        db.update_batch_item_image(tag_id, image_path, image_url)

        # Process with OCR
        hallmark_info = ocr_engine_v2.extract_with_hallmark_info(image)

        # Extract actual HUID
        actual_huid = hallmark_info.huid
        expected_huid = batch_item.expected_huid

        # Compare HUIDs
        huid_match = False
        if actual_huid and expected_huid:
            huid_match = actual_huid.upper().strip() == expected_huid.upper().strip()

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

            # Upload image
            image_path, image_url = storage.upload_image(
                contents,
                tag_id,
                file.filename,
                prefix="originals"
            )

            db.update_batch_item_image(tag_id, image_path, image_url)

            # Process OCR
            hallmark_info = ocr_engine_v2.extract_with_hallmark_info(image)
            actual_huid = hallmark_info.huid
            huid_match = (
                actual_huid and
                actual_huid.upper().strip() == batch_item.expected_huid.upper().strip()
            )

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


# =============================================================================
# STAGE 3: Results Retrieval
# =============================================================================

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
    # This would be enhanced with proper search functionality
    if tag_id:
        result = db.get_full_result_by_tag(tag_id)
        if result:
            return {"status": "success", "results": [result]}
        return {"status": "success", "results": []}

    if batch_id:
        results = db.get_results_by_batch(batch_id)
        if decision:
            results = [r for r in results if r.get("decision") == decision]
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
# Legacy Endpoints (kept for backward compatibility)
# =============================================================================

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """Legacy: Extract text from an uploaded image."""
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        ocr_results = ocr_engine.extract_text_with_confidence(image)

        results = [
            {"text": r.text, "confidence": r.confidence, "approved": r.confidence >= 0.75}
            for r in ocr_results
        ]

        return {
            "success": True,
            "results": results,
            "full_text": "\n".join([r.text for r in ocr_results]),
            "average_confidence": sum(r.confidence for r in ocr_results) / len(ocr_results) if ocr_results else 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
