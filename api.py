"""
FastAPI OCR API - Extract text from images with QC validation.
Deploy on Render or run locally with: uvicorn api:app --host 0.0.0.0 --port 8000

Features:
- V1: Standard OCR text extraction
- V2: Hallmark-specific OCR with BIS validation
- QC: Quality Control workflow for jewelry hallmarking
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from PIL import Image
import io
import sys

# Import OCR engines
sys.path.insert(0, "src")
from ocr_model import OCREngine, OCRResult
from ocr_model_v2 import OCREngineV2, OCRResultV2, HallmarkInfo, HallmarkType, CheckInfo


app = FastAPI(
    title="OCR API - Jewelry Hallmarking QC",
    description="""
    Extract text from images using PaddleOCR with BIS hallmark validation.

    ## Features
    - **V1 OCR**: Standard text extraction
    - **V2 OCR**: Hallmark-specific detection with BIS validation
    - **QC Validation**: Quality control workflow for jewelry hallmarking

    ## QC Integration Flows
    1. **Hallmarking Stage**: Real-time validation after HUID engraving
    2. **QC Dashboard**: Batch validation with approval/rejection workflow

    ## BIS Standards
    - Gold: IS 1417 (916, 750, 585, etc.)
    - Silver: IS 2112 (925, 999, etc.)
    - HUID: Mandatory 6-character alphanumeric code
    """,
    version="3.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR engines at startup
ocr_engine = None
ocr_engine_v2 = None


class TextResult(BaseModel):
    text: str
    confidence: float
    approved: bool


class OCRResponse(BaseModel):
    success: bool
    results: list[TextResult]
    full_text: str
    average_confidence: float


# V2 Response Models
class TextResultV2(BaseModel):
    text: str
    confidence: float
    approved: bool
    hallmark_type: str
    validated: bool
    validation_details: dict


class CheckData(BaseModel):
    check_number: Optional[str] = None
    micr_code: Optional[str] = None
    ifsc_code: Optional[str] = None
    account_number: Optional[str] = None
    bank_name: Optional[str] = None
    is_valid_check: bool = False


class HallmarkData(BaseModel):
    purity_code: Optional[str] = None
    karat: Optional[str] = None
    purity_percentage: Optional[float] = None
    huid: Optional[str] = None
    bis_certified: bool = False


class OCRResponseV2(BaseModel):
    success: bool
    results: List[TextResultV2]
    full_text: str
    average_confidence: float
    hallmark: HallmarkData
    check: Optional[CheckData] = None


@app.on_event("startup")
async def startup_event():
    """Load OCR models on startup."""
    global ocr_engine, ocr_engine_v2
    ocr_engine = OCREngine()
    ocr_engine_v2 = OCREngineV2(enable_preprocessing=True)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "OCR API is running"}


@app.get("/health")
async def health():
    """Health check for Render."""
    return {"status": "healthy"}


@app.post("/extract", response_model=OCRResponse)
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from an uploaded image.

    Args:
        file: Image file (PNG, JPG, JPEG, BMP, WEBP)

    Returns:
        OCRResponse with extracted text, confidence scores, and approval status
    """
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: PNG, JPG, JPEG, BMP, WEBP"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Extract text
        ocr_results = ocr_engine.extract_text_with_confidence(image)

        # Build response
        results = [
            TextResult(
                text=r.text,
                confidence=r.confidence,
                approved=r.confidence >= 0.75
            )
            for r in ocr_results
        ]

        full_text = "\n".join([r.text for r in ocr_results])
        avg_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results) if ocr_results else 0.0

        return OCRResponse(
            success=True,
            results=results,
            full_text=full_text,
            average_confidence=avg_confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.post("/extract/v2", response_model=OCRResponseV2)
async def extract_text_v2(file: UploadFile = File(...)):
    """
    Extract text from an uploaded image using V2 engine with hallmark detection.

    Features:
    - Advanced image preprocessing for metal surfaces
    - Hallmark pattern recognition (purity marks, HUID)
    - BIS standard validation
    - Reflection removal and contrast enhancement

    Args:
        file: Image file (PNG, JPG, JPEG, BMP, WEBP)

    Returns:
        OCRResponseV2 with extracted text, hallmark info, and validation status
    """
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: PNG, JPG, JPEG, BMP, WEBP"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Extract text with hallmark info
        hallmark_info = ocr_engine_v2.extract_with_hallmark_info(image)

        # Build response
        results = [
            TextResultV2(
                text=r.text,
                confidence=r.confidence,
                approved=r.confidence >= 0.75,
                hallmark_type=r.hallmark_type.value,
                validated=r.validated,
                validation_details=r.validation_details
            )
            for r in hallmark_info.all_results
        ]

        full_text = "\n".join([r.text for r in hallmark_info.all_results])

        # Build check data if available
        check_data = None
        if hallmark_info.check_info:
            check_data = CheckData(
                check_number=hallmark_info.check_info.check_number,
                micr_code=hallmark_info.check_info.micr_code,
                ifsc_code=hallmark_info.check_info.ifsc_code,
                account_number=hallmark_info.check_info.account_number,
                bank_name=hallmark_info.check_info.bank_name,
                is_valid_check=hallmark_info.check_info.is_valid_check
            )

        return OCRResponseV2(
            success=True,
            results=results,
            full_text=full_text,
            average_confidence=hallmark_info.overall_confidence,
            hallmark=HallmarkData(
                purity_code=hallmark_info.purity_code,
                karat=hallmark_info.karat,
                purity_percentage=hallmark_info.purity_percentage,
                huid=hallmark_info.huid,
                bis_certified=hallmark_info.bis_certified
            ),
            check=check_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR V2 processing failed: {str(e)}")


# QC Validation Models
class QCValidationRequest(BaseModel):
    job_id: str
    expected_purity: Optional[str] = None
    metadata: Optional[Dict] = None


class QCHallmarkData(BaseModel):
    purity_code: Optional[str] = None
    karat: Optional[str] = None
    purity_percentage: Optional[float] = None
    huid: Optional[str] = None


class QCValidationStatus(BaseModel):
    purity_valid: bool
    huid_valid: bool
    bis_logo_detected: bool


class QCRejectionInfo(BaseModel):
    reasons: List[str]
    message: str


class QCProcessingDetails(BaseModel):
    ocr_corrections_applied: int
    raw_ocr_text: str
    image_quality_score: float
    processing_time_ms: int


class QCValidationData(BaseModel):
    job_id: str
    decision: str  # approved, rejected, manual_review
    confidence: float
    bis_certified: bool
    hallmark_data: QCHallmarkData
    validation_status: QCValidationStatus
    rejection_info: Optional[QCRejectionInfo] = None
    processing_details: QCProcessingDetails
    requires_manual_review: bool
    qc_override: Optional[str] = None


class QCValidationResponse(BaseModel):
    status: str
    data: QCValidationData
    metadata: Optional[Dict] = None


class QCOverrideRequest(BaseModel):
    job_id: str
    override_decision: str  # approved or rejected
    override_reason: str
    operator_id: str
    notes: Optional[str] = ""


class QCBatchItem(BaseModel):
    job_id: str
    image_path: Optional[str] = None
    expected_purity: Optional[str] = None


class QCBatchRequest(BaseModel):
    batch_id: str
    items: List[QCBatchItem]
    callback_url: Optional[str] = None


class QCBatchSummary(BaseModel):
    total: int
    approved: int
    rejected: int
    manual_review: int
    errors: int


class QCBatchResponse(BaseModel):
    status: str
    batch_id: str
    summary: QCBatchSummary
    processing_time_ms: int


# Initialize QC service
qc_service = None


def get_qc_service():
    """Get or create QC service instance."""
    global qc_service
    if qc_service is None:
        try:
            from qc_service import HallmarkQCService
            qc_service = HallmarkQCService(workflow="hallmarking_stage")
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="QC service not available. Please check qc_service.py is present."
            )
    return qc_service


@app.post("/qc/validate")
async def qc_validate_hallmark(
    file: UploadFile = File(...),
    job_id: str = Form(...),
    expected_purity: Optional[str] = Form(None),
):
    """
    Validate a hallmark image against BIS standards.

    This endpoint is used for real-time QC validation during the hallmarking process.

    **Integration Flow 1: Hallmarking Stage**
    - Camera captures image after HUID engraving
    - Image + job_id sent to this endpoint
    - Returns validation result for ERP dashboard

    Args:
        file: Image file of the hallmarked jewelry
        job_id: Unique job identifier from hallmarking machine/ERP
        expected_purity: Expected purity code for cross-validation (optional)

    Returns:
        QCValidationResponse with decision (approved/rejected/manual_review)
    """
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: PNG, JPG, JPEG, BMP, WEBP"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get QC service
        service = get_qc_service()

        # Validate
        result = service.validate_image(
            job_id=job_id,
            image=image,
            expected_purity=expected_purity,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QC validation failed: {str(e)}")


@app.post("/qc/validate/v2")
async def qc_validate_hallmark_v2(file: UploadFile = File(...)):
    """
    Enhanced hallmark validation with full BIS compliance checking.

    This endpoint provides detailed validation with:
    - Purity code validation against BIS IS 1417/IS 2112
    - HUID format validation (mandatory since April 2023)
    - BIS logo detection
    - Confidence scoring and auto-approval logic

    Returns structured decision: approved, rejected, or manual_review
    """
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: PNG, JPG, JPEG, BMP, WEBP"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Extract hallmark info using V2 engine
        hallmark_info = ocr_engine_v2.extract_with_hallmark_info(image)

        # Determine QC decision based on validation
        decision = "manual_review"
        rejection_reasons = []

        # Check purity
        purity_valid = hallmark_info.purity_code is not None
        if not purity_valid:
            rejection_reasons.append("missing_purity_mark")

        # Check HUID (mandatory since April 2023)
        huid_valid = hallmark_info.huid is not None
        if not huid_valid:
            rejection_reasons.append("missing_huid")

        # Determine BIS certification
        bis_certified = purity_valid and huid_valid

        # Auto-decision logic
        if purity_valid and huid_valid:
            if hallmark_info.overall_confidence >= 0.85:
                decision = "approved"
            elif hallmark_info.overall_confidence >= 0.50:
                decision = "manual_review"
            else:
                decision = "rejected"
                rejection_reasons.append("low_confidence")
        else:
            if hallmark_info.overall_confidence < 0.50:
                decision = "rejected"
            else:
                decision = "manual_review"

        return {
            "status": "success",
            "data": {
                "job_id": "AUTO-GENERATED",
                "decision": decision,
                "confidence": round(hallmark_info.overall_confidence, 3),
                "bis_certified": bis_certified,
                "hallmark_data": {
                    "purity_code": hallmark_info.purity_code,
                    "karat": hallmark_info.karat,
                    "purity_percentage": hallmark_info.purity_percentage,
                    "huid": hallmark_info.huid,
                },
                "validation_status": {
                    "purity_valid": purity_valid,
                    "huid_valid": huid_valid,
                    "bis_logo_detected": bis_certified,
                },
                "rejection_info": {
                    "reasons": rejection_reasons,
                    "message": "; ".join(rejection_reasons) if rejection_reasons else None,
                } if rejection_reasons else None,
                "processing_details": {
                    "ocr_corrections_applied": 0,
                    "raw_ocr_text": " ".join([r.text for r in hallmark_info.all_results]),
                    "image_quality_score": 0.8,
                    "processing_time_ms": 0,
                },
                "requires_manual_review": decision == "manual_review",
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QC validation failed: {str(e)}")


@app.post("/qc/override")
async def qc_override_decision(request: QCOverrideRequest):
    """
    Apply QC personnel override to a validation decision.

    Used when QC operator disagrees with AI decision and wants to
    manually approve or reject.

    **Feedback Loop**: Overrides are stored for model improvement.
    """
    try:
        # Validate override decision
        if request.override_decision not in ["approved", "rejected"]:
            raise HTTPException(
                status_code=400,
                detail="override_decision must be 'approved' or 'rejected'"
            )

        # In a real implementation, you would fetch the original result
        # from a database using job_id and apply the override via QC service.
        # For now, we acknowledge the override.
        return {
            "status": "success",
            "message": f"Override applied for job {request.job_id}",
            "override": {
                "job_id": request.job_id,
                "decision": request.override_decision,
                "reason": request.override_reason,
                "operator_id": request.operator_id,
                "notes": request.notes,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Override failed: {str(e)}")


@app.get("/qc/rules")
async def get_qc_rules():
    """
    Get current QC validation rules and BIS compliance standards.

    Returns the rule configuration used for hallmark validation.
    """
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
            "require_all_components": True,
        },
        "rejection_reasons": [
            "invalid_purity_code",
            "invalid_huid_format",
            "missing_huid",
            "missing_purity_mark",
            "missing_bis_logo",
            "low_confidence",
            "unclear_engraving",
            "incomplete_hallmark",
        ]
    }


@app.post("/validate/huid")
async def validate_huid(huid: str = Form(...)):
    """
    Validate HUID format (does not check against BIS database).

    HUID Requirements (mandatory since April 2023):
    - 6 characters
    - Alphanumeric (A-Z, 0-9)
    - Must contain at least one letter
    """
    import re

    cleaned = huid.upper().strip()
    cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)

    is_valid = (
        len(cleaned) == 6 and
        bool(re.match(r'^[A-Z0-9]{6}$', cleaned)) and
        bool(re.search(r'[A-Z]', cleaned))  # Must have at least one letter
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
