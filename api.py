"""
FastAPI OCR API - Extract text from images.
Deploy on Render or run locally with: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import io
import sys

# Import OCR engines
sys.path.insert(0, "src")
from ocr_model import OCREngine, OCRResult
from ocr_model_v2 import OCREngineV2, OCRResultV2, HallmarkInfo, HallmarkType, CheckInfo


app = FastAPI(
    title="OCR API",
    description="Extract text from images using PaddleOCR (v1 and v2 with hallmark detection)",
    version="2.0.0"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
