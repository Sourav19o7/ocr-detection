"""
FastAPI OCR API - Extract text from images.
Deploy on Render or run locally with: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import sys

# Import OCR engine
sys.path.insert(0, "src")
from ocr_model import OCREngine, OCRResult


app = FastAPI(
    title="OCR API",
    description="Extract text from images using PaddleOCR",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR engine at startup
ocr_engine = None


class TextResult(BaseModel):
    text: str
    confidence: float
    approved: bool


class OCRResponse(BaseModel):
    success: bool
    results: list[TextResult]
    full_text: str
    average_confidence: float


@app.on_event("startup")
async def startup_event():
    """Load OCR model on startup."""
    global ocr_engine
    ocr_engine = OCREngine()


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
