"""
OCR Model Module - PaddleOCR wrapper for text detection from images.
Structured for easy fine-tuning with custom datasets.
"""

import sys
import numpy as np
from PIL import Image
from dataclasses import dataclass

# Cache the PaddleOCR module to prevent reinitialization errors with Streamlit
if "paddleocr" not in sys.modules:
    from paddleocr import PaddleOCR
else:
    PaddleOCR = sys.modules["paddleocr"].PaddleOCR


@dataclass
class OCRResult:
    """OCR result with text and confidence score."""
    text: str
    confidence: float


class OCREngine:
    """
    OCR Engine using PaddleOCR.

    To fine-tune with your own dataset:
    1. Place your trained model files in the 'models/' directory
    2. Update the model paths in the __init__ method
    """

    def __init__(self, model_dir: str = None):
        """
        Initialize OCR engine.

        Args:
            model_dir: Path to custom model directory (for fine-tuned models)
        """
        # Default PaddleOCR configuration
        # For fine-tuned models, specify: text_detection_model_dir, text_recognition_model_dir
        config = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "lang": "en",
        }

        # Add custom model paths if provided
        if model_dir:
            config["text_detection_model_dir"] = f"{model_dir}/det"
            config["text_recognition_model_dir"] = f"{model_dir}/rec"

        self.ocr = PaddleOCR(**config)

    def extract_text_with_confidence(self, image: Image.Image) -> list[OCRResult]:
        """
        Extract text from an image with confidence scores.

        Args:
            image: PIL Image object

        Returns:
            List of OCRResult objects with text and confidence
        """
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run OCR
        result = self.ocr.predict(img_array)

        # Parse results with confidence
        if not result:
            return []

        results = []
        for item in result:
            if "rec_texts" in item and "rec_scores" in item:
                texts = item["rec_texts"]
                scores = item["rec_scores"]
                for text, score in zip(texts, scores):
                    results.append(OCRResult(text=text, confidence=float(score)))

        return results

    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from an image (simple version without confidence).

        Args:
            image: PIL Image object

        Returns:
            Extracted text as a single string
        """
        results = self.extract_text_with_confidence(image)
        return "\n".join([r.text for r in results])
