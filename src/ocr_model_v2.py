"""
OCR Model V2 - Hallmark-specific OCR with preprocessing and validation.
Enhanced version with image preprocessing, hallmark pattern recognition,
and BIS standard validation.
"""

import sys
import re
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum

# Cache the PaddleOCR module to prevent reinitialization errors with Streamlit
if "paddleocr" not in sys.modules:
    from paddleocr import PaddleOCR
else:
    PaddleOCR = sys.modules["paddleocr"].PaddleOCR


class HallmarkType(Enum):
    """Types of hallmark components."""
    PURITY_MARK = "purity_mark"
    HUID = "huid"
    BIS_LOGO = "bis_logo"
    JEWELER_MARK = "jeweler_mark"
    CHECK = "check"
    UNKNOWN = "unknown"


@dataclass
class OCRResultV2:
    """Enhanced OCR result with hallmark-specific information."""
    text: str
    confidence: float
    hallmark_type: HallmarkType = HallmarkType.UNKNOWN
    validated: bool = False
    validation_details: Dict = field(default_factory=dict)
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class CheckInfo:
    """Check/Cheque specific information."""
    check_number: Optional[str] = None
    micr_code: Optional[str] = None
    ifsc_code: Optional[str] = None
    account_number: Optional[str] = None
    bank_name: Optional[str] = None
    is_valid_check: bool = False


@dataclass
class HallmarkInfo:
    """Complete hallmark information after validation."""
    purity_code: Optional[str] = None
    karat: Optional[str] = None
    purity_percentage: Optional[float] = None
    huid: Optional[str] = None
    bis_certified: bool = False
    overall_confidence: float = 0.0
    all_results: List[OCRResultV2] = field(default_factory=list)
    # Check information
    check_info: Optional[CheckInfo] = None


@dataclass
class PreprocessConfigV2:
    """Configuration for V2 preprocessing pipeline."""
    target_size: Tuple[int, int] = (1280, 1280)
    denoise_strength: int = 10
    clahe_clip_limit: float = 3.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    sharpen_strength: float = 1.5
    enable_reflection_removal: bool = True


class HallmarkPatterns:
    """BIS Hallmark pattern definitions and validation rules."""

    # Valid gold purity grades (BIS recognized)
    GOLD_PURITY_GRADES = {
        "375": {"karat": "9K", "purity": 37.5},
        "585": {"karat": "14K", "purity": 58.5},
        "750": {"karat": "18K", "purity": 75.0},
        "875": {"karat": "21K", "purity": 87.5},
        "916": {"karat": "22K", "purity": 91.6},
        "958": {"karat": "23K", "purity": 95.8},
        "999": {"karat": "24K", "purity": 99.9},
    }

    # Silver purity grades
    SILVER_PURITY_GRADES = {
        "800": {"purity": 80.0, "grade": "Silver 800"},
        "925": {"purity": 92.5, "grade": "Sterling Silver"},
        "950": {"purity": 95.0, "grade": "Britannia Silver"},
        "999": {"purity": 99.9, "grade": "Fine Silver"},
    }

    # Alternative representations
    PURITY_ALIASES = {
        "22K916": "916", "22K": "916", "22KT": "916",
        "18K750": "750", "18K": "750", "18KT": "750",
        "14K585": "585", "14K": "585", "14KT": "585",
        "9K375": "375", "9K": "375", "9KT": "375",
        "STERLING": "925", "STG": "925", "SS925": "925",
    }

    # HUID pattern: 6 alphanumeric characters
    HUID_PATTERN = re.compile(r'^[A-Z0-9]{6}$')

    # Common OCR errors and corrections
    OCR_CORRECTIONS = {
        'O': '0', 'I': '1', 'l': '1', 'S': '5', 'B': '8', 'G': '6',
    }

    # Check/Cheque patterns (Indian banking)
    # MICR code: 9 digits (City code 3 + Bank code 3 + Branch code 3)
    MICR_PATTERN = re.compile(r'\b\d{9}\b')

    # Check number: typically 6 digits
    CHECK_NUMBER_PATTERN = re.compile(r'\b\d{6}\b')

    # Account number: 9-18 digits (varies by bank)
    ACCOUNT_NUMBER_PATTERN = re.compile(r'\b\d{9,18}\b')

    # IFSC code: 4 letters + 0 + 6 alphanumeric (e.g., SBIN0001234)
    IFSC_PATTERN = re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b')

    # Common check keywords
    CHECK_KEYWORDS = [
        "PAY", "BEARER", "ORDER", "ACCOUNT", "PAYEE", "A/C", "CHEQUE",
        "CHECK", "BANK", "BRANCH", "IFSC", "MICR", "DATE", "RUPEES",
        "RS", "INR", "ONLY", "LAKH", "CRORE", "THOUSAND", "HUNDRED"
    ]

    # Bank names (common Indian banks)
    BANK_NAMES = [
        "STATE BANK", "SBI", "HDFC", "ICICI", "AXIS", "KOTAK", "PNB",
        "BANK OF BARODA", "BOB", "CANARA", "UNION BANK", "IDBI", "YES BANK",
        "INDUSIND", "FEDERAL BANK", "RBL", "BANDHAN", "IDFC", "AU BANK"
    ]


class ImagePreprocessorV2:
    """Image preprocessing pipeline optimized for hallmark detection on metal surfaces."""

    def __init__(self, config: Optional[PreprocessConfigV2] = None):
        self.config = config or PreprocessConfigV2()

    def process(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """Full preprocessing pipeline."""
        # Convert PIL Image to numpy array (BGR for OpenCV)
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            # Grayscale to BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            # RGBA to BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:
            # RGB to BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        results = {"original": img_array.copy()}

        # Step 1: Resize while maintaining aspect ratio
        resized = self._resize_image(img_array)
        results["resized"] = resized

        # Step 2: Remove reflections (for metal surfaces)
        if self.config.enable_reflection_removal:
            reflection_removed = self._remove_reflections(resized)
        else:
            reflection_removed = resized
        results["reflection_removed"] = reflection_removed

        # Step 3: Denoise
        denoised = self._denoise(reflection_removed)
        results["denoised"] = denoised

        # Step 4: Convert to grayscale
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        results["grayscale"] = gray

        # Step 5: CLAHE for contrast enhancement
        enhanced = self._apply_clahe(gray)
        results["contrast_enhanced"] = enhanced

        # Step 6: Sharpen for engraved text
        sharpened = self._sharpen(enhanced)
        results["sharpened"] = sharpened

        # Step 7: Adaptive binarization (for OCR)
        binary = self._adaptive_binarize(sharpened)
        results["binary"] = binary

        # Final outputs
        results["for_ocr"] = denoised  # Color image works better with PaddleOCR
        results["for_ocr_enhanced"] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return results

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        h, w = image.shape[:2]
        target_w, target_h = self.config.target_size

        scale = min(target_w / w, target_h / h)

        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif scale > 1:
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return image

    def _remove_reflections(self, image: np.ndarray) -> np.ndarray:
        """Remove specular reflections from metal surfaces."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        threshold = int(np.percentile(l_channel, 98))
        _, reflection_mask = cv2.threshold(l_channel, threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        reflection_mask = cv2.dilate(reflection_mask, kernel, iterations=1)

        result = cv2.inpaint(image, reflection_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return result

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving edges using bilateral filter."""
        return cv2.bilateralFilter(
            image,
            d=9,
            sigmaColor=self.config.denoise_strength * 7.5,
            sigmaSpace=self.config.denoise_strength * 7.5
        )

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        return clahe.apply(gray)

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        """Sharpen engraved text using unsharp masking."""
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(
            gray, 1 + self.config.sharpen_strength,
            blurred, -self.config.sharpen_strength,
            0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _adaptive_binarize(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for handling uneven illumination."""
        return cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.config.adaptive_block_size,
            C=self.config.adaptive_c
        )


class HallmarkValidator:
    """Validates hallmark information against BIS standards."""

    def __init__(self):
        self.patterns = HallmarkPatterns()

    def classify_text(self, text: str) -> HallmarkType:
        """Classify the type of hallmark text."""
        cleaned = text.upper().strip()
        cleaned_alphanum = re.sub(r'[^A-Z0-9]', '', cleaned)

        # Check for check/cheque indicators first (higher priority for document detection)
        if self.is_check_text(text):
            return HallmarkType.CHECK

        # Check for purity marks
        for alias in self.patterns.PURITY_ALIASES:
            if alias in cleaned_alphanum:
                return HallmarkType.PURITY_MARK

        # Check for 3-digit purity codes
        purity_codes = re.findall(r'\d{3}', cleaned_alphanum)
        for code in purity_codes:
            if code in self.patterns.GOLD_PURITY_GRADES or code in self.patterns.SILVER_PURITY_GRADES:
                return HallmarkType.PURITY_MARK

        # Check for HUID (6 alphanumeric) - must contain at least one letter
        # HUID is typically all letters or alphanumeric, not pure digits
        if len(cleaned_alphanum) == 6 and self.patterns.HUID_PATTERN.match(cleaned_alphanum):
            # Ensure it's not just digits (those are likely other codes)
            if re.search(r'[A-Z]', cleaned_alphanum):
                return HallmarkType.HUID

        # Check for potential HUID within text (6 consecutive alphanumeric with at least one letter)
        huid_matches = re.findall(r'[A-Z0-9]{6}', cleaned_alphanum)
        for match in huid_matches:
            if self.patterns.HUID_PATTERN.match(match) and re.search(r'[A-Z]', match):
                # Skip if it's a known purity alias like "22K916"
                if match not in self.patterns.PURITY_ALIASES:
                    return HallmarkType.HUID

        return HallmarkType.UNKNOWN

    def validate_purity_mark(self, text: str) -> Dict:
        """Validate purity mark against BIS standards."""
        cleaned = text.upper().strip()
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)

        # Check aliases first
        purity_code = None
        if cleaned in self.patterns.PURITY_ALIASES:
            purity_code = self.patterns.PURITY_ALIASES[cleaned]
        else:
            # Extract 3-digit code
            matches = re.findall(r'\d{3}', cleaned)
            for m in matches:
                if m in self.patterns.GOLD_PURITY_GRADES or m in self.patterns.SILVER_PURITY_GRADES:
                    purity_code = m
                    break

        if purity_code:
            if purity_code in self.patterns.GOLD_PURITY_GRADES:
                grade_info = self.patterns.GOLD_PURITY_GRADES[purity_code]
                return {
                    "valid": True,
                    "purity_code": purity_code,
                    "karat": grade_info["karat"],
                    "purity_percentage": grade_info["purity"],
                    "metal": "gold"
                }
            elif purity_code in self.patterns.SILVER_PURITY_GRADES:
                grade_info = self.patterns.SILVER_PURITY_GRADES[purity_code]
                return {
                    "valid": True,
                    "purity_code": purity_code,
                    "purity_percentage": grade_info["purity"],
                    "grade": grade_info["grade"],
                    "metal": "silver"
                }

        # Try OCR correction
        corrected = self._apply_ocr_corrections(cleaned)
        corrected_matches = re.findall(r'\d{3}', corrected)
        for m in corrected_matches:
            if m in self.patterns.GOLD_PURITY_GRADES:
                grade_info = self.patterns.GOLD_PURITY_GRADES[m]
                return {
                    "valid": True,
                    "purity_code": m,
                    "karat": grade_info["karat"],
                    "purity_percentage": grade_info["purity"],
                    "metal": "gold",
                    "ocr_corrected": True
                }

        return {"valid": False, "raw_text": text}

    def validate_huid(self, text: str) -> Dict:
        """Validate HUID format."""
        cleaned = text.upper().strip()
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)

        # Find 6-character alphanumeric sequences
        matches = re.findall(r'[A-Z0-9]{6}', cleaned)

        for match in matches:
            if self.patterns.HUID_PATTERN.match(match):
                # Skip known purity aliases (like 22K916)
                if match in self.patterns.PURITY_ALIASES:
                    continue
                # HUID should have at least one letter (not pure digits)
                if re.search(r'[A-Z]', match):
                    return {
                        "valid": True,
                        "huid": match,
                        "format_valid": True
                    }

        # Also try to find HUID patterns that are purely alphabetic (common format)
        alpha_matches = re.findall(r'[A-Z]{6}', cleaned)
        for match in alpha_matches:
            return {
                "valid": True,
                "huid": match,
                "format_valid": True
            }

        return {"valid": False, "raw_text": text}

    def _apply_ocr_corrections(self, text: str) -> str:
        """Apply common OCR error corrections."""
        result = text
        for wrong, correct in self.patterns.OCR_CORRECTIONS.items():
            result = result.replace(wrong, correct)
        return result

    def is_check_text(self, text: str) -> bool:
        """Check if text contains check/cheque related content."""
        upper_text = text.upper()

        # Check for check keywords
        keyword_count = sum(1 for kw in self.patterns.CHECK_KEYWORDS if kw in upper_text)

        # Check for bank names
        bank_found = any(bank in upper_text for bank in self.patterns.BANK_NAMES)

        # Check for IFSC pattern
        ifsc_found = bool(self.patterns.IFSC_PATTERN.search(upper_text))

        # Check for MICR pattern (9 digits)
        micr_found = bool(self.patterns.MICR_PATTERN.search(text))

        # Classify as check if multiple indicators are present
        return (keyword_count >= 2) or bank_found or ifsc_found or (keyword_count >= 1 and micr_found)

    def validate_check(self, text: str) -> Dict:
        """Validate and extract check information from text."""
        upper_text = text.upper()
        result = {
            "valid": False,
            "check_number": None,
            "micr_code": None,
            "ifsc_code": None,
            "account_number": None,
            "bank_name": None
        }

        # Extract IFSC code
        ifsc_match = self.patterns.IFSC_PATTERN.search(upper_text)
        if ifsc_match:
            result["ifsc_code"] = ifsc_match.group()

        # Extract MICR code (9 digits)
        micr_matches = self.patterns.MICR_PATTERN.findall(text)
        if micr_matches:
            # MICR is typically the first 9-digit number
            result["micr_code"] = micr_matches[0]

        # Extract account number (9-18 digits)
        account_matches = self.patterns.ACCOUNT_NUMBER_PATTERN.findall(text)
        for acc in account_matches:
            # Skip if it's the MICR code
            if acc != result.get("micr_code") and len(acc) >= 10:
                result["account_number"] = acc
                break

        # Extract check number (6 digits)
        check_matches = self.patterns.CHECK_NUMBER_PATTERN.findall(text)
        for chk in check_matches:
            # Skip if it's part of MICR or account
            if chk not in str(result.get("micr_code", "")) and chk not in str(result.get("account_number", "")):
                result["check_number"] = chk
                break

        # Detect bank name
        for bank in self.patterns.BANK_NAMES:
            if bank in upper_text:
                result["bank_name"] = bank
                break

        # Mark as valid if we found key check indicators
        if result["ifsc_code"] or result["micr_code"] or (result["check_number"] and result["bank_name"]):
            result["valid"] = True

        return result


class OCREngineV2:
    """
    OCR Engine V2 with hallmark-specific preprocessing and validation.

    Features:
    - Advanced image preprocessing for metal surfaces
    - Hallmark pattern recognition
    - BIS standard validation
    - Purity code and HUID extraction
    """

    def __init__(self, model_dir: str = None, enable_preprocessing: bool = True):
        """
        Initialize OCR engine V2.

        Args:
            model_dir: Path to custom model directory (for fine-tuned models)
            enable_preprocessing: Enable advanced preprocessing pipeline
        """
        self.enable_preprocessing = enable_preprocessing
        self.preprocessor = ImagePreprocessorV2()
        self.validator = HallmarkValidator()

        # PaddleOCR configuration
        config = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "lang": "en",
        }

        if model_dir:
            config["text_detection_model_dir"] = f"{model_dir}/det"
            config["text_recognition_model_dir"] = f"{model_dir}/rec"

        self.ocr = PaddleOCR(**config)

    def extract_with_hallmark_info(self, image: Image.Image) -> HallmarkInfo:
        """
        Extract text from image with full hallmark validation.

        Args:
            image: PIL Image object

        Returns:
            HallmarkInfo with validated hallmark data
        """
        results = self.extract_text_with_confidence(image)

        hallmark_info = HallmarkInfo(all_results=results)

        # Collect all text for comprehensive HUID search
        all_text = " ".join([r.text for r in results])

        # Process each result
        check_data = {}
        for r in results:
            if r.hallmark_type == HallmarkType.PURITY_MARK and r.validated:
                details = r.validation_details
                if details.get("valid"):
                    hallmark_info.purity_code = details.get("purity_code")
                    hallmark_info.karat = details.get("karat")
                    hallmark_info.purity_percentage = details.get("purity_percentage")

            elif r.hallmark_type == HallmarkType.HUID and r.validated:
                details = r.validation_details
                if details.get("valid"):
                    hallmark_info.huid = details.get("huid")

            elif r.hallmark_type == HallmarkType.CHECK and r.validated:
                details = r.validation_details
                # Merge check data from multiple text regions
                for key in ["check_number", "micr_code", "ifsc_code", "account_number", "bank_name"]:
                    if details.get(key) and not check_data.get(key):
                        check_data[key] = details.get(key)

        # If HUID not found yet, scan all text for potential HUIDs
        # This handles cases where HUID appears alongside purity marks in same text region
        if not hallmark_info.huid:
            huid_result = self.validator.validate_huid(all_text)
            if huid_result.get("valid"):
                potential_huid = huid_result.get("huid")
                # Make sure it's not part of a purity code
                if potential_huid and potential_huid not in HallmarkPatterns.PURITY_ALIASES:
                    # Check it has at least one letter (not pure digits)
                    if re.search(r'[A-Z]', potential_huid):
                        hallmark_info.huid = potential_huid

        # Build CheckInfo if check data was found
        if check_data:
            hallmark_info.check_info = CheckInfo(
                check_number=check_data.get("check_number"),
                micr_code=check_data.get("micr_code"),
                ifsc_code=check_data.get("ifsc_code"),
                account_number=check_data.get("account_number"),
                bank_name=check_data.get("bank_name"),
                is_valid_check=bool(check_data.get("ifsc_code") or check_data.get("micr_code"))
            )

        # Calculate overall confidence
        if results:
            hallmark_info.overall_confidence = sum(r.confidence for r in results) / len(results)

        # Check BIS certification (purity mark + HUID present)
        hallmark_info.bis_certified = bool(hallmark_info.purity_code and hallmark_info.huid)

        return hallmark_info

    def extract_text_with_confidence(self, image: Image.Image) -> List[OCRResultV2]:
        """
        Extract text from image with confidence scores and hallmark validation.

        Args:
            image: PIL Image object

        Returns:
            List of OCRResultV2 objects with hallmark-specific information
        """
        # Use raw image directly without preprocessing
        img_for_ocr = np.array(image)

        # Run OCR
        result = self.ocr.predict(img_for_ocr)

        if not result:
            return []

        results = []
        for item in result:
            if "rec_texts" in item and "rec_scores" in item:
                texts = item["rec_texts"]
                scores = item["rec_scores"]

                # Get bounding boxes if available
                bboxes = item.get("dt_polys", [None] * len(texts))

                for i, (text, score) in enumerate(zip(texts, scores)):
                    # Classify hallmark type
                    hallmark_type = self.validator.classify_text(text)

                    # Validate based on type
                    validated = False
                    validation_details = {}

                    if hallmark_type == HallmarkType.PURITY_MARK:
                        validation_details = self.validator.validate_purity_mark(text)
                        validated = validation_details.get("valid", False)

                    elif hallmark_type == HallmarkType.HUID:
                        validation_details = self.validator.validate_huid(text)
                        validated = validation_details.get("valid", False)

                    elif hallmark_type == HallmarkType.CHECK:
                        validation_details = self.validator.validate_check(text)
                        validated = validation_details.get("valid", False)

                    # Get bounding box
                    bbox = None
                    if bboxes[i] is not None:
                        try:
                            poly = bboxes[i]
                            x_coords = [p[0] for p in poly]
                            y_coords = [p[1] for p in poly]
                            bbox = (int(min(x_coords)), int(min(y_coords)),
                                   int(max(x_coords)), int(max(y_coords)))
                        except (IndexError, TypeError):
                            pass

                    results.append(OCRResultV2(
                        text=text,
                        confidence=float(score),
                        hallmark_type=hallmark_type,
                        validated=validated,
                        validation_details=validation_details,
                        bbox=bbox
                    ))

        return results

    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from image (simple version).

        Args:
            image: PIL Image object

        Returns:
            Extracted text as a single string
        """
        results = self.extract_text_with_confidence(image)
        return "\n".join([r.text for r in results])

    def get_preprocessing_stages(self, image: Image.Image) -> Dict[str, Image.Image]:
        """
        Get all preprocessing stages as PIL Images (for visualization).

        Args:
            image: PIL Image object

        Returns:
            Dictionary of stage names to PIL Images
        """
        processed = self.preprocessor.process(image)

        pil_images = {}
        for name, img_array in processed.items():
            if len(img_array.shape) == 2:
                # Grayscale
                pil_images[name] = Image.fromarray(img_array)
            else:
                # Color (BGR to RGB)
                pil_images[name] = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

        return pil_images
