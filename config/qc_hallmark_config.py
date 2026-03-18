"""
QC Hallmark Configuration and Rule Sets for Jewelry Hallmarking Process

This module defines the comprehensive rule sets, validation logic, and configuration
for AI-powered OCR validation in the jewelry hallmarking QC process.

Supports two integration flows:
1. Hallmarking Stage Integration: Camera captures image after engraving, OCR validates
   against BIS rules, result shown in ERP dashboard for QC cross-check
2. Separate QC Dashboard: Dedicated dashboard for OCR results with approval/rejection workflow

BIS Standards Reference:
- IS 1417 (Gold and Gold Alloys)
- IS 2112 (Silver and Silver Alloys)
- BIS Hallmarking Scheme (April 2023 onwards - HUID mandatory)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re


class QCDecision(Enum):
    """QC Decision outcomes."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"
    PENDING = "pending"


class RejectionReason(Enum):
    """Standardized rejection reasons for QC."""
    INVALID_PURITY_CODE = "invalid_purity_code"
    INVALID_HUID_FORMAT = "invalid_huid_format"
    MISSING_HUID = "missing_huid"
    MISSING_PURITY_MARK = "missing_purity_mark"
    MISSING_BIS_LOGO = "missing_bis_logo"
    LOW_CONFIDENCE = "low_confidence"
    UNCLEAR_ENGRAVING = "unclear_engraving"
    OCR_MISMATCH = "ocr_mismatch"
    INCOMPLETE_HALLMARK = "incomplete_hallmark"
    NON_COMPLIANT_FORMAT = "non_compliant_format"
    PURITY_HUID_MISMATCH = "purity_huid_mismatch"


@dataclass
class BISComplianceRules:
    """
    BIS (Bureau of Indian Standards) Compliance Rules for Hallmarking.

    Reference: https://www.bis.gov.in/hallmarking-overview/
    """

    # Valid gold purity grades (BIS IS 1417)
    GOLD_PURITY_GRADES: Dict[str, Dict] = field(default_factory=lambda: {
        "375": {"karat": "9K", "purity": 37.5, "min_fineness": 375, "tolerance": 3},
        "585": {"karat": "14K", "purity": 58.5, "min_fineness": 583, "tolerance": 3},
        "750": {"karat": "18K", "purity": 75.0, "min_fineness": 750, "tolerance": 3},
        "875": {"karat": "21K", "purity": 87.5, "min_fineness": 875, "tolerance": 3},
        "916": {"karat": "22K", "purity": 91.6, "min_fineness": 916, "tolerance": 3},
        "958": {"karat": "23K", "purity": 95.8, "min_fineness": 958, "tolerance": 3},
        "999": {"karat": "24K", "purity": 99.9, "min_fineness": 990, "tolerance": 1},
    })

    # Silver purity grades (BIS IS 2112)
    SILVER_PURITY_GRADES: Dict[str, Dict] = field(default_factory=lambda: {
        "800": {"purity": 80.0, "grade": "Silver 800", "tolerance": 3},
        "835": {"purity": 83.5, "grade": "Silver 835", "tolerance": 3},
        "900": {"purity": 90.0, "grade": "Silver 900", "tolerance": 3},
        "925": {"purity": 92.5, "grade": "Sterling Silver", "tolerance": 3},
        "950": {"purity": 95.0, "grade": "Britannia Silver", "tolerance": 3},
        "999": {"purity": 99.9, "grade": "Fine Silver", "tolerance": 1},
    })

    # HUID Requirements (mandatory from April 2023)
    HUID_MANDATORY: bool = True
    HUID_LENGTH: int = 6
    HUID_PATTERN: str = r'^[A-Z0-9]{6}$'
    HUID_MUST_CONTAIN_LETTER: bool = True  # HUID must have at least one letter

    # BIS Logo requirements
    BIS_LOGO_REQUIRED: bool = True
    BIS_LOGO_FORMATS: List[str] = field(default_factory=lambda: [
        "BIS",
        "HALLMARK",
        "triangle_logo"  # The characteristic BIS triangle
    ])

    # Assaying & Hallmarking Centre (AHC) requirements
    AHC_MARK_REQUIRED: bool = False  # Optional but recommended

    # Jeweler identification mark
    JEWELER_MARK_REQUIRED: bool = False  # Optional


@dataclass
class QCValidationRules:
    """
    QC Validation Rules for automated approval/rejection decisions.
    """

    # Confidence thresholds
    AUTO_APPROVE_CONFIDENCE: float = 0.85  # Auto-approve if OCR confidence >= 85%
    AUTO_REJECT_CONFIDENCE: float = 0.50   # Auto-reject if OCR confidence < 50%
    MANUAL_REVIEW_RANGE: Tuple[float, float] = (0.50, 0.85)  # Manual review needed

    # Individual component confidence thresholds
    PURITY_MARK_MIN_CONFIDENCE: float = 0.80
    HUID_MIN_CONFIDENCE: float = 0.80
    BIS_LOGO_MIN_CONFIDENCE: float = 0.70

    # Validation strictness
    REQUIRE_ALL_COMPONENTS: bool = True  # Purity + HUID + BIS Logo
    ALLOW_PARTIAL_HALLMARK: bool = False  # If False, incomplete hallmarks are rejected

    # OCR correction tolerance
    ALLOW_OCR_CORRECTION: bool = True
    MAX_OCR_CORRECTIONS: int = 2  # Maximum character corrections allowed

    # Image quality requirements
    MIN_IMAGE_RESOLUTION: Tuple[int, int] = (640, 480)
    MAX_BLUR_SCORE: float = 100.0  # Laplacian variance threshold

    # Time constraints for QC
    MAX_PROCESSING_TIME_SECONDS: int = 30

    # Retry policy
    MAX_RETRIES_ON_LOW_CONFIDENCE: int = 2

    # Job ID validation
    VALIDATE_JOB_ID_FORMAT: bool = True
    JOB_ID_PATTERN: str = r'^[A-Z]{2,4}-\d{6,12}$'  # e.g., HM-123456789


@dataclass
class QCWorkflowConfig:
    """
    Configuration for QC Workflow Integration.
    """

    # Integration mode
    INTEGRATION_MODE: str = "hallmarking_stage"  # or "separate_qc_dashboard"

    # ERP Integration
    ERP_CALLBACK_URL: Optional[str] = None
    ERP_WEBHOOK_ENABLED: bool = True
    SEND_RESULT_ON_COMPLETION: bool = True

    # Dashboard settings
    SHOW_IMAGE_PREVIEW: bool = True
    SHOW_CONFIDENCE_BREAKDOWN: bool = True
    SHOW_REJECTION_REASONS: bool = True
    ALLOW_MANUAL_OVERRIDE: bool = True  # QC person can override AI decision

    # Feedback loop
    COLLECT_QC_FEEDBACK: bool = True  # Collect human QC decisions for model improvement
    FEEDBACK_STORAGE_PATH: str = "qc_feedback/"

    # Batch processing
    ENABLE_BATCH_MODE: bool = True
    BATCH_SIZE: int = 50

    # Notifications
    NOTIFY_ON_REJECTION: bool = True
    NOTIFICATION_CHANNELS: List[str] = field(default_factory=lambda: ["email", "dashboard"])


@dataclass
class QCResult:
    """
    Result structure for QC validation.
    """
    job_id: str
    decision: QCDecision
    confidence: float

    # Extracted data
    purity_code: Optional[str] = None
    karat: Optional[str] = None
    purity_percentage: Optional[float] = None
    huid: Optional[str] = None
    bis_certified: bool = False

    # Validation details
    purity_valid: bool = False
    huid_valid: bool = False
    bis_logo_detected: bool = False

    # Rejection info
    rejection_reasons: List[RejectionReason] = field(default_factory=list)
    rejection_message: str = ""

    # OCR details
    ocr_corrections_applied: int = 0
    raw_ocr_text: str = ""

    # Processing metadata
    processing_time_ms: int = 0
    image_quality_score: float = 0.0
    requires_manual_review: bool = False

    # Feedback fields
    qc_override: Optional[QCDecision] = None
    qc_feedback_notes: str = ""


# OCR Character Corrections for common hallmark misreads
OCR_CORRECTIONS = {
    # Letter to number
    'O': '0',
    'I': '1',
    'l': '1',
    'S': '5',
    'B': '8',
    'G': '6',
    'Z': '2',
    'T': '7',

    # Number to letter (for HUID)
    '0': 'O',
    '1': 'I',
    '5': 'S',
    '8': 'B',
    '6': 'G',
}

# Purity mark aliases and alternative representations
PURITY_ALIASES = {
    # Gold
    "22K916": "916", "22K": "916", "22KT": "916", "22 KT": "916",
    "18K750": "750", "18K": "750", "18KT": "750", "18 KT": "750",
    "14K585": "585", "14K": "585", "14KT": "585", "14 KT": "585",
    "9K375": "375", "9K": "375", "9KT": "375", "9 KT": "375",
    "21K875": "875", "21K": "875", "21KT": "875",
    "23K958": "958", "23K": "958", "23KT": "958",
    "24K999": "999", "24K": "999", "24KT": "999",

    # Silver
    "STERLING": "925", "STG": "925", "SS925": "925", "STER": "925",
    "925S": "925", "S925": "925",
    "FINE SILVER": "999", "FS": "999",
}


class HallmarkQCValidator:
    """
    Main validator class for Jewelry Hallmarking QC Process.

    This class implements the AI-powered validation logic for both integration flows:
    1. Hallmarking stage integration (camera capture after engraving)
    2. Separate QC dashboard (dedicated approval/rejection workflow)
    """

    def __init__(
        self,
        bis_rules: Optional[BISComplianceRules] = None,
        qc_rules: Optional[QCValidationRules] = None,
        workflow_config: Optional[QCWorkflowConfig] = None
    ):
        self.bis_rules = bis_rules or BISComplianceRules()
        self.qc_rules = qc_rules or QCValidationRules()
        self.workflow_config = workflow_config or QCWorkflowConfig()

    def validate_hallmark(
        self,
        job_id: str,
        purity_text: Optional[str],
        huid_text: Optional[str],
        bis_logo_detected: bool,
        ocr_confidence: float,
        purity_confidence: float = 1.0,
        huid_confidence: float = 1.0,
        bis_confidence: float = 1.0,
        image_quality_score: float = 1.0
    ) -> QCResult:
        """
        Validate hallmark against BIS rules and determine QC decision.

        Args:
            job_id: Unique job identifier from hallmarking stage
            purity_text: Detected purity mark text (e.g., "916", "22K")
            huid_text: Detected HUID text (6 alphanumeric chars)
            bis_logo_detected: Whether BIS logo was detected
            ocr_confidence: Overall OCR confidence score (0-1)
            purity_confidence: Confidence for purity detection
            huid_confidence: Confidence for HUID detection
            bis_confidence: Confidence for BIS logo detection
            image_quality_score: Image quality assessment score

        Returns:
            QCResult with decision, validation details, and any rejection reasons
        """
        result = QCResult(
            job_id=job_id,
            decision=QCDecision.PENDING,
            confidence=ocr_confidence,
            image_quality_score=image_quality_score
        )

        rejection_reasons = []

        # Validate job ID format
        if self.qc_rules.VALIDATE_JOB_ID_FORMAT:
            if not re.match(self.qc_rules.JOB_ID_PATTERN, job_id):
                # Don't reject, but flag for review
                result.requires_manual_review = True

        # Step 1: Validate Purity Mark
        purity_result = self._validate_purity(purity_text)
        result.purity_valid = purity_result["valid"]
        result.purity_code = purity_result.get("purity_code")
        result.karat = purity_result.get("karat")
        result.purity_percentage = purity_result.get("purity_percentage")
        result.ocr_corrections_applied = purity_result.get("corrections", 0)

        if not result.purity_valid:
            if not purity_text:
                rejection_reasons.append(RejectionReason.MISSING_PURITY_MARK)
            else:
                rejection_reasons.append(RejectionReason.INVALID_PURITY_CODE)
        elif purity_confidence < self.qc_rules.PURITY_MARK_MIN_CONFIDENCE:
            rejection_reasons.append(RejectionReason.LOW_CONFIDENCE)

        # Step 2: Validate HUID
        huid_result = self._validate_huid(huid_text)
        result.huid_valid = huid_result["valid"]
        result.huid = huid_result.get("huid")

        if self.bis_rules.HUID_MANDATORY:
            if not result.huid_valid:
                if not huid_text:
                    rejection_reasons.append(RejectionReason.MISSING_HUID)
                else:
                    rejection_reasons.append(RejectionReason.INVALID_HUID_FORMAT)
            elif huid_confidence < self.qc_rules.HUID_MIN_CONFIDENCE:
                rejection_reasons.append(RejectionReason.LOW_CONFIDENCE)

        # Step 3: Validate BIS Logo
        result.bis_logo_detected = bis_logo_detected

        if self.bis_rules.BIS_LOGO_REQUIRED and not bis_logo_detected:
            rejection_reasons.append(RejectionReason.MISSING_BIS_LOGO)
        elif bis_logo_detected and bis_confidence < self.qc_rules.BIS_LOGO_MIN_CONFIDENCE:
            rejection_reasons.append(RejectionReason.LOW_CONFIDENCE)

        # Step 4: Check overall completeness
        if self.qc_rules.REQUIRE_ALL_COMPONENTS:
            has_all = result.purity_valid and result.huid_valid and result.bis_logo_detected
            if not has_all and not self.qc_rules.ALLOW_PARTIAL_HALLMARK:
                rejection_reasons.append(RejectionReason.INCOMPLETE_HALLMARK)

        # Step 5: Determine BIS certification status
        result.bis_certified = (
            result.purity_valid and
            result.huid_valid and
            result.bis_logo_detected
        )

        # Step 6: Make final decision
        result.rejection_reasons = list(set(rejection_reasons))  # Remove duplicates

        if len(result.rejection_reasons) > 0:
            # Check if it's a hard rejection or needs manual review
            hard_rejections = [
                RejectionReason.INVALID_PURITY_CODE,
                RejectionReason.INVALID_HUID_FORMAT,
                RejectionReason.NON_COMPLIANT_FORMAT
            ]

            if any(r in hard_rejections for r in result.rejection_reasons):
                result.decision = QCDecision.REJECTED
            elif ocr_confidence < self.qc_rules.AUTO_REJECT_CONFIDENCE:
                result.decision = QCDecision.REJECTED
            elif ocr_confidence >= self.qc_rules.MANUAL_REVIEW_RANGE[0]:
                result.decision = QCDecision.MANUAL_REVIEW
                result.requires_manual_review = True
            else:
                result.decision = QCDecision.REJECTED
        else:
            # No rejection reasons - check confidence for auto-approve
            if ocr_confidence >= self.qc_rules.AUTO_APPROVE_CONFIDENCE:
                result.decision = QCDecision.APPROVED
            elif ocr_confidence >= self.qc_rules.MANUAL_REVIEW_RANGE[0]:
                result.decision = QCDecision.MANUAL_REVIEW
                result.requires_manual_review = True
            else:
                result.decision = QCDecision.REJECTED
                result.rejection_reasons.append(RejectionReason.LOW_CONFIDENCE)

        # Generate rejection message
        if result.rejection_reasons:
            result.rejection_message = self._generate_rejection_message(result.rejection_reasons)

        return result

    def _validate_purity(self, text: Optional[str]) -> Dict:
        """Validate purity mark against BIS standards."""
        if not text:
            return {"valid": False}

        cleaned = text.upper().strip()
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
        corrections = 0

        # Check aliases first
        if cleaned in PURITY_ALIASES:
            purity_code = PURITY_ALIASES[cleaned]
        else:
            # Try to extract 3-digit purity code
            matches = re.findall(r'\d{3}', cleaned)
            if matches:
                purity_code = matches[0]
            else:
                # Apply OCR corrections and retry
                if self.qc_rules.ALLOW_OCR_CORRECTION:
                    corrected = self._apply_ocr_corrections(cleaned)
                    corrections = sum(1 for a, b in zip(cleaned, corrected) if a != b)

                    if corrections <= self.qc_rules.MAX_OCR_CORRECTIONS:
                        matches = re.findall(r'\d{3}', corrected)
                        if matches:
                            purity_code = matches[0]
                        else:
                            return {"valid": False, "corrections": corrections}
                    else:
                        return {"valid": False, "corrections": corrections}
                else:
                    return {"valid": False}

        # Validate against BIS grades
        if purity_code in self.bis_rules.GOLD_PURITY_GRADES:
            grade = self.bis_rules.GOLD_PURITY_GRADES[purity_code]
            return {
                "valid": True,
                "purity_code": purity_code,
                "karat": grade["karat"],
                "purity_percentage": grade["purity"],
                "metal": "gold",
                "corrections": corrections
            }
        elif purity_code in self.bis_rules.SILVER_PURITY_GRADES:
            grade = self.bis_rules.SILVER_PURITY_GRADES[purity_code]
            return {
                "valid": True,
                "purity_code": purity_code,
                "purity_percentage": grade["purity"],
                "grade": grade["grade"],
                "metal": "silver",
                "corrections": corrections
            }

        return {"valid": False, "corrections": corrections}

    def _validate_huid(self, text: Optional[str]) -> Dict:
        """Validate HUID format."""
        if not text:
            return {"valid": False}

        cleaned = text.upper().strip()
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)

        # Find 6-character sequences
        matches = re.findall(r'[A-Z0-9]{6}', cleaned)

        for match in matches:
            # Validate against pattern
            if re.match(self.bis_rules.HUID_PATTERN, match):
                # Check if it contains at least one letter (not pure digits)
                if self.bis_rules.HUID_MUST_CONTAIN_LETTER:
                    if re.search(r'[A-Z]', match):
                        # Skip known purity aliases
                        if match not in PURITY_ALIASES:
                            return {"valid": True, "huid": match}
                else:
                    if match not in PURITY_ALIASES:
                        return {"valid": True, "huid": match}

        return {"valid": False}

    def _apply_ocr_corrections(self, text: str) -> str:
        """Apply common OCR error corrections."""
        result = text
        for wrong, correct in OCR_CORRECTIONS.items():
            result = result.replace(wrong, correct)
        return result

    def _generate_rejection_message(self, reasons: List[RejectionReason]) -> str:
        """Generate human-readable rejection message."""
        messages = {
            RejectionReason.INVALID_PURITY_CODE: "Purity code does not match BIS standards",
            RejectionReason.INVALID_HUID_FORMAT: "HUID format is invalid (must be 6 alphanumeric characters)",
            RejectionReason.MISSING_HUID: "HUID is missing (mandatory since April 2023)",
            RejectionReason.MISSING_PURITY_MARK: "Purity mark not detected",
            RejectionReason.MISSING_BIS_LOGO: "BIS logo not detected",
            RejectionReason.LOW_CONFIDENCE: "OCR confidence below acceptable threshold",
            RejectionReason.UNCLEAR_ENGRAVING: "Engraving is unclear or illegible",
            RejectionReason.OCR_MISMATCH: "OCR result does not match expected value",
            RejectionReason.INCOMPLETE_HALLMARK: "Hallmark is incomplete (missing required components)",
            RejectionReason.NON_COMPLIANT_FORMAT: "Hallmark format does not comply with BIS standards",
            RejectionReason.PURITY_HUID_MISMATCH: "Purity code and HUID combination is invalid",
        }

        reason_texts = [messages.get(r, str(r.value)) for r in reasons]
        return "; ".join(reason_texts)

    def apply_qc_override(
        self,
        result: QCResult,
        override_decision: QCDecision,
        feedback_notes: str = ""
    ) -> QCResult:
        """
        Apply QC personnel override to the AI decision.

        This is used when the QC person disagrees with the AI decision
        and wants to manually approve or reject.
        """
        result.qc_override = override_decision
        result.qc_feedback_notes = feedback_notes

        # The override becomes the final decision
        result.decision = override_decision

        return result

    def get_validation_summary(self, result: QCResult) -> Dict:
        """Get a summary suitable for ERP/dashboard display."""
        return {
            "job_id": result.job_id,
            "decision": result.decision.value,
            "confidence": round(result.confidence * 100, 1),
            "bis_certified": result.bis_certified,
            "hallmark_data": {
                "purity_code": result.purity_code,
                "karat": result.karat,
                "purity_percentage": result.purity_percentage,
                "huid": result.huid,
            },
            "validation_status": {
                "purity_valid": result.purity_valid,
                "huid_valid": result.huid_valid,
                "bis_logo_detected": result.bis_logo_detected,
            },
            "rejection_info": {
                "reasons": [r.value for r in result.rejection_reasons],
                "message": result.rejection_message,
            } if result.rejection_reasons else None,
            "requires_manual_review": result.requires_manual_review,
            "qc_override": result.qc_override.value if result.qc_override else None,
            "processing_time_ms": result.processing_time_ms,
        }


# Default configurations for different use cases
DEFAULT_STRICT_CONFIG = QCValidationRules(
    AUTO_APPROVE_CONFIDENCE=0.90,
    AUTO_REJECT_CONFIDENCE=0.60,
    REQUIRE_ALL_COMPONENTS=True,
    ALLOW_PARTIAL_HALLMARK=False,
)

DEFAULT_LENIENT_CONFIG = QCValidationRules(
    AUTO_APPROVE_CONFIDENCE=0.75,
    AUTO_REJECT_CONFIDENCE=0.40,
    REQUIRE_ALL_COMPONENTS=False,
    ALLOW_PARTIAL_HALLMARK=True,
)

HALLMARKING_STAGE_WORKFLOW = QCWorkflowConfig(
    INTEGRATION_MODE="hallmarking_stage",
    ERP_WEBHOOK_ENABLED=True,
    SEND_RESULT_ON_COMPLETION=True,
    SHOW_IMAGE_PREVIEW=True,
    ALLOW_MANUAL_OVERRIDE=True,
    COLLECT_QC_FEEDBACK=True,
)

SEPARATE_QC_DASHBOARD_WORKFLOW = QCWorkflowConfig(
    INTEGRATION_MODE="separate_qc_dashboard",
    ERP_WEBHOOK_ENABLED=True,
    SEND_RESULT_ON_COMPLETION=True,
    SHOW_IMAGE_PREVIEW=True,
    SHOW_CONFIDENCE_BREAKDOWN=True,
    SHOW_REJECTION_REASONS=True,
    ALLOW_MANUAL_OVERRIDE=True,
    COLLECT_QC_FEEDBACK=True,
    ENABLE_BATCH_MODE=True,
)
