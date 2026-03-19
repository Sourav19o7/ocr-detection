"""
Jewelry-Specific Rulesets for Hallmark QC Validation.

This module defines item-specific rules for different types of jewelry,
including marking positions, error categories, and validation criteria.

Each jewelry type has unique requirements for:
- Hallmark positioning
- Acceptable marking areas
- Size constraints
- Common defect patterns
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class JewelryType(Enum):
    """Types of jewelry items."""
    RING = "ring"
    BANGLE = "bangle"
    CHAIN = "chain"
    NECKLACE = "necklace"
    PENDANT = "pendant"
    EARRING = "earring"
    BRACELET = "bracelet"
    ANKLET = "anklet"
    NOSE_PIN = "nose_pin"
    MANGALSUTRA = "mangalsutra"
    COIN = "coin"
    BAR = "bar"
    OTHER = "other"


class ErrorCategory(Enum):
    """Categories of QC errors/issues."""
    # Image Quality Errors
    IMAGE_BLURRED = "image_blurred"
    IMAGE_OVEREXPOSED = "image_overexposed"
    IMAGE_UNDEREXPOSED = "image_underexposed"
    IMAGE_REFLECTION = "image_reflection"
    IMAGE_LOW_RESOLUTION = "image_low_resolution"
    IMAGE_OUT_OF_FOCUS = "image_out_of_focus"

    # OCR/Detection Errors
    LOW_CONFIDENCE = "low_confidence"
    PARTIAL_DETECTION = "partial_detection"
    OCR_MISREAD = "ocr_misread"
    MULTIPLE_DETECTIONS = "multiple_detections"
    NO_DETECTION = "no_detection"

    # Hallmark Content Errors
    INVALID_PURITY_CODE = "invalid_purity_code"
    INVALID_HUID_FORMAT = "invalid_huid_format"
    MISSING_HUID = "missing_huid"
    MISSING_PURITY_MARK = "missing_purity_mark"
    MISSING_BIS_LOGO = "missing_bis_logo"
    INCOMPLETE_HALLMARK = "incomplete_hallmark"

    # Position/Placement Errors
    WRONG_POSITION = "wrong_position"
    MARKING_AT_6_OCLOCK = "marking_at_6_oclock"  # Ring-specific: bottom inside
    MARKING_OBSCURED = "marking_obscured"
    MARKING_ON_DECORATIVE_AREA = "marking_on_decorative_area"
    MARKING_TOO_CLOSE_TO_EDGE = "marking_too_close_to_edge"

    # Engraving Quality Errors
    SHALLOW_ENGRAVING = "shallow_engraving"
    UNEVEN_ENGRAVING = "uneven_engraving"
    DOUBLE_STRIKE = "double_strike"
    SMUDGED_MARKING = "smudged_marking"
    WORN_MARKING = "worn_marking"

    # Size/Dimension Errors
    MARKING_TOO_SMALL = "marking_too_small"
    MARKING_TOO_LARGE = "marking_too_large"
    INCONSISTENT_SIZE = "inconsistent_size"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    CRITICAL = "critical"      # Auto-reject, cannot proceed
    MAJOR = "major"            # Needs correction before approval
    MINOR = "minor"            # Warning, can be approved with note
    INFO = "info"              # Informational only


@dataclass
class ErrorDetail:
    """Detailed error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[str] = None
    suggestion: Optional[str] = None
    confidence_impact: float = 0.0  # How much this affects confidence (0-1)


@dataclass
class ConfidenceBenchmark:
    """Confidence score benchmarks and thresholds."""
    # Overall thresholds
    auto_approve_threshold: float = 0.85
    manual_review_threshold: float = 0.50
    auto_reject_threshold: float = 0.50

    # Component-specific thresholds
    purity_mark_min: float = 0.80
    huid_min: float = 0.80
    bis_logo_min: float = 0.70

    # Image quality thresholds
    blur_score_max: float = 100.0  # Laplacian variance
    min_resolution: Tuple[int, int] = (640, 480)

    # Confidence score interpretation
    SCORE_RANGES: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "excellent": (0.95, 1.0),
        "good": (0.85, 0.95),
        "acceptable": (0.70, 0.85),
        "poor": (0.50, 0.70),
        "unacceptable": (0.0, 0.50),
    })

    def get_score_interpretation(self, score: float) -> str:
        """Get human-readable interpretation of confidence score."""
        for label, (low, high) in self.SCORE_RANGES.items():
            if low <= score <= high:
                return label
        return "unknown"

    def get_score_color(self, score: float) -> str:
        """Get color code for score visualization."""
        if score >= 0.85:
            return "#2ecc71"  # Green
        elif score >= 0.70:
            return "#f1c40f"  # Yellow
        elif score >= 0.50:
            return "#e67e22"  # Orange
        else:
            return "#e74c3c"  # Red


@dataclass
class MarkingPosition:
    """Defines acceptable marking positions for jewelry."""
    name: str
    description: str
    is_acceptable: bool = True
    is_preferred: bool = False
    notes: Optional[str] = None


@dataclass
class JewelryRuleset:
    """Complete ruleset for a specific jewelry type."""
    jewelry_type: JewelryType
    display_name: str
    description: str

    # Marking position rules
    acceptable_positions: List[MarkingPosition] = field(default_factory=list)
    forbidden_positions: List[MarkingPosition] = field(default_factory=list)

    # Size constraints for marking
    min_marking_size_mm: float = 0.5
    max_marking_size_mm: float = 3.0

    # Special rules
    special_rules: List[str] = field(default_factory=list)

    # Common issues for this jewelry type
    common_issues: List[ErrorCategory] = field(default_factory=list)

    # Weight threshold for mandatory hallmarking (in grams)
    min_weight_for_hallmark: float = 2.0

    # Additional validation rules
    requires_both_sides_check: bool = False
    requires_clasp_check: bool = False
    requires_joint_check: bool = False


# Pre-defined rulesets for common jewelry types
RING_RULESET = JewelryRuleset(
    jewelry_type=JewelryType.RING,
    display_name="Ring",
    description="Finger rings including engagement, wedding, and fashion rings",
    acceptable_positions=[
        MarkingPosition(
            name="12_oclock",
            description="Inside of ring at 12 o'clock position (top)",
            is_acceptable=True,
            is_preferred=True,
            notes="Preferred position for rings"
        ),
        MarkingPosition(
            name="3_oclock",
            description="Inside of ring at 3 o'clock position (right side)",
            is_acceptable=True,
            is_preferred=False
        ),
        MarkingPosition(
            name="9_oclock",
            description="Inside of ring at 9 o'clock position (left side)",
            is_acceptable=True,
            is_preferred=False
        ),
    ],
    forbidden_positions=[
        MarkingPosition(
            name="6_oclock",
            description="Inside of ring at 6 o'clock position (bottom)",
            is_acceptable=False,
            notes="FORBIDDEN: Bottom position causes wear and illegibility over time. Marking gets scratched against surfaces when ring is placed down."
        ),
        MarkingPosition(
            name="outside_visible",
            description="On the outer visible surface of the ring",
            is_acceptable=False,
            notes="FORBIDDEN: Affects aesthetics and design"
        ),
    ],
    min_marking_size_mm=0.8,
    max_marking_size_mm=2.0,
    special_rules=[
        "Hallmark must be inside the band",
        "Marking should not interfere with stone settings",
        "For thin bands (<2mm width), marking may be smaller",
        "Solitaire rings: avoid area near prongs",
        "6 o'clock position (bottom inside) is NOT acceptable",
    ],
    common_issues=[
        ErrorCategory.MARKING_AT_6_OCLOCK,
        ErrorCategory.SHALLOW_ENGRAVING,
        ErrorCategory.MARKING_TOO_SMALL,
        ErrorCategory.IMAGE_REFLECTION,
    ],
    min_weight_for_hallmark=2.0,
)

BANGLE_RULESET = JewelryRuleset(
    jewelry_type=JewelryType.BANGLE,
    display_name="Bangle",
    description="Rigid circular bracelets worn on the wrist",
    acceptable_positions=[
        MarkingPosition(
            name="inside_near_opening",
            description="Inside surface near the opening/clasp",
            is_acceptable=True,
            is_preferred=True
        ),
        MarkingPosition(
            name="inside_flat_area",
            description="Inside flat surface area",
            is_acceptable=True,
            is_preferred=False
        ),
    ],
    forbidden_positions=[
        MarkingPosition(
            name="outside_decorative",
            description="On decorative outer surface",
            is_acceptable=False,
            notes="Must not interfere with design"
        ),
    ],
    min_marking_size_mm=1.0,
    max_marking_size_mm=3.0,
    special_rules=[
        "Marking should be on inner surface",
        "For kada (thick bangles), marking near opening",
        "Should not be on carved/embossed areas",
    ],
    common_issues=[
        ErrorCategory.MARKING_ON_DECORATIVE_AREA,
        ErrorCategory.UNEVEN_ENGRAVING,
        ErrorCategory.IMAGE_REFLECTION,
    ],
    min_weight_for_hallmark=2.0,
)

CHAIN_RULESET = JewelryRuleset(
    jewelry_type=JewelryType.CHAIN,
    display_name="Chain",
    description="Neck chains and rope chains",
    acceptable_positions=[
        MarkingPosition(
            name="clasp_area",
            description="On or near the clasp/lobster claw",
            is_acceptable=True,
            is_preferred=True
        ),
        MarkingPosition(
            name="tag_attached",
            description="On a small tag attached near clasp",
            is_acceptable=True,
            is_preferred=True
        ),
        MarkingPosition(
            name="end_link",
            description="On the end link before clasp",
            is_acceptable=True,
            is_preferred=False
        ),
    ],
    forbidden_positions=[
        MarkingPosition(
            name="middle_links",
            description="On middle chain links",
            is_acceptable=False,
            notes="Would weaken chain structure"
        ),
    ],
    min_marking_size_mm=0.5,
    max_marking_size_mm=2.0,
    special_rules=[
        "Marking on clasp or attached tag",
        "Must not weaken any chain link",
        "For very fine chains, tag is mandatory",
    ],
    common_issues=[
        ErrorCategory.MARKING_TOO_SMALL,
        ErrorCategory.IMAGE_OUT_OF_FOCUS,
        ErrorCategory.PARTIAL_DETECTION,
    ],
    min_weight_for_hallmark=2.0,
    requires_clasp_check=True,
)

NECKLACE_RULESET = JewelryRuleset(
    jewelry_type=JewelryType.NECKLACE,
    display_name="Necklace",
    description="Decorative necklaces with pendants or elaborate designs",
    acceptable_positions=[
        MarkingPosition(
            name="clasp_area",
            description="On or near the clasp",
            is_acceptable=True,
            is_preferred=True
        ),
        MarkingPosition(
            name="back_of_pendant",
            description="On the back of main pendant",
            is_acceptable=True,
            is_preferred=True
        ),
        MarkingPosition(
            name="tag_attached",
            description="On attached hallmark tag",
            is_acceptable=True,
            is_preferred=False
        ),
    ],
    forbidden_positions=[
        MarkingPosition(
            name="front_visible",
            description="On front visible decorative area",
            is_acceptable=False
        ),
    ],
    min_marking_size_mm=0.8,
    max_marking_size_mm=2.5,
    special_rules=[
        "Heavy necklaces may need multiple marks",
        "Each detachable component should be marked",
        "Kundan/Polki: mark on frame, not stones",
    ],
    common_issues=[
        ErrorCategory.MARKING_ON_DECORATIVE_AREA,
        ErrorCategory.MULTIPLE_DETECTIONS,
    ],
    min_weight_for_hallmark=2.0,
)

EARRING_RULESET = JewelryRuleset(
    jewelry_type=JewelryType.EARRING,
    display_name="Earring",
    description="Earrings including studs, drops, and hoops",
    acceptable_positions=[
        MarkingPosition(
            name="back_plate",
            description="On the back plate or post base",
            is_acceptable=True,
            is_preferred=True
        ),
        MarkingPosition(
            name="earring_back",
            description="On the earring back/stopper",
            is_acceptable=True,
            is_preferred=False
        ),
        MarkingPosition(
            name="hook_base",
            description="On the base of hook/wire",
            is_acceptable=True,
            is_preferred=False
        ),
    ],
    forbidden_positions=[
        MarkingPosition(
            name="visible_front",
            description="On the visible front design",
            is_acceptable=False
        ),
        MarkingPosition(
            name="on_stones",
            description="On or near stone settings",
            is_acceptable=False
        ),
    ],
    min_marking_size_mm=0.5,
    max_marking_size_mm=1.5,
    special_rules=[
        "Both earrings in pair should be marked",
        "For small studs (<2g), one mark acceptable",
        "Mark should not affect wearability",
    ],
    common_issues=[
        ErrorCategory.MARKING_TOO_SMALL,
        ErrorCategory.SHALLOW_ENGRAVING,
        ErrorCategory.IMAGE_OUT_OF_FOCUS,
    ],
    min_weight_for_hallmark=2.0,
)

PENDANT_RULESET = JewelryRuleset(
    jewelry_type=JewelryType.PENDANT,
    display_name="Pendant",
    description="Pendants and lockets",
    acceptable_positions=[
        MarkingPosition(
            name="back_surface",
            description="On the back flat surface",
            is_acceptable=True,
            is_preferred=True
        ),
        MarkingPosition(
            name="bail_area",
            description="On or near the bail (loop)",
            is_acceptable=True,
            is_preferred=False
        ),
        MarkingPosition(
            name="inside_locket",
            description="Inside of locket (if applicable)",
            is_acceptable=True,
            is_preferred=False
        ),
    ],
    forbidden_positions=[
        MarkingPosition(
            name="front_design",
            description="On front decorative surface",
            is_acceptable=False
        ),
    ],
    min_marking_size_mm=0.6,
    max_marking_size_mm=2.0,
    special_rules=[
        "Back surface marking preferred",
        "For lockets, inside marking acceptable",
        "Should not interfere with stone settings",
    ],
    common_issues=[
        ErrorCategory.MARKING_ON_DECORATIVE_AREA,
        ErrorCategory.IMAGE_REFLECTION,
    ],
    min_weight_for_hallmark=2.0,
)

MANGALSUTRA_RULESET = JewelryRuleset(
    jewelry_type=JewelryType.MANGALSUTRA,
    display_name="Mangalsutra",
    description="Traditional Indian wedding necklace",
    acceptable_positions=[
        MarkingPosition(
            name="pendant_back",
            description="On the back of main pendant",
            is_acceptable=True,
            is_preferred=True
        ),
        MarkingPosition(
            name="clasp_area",
            description="On or near clasp",
            is_acceptable=True,
            is_preferred=True
        ),
        MarkingPosition(
            name="vati_back",
            description="On back of vati (if gold)",
            is_acceptable=True,
            is_preferred=False
        ),
    ],
    forbidden_positions=[
        MarkingPosition(
            name="black_beads",
            description="On black bead sections",
            is_acceptable=False
        ),
        MarkingPosition(
            name="front_pendant",
            description="On front of pendant",
            is_acceptable=False
        ),
    ],
    min_marking_size_mm=0.8,
    max_marking_size_mm=2.0,
    special_rules=[
        "Each gold component should be marked",
        "Pendant and clasp both need marking if separate",
        "Black bead (mangal) sections excluded from marking",
    ],
    common_issues=[
        ErrorCategory.PARTIAL_DETECTION,
        ErrorCategory.MULTIPLE_DETECTIONS,
    ],
    min_weight_for_hallmark=2.0,
)


# Registry of all rulesets
JEWELRY_RULESETS: Dict[JewelryType, JewelryRuleset] = {
    JewelryType.RING: RING_RULESET,
    JewelryType.BANGLE: BANGLE_RULESET,
    JewelryType.CHAIN: CHAIN_RULESET,
    JewelryType.NECKLACE: NECKLACE_RULESET,
    JewelryType.EARRING: EARRING_RULESET,
    JewelryType.PENDANT: PENDANT_RULESET,
    JewelryType.MANGALSUTRA: MANGALSUTRA_RULESET,
}


# Error messages and suggestions
ERROR_DETAILS: Dict[ErrorCategory, Dict] = {
    ErrorCategory.IMAGE_BLURRED: {
        "message": "Image is blurred or out of focus",
        "severity": ErrorSeverity.MAJOR,
        "suggestion": "Retake image with camera held steady, ensure proper focus on hallmark area",
        "confidence_impact": 0.3,
    },
    ErrorCategory.IMAGE_OVEREXPOSED: {
        "message": "Image is overexposed (too bright)",
        "severity": ErrorSeverity.MAJOR,
        "suggestion": "Reduce lighting or adjust camera exposure settings",
        "confidence_impact": 0.2,
    },
    ErrorCategory.IMAGE_UNDEREXPOSED: {
        "message": "Image is underexposed (too dark)",
        "severity": ErrorSeverity.MAJOR,
        "suggestion": "Increase lighting or adjust camera exposure settings",
        "confidence_impact": 0.25,
    },
    ErrorCategory.IMAGE_REFLECTION: {
        "message": "Specular reflection detected on metal surface",
        "severity": ErrorSeverity.MINOR,
        "suggestion": "Use diffused lighting or change angle to avoid reflections",
        "confidence_impact": 0.15,
    },
    ErrorCategory.IMAGE_LOW_RESOLUTION: {
        "message": "Image resolution too low for accurate OCR",
        "severity": ErrorSeverity.MAJOR,
        "suggestion": "Use higher resolution camera or move closer to subject",
        "confidence_impact": 0.25,
    },
    ErrorCategory.LOW_CONFIDENCE: {
        "message": "OCR confidence score below acceptable threshold",
        "severity": ErrorSeverity.MAJOR,
        "suggestion": "Retake image with better lighting and focus, or verify manually",
        "confidence_impact": 0.0,  # Already reflected in score
    },
    ErrorCategory.INVALID_PURITY_CODE: {
        "message": "Detected purity code does not match BIS standards",
        "severity": ErrorSeverity.CRITICAL,
        "suggestion": "Verify purity code manually. Valid codes: 375, 585, 750, 875, 916, 958, 999",
        "confidence_impact": 0.0,
    },
    ErrorCategory.INVALID_HUID_FORMAT: {
        "message": "HUID format is invalid (must be 6 alphanumeric characters)",
        "severity": ErrorSeverity.CRITICAL,
        "suggestion": "HUID must be exactly 6 characters with at least one letter",
        "confidence_impact": 0.0,
    },
    ErrorCategory.MISSING_HUID: {
        "message": "HUID not detected (mandatory since April 2023)",
        "severity": ErrorSeverity.CRITICAL,
        "suggestion": "Ensure HUID is properly engraved and visible in image",
        "confidence_impact": 0.0,
    },
    ErrorCategory.MISSING_PURITY_MARK: {
        "message": "Purity mark not detected",
        "severity": ErrorSeverity.CRITICAL,
        "suggestion": "Ensure purity code (916, 750, etc.) is clearly visible",
        "confidence_impact": 0.0,
    },
    ErrorCategory.MARKING_AT_6_OCLOCK: {
        "message": "Ring hallmark detected at 6 o'clock position (bottom inside)",
        "severity": ErrorSeverity.CRITICAL,
        "suggestion": "Hallmark must be at 12, 3, or 9 o'clock position. 6 o'clock (bottom) is NOT acceptable as it wears off quickly",
        "confidence_impact": 0.0,
    },
    ErrorCategory.WRONG_POSITION: {
        "message": "Hallmark is in an unacceptable position for this jewelry type",
        "severity": ErrorSeverity.MAJOR,
        "suggestion": "Refer to jewelry-specific positioning guidelines",
        "confidence_impact": 0.0,
    },
    ErrorCategory.SHALLOW_ENGRAVING: {
        "message": "Engraving appears too shallow",
        "severity": ErrorSeverity.MAJOR,
        "suggestion": "Engraving must be deep enough to remain legible over time",
        "confidence_impact": 0.1,
    },
    ErrorCategory.MARKING_TOO_SMALL: {
        "message": "Hallmark marking is too small to read reliably",
        "severity": ErrorSeverity.MAJOR,
        "suggestion": "Marking size should meet minimum requirements for jewelry type",
        "confidence_impact": 0.2,
    },
}


class JewelryRulesetValidator:
    """Validates hallmarks against jewelry-specific rulesets."""

    def __init__(self):
        self.rulesets = JEWELRY_RULESETS
        self.error_details = ERROR_DETAILS
        self.confidence_benchmark = ConfidenceBenchmark()

    def get_ruleset(self, jewelry_type: JewelryType) -> JewelryRuleset:
        """Get ruleset for a jewelry type."""
        return self.rulesets.get(jewelry_type)

    def validate_position(
        self,
        jewelry_type: JewelryType,
        position: str
    ) -> Tuple[bool, Optional[ErrorDetail]]:
        """
        Validate if marking position is acceptable for jewelry type.

        Returns:
            Tuple of (is_valid, error_detail if invalid)
        """
        ruleset = self.get_ruleset(jewelry_type)
        if not ruleset:
            return True, None  # No specific rules

        # Check forbidden positions
        for forbidden in ruleset.forbidden_positions:
            if position.lower() == forbidden.name.lower():
                error_info = self.error_details.get(
                    ErrorCategory.MARKING_AT_6_OCLOCK if "6_oclock" in position.lower()
                    else ErrorCategory.WRONG_POSITION
                )
                return False, ErrorDetail(
                    category=ErrorCategory.MARKING_AT_6_OCLOCK if "6_oclock" in position.lower()
                             else ErrorCategory.WRONG_POSITION,
                    severity=ErrorSeverity.CRITICAL,
                    message=error_info["message"],
                    details=forbidden.notes,
                    suggestion=error_info["suggestion"],
                )

        return True, None

    def assess_image_quality(
        self,
        blur_score: float,
        resolution: Tuple[int, int],
        brightness: float = 0.5,
        has_reflection: bool = False
    ) -> List[ErrorDetail]:
        """
        Assess image quality and return list of issues.

        Args:
            blur_score: Laplacian variance (higher = sharper)
            resolution: Image (width, height)
            brightness: Average brightness 0-1
            has_reflection: Whether reflection detected

        Returns:
            List of ErrorDetail for any issues found
        """
        errors = []

        # Check blur
        if blur_score < self.confidence_benchmark.blur_score_max:
            error_info = self.error_details[ErrorCategory.IMAGE_BLURRED]
            errors.append(ErrorDetail(
                category=ErrorCategory.IMAGE_BLURRED,
                severity=ErrorSeverity(error_info["severity"].value),
                message=error_info["message"],
                details=f"Blur score: {blur_score:.1f} (min required: {self.confidence_benchmark.blur_score_max})",
                suggestion=error_info["suggestion"],
                confidence_impact=error_info["confidence_impact"],
            ))

        # Check resolution
        min_res = self.confidence_benchmark.min_resolution
        if resolution[0] < min_res[0] or resolution[1] < min_res[1]:
            error_info = self.error_details[ErrorCategory.IMAGE_LOW_RESOLUTION]
            errors.append(ErrorDetail(
                category=ErrorCategory.IMAGE_LOW_RESOLUTION,
                severity=ErrorSeverity(error_info["severity"].value),
                message=error_info["message"],
                details=f"Resolution: {resolution[0]}x{resolution[1]} (min required: {min_res[0]}x{min_res[1]})",
                suggestion=error_info["suggestion"],
                confidence_impact=error_info["confidence_impact"],
            ))

        # Check brightness
        if brightness > 0.85:
            error_info = self.error_details[ErrorCategory.IMAGE_OVEREXPOSED]
            errors.append(ErrorDetail(
                category=ErrorCategory.IMAGE_OVEREXPOSED,
                severity=ErrorSeverity(error_info["severity"].value),
                message=error_info["message"],
                suggestion=error_info["suggestion"],
                confidence_impact=error_info["confidence_impact"],
            ))
        elif brightness < 0.15:
            error_info = self.error_details[ErrorCategory.IMAGE_UNDEREXPOSED]
            errors.append(ErrorDetail(
                category=ErrorCategory.IMAGE_UNDEREXPOSED,
                severity=ErrorSeverity(error_info["severity"].value),
                message=error_info["message"],
                suggestion=error_info["suggestion"],
                confidence_impact=error_info["confidence_impact"],
            ))

        # Check reflection
        if has_reflection:
            error_info = self.error_details[ErrorCategory.IMAGE_REFLECTION]
            errors.append(ErrorDetail(
                category=ErrorCategory.IMAGE_REFLECTION,
                severity=ErrorSeverity(error_info["severity"].value),
                message=error_info["message"],
                suggestion=error_info["suggestion"],
                confidence_impact=error_info["confidence_impact"],
            ))

        return errors

    def get_rejection_reasoning(
        self,
        errors: List[ErrorDetail],
        confidence: float,
        jewelry_type: Optional[JewelryType] = None
    ) -> Dict:
        """
        Generate detailed rejection reasoning.

        Returns structured reasoning for why item was not approved.
        """
        critical_errors = [e for e in errors if e.severity == ErrorSeverity.CRITICAL]
        major_errors = [e for e in errors if e.severity == ErrorSeverity.MAJOR]
        minor_errors = [e for e in errors if e.severity == ErrorSeverity.MINOR]

        # Determine primary reason
        if critical_errors:
            primary_reason = critical_errors[0].message
            can_be_fixed = False
        elif major_errors:
            primary_reason = major_errors[0].message
            can_be_fixed = True
        elif confidence < self.confidence_benchmark.auto_reject_threshold:
            primary_reason = "Overall confidence score too low"
            can_be_fixed = True
        else:
            primary_reason = "Unknown issue"
            can_be_fixed = True

        # Build reasoning
        reasoning = {
            "primary_reason": primary_reason,
            "can_be_corrected": can_be_fixed,
            "confidence_assessment": {
                "score": round(confidence, 3),
                "interpretation": self.confidence_benchmark.get_score_interpretation(confidence),
                "threshold_for_approval": self.confidence_benchmark.auto_approve_threshold,
                "gap_to_approval": round(max(0, self.confidence_benchmark.auto_approve_threshold - confidence), 3),
            },
            "errors_by_severity": {
                "critical": [
                    {
                        "category": e.category.value,
                        "message": e.message,
                        "suggestion": e.suggestion,
                    }
                    for e in critical_errors
                ],
                "major": [
                    {
                        "category": e.category.value,
                        "message": e.message,
                        "suggestion": e.suggestion,
                    }
                    for e in major_errors
                ],
                "minor": [
                    {
                        "category": e.category.value,
                        "message": e.message,
                        "suggestion": e.suggestion,
                    }
                    for e in minor_errors
                ],
            },
            "total_errors": len(errors),
            "suggestions": list(set(e.suggestion for e in errors if e.suggestion)),
        }

        # Add jewelry-specific notes
        if jewelry_type:
            ruleset = self.get_ruleset(jewelry_type)
            if ruleset:
                reasoning["jewelry_specific_notes"] = ruleset.special_rules

        return reasoning

    def get_all_rulesets_summary(self) -> List[Dict]:
        """Get summary of all jewelry rulesets for display."""
        summaries = []
        for jtype, ruleset in self.rulesets.items():
            summaries.append({
                "type": jtype.value,
                "display_name": ruleset.display_name,
                "description": ruleset.description,
                "acceptable_positions": [
                    {"name": p.name, "description": p.description, "preferred": p.is_preferred}
                    for p in ruleset.acceptable_positions
                ],
                "forbidden_positions": [
                    {"name": p.name, "description": p.description, "reason": p.notes}
                    for p in ruleset.forbidden_positions
                ],
                "special_rules": ruleset.special_rules,
                "common_issues": [e.value for e in ruleset.common_issues],
                "min_weight_grams": ruleset.min_weight_for_hallmark,
            })
        return summaries
