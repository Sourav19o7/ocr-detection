"""
QC Service for Jewelry Hallmarking OCR Validation.

This service integrates the OCR engine with the QC validation rules
to provide a complete quality control workflow for hallmark verification.

Supports:
1. Hallmarking stage integration (real-time validation after engraving)
2. Separate QC dashboard (batch validation with approval workflow)
"""

import time
from typing import Dict, Optional, List
from PIL import Image
from dataclasses import asdict
import json

# Import OCR engine
from ocr_model_v2 import OCREngineV2, HallmarkInfo, HallmarkType

# Import QC configuration
import sys
sys.path.insert(0, "..")
from config.qc_hallmark_config import (
    HallmarkQCValidator,
    QCResult,
    QCDecision,
    RejectionReason,
    BISComplianceRules,
    QCValidationRules,
    QCWorkflowConfig,
    HALLMARKING_STAGE_WORKFLOW,
    SEPARATE_QC_DASHBOARD_WORKFLOW,
)


class HallmarkQCService:
    """
    Main service class for Hallmark QC validation.

    Usage:
        # For hallmarking stage integration
        service = HallmarkQCService(workflow="hallmarking_stage")
        result = service.validate_image(job_id, image)

        # For QC dashboard
        service = HallmarkQCService(workflow="qc_dashboard")
        results = service.validate_batch(items)
    """

    def __init__(
        self,
        workflow: str = "hallmarking_stage",
        custom_bis_rules: Optional[BISComplianceRules] = None,
        custom_qc_rules: Optional[QCValidationRules] = None,
    ):
        """
        Initialize the QC service.

        Args:
            workflow: "hallmarking_stage" or "qc_dashboard"
            custom_bis_rules: Custom BIS compliance rules (optional)
            custom_qc_rules: Custom QC validation rules (optional)
        """
        # Select workflow configuration
        if workflow == "hallmarking_stage":
            self.workflow_config = HALLMARKING_STAGE_WORKFLOW
        else:
            self.workflow_config = SEPARATE_QC_DASHBOARD_WORKFLOW

        # Initialize validator
        self.validator = HallmarkQCValidator(
            bis_rules=custom_bis_rules or BISComplianceRules(),
            qc_rules=custom_qc_rules or QCValidationRules(),
            workflow_config=self.workflow_config,
        )

        # Initialize OCR engine
        self.ocr_engine = OCREngineV2(enable_preprocessing=True)

    def validate_image(
        self,
        job_id: str,
        image: Image.Image,
        expected_purity: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Validate a single hallmark image.

        This is the main entry point for both integration flows.

        Args:
            job_id: Unique job identifier from ERP/hallmarking machine
            image: PIL Image of the hallmarked jewelry
            expected_purity: Expected purity code for cross-validation (optional)
            metadata: Additional metadata (machine_id, operator_id, etc.)

        Returns:
            Complete validation result with decision, confidence, and details
        """
        start_time = time.time()

        # Step 1: Run OCR extraction
        hallmark_info = self.ocr_engine.extract_with_hallmark_info(image)

        # Step 2: Extract detection confidences
        purity_confidence = 0.0
        huid_confidence = 0.0
        bis_confidence = 0.0

        for result in hallmark_info.all_results:
            if result.hallmark_type == HallmarkType.PURITY_MARK and result.validated:
                purity_confidence = max(purity_confidence, result.confidence)
            elif result.hallmark_type == HallmarkType.HUID and result.validated:
                huid_confidence = max(huid_confidence, result.confidence)

        # BIS logo detection (check if we have both purity and HUID)
        bis_logo_detected = hallmark_info.bis_certified or (
            hallmark_info.purity_code is not None and
            hallmark_info.huid is not None
        )
        if bis_logo_detected:
            bis_confidence = 0.80  # Default confidence for implied BIS certification

        # Step 3: Calculate image quality score
        image_quality_score = self._assess_image_quality(image)

        # Step 4: Run QC validation
        qc_result = self.validator.validate_hallmark(
            job_id=job_id,
            purity_text=hallmark_info.purity_code,
            huid_text=hallmark_info.huid,
            bis_logo_detected=bis_logo_detected,
            ocr_confidence=hallmark_info.overall_confidence,
            purity_confidence=purity_confidence,
            huid_confidence=huid_confidence,
            bis_confidence=bis_confidence,
            image_quality_score=image_quality_score,
        )

        # Step 5: Cross-validate with expected purity if provided
        if expected_purity and hallmark_info.purity_code:
            if hallmark_info.purity_code != expected_purity:
                qc_result.rejection_reasons.append(RejectionReason.OCR_MISMATCH)
                qc_result.requires_manual_review = True
                if qc_result.decision == QCDecision.APPROVED:
                    qc_result.decision = QCDecision.MANUAL_REVIEW

        # Step 6: Record processing time
        qc_result.processing_time_ms = int((time.time() - start_time) * 1000)
        qc_result.raw_ocr_text = " ".join([r.text for r in hallmark_info.all_results])

        # Build response
        return self._build_response(qc_result, hallmark_info, metadata)

    def validate_batch(
        self,
        items: List[Dict],
        callback_url: Optional[str] = None
    ) -> Dict:
        """
        Validate a batch of hallmark images.

        Used by the QC dashboard for bulk processing.

        Args:
            items: List of {"job_id": str, "image": Image or "image_path": str}
            callback_url: Webhook URL for async notification

        Returns:
            Batch results with summary statistics
        """
        start_time = time.time()
        results = []
        summary = {
            "total": len(items),
            "approved": 0,
            "rejected": 0,
            "manual_review": 0,
            "errors": 0,
        }

        for item in items:
            try:
                job_id = item.get("job_id", f"BATCH-{len(results)}")

                # Load image
                if "image" in item:
                    image = item["image"]
                elif "image_path" in item:
                    image = Image.open(item["image_path"])
                else:
                    raise ValueError("No image provided")

                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Validate
                result = self.validate_image(
                    job_id=job_id,
                    image=image,
                    expected_purity=item.get("expected_purity"),
                    metadata=item.get("metadata"),
                )

                results.append(result)

                # Update summary
                decision = result.get("data", {}).get("decision", "error")
                if decision == "approved":
                    summary["approved"] += 1
                elif decision == "rejected":
                    summary["rejected"] += 1
                elif decision == "manual_review":
                    summary["manual_review"] += 1
                else:
                    summary["errors"] += 1

            except Exception as e:
                results.append({
                    "status": "error",
                    "job_id": item.get("job_id", "unknown"),
                    "error": str(e),
                })
                summary["errors"] += 1

        total_time = int((time.time() - start_time) * 1000)

        return {
            "status": "completed",
            "summary": summary,
            "results": results,
            "processing_time_ms": total_time,
            "callback_url": callback_url,
        }

    def apply_override(
        self,
        job_id: str,
        original_result: Dict,
        override_decision: str,
        override_reason: str,
        operator_id: str,
        notes: str = ""
    ) -> Dict:
        """
        Apply QC personnel override to a validation result.

        Used when QC person disagrees with AI decision.

        Args:
            job_id: Job identifier
            original_result: Original validation result
            override_decision: "approved" or "rejected"
            override_reason: Reason for override
            operator_id: QC operator identifier
            notes: Additional notes

        Returns:
            Updated result with override applied
        """
        # Convert override decision string to enum
        if override_decision == "approved":
            override_enum = QCDecision.APPROVED
        elif override_decision == "rejected":
            override_enum = QCDecision.REJECTED
        else:
            override_enum = QCDecision.MANUAL_REVIEW

        # Update the result
        result = original_result.copy()
        result["qc_override"] = {
            "decision": override_decision,
            "reason": override_reason,
            "operator_id": operator_id,
            "notes": notes,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # Update the final decision
        result["data"]["decision"] = override_decision
        result["data"]["qc_override"] = override_decision

        # Log for feedback loop
        if self.workflow_config.COLLECT_QC_FEEDBACK:
            self._store_feedback(job_id, original_result, result["qc_override"])

        return result

    def get_qc_statistics(self, time_range: str = "today") -> Dict:
        """
        Get QC statistics for dashboard display.

        Args:
            time_range: "today", "week", "month"

        Returns:
            Statistics summary
        """
        # This would typically query a database
        # For now, return placeholder structure
        return {
            "time_range": time_range,
            "total_processed": 0,
            "auto_approved": 0,
            "auto_rejected": 0,
            "manual_reviews": 0,
            "overrides": 0,
            "average_confidence": 0.0,
            "average_processing_time_ms": 0,
            "top_rejection_reasons": [],
        }

    def _build_response(
        self,
        qc_result: QCResult,
        hallmark_info: HallmarkInfo,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Build the API response structure."""
        return {
            "status": "success",
            "data": {
                "job_id": qc_result.job_id,
                "decision": qc_result.decision.value,
                "confidence": round(qc_result.confidence, 3),
                "bis_certified": qc_result.bis_certified,

                "hallmark_data": {
                    "purity_code": qc_result.purity_code,
                    "karat": qc_result.karat,
                    "purity_percentage": qc_result.purity_percentage,
                    "huid": qc_result.huid,
                },

                "validation_status": {
                    "purity_valid": qc_result.purity_valid,
                    "huid_valid": qc_result.huid_valid,
                    "bis_logo_detected": qc_result.bis_logo_detected,
                },

                "rejection_info": {
                    "reasons": [r.value for r in qc_result.rejection_reasons],
                    "message": qc_result.rejection_message,
                } if qc_result.rejection_reasons else None,

                "processing_details": {
                    "ocr_corrections_applied": qc_result.ocr_corrections_applied,
                    "raw_ocr_text": qc_result.raw_ocr_text,
                    "image_quality_score": round(qc_result.image_quality_score, 3),
                    "processing_time_ms": qc_result.processing_time_ms,
                },

                "requires_manual_review": qc_result.requires_manual_review,
                "qc_override": qc_result.qc_override.value if qc_result.qc_override else None,
            },
            "metadata": metadata,
        }

    def _assess_image_quality(self, image: Image.Image) -> float:
        """
        Assess image quality for OCR reliability.

        Returns score between 0 and 1.
        """
        import numpy as np

        # Convert to grayscale numpy array
        gray = np.array(image.convert("L"))

        # Calculate Laplacian variance (blur detection)
        # Higher variance = sharper image
        try:
            import cv2
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize to 0-1 range (100 is typical threshold)
            blur_score = min(laplacian_var / 200.0, 1.0)
        except ImportError:
            # Fallback without OpenCV
            blur_score = 0.7

        # Check resolution
        width, height = image.size
        min_dim = min(width, height)
        resolution_score = min(min_dim / 640.0, 1.0)

        # Combined score
        quality_score = (blur_score * 0.6) + (resolution_score * 0.4)

        return quality_score

    def _store_feedback(
        self,
        job_id: str,
        original_result: Dict,
        override_info: Dict
    ):
        """Store QC feedback for model improvement."""
        feedback_path = self.workflow_config.FEEDBACK_STORAGE_PATH

        try:
            import os
            os.makedirs(feedback_path, exist_ok=True)

            feedback_entry = {
                "job_id": job_id,
                "original_decision": original_result.get("data", {}).get("decision"),
                "override": override_info,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            # Append to feedback file
            feedback_file = os.path.join(feedback_path, "feedback_log.jsonl")
            with open(feedback_file, "a") as f:
                f.write(json.dumps(feedback_entry) + "\n")

        except Exception as e:
            print(f"Warning: Could not store feedback: {e}")


# Convenience functions for quick usage
def validate_hallmark_image(
    job_id: str,
    image_path: str,
    workflow: str = "hallmarking_stage"
) -> Dict:
    """
    Quick function to validate a hallmark image.

    Args:
        job_id: Job identifier
        image_path: Path to the image file
        workflow: "hallmarking_stage" or "qc_dashboard"

    Returns:
        Validation result
    """
    service = HallmarkQCService(workflow=workflow)
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return service.validate_image(job_id, image)


def validate_hallmark_batch(
    items: List[Dict],
    workflow: str = "qc_dashboard"
) -> Dict:
    """
    Quick function to validate a batch of images.

    Args:
        items: List of {"job_id": str, "image_path": str}
        workflow: Workflow mode

    Returns:
        Batch results
    """
    service = HallmarkQCService(workflow=workflow)
    return service.validate_batch(items)
