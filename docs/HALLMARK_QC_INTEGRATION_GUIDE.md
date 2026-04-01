# Jewelry Hallmarking QC - AI-Powered OCR Integration Guide

## Executive Summary

This document outlines the comprehensive approach for integrating AI-powered OCR validation into the jewelry hallmarking quality control (QC) process. The system validates hallmark engravings against Bureau of Indian Standards (BIS) regulations, ensuring compliance and automating the QC workflow.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Current Process Analysis](#2-current-process-analysis)
3. [Proposed AI Integration Approach](#3-proposed-ai-integration-approach)
4. [Integration Type 1: Hallmarking Stage Integration](#4-integration-type-1-hallmarking-stage-integration)
5. [Integration Type 2: Separate QC Dashboard](#5-integration-type-2-separate-qc-dashboard)
6. [Technical Architecture](#6-technical-architecture)
7. [BIS Compliance Rules](#7-bis-compliance-rules)
8. [Error Categories & Handling](#8-error-categories--handling)
9. [Jewelry-Specific Rulesets](#9-jewelry-specific-rulesets)
10. [API Reference](#10-api-reference)
11. [Decision Logic & Confidence Scoring](#11-decision-logic--confidence-scoring)
12. [Feedback Loop & Model Improvement](#12-feedback-loop--model-improvement)
13. [Implementation Roadmap](#13-implementation-roadmap)

---

## 1. Overview

### 1.1 Purpose

The AI-powered OCR system automates the quality control process for jewelry hallmarking by:
- Reading and validating hallmark engravings (Purity Code, HUID, BIS Logo)
- Ensuring compliance with BIS standards
- Providing real-time feedback to QC personnel
- Reducing manual inspection time and human error

### 1.2 Key Components

| Component | Description |
|-----------|-------------|
| **OCR Engine** | PaddleOCR-based text extraction with metal surface preprocessing |
| **BIS Validator** | Rule engine for purity codes, HUID format, and compliance |
| **QC Service** | Decision logic for approve/reject/manual review |
| **API Layer** | FastAPI endpoints for ERP/dashboard integration |
| **QC Dashboard** | Streamlit-based UI for testing and QC operations |

### 1.3 Supported Standards

- **Gold**: IS 1417 (Purity codes: 375, 585, 750, 875, 916, 958, 999)
- **Silver**: IS 2112 (Purity codes: 800, 925, 950, 999)
- **HUID**: Mandatory 6-character alphanumeric code (since April 2023)

---

## 2. Current Process Analysis

### 2.1 Existing Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Hallmarking   │────▶│  HUID Engraving │────▶│   Manual QC     │
│     Stage       │     │    on Jewelry   │     │   Inspection    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  Approve/Reject │
                                                │   in ERP        │
                                                └─────────────────┘
```

### 2.2 Current Pain Points

| Issue | Impact |
|-------|--------|
| Manual inspection is time-consuming | Low throughput, delays |
| Human error in reading small engravings | Inconsistent quality |
| No standardized rejection documentation | Difficult to track issues |
| No feedback loop for engraving quality | Recurring defects |
| Subjective approval decisions | Inconsistent standards |

### 2.3 QC Personnel Responsibilities

1. Visually inspect hallmark engraving
2. Verify purity code matches job order
3. Confirm HUID is present and readable
4. Check BIS logo presence
5. Ensure marking position is correct
6. Document any rejections with reasons

---

## 3. Proposed AI Integration Approach

### 3.1 Core Philosophy

The AI system is designed to **assist, not replace** QC personnel. It provides:
- Automated first-pass validation
- Confidence-based decision recommendations
- Detailed error categorization
- Human override capability for edge cases

### 3.2 Decision Framework

```
                    ┌─────────────────────────────────────┐
                    │         Image Captured              │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │      OCR Processing & Validation    │
                    │  • Extract Purity Code              │
                    │  • Extract HUID                     │
                    │  • Detect BIS Logo                  │
                    │  • Assess Image Quality             │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │      Calculate Confidence Score     │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  ≥ 85%    │   │  50-85%   │   │   < 50%   │
            │  AUTO     │   │  MANUAL   │   │   AUTO    │
            │  APPROVE  │   │  REVIEW   │   │  REJECT   │
            └───────────┘   └───────────┘   └───────────┘
                    │               │               │
                    ▼               ▼               ▼
            ┌─────────────────────────────────────────────┐
            │           ERP Dashboard Update              │
            │    QC Personnel Cross-Check & Feedback      │
            └─────────────────────────────────────────────┘
```

### 3.3 Key Benefits

| Benefit | Description |
|---------|-------------|
| **Speed** | 80%+ of items auto-processed without manual review |
| **Consistency** | Same validation rules applied uniformly |
| **Traceability** | Every decision logged with reasoning |
| **Feedback Loop** | Human corrections improve model over time |
| **Compliance** | Automated BIS standard enforcement |

---

## 4. Integration Type 1: Hallmarking Stage Integration

### 4.1 Overview

In this integration, the AI validation happens **immediately after engraving** at the hallmarking machine. The result is displayed in the ERP dashboard for the QC person to cross-check.

### 4.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      HALLMARKING STATION                            │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐       │
│  │  Hallmarking  │───▶│    Camera     │───▶│  Edge Device  │       │
│  │   Machine     │    │   Capture     │    │  (Optional)   │       │
│  └───────────────┘    └───────────────┘    └───────────────┘       │
│                                                    │                │
└────────────────────────────────────────────────────┼────────────────┘
                                                     │
                                                     ▼
                    ┌─────────────────────────────────────────────────┐
                    │              API SERVER                         │
                    │  POST /qc/validate                              │
                    │  • Image + Job ID                               │
                    │  • Expected Purity (optional)                   │
                    └─────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                    ┌─────────────────────────────────────────────────┐
                    │           OCR + VALIDATION ENGINE               │
                    │  • PaddleOCR extraction                         │
                    │  • BIS compliance check                         │
                    │  • Confidence scoring                           │
                    └─────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                    ┌─────────────────────────────────────────────────┐
                    │              ERP DASHBOARD                      │
                    │  ┌─────────────────────────────────────┐       │
                    │  │  Job: HM-123456789                  │       │
                    │  │  Status: ✅ APPROVED (92% conf)     │       │
                    │  │  Purity: 916 (22K)  HUID: AB1234   │       │
                    │  │  [Override] [View Details]          │       │
                    │  └─────────────────────────────────────┘       │
                    │                                                 │
                    │  QC Person reviews and provides feedback        │
                    └─────────────────────────────────────────────────┘
```

### 4.3 Workflow Steps

| Step | Actor | Action | System Response |
|------|-------|--------|-----------------|
| 1 | Machine | Complete HUID engraving | Trigger camera capture |
| 2 | Camera | Capture hallmark image | Send to API with job_id |
| 3 | API | Process image | Extract text, validate, score |
| 4 | API | Return result | Decision + confidence + details |
| 5 | ERP | Display result | Show in QC section of dashboard |
| 6 | QC Person | Review result | Cross-check visually |
| 7 | QC Person | Confirm or Override | Update final status |
| 8 | System | Log feedback | Store for model improvement |

### 4.4 API Integration

**Endpoint**: `POST /qc/validate`

**Request**:
```http
POST /qc/validate HTTP/1.1
Content-Type: multipart/form-data

file: <image_binary>
job_id: HM-123456789
expected_purity: 916  (optional)
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "job_id": "HM-123456789",
    "decision": "approved",
    "confidence": 0.92,
    "bis_certified": true,
    "hallmark_data": {
      "purity_code": "916",
      "karat": "22K",
      "purity_percentage": 91.6,
      "huid": "AB1234"
    },
    "validation_status": {
      "purity_valid": true,
      "huid_valid": true,
      "bis_logo_detected": true
    },
    "rejection_info": null,
    "processing_details": {
      "ocr_corrections_applied": 0,
      "raw_ocr_text": "916 AB1234 BIS",
      "image_quality_score": 0.85,
      "processing_time_ms": 450
    },
    "requires_manual_review": false
  }
}
```

### 4.5 ERP Dashboard Integration

The ERP dashboard should display:

```
┌─────────────────────────────────────────────────────────────────┐
│  HALLMARKING QC - Job HM-123456789                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  AI VALIDATION RESULT                        │
│  │              │  ═══════════════════════════════════════     │
│  │   [IMAGE]    │                                               │
│  │              │  Decision:  ✅ APPROVED                       │
│  │              │  Confidence: ████████████░░ 92%              │
│  └──────────────┘                                               │
│                                                                 │
│  DETECTED VALUES                    VALIDATION STATUS           │
│  ─────────────────                  ──────────────────          │
│  Purity Code: 916                   ✅ Purity Valid            │
│  Karat: 22K                         ✅ HUID Valid              │
│  HUID: AB1234                       ✅ BIS Certified           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  QC ACTION                                               │   │
│  │  ○ Confirm AI Decision    ○ Override to REJECT          │   │
│  │                                                          │   │
│  │  Override Reason: [________________________]             │   │
│  │                                                          │   │
│  │  [SUBMIT]                                                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.6 Advantages

| Advantage | Description |
|-----------|-------------|
| Real-time feedback | Immediate validation after engraving |
| Integrated workflow | No separate system to check |
| Fast throughput | Auto-approved items move quickly |
| Minimal disruption | Fits into existing ERP workflow |

### 4.7 Considerations

| Consideration | Mitigation |
|---------------|------------|
| Camera placement at each station | Use standardized mounting brackets |
| Network connectivity required | Local edge processing fallback |
| ERP modification needed | Simple API integration module |

---

## 5. Integration Type 2: Separate QC Dashboard

### 5.1 Overview

In this integration, a **dedicated QC dashboard** is introduced alongside the existing ERP. Images are queued for review, and QC personnel use the specialized interface to approve or reject items.

### 5.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HALLMARKING STATIONS (Multiple)                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│  │Station 1│  │Station 2│  │Station 3│  │Station 4│               │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘               │
│       │            │            │            │                      │
└───────┼────────────┼────────────┼────────────┼──────────────────────┘
        │            │            │            │
        └────────────┴────────────┴────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                    IMAGE QUEUE / STORAGE                    │
        │  • Images uploaded from all stations                        │
        │  • Job metadata attached                                    │
        │  • Pending validation status                                │
        └─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                    BATCH PROCESSING                         │
        │  POST /qc/validate/batch                                    │
        │  • Process multiple images                                  │
        │  • Generate validation results                              │
        └─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                 DEDICATED QC DASHBOARD                      │
        │  ┌─────────────────────────────────────────────────────┐   │
        │  │  PENDING REVIEW (15)  │  APPROVED (142)  │ REJECTED (8)│ │
        │  └─────────────────────────────────────────────────────┘   │
        │                                                             │
        │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
        │  │ HM-001  │ │ HM-002  │ │ HM-003  │ │ HM-004  │          │
        │  │ 78% ⚠️  │ │ 92% ✅  │ │ 45% ❌  │ │ 88% ✅  │          │
        │  │[Review] │ │[Confirm]│ │[Details]│ │[Confirm]│          │
        │  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
        │                                                             │
        │  [BULK APPROVE SELECTED]  [EXPORT REPORT]                  │
        └─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                      ERP SYNC                               │
        │  • Final decisions synced back to ERP                       │
        │  • Status updates via webhook                               │
        └─────────────────────────────────────────────────────────────┘
```

### 5.3 Dashboard Features

#### 5.3.1 Queue View

```
┌─────────────────────────────────────────────────────────────────────┐
│  QC DASHBOARD - Hallmark Validation Queue                          │
├─────────────────────────────────────────────────────────────────────┤
│  Filter: [All ▼] [Today ▼] [Pending Review ▼]    Search: [______]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────┬────────────┬────────┬────────┬──────────┬────────┬───────┐│
│  │ ☐  │ Job ID     │ Purity │ HUID   │ Confidence│ Status │Action ││
│  ├────┼────────────┼────────┼────────┼──────────┼────────┼───────┤│
│  │ ☐  │ HM-001234  │ 916    │ AB1234 │ 92%  🟢  │ Auto ✅│[View] ││
│  │ ☐  │ HM-001235  │ 750    │ CD5678 │ 78%  🟡  │ Review │[Open] ││
│  │ ☐  │ HM-001236  │ ---    │ ---    │ 42%  🔴  │ Auto ❌│[View] ││
│  │ ☐  │ HM-001237  │ 916    │ EF9012 │ 88%  🟢  │ Auto ✅│[View] ││
│  │ ☐  │ HM-001238  │ 585    │ GH3456 │ 65%  🟡  │ Review │[Open] ││
│  └────┴────────────┴────────┴────────┴──────────┴────────┴───────┘│
│                                                                     │
│  Selected: 2  [Bulk Approve] [Bulk Reject] [Export CSV]            │
└─────────────────────────────────────────────────────────────────────┘
```

#### 5.3.2 Detail View

```
┌─────────────────────────────────────────────────────────────────────┐
│  JOB DETAIL - HM-001235                              [← Back]       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐    ┌─────────────────────────────────────┐│
│  │                     │    │  AI ANALYSIS                        ││
│  │                     │    │  ═══════════════════════════════    ││
│  │    [HALLMARK        │    │                                     ││
│  │     IMAGE]          │    │  Decision: ⚠️ MANUAL REVIEW         ││
│  │                     │    │  Confidence: ███████░░░░ 78%        ││
│  │                     │    │                                     ││
│  └─────────────────────┘    │  DETECTED VALUES                    ││
│                             │  ─────────────────                   ││
│  ┌─────────────────────┐    │  Purity: 750 (18K) ✅               ││
│  │  PROCESSED IMAGE    │    │  HUID: CD5678 ✅                    ││
│  │  (Enhanced)         │    │  BIS Logo: Detected ✅              ││
│  │                     │    │                                     ││
│  └─────────────────────┘    │  ⚠️ REVIEW REASON                   ││
│                             │  Confidence between 50-85%          ││
│                             │  Image has minor reflection         ││
│                             └─────────────────────────────────────┘│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  QC DECISION                                                    ││
│  │                                                                 ││
│  │  ○ APPROVE - Hallmark is correct                               ││
│  │  ○ REJECT - Select reason:                                     ││
│  │      ☐ Purity code incorrect                                   ││
│  │      ☐ HUID not readable                                       ││
│  │      ☐ Wrong marking position                                  ││
│  │      ☐ Poor engraving quality                                  ││
│  │      ☐ Other: [_______________]                                ││
│  │                                                                 ││
│  │  Notes: [_________________________________________________]    ││
│  │                                                                 ││
│  │  [SUBMIT DECISION]                                             ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Workflow Steps

| Step | Actor | Action | System Response |
|------|-------|--------|-----------------|
| 1 | Stations | Upload images to queue | Store with job metadata |
| 2 | System | Batch process images | Generate AI validation results |
| 3 | Dashboard | Display queue | Show items sorted by status |
| 4 | QC Person | Open pending item | Show detail view |
| 5 | QC Person | Make decision | Approve/Reject with reason |
| 6 | System | Update status | Sync to ERP via webhook |
| 7 | System | Log decision | Store for analytics & training |

### 5.5 Batch Processing API

**Endpoint**: `POST /qc/validate/batch`

**Request**:
```json
{
  "batch_id": "BATCH-2024011501",
  "items": [
    {"job_id": "HM-001234", "image_path": "/uploads/img1.jpg"},
    {"job_id": "HM-001235", "image_path": "/uploads/img2.jpg"},
    {"job_id": "HM-001236", "image_path": "/uploads/img3.jpg"}
  ],
  "callback_url": "https://erp.example.com/webhook/qc"
}
```

**Response**:
```json
{
  "batch_id": "BATCH-2024011501",
  "status": "completed",
  "summary": {
    "total": 50,
    "approved": 42,
    "rejected": 5,
    "manual_review": 3
  },
  "processing_time_ms": 15000
}
```

### 5.6 Advantages

| Advantage | Description |
|-----------|-------------|
| Centralized QC | All items reviewed in one place |
| Batch processing | Efficient handling of large volumes |
| Specialized interface | Optimized for QC workflow |
| Detailed analytics | Track trends and issues |
| Flexible scheduling | Process during off-peak hours |

### 5.7 Considerations

| Consideration | Mitigation |
|---------------|------------|
| Additional system to maintain | Cloud-hosted option available |
| Not real-time at station | Batch intervals can be frequent |
| Training required | Intuitive UI minimizes learning curve |

---

## 6. Technical Architecture

### 6.1 System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      PRESENTATION LAYER                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│  │  │ ERP Module  │  │ QC Dashboard│  │ Mobile App  │         │   │
│  │  │ (Existing)  │  │ (Streamlit) │  │ (Future)    │         │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                         API LAYER                            │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │                    FastAPI Server                    │    │   │
│  │  │  /qc/validate    /qc/validate/v2   /qc/override     │    │   │
│  │  │  /qc/rules       /validate/huid    /extract/v2      │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      SERVICE LAYER                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│  │  │ QC Service  │  │ OCR Engine  │  │ Validator   │         │   │
│  │  │             │  │ (V2)        │  │ (BIS Rules) │         │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     PROCESSING LAYER                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│  │  │ PaddleOCR   │  │ Image       │  │ Confidence  │         │   │
│  │  │ Engine      │  │ Preprocessor│  │ Scorer      │         │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                       DATA LAYER                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│  │  │ Config      │  │ Feedback    │  │ History     │         │   │
│  │  │ (Rulesets)  │  │ Storage     │  │ (JSON/DB)   │         │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Image Preprocessing Pipeline

```
INPUT IMAGE
     │
     ▼
┌─────────────────┐
│ Resize & Aspect │  Target: 1280x1280
│ Ratio Preserve  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reflection      │  LAB color space
│ Removal         │  + Inpainting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bilateral       │  Noise reduction
│ Filtering       │  while preserving edges
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CLAHE           │  Contrast Limited
│ Enhancement     │  Adaptive Histogram
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Unsharp         │  Enhance engraved
│ Masking         │  text edges
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Adaptive        │  For uneven
│ Binarization    │  illumination
└────────┬────────┘
         │
         ▼
PROCESSED IMAGE → OCR
```

### 6.3 File Structure

```
ocr-detection/
├── api.py                          # FastAPI REST API
├── run.py                          # Streamlit entry point
├── requirements.txt                # Dependencies
│
├── src/
│   ├── ocr_model.py               # V1: Basic OCR
│   ├── ocr_model_v2.py            # V2: Hallmark detection
│   ├── dashboard.py               # Original Streamlit UI
│   ├── qc_dashboard.py            # QC-specific dashboard
│   ├── qc_service.py              # QC validation service
│   └── history.py                 # History management
│
├── config/
│   ├── qc_hallmark_config.py      # BIS rules & QC config
│   └── jewelry_rulesets.py        # Jewelry-specific rules
│
├── prompts/
│   └── hallmark_qc_prompt.md      # AI training prompt
│
└── docs/
    └── HALLMARK_QC_INTEGRATION_GUIDE.md  # This document
```

---

## 7. BIS Compliance Rules

### 7.1 Gold Purity Grades (IS 1417)

| Code | Karat | Purity % | Min Fineness | Tolerance |
|------|-------|----------|--------------|-----------|
| 375 | 9K | 37.5% | 375 | ±3 |
| 585 | 14K | 58.5% | 583 | ±3 |
| 750 | 18K | 75.0% | 750 | ±3 |
| 875 | 21K | 87.5% | 875 | ±3 |
| 916 | 22K | 91.6% | 916 | ±3 |
| 958 | 23K | 95.8% | 958 | ±3 |
| 999 | 24K | 99.9% | 990 | ±1 |

### 7.2 Silver Purity Grades (IS 2112)

| Code | Purity % | Grade | Tolerance |
|------|----------|-------|-----------|
| 800 | 80.0% | Silver 800 | ±3 |
| 835 | 83.5% | Silver 835 | ±3 |
| 900 | 90.0% | Silver 900 | ±3 |
| 925 | 92.5% | Sterling Silver | ±3 |
| 950 | 95.0% | Britannia Silver | ±3 |
| 999 | 99.9% | Fine Silver | ±1 |

### 7.3 HUID Requirements

| Requirement | Specification |
|-------------|---------------|
| Length | Exactly 6 characters |
| Characters | Alphanumeric (A-Z, 0-9) |
| Constraint | Must contain at least one letter |
| Mandatory Since | April 2023 |
| Purpose | Unique identification & traceability |

### 7.4 Hallmark Components

A valid BIS hallmark must contain:

1. **BIS Logo** - Triangle mark indicating BIS certification
2. **Purity/Fineness** - 3-digit code (e.g., 916)
3. **HUID** - 6-character unique identifier

---

## 8. Error Categories & Handling

### 8.1 Error Category Reference

#### Image Quality Errors

| Code | Description | Severity | Suggestion |
|------|-------------|----------|------------|
| `image_blurred` | Image is blurred or out of focus | MAJOR | Retake with steady camera |
| `image_overexposed` | Image too bright | MAJOR | Reduce lighting |
| `image_underexposed` | Image too dark | MAJOR | Increase lighting |
| `image_reflection` | Specular reflection on metal | MINOR | Use diffused lighting |
| `image_low_resolution` | Resolution too low | MAJOR | Higher resolution camera |

#### OCR/Detection Errors

| Code | Description | Severity | Suggestion |
|------|-------------|----------|------------|
| `low_confidence` | OCR confidence below threshold | MAJOR | Retake with better conditions |
| `partial_detection` | Only partial hallmark detected | MAJOR | Ensure full hallmark visible |
| `ocr_misread` | Characters incorrectly read | MAJOR | Verify manually |
| `no_detection` | No text detected | CRITICAL | Check if hallmark exists |

#### Hallmark Content Errors

| Code | Description | Severity | Suggestion |
|------|-------------|----------|------------|
| `invalid_purity_code` | Code not in BIS standards | CRITICAL | Valid: 375, 585, 750, etc. |
| `invalid_huid_format` | HUID format incorrect | CRITICAL | Must be 6 alphanumeric |
| `missing_huid` | HUID not present | CRITICAL | HUID mandatory since April 2023 |
| `missing_purity_mark` | Purity code not found | CRITICAL | Must be visible |
| `missing_bis_logo` | BIS logo not detected | MAJOR | Check marking |

#### Position/Placement Errors

| Code | Description | Severity | Suggestion |
|------|-------------|----------|------------|
| `marking_at_6_oclock` | Ring marked at bottom | CRITICAL | Use 12, 3, or 9 o'clock |
| `wrong_position` | Wrong area for jewelry type | MAJOR | Refer to jewelry rules |
| `marking_on_decorative` | Marking on design area | MAJOR | Use non-decorative surface |

### 8.2 Severity Levels

| Severity | Action | Description |
|----------|--------|-------------|
| **CRITICAL** | Auto-Reject | Cannot proceed, fundamental issue |
| **MAJOR** | Manual Review | Needs correction or verification |
| **MINOR** | Warning | Can be approved with note |
| **INFO** | Informational | No action required |

---

## 9. Jewelry-Specific Rulesets

### 9.1 Ring Rules

| Aspect | Rule |
|--------|------|
| **Acceptable Positions** | 12 o'clock (preferred), 3 o'clock, 9 o'clock |
| **Forbidden Positions** | 6 o'clock (bottom), outer surface |
| **Marking Size** | 0.8mm - 2.0mm |
| **Special Rules** | Must be inside band, avoid stone settings |

**Why 6 o'clock is forbidden**: The bottom inside of a ring contacts surfaces when placed down (tables, counters), causing the marking to wear off quickly. This position also gets the most friction during daily wear.

### 9.2 Bangle Rules

| Aspect | Rule |
|--------|------|
| **Acceptable Positions** | Inside near opening (preferred), inside flat area |
| **Forbidden Positions** | Outside decorative surface |
| **Marking Size** | 1.0mm - 3.0mm |
| **Special Rules** | Should not be on carved/embossed areas |

### 9.3 Chain Rules

| Aspect | Rule |
|--------|------|
| **Acceptable Positions** | Clasp area (preferred), attached tag, end link |
| **Forbidden Positions** | Middle chain links |
| **Marking Size** | 0.5mm - 2.0mm |
| **Special Rules** | Must not weaken any link, tag for fine chains |

### 9.4 Earring Rules

| Aspect | Rule |
|--------|------|
| **Acceptable Positions** | Back plate, earring back/stopper, hook base |
| **Forbidden Positions** | Visible front, near stones |
| **Marking Size** | 0.5mm - 1.5mm |
| **Special Rules** | Both in pair should be marked |

### 9.5 Pendant Rules

| Aspect | Rule |
|--------|------|
| **Acceptable Positions** | Back surface (preferred), bail area, inside locket |
| **Forbidden Positions** | Front decorative surface |
| **Marking Size** | 0.6mm - 2.0mm |
| **Special Rules** | Avoid stone settings |

### 9.6 Mangalsutra Rules

| Aspect | Rule |
|--------|------|
| **Acceptable Positions** | Pendant back (preferred), clasp area, vati back |
| **Forbidden Positions** | Black bead sections, front of pendant |
| **Marking Size** | 0.8mm - 2.0mm |
| **Special Rules** | Each gold component should be marked |

---

## 10. API Reference

### 10.1 Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Service health status |
| `/extract` | POST | V1 standard OCR |
| `/extract/v2` | POST | V2 hallmark OCR |
| `/qc/validate` | POST | QC validation with job_id |
| `/qc/validate/v2` | POST | Enhanced QC validation |
| `/qc/override` | POST | Apply QC override |
| `/qc/rules` | GET | Get BIS compliance rules |
| `/validate/huid` | POST | Validate HUID format |

### 10.2 Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 500 | Server Error |
| 503 | Service Unavailable |

### 10.3 Decision Values

| Decision | Description |
|----------|-------------|
| `approved` | Passed all validations, confidence ≥ 85% |
| `rejected` | Failed critical validation or confidence < 50% |
| `manual_review` | Needs human verification (50-85% confidence) |

---

## 11. Decision Logic & Confidence Scoring

### 11.1 Confidence Thresholds

```
100% ┬────────────────────────────┐
     │        EXCELLENT           │
 95% ├────────────────────────────┤
     │          GOOD              │  ← AUTO-APPROVE
 85% ├════════════════════════════╡     (≥ 85%)
     │        ACCEPTABLE          │
 70% ├────────────────────────────┤  ← MANUAL REVIEW
     │          POOR              │     (50-85%)
 50% ├════════════════════════════╡
     │       UNACCEPTABLE         │  ← AUTO-REJECT
  0% └────────────────────────────┘     (< 50%)
```

### 11.2 Decision Algorithm

```python
def make_decision(validation_result):
    # Critical errors always reject
    if has_critical_errors(validation_result):
        return "rejected"

    confidence = validation_result.confidence

    # Check if all components valid
    all_valid = (
        validation_result.purity_valid and
        validation_result.huid_valid and
        validation_result.bis_logo_detected
    )

    if all_valid:
        if confidence >= 0.85:
            return "approved"
        elif confidence >= 0.50:
            return "manual_review"
        else:
            return "rejected"
    else:
        # Missing components
        if confidence < 0.50:
            return "rejected"
        else:
            return "manual_review"
```

### 11.3 Component-Specific Thresholds

| Component | Minimum Confidence |
|-----------|-------------------|
| Purity Mark | 80% |
| HUID | 80% |
| BIS Logo | 70% |

---

## 12. Feedback Loop & Model Improvement

### 12.1 Override Tracking

When QC personnel override an AI decision:

```json
{
  "job_id": "HM-123456789",
  "original_decision": "rejected",
  "qc_override": "approved",
  "override_reason": "Engraving clear upon manual inspection",
  "operator_id": "QC-001",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 12.2 Feedback Storage

All overrides are stored for:
- Model retraining (false positives/negatives)
- Pattern analysis (common OCR errors)
- QC performance metrics
- Continuous improvement

### 12.3 Improvement Cycle

```
┌─────────────────┐
│  AI Validation  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  QC Review      │
│  & Override     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feedback       │
│  Collection     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Pattern        │
│  Analysis       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model          │
│  Retraining     │
└────────┬────────┘
         │
         └──────────▶ Back to AI Validation
```

---

## 13. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

- [x] OCR engine with hallmark preprocessing
- [x] BIS validation rule engine
- [x] Basic API endpoints
- [x] Confidence scoring logic

### Phase 2: Integration Type 1 (Weeks 3-4)

- [ ] Camera integration at hallmarking stations
- [ ] ERP API module development
- [ ] Dashboard widget for QC display
- [ ] Real-time validation flow testing

### Phase 3: Integration Type 2 (Weeks 5-6)

- [x] QC Dashboard development
- [ ] Batch processing implementation
- [ ] Queue management system
- [ ] ERP webhook integration

### Phase 4: Enhancement (Weeks 7-8)

- [ ] Jewelry-specific ruleset UI
- [ ] Analytics dashboard
- [ ] Mobile app prototype
- [ ] Performance optimization

### Phase 5: Production (Weeks 9-10)

- [ ] Production deployment
- [ ] User training
- [ ] Monitoring setup
- [ ] Documentation finalization

---

## Appendix A: Quick Start Guide

### Starting the Services

```bash
# Start API Server (Port 8000)
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# Start QC Dashboard (Port 8501)
streamlit run src/qc_dashboard.py --server.port 8501
```

### Testing Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Get BIS rules
curl http://localhost:8000/qc/rules

# Validate HUID
curl -X POST http://localhost:8000/validate/huid -F "huid=AB1234"

# Validate hallmark image
curl -X POST http://localhost:8000/qc/validate/v2 -F "file=@hallmark.jpg"
```

### Dashboard Access

- **QC Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **BIS** | Bureau of Indian Standards |
| **HUID** | Hallmark Unique Identification |
| **AHC** | Assaying & Hallmarking Centre |
| **IS 1417** | Indian Standard for Gold |
| **IS 2112** | Indian Standard for Silver |
| **Karat** | Measure of gold purity (24K = pure) |
| **Fineness** | Parts per thousand of precious metal |
| **OCR** | Optical Character Recognition |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-01-15 | AI System | Initial documentation |

---

*This document is maintained as part of the Jewelry Hallmarking QC OCR system.*
