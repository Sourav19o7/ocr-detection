# Hallmark QC System - Implementation Roadmap

## Document Version: 1.0 | Date: 2026-04-22

---

## Executive Summary

This document outlines the implementation plan for the complete Hallmark QC System covering 5 modules:
1. Article Photo Capture
2. HUID Photo Capture
3. OCR Verification
4. Manakonline Portal Upload (Semi-automated)
5. Dashboard & User Management

**Total Estimated Timeline: 5-6 weeks** (with AI-assisted development)

---

## Current System State

### What's Already Working
- OCR engine with PaddleOCR (hallmark-specific)
- HUID extraction and purity/karat detection
- Batch upload via Excel/CSV
- Manual image upload with OCR processing
- S3 cloud storage integration
- Basic dashboard UI
- Accept/Reject workflow

### What's Missing
- External APIs for ERP/camera integration
- Manakonline Excel comparison
- Rework/Upload queue management
- Portal upload automation
- User authentication & roles
- Global dashboard metrics

---

## Module 1 & 2: External APIs for Image Capture

### Architecture Overview

Since ERP team will NOT implement POST APIs, we provide two options:

**Option A: Direct S3 Upload (Recommended)**
```
┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Camera Software  │────▶│ Our API         │────▶│ S3 Bucket    │
│ (BAC Team)       │     │ (Presigned URL) │     │              │
└──────────────────┘     └─────────────────┘     └──────────────┘
        │                         │
        │                         ▼
        │                 ┌─────────────────┐
        └────────────────▶│ Webhook/Poll    │
                          │ (Trigger OCR)   │
                          └─────────────────┘
```

**Option B: Multipart Upload**
```
┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Camera Software  │────▶│ Our API         │────▶│ S3 Bucket    │
│ (BAC Team)       │     │ (multipart)     │     │              │
└──────────────────┘     └─────────────────┘     └──────────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │ OCR Processing  │
                          │ (automatic)     │
                          └─────────────────┘
```

### API Specifications

See: [API_SPECIFICATION_FOR_BAC.md](./API_SPECIFICATION_FOR_BAC.md)

---

## Module 3: OCR Verification Enhancement

### New Features Required
1. Parse manakonline Excel (BIS Job No from filename)
2. Compare OCR HUID vs Excel HUID
3. Compare OCR Karatage vs Declared Purity
4. Rework queue for mismatches
5. Upload queue for matches

### Data Flow
```
Manakonline Excel ──▶ Parse ──▶ Create Comparison Batch
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │ For each Tag ID:    │
                              │ - Get OCR result    │
                              │ - Compare HUID      │
                              │ - Compare Purity    │
                              └─────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              ┌──────────┐       ┌──────────┐       ┌──────────┐
              │ MATCH    │       │ MISMATCH │       │ MISSING  │
              │ → Upload │       │ → Rework │       │ → Alert  │
              │   Queue  │       │   Queue  │       │          │
              └──────────┘       └──────────┘       └──────────┘
```

---

## Module 4: Portal Upload Automation

### Semi-Automated Browser Extension

See: [PORTAL_AUTOMATION_OVERVIEW.md](./PORTAL_AUTOMATION_OVERVIEW.md)

---

## Module 5: Dashboard & Authentication

### User Roles
| Role | Permissions |
|------|-------------|
| Operator | View own queue, upload images, accept/reject |
| Supervisor | View all items, override decisions, view reports |
| Admin | Full access, user management, all exports |

### Dashboard Metrics
- Total items processed
- Pass count (HUID match + approved)
- Refer count (mismatch + manual review)
- Upload status (pending/uploaded/failed)
- Date range and batch filters

---

## Implementation Phases

### Phase 1: External APIs (Week 1)
- Article photo upload API
- HUID photo upload API
- Auto-OCR on HUID upload
- File naming convention change
- API documentation for BAC team

### Phase 2: Excel Comparison (Week 2)
- Manakonline Excel parser
- Comparison logic
- Rework queue
- Upload queue
- Status tracking

### Phase 3: Dashboard & Auth (Week 3)
- User authentication (simple role-based)
- Global metrics
- Filters (date, batch)
- Export functionality

### Phase 4: Portal Automation (Week 4-5)
- Browser extension skeleton
- Session detection
- Upload automation
- Error handling
- Status sync

### Phase 5: Testing & Deployment (Week 6)
- End-to-end testing
- Bug fixes
- Documentation
- Operator training

---

## Roadblocks & Dependencies

### Critical Roadblocks

| # | Roadblock | Impact | Owner | Status |
|---|-----------|--------|-------|--------|
| 1 | ERP APIs not working | Cannot fetch tag data, HUID assignments | BAC Team | BLOCKED |
| 2 | BIS export format unknown | Cannot implement export | BAC Team | WAITING |
| 3 | No POST APIs from ERP | Must use presigned URL approach | BAC Team | WORKAROUND |

### Dependencies on BAC Team

| # | Dependency | Required By | Priority |
|---|------------|-------------|----------|
| 1 | Working ERP APIs (Receipt, GRN, HUID_API) | Phase 1 | HIGH |
| 2 | Sample API responses | Phase 1 | HIGH |
| 3 | Camera software integration | Phase 1 | HIGH |
| 4 | BIS export template | Phase 3 | MEDIUM |
| 5 | Manakonline portal test credentials | Phase 4 | HIGH |

### Dependencies on Us

| # | Deliverable | For BAC Team | Timeline |
|---|-------------|--------------|----------|
| 1 | API specification document | Camera integration | Day 1 |
| 2 | S3 bucket access credentials | Image uploads | Day 1 |
| 3 | Webhook endpoint (optional) | Upload notifications | Week 1 |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| ERP APIs remain broken | Medium | High | Escalate immediately |
| Portal UI changes | Medium | High | Design for resilience |
| CAPTCHA blocks automation | Low | Medium | Manual login fallback |
| OCR accuracy issues | Low | Medium | Manual review queue |
| S3 quota limits | Low | Low | Monitor usage |

---

## Questions for BAC Team Meeting

See: [MEETING_AGENDA_BAC.md](./MEETING_AGENDA_BAC.md)

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | External APIs | APIs ready, documentation sent to BAC |
| 2 | Excel Comparison | Manakonline parsing, queue management |
| 3 | Dashboard & Auth | User login, metrics, filters |
| 4-5 | Portal Automation | Browser extension for uploads |
| 6 | Testing | E2E testing, bug fixes, training |

**Go-Live Target: 6 weeks from start**

---

## Next Steps

1. Send API specification to BAC team
2. Schedule meeting to discuss roadblocks
3. Get ERP API access fixed
4. Begin Phase 1 development
