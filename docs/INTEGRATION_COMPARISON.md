# Integration Type Comparison: Quick Reference

## Overview

This document provides a quick comparison of the two integration approaches for AI-powered hallmark QC validation.

---

## Side-by-Side Comparison

| Aspect | Type 1: Hallmarking Stage | Type 2: Separate QC Dashboard |
|--------|---------------------------|-------------------------------|
| **When validation happens** | Immediately after engraving | Batch processing (periodic) |
| **Where results appear** | Existing ERP dashboard | Dedicated QC dashboard |
| **Processing mode** | Real-time, single item | Batch, multiple items |
| **QC workflow** | Review within ERP | Dedicated queue interface |
| **Best for** | High-volume, fast turnaround | Detailed review, analytics |
| **ERP changes required** | Yes (add QC widget) | Minimal (webhook only) |
| **Hardware needed** | Camera at each station | Centralized image storage |

---

## Integration Type 1: Hallmarking Stage

### Flow Diagram

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Hallmarking │───▶│   Camera     │───▶│  API Call    │───▶│  ERP Shows   │
│  Complete    │    │   Capture    │    │  /qc/validate│    │  Result      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                   │
                                                                   ▼
                                                            ┌──────────────┐
                                                            │  QC Person   │
                                                            │  Confirms    │
                                                            └──────────────┘
```

### Key Characteristics

- **Latency**: ~2 seconds per item
- **Integration Point**: POST /qc/validate with job_id
- **UI Location**: Widget in existing ERP QC section
- **Feedback**: Immediate in same workflow

### When to Use

✅ High-volume production lines
✅ Need immediate go/no-go decision
✅ QC person stationed at/near hallmarking machine
✅ Existing ERP can be extended easily

### Sample API Call

```bash
curl -X POST http://api/qc/validate \
  -F "file=@hallmark.jpg" \
  -F "job_id=HM-123456789" \
  -F "expected_purity=916"
```

---

## Integration Type 2: Separate QC Dashboard

### Flow Diagram

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Multiple    │───▶│   Image      │───▶│   Batch      │───▶│  QC Dashboard│
│  Stations    │    │   Queue      │    │   Process    │    │  Review      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                   │
                                                                   ▼
                                                            ┌──────────────┐
                                                            │  Sync to ERP │
                                                            │  via Webhook │
                                                            └──────────────┘
```

### Key Characteristics

- **Latency**: Batch intervals (e.g., every 5-15 minutes)
- **Integration Point**: Webhook callback to ERP
- **UI Location**: Standalone dashboard (Streamlit/Web)
- **Feedback**: Centralized queue management

### When to Use

✅ Multiple hallmarking stations
✅ Centralized QC team
✅ Need detailed analytics and reporting
✅ Want minimal ERP modifications
✅ Batch review is acceptable

### Sample Batch Request

```bash
curl -X POST http://api/qc/validate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "BATCH-001",
    "items": [
      {"job_id": "HM-001", "image_path": "/img1.jpg"},
      {"job_id": "HM-002", "image_path": "/img2.jpg"}
    ],
    "callback_url": "https://erp/webhook"
  }'
```

---

## Decision Matrix

| Your Situation | Recommended Integration |
|----------------|------------------------|
| Single hallmarking station | Type 1 |
| Multiple stations, central QC | Type 2 |
| Need instant feedback | Type 1 |
| Detailed review required | Type 2 |
| Minimal IT changes | Type 2 |
| High-speed production | Type 1 |
| Analytics & reporting focus | Type 2 |
| Existing ERP extensible | Type 1 |

---

## Hybrid Approach

You can also implement **both** integrations:

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HALLMARKING STATIONS                                       │
│  ───────────────────                                        │
│  • Real-time validation (Type 1)                           │
│  • Immediate go/no-go for operators                        │
│  • Auto-approved items proceed immediately                 │
│                                                             │
│                    │                                        │
│                    ▼                                        │
│                                                             │
│  QC DASHBOARD (for "Manual Review" items)                  │
│  ─────────────────────────────────────────                 │
│  • Items with 50-85% confidence routed here                │
│  • Detailed review interface (Type 2)                      │
│  • Batch processing for edge cases                         │
│  • Analytics and reporting                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Benefits of Hybrid**:
- Fast throughput for clear cases (≥85% confidence)
- Detailed review for uncertain cases (50-85%)
- Best of both worlds

---

## Implementation Effort

| Component | Type 1 | Type 2 | Hybrid |
|-----------|--------|--------|--------|
| API Development | ✅ Done | ✅ Done | ✅ Done |
| Camera Setup | Per station | Centralized | Both |
| ERP Integration | Widget + API | Webhook only | Both |
| QC Dashboard | Not needed | Required | Required |
| Training | Minimal | Moderate | Moderate |
| **Timeline** | 2-3 weeks | 3-4 weeks | 4-5 weeks |

---

## Quick Links

- [Full Integration Guide](./HALLMARK_QC_INTEGRATION_GUIDE.md)
- [API Documentation](http://localhost:8000/docs)
- [QC Dashboard](http://localhost:8501)
- [Jewelry Rulesets](../config/jewelry_rulesets.py)
- [BIS Configuration](../config/qc_hallmark_config.py)

---

## Contact

For implementation support or questions about integration options, refer to the full documentation or API specifications.
