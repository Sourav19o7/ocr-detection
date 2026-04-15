# ERP Integration Guide

This document describes how to integrate the Hallmark OCR API with your existing ERP system.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   ERP System    │────>│   AWS S3 Bucket  │────>│  OCR API (EC2)      │
│                 │     │                  │     │  Public IP:8000     │
└────────┬────────┘     └──────────────────┘     └─────────┬───────────┘
         │                                                  │
         │              ┌──────────────────┐                │
         └──────────────│  Callback to ERP │<───────────────┘
                        │  (JSON payload)  │
                        └──────────────────┘
```

## Integration Options

### Option 1: Direct Upload (Recommended for Simplicity)

Send the image directly to the OCR API. The API handles everything.

**Endpoint:** `POST /api/erp/upload-and-process`

```bash
curl -X POST "http://YOUR_EC2_IP:8000/api/erp/upload-and-process" \
  -F "file=@/path/to/image.jpg" \
  -F "tag_id=TAG001" \
  -F "expected_huid=AB1234" \
  -F "callback_url=https://your-erp.com/webhook/ocr-result"
```

**Response:**
```json
{
  "status": "success",
  "tag_id": "TAG001",
  "expected_huid": "AB1234",
  "actual_huid": "AB1234",
  "huid_match": true,
  "confidence": 0.923,
  "decision": "approved",
  "purity_code": "916",
  "karat": "22K",
  "purity_percentage": 91.6,
  "rejection_reasons": [],
  "image_url": "https://s3.amazonaws.com/...",
  "processing_time_ms": 2150
}
```

### Option 2: S3 Upload with API Trigger

1. ERP uploads image to S3
2. ERP calls API to process

**Step 1: Register Item**
```bash
curl -X POST "http://YOUR_EC2_IP:8000/api/erp/register-item" \
  -H "Content-Type: application/json" \
  -d '{
    "tag_id": "TAG001",
    "expected_huid": "AB1234",
    "s3_key": "erp-images/TAG001_AB1234.jpg",
    "callback_url": "https://your-erp.com/webhook/ocr-result"
  }'
```

**Step 2: Upload to S3** (from your ERP system)

**Step 3: Trigger Processing**
```bash
curl -X POST "http://YOUR_EC2_IP:8000/api/erp/process-image" \
  -F "tag_id=TAG001"
```

### Option 3: S3 Event Webhook (Automatic)

Configure S3 to send event notifications when images are uploaded.

**S3 Key Format:** `erp-images/{TAG_ID}_{EXPECTED_HUID}.jpg`
Example: `erp-images/TAG001_AB1234.jpg`

**Webhook Endpoint:** `POST /api/erp/s3-webhook`

Configure in AWS S3:
1. Go to S3 Bucket → Properties → Event notifications
2. Create event: `s3:ObjectCreated:*`
3. Destination: HTTP endpoint → `http://YOUR_EC2_IP:8000/api/erp/s3-webhook`
4. Prefix filter: `erp-images/`

## Callback Payload

When processing completes, the API sends this JSON to your callback URL:

```json
{
  "tag_id": "TAG001",
  "expected_huid": "AB1234",
  "actual_huid": "AB1234",
  "huid_match": true,
  "confidence": 0.923,
  "decision": "approved",
  "purity_code": "916",
  "karat": "22K",
  "rejection_reasons": [],
  "image_url": "https://...",
  "processed_at": "2024-01-15T10:30:00Z"
}
```

## Decision Values

| Decision | Description |
|----------|-------------|
| `approved` | Confidence >= 85%, HUID matches, purity detected |
| `rejected` | HUID mismatch or confidence < 50% |
| `manual_review` | Confidence 50-85% or missing data |
| `pending` | Not yet processed |

## Rejection Reasons

| Reason | Description |
|--------|-------------|
| `missing_huid` | HUID not detected in image |
| `missing_purity_mark` | Purity code (916, 750, etc.) not found |
| `huid_mismatch` | Detected HUID doesn't match expected |
| `low_confidence` | OCR confidence below threshold |

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/erp/register-item` | POST | Register an item before S3 upload |
| `/api/erp/process-image` | POST | Process an image from S3 |
| `/api/erp/upload-and-process` | POST | Upload and process in one call |
| `/api/erp/s3-webhook` | POST | S3 event notification handler |
| `/api/erp/pending-items` | GET | Get items needing review |
| `/api/erp/manual-decision` | POST | Set manual decision |
| `/api/erp/statistics` | GET | Get processing statistics |
| `/stage3/result/{tag_id}` | GET | Get result by tag ID |

## EC2 Deployment

Your API is deployed at: `http://YOUR_EC2_PUBLIC_IP:8000`

### Required Security Group Rules

| Port | Protocol | Source | Description |
|------|----------|--------|-------------|
| 8000 | TCP | Your ERP IP or 0.0.0.0/0 | API access |
| 8501 | TCP | Your IP | Dashboard access |
| 22 | TCP | Your IP | SSH access |

### Environment Variables

Set these in `/opt/hallmark-ocr/.env`:

```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=ap-south-1
S3_BUCKET_NAME=your-bucket
API_BASE_URL=http://YOUR_EC2_IP:8000
```

## Testing

### Health Check
```bash
curl http://YOUR_EC2_IP:8000/health
```

### Process Single Image
```bash
curl -X POST "http://YOUR_EC2_IP:8000/api/erp/upload-and-process" \
  -F "file=@test_image.jpg" \
  -F "tag_id=TEST001" \
  -F "expected_huid=XY9876"
```

### Get Statistics
```bash
curl "http://YOUR_EC2_IP:8000/api/erp/statistics"
```

## Dashboard Access

Access the QC Dashboard at: `http://YOUR_EC2_IP:8501`

The dashboard includes:
- **Stage 1:** Batch upload (CSV/Excel)
- **Stage 2:** Image processing
- **Stage 3:** Results view
- **ERP Monitor:** Real-time monitoring and manual review queue

## Error Handling

The API returns standard HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid data) |
| 404 | Tag ID not found |
| 500 | Server error |

Always check the `status` field in responses:
- `"status": "success"` - Operation completed
- `"status": "error"` - Check `detail` field for error message

## Support

For issues, check:
1. API logs: `sudo journalctl -u hallmark-api -f`
2. Dashboard logs: `sudo journalctl -u hallmark-dashboard -f`
3. API docs: `http://YOUR_EC2_IP:8000/docs`
