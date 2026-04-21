# API Specification for BAC Team Integration

## Document Version: 1.0 | Date: 2026-04-22

---

## Overview

This document provides API specifications for BAC team to integrate their camera software with the Hallmark QC System. Two types of image uploads are supported:

1. **Article Photo** - Jewelry item photos (pre-HUID marking)
2. **HUID Photo** - Microscopic hallmark photos (post-HUID marking)

---

## Authentication

All API calls require an API key in the header:

```
x-api-key: {your_api_key}
```

API keys will be provided separately.

---

## Base URL

```
Production: https://[TBD]/api/external
Development: http://[TBD]:8000/api/external
```

---

## Option A: Presigned URL Upload (Recommended)

This approach allows direct upload to S3 without routing through our server. Best for large files and high throughput.

### Step 1: Get Upload URL

**Endpoint:** `POST /get-upload-url`

**Request:**
```json
{
  "tag_id": "51D5M2NPSAAA002JA000045_1",
  "job_no": "125554301",
  "branch": "Hosur",
  "image_type": "article",  // "article" or "huid"
  "device_source": "camera_station_1",
  "content_type": "image/jpeg"
}
```

**Response:**
```json
{
  "success": true,
  "upload_url": "https://s3.ap-south-1.amazonaws.com/bucket/...?X-Amz-Signature=...",
  "s3_key": "article-photos/Hosur/125554301/51D5M2NPSAAA002JA000045_1.jpg",
  "expires_in": 3600
}
```

### Step 2: Upload Image to S3

**Method:** `PUT {upload_url}`

**Headers:**
```
Content-Type: image/jpeg
```

**Body:** Raw image binary

**Response:** HTTP 200 on success

### Step 3: Confirm Upload (Required)

**Endpoint:** `POST /confirm-upload`

**Request:**
```json
{
  "tag_id": "51D5M2NPSAAA002JA000045_1",
  "s3_key": "article-photos/Hosur/125554301/51D5M2NPSAAA002JA000045_1.jpg",
  "image_type": "article"
}
```

**Response (Article Photo):**
```json
{
  "success": true,
  "tag_id": "51D5M2NPSAAA002JA000045_1",
  "image_url": "https://...",
  "is_detachable": true,
  "detachable_suffix": "_1"
}
```

**Response (HUID Photo - includes OCR):**
```json
{
  "success": true,
  "tag_id": "51D5M2NPSAAA002JA000045_1",
  "image_url": "https://...",
  "ocr_result": {
    "huid_detected": "YY6DUG",
    "purity_code": "916",
    "karat": "22K",
    "confidence": 0.92,
    "raw_text": "22K916YY6DUG"
  }
}
```

---

## Option B: Direct Multipart Upload

Simpler integration but image routes through our server.

### Article Photo Upload

**Endpoint:** `POST /article-photo`

**Content-Type:** `multipart/form-data`

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| tag_id | string | Yes | Tag ID (e.g., "TAG001_1") |
| job_no | string | Yes | BIS Job Number |
| branch | string | Yes | Branch name (e.g., "Hosur") |
| device_source | string | No | Camera identifier |
| image | file | Yes | Image file (JPEG/PNG) |

**Example (cURL):**
```bash
curl -X POST https://api.example.com/api/external/article-photo \
  -H "x-api-key: your_api_key" \
  -F "tag_id=51D5M2NPSAAA002JA000045_1" \
  -F "job_no=125554301" \
  -F "branch=Hosur" \
  -F "device_source=camera_1" \
  -F "image=@/path/to/photo.jpg"
```

**Response:**
```json
{
  "success": true,
  "tag_id": "51D5M2NPSAAA002JA000045_1",
  "job_no": "125554301",
  "image_url": "https://s3.../article-photos/51D5M2NPSAAA002JA000045_1.jpg",
  "is_detachable": true,
  "detachable_suffix": "_1",
  "uploaded_at": "2026-04-22T10:30:00Z"
}
```

---

### HUID Photo Upload

**Endpoint:** `POST /huid-photo`

**Content-Type:** `multipart/form-data`

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| tag_id | string | Yes | Tag ID (e.g., "TAG001_1") |
| job_no | string | Yes | BIS Job Number |
| branch | string | Yes | Branch name |
| device_source | string | No | Camera identifier (e.g., "microscope_1") |
| image | file | Yes | Microscopic image (JPEG/PNG) |

**Example (cURL):**
```bash
curl -X POST https://api.example.com/api/external/huid-photo \
  -H "x-api-key: your_api_key" \
  -F "tag_id=51D5M2NPSAAA002JA000045_1" \
  -F "job_no=125554301" \
  -F "branch=Hosur" \
  -F "device_source=microscope_cam_1" \
  -F "image=@/path/to/huid_photo.jpg"
```

**Response:**
```json
{
  "success": true,
  "tag_id": "51D5M2NPSAAA002JA000045_1",
  "job_no": "125554301",
  "image_url": "https://s3.../huid-photos/51D5M2NPSAAA002JA000045_1_HUID.jpg",
  "uploaded_at": "2026-04-22T10:35:00Z",
  "ocr_result": {
    "huid_detected": "YY6DUG",
    "purity_code": "916",
    "karat": "22K",
    "purity_percentage": 91.6,
    "confidence": 0.92,
    "raw_text": "22K916YY6DUG",
    "decision": "approved"
  }
}
```

**OCR Decision Values:**
- `approved` - Confidence >= 85%, HUID readable
- `manual_review` - Confidence 50-85%, needs human verification
- `rejected` - Confidence < 50% or HUID not detected

---

## Utility Endpoints

### Check Photo Exists

**Endpoint:** `GET /photo-exists/{tag_id}`

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| image_type | string | "article" or "huid" (default: both) |

**Response:**
```json
{
  "tag_id": "51D5M2NPSAAA002JA000045_1",
  "article_photo": {
    "exists": true,
    "image_url": "https://...",
    "uploaded_at": "2026-04-22T10:30:00Z"
  },
  "huid_photo": {
    "exists": true,
    "image_url": "https://...",
    "uploaded_at": "2026-04-22T10:35:00Z",
    "ocr_status": "approved"
  }
}
```

### Get OCR Result

**Endpoint:** `GET /ocr-result/{tag_id}`

**Response:**
```json
{
  "tag_id": "51D5M2NPSAAA002JA000045_1",
  "expected_huid": "YY6DUG",
  "actual_huid": "YY6DUG",
  "huid_match": true,
  "purity_code": "916",
  "karat": "22K",
  "confidence": 0.92,
  "decision": "approved",
  "processed_at": "2026-04-22T10:35:00Z"
}
```

---

## File Naming Convention

Images are stored with the following naming pattern:

| Type | Pattern | Example |
|------|---------|---------|
| Article Photo | `{tag_id}.jpg` | `51D5M2NPSAAA002JA000045_1.jpg` |
| Article Photo (2nd) | `{tag_id}(1).jpg` | `51D5M2NPSAAA002JA000045_1(1).jpg` |
| HUID Photo | `{tag_id}_HUID.jpg` | `51D5M2NPSAAA002JA000045_1_HUID.jpg` |

### Detachable Items

Tag IDs with suffix `_1`, `_2` indicate detachable parts:
- `TAG001_1` - First detachable part
- `TAG001_2` - Second detachable part

Both parts require separate Article and HUID photos.

---

## S3 Folder Structure

```
hallmark-images/
├── article-photos/
│   └── {branch}/
│       └── {job_no}/
│           ├── TAG001_1.jpg
│           ├── TAG001_2.jpg
│           └── TAG002.jpg
└── huid-photos/
    └── {branch}/
        └── {job_no}/
            ├── TAG001_1_HUID.jpg
            ├── TAG001_2_HUID.jpg
            └── TAG002_HUID.jpg
```

---

## Error Responses

All errors return JSON with this structure:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_TAG_ID",
    "message": "Tag ID not found in system",
    "details": {}
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_API_KEY | 401 | API key missing or invalid |
| INVALID_TAG_ID | 404 | Tag ID not found |
| INVALID_IMAGE_TYPE | 400 | Must be JPEG, PNG, BMP, or WEBP |
| UPLOAD_FAILED | 500 | S3 upload failed |
| OCR_FAILED | 500 | OCR processing failed |
| DUPLICATE_UPLOAD | 409 | Image already exists (use replace flag) |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| Upload APIs | 100 requests/minute |
| Query APIs | 500 requests/minute |

---

## Integration Checklist

- [ ] Obtain API key from QC team
- [ ] Test with development environment first
- [ ] Implement error handling for all error codes
- [ ] Log all upload attempts for debugging
- [ ] Implement retry logic (3 attempts, exponential backoff)
- [ ] Validate image size before upload (max 10MB)

---

## Support

For integration support, contact: [TBD]
