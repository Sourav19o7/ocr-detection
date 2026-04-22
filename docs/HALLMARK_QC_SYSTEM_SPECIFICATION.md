# Hallmark QC System - Complete Specification

## Project Overview

The Hallmark QC System is designed to automate the quality control process for hallmarked jewelry items. It performs OCR on jewelry images to extract HUID (Hallmark Unique Identification) codes and purity values, compares them with official Manakonline portal data, and facilitates semi-automated upload of verified images back to the portal.

---

## Module 1: Core OCR System

### 1.1 Requirements
- Upload jewelry images (article photos and HUID close-ups)
- Extract HUID codes (6-character alphanumeric) from images
- Extract purity values (22K916, 18K750, 925, etc.) from images
- Support batch processing of multiple items
- Store results in database for comparison

### 1.2 Implementation Status

| Feature | Status | Location |
|---------|--------|----------|
| Image Upload | ✅ Done | `api.py` - `/api/items/{id}/images` |
| HUID Extraction | ✅ Done | `services/ocr_service.py` |
| Purity Extraction | ✅ Done | `services/ocr_service.py` |
| Batch Processing | ✅ Done | `api.py` - `/api/process/{batch_id}` |
| Database Storage | ✅ Done | `database.py`, `ocr_results` table |

### 1.3 API Endpoints

```
POST /api/batches
  - Create a new batch
  - Request: { "name": "Batch Name", "branch": "Branch Name" }
  - Response: { "batch_id": "uuid", "status": "created" }

GET /api/batches
  - List all batches
  - Response: [{ "id": "uuid", "name": "...", "status": "...", "item_count": N }]

GET /api/batches/{batch_id}
  - Get batch details with items
  - Response: { "id": "...", "items": [...], "stats": {...} }

POST /api/batches/{batch_id}/items
  - Add items to batch
  - Request: { "items": [{ "tag_id": "ABC123", "bis_job_no": "100166462" }] }

POST /api/items/{item_id}/images
  - Upload images for an item
  - Request: multipart/form-data with image files
  - Params: image_type = "article" | "huid"

POST /api/process/{batch_id}
  - Process all images in batch with OCR
  - Response: { "processed": N, "results": [...] }
```

### 1.4 Database Schema

```sql
-- Batches table
CREATE TABLE batches (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    branch TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP,
    processed_at TIMESTAMP
);

-- Batch items
CREATE TABLE batch_items (
    id TEXT PRIMARY KEY,
    batch_id TEXT REFERENCES batches(id),
    tag_id TEXT NOT NULL,
    bis_job_no TEXT,
    status TEXT DEFAULT 'pending'
);

-- Item images
CREATE TABLE item_images (
    id TEXT PRIMARY KEY,
    item_id TEXT REFERENCES batch_items(id),
    image_type TEXT CHECK (image_type IN ('article', 'huid')),
    file_path TEXT,
    s3_key TEXT
);

-- OCR results
CREATE TABLE ocr_results (
    id TEXT PRIMARY KEY,
    item_id TEXT REFERENCES batch_items(id),
    image_id TEXT REFERENCES item_images(id),
    detected_huid TEXT,
    detected_purity TEXT,
    confidence REAL,
    raw_text TEXT
);
```

---

## Module 2: Manakonline Excel Parser

### 2.1 Requirements
- Parse Excel files exported from Manakonline portal
- Extract: BIS Job No, AHC Tag, HUID, Purity, Weight, Article Type
- Handle various Excel formats (different column orders, sheet names)
- Store parsed data for comparison with OCR results

### 2.2 Implementation Status

| Feature | Status | Location |
|---------|--------|----------|
| Excel Parsing | ✅ Done | `config/manak_parser.py` |
| Column Auto-detection | ✅ Done | `config/manak_parser.py` |
| Multiple Sheet Support | ✅ Done | `config/manak_parser.py` |
| API Endpoint | ✅ Done | `api.py` - `/api/manak/upload-excel` |

### 2.3 API Endpoints

```
POST /api/manak/upload-excel
  - Upload and parse Manakonline Excel file
  - Request: multipart/form-data with Excel file
  - Response: {
      "bis_job_no": "100166462",
      "total_items": 150,
      "items": [
        { "tag_id": "570001123111", "huid": "ABC123", "purity": "22K916" },
        ...
      ]
    }
```

### 2.4 Excel Format Expected

| BIS Job No | AHC Tag | HUID | Purity | Gross Wt | Net Wt | Article |
|------------|---------|------|--------|----------|--------|---------|
| 100166462 | 570001123111 | ABC123 | 22K916 | 5.5 | 5.2 | Ring |

---

## Module 3: Comparison Engine

### 3.1 Requirements
- Compare OCR-extracted values with Manakonline Excel data
- Match by AHC Tag ID
- Report: matches, mismatches, partial matches, missing OCR
- Generate comparison reports

### 3.2 Implementation Status

| Feature | Status | Location |
|---------|--------|----------|
| HUID Comparison | ✅ Done | `api.py` - `/api/manak/compare/{bis_job_no}` |
| Purity Comparison | ✅ Done | `api.py` - `/api/manak/compare/{bis_job_no}` |
| Match Status | ✅ Done | `manak_comparisons` table |
| Comparison Report | ✅ Done | API returns detailed comparison |

### 3.3 API Endpoints

```
GET /api/manak/compare/{bis_job_no}
  - Compare OCR results with Manak Excel data
  - Response: {
      "bis_job_no": "100166462",
      "total_items": 150,
      "matches": 140,
      "mismatches": 5,
      "partial_matches": 3,
      "missing_ocr": 2,
      "items": [
        {
          "tag_id": "570001123111",
          "manak_huid": "ABC123",
          "ocr_huid": "ABC123",
          "huid_match": true,
          "manak_purity": "22K916",
          "ocr_purity": "22K916",
          "purity_match": true,
          "status": "match"
        },
        ...
      ]
    }
```

### 3.4 Match Status Types

| Status | Description |
|--------|-------------|
| `match` | Both HUID and Purity match |
| `partial_match` | Either HUID or Purity matches |
| `mismatch` | Neither HUID nor Purity matches |
| `missing_ocr` | No OCR result for this tag |

---

## Module 4: Upload Queue Management

### 4.1 Requirements
- Queue items that need to be uploaded to Manakonline portal
- Track upload status (pending, uploading, uploaded, failed)
- Support retry logic for failed uploads
- Store S3/local URLs for images

### 4.2 Implementation Status

| Feature | Status | Location |
|---------|--------|----------|
| Queue Table | ✅ Done | `upload_queue` table |
| Add to Queue | ✅ Done | `api.py` |
| Get Queue | ✅ Done | `/api/manak/upload-queue` |
| Update Status | ✅ Done | `/api/manak/upload-result` |
| Retry Logic | ✅ Done | `background.js` |

### 4.3 API Endpoints

```
GET /api/manak/upload-queue
  - Get pending upload items
  - Params: status=pending, limit=100
  - Response: {
      "items": [
        {
          "id": 1,
          "tag_id": "570001123111",
          "bis_job_no": "100166462",
          "image_type": "article",
          "image_url": "http://localhost:8000/uploads/...",
          "status": "pending",
          "retry_count": 0
        },
        ...
      ],
      "total_pending": 50,
      "stats": { "pending": 50, "uploaded": 100, "failed": 2 }
    }

POST /api/manak/upload-result
  - Report upload result
  - Request: {
      "tag_id": "570001123111",
      "image_type": "article",
      "status": "success" | "failed",
      "error_message": "optional error"
    }
```

### 4.4 Database Schema

```sql
CREATE TABLE upload_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_id TEXT NOT NULL,
    bis_job_no TEXT NOT NULL,
    branch TEXT,
    image_type TEXT CHECK (image_type IN ('article', 'huid')),
    s3_key TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'queued', 'uploading', 'uploaded', 'failed')),
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    portal_reference TEXT,
    queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_at TIMESTAMP,
    UNIQUE(tag_id, image_type)
);
```

---

## Module 5: Browser Extension (Portal Automation)

### 5.1 Requirements
- Detect login state on Manakonline portal
- Navigate through portal pages (Dashboard → Job List → Tag List → Upload)
- Download images from API/S3
- Automate file selection and upload
- Report success/failure back to API
- Handle multiple items in queue

### 5.2 Implementation Status

| Feature | Status | Location |
|---------|--------|----------|
| Extension Manifest | ✅ Done | `browser-extension/manifest.json` |
| Background Worker | ✅ Done | `browser-extension/background.js` |
| Content Script | ⚠️ Partial | `browser-extension/content.js` |
| Popup UI | ✅ Done | `browser-extension/popup/` |
| Login Detection | ✅ Done | `content.js` - `isLoggedIn()` |
| Page Type Detection | ✅ Done | `content.js` - `getPageType()` |
| Queue Fetch | ✅ Done | `background.js` - `fetchUploadQueue()` |
| Job Navigation | ❌ Not Working | Needs portal HTML |
| Tag Navigation | ❌ Not Working | Needs portal HTML |
| File Upload | ❌ Not Working | Needs portal HTML |
| Auto-Upload Flow | ❌ Not Working | Depends on above |

### 5.3 Portal Page Flow

```
1. Login Page
   URL: https://newmanak.uat.dcservices.in/MANAK/eBISLogin

2. Dashboard
   URL: https://newmanak.uat.dcservices.in/MANAK/JewellerHM
   Action: Click "Hallmarking" → "Upload Image" → "View"

3. Job List Page
   URL: https://newmanak.uat.dcservices.in/MANAK/getListForUploadImage
   Action: Find row with BIS Job No, click "View"

4. Tag List Page
   URL: https://newmanak.uat.dcservices.in/MANAK/getHuidListUploadImage?param=...
   Action: Find row with AHC Tag, click to go to upload

5. Upload Page
   URL: https://newmanak.uat.dcservices.in/MANAK/getUploadImage?param=...
   Action: Select file, click "Upload Photos"
```

### 5.4 Extension Files

```
browser-extension/
├── manifest.json           # Chrome extension manifest (v3)
├── background.js           # Service worker
│   ├── State management
│   ├── Queue fetch from API
│   ├── Message routing
│   └── Upload result reporting
├── content.js              # Content script (injected into portal)
│   ├── Login detection
│   ├── Page type detection
│   ├── DOM interaction (NOT WORKING)
│   └── File upload (NOT WORKING)
├── content.css             # Status bar styling
└── popup/
    ├── popup.html          # Extension popup UI
    ├── popup.js            # Popup logic
    └── popup.css           # Popup styling
```

### 5.5 What's Needed to Complete

To make the browser extension work, we need:

#### A. Portal HTML Structure
```
1. Job List Page HTML
   - How to identify the table containing jobs
   - How to find a specific BIS Job No row
   - What element to click to navigate to Tag List

2. Tag List Page HTML
   - How to identify the table containing tags
   - How to find a specific AHC Tag row
   - What element to click to navigate to Upload Page

3. Upload Page HTML
   - What is the file input element (id, name, class)
   - What is the upload button element
   - Is it a form POST or AJAX request
   - Are there any hidden fields required
```

#### B. Example Selectors Needed
```javascript
// Job List Page
const jobTable = document.querySelector('???');
const jobRow = jobTable.querySelector('tr:contains("100166462")');
const viewButton = jobRow.querySelector('???');

// Tag List Page
const tagTable = document.querySelector('???');
const tagRow = tagTable.querySelector('tr:contains("570001123111")');
const uploadLink = tagRow.querySelector('???');

// Upload Page
const fileInput = document.querySelector('???');
const uploadButton = document.querySelector('???');
```

---

## Module 6: External Photo Upload API

### 6.1 Requirements
- Allow external systems to upload photos
- Associate photos with tag IDs
- Store in S3 or local storage
- Add to upload queue automatically

### 6.2 Implementation Status

| Feature | Status | Location |
|---------|--------|----------|
| Photo Upload Endpoint | ✅ Done | `/api/external/photo` |
| S3 Storage | ✅ Done | `services/storage.py` |
| Local Storage Fallback | ✅ Done | `services/storage.py` |
| Auto-Queue | ✅ Done | Adds to `upload_queue` |

### 6.3 API Endpoint

```
POST /api/external/photo
  - Upload photo from external system
  - Headers: x-api-key: <api_key>
  - Request: multipart/form-data
    - file: image file
    - tag_id: "570001123111"
    - bis_job_no: "100166462"
    - image_type: "article" | "huid"
    - branch: "Branch Name" (optional)
  - Response: {
      "success": true,
      "tag_id": "570001123111",
      "image_type": "article",
      "s3_key": "images/100166462/570001123111_article.jpg",
      "queued": true
    }
```

---

## Summary Table

| Module | Description | Status |
|--------|-------------|--------|
| 1. Core OCR | Image upload, HUID/Purity extraction | ✅ Complete |
| 2. Excel Parser | Parse Manakonline exports | ✅ Complete |
| 3. Comparison | Compare OCR vs Manak data | ✅ Complete |
| 4. Upload Queue | Queue management for uploads | ✅ Complete |
| 5. Browser Extension | Portal automation | ⚠️ Partial (needs portal HTML) |
| 6. External API | Photo upload from external systems | ✅ Complete |

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.11, FastAPI |
| Database | SQLite |
| OCR | Google Vision API / Tesseract |
| Storage | AWS S3 / Local filesystem |
| Extension | Chrome Extension Manifest V3 |
| Frontend | HTML, CSS, JavaScript |

---

## File Structure

```
ocr-detection/
├── api.py                      # FastAPI application
├── database.py                 # Database connection
├── config/
│   ├── manak_parser.py         # Excel parser
│   └── settings.py             # Configuration
├── services/
│   ├── ocr_service.py          # OCR processing
│   └── storage.py              # S3/local storage
├── migrations/
│   ├── m001_initial.py         # Initial schema
│   └── m002_external_api.py    # External API tables
├── browser-extension/          # Chrome extension
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   ├── content.css
│   └── popup/
├── uploads/                    # Local file storage
├── docs/                       # Documentation
└── hallmark_qc.db              # SQLite database
```

---

## Next Steps

1. **Get Portal HTML**: Capture HTML from Job List, Tag List, and Upload pages
2. **Update content.js**: Write correct selectors based on actual HTML
3. **Test on UAT**: Test the extension on UAT portal
4. **Handle Edge Cases**: Add error handling, retries, timeouts
5. **Production Deploy**: Deploy API to production server
