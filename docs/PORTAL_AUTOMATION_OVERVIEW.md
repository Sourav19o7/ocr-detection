# Manakonline Portal Upload - Semi-Automated Tool Overview

## Document Version: 1.0 | Date: 2026-04-22

---

## Problem Statement

- 1000+ images need to be uploaded to manakonline portal daily
- Portal has CAPTCHA on login (cannot be fully automated)
- Each upload requires: finding BIS Job No → finding Tag ID → uploading image
- Current process is 100% manual, time-consuming, error-prone

---

## Solution: Semi-Automated Browser Extension

A Chrome/Edge browser extension that:
1. Detects when operator manually logs in (solves CAPTCHA)
2. Takes over repetitive upload tasks automatically
3. Reports status back to our system

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Operator's Browser                            │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Browser Extension                             │ │
│  │                                                                   │ │
│  │  ┌───────────────┐    ┌───────────────┐    ┌─────────────────┐  │ │
│  │  │ Login Monitor │    │ Upload Engine │    │ Status Reporter │  │ │
│  │  │               │    │               │    │                 │  │ │
│  │  │ - Detects     │    │ - Navigate    │    │ - Success/Fail  │  │ │
│  │  │   successful  │───▶│ - Find tag    │───▶│ - Sync to API   │  │ │
│  │  │   login       │    │ - Upload file │    │ - Update queue  │  │ │
│  │  │ - Stores      │    │ - Confirm     │    │                 │  │ │
│  │  │   session     │    │               │    │                 │  │ │
│  │  └───────────────┘    └───────────────┘    └─────────────────┘  │ │
│  │                              │                                    │ │
│  └──────────────────────────────┼────────────────────────────────────┘ │
│                                 │                                       │
└─────────────────────────────────┼───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Manakonline Portal                               │
│  ┌─────────────┐    ┌─────────────────┐    ┌───────────────────────┐   │
│  │ Login Page  │    │ BIS Job List    │    │ Upload Article Image  │   │
│  │ (CAPTCHA)   │───▶│                 │───▶│                       │   │
│  │             │    │ - Search job no │    │ - AHC Tag list        │   │
│  │ [MANUAL]    │    │ - Click View    │    │ - Browse/Capture      │   │
│  └─────────────┘    └─────────────────┘    └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ Images fetched from
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Our Backend System                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌───────────────────┐   │
│  │ Upload Queue    │    │ S3 Storage      │    │ Status Tracker    │   │
│  │                 │    │                 │    │                   │   │
│  │ - Pending items │    │ - Article imgs  │    │ - Mark uploaded   │   │
│  │ - Priority      │    │ - HUID imgs     │    │ - Log errors      │   │
│  │ - Retry count   │    │ - Presigned URL │    │ - Retry failed    │   │
│  └─────────────────┘    └─────────────────┘    └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## User Flow

### Step 1: Operator Starts Session
```
1. Operator opens manakonline portal
2. Logs in manually (solves CAPTCHA)
3. Extension detects successful login
4. Extension shows: "Session active - Ready to upload"
```

### Step 2: Automated Upload Process
```
1. Extension fetches upload queue from our API
2. For each item in queue:
   a. Navigate to BIS Job No page
   b. Find Tag ID row in table
   c. Download image from S3 (presigned URL)
   d. Click "Browse" button
   e. Upload the image file
   f. Wait for confirmation
   g. Report success/failure to our API
   h. Move to next item
3. On completion, show summary
```

### Step 3: Session Expiry Handling
```
1. Extension detects session timeout/logout
2. Pauses upload queue
3. Notifies operator: "Please login again"
4. Operator logs in (solves CAPTCHA)
5. Extension resumes from where it stopped
```

---

## Extension UI Components

### Popup Panel
```
┌─────────────────────────────────┐
│  Hallmark QC Uploader     [≡]  │
├─────────────────────────────────┤
│                                 │
│  Session: ● Active              │
│  Logged in as: userquam2        │
│                                 │
├─────────────────────────────────┤
│  Upload Queue                   │
│  ┌───────────────────────────┐  │
│  │ Pending:    45            │  │
│  │ Uploading:  1             │  │
│  │ Completed:  123           │  │
│  │ Failed:     2             │  │
│  └───────────────────────────┘  │
│                                 │
│  [▶ Start Upload] [⏸ Pause]    │
│                                 │
├─────────────────────────────────┤
│  Current: TAG001_1              │
│  Progress: ████████░░ 80%       │
│  Status: Uploading image...     │
│                                 │
├─────────────────────────────────┤
│  Speed: ~15 uploads/min         │
│  ETA: 3 minutes                 │
│                                 │
└─────────────────────────────────┘
```

### Status Bar (In-page overlay)
```
┌────────────────────────────────────────────────────────────────┐
│ 🔄 Uploading TAG001_1 (45/168) │ ⏸ Pause │ ⏹ Stop │ ⚙ Settings │
└────────────────────────────────────────────────────────────────┘
```

---

## Technical Implementation

### Extension Components

```
hallmark-qc-extension/
├── manifest.json          # Chrome extension manifest
├── background.js          # Service worker (session management)
├── content.js             # Page automation scripts
├── popup/
│   ├── popup.html         # Extension popup UI
│   ├── popup.js           # Popup logic
│   └── popup.css          # Popup styles
├── lib/
│   ├── api-client.js      # Communication with our backend
│   ├── session-manager.js # Login detection & session storage
│   └── upload-engine.js   # DOM automation for uploads
└── assets/
    └── icons/             # Extension icons
```

### Key Technical Challenges

| Challenge | Solution |
|-----------|----------|
| CAPTCHA on login | Manual login by operator, extension detects success |
| Session timeout | Monitor for logout, pause queue, prompt re-login |
| Dynamic page content | Wait for elements, retry on failure |
| File upload dialog | Use Chrome extension file API |
| Portal UI changes | Configurable selectors, fallback strategies |
| Rate limiting | Configurable delay between uploads |
| Network failures | Retry logic with exponential backoff |

---

## API Endpoints (Backend)

### Get Upload Queue
```
GET /api/manak/upload-queue?limit=50

Response:
{
  "items": [
    {
      "id": 123,
      "tag_id": "TAG001_1",
      "bis_job_no": "125554301",
      "article_image_url": "https://s3.../presigned...",
      "huid_image_url": "https://s3.../presigned...",
      "priority": 1,
      "retry_count": 0
    }
  ],
  "total_pending": 168
}
```

### Report Upload Result
```
POST /api/manak/upload-result

Body:
{
  "tag_id": "TAG001_1",
  "image_type": "article",  // or "huid"
  "status": "success",      // or "failed"
  "error_message": null,
  "portal_reference": "...",
  "uploaded_at": "2026-04-22T10:30:00Z"
}
```

### Get Upload Statistics
```
GET /api/manak/upload-stats?date=2026-04-22

Response:
{
  "date": "2026-04-22",
  "total_queued": 1000,
  "uploaded": 850,
  "failed": 12,
  "pending": 138,
  "upload_rate_per_hour": 150
}
```

---

## Performance Estimates

| Metric | Value |
|--------|-------|
| Uploads per minute | 10-15 |
| Uploads per hour | 600-900 |
| Daily capacity (8 hrs) | 4800-7200 |
| Your requirement | 1000/day |
| Time needed | ~1-2 hours/day |

---

## Failure Handling

### Retry Strategy
```
Attempt 1: Immediate
Attempt 2: Wait 5 seconds
Attempt 3: Wait 30 seconds
After 3 failures: Mark as failed, move to error queue
```

### Error Categories

| Error Type | Action |
|------------|--------|
| Session expired | Pause, prompt re-login |
| Tag not found | Log error, skip, continue |
| Upload button missing | Retry with different selector |
| Network timeout | Retry up to 3 times |
| Portal error | Log details, skip, continue |

---

## Security Considerations

1. **Credentials**: Never stored in extension, uses portal's session
2. **API Key**: Stored securely in extension storage
3. **Images**: Downloaded via presigned URLs (time-limited)
4. **Session**: Monitored for validity, auto-pause on expiry

---

## Operator Instructions (Draft)

### Daily Workflow

1. **Start of Day**
   - Open Chrome
   - Go to manakonline portal
   - Login with your credentials (solve CAPTCHA)
   - Click extension icon, verify "Session Active"

2. **Run Uploads**
   - Click "Start Upload" in extension
   - Monitor progress
   - Extension handles everything automatically

3. **If Session Expires**
   - Extension will pause and notify you
   - Login again in the portal
   - Click "Resume" in extension

4. **End of Day**
   - Check completion summary
   - Note any failed uploads for investigation

---

## Development Timeline

| Task | Effort |
|------|--------|
| Extension skeleton & manifest | 0.5 days |
| Session detection | 1 day |
| Upload queue integration | 1 day |
| DOM automation (navigate, find, upload) | 2 days |
| Error handling & retry | 1 day |
| Popup UI | 1 day |
| Testing & refinement | 1.5 days |
| **Total** | **8 days** |

---

## Limitations

1. **Cannot bypass CAPTCHA** - Operator must login manually
2. **Portal changes may break automation** - Need maintenance
3. **One browser session at a time** - Cannot parallelize
4. **Requires Chrome/Edge** - No Firefox support initially
