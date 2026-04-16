# EC2 Server Update Steps

## Updates Made
1. Made `expected_huid` parameter optional in `/api/erp/upload-and-process` endpoint
2. Created `capture_image.py` for Windows camera capture with ERP integration
3. Created test script to verify live API

## Steps to Update EC2 Server

### 1. Pull Latest Changes
```bash
cd /home/ubuntu/ocr-detection
git pull origin main
```

### 2. Restart API Service
```bash
pm2 restart hallmark-api
```

### 3. Verify API is Running
```bash
pm2 logs hallmark-api --lines 20
```

### 4. Test the API
From your local machine, run:
```bash
python test_api_live.py
```

## Expected Result
The API should now accept requests with or without `expected_huid`:
- With `expected_huid=""` (empty string) - OCR will process without HUID validation
- With `expected_huid="ABC123"` - OCR will validate detected HUID against expected

## API Endpoint
```
POST http://65.2.187.3:8000/api/erp/upload-and-process
```

**Form Data:**
- `file`: Image file (JPG/PNG)
- `tag_id`: Unique identifier for the image
- `expected_huid`: Expected HUID (optional, can be empty string)

**Response:**
```json
{
  "status": "completed",
  "tag_id": "TAG_001",
  "decision": "approved",
  "actual_huid": "ABC123",
  "huid_match": true,
  "confidence": 0.92,
  "processing_time_ms": 1234
}
```

## PM2 Management Commands
```bash
# Check status
pm2 status

# View logs
pm2 logs hallmark-api
pm2 logs hallmark-dashboard

# Restart services
pm2 restart all
pm2 restart hallmark-api
pm2 restart hallmark-dashboard

# Stop services
pm2 stop all

# Monitor
pm2 monit
```
