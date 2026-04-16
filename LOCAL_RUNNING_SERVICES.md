# Local Services Running

## Current Status ✓

Both the API and Dashboard are now running locally on your machine.

### 1. FastAPI Server (Backend)
- **URL**: http://localhost:8000
- **Process**: Running in background (PID: check with `ps aux | grep uvicorn`)
- **Status**: ✓ Healthy
- **Logs**: `/private/tmp/claude/-Users-souravdey-Projects-ocr-detection/tasks/b01210d.output`

**Endpoints:**
- Health: http://localhost:8000/health
- Docs: http://localhost:8000/docs
- ERP Upload: http://localhost:8000/api/erp/upload-and-process

### 2. Streamlit Dashboard (Frontend)
- **URL**: http://localhost:8501
- **Process**: Running in background (PID: check with `ps aux | grep streamlit`)
- **Status**: ✓ Running
- **Logs**: `/private/tmp/claude/-Users-souravdey-Projects-ocr-detection/tasks/bf9d3b1.output`

**Features:**
- QC Dashboard tab
- Extract tab
- ERP Monitor tab (new)

## Access the Services

### Dashboard (Main Interface)
Open in your browser: **http://localhost:8501**

The dashboard has three tabs:
1. **QC Dashboard** - Quality control and batch management
2. **Extract** - OCR extraction interface
3. **ERP Monitor** - Monitor ERP integration uploads

### API Documentation
Open in your browser: **http://localhost:8000/docs**

Interactive API documentation with Swagger UI.

## Test the Local API

Run the test script:
```bash
python test_api_live.py
```

Or test manually with curl:
```bash
# Health check
curl http://localhost:8000/health

# Upload test image
curl -X POST http://localhost:8000/api/erp/upload-and-process \
  -F "file=@test.jpeg" \
  -F "tag_id=LOCAL_TEST_001" \
  -F "expected_huid="
```

## Stop the Services

To stop the running services:
```bash
# Find process IDs
ps aux | grep -E "(uvicorn|streamlit)" | grep -v grep

# Kill processes
pkill -f "uvicorn api:app"
pkill -f "streamlit run"
```

Or use the task IDs:
```bash
# Check running tasks
claude-code /tasks

# Or manually kill
kill <PID>
```

## Environment Configuration

The dashboard uses `API_BASE_URL` environment variable:
- **Current**: `http://localhost:8000` (default)
- **EC2 Production**: `http://65.2.187.3:8000`

To point to EC2 instead:
```bash
export API_BASE_URL=http://65.2.187.3:8000
venv/bin/streamlit run src/qc_dashboard.py --server.port 8501
```

## Next Steps

1. **Update EC2 Server**: Follow [EC2_UPDATE_STEPS.md](EC2_UPDATE_STEPS.md)
2. **Test Capture Tool**: Use `capture_image.py` on Windows machine
3. **Monitor Dashboard**: Check ERP Monitor tab for uploaded images

## Troubleshooting

### API not loading
- Check logs: `tail -f /private/tmp/claude/-Users-souravdey-Projects-ocr-detection/tasks/b01210d.output`
- Verify port 8000 is free: `lsof -i :8000`

### Dashboard not loading
- Check logs: `tail -f /private/tmp/claude/-Users-souravdey-Projects-ocr-detection/tasks/bf9d3b1.output`
- Verify port 8501 is free: `lsof -i :8501`

### Database issues
- Check if `hallmark_results.db` exists
- Check write permissions
