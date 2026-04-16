# Batch Upload Guide

Comprehensive guide for uploading batch data (CSV and Excel files) with tag IDs and HUIDs.

## 📋 Supported File Formats

### CSV Files
- **Extension**: `.csv`
- **Encoding**: UTF-8 (recommended)
- **Delimiter**: Comma (`,`)

### Excel Files
- **Extensions**: `.xlsx`, `.xls`
- **Sheet Detection**: Automatically detects HUID sheets
- **Compatibility**: Microsoft Excel 2007+ format

## 🎯 Automatic HUID Sheet Detection

The system intelligently detects and reads the correct sheet from Excel files:

### Detection Logic
1. Scans all sheet names in the Excel file
2. Looks for sheets containing **"HUID"** or **"PRINT"** (case-insensitive)
3. Reads the first matching sheet
4. Falls back to the first sheet if no match found

### Examples of Detected Sheet Names
- ✅ `HUID_PRINT_125554301`
- ✅ `HUID_Print_125554301`
- ✅ `huid_data`
- ✅ `Print_Sheet`
- ✅ `HUID_Master`
- ✅ `Daily_Print_Report`

## 📊 Required Columns

Your file must contain these two columns (case-insensitive):

### Tag ID Column
Accepted variations:
- `tag_id`
- `tagid`
- `tag`
- `id`
- `item_id`

### HUID Column
Accepted variations:
- `expected_huid`
- `huid`
- `expected_id`
- `expectedhuid`

## 📝 File Format Examples

### CSV Format
```csv
tag_id,expected_huid
TAG001,AB1234567890123456
TAG002,CD9876543210987654
TAG003,EF5555555555555555
```

### Excel Format (HUID_PRINT sheet)
| tag_id | expected_huid |
|--------|---------------|
| TAG001 | AB1234567890123456 |
| TAG002 | CD9876543210987654 |
| TAG003 | EF5555555555555555 |

### Alternative Column Names
```csv
tagid,huid
TAG001,AB1234567890123456
TAG002,CD9876543210987654
```

## 🚀 Upload Process

### Via Dashboard (http://localhost:8501)

1. Navigate to **"Stage 1: Upload Data"** tab
2. Click **"Drop your file here or click to browse"**
3. Select your CSV or Excel file
4. (Optional) Enter a batch name
5. Preview the data
6. Click **"Upload Batch"**

### Via API (http://localhost:8000)

```bash
curl -X POST http://localhost:8000/stage1/upload-batch \
  -F "file=@HUID_Print_125554301.xlsx" \
  -F "batch_name=Morning_Batch_001"
```

## ✅ Data Validation

The system automatically:

1. **Normalizes column names**: Converts to lowercase, removes spaces
2. **Removes duplicates**: Based on `tag_id`
3. **Removes empty rows**: Filters out rows with missing data
4. **Validates columns**: Ensures required columns exist

## 📂 Example Files

### Minimal CSV
```csv
tag_id,expected_huid
TAG001,AB1234567890123456
```

### CSV with Alternative Names
```csv
id,huid
TAG001,AB1234567890123456
TAG002,CD9876543210987654
```

### Excel with Multiple Sheets
If your Excel file has multiple sheets:
```
Sheet1: General_Data
Sheet2: HUID_PRINT_125554301  ← This will be auto-selected
Sheet3: Summary
```

## ❌ Common Issues

### Issue: "Missing required column"
**Cause**: Column names don't match expected variations
**Solution**: Use one of the accepted column names (see Required Columns)

### Issue: "No valid rows found"
**Cause**: All rows have missing tag_id or expected_huid
**Solution**: Ensure every row has both values filled

### Issue: "Invalid file type"
**Cause**: Unsupported file format
**Solution**: Use CSV (.csv) or Excel (.xlsx, .xls)

### Issue: Excel file won't open
**Cause**: Corrupted or incompatible Excel format
**Solution**:
1. Open in Excel and Save As → Excel Workbook (.xlsx)
2. Or export to CSV format

## 🔧 Technical Details

### Column Mapping Process
```python
# Example transformation
"Tag ID" → "tag_id"
"Expected HUID" → "expected_huid"
"TagID" → "tagid" → "tag_id"
```

### Sheet Detection Code
```python
# Automatically finds HUID sheets
for sheet in sheet_names:
    if 'HUID' in sheet.upper() or 'PRINT' in sheet.upper():
        use_this_sheet = sheet
        break
```

## 📊 Batch Upload Response

Successful upload returns:
```json
{
  "batch_id": 1,
  "batch_name": "Morning_Batch_001",
  "total_items": 150,
  "status": "pending",
  "created_at": "2026-04-16T14:30:00Z"
}
```

## 🎯 Best Practices

1. **Use consistent naming**: Stick to `tag_id` and `expected_huid`
2. **Remove duplicates**: Before uploading
3. **Validate HUIDs**: Ensure correct format (16 characters)
4. **Use descriptive batch names**: e.g., "Morning_Batch_2026-04-16"
5. **Keep sheets organized**: Name HUID sheets clearly

## 📱 Dashboard Features

After upload, you can:
- View batch statistics
- See uploaded items count
- Track processing status
- View recent batches in sidebar

## 🔗 API Endpoints

### Upload Batch
```
POST /stage1/upload-batch
Content-Type: multipart/form-data

Parameters:
- file: CSV or Excel file
- batch_name: Optional batch name (string)
```

### List Batches
```
GET /stage1/batches
```

### Get Batch Details
```
GET /stage1/batches/{batch_id}
```

## 💡 Tips

1. **Large files**: Split into multiple batches if > 1000 rows
2. **Testing**: Upload a small test batch first (5-10 rows)
3. **Backup**: Keep original files before uploading
4. **Naming**: Use ISO date format in batch names (YYYY-MM-DD)

## 🎨 Premium Dashboard

The dashboard now features:
- **Lexend font** for premium typography
- **Clean, modern UI** with refined spacing
- **Smart file preview** showing first 10 rows
- **Real-time validation** before upload
- **Batch statistics** in sidebar

---

**Need Help?**
- Check [API Documentation](http://localhost:8000/docs)
- View [Design Updates](DESIGN_UPDATES.md)
- See [Local Services Guide](LOCAL_RUNNING_SERVICES.md)
