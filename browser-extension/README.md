# Hallmark QC Portal Uploader - Browser Extension

Semi-automated upload tool for Manakonline portal that handles bulk image uploads with manual login (CAPTCHA).

## Features

- **Session Detection**: Automatically detects when you're logged in
- **Queue Management**: Fetches pending uploads from backend
- **Automated Upload**: Navigates to tags and uploads images
- **Progress Tracking**: Real-time stats and progress
- **Error Handling**: Retry logic with error reporting
- **Pause/Resume**: Control upload process anytime

## Installation

### Development Mode (Chrome)

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right toggle)
3. Click "Load unpacked"
4. Select the `browser-extension` folder
5. The extension icon should appear in your toolbar

### Development Mode (Edge)

1. Open Edge and go to `edge://extensions/`
2. Enable "Developer mode" (left sidebar)
3. Click "Load unpacked"
4. Select the `browser-extension` folder

## Configuration

1. Click the extension icon
2. Click the settings gear icon
3. Enter:
   - **API Base URL**: Your backend URL (e.g., `http://localhost:8000`)
   - **API Key**: Your external API key (set in backend `.env`)
4. Click Save

## Usage

### Daily Workflow

1. **Login to Portal**
   - Open the Manakonline portal in your browser
   - Login manually (solve CAPTCHA)
   - Extension will detect login automatically

2. **Check Queue**
   - Click extension icon
   - Verify "Session Active" status
   - Click "Refresh Queue" to load pending items

3. **Start Upload**
   - Click "Start Upload"
   - Extension handles navigation and file uploads
   - Monitor progress in the popup

4. **Session Expiry**
   - If session expires, extension pauses automatically
   - Login again in the portal
   - Click "Resume" to continue

### Status Indicators

- **Green dot**: Logged in, ready to upload
- **Red dot**: Not logged in, please login first

### Stats

- **Pending**: Items waiting to upload
- **Uploading**: Currently processing
- **Completed**: Successfully uploaded
- **Failed**: Upload errors (will retry)

## Troubleshooting

### Extension not detecting login

- Refresh the portal page
- Check if you can see your username/logout button
- The extension looks for logout buttons to detect login state

### Upload fails

- Check if the tag ID exists in the portal
- Verify the BIS Job Number is correct
- Check browser console for errors

### Queue not loading

- Verify API URL in settings
- Check API key is correct
- Ensure backend server is running

## Technical Details

### Files

```
browser-extension/
├── manifest.json       # Extension manifest (v3)
├── background.js       # Service worker for state management
├── content.js          # DOM automation for portal
├── content.css         # Status bar styles
├── popup/
│   ├── popup.html      # Extension popup UI
│   ├── popup.css       # Popup styles
│   └── popup.js        # Popup logic
└── assets/
    └── icons/          # Extension icons
```

### Permissions

- `storage`: Save settings
- `activeTab`: Access current tab
- `scripting`: Run content scripts
- `downloads`: Download images from S3

### Host Permissions

- Manakonline portal URLs
- Backend API URL
- S3 for image downloads

## Creating Icons

The extension needs icons. Create these PNG files:

- `assets/icon16.png` (16x16)
- `assets/icon48.png` (48x48)
- `assets/icon128.png` (128x128)

Use any icon generator or create simple "HQC" badges.

## Backend Integration

The extension communicates with these endpoints:

```
GET  /api/manak/upload-queue     - Fetch pending uploads
POST /api/manak/upload-result    - Report upload status
GET  /api/manak/upload-stats     - Get statistics
```

## Security Notes

- API keys are stored in Chrome's local storage
- Never commit API keys to version control
- Use separate keys for development and production
- Rotate keys periodically

## Known Limitations

1. Cannot bypass CAPTCHA - manual login required
2. Single tab operation - only uploads to one tab at a time
3. Portal-specific selectors may need updates if UI changes
4. Chrome/Edge only (no Firefox support yet)
