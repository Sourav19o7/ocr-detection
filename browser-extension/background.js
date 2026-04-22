/**
 * Hallmark QC Portal Uploader - Background Service Worker
 *
 * Manages:
 * - Session state across tabs
 * - Communication between popup and content scripts
 * - Upload queue management
 * - API communication with backend
 */

// Configuration
const CONFIG = {
  apiBaseUrl: 'http://localhost:8000',  // Change for production
  apiKey: '',  // Set via popup settings
  uploadDelayMs: 2000,  // Delay between uploads to avoid rate limiting
  maxRetries: 3,
  sessionCheckIntervalMs: 30000,  // Check session every 30 seconds
};

// State
let state = {
  isLoggedIn: false,
  isUploading: false,
  isPaused: false,
  currentTagId: null,
  uploadQueue: [],
  stats: {
    pending: 0,
    uploading: 0,
    completed: 0,
    failed: 0
  },
  lastError: null
};

// Initialize
chrome.runtime.onInstalled.addListener(() => {
  console.log('Hallmark QC Portal Uploader installed');
  loadConfig();
});

// Load saved configuration
async function loadConfig() {
  const saved = await chrome.storage.local.get(['apiBaseUrl', 'apiKey']);
  if (saved.apiBaseUrl) CONFIG.apiBaseUrl = saved.apiBaseUrl;
  if (saved.apiKey) CONFIG.apiKey = saved.apiKey;
}

// Save configuration
async function saveConfig(newConfig) {
  Object.assign(CONFIG, newConfig);
  await chrome.storage.local.set({
    apiBaseUrl: CONFIG.apiBaseUrl,
    apiKey: CONFIG.apiKey
  });
}

// Message handler
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender).then(sendResponse);
  return true; // Keep channel open for async response
});

async function handleMessage(message, sender) {
  switch (message.type) {
    case 'GET_STATE':
      return { success: true, state };

    case 'GET_CONFIG':
      return { success: true, config: CONFIG };

    case 'SAVE_CONFIG':
      await saveConfig(message.config);
      return { success: true };

    case 'LOGIN_DETECTED':
      state.isLoggedIn = true;
      state.lastError = null;
      notifyPopup({ type: 'STATE_CHANGED', state });
      return { success: true };

    case 'LOGOUT_DETECTED':
      state.isLoggedIn = false;
      if (state.isUploading) {
        state.isPaused = true;
        state.isUploading = false;
      }
      notifyPopup({ type: 'STATE_CHANGED', state });
      return { success: true };

    case 'START_UPLOAD':
      return await startUpload();

    case 'PAUSE_UPLOAD':
      state.isPaused = true;
      state.isUploading = false;
      notifyPopup({ type: 'STATE_CHANGED', state });
      return { success: true };

    case 'RESUME_UPLOAD':
      state.isPaused = false;
      return await startUpload();

    case 'FETCH_QUEUE':
      return await fetchUploadQueue();

    case 'UPLOAD_COMPLETE':
      return await handleUploadComplete(message);

    case 'UPLOAD_FAILED':
      return await handleUploadFailed(message);

    case 'GET_NEXT_ITEM':
      return getNextQueueItem();

    default:
      return { success: false, error: 'Unknown message type' };
  }
}

// Notify popup of state changes
async function notifyPopup(message) {
  try {
    await chrome.runtime.sendMessage(message);
  } catch (e) {
    // Popup might be closed, ignore
  }
}

// Fetch upload queue from backend
async function fetchUploadQueue() {
  try {
    const response = await fetch(`${CONFIG.apiBaseUrl}/api/manak/upload-queue?limit=100&status=pending`, {
      headers: {
        'x-api-key': CONFIG.apiKey
      }
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    state.uploadQueue = data.items || [];
    state.stats.pending = data.total_pending || 0;

    notifyPopup({ type: 'QUEUE_UPDATED', queue: state.uploadQueue, stats: state.stats });

    return { success: true, queue: state.uploadQueue, stats: state.stats };
  } catch (error) {
    state.lastError = error.message;
    return { success: false, error: error.message };
  }
}

// Start upload process
async function startUpload() {
  if (!state.isLoggedIn) {
    return { success: false, error: 'Not logged in to portal' };
  }

  if (state.uploadQueue.length === 0) {
    await fetchUploadQueue();
  }

  if (state.uploadQueue.length === 0) {
    return { success: false, error: 'No items in queue' };
  }

  state.isUploading = true;
  state.isPaused = false;
  notifyPopup({ type: 'STATE_CHANGED', state });

  // Send message to content script to start processing
  const tabs = await chrome.tabs.query({
    url: [
      'https://*.dcservices.in/*',
      'http://*.dcservices.in/*',
      'https://manak.bis.gov.in/*',
      'http://manak.bis.gov.in/*',
      'https://*.manakonline.in/*',
      'http://*.manakonline.in/*'
    ]
  });

  if (tabs.length === 0) {
    state.isUploading = false;
    return { success: false, error: 'No Manakonline tab open' };
  }

  // Start processing in the first matching tab
  chrome.tabs.sendMessage(tabs[0].id, {
    type: 'START_PROCESSING',
    item: state.uploadQueue[0]
  });

  return { success: true };
}

// Get next item from queue
function getNextQueueItem() {
  if (state.isPaused || !state.isUploading) {
    return { success: false, paused: true };
  }

  if (state.uploadQueue.length === 0) {
    state.isUploading = false;
    notifyPopup({ type: 'STATE_CHANGED', state });
    return { success: false, queueEmpty: true };
  }

  const item = state.uploadQueue[0];
  state.currentTagId = item.tag_id;
  notifyPopup({ type: 'PROCESSING', item });

  return { success: true, item };
}

// Handle successful upload
async function handleUploadComplete(message) {
  const { tag_id, image_type, portal_reference } = message;

  // Report to backend
  try {
    const formData = new FormData();
    formData.append('tag_id', tag_id);
    formData.append('image_type', image_type);
    formData.append('status', 'success');
    if (portal_reference) formData.append('portal_reference', portal_reference);

    await fetch(`${CONFIG.apiBaseUrl}/api/manak/upload-result`, {
      method: 'POST',
      headers: {
        'x-api-key': CONFIG.apiKey
      },
      body: formData
    });
  } catch (e) {
    console.error('Failed to report upload result:', e);
  }

  // Remove from queue
  state.uploadQueue = state.uploadQueue.filter(
    item => !(item.tag_id === tag_id && item.image_type === image_type)
  );
  state.stats.completed++;
  state.stats.pending = Math.max(0, state.stats.pending - 1);
  state.currentTagId = null;

  notifyPopup({ type: 'STATE_CHANGED', state });

  // Process next item after delay
  if (state.isUploading && !state.isPaused && state.uploadQueue.length > 0) {
    setTimeout(() => {
      const tabs = chrome.tabs.query({
        url: ['*://*.dcservices.in/*', '*://manak.bis.gov.in/*', '*://*.manakonline.in/*']
      }).then(tabs => {
        if (tabs.length > 0) {
          chrome.tabs.sendMessage(tabs[0].id, {
            type: 'PROCESS_NEXT',
            item: state.uploadQueue[0]
          });
        }
      });
    }, CONFIG.uploadDelayMs);
  } else if (state.uploadQueue.length === 0) {
    state.isUploading = false;
    notifyPopup({ type: 'UPLOAD_COMPLETE_ALL', stats: state.stats });
  }

  return { success: true };
}

// Handle failed upload
async function handleUploadFailed(message) {
  const { tag_id, image_type, error } = message;

  // Report to backend
  try {
    const formData = new FormData();
    formData.append('tag_id', tag_id);
    formData.append('image_type', image_type);
    formData.append('status', 'failed');
    formData.append('error_message', error || 'Unknown error');

    await fetch(`${CONFIG.apiBaseUrl}/api/manak/upload-result`, {
      method: 'POST',
      headers: {
        'x-api-key': CONFIG.apiKey
      },
      body: formData
    });
  } catch (e) {
    console.error('Failed to report upload error:', e);
  }

  // Move to end of queue or remove after max retries
  const item = state.uploadQueue.find(
    i => i.tag_id === tag_id && i.image_type === image_type
  );

  if (item) {
    item.retry_count = (item.retry_count || 0) + 1;

    if (item.retry_count >= CONFIG.maxRetries) {
      state.uploadQueue = state.uploadQueue.filter(
        i => !(i.tag_id === tag_id && i.image_type === image_type)
      );
      state.stats.failed++;
    } else {
      // Move to end of queue
      state.uploadQueue = state.uploadQueue.filter(
        i => !(i.tag_id === tag_id && i.image_type === image_type)
      );
      state.uploadQueue.push(item);
    }
  }

  state.stats.pending = state.uploadQueue.length;
  state.currentTagId = null;
  state.lastError = error;

  notifyPopup({ type: 'STATE_CHANGED', state });

  // Continue with next item
  if (state.isUploading && !state.isPaused && state.uploadQueue.length > 0) {
    setTimeout(() => {
      const tabs = chrome.tabs.query({
        url: ['*://*.dcservices.in/*', '*://manak.bis.gov.in/*', '*://*.manakonline.in/*']
      }).then(tabs => {
        if (tabs.length > 0) {
          chrome.tabs.sendMessage(tabs[0].id, {
            type: 'PROCESS_NEXT',
            item: state.uploadQueue[0]
          });
        }
      });
    }, CONFIG.uploadDelayMs);
  }

  return { success: true };
}
