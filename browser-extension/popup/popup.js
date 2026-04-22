/**
 * Hallmark QC Portal Uploader - Popup Script
 */

// DOM Elements
const elements = {
  // Session
  statusDot: document.getElementById('statusDot'),
  statusText: document.getElementById('statusText'),

  // Stats
  statPending: document.getElementById('statPending'),
  statUploading: document.getElementById('statUploading'),
  statCompleted: document.getElementById('statCompleted'),
  statFailed: document.getElementById('statFailed'),

  // Current item
  currentItem: document.getElementById('currentItem'),
  currentTag: document.getElementById('currentTag'),
  progressFill: document.getElementById('progressFill'),

  // Actions
  startBtn: document.getElementById('startBtn'),
  pauseBtn: document.getElementById('pauseBtn'),
  refreshBtn: document.getElementById('refreshBtn'),

  // Error
  errorMessage: document.getElementById('errorMessage'),
  errorText: document.getElementById('errorText'),

  // Queue
  queueList: document.getElementById('queueList'),
  queueCount: document.getElementById('queueCount'),

  // Settings
  settingsBtn: document.getElementById('settingsBtn'),
  settingsPanel: document.getElementById('settingsPanel'),
  mainContent: document.getElementById('mainContent'),
  apiUrl: document.getElementById('apiUrl'),
  apiKey: document.getElementById('apiKey'),
  saveSettings: document.getElementById('saveSettings'),
  cancelSettings: document.getElementById('cancelSettings'),
};

// Initialize
document.addEventListener('DOMContentLoaded', init);

async function init() {
  // Load config
  await loadConfig();

  // Get initial state
  await refreshState();

  // Set up event listeners
  setupEventListeners();

  // Set up message listener for state updates
  chrome.runtime.onMessage.addListener(handleBackgroundMessage);

  // Refresh queue
  await refreshQueue();
}

async function loadConfig() {
  const response = await chrome.runtime.sendMessage({ type: 'GET_CONFIG' });
  if (response.success && response.config) {
    elements.apiUrl.value = response.config.apiBaseUrl || '';
    elements.apiKey.value = response.config.apiKey || '';
  }
}

async function refreshState() {
  const response = await chrome.runtime.sendMessage({ type: 'GET_STATE' });
  if (response.success && response.state) {
    updateUI(response.state);
  }
}

async function refreshQueue() {
  elements.refreshBtn.disabled = true;
  const response = await chrome.runtime.sendMessage({ type: 'FETCH_QUEUE' });
  elements.refreshBtn.disabled = false;

  if (response.success) {
    renderQueue(response.queue);
    updateStats(response.stats);
  } else {
    showError(response.error || 'Failed to fetch queue');
  }
}

function setupEventListeners() {
  // Start button
  elements.startBtn.addEventListener('click', async () => {
    hideError();
    elements.startBtn.disabled = true;
    const response = await chrome.runtime.sendMessage({ type: 'START_UPLOAD' });
    if (!response.success) {
      showError(response.error);
      elements.startBtn.disabled = false;
    }
  });

  // Pause button
  elements.pauseBtn.addEventListener('click', async () => {
    await chrome.runtime.sendMessage({ type: 'PAUSE_UPLOAD' });
  });

  // Refresh button
  elements.refreshBtn.addEventListener('click', refreshQueue);

  // Settings button
  elements.settingsBtn.addEventListener('click', () => {
    elements.mainContent.style.display = 'none';
    elements.settingsPanel.style.display = 'block';
  });

  // Cancel settings
  elements.cancelSettings.addEventListener('click', () => {
    elements.settingsPanel.style.display = 'none';
    elements.mainContent.style.display = 'flex';
    loadConfig(); // Reset to saved values
  });

  // Save settings
  elements.saveSettings.addEventListener('click', async () => {
    const config = {
      apiBaseUrl: elements.apiUrl.value.trim(),
      apiKey: elements.apiKey.value.trim(),
    };

    await chrome.runtime.sendMessage({ type: 'SAVE_CONFIG', config });

    elements.settingsPanel.style.display = 'none';
    elements.mainContent.style.display = 'flex';

    // Refresh queue with new settings
    await refreshQueue();
  });
}

function handleBackgroundMessage(message) {
  switch (message.type) {
    case 'STATE_CHANGED':
      updateUI(message.state);
      break;

    case 'QUEUE_UPDATED':
      renderQueue(message.queue);
      updateStats(message.stats);
      break;

    case 'PROCESSING':
      showProcessing(message.item);
      break;

    case 'UPLOAD_COMPLETE_ALL':
      showComplete(message.stats);
      break;
  }
}

function updateUI(state) {
  // Session status
  if (state.isLoggedIn) {
    elements.statusDot.className = 'status-dot active';
    elements.statusText.textContent = 'Session Active';
    elements.startBtn.disabled = false;
  } else {
    elements.statusDot.className = 'status-dot inactive';
    elements.statusText.textContent = 'Not logged in - Please login to portal';
    elements.startBtn.disabled = true;
  }

  // Uploading state
  if (state.isUploading) {
    elements.startBtn.style.display = 'none';
    elements.pauseBtn.style.display = 'flex';
  } else if (state.isPaused) {
    elements.startBtn.textContent = 'Resume';
    elements.startBtn.style.display = 'flex';
    elements.pauseBtn.style.display = 'none';
  } else {
    elements.startBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>
      Start Upload
    `;
    elements.startBtn.style.display = 'flex';
    elements.pauseBtn.style.display = 'none';
  }

  // Stats
  updateStats(state.stats);

  // Current item
  if (state.currentTagId) {
    showProcessing({ tag_id: state.currentTagId });
  } else {
    elements.currentItem.style.display = 'none';
  }

  // Error
  if (state.lastError) {
    showError(state.lastError);
  }
}

function updateStats(stats) {
  if (!stats) return;
  elements.statPending.textContent = stats.pending || 0;
  elements.statUploading.textContent = stats.uploading || 0;
  elements.statCompleted.textContent = stats.completed || 0;
  elements.statFailed.textContent = stats.failed || 0;
}

function renderQueue(queue) {
  if (!queue || queue.length === 0) {
    elements.queueList.innerHTML = '<div class="queue-empty">No items in queue</div>';
    elements.queueCount.textContent = '0 items';
    return;
  }

  elements.queueCount.textContent = `${queue.length} items`;

  // Show only first 10 items
  const displayItems = queue.slice(0, 10);

  elements.queueList.innerHTML = displayItems.map(item => `
    <div class="queue-item">
      <span class="queue-item-tag">${escapeHtml(item.tag_id)}</span>
      <span class="queue-item-type">${item.image_type}</span>
    </div>
  `).join('');

  if (queue.length > 10) {
    elements.queueList.innerHTML += `
      <div class="queue-item" style="justify-content: center; color: #718096;">
        ... and ${queue.length - 10} more
      </div>
    `;
  }
}

function showProcessing(item) {
  elements.currentItem.style.display = 'block';
  elements.currentTag.textContent = item.tag_id;
  elements.progressFill.style.width = '50%';
}

function showComplete(stats) {
  elements.currentItem.style.display = 'none';
  elements.startBtn.disabled = false;
  elements.startBtn.style.display = 'flex';
  elements.pauseBtn.style.display = 'none';

  // Show success message
  const total = (stats.completed || 0) + (stats.failed || 0);
  if (stats.failed === 0) {
    elements.statusText.textContent = `Completed! ${stats.completed} uploads successful`;
  } else {
    showError(`Completed with ${stats.failed} failures out of ${total}`);
  }
}

function showError(message) {
  elements.errorMessage.style.display = 'flex';
  elements.errorText.textContent = message;
}

function hideError() {
  elements.errorMessage.style.display = 'none';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
