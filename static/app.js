// DOM Elements
const loginSection = document.getElementById('login-section');
const uploadSection = document.getElementById('upload-section');
const loginForm = document.getElementById('login-form');
const loginError = document.getElementById('login-error');
const logoutBtn = document.getElementById('logout-btn');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileList = document.getElementById('file-list');
const uploadBtn = document.getElementById('upload-btn');
const uploadProgress = document.getElementById('upload-progress');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const uploadResults = document.getElementById('upload-results');

// State
let selectedFiles = [];

// Check authentication status on page load
async function checkAuth() {
  try {
    const response = await fetch('/api/auth/status');
    const data = await response.json();

    if (data.authenticated) {
      showUploadSection();
    } else {
      showLoginSection();
    }
  } catch (error) {
    showLoginSection();
  }
}

function showLoginSection() {
  loginSection.classList.remove('hidden');
  uploadSection.classList.add('hidden');
}

function showUploadSection() {
  loginSection.classList.add('hidden');
  uploadSection.classList.remove('hidden');
  resetUploadState();
}

function resetUploadState() {
  selectedFiles = [];
  renderFileList();
  uploadProgress.classList.add('hidden');
  uploadResults.classList.add('hidden');
  uploadBtn.classList.add('hidden');
  progressFill.style.width = '0%';
}

// Login handler
loginForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  loginError.textContent = '';

  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;

  try {
    const response = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    if (response.ok) {
      showUploadSection();
      loginForm.reset();
    } else {
      loginError.textContent = 'Invalid username or password';
    }
  } catch (error) {
    loginError.textContent = 'Login failed. Please try again.';
  }
});

// Logout handler
logoutBtn.addEventListener('click', async () => {
  try {
    await fetch('/api/logout', { method: 'POST' });
    showLoginSection();
  } catch (error) {
    console.error('Logout failed:', error);
  }
});

// Drag and drop handlers
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');

  const files = Array.from(e.dataTransfer.files);
  addFiles(files);
});

// Click to browse
dropZone.addEventListener('click', (e) => {
  if (e.target.tagName !== 'LABEL' && e.target.tagName !== 'INPUT') {
    fileInput.click();
  }
});

fileInput.addEventListener('change', (e) => {
  const files = Array.from(e.target.files);
  addFiles(files);
  fileInput.value = ''; // Reset input
});

// Add files to selected list
function addFiles(files) {
  files.forEach(file => {
    // Check if file already exists
    const exists = selectedFiles.some(f => f.name === file.name && f.size === file.size);
    if (!exists) {
      selectedFiles.push(file);
    }
  });

  renderFileList();
  updateUploadButton();
  uploadResults.classList.add('hidden');
}

// Remove file from list
function removeFile(index) {
  selectedFiles.splice(index, 1);
  renderFileList();
  updateUploadButton();
}

// Format file size
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Render file list
function renderFileList() {
  if (selectedFiles.length === 0) {
    fileList.innerHTML = '';
    return;
  }

  fileList.innerHTML = selectedFiles.map((file, index) => `
    <div class="file-item">
      <div class="file-info">
        <svg class="file-icon" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
          <polyline points="13 2 13 9 20 9"></polyline>
        </svg>
        <span class="file-name" title="${file.name}">${file.name}</span>
        <span class="file-size">${formatFileSize(file.size)}</span>
      </div>
      <button class="file-remove" onclick="removeFile(${index})" title="Remove file">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
  `).join('');
}

// Update upload button visibility
function updateUploadButton() {
  if (selectedFiles.length > 0) {
    uploadBtn.classList.remove('hidden');
    uploadBtn.textContent = `Upload ${selectedFiles.length} file${selectedFiles.length > 1 ? 's' : ''}`;
  } else {
    uploadBtn.classList.add('hidden');
  }
}

// Upload a single file directly to S3
async function uploadFileToS3(file, onProgress) {
  // Get presigned URL from server
  const urlResponse = await fetch('/api/get-upload-url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      fileName: file.name,
      contentType: file.type || 'application/octet-stream',
    }),
  });

  if (!urlResponse.ok) {
    const error = await urlResponse.json();
    throw new Error(error.error || 'Failed to get upload URL');
  }

  const { uploadUrl, fileName } = await urlResponse.json();

  // Upload directly to S3 using XMLHttpRequest for progress tracking
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(e.loaded / e.total);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve({ originalName: file.name, uploadedAs: fileName, size: file.size, success: true });
      } else {
        reject(new Error(`Upload failed with status ${xhr.status}`));
      }
    });

    xhr.addEventListener('error', () => reject(new Error('Network error during upload')));
    xhr.addEventListener('abort', () => reject(new Error('Upload aborted')));

    xhr.open('PUT', uploadUrl);
    xhr.setRequestHeader('Content-Type', file.type || 'application/octet-stream');
    xhr.send(file);
  });
}

// Upload files
uploadBtn.addEventListener('click', async () => {
  if (selectedFiles.length === 0) return;

  uploadBtn.disabled = true;
  uploadProgress.classList.remove('hidden');
  uploadResults.classList.add('hidden');
  progressFill.style.width = '0%';
  progressText.textContent = 'Preparing upload...';

  const uploadResultsList = [];
  const errors = [];
  const totalFiles = selectedFiles.length;
  let completedFiles = 0;

  for (const file of selectedFiles) {
    try {
      const result = await uploadFileToS3(file, (fileProgress) => {
        const overallProgress = ((completedFiles + fileProgress) / totalFiles) * 100;
        const percentage = Math.round(overallProgress);
        progressFill.style.width = `${overallProgress}%`;
        progressText.textContent = `Uploading ${file.name} (${completedFiles + 1}/${totalFiles}) - ${percentage}%`;
      });

      uploadResultsList.push(result);
      completedFiles++;
    } catch (error) {
      console.error(`Error uploading ${file.name}:`, error);
      errors.push({ originalName: file.name, error: error.message, success: false });
      completedFiles++;
    }
  }

  progressFill.style.width = '100%';
  progressText.textContent = 'Upload complete!';

  showUploadResults({
    uploaded: uploadResultsList,
    errors: errors,
    totalUploaded: uploadResultsList.length,
    totalErrors: errors.length,
  });

  selectedFiles = [];
  renderFileList();
  updateUploadButton();
  uploadBtn.disabled = false;
});

// Show upload results
function showUploadResults(result) {
  const { uploaded, errors, totalUploaded, totalErrors } = result;

  let className = 'success';
  let title = 'Upload Successful!';

  if (totalErrors > 0 && totalUploaded === 0) {
    className = 'error';
    title = 'Upload Failed';
  } else if (totalErrors > 0) {
    className = 'mixed';
    title = 'Partial Upload';
  }

  let html = `<p class="result-title ${className}">${title}</p>`;

  if (totalUploaded > 0) {
    html += `<p>${totalUploaded} file${totalUploaded > 1 ? 's' : ''} uploaded successfully</p>`;
    html += '<ul class="result-list">';
    uploaded.forEach(file => {
      html += `<li>${file.originalName}</li>`;
    });
    html += '</ul>';
  }

  if (totalErrors > 0) {
    html += `<p style="margin-top: 0.5rem; color: #dc2626;">${totalErrors} file${totalErrors > 1 ? 's' : ''} failed:</p>`;
    html += '<ul class="result-list">';
    errors.forEach(file => {
      html += `<li>${file.originalName}: ${file.error}</li>`;
    });
    html += '</ul>';
  }

  uploadResults.innerHTML = html;
  uploadResults.className = `upload-results ${className}`;
  uploadResults.classList.remove('hidden');
}

// Show error
function showError(message) {
  uploadResults.innerHTML = `<p class="result-title error">${message}</p>`;
  uploadResults.className = 'upload-results error';
  uploadResults.classList.remove('hidden');
}

// Initialize
checkAuth();
