/**
 * Hallmark QC Portal Uploader - Content Script
 *
 * Handles the multi-page Manakonline portal workflow:
 * 1. Login page: https://newmanak.uat.dcservices.in/MANAK/eBISLogin
 * 2. Dashboard: https://newmanak.uat.dcservices.in/MANAK/JewellerHM
 * 3. Job List: https://newmanak.uat.dcservices.in/MANAK/getListForUploadImage
 * 4. Tag List: https://newmanak.uat.dcservices.in/MANAK/getHuidListUploadImage
 * 5. Upload Page: https://newmanak.uat.dcservices.in/MANAK/getUploadImage
 */

console.log('Hallmark QC Portal Uploader: Content script loaded');
console.log('Current URL:', window.location.href);

// State variables
let statusBar = null;
let isProcessing = false;
let currentItem = null;

// Page type detection
function getPageType() {
  const url = window.location.href;
  const pathname = window.location.pathname;

  if (pathname.includes('eBISLogin') || pathname.includes('login')) {
    return 'login';
  }
  if (pathname.includes('JewellerHM') || pathname.includes('dashboard')) {
    return 'dashboard';
  }
  if (pathname.includes('getListForUploadImage')) {
    return 'job_list';
  }
  if (pathname.includes('getHuidListUploadImage')) {
    return 'tag_list';
  }
  if (pathname.includes('getUploadImage')) {
    return 'upload_page';
  }

  // Check for login form elements as fallback
  if (document.querySelector('input[name="username"]') ||
      document.querySelector('input[name="password"]') ||
      document.querySelector('#loginForm') ||
      document.querySelector('form[action*="login"]')) {
    return 'login';
  }

  return 'unknown';
}

// Debug: Log page type and elements
setTimeout(() => {
  const pageType = getPageType();
  console.log('=== DEBUG: Page Analysis ===');
  console.log('Page Type:', pageType);
  console.log('URL:', window.location.href);

  // Look for logout elements
  const allElements = document.querySelectorAll('a, button, input[type="submit"], span, div');
  let foundLogout = false;
  allElements.forEach(el => {
    const text = el.textContent?.toLowerCase() || '';
    const href = el.getAttribute?.('href')?.toLowerCase() || '';
    const onclick = el.getAttribute?.('onclick')?.toLowerCase() || '';
    if (text.includes('logout') || text.includes('log out') || text.includes('sign out') ||
        href.includes('logout') || onclick.includes('logout')) {
      console.log('Found logout element:', el.tagName, el.className, el.id, el.textContent?.slice(0, 50));
      foundLogout = true;
    }
  });

  if (!foundLogout) {
    console.log('No logout elements found');
  }
  console.log('=== END DEBUG ===');
}, 2000);

// Detect login state on page load
detectLoginState();

// Watch for navigation changes (SPA behavior)
const observer = new MutationObserver(() => {
  detectLoginState();
});
observer.observe(document.body, { childList: true, subtree: true });

// Message handler
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message).then(sendResponse);
  return true;
});

async function handleMessage(message) {
  console.log('Content script received message:', message.type);

  switch (message.type) {
    case 'START_PROCESSING':
    case 'PROCESS_NEXT':
      return await processItem(message.item);

    case 'CHECK_LOGIN':
      return { isLoggedIn: isLoggedIn() };

    case 'GET_PAGE_TYPE':
      return { pageType: getPageType() };

    case 'NAVIGATE_TO_UPLOAD':
      return await navigateToUploadSection();

    default:
      return { success: false, error: 'Unknown message type' };
  }
}

// Detect if user is logged in
function detectLoginState() {
  const loggedIn = isLoggedIn();
  const pageType = getPageType();

  console.log('Login state:', loggedIn, 'Page type:', pageType);

  chrome.runtime.sendMessage({
    type: loggedIn ? 'LOGIN_DETECTED' : 'LOGOUT_DETECTED',
    pageType: pageType
  });

  if (loggedIn) {
    showStatusBar();
  } else {
    hideStatusBar();
  }
}

function isLoggedIn() {
  const pageType = getPageType();

  // If on login page, not logged in
  if (pageType === 'login') {
    return false;
  }

  // Check for logout button or user menu
  const logoutSelectors = [
    'a[href*="logout"]',
    'a[href*="Logout"]',
    'a[href*="signout"]',
    'button[onclick*="logout"]',
    'button[onclick*="Logout"]',
    '.logout-btn',
    '#logout',
    '#btnLogout',
    '[title*="Logout"]',
    '[title*="Log out"]',
    '.btn-logout',
    '#lnkLogout'
  ];

  const userMenuSelectors = [
    '.user-menu',
    '.user-info',
    '#userInfo',
    '.navbar-user',
    '.user-name',
    '.username',
    '#userName',
    '.profile-info',
    '[class*="userInfo"]',
    '[class*="user-profile"]',
    '.welcome-user'
  ];

  // Check for logout elements
  let logoutBtn = null;
  for (const selector of logoutSelectors) {
    try {
      logoutBtn = document.querySelector(selector);
      if (logoutBtn) {
        console.log('Found logout element with selector:', selector);
        break;
      }
    } catch (e) {
      // Invalid selector, skip
    }
  }

  // Also check by text content
  if (!logoutBtn) {
    const allLinks = document.querySelectorAll('a, button');
    for (const el of allLinks) {
      const text = (el.textContent || '').toLowerCase().trim();
      if (text === 'logout' || text === 'log out' || text === 'sign out' || text === 'signout') {
        logoutBtn = el;
        console.log('Found logout by text content:', el.textContent);
        break;
      }
    }
  }

  // Check for user menu
  let userMenu = null;
  for (const selector of userMenuSelectors) {
    userMenu = document.querySelector(selector);
    if (userMenu) {
      console.log('Found user menu with selector:', selector);
      break;
    }
  }

  // Check for login form (indicates NOT logged in)
  const loginForm = document.querySelector('form[action*="login"], #loginForm, .login-form, input[name="username"]');

  console.log('Login detection - logoutBtn:', !!logoutBtn, 'userMenu:', !!userMenu, 'loginForm:', !!loginForm);

  // If login form is visible and no logout button, not logged in
  if (loginForm && !logoutBtn && !userMenu) {
    return false;
  }

  // If we're on a known portal page (job_list, tag_list, upload_page), assume logged in
  if (['job_list', 'tag_list', 'upload_page', 'dashboard'].includes(pageType)) {
    return true;
  }

  return !!(logoutBtn || userMenu);
}

// Navigate to the Upload Image section from dashboard
async function navigateToUploadSection() {
  const pageType = getPageType();
  console.log('navigateToUploadSection called, current page:', pageType);

  if (pageType === 'dashboard') {
    // Find and click on "Hallmarking" section first
    const hallmarkingLinks = document.querySelectorAll('a, div, span');
    for (const el of hallmarkingLinks) {
      const text = (el.textContent || '').toLowerCase();
      if (text.includes('hallmarking') || text.includes('upload image')) {
        console.log('Clicking on:', el.textContent);
        el.click();
        await sleep(1000);
        break;
      }
    }

    // Then find "Upload Image" tile and click "View"
    await sleep(1000);
    const viewButtons = document.querySelectorAll('a, button');
    for (const el of viewButtons) {
      const parentText = el.closest('.tile, .card, .box, div')?.textContent || '';
      if (parentText.toLowerCase().includes('upload image')) {
        const text = (el.textContent || '').toLowerCase();
        if (text === 'view' || text.includes('view')) {
          console.log('Clicking View for Upload Image');
          el.click();
          return { success: true, message: 'Navigating to Upload Image' };
        }
      }
    }

    return { success: false, error: 'Could not find Upload Image section' };
  }

  return { success: true, message: 'Already on correct page type: ' + pageType };
}

// Create and show status bar
function showStatusBar() {
  if (statusBar) return;

  statusBar = document.createElement('div');
  statusBar.id = 'hqc-status-bar';
  statusBar.innerHTML = `
    <div class="hqc-status-content">
      <span class="hqc-logo">HQC</span>
      <span class="hqc-status-text">Ready</span>
      <span class="hqc-current"></span>
      <span class="hqc-progress"></span>
      <button class="hqc-btn hqc-pause" style="display:none">Pause</button>
    </div>
  `;
  document.body.appendChild(statusBar);

  // Add event listener for pause button
  statusBar.querySelector('.hqc-pause').addEventListener('click', () => {
    chrome.runtime.sendMessage({ type: 'PAUSE_UPLOAD' });
  });
}

function hideStatusBar() {
  if (statusBar) {
    statusBar.remove();
    statusBar = null;
  }
}

function updateStatusBar(text, current = '', progress = '') {
  if (!statusBar) return;

  statusBar.querySelector('.hqc-status-text').textContent = text;
  statusBar.querySelector('.hqc-current').textContent = current;
  statusBar.querySelector('.hqc-progress').textContent = progress;

  const pauseBtn = statusBar.querySelector('.hqc-pause');
  pauseBtn.style.display = isProcessing ? 'inline-block' : 'none';
}

// Process a single item
async function processItem(item) {
  if (!item) {
    return { success: false, error: 'No item provided' };
  }

  isProcessing = true;
  currentItem = item;
  const pageType = getPageType();

  console.log('Processing item:', item, 'Current page type:', pageType);
  updateStatusBar('Processing...', item.tag_id, '');

  try {
    // Step 1: Navigate to the correct page based on current location
    if (pageType === 'dashboard') {
      updateStatusBar('Navigating to Upload Image...', item.tag_id, '');
      await navigateToUploadSection();
      await sleep(2000);
    }

    // Step 2: If on job list, find and click the job
    if (pageType === 'job_list' || getPageType() === 'job_list') {
      updateStatusBar('Finding job...', item.bis_job_no, '');
      const foundJob = await navigateToJob(item.bis_job_no);
      if (!foundJob) {
        throw new Error(`Could not find job: ${item.bis_job_no}`);
      }
      await sleep(2000);
    }

    // Step 3: If on tag list, find and click the tag row
    if (getPageType() === 'tag_list') {
      updateStatusBar('Finding tag...', item.tag_id, '');
      const foundTag = await navigateToTag(item.tag_id);
      if (!foundTag) {
        throw new Error(`Could not find tag: ${item.tag_id}`);
      }
      await sleep(2000);
    }

    // Step 4: If on upload page, perform the upload
    if (getPageType() === 'upload_page') {
      // Download image from S3
      updateStatusBar('Downloading image...', item.tag_id, '');
      const imageBlob = await downloadImage(item.image_url);
      if (!imageBlob) {
        throw new Error('Could not download image');
      }

      // Upload the image
      updateStatusBar('Uploading...', item.tag_id, '');
      const uploaded = await uploadImageOnPage(imageBlob, item);
      if (!uploaded) {
        throw new Error('Upload failed');
      }

      // Report success
      isProcessing = false;
      currentItem = null;
      updateStatusBar('Upload complete', item.tag_id, '');

      chrome.runtime.sendMessage({
        type: 'UPLOAD_COMPLETE',
        tag_id: item.tag_id,
        image_type: item.image_type
      });

      return { success: true };
    } else {
      // Not on upload page yet, need to continue navigation
      return { success: false, error: 'Navigation in progress, not yet on upload page' };
    }

  } catch (error) {
    console.error('Upload failed:', error);
    isProcessing = false;
    currentItem = null;
    updateStatusBar('Error', item.tag_id, error.message);

    chrome.runtime.sendMessage({
      type: 'UPLOAD_FAILED',
      tag_id: item.tag_id,
      image_type: item.image_type,
      error: error.message
    });

    return { success: false, error: error.message };
  }
}

// Navigate to a specific job on the job list page
async function navigateToJob(bisJobNo) {
  console.log('Looking for job:', bisJobNo);

  // Look for the job in the table
  const tables = document.querySelectorAll('table');
  for (const table of tables) {
    const rows = table.querySelectorAll('tr');
    for (const row of rows) {
      const cells = row.querySelectorAll('td');
      for (const cell of cells) {
        if (cell.textContent.includes(bisJobNo)) {
          console.log('Found job row:', bisJobNo);

          // Find View/Action link in this row
          const viewLink = row.querySelector('a[href*="getHuidListUploadImage"], a:contains("View"), button:contains("View")');
          if (viewLink) {
            viewLink.click();
            return true;
          }

          // Try clicking any link in the row
          const anyLink = row.querySelector('a');
          if (anyLink) {
            anyLink.click();
            return true;
          }

          // Try clicking the row itself
          row.click();
          return true;
        }
      }
    }
  }

  // Try finding by text content anywhere
  const allElements = document.querySelectorAll('*');
  for (const el of allElements) {
    if (el.children.length === 0 && el.textContent.trim() === bisJobNo) {
      // Found the job number, look for nearby View link
      const parent = el.closest('tr, .row, .item');
      if (parent) {
        const link = parent.querySelector('a, button');
        if (link) {
          link.click();
          return true;
        }
      }
    }
  }

  return false;
}

// Navigate to a specific tag on the tag list page
async function navigateToTag(tagId) {
  console.log('Looking for tag:', tagId);

  // Look for the tag in the table
  const tables = document.querySelectorAll('table');
  for (const table of tables) {
    const rows = table.querySelectorAll('tr');
    for (const row of rows) {
      const cells = row.querySelectorAll('td');
      for (const cell of cells) {
        if (cell.textContent.trim() === tagId || cell.textContent.includes(tagId)) {
          console.log('Found tag row:', tagId);

          // Find the link to upload page
          const uploadLink = row.querySelector('a[href*="getUploadImage"], a');
          if (uploadLink) {
            uploadLink.click();
            return true;
          }

          // Try clicking the row
          row.click();
          return true;
        }
      }
    }
  }

  return false;
}

// Download image from URL
async function downloadImage(url) {
  try {
    console.log('Downloading image from:', url);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.blob();
  } catch (error) {
    console.error('Image download failed:', error);
    return null;
  }
}

// Upload image on the upload page
async function uploadImageOnPage(imageBlob, item) {
  console.log('Uploading image on page for:', item.tag_id);

  // Find the file input
  let fileInput = document.querySelector('input[type="file"]');

  if (!fileInput) {
    // Look for Browse/Choose file button
    const browseButtons = document.querySelectorAll('button, a, input[type="button"]');
    for (const btn of browseButtons) {
      const text = (btn.textContent || btn.value || '').toLowerCase();
      if (text.includes('browse') || text.includes('choose') || text.includes('select')) {
        console.log('Clicking browse button:', btn.textContent || btn.value);
        btn.click();
        await sleep(500);
        break;
      }
    }

    fileInput = document.querySelector('input[type="file"]');
  }

  if (!fileInput) {
    console.error('Could not find file input');
    return false;
  }

  // Create a file from the blob
  const fileName = item.image_type === 'huid'
    ? `${item.tag_id}_HUID.jpg`
    : `${item.tag_id}.jpg`;

  const file = new File([imageBlob], fileName, { type: 'image/jpeg' });

  // Set the file on the input
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  fileInput.files = dataTransfer.files;

  // Trigger change event
  fileInput.dispatchEvent(new Event('change', { bubbles: true }));
  console.log('File set on input:', fileName);

  await sleep(1000);

  // Find and click the Upload button
  const uploadButtons = document.querySelectorAll('button, input[type="submit"], input[type="button"], a');
  for (const btn of uploadButtons) {
    const text = (btn.textContent || btn.value || '').toLowerCase();
    if (text.includes('upload') && !text.includes('choose')) {
      console.log('Clicking upload button:', btn.textContent || btn.value);
      btn.click();
      await sleep(2000);
      break;
    }
  }

  // Check for success/error indicators
  const errorIndicators = document.querySelectorAll('.error, .alert-danger, .upload-error, [class*="error"]');
  if (errorIndicators.length > 0) {
    const errorText = errorIndicators[0].textContent;
    console.error('Upload error detected:', errorText);
    return false;
  }

  // Check for success indicators
  const successIndicators = document.querySelectorAll('.success, .alert-success, [class*="success"]');
  if (successIndicators.length > 0) {
    console.log('Upload success detected');
    return true;
  }

  // Assume success if no error appeared
  console.log('No error detected, assuming success');
  return true;
}

// Utility functions
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForElement(selector, timeout = 5000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const element = document.querySelector(selector);
    if (element) return element;
    await sleep(100);
  }
  return null;
}
