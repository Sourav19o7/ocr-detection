// Camera capture component with device selection and live preview.
// Opens in a modal, shows live feed, and returns captured image as File.

import { modal } from "./modal.js";
import { toast } from "./toast.js";
import { refreshIcons } from "../lib/format.js";

let activeStream = null;
let cameraKeyHandler = null;

// Helper to get capture button (it's rendered after body, so we query dynamically)
function getCaptureBtn() {
  return document.querySelector("#camera-capture");
}

/**
 * Opens a camera capture modal with device selection and preview.
 * @param {Object} options
 * @param {function(File): void} options.onCapture - Called with the captured image File
 * @param {string} [options.filename] - Filename for the captured image (default: capture.jpg)
 * @returns {void}
 */
export function openCameraCapture({ onCapture, filename = "capture.jpg" } = {}) {
  modal.open({
    title: "Capture Image",
    width: 560,
    onClose: () => {
      stopStream();
      // Remove keyboard handler
      if (cameraKeyHandler) {
        document.removeEventListener("keydown", cameraKeyHandler);
        cameraKeyHandler = null;
      }
    },
    body: (container) => {
      container.innerHTML = `
        <div class="camera-container">
          <div class="camera-select-row">
            <label for="camera-select">Camera</label>
            <select id="camera-select" class="camera-select">
              <option value="">Detecting cameras...</option>
            </select>
          </div>
          <div class="camera-preview-wrap">
            <video id="camera-video" autoplay playsinline muted></video>
            <canvas id="camera-canvas" style="display:none"></canvas>
            <div id="camera-placeholder" class="camera-placeholder">
              <i data-lucide="camera-off"></i>
              <span>Initializing camera...</span>
            </div>
          </div>
          <div id="camera-error" class="camera-error hidden"></div>
        </div>
      `;
      refreshIcons(container);
      // Delay init slightly to ensure footer is rendered
      setTimeout(() => initCamera(container, onCapture, filename), 50);
    },
    footer: (container) => {
      container.innerHTML = `
        <button class="btn btn-tertiary" id="camera-cancel">Cancel <kbd>Esc</kbd></button>
        <button class="btn btn-primary" id="camera-capture" disabled>
          <i data-lucide="camera"></i> Capture <kbd>Space</kbd>
        </button>
      `;
      refreshIcons(container);
      container.querySelector("#camera-cancel").addEventListener("click", () => modal.close());
      container.querySelector("#camera-capture").addEventListener("click", () => {
        captureImage(onCapture, filename);
      });

      // Keyboard shortcut for capture (Space bar)
      cameraKeyHandler = (e) => {
        if (e.key === " " || e.key === "Spacebar") {
          e.preventDefault();
          const btn = getCaptureBtn();
          if (btn && !btn.disabled) {
            captureImage(onCapture, filename);
          }
        }
      };
      document.addEventListener("keydown", cameraKeyHandler);
    },
  });
}

async function initCamera(container, onCapture, filename) {
  const select = container.querySelector("#camera-select");
  const video = container.querySelector("#camera-video");
  const placeholder = container.querySelector("#camera-placeholder");
  const errorEl = container.querySelector("#camera-error");

  // Check if camera API is available
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError(errorEl, placeholder, "Camera API not supported in this browser");
    return;
  }

  try {
    // First request permission to access any camera to enumerate devices
    placeholder.querySelector("span").textContent = "Requesting camera access...";
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
    tempStream.getTracks().forEach(track => track.stop());

    // Now enumerate video input devices
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === "videoinput");

    if (videoDevices.length === 0) {
      showError(errorEl, placeholder, "No cameras found on this device");
      return;
    }

    // Populate select
    select.innerHTML = videoDevices.map((device, i) =>
      `<option value="${device.deviceId}">${device.label || `Camera ${i + 1}`}</option>`
    ).join("");

    // Start with first camera
    select.addEventListener("change", () => startStream(select.value, video, placeholder, errorEl));
    await startStream(videoDevices[0].deviceId, video, placeholder, errorEl);

  } catch (err) {
    console.error("Camera init error:", err);
    if (err.name === "NotAllowedError") {
      showError(errorEl, placeholder, "Camera access denied. Please allow camera permissions.");
    } else if (err.name === "NotFoundError") {
      showError(errorEl, placeholder, "No camera found on this device");
    } else {
      showError(errorEl, placeholder, `Camera error: ${err.message}`);
    }
  }
}

async function startStream(deviceId, video, placeholder, errorEl) {
  // Stop any existing stream
  stopStream();

  // Disable capture button while switching
  const captureBtn = getCaptureBtn();
  if (captureBtn) captureBtn.disabled = true;

  try {
    // Build constraints - don't mix exact deviceId with facingMode
    const videoConstraints = {
      width: { ideal: 1920 },
      height: { ideal: 1080 }
    };

    if (deviceId) {
      videoConstraints.deviceId = { exact: deviceId };
    } else {
      videoConstraints.facingMode = "environment"; // Prefer back camera on mobile
    }

    activeStream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints });
    video.srcObject = activeStream;
    video.style.display = "block";
    placeholder.style.display = "none";
    errorEl.classList.add("hidden");

    // Enable capture button once video is actually playing
    video.onloadedmetadata = () => {
      video.play().then(() => {
        const btn = getCaptureBtn();
        if (btn) btn.disabled = false;
      }).catch(err => {
        console.error("Video play error:", err);
        // Still try to enable button - autoplay might be restricted but video should work
        const btn = getCaptureBtn();
        if (btn) btn.disabled = false;
      });
    };

    // Also listen for canplay as a fallback
    video.oncanplay = () => {
      const btn = getCaptureBtn();
      if (btn && btn.disabled) btn.disabled = false;
    };

  } catch (err) {
    console.error("Stream error:", err);
    showError(errorEl, placeholder, `Could not start camera: ${err.message}`);
    const btn = getCaptureBtn();
    if (btn) btn.disabled = true;
  }
}

function stopStream() {
  if (activeStream) {
    activeStream.getTracks().forEach(track => track.stop());
    activeStream = null;
  }
}

function showError(errorEl, placeholder, message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
  placeholder.style.display = "flex";
  const video = document.querySelector("#camera-video");
  if (video) video.style.display = "none";
}

function captureImage(onCapture, filename) {
  const video = document.querySelector("#camera-video");
  const canvas = document.querySelector("#camera-canvas");

  if (!video || !canvas || !activeStream) {
    toast.error("Camera not ready");
    return;
  }

  // Check if video has valid dimensions
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    toast.error("Video not ready yet, please wait");
    return;
  }

  // Set canvas size to video dimensions
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Draw video frame to canvas
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert to blob and create File
  canvas.toBlob((blob) => {
    if (!blob) {
      toast.error("Failed to capture image");
      return;
    }

    const file = new File([blob], filename, { type: "image/jpeg" });

    // Stop stream and close modal first
    stopStream();
    modal.close();

    // Then call the callback
    if (typeof onCapture === "function") {
      onCapture(file);
    }
  }, "image/jpeg", 0.92);
}

/**
 * Opens a choice dialog: upload from files or capture from camera.
 * @param {Object} options
 * @param {string} options.accept - File input accept attribute (default: "image/*")
 * @param {boolean} options.multiple - Allow multiple file selection (default: false)
 * @param {function(File[]): void} options.onFiles - Called with selected/captured files
 * @param {string} [options.captureFilename] - Filename for captured image
 * @returns {void}
 */
export function openImageSourcePicker({ accept = "image/*", multiple = false, onFiles, captureFilename = "capture.jpg" } = {}) {
  modal.open({
    title: "Add Image",
    width: 400,
    body: (container) => {
      container.innerHTML = `
        <div class="image-source-picker">
          <button class="source-option" id="source-upload">
            <div class="source-icon"><i data-lucide="upload"></i></div>
            <div class="source-text">
              <strong>Upload from files</strong>
              <span>Select image from your device</span>
            </div>
          </button>
          <button class="source-option" id="source-camera">
            <div class="source-icon"><i data-lucide="camera"></i></div>
            <div class="source-text">
              <strong>Capture from camera</strong>
              <span>Take a photo using your camera</span>
            </div>
          </button>
        </div>
      `;
      refreshIcons(container);

      // Upload from files
      container.querySelector("#source-upload").addEventListener("click", () => {
        modal.close();
        const input = document.createElement("input");
        input.type = "file";
        input.accept = accept;
        input.multiple = multiple;
        input.addEventListener("change", () => {
          if (input.files && input.files.length > 0) {
            onFiles([...input.files]);
          }
        });
        input.click();
      });

      // Capture from camera
      container.querySelector("#source-camera").addEventListener("click", () => {
        modal.close();
        openCameraCapture({
          filename: captureFilename,
          onCapture: (file) => onFiles([file])
        });
      });
    },
    footer: (container) => {
      container.innerHTML = `<button class="btn btn-tertiary" id="picker-cancel">Cancel</button>`;
      container.querySelector("#picker-cancel").addEventListener("click", () => modal.close());
    },
  });
}
