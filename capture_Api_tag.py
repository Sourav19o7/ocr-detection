import os
import sys
import time
import threading
import cv2
import requests
from datetime import datetime
from pynput import keyboard as kb

# Splash UI
import tkinter as tk
from tkinter import ttk

# Tray icon
import pystray
from PIL import Image, ImageTk

# Windows focus control
import win32gui
import win32con

# ================= CONFIG =================
BASE_SAVE_DIR = r"D:\LaserCapture\captures"
DEFAULT_CAM_INDEX = 0
DEFAULT_BACKEND_NAME = "DSHOW"   # MSMF / DSHOW / DEFAULT

JPEG_QUALITY = 95
DELAY_AFTER_TRIGGER_MS = 0
DEBOUNCE_SEC = 0.4
PREVIEW_SECONDS = 10

LIVE_PREVIEW_WINDOW_NAME = "BAC Live Preview - Press Q to hide preview"

# Assets (packed with exe)
LOGO_FILE = "bac_logo.png"
TRAY_ICON_FILE = "tray.ico"
APP_NAME = "LaserCapture"

# OCR Upload API (S3 presigned URL for OCR processing)
OCR_UPLOAD_API = "https://65-2-187-3.nip.io/api/get-ocr-upload-url"

# ===== Hallmark QC API (for Manakonline upload queue) =====
HQC_API_BASE     = "https://65-2-187-3.nip.io"  # Production EC2 server
HQC_API_KEY      = "test-camera-key-2024"       # API key for external access
HQC_UPLOAD_TIMEOUT = 30

# ===== HUID API =====
HUID_API_BASE            = "http://103.133.214.232:8026/api/grn/HUID_API"
HUID_API_KEY             = "9f8c7a6b5e4d3c2b1a"  # BAC API key
BRANCH_NAME              = "Hosur"
HUID_API_TIMEOUT_SEC     = 10
HUID_RESPONSE_TAGID_KEY  = "Tagid"
HUID_RESPONSE_JOBNO_KEY  = "Jobno"

# ===== Marking software automation =====
MARKING_WINDOW_TITLE_KEYWORD = "EzCad-Lite"
PREVIEW_OFF_WAIT_SEC         = 1.0
PREVIEW_ON_WAIT_SEC          = 5.0
AUTOMATION_COOLDOWN_SEC      = 1.5

# ===== Debug log config =====
DEBUG_LOG_DIR       = r"D:\LaserCapture\logs"  # log files saved here
DEBUG_LOG_MAX_LINES = 300                       # max lines kept on-screen
DEBUG_WINDOW_W      = 560
DEBUG_WINDOW_H      = 300
DEBUG_WINDOW_ALPHA  = 0.92                      # 0.0=transparent … 1.0=solid

BACKENDS = {
    "DEFAULT": None,
    "DSHOW":   cv2.CAP_DSHOW,
    "MSMF":    cv2.CAP_MSMF,
}


# ===========================================================
#  DEBUG LOGGER
#  - Floating always-on-top tkinter window (bottom-right)
#  - Colour-coded lines: INFO / OK / WARN / ERROR
#  - Every message also written to  logs/YYYY-MM-DD.log
#  - Thread-safe: call log() from any thread at any time
# ===========================================================

class DebugLog:
    LEVELS = {
        "INFO":  ("[INFO ]", "#A8C8FF"),   # blue
        "OK":    ("[OK   ]", "#6EE7B7"),   # green
        "WARN":  ("[WARN ]", "#FCD34D"),   # amber
        "ERROR": ("[ERROR]", "#F87171"),   # red
    }

    def __init__(self):
        self._lines  = []
        self._lock   = threading.Lock()
        self._root   = None
        self._text   = None
        self._log_fh = None
        self._open_log_file()

    # ── file helpers ─────────────────────────────────────────

    def _open_log_file(self):
        try:
            os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
            fname       = datetime.now().strftime("%Y-%m-%d") + ".log"
            path        = os.path.join(DEBUG_LOG_DIR, fname)
            self._log_fh = open(path, "a", encoding="utf-8", buffering=1)
            self._log_fh.write(
                f"\n{'='*60}\n"
                f"  Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'='*60}\n"
            )
        except Exception as e:
            print(f"[DebugLog] Cannot open log file: {e}")

    def _write_file(self, line: str):
        try:
            if self._log_fh:
                self._log_fh.write(line + "\n")
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────

    def log(self, message: str, level: str = "INFO"):
        level = level.upper()
        if level not in self.LEVELS:
            level = "INFO"
        ts   = datetime.now().strftime("%H:%M:%S")
        tag  = self.LEVELS[level][0]
        line = f"{ts} {tag} {message}"
        self._write_file(line)
        with self._lock:
            self._lines.append((ts, level, message))
            if len(self._lines) > DEBUG_LOG_MAX_LINES:
                self._lines.pop(0)
        if self._root:
            try:
                self._root.after(0, self._refresh_text)
            except Exception:
                pass

    def info(self,  msg): self.log(msg, "INFO")
    def ok(self,    msg): self.log(msg, "OK")
    def warn(self,  msg): self.log(msg, "WARN")
    def error(self, msg): self.log(msg, "ERROR")

    # ── window ────────────────────────────────────────────────

    def start_window(self):
        """Build and run the floating debug window (dedicated thread)."""
        self._root = tk.Tk()
        root = self._root

        root.title("LaserCapture – Debug Log")
        root.configure(bg="#0D1117")
        root.attributes("-topmost", True)
        root.attributes("-alpha", DEBUG_WINDOW_ALPHA)
        root.resizable(True, True)

        # Position: bottom-right corner of screen
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x  = sw - DEBUG_WINDOW_W - 16
        y  = sh - DEBUG_WINDOW_H - 50
        root.geometry(f"{DEBUG_WINDOW_W}x{DEBUG_WINDOW_H}+{x}+{y}")

        # ── header bar ──────────────────────────────────────
        hdr = tk.Frame(root, bg="#161B22", height=28)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text="  BAC LaserCapture  |  Debug Log",
            bg="#161B22", fg="#8BA3FF", font=("Consolas", 9, "bold")
        ).pack(side="left", pady=4)

        tk.Button(
            hdr, text="Open Logs", bg="#21262D", fg="#8BA3FF",
            relief="flat", font=("Consolas", 8), cursor="hand2", padx=6,
            command=lambda: (
                os.makedirs(DEBUG_LOG_DIR, exist_ok=True),
                os.startfile(DEBUG_LOG_DIR)
            )
        ).pack(side="right", padx=4, pady=3)

        tk.Button(
            hdr, text="Clear", bg="#21262D", fg="#8BA3FF",
            relief="flat", font=("Consolas", 8), cursor="hand2", padx=6,
            command=self._clear
        ).pack(side="right", padx=0, pady=3)

        # ── scrollable text area ─────────────────────────────
        frame     = tk.Frame(root, bg="#0D1117")
        frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(frame, bg="#21262D", troughcolor="#161B22",
                                 relief="flat", width=8)
        scrollbar.pack(side="right", fill="y")

        self._text = tk.Text(
            frame,
            bg="#0D1117", fg="#C9D1D9",
            font=("Consolas", 9),
            state="disabled", wrap="word",
            relief="flat", padx=6, pady=4,
            yscrollcommand=scrollbar.set,
            insertbackground="#C9D1D9",
            selectbackground="#264F78",
        )
        self._text.pack(fill="both", expand=True)
        scrollbar.config(command=self._text.yview)

        # colour tags
        for lvl, (_, colour) in self.LEVELS.items():
            self._text.tag_config(lvl, foreground=colour)
        self._text.tag_config("TS", foreground="#555E6B")  # dim timestamp

        # dump any lines already buffered before window existed
        self._refresh_text()

        # draggable header
        self._dx = self._dy = 0

        def drag_start(e):
            self._dx = e.x_root - root.winfo_x()
            self._dy = e.y_root - root.winfo_y()

        def drag_move(e):
            root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

        hdr.bind("<ButtonPress-1>", drag_start)
        hdr.bind("<B1-Motion>",     drag_move)

        root.mainloop()

    def _refresh_text(self):
        if not self._text:
            return
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        with self._lock:
            lines = list(self._lines)
        for ts, level, message in lines:
            self._text.insert("end", ts + " ", "TS")
            self._text.insert("end", self.LEVELS[level][0] + " ", level)
            self._text.insert("end", message + "\n")
        self._text.config(state="disabled")
        self._text.see("end")

    def _clear(self):
        with self._lock:
            self._lines.clear()
        self._refresh_text()

    def close(self):
        try:
            if self._log_fh:
                self._log_fh.write(
                    f"  Session ended:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                self._log_fh.close()
        except Exception:
            pass
        try:
            if self._root:
                self._root.destroy()
        except Exception:
            pass


# ── global logger ──────────────────────────────────────────
dlog = DebugLog()

def log(message: str, level: str = "INFO"):
    """Use everywhere instead of print()."""
    dlog.log(message, level)
    print(message)          # also visible when running from cmd / terminal


# ===========================================================
#  Helpers
# ===========================================================

def today_folder():
    d = datetime.now().strftime("%Y-%m-%d")
    p = os.path.join(BASE_SAVE_DIR, d)
    os.makedirs(p, exist_ok=True)
    return p

def jobno_folder(jobno: str) -> str:
    p = os.path.join(BASE_SAVE_DIR, str(jobno))
    os.makedirs(p, exist_ok=True)
    return p

def open_folder(path: str):
    try:
        os.startfile(path)
    except Exception:
        pass

def resource_path(filename: str) -> str:
    try:
        base = sys._MEIPASS          # type: ignore[attr-defined]
        return os.path.join(base, filename)
    except Exception:
        return os.path.join(os.path.dirname(__file__), filename)

def open_camera(index: int, backend_name: str):
    backend = BACKENDS.get(backend_name)
    return cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)

def list_dshow_device_names():
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return []

def scan_cameras(max_index: int = 10, backend_name: str = "MSMF"):
    dshow_names = list_dshow_device_names()
    found = []
    for i in range(max_index):
        cap = open_camera(i, backend_name)
        if cap.isOpened():
            ok, frame = cap.read()
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or (frame.shape[1] if ok else 0)
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or (frame.shape[0] if ok else 0)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            name  = dshow_names[i] if i < len(dshow_names) else f"Camera {i}"
            label = (
                f"[{i}] {name} [{backend_name}] ({w}x{h}, {fps:.0f}fps)"
                if w and h else f"[{i}] {name} [{backend_name}]"
            )
            found.append((i, label))
    return found

def preview_camera(index: int, backend_name: str, seconds: int = 5):
    cap = open_camera(index, backend_name)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    start = time.time()
    ok_any = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ok_any = True
        cv2.imshow(f"Preview - Camera {index} [{backend_name}] (press Q to close)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
        if time.time() - start >= seconds:
            break
    cap.release()
    cv2.destroyAllWindows()
    return ok_any


# ===========================================================
#  HUID API
# ===========================================================

def fetch_huid_metadata(jobno: str = None, tagid: str = None) -> dict:
    params = {"branchName": BRANCH_NAME}
    if jobno:
        params["Jobno"] = jobno
    if tagid:
        params["Tagid"] = tagid

    log(f"HUID API call → {params}", "INFO")

    # Build headers with API key for authentication
    headers = {}
    if HUID_API_KEY:
        headers["x-api-key"] = HUID_API_KEY

    try:
        resp = requests.get(HUID_API_BASE, params=params, headers=headers, timeout=HUID_API_TIMEOUT_SEC)
        log(f"HUID API HTTP {resp.status_code}", "INFO")
        resp.raise_for_status()
        data   = resp.json()
        record = data[0] if isinstance(data, list) and data else data

        fetched_tagid = (
            record.get(HUID_RESPONSE_TAGID_KEY)
            or record.get("tagid") or record.get("TagId") or record.get("tag_id")
        )
        fetched_jobno = (
            record.get(HUID_RESPONSE_JOBNO_KEY)
            or record.get("jobno") or record.get("JobNo") or record.get("job_no")
        )

        if fetched_tagid and fetched_jobno:
            log(f"HUID OK → Tagid: {fetched_tagid}  Jobno: {fetched_jobno}", "OK")
            return {"tagid": str(fetched_tagid), "jobno": str(fetched_jobno),
                    "raw": record, "success": True}
        else:
            log(f"HUID response missing fields. Raw: {data}", "WARN")

    except requests.exceptions.Timeout:
        log(f"HUID API timed out after {HUID_API_TIMEOUT_SEC}s", "ERROR")
    except requests.exceptions.ConnectionError as e:
        log(f"HUID API connection error: {e}", "ERROR")
    except Exception as e:
        log(f"HUID API failed: {e}", "ERROR")

    # Fallback: use timestamp-based filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    log("Fallback: using timestamp-based filename", "WARN")
    return {"tagid": f"mark_{ts}", "jobno": datetime.now().strftime("%Y-%m-%d"),
            "raw": {}, "success": False}


# ===========================================================
#  S3 Upload
# ===========================================================

def upload_image_async(filepath: str, unique_id: str):
    def do_upload():
        try:
            filename = f"{unique_id}.jpg"
            log(f"S3 upload → requesting presigned URL for {filename}", "INFO")
            resp = requests.post(
                OCR_UPLOAD_API,
                json={"fileName": filename, "contentType": "image/jpeg"},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            resp.raise_for_status()
            upload_url = resp.json().get("uploadUrl")

            if upload_url:
                with open(filepath, "rb") as f:
                    data = f.read()
                put = requests.put(upload_url, data=data,
                                   headers={"Content-Type": "image/jpeg"}, timeout=60)
                put.raise_for_status()
                log(f"S3 upload OK: {filename}", "OK")
            else:
                log(f"S3: no uploadUrl in response for {filename}", "WARN")

        except requests.exceptions.Timeout:
            log(f"S3 upload timed out for {unique_id}", "ERROR")
        except Exception as e:
            log(f"S3 upload failed: {e}", "ERROR")

    threading.Thread(target=do_upload, daemon=True).start()


def upload_to_hqc_api(filepath: str, tag_id: str, bis_job_no: str, image_type: str = "huid", branch: str = None):
    """
    Upload image to Hallmark QC API for Manakonline upload queue.

    Args:
        filepath: Path to the image file
        tag_id: AHC Tag ID (e.g., "570001123111")
        bis_job_no: BIS Job Number (e.g., "100166462")
        image_type: "article" or "huid" (default: "huid")
        branch: Branch name (optional)
    """
    def do_hqc_upload():
        try:
            log(f"HQC API → uploading {tag_id} ({image_type}) to queue", "INFO")

            with open(filepath, "rb") as f:
                files = {"file": (os.path.basename(filepath), f, "image/jpeg")}
                data = {
                    "tag_id": tag_id,
                    "job_no": bis_job_no,
                }
                if branch:
                    data["branch"] = branch

                headers = {}
                if HQC_API_KEY:
                    headers["x-api-key"] = HQC_API_KEY

                # Use huid-photo endpoint for HUID images, article-photo for article images
                endpoint = "/api/external/huid-photo" if image_type == "huid" else "/api/external/article-photo"
                resp = requests.post(
                    f"{HQC_API_BASE}{endpoint}",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=HQC_UPLOAD_TIMEOUT
                )
                resp.raise_for_status()
                result = resp.json()

                if result.get("success"):
                    log(f"HQC API OK: {tag_id} queued for Manakonline upload", "OK")
                else:
                    log(f"HQC API response: {result}", "WARN")

        except requests.exceptions.Timeout:
            log(f"HQC API timed out for {tag_id}", "ERROR")
        except requests.exceptions.ConnectionError:
            log(f"HQC API connection failed - is server running at {HQC_API_BASE}?", "ERROR")
        except Exception as e:
            log(f"HQC API upload failed: {e}", "ERROR")

    threading.Thread(target=do_hqc_upload, daemon=True).start()


# ===========================================================
#  Marking software focus
# ===========================================================

def find_window_by_title(keyword: str):
    matches = []
    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if keyword.lower() in title.lower():
                matches.append((hwnd, title))
    win32gui.EnumWindows(enum_handler, None)
    if matches:
        log(f"Window found: '{matches[0][1]}'", "INFO")
    else:
        log(f"Window not found for keyword: '{keyword}'", "WARN")
    return matches[0][0] if matches else None

import ctypes
import win32api
import win32con
import win32gui
import win32process

def bring_window_to_front(hwnd):
    try:
        # Get thread IDs
        fore_hwnd  = win32gui.GetForegroundWindow()
        target_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
        curr_tid   = ctypes.windll.kernel32.GetCurrentThreadId()
        fore_tid, _ = win32process.GetWindowThreadProcessId(fore_hwnd)

        # Attach our thread + foreground thread to the target thread
        # This grants permission to steal the foreground
        if target_tid != curr_tid:
            ctypes.windll.user32.AttachThreadInput(curr_tid,  target_tid, True)
            ctypes.windll.user32.AttachThreadInput(fore_tid, target_tid, True)

        # Restore if minimised
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.15)

        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.25)

        # Detach threads
        if target_tid != curr_tid:
            ctypes.windll.user32.AttachThreadInput(curr_tid,  target_tid, False)
            ctypes.windll.user32.AttachThreadInput(fore_tid, target_tid, False)

        log("Marking software focused", "OK")
        return True

    except Exception as e:
        log(f"Focus failed: {e}", "ERROR")
        return False

def send_key_to_foreground(key_obj):
    controller = kb.Controller()
    controller.press(key_obj)
    time.sleep(0.08)
    controller.release(key_obj)


# ===========================================================
#  Splash Screen
# ===========================================================

class Splash:
    def __init__(self, default_cam_index=0):
        self.selected_cam     = default_cam_index
        self.selected_backend = DEFAULT_BACKEND_NAME
        self.start_clicked    = False
        self._cam_map         = {}

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        w, h = 620, 320
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.root.configure(bg="#0B1220")

        frame = tk.Frame(self.root, bg="#0B1220")
        frame.pack(fill="both", expand=True, padx=18, pady=18)

        try:
            img = Image.open(resource_path(LOGO_FILE)).convert("RGBA").resize((78, 78))
            self.logo_img = ImageTk.PhotoImage(img)
            tk.Label(frame, image=self.logo_img, bg="#0B1220").grid(
                row=0, column=0, rowspan=4, padx=(0, 16), sticky="n")
        except Exception:
            self.logo_img = None

        tk.Label(frame, text="BAC Laser Capture", fg="#E0C37E", bg="#0B1220",
                 font=("Segoe UI", 15, "bold")).grid(row=0, column=1, sticky="w")

        self.status = tk.Label(frame, text="Select Camera, Backend and Start",
                               fg="#C7D2FE", bg="#0B1220", font=("Segoe UI", 11))
        self.status.grid(row=1, column=1, sticky="w", pady=(6, 12))

        cam_row = tk.Frame(frame, bg="#0B1220")
        cam_row.grid(row=2, column=1, sticky="w", pady=(0, 10))
        tk.Label(cam_row, text="Camera:", fg="#8BA3FF", bg="#0B1220",
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 8))
        self.cam_var   = tk.StringVar(value="")
        self.cam_combo = ttk.Combobox(cam_row, textvariable=self.cam_var,
                                      width=42, state="readonly")
        self.cam_combo.pack(side="left")
        ttk.Button(cam_row, text="Scan Cameras",
                   command=self._scan).pack(side="left", padx=(10, 0))

        backend_row = tk.Frame(frame, bg="#0B1220")
        backend_row.grid(row=3, column=1, sticky="w", pady=(0, 10))
        tk.Label(backend_row, text="Backend:", fg="#8BA3FF", bg="#0B1220",
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 8))
        self.backend_var   = tk.StringVar(value=DEFAULT_BACKEND_NAME)
        self.backend_combo = ttk.Combobox(backend_row, textvariable=self.backend_var,
                                          width=12, state="readonly",
                                          values=["MSMF", "DSHOW", "DEFAULT"])
        self.backend_combo.pack(side="left")

        btn_row = tk.Frame(frame, bg="#0B1220")
        btn_row.grid(row=4, column=1, sticky="w", pady=(14, 0))
        ttk.Button(btn_row, text="Start",   command=self._start).pack(side="left")
        ttk.Button(btn_row, text="Preview", command=self._preview).pack(side="left", padx=(10, 0))
        ttk.Button(btn_row, text="Exit",    command=self._exit).pack(side="left", padx=(10, 0))

        self.dots = tk.Label(frame, text="", fg="#8BA3FF", bg="#0B1220",
                             font=("Segoe UI", 10))
        self.dots.grid(row=5, column=1, sticky="w", pady=(16, 0))

        self._dot_count = 0
        self._running   = True
        self._animate()
        self._scan()
        self.root.update()

    def _scan(self):
        backend_name = self.backend_var.get().strip() or DEFAULT_BACKEND_NAME
        self.set_status(f"Scanning cameras with {backend_name}…")
        cams = scan_cameras(max_index=10, backend_name=backend_name)
        if not cams:
            self.cam_combo["values"] = []
            self.cam_var.set("")
            self.set_status(f"No camera found with {backend_name} ❌")
            return
        values        = [label for _, label in cams]
        self._cam_map = {label: idx for idx, label in cams}
        self.cam_combo["values"] = values
        preferred = next((lbl for idx, lbl in cams if idx == self.selected_cam), None)
        self.cam_var.set(preferred or values[0])
        self.set_status(f"Camera list updated with {backend_name} ✅")

    def _start(self):
        label = self.cam_var.get().strip()
        if not label:
            self.set_status("Select a camera first ❗")
            return
        self.selected_cam     = self._cam_map.get(label, 0)
        self.selected_backend = self.backend_var.get().strip() or DEFAULT_BACKEND_NAME
        self.start_clicked    = True
        self._running         = False
        self.root.destroy()

    def _preview(self):
        label = self.cam_var.get().strip()
        if not label:
            self.set_status("Select a camera first ❗")
            return
        idx          = self._cam_map.get(label, 0)
        backend_name = self.backend_var.get().strip() or DEFAULT_BACKEND_NAME
        self.set_status(f"Opening preview for Camera {idx} [{backend_name}]…")
        ok = preview_camera(idx, backend_name, seconds=PREVIEW_SECONDS)
        self.set_status(
            "Preview closed ✅ Select correct camera & Start" if ok
            else "Preview failed ❌ Camera busy / backend mismatch"
        )

    def _exit(self):
        self.start_clicked = False
        self._running      = False
        self.root.destroy()

    def set_status(self, text: str):
        self.status.config(text=text)
        self.root.update()

    def _animate(self):
        if not self._running:
            return
        self._dot_count = (self._dot_count + 1) % 4
        self.dots.config(text="Ready" + ("." * self._dot_count))
        self.root.after(350, self._animate)


# ===========================================================
#  Main capture engine
# ===========================================================

latest_frame         = None
frame_lock           = threading.Lock()
running              = True
last_capture_time    = 0
preview_running      = False
preview_thread       = None
last_automation_time = 0


def reader_loop(cap):
    global latest_frame, running
    for _ in range(25):          # flush initial stale frames
        cap.read()
        time.sleep(0.01)
    log("Camera reader loop ready", "OK")
    while running:
        ok, frame = cap.read()
        if ok:
            with frame_lock:
                latest_frame = frame
        else:
            time.sleep(0.01)


def preview_loop():
    global preview_running, latest_frame
    cv2.namedWindow(LIVE_PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_PREVIEW_WINDOW_NAME, 960, 540)
    log("Live preview window opened (press Q to hide)", "OK")
    while preview_running:
        frame_to_show = None
        with frame_lock:
            if latest_frame is not None:
                frame_to_show = latest_frame.copy()
        if frame_to_show is not None:
            cv2.imshow(LIVE_PREVIEW_WINDOW_NAME, frame_to_show)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            preview_running = False
            log("Live preview hidden by user (Q)", "INFO")
            break
        time.sleep(0.01)
    try:
        cv2.destroyWindow(LIVE_PREVIEW_WINDOW_NAME)
    except Exception:
        pass


def start_preview():
    global preview_running, preview_thread
    if preview_running:
        return
    preview_running = True
    preview_thread  = threading.Thread(target=preview_loop, daemon=True)
    preview_thread.start()


def stop_preview():
    global preview_running
    preview_running = False


# ===========================================================
#  Tag ID Input Dialog
# ===========================================================

def ask_tag_id() -> str:
    """
    Show a dialog to ask for Tag ID input.
    Returns the entered tag ID or None if cancelled.
    """
    result = {"value": None}

    def on_submit():
        value = entry.get().strip()
        if value:
            result["value"] = value
            dialog.destroy()
        else:
            entry.focus_set()

    def on_cancel():
        dialog.destroy()

    def on_key(event):
        if event.keysym == "Return":
            on_submit()
        elif event.keysym == "Escape":
            on_cancel()

    dialog = tk.Tk()
    dialog.title("Enter Tag ID")
    dialog.configure(bg="#0B1220")
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)

    # Center the dialog
    w, h = 400, 150
    sw, sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
    dialog.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    # Main frame
    frame = tk.Frame(dialog, bg="#0B1220", padx=20, pady=20)
    frame.pack(fill="both", expand=True)

    # Label
    tk.Label(
        frame,
        text="Enter Tag ID:",
        fg="#E0C37E",
        bg="#0B1220",
        font=("Segoe UI", 12, "bold")
    ).pack(anchor="w", pady=(0, 10))

    # Entry field
    entry = tk.Entry(
        frame,
        font=("Consolas", 14),
        bg="#1E293B",
        fg="#FFFFFF",
        insertbackground="#FFFFFF",
        relief="flat",
        width=30
    )
    entry.pack(fill="x", pady=(0, 15))
    entry.focus_set()
    entry.bind("<Key>", on_key)

    # Button frame
    btn_frame = tk.Frame(frame, bg="#0B1220")
    btn_frame.pack(fill="x")

    tk.Button(
        btn_frame,
        text="Submit",
        command=on_submit,
        bg="#22C55E",
        fg="#FFFFFF",
        font=("Segoe UI", 10, "bold"),
        relief="flat",
        padx=20,
        pady=5,
        cursor="hand2"
    ).pack(side="left", padx=(0, 10))

    tk.Button(
        btn_frame,
        text="Cancel",
        command=on_cancel,
        bg="#EF4444",
        fg="#FFFFFF",
        font=("Segoe UI", 10),
        relief="flat",
        padx=20,
        pady=5,
        cursor="hand2"
    ).pack(side="left")

    dialog.mainloop()

    return result["value"]


def save_image(cap):
    global last_capture_time, latest_frame

    now = time.time()
    if now - last_capture_time < DEBOUNCE_SEC:
        log("Capture skipped — debounce active", "WARN")
        return
    last_capture_time = now

    if DELAY_AFTER_TRIGGER_MS > 0:
        time.sleep(DELAY_AFTER_TRIGGER_MS / 1000.0)

    log("Grabbing frame from camera…", "INFO")
    for _ in range(3):
        cap.grab()
    ok, frame = cap.retrieve()

    if not ok:
        log("cap.retrieve() failed — using buffered frame", "WARN")
        with frame_lock:
            frame = latest_frame

    if frame is None:
        log("No frame available — capture aborted", "ERROR")
        return

    # Ask for Tag ID after capturing the image
    log("Waiting for Tag ID input...", "INFO")
    tag_id = ask_tag_id()

    if not tag_id:
        log("Tag ID input cancelled — image discarded", "WARN")
        return

    log(f"Tag ID entered: {tag_id}", "OK")

    # Use today's date as job number
    job_no = datetime.now().strftime("%Y-%m-%d")

    # Save the captured frame using the entered tag ID
    folder = jobno_folder(job_no)
    filename = f"{tag_id}.jpg"
    filepath = os.path.join(folder, filename)

    cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    log(f"Image saved: {filepath}", "OK")

    # Upload to Hallmark QC API using the entered tag ID
    upload_to_hqc_api(
        filepath=filepath,
        tag_id=tag_id,
        bis_job_no=job_no,
        image_type="huid",
        branch=BRANCH_NAME if BRANCH_NAME != "branchName" else None
    )


def automated_capture_flow(cap):
    global last_automation_time

    now = time.time()
    if now - last_automation_time < AUTOMATION_COOLDOWN_SEC:
        log("Automation cooldown — trigger ignored", "WARN")
        return
    last_automation_time = now

    log("F12 triggered — automated capture flow starting", "INFO")

    # Ask for Tag ID FIRST before doing anything
    log("Waiting for Tag ID input...", "INFO")
    tag_id = ask_tag_id()

    if not tag_id:
        log("Tag ID input cancelled — aborting", "WARN")
        return

    log(f"Tag ID entered: {tag_id}", "OK")

    # Now proceed with the capture flow
    hwnd = find_window_by_title(MARKING_WINDOW_TITLE_KEYWORD)
    if not hwnd:
        log(f"EZCAD2 window not found (keyword='{MARKING_WINDOW_TITLE_KEYWORD}')", "ERROR")
        return

    if not bring_window_to_front(hwnd):
        log("Could not focus marking software — aborting", "ERROR")
        return

    log("Sending ESC → EZCAD2 preview OFF", "INFO")
    send_key_to_foreground(kb.Key.esc)
    time.sleep(PREVIEW_OFF_WAIT_SEC)

    # Capture the image
    log("Capturing image...", "INFO")

    # Grab frame from camera
    global latest_frame, last_capture_time
    for _ in range(3):
        cap.grab()
    ok, frame = cap.retrieve()

    if not ok:
        log("cap.retrieve() failed — using buffered frame", "WARN")
        with frame_lock:
            frame = latest_frame

    if frame is None:
        log("No frame available — capture aborted", "ERROR")
        # Still send F1 to restore preview
        log("Sending F1 → EZCAD2 preview ON", "INFO")
        send_key_to_foreground(kb.Key.f1)
        return

    # Use today's date as job number
    job_no = datetime.now().strftime("%Y-%m-%d")

    # Save the captured frame using the entered tag ID
    folder = jobno_folder(job_no)
    filename = f"{tag_id}.jpg"
    filepath = os.path.join(folder, filename)

    last_capture_time = time.time()
    cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    log(f"Image saved: {filepath}", "OK")

    # Upload to Hallmark QC API using the entered tag ID
    upload_to_hqc_api(
        filepath=filepath,
        tag_id=tag_id,
        bis_job_no=job_no,
        image_type="huid",
        branch=BRANCH_NAME if BRANCH_NAME != "branchName" else None
    )

    log(f"Waiting {PREVIEW_ON_WAIT_SEC}s before sending F1…", "INFO")
    time.sleep(PREVIEW_ON_WAIT_SEC)

    log("Sending F1 → EZCAD2 preview ON", "INFO")
    send_key_to_foreground(kb.Key.f1)

    log("Automated cycle complete", "OK")


# ===========================================================
#  Tray Icon
# ===========================================================

tray_icon = None

def tray_setup():
    global tray_icon
    try:
        img = Image.open(resource_path(TRAY_ICON_FILE))
    except Exception:
        img = Image.new("RGBA", (64, 64), (11, 18, 32, 255))

    menu = pystray.Menu(
        pystray.MenuItem("Open Today Folder",    lambda i, m: open_folder(today_folder())),
        pystray.MenuItem("Open Captures Folder", lambda i, m: (
            os.makedirs(BASE_SAVE_DIR, exist_ok=True), open_folder(BASE_SAVE_DIR)
        )),
        pystray.MenuItem("Open Log Folder",      lambda i, m: (
            os.makedirs(DEBUG_LOG_DIR, exist_ok=True), open_folder(DEBUG_LOG_DIR)
        )),
        pystray.MenuItem("Exit", _tray_exit),
    )
    tray_icon = pystray.Icon(APP_NAME, img, "BAC Laser Capture", menu)
    tray_icon.run()

def _tray_exit(icon, item):
    global running
    log("Exit requested via tray icon", "INFO")
    running = False
    stop_preview()
    try:
        icon.stop()
    except Exception:
        pass


# ===========================================================
#  Keyboard listener
# ===========================================================
# ESC is intentionally NOT the exit key — the automated flow sends
# a synthetic ESC to EZCAD2 and pynput would catch it here.
# Safe exits: tray → Exit   OR   Ctrl+Shift+Q

def start_listener(cap):
    def on_press(key):
        if key == kb.Key.f9:
            log("F9 pressed — manual capture triggered", "INFO")
            save_image(cap)

        elif key == kb.Key.f12:
            log("F12 pressed — automated flow triggered", "INFO")
            threading.Thread(target=automated_capture_flow,
                             args=(cap,), daemon=True).start()

        elif key == kb.KeyCode.from_char('Q') and {kb.Key.ctrl, kb.Key.shift}.issubset(
            getattr(start_listener, '_pressed', set())
        ):
            log("Ctrl+Shift+Q pressed — exiting", "INFO")
            return False

    def on_press_track(key):
        if not hasattr(start_listener, '_pressed'):
            start_listener._pressed = set()
        start_listener._pressed.add(key)
        return on_press(key)

    def on_release_track(key):
        if hasattr(start_listener, '_pressed'):
            start_listener._pressed.discard(key)

    with kb.Listener(on_press=on_press_track, on_release=on_release_track) as listener:
        listener.join()


# ===========================================================
#  Entry point
# ===========================================================

def main():
    global running

    # Start debug window FIRST so all logs appear from the beginning
    debug_thread = threading.Thread(target=dlog.start_window, daemon=True)
    debug_thread.start()
    time.sleep(0.35)   # let the window render before splash appears

    log("BAC LaserCapture starting…", "INFO")
    log(f"Save dir : {BASE_SAVE_DIR}", "INFO")
    log(f"Log dir  : {DEBUG_LOG_DIR}", "INFO")
    log(f"HUID API : {HUID_API_BASE}", "INFO")
    log(f"Branch   : {BRANCH_NAME}", "INFO")

    splash = Splash(default_cam_index=DEFAULT_CAM_INDEX)
    splash.root.mainloop()

    if not splash.start_clicked:
        log("Cancelled at splash screen", "WARN")
        dlog.close()
        return

    cam_index    = splash.selected_cam
    backend_name = splash.selected_backend
    log(f"Opening camera {cam_index} with backend {backend_name}", "INFO")

    cap = open_camera(cam_index, backend_name)
    if not cap.isOpened():
        log(f"Camera {cam_index} failed to open ({backend_name})", "ERROR")
        dlog.close()
        raise RuntimeError(
            f"Camera not opened on index {cam_index} with backend {backend_name}."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    log("Camera opened — 1280x720", "OK")

    threading.Thread(target=tray_setup, daemon=True).start()
    threading.Thread(target=reader_loop, args=(cap,), daemon=True).start()

    start_preview()

    log("App ready — listening for keypress", "OK")
    log("F9  = manual capture", "INFO")
    log("F12 = automated flow (EZCAD2 → ESC → capture → F1)", "INFO")
    log("Tray → Exit  or  Ctrl+Shift+Q to quit", "INFO")

    start_listener(cap)

    # Cleanup
    log("Shutting down…", "INFO")
    running = False
    stop_preview()
    time.sleep(0.2)
    cap.release()
    log("Camera released — goodbye", "OK")

    try:
        if tray_icon:
            tray_icon.stop()
    except Exception:
        pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    dlog.close()


if __name__ == "__main__":
    main()