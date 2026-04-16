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

# ================= CONFIG =================
BASE_SAVE_DIR = r"D:\LaserCapture\captures"
DEFAULT_CAM_INDEX = 0
DEFAULT_BACKEND_NAME = "MSMF"   # MSMF / DSHOW / DEFAULT

JPEG_QUALITY = 95
DELAY_AFTER_TRIGGER_MS = 0
DEBOUNCE_SEC = 0.4
PREVIEW_SECONDS = 10

# Assets (packed with exe)
LOGO_FILE = "bac_logo.png"
TRAY_ICON_FILE = "tray.ico"
APP_NAME = "LaserCapture"

# OCR Upload API - ERP Integration endpoint
OCR_API_BASE = "http://65.2.187.3:8000"
OCR_UPLOAD_API = f"{OCR_API_BASE}/api/erp/upload-and-process"
# ==========================================

BACKENDS = {
    "DEFAULT": None,
    "DSHOW": cv2.CAP_DSHOW,
    "MSMF": cv2.CAP_MSMF,
}

# ---------- Helpers ----------
def today_folder():
    d = datetime.now().strftime("%Y-%m-%d")
    p = os.path.join(BASE_SAVE_DIR, d)
    os.makedirs(p, exist_ok=True)
    return p

def open_folder(path: str):
    try:
        os.startfile(path)
    except Exception:
        pass

def resource_path(filename: str) -> str:
    try:
        base = sys._MEIPASS  # type: ignore[attr-defined]
        return os.path.join(base, filename)
    except Exception:
        return os.path.join(os.path.dirname(__file__), filename)

def open_camera(index: int, backend_name: str):
    backend = BACKENDS.get(backend_name)
    if backend is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)

def list_dshow_device_names():
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        return graph.get_input_devices()
    except Exception:
        return []

def scan_cameras(max_index: int = 10, backend_name: str = "MSMF"):
    dshow_names = list_dshow_device_names()
    found = []

    for i in range(max_index):
        cap = open_camera(i, backend_name)
        if cap.isOpened():
            ok, frame = cap.read()

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or (frame.shape[1] if ok else 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or (frame.shape[0] if ok else 0)
            fps = cap.get(cv2.CAP_PROP_FPS)

            cap.release()

            name = dshow_names[i] if i < len(dshow_names) else f"Camera {i}"
            if w and h:
                label = f"[{i}] {name} [{backend_name}] ({w}x{h}, {fps:.0f}fps)"
            else:
                label = f"[{i}] {name} [{backend_name}]"

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

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

        if time.time() - start >= seconds:
            break

    cap.release()
    cv2.destroyAllWindows()
    return ok_any

def upload_image_async(filepath: str, tag_id: str, expected_huid: str = ""):
    """Asynchronously upload image to OCR API for processing with tag_id."""
    def do_upload():
        try:
            with open(filepath, "rb") as f:
                files = {"file": (os.path.basename(filepath), f, "image/jpeg")}
                data = {
                    "tag_id": tag_id,
                    "expected_huid": expected_huid
                }

                response = requests.post(
                    OCR_UPLOAD_API,
                    files=files,
                    data=data,
                    timeout=60
                )
                response.raise_for_status()

                result = response.json()
                decision = result.get("decision", "unknown")
                huid_match = result.get("huid_match", False)
                actual_huid = result.get("actual_huid", "")
                confidence = result.get("confidence", 0)

                match_status = "MATCH" if huid_match else "MISMATCH"
                print(f"[{tag_id}] OCR Result: {decision.upper()} | "
                      f"HUID: {actual_huid} ({match_status}) | "
                      f"Confidence: {confidence*100:.1f}%")

        except requests.exceptions.RequestException as e:
            print(f"[{tag_id}] Upload failed: {e}")
        except Exception as e:
            print(f"[{tag_id}] Error: {e}")

    threading.Thread(target=do_upload, daemon=True).start()

# ---------- Splash Screen ----------
class Splash:
    def __init__(self, default_cam_index=0):
        self.selected_cam = default_cam_index
        self.selected_backend = DEFAULT_BACKEND_NAME
        self.start_clicked = False
        self._cam_map = {}

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        w, h = 620, 320
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.configure(bg="#0B1220")

        frame = tk.Frame(self.root, bg="#0B1220")
        frame.pack(fill="both", expand=True, padx=18, pady=18)

        try:
            img = Image.open(resource_path(LOGO_FILE)).convert("RGBA").resize((78, 78))
            self.logo_img = ImageTk.PhotoImage(img)
            tk.Label(frame, image=self.logo_img, bg="#0B1220").grid(
                row=0, column=0, rowspan=4, padx=(0, 16), sticky="n"
            )
        except Exception:
            self.logo_img = None

        tk.Label(
            frame,
            text="BAC Laser Capture",
            fg="#E0C37E",
            bg="#0B1220",
            font=("Segoe UI", 15, "bold")
        ).grid(row=0, column=1, sticky="w")

        self.status = tk.Label(
            frame,
            text="Select Camera, Backend and Start",
            fg="#C7D2FE",
            bg="#0B1220",
            font=("Segoe UI", 11)
        )
        self.status.grid(row=1, column=1, sticky="w", pady=(6, 12))

        cam_row = tk.Frame(frame, bg="#0B1220")
        cam_row.grid(row=2, column=1, sticky="w", pady=(0, 10))

        tk.Label(
            cam_row,
            text="Camera:",
            fg="#8BA3FF",
            bg="#0B1220",
            font=("Segoe UI", 10)
        ).pack(side="left", padx=(0, 8))

        self.cam_var = tk.StringVar(value="")
        self.cam_combo = ttk.Combobox(
            cam_row,
            textvariable=self.cam_var,
            width=42,
            state="readonly"
        )
        self.cam_combo.pack(side="left")

        self.scan_btn = ttk.Button(cam_row, text="Scan Cameras", command=self._scan)
        self.scan_btn.pack(side="left", padx=(10, 0))

        backend_row = tk.Frame(frame, bg="#0B1220")
        backend_row.grid(row=3, column=1, sticky="w", pady=(0, 10))

        tk.Label(
            backend_row,
            text="Backend:",
            fg="#8BA3FF",
            bg="#0B1220",
            font=("Segoe UI", 10)
        ).pack(side="left", padx=(0, 8))

        self.backend_var = tk.StringVar(value=DEFAULT_BACKEND_NAME)
        self.backend_combo = ttk.Combobox(
            backend_row,
            textvariable=self.backend_var,
            width=12,
            state="readonly",
            values=["MSMF", "DSHOW", "DEFAULT"]
        )
        self.backend_combo.pack(side="left")

        btn_row = tk.Frame(frame, bg="#0B1220")
        btn_row.grid(row=4, column=1, sticky="w", pady=(14, 0))

        self.start_btn = ttk.Button(btn_row, text="Start", command=self._start)
        self.start_btn.pack(side="left")

        self.preview_btn = ttk.Button(btn_row, text="Preview", command=self._preview)
        self.preview_btn.pack(side="left", padx=(10, 0))

        self.exit_btn = ttk.Button(btn_row, text="Exit", command=self._exit)
        self.exit_btn.pack(side="left", padx=(10, 0))

        self.dots = tk.Label(
            frame,
            text="",
            fg="#8BA3FF",
            bg="#0B1220",
            font=("Segoe UI", 10)
        )
        self.dots.grid(row=5, column=1, sticky="w", pady=(16, 0))

        self._dot_count = 0
        self._running = True
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

        values = [label for _, label in cams]
        self._cam_map = {label: idx for idx, label in cams}
        self.cam_combo["values"] = values

        preferred_label = None
        for idx, label in cams:
            if idx == self.selected_cam:
                preferred_label = label
                break

        self.cam_var.set(preferred_label or values[0])
        self.set_status(f"Camera list updated with {backend_name} ✅")

    def _start(self):
        label = self.cam_var.get().strip()
        if not label:
            self.set_status("Select a camera first ❗")
            return

        self.selected_cam = self._cam_map.get(label, 0)
        self.selected_backend = self.backend_var.get().strip() or DEFAULT_BACKEND_NAME
        self.start_clicked = True
        self._running = False
        self.root.destroy()

    def _preview(self):
        label = self.cam_var.get().strip()
        if not label:
            self.set_status("Select a camera first ❗")
            return

        idx = self._cam_map.get(label, 0)
        backend_name = self.backend_var.get().strip() or DEFAULT_BACKEND_NAME

        self.set_status(f"Opening preview for Camera {idx} [{backend_name}]…")
        ok = preview_camera(idx, backend_name, seconds=PREVIEW_SECONDS)

        if ok:
            self.set_status("Preview closed ✅ Select correct camera & Start")
        else:
            self.set_status("Preview failed ❌ Camera busy / backend mismatch")

    def _exit(self):
        self.start_clicked = False
        self._running = False
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

# ---------- Main capture ----------
latest_frame = None
frame_lock = threading.Lock()
running = True
last_capture_time = 0

def reader_loop(cap):
    global latest_frame, running

    for _ in range(25):
        cap.read()
        time.sleep(0.01)

    while running:
        ok, frame = cap.read()
        if ok:
            with frame_lock:
                latest_frame = frame
        else:
            time.sleep(0.01)

def save_image(cap, expected_huid: str = ""):
    """Capture image and upload to OCR API with tag_id."""
    global last_capture_time, latest_frame

    now = time.time()
    if now - last_capture_time < DEBOUNCE_SEC:
        return
    last_capture_time = now

    if DELAY_AFTER_TRIGGER_MS > 0:
        time.sleep(DELAY_AFTER_TRIGGER_MS / 1000.0)

    for _ in range(3):
        cap.grab()
    ok, frame = cap.retrieve()

    if not ok:
        with frame_lock:
            frame = latest_frame

    if frame is None:
        print("No frame available.")
        return

    folder = today_folder()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    tag_id = f"TAG_{ts}"
    filepath = os.path.join(folder, f"{tag_id}.jpg")

    cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    print(f"Saved: {filepath}")

    # Upload to OCR API with tag_id
    upload_image_async(filepath, tag_id, expected_huid)

# ---------- Tray Icon ----------
tray_icon = None

def tray_setup():
    global tray_icon
    try:
        icon_path = resource_path(TRAY_ICON_FILE)
        img = Image.open(icon_path)
    except Exception:
        img = Image.new("RGBA", (64, 64), (11, 18, 32, 255))

    def on_open_today(icon, item):
        open_folder(today_folder())

    def on_open_base(icon, item):
        os.makedirs(BASE_SAVE_DIR, exist_ok=True)
        open_folder(BASE_SAVE_DIR)

    def on_exit(icon, item):
        global running
        running = False
        try:
            icon.stop()
        except Exception:
            pass

    menu = pystray.Menu(
        pystray.MenuItem("Open Today Folder", on_open_today),
        pystray.MenuItem("Open Captures Folder", on_open_base),
        pystray.MenuItem("Exit", on_exit),
    )

    tray_icon = pystray.Icon(APP_NAME, img, "BAC Laser Capture", menu)
    tray_icon.run()

# ---------- Keyboard listener ----------
def start_listener(cap):
    def on_press(key):
        if key == kb.Key.f9:
            save_image(cap)
        if key == kb.Key.esc:
            return False

    with kb.Listener(on_press=on_press) as listener:
        listener.join()

# ---------- App start ----------
def main():
    global running

    splash = Splash(default_cam_index=DEFAULT_CAM_INDEX)
    splash.root.mainloop()

    if not splash.start_clicked:
        return

    cam_index = splash.selected_cam
    backend_name = splash.selected_backend

    cap = open_camera(cam_index, backend_name)
    if not cap.isOpened():
        raise RuntimeError(
            f"Camera not opened on index {cam_index} with backend {backend_name}. "
            f"Try MSMF / DSHOW / DEFAULT and close other camera apps."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tray_thread = threading.Thread(target=tray_setup, daemon=True)
    tray_thread.start()

    t = threading.Thread(target=reader_loop, args=(cap,), daemon=True)
    t.start()

    start_listener(cap)

    running = False
    time.sleep(0.1)
    cap.release()

    try:
        if tray_icon:
            tray_icon.stop()
    except Exception:
        pass

if __name__ == "__main__":
    main()