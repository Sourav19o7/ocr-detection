# Hallmark OCR Detection System - Complete Implementation Guide

A comprehensive guide for building an AI-powered hallmark detection and recognition system for jewelry authentication.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Stage 1: Image Preprocessing](#3-stage-1-image-preprocessing)
4. [Stage 2: YOLO Detection Model](#4-stage-2-yolo-detection-model)
5. [Stage 3: OCR Recognition Layer](#5-stage-3-ocr-recognition-layer)
6. [Stage 4: Validation Layer](#6-stage-4-validation-layer)
7. [Dataset Preparation](#7-dataset-preparation)
8. [Training Pipeline](#8-training-pipeline)
9. [Deployment & API Integration](#9-deployment--api-integration)
10. [Performance Optimization](#10-performance-optimization)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                                      │
│                   (Jewelry photo with hallmark)                          │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: IMAGE PREPROCESSING                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Resize    │→ │  Denoise    │→ │  Contrast   │→ │ Binarize    │    │
│  │   & Crop    │  │  (Bilateral)│  │  (CLAHE)    │  │ (Adaptive)  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: YOLO DETECTION                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  YOLOv8/v10 Custom Model                                        │    │
│  │  - Detects hallmark regions (BIS logo, purity marks, HUID)      │    │
│  │  - Returns bounding boxes with confidence scores                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: OCR RECOGNITION                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  PaddleOCR (Fine-tuned) or TrOCR                                │    │
│  │  - Reads text from cropped hallmark regions                     │    │
│  │  - Outputs: "916", "750", "585", HUID codes, etc.               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: VALIDATION LAYER                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  - Validate against known BIS hallmark patterns                 │    │
│  │  - Cross-check purity codes (916, 750, 585, etc.)               │    │
│  │  - Verify HUID format (6-digit alphanumeric)                    │    │
│  │  - Optional: BIS API verification                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                           │
│  {                                                                       │
│    "purity_code": "916",                                                │
│    "karat": "22K",                                                      │
│    "purity_percentage": 91.6,                                           │
│    "huid": "AB1234",                                                    │
│    "bis_certified": true,                                               │
│    "confidence": 0.95                                                   │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Prerequisites & Environment Setup

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060 (6GB) | NVIDIA RTX 3080+ (10GB+) |
| RAM | 16GB | 32GB |
| Storage | 50GB SSD | 100GB NVMe SSD |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |

### Software Requirements

```bash
# Python version
Python 3.8 - 3.11 (3.10 recommended)

# CUDA (for GPU training)
CUDA 11.8 or 12.x
cuDNN 8.x
```

### Project Structure

```
hallmark-ocr/
├── config/
│   ├── yolo_config.yaml
│   ├── paddleocr_config.yaml
│   └── hallmark_patterns.json
├── data/
│   ├── raw/                    # Original images
│   ├── processed/              # Preprocessed images
│   ├── annotations/            # YOLO format annotations
│   └── ocr_labels/             # OCR training labels
├── models/
│   ├── yolo/                   # Trained YOLO weights
│   ├── ocr/                    # Fine-tuned OCR models
│   └── pretrained/             # Downloaded pretrained models
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── image_processor.py
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py
│   ├── recognition/
│   │   ├── __init__.py
│   │   └── ocr_engine.py
│   ├── validation/
│   │   ├── __init__.py
│   │   └── hallmark_validator.py
│   └── pipeline.py
├── training/
│   ├── train_yolo.py
│   ├── train_ocr.py
│   └── augmentation.py
├── api/
│   ├── app.py
│   └── routes.py
├── tests/
├── requirements.txt
└── README.md
```

### Installation

```bash
# Create virtual environment
python -m venv hallmark-env
source hallmark-env/bin/activate  # Linux/Mac
# or
hallmark-env\Scripts\activate  # Windows

# Install core dependencies
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install YOLO
pip install ultralytics

# Install PaddleOCR
pip install paddlepaddle-gpu  # For GPU
# or
pip install paddlepaddle  # For CPU only
pip install paddleocr

# Install additional dependencies
pip install opencv-python-headless
pip install numpy
pip install Pillow
pip install fastapi uvicorn
pip install python-multipart
pip install pydantic
```

### requirements.txt

```txt
# Core ML/DL
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
paddlepaddle-gpu>=2.5.0
paddleocr>=2.7.0

# Image Processing
opencv-python-headless>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
scikit-image>=0.21.0

# API
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
pydantic>=2.0.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
requests>=2.31.0

# Optional: For TrOCR
transformers>=4.30.0
```

---

## 3. Stage 1: Image Preprocessing

Metal surfaces with hallmarks present unique challenges: reflections, uneven lighting, and small engraved text. Proper preprocessing is critical.

### src/preprocessing/image_processor.py

```python
"""
Image Preprocessing Module for Hallmark Detection

Handles the unique challenges of metal surfaces:
- Reflection removal
- Contrast enhancement for engraved text
- Noise reduction
- Adaptive binarization

References:
- https://pyimagesearch.com/2021/11/22/improving-ocr-results-with-basic-image-processing/
- https://www.nitorinfotech.com/blog/improve-ocr-accuracy-using-advanced-preprocessing-techniques/
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""
    target_size: Tuple[int, int] = (1280, 1280)  # Larger for small text
    denoise_strength: int = 10
    clahe_clip_limit: float = 3.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    sharpen_strength: float = 1.5
    enable_reflection_removal: bool = True


class HallmarkImageProcessor:
    """
    Preprocessing pipeline optimized for hallmark detection on metal surfaces.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def process(self, image: np.ndarray) -> dict:
        """
        Full preprocessing pipeline.

        Args:
            image: BGR image from cv2.imread()

        Returns:
            Dictionary containing processed images for different stages
        """
        results = {"original": image.copy()}

        # Step 1: Resize while maintaining aspect ratio
        resized = self._resize_image(image)
        results["resized"] = resized

        # Step 2: Remove reflections (for metal surfaces)
        if self.config.enable_reflection_removal:
            reflection_removed = self._remove_reflections(resized)
        else:
            reflection_removed = resized
        results["reflection_removed"] = reflection_removed

        # Step 3: Denoise
        denoised = self._denoise(reflection_removed)
        results["denoised"] = denoised

        # Step 4: Convert to grayscale
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        results["grayscale"] = gray

        # Step 5: CLAHE for contrast enhancement
        enhanced = self._apply_clahe(gray)
        results["contrast_enhanced"] = enhanced

        # Step 6: Sharpen for engraved text
        sharpened = self._sharpen(enhanced)
        results["sharpened"] = sharpened

        # Step 7: Adaptive binarization (for OCR stage)
        binary = self._adaptive_binarize(sharpened)
        results["binary"] = binary

        # Final output for detection (color, enhanced)
        results["for_detection"] = denoised
        results["for_ocr"] = binary

        return results

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        Uses 1280px for better small text detection.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.config.target_size

        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)

        if scale < 1:  # Only downscale if larger
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif scale > 1:  # Upscale if smaller (important for small hallmarks)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return image

    def _remove_reflections(self, image: np.ndarray) -> np.ndarray:
        """
        Remove specular reflections from metal surfaces using
        morphological operations and inpainting.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Detect bright spots (reflections) - threshold at 95% of max
        threshold = int(np.percentile(l_channel, 98))
        _, reflection_mask = cv2.threshold(l_channel, threshold, 255, cv2.THRESH_BINARY)

        # Dilate mask slightly to cover reflection edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        reflection_mask = cv2.dilate(reflection_mask, kernel, iterations=1)

        # Inpaint the reflection areas
        result = cv2.inpaint(image, reflection_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return result

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise while preserving edges using bilateral filter.
        Better than Gaussian blur for text preservation.
        """
        return cv2.bilateralFilter(
            image,
            d=9,
            sigmaColor=self.config.denoise_strength * 7.5,
            sigmaSpace=self.config.denoise_strength * 7.5
        )

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        Essential for uneven lighting on metal surfaces.

        Reference: https://medium.com/@Hiadore/preprocessing-images-for-ocr
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        return clahe.apply(gray)

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        """
        Sharpen engraved text using unsharp masking.
        """
        # Gaussian blur for unsharp mask
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)

        # Unsharp mask: original + strength * (original - blurred)
        sharpened = cv2.addWeighted(
            gray, 1 + self.config.sharpen_strength,
            blurred, -self.config.sharpen_strength,
            0
        )

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _adaptive_binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        Adaptive thresholding for handling uneven illumination.
        Better than global Otsu for metal surfaces.
        """
        return cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.config.adaptive_block_size,
            C=self.config.adaptive_c
        )

    def process_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Quick preprocessing for YOLO detection stage."""
        resized = self._resize_image(image)
        if self.config.enable_reflection_removal:
            resized = self._remove_reflections(resized)
        return self._denoise(resized)

    def process_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Quick preprocessing for OCR recognition stage."""
        results = self.process(image)
        return results["for_ocr"]


# Convenience functions
def preprocess_image(image_path: str, config: Optional[PreprocessConfig] = None) -> dict:
    """Load and preprocess an image from file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    processor = HallmarkImageProcessor(config)
    return processor.process(image)


def preprocess_batch(image_paths: list, config: Optional[PreprocessConfig] = None) -> list:
    """Preprocess multiple images."""
    processor = HallmarkImageProcessor(config)
    results = []

    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            results.append({
                "path": path,
                "processed": processor.process(image)
            })

    return results
```

### Preprocessing Tips for Metal Hallmarks

| Challenge | Solution | OpenCV Function |
|-----------|----------|-----------------|
| Specular reflections | Inpainting + morphological ops | `cv2.inpaint()` |
| Uneven lighting | CLAHE | `cv2.createCLAHE()` |
| Engraved (not printed) text | Unsharp masking | `cv2.addWeighted()` |
| Surface noise | Bilateral filter | `cv2.bilateralFilter()` |
| Low contrast | Histogram stretching | `cv2.normalize()` |

---

## 4. Stage 2: YOLO Detection Model

The detection stage locates hallmark regions in the image. We use YOLOv8 for its excellent small object detection.

### Why YOLOv8?

- **Anchor-free detection** - Better for small objects like hallmarks
- **80.2% mAP on domain-specific tasks** (vs 73.5% for YOLOv5)
- **Easy training and deployment** - Ultralytics ecosystem
- **Export flexibility** - ONNX, TensorRT, CoreML

Reference: [How to Train YOLOv8 on Custom Dataset](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/)

### Detection Classes

Define these classes for hallmark detection:

```yaml
# config/yolo_config.yaml
# Dataset configuration for YOLO training

path: ../data  # Dataset root directory
train: images/train
val: images/val
test: images/test

# Classes for hallmark detection
names:
  0: hallmark_region      # General hallmark area
  1: bis_logo             # BIS triangle logo
  2: purity_mark          # 916, 750, 585, etc.
  3: huid_code            # 6-digit HUID
  4: jeweler_mark         # Jeweler identification
  5: ahc_mark             # Assaying & Hallmarking Centre mark

# Training parameters (optimized for small objects)
# Use these when running training
# imgsz: 1280  # Larger image size for small text
# batch: 8     # Adjust based on GPU memory
# epochs: 100
# patience: 20
```

### src/detection/yolo_detector.py

```python
"""
YOLO-based Hallmark Detection Module

Uses YOLOv8 for detecting hallmark regions on jewelry.
Optimized for small object detection on metal surfaces.

References:
- https://learnopencv.com/train-yolov8-on-custom-dataset/
- https://docs.ultralytics.com/
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    cropped_image: Optional[np.ndarray] = None


@dataclass
class DetectorConfig:
    """Configuration for YOLO detector."""
    model_path: str = "models/yolo/hallmark_detector.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    image_size: int = 1280  # Larger for small objects
    device: str = "auto"  # "cpu", "cuda", "mps", or "auto"
    classes: Optional[List[int]] = None  # Filter specific classes


class HallmarkDetector:
    """
    YOLOv8-based detector for hallmark regions.
    """

    # Class name mapping
    CLASS_NAMES = {
        0: "hallmark_region",
        1: "bis_logo",
        2: "purity_mark",
        3: "huid_code",
        4: "jeweler_mark",
        5: "ahc_mark"
    }

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """Load YOLO model from path or download pretrained."""
        model_path = Path(self.config.model_path)

        if model_path.exists():
            print(f"Loading custom model from: {model_path}")
            return YOLO(str(model_path))
        else:
            print("Custom model not found. Loading pretrained YOLOv8n...")
            print("Note: Train a custom model for hallmark detection.")
            return YOLO("yolov8n.pt")

    def detect(self, image: np.ndarray, return_crops: bool = True) -> List[Detection]:
        """
        Detect hallmarks in an image.

        Args:
            image: BGR image (from cv2.imread or preprocessed)
            return_crops: Whether to include cropped regions

        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            classes=self.config.classes,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                # Get box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

                # Get class and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                class_name = self.CLASS_NAMES.get(class_id, f"class_{class_id}")

                # Crop region if requested
                cropped = None
                if return_crops:
                    # Add padding for better OCR
                    pad = 5
                    h, w = image.shape[:2]
                    x1_pad = max(0, x1 - pad)
                    y1_pad = max(0, y1 - pad)
                    x2_pad = min(w, x2 + pad)
                    y2_pad = min(h, y2 + pad)
                    cropped = image[y1_pad:y2_pad, x1_pad:x2_pad].copy()

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    cropped_image=cropped
                ))

        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)

        return detections

    def detect_from_path(self, image_path: str, return_crops: bool = True) -> List[Detection]:
        """Detect hallmarks from image file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.detect(image, return_crops)

    def visualize(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes on image."""
        result = image.copy()

        colors = {
            "hallmark_region": (0, 255, 0),
            "bis_logo": (255, 0, 0),
            "purity_mark": (0, 0, 255),
            "huid_code": (255, 255, 0),
            "jeweler_mark": (255, 0, 255),
            "ahc_mark": (0, 255, 255),
        }

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (128, 128, 128))

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return result


class YOLOTrainer:
    """
    Training utilities for custom hallmark detection model.
    """

    @staticmethod
    def train(
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 1280,
        batch: int = 8,
        model: str = "yolov8s.pt",  # Start with small model
        project: str = "runs/hallmark",
        name: str = "train",
        patience: int = 20,
        device: str = "auto"
    ) -> str:
        """
        Train YOLOv8 on custom hallmark dataset.

        Args:
            data_yaml: Path to dataset configuration YAML
            epochs: Number of training epochs
            imgsz: Image size (use 1280 for small objects)
            batch: Batch size (reduce if OOM)
            model: Base model to fine-tune
            project: Project directory for outputs
            name: Experiment name
            patience: Early stopping patience
            device: Training device

        Returns:
            Path to best model weights

        Example:
            >>> trainer = YOLOTrainer()
            >>> best_model = trainer.train(
            ...     data_yaml="config/yolo_config.yaml",
            ...     epochs=100,
            ...     imgsz=1280
            ... )
        """
        # Load base model
        model = YOLO(model)

        # Train
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            patience=patience,
            device=device,
            # Optimizations for small objects
            augment=True,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            # Save settings
            save=True,
            save_period=10,
            plots=True,
        )

        # Return path to best weights
        best_path = Path(project) / name / "weights" / "best.pt"
        print(f"Training complete! Best model saved to: {best_path}")

        return str(best_path)

    @staticmethod
    def validate(model_path: str, data_yaml: str) -> dict:
        """Validate model on test set."""
        model = YOLO(model_path)
        results = model.val(data=data_yaml)
        return {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }

    @staticmethod
    def export(model_path: str, format: str = "onnx") -> str:
        """
        Export model for deployment.

        Supported formats: onnx, torchscript, tensorrt, coreml, etc.
        """
        model = YOLO(model_path)
        export_path = model.export(format=format)
        print(f"Model exported to: {export_path}")
        return export_path
```

### Training Command (CLI)

```bash
# Train from command line
yolo train \
    data=config/yolo_config.yaml \
    model=yolov8s.pt \
    epochs=100 \
    imgsz=1280 \
    batch=8 \
    patience=20 \
    project=runs/hallmark \
    name=experiment1

# Validate
yolo val \
    model=runs/hallmark/experiment1/weights/best.pt \
    data=config/yolo_config.yaml

# Export for deployment
yolo export \
    model=runs/hallmark/experiment1/weights/best.pt \
    format=onnx
```

---

## 5. Stage 3: OCR Recognition Layer

After detecting hallmark regions, we use OCR to read the text (916, 750, HUID codes, etc.).

### Option A: PaddleOCR (Recommended)

Best balance of accuracy and ease of fine-tuning.

Reference: [PaddleOCR Fine-tuning Guide](https://hackernoon.com/ocr-fine-tuning-from-raw-data-to-custom-paddle-ocr-model)

### Option B: TrOCR (Highest Accuracy)

Microsoft's Transformer-based OCR, best for unusual fonts.

### src/recognition/ocr_engine.py

```python
"""
OCR Recognition Module for Hallmark Text

Supports multiple OCR backends:
- PaddleOCR (recommended for fine-tuning)
- TrOCR (highest accuracy)
- Tesseract (fallback)

References:
- https://hackernoon.com/ocr-fine-tuning-from-raw-data-to-custom-paddle-ocr-model
- https://anushsom.medium.com/finetuning-paddleocrs-recognition-model-for-dummies
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re


@dataclass
class OCRResult:
    """Single OCR recognition result."""
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2


@dataclass
class OCRConfig:
    """Configuration for OCR engine."""
    backend: str = "paddleocr"  # "paddleocr", "trocr", "tesseract"
    language: str = "en"
    use_gpu: bool = True
    model_dir: Optional[str] = None  # Custom model directory
    det_model_dir: Optional[str] = None
    rec_model_dir: Optional[str] = None
    use_angle_cls: bool = True
    det: bool = True  # Run detection (False if already cropped)


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """Recognize text in image."""
        pass

    @abstractmethod
    def recognize_batch(self, images: List[np.ndarray]) -> List[List[OCRResult]]:
        """Recognize text in multiple images."""
        pass


class PaddleOCREngine(OCREngine):
    """
    PaddleOCR-based recognition engine.

    Recommended for hallmark detection due to:
    - Easy fine-tuning with PPOCRLabel
    - Good accuracy on engraved text
    - Multi-language support
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.ocr = self._initialize()

    def _initialize(self):
        """Initialize PaddleOCR with configuration."""
        from paddleocr import PaddleOCR

        kwargs = {
            "use_angle_cls": self.config.use_angle_cls,
            "lang": self.config.language,
            "use_gpu": self.config.use_gpu,
            "show_log": False,
        }

        # Use custom models if provided
        if self.config.det_model_dir:
            kwargs["det_model_dir"] = self.config.det_model_dir
        if self.config.rec_model_dir:
            kwargs["rec_model_dir"] = self.config.rec_model_dir

        return PaddleOCR(**kwargs)

    def recognize(self, image: np.ndarray, det: bool = True) -> List[OCRResult]:
        """
        Recognize text in image.

        Args:
            image: BGR image
            det: Whether to run detection (False for pre-cropped regions)
        """
        # PaddleOCR expects RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        result = self.ocr.ocr(image_rgb, det=det, cls=self.config.use_angle_cls)

        ocr_results = []

        if result and result[0]:
            for line in result[0]:
                if det:
                    # Full detection + recognition
                    bbox_points = line[0]  # 4 corner points
                    text, confidence = line[1]

                    # Convert to x1, y1, x2, y2
                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]
                    bbox = (
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords)),
                        int(max(y_coords))
                    )
                else:
                    # Recognition only
                    text, confidence = line
                    bbox = None

                ocr_results.append(OCRResult(
                    text=text.strip(),
                    confidence=confidence,
                    bbox=bbox
                ))

        return ocr_results

    def recognize_batch(self, images: List[np.ndarray], det: bool = True) -> List[List[OCRResult]]:
        """Recognize text in multiple images."""
        return [self.recognize(img, det) for img in images]

    def recognize_cropped(self, image: np.ndarray) -> OCRResult:
        """
        Recognize text from a pre-cropped hallmark region.
        Skips detection for efficiency.
        """
        results = self.recognize(image, det=False)

        if results:
            # Return highest confidence result
            return max(results, key=lambda x: x.confidence)

        return OCRResult(text="", confidence=0.0)


class TrOCREngine(OCREngine):
    """
    Microsoft TrOCR-based recognition engine.

    Best accuracy for unusual fonts and handwritten text.
    Requires more GPU memory than PaddleOCR.

    Reference: https://huggingface.co/microsoft/trocr-base-printed
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.processor, self.model = self._initialize()

    def _initialize(self):
        """Initialize TrOCR model."""
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        # Use custom model or default
        model_name = self.config.model_dir or "microsoft/trocr-base-printed"

        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)

        if self.config.use_gpu and torch.cuda.is_available():
            model = model.cuda()

        model.eval()

        return processor, model

    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """Recognize text using TrOCR."""
        import torch
        from PIL import Image

        # Convert BGR to RGB PIL Image
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        pil_image = Image.fromarray(image_rgb)

        # Process image
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values

        if self.config.use_gpu and torch.cuda.is_available():
            pixel_values = pixel_values.cuda()

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # TrOCR doesn't provide confidence directly
        return [OCRResult(text=text.strip(), confidence=0.95)]

    def recognize_batch(self, images: List[np.ndarray]) -> List[List[OCRResult]]:
        """Recognize text in batch."""
        return [self.recognize(img) for img in images]


class HallmarkOCR:
    """
    High-level OCR interface for hallmark recognition.
    Combines detection crops with OCR and post-processing.
    """

    # Known hallmark patterns for post-processing
    PURITY_PATTERNS = {
        "916": {"karat": "22K", "purity": 91.6},
        "750": {"karat": "18K", "purity": 75.0},
        "585": {"karat": "14K", "purity": 58.5},
        "375": {"karat": "9K", "purity": 37.5},
        "958": {"karat": "23K", "purity": 95.8},
        "875": {"karat": "21K", "purity": 87.5},
        "999": {"karat": "24K", "purity": 99.9},
    }

    # HUID pattern: 6 alphanumeric characters
    HUID_PATTERN = re.compile(r'^[A-Z0-9]{6}$')

    def __init__(self, backend: str = "paddleocr", config: Optional[OCRConfig] = None):
        if config is None:
            config = OCRConfig(backend=backend)

        if backend == "paddleocr":
            self.engine = PaddleOCREngine(config)
        elif backend == "trocr":
            self.engine = TrOCREngine(config)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def read_purity_mark(self, image: np.ndarray) -> Dict:
        """
        Read purity mark from cropped image.
        Returns structured purity information.
        """
        results = self.engine.recognize(image)

        if not results:
            return {"raw": "", "purity_code": None, "karat": None, "purity": None, "confidence": 0}

        best = max(results, key=lambda x: x.confidence)
        raw_text = best.text.upper().strip()

        # Extract numeric purity code
        numbers = re.findall(r'\d{3}', raw_text)

        purity_code = None
        purity_info = None

        for num in numbers:
            if num in self.PURITY_PATTERNS:
                purity_code = num
                purity_info = self.PURITY_PATTERNS[num]
                break

        return {
            "raw": raw_text,
            "purity_code": purity_code,
            "karat": purity_info["karat"] if purity_info else None,
            "purity": purity_info["purity"] if purity_info else None,
            "confidence": best.confidence
        }

    def read_huid(self, image: np.ndarray) -> Dict:
        """
        Read HUID (Hallmark Unique Identification) code.
        HUID is a 6-digit alphanumeric code.
        """
        results = self.engine.recognize(image)

        if not results:
            return {"raw": "", "huid": None, "valid": False, "confidence": 0}

        best = max(results, key=lambda x: x.confidence)
        raw_text = best.text.upper().strip()

        # Remove spaces and special characters
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text)

        # Find 6-character alphanumeric sequence
        matches = re.findall(r'[A-Z0-9]{6}', cleaned)

        huid = matches[0] if matches else None
        valid = bool(huid and self.HUID_PATTERN.match(huid))

        return {
            "raw": raw_text,
            "huid": huid,
            "valid": valid,
            "confidence": best.confidence
        }

    def read_all(self, image: np.ndarray) -> List[OCRResult]:
        """Read all text from image."""
        return self.engine.recognize(image)


# PaddleOCR Fine-tuning utilities
class PaddleOCRTrainer:
    """
    Utilities for fine-tuning PaddleOCR on hallmark data.

    Dataset format expected:
    - Images in data/images/
    - Labels in data/labels.txt (format: image_path\ttext)

    Reference: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/training_en.md
    """

    @staticmethod
    def prepare_dataset(
        images_dir: str,
        labels_csv: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ):
        """
        Prepare dataset in PaddleOCR format.

        Expected CSV format:
        filename,text
        img001.jpg,916
        img002.jpg,750
        """
        import pandas as pd
        from pathlib import Path
        import shutil

        df = pd.read_csv(labels_csv)
        df = df.sample(frac=1, random_state=42)  # Shuffle

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {
            "train": df[:train_end],
            "val": df[train_end:val_end],
            "test": df[val_end:]
        }

        output_path = Path(output_dir)

        for split_name, split_df in splits.items():
            split_dir = output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            # Create label file
            label_file = output_path / f"{split_name}_label.txt"

            with open(label_file, 'w') as f:
                for _, row in split_df.iterrows():
                    src_path = Path(images_dir) / row['filename']
                    dst_path = split_dir / row['filename']

                    if src_path.exists():
                        shutil.copy(src_path, dst_path)
                        # PaddleOCR format: path\ttext
                        f.write(f"{dst_path}\t{row['text']}\n")

        print(f"Dataset prepared in {output_dir}")
        print(f"  Train: {len(splits['train'])} samples")
        print(f"  Val: {len(splits['val'])} samples")
        print(f"  Test: {len(splits['test'])} samples")

    @staticmethod
    def get_training_command(
        config_path: str = "configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml",
        pretrained_model: str = "pretrain_models/en_PP-OCRv4_rec_train/best_accuracy",
        train_data: str = "data/train_label.txt",
        eval_data: str = "data/val_label.txt",
        save_dir: str = "output/hallmark_rec",
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.0001
    ) -> str:
        """
        Generate PaddleOCR training command.

        Returns command to run in terminal.
        """
        cmd = f"""
# Clone PaddleOCR if not already done
# git clone https://github.com/PaddlePaddle/PaddleOCR.git
# cd PaddleOCR

python tools/train.py \\
    -c {config_path} \\
    -o Global.pretrained_model={pretrained_model} \\
    -o Train.dataset.data_dir=./ \\
    -o Train.dataset.label_file_list=["{train_data}"] \\
    -o Eval.dataset.data_dir=./ \\
    -o Eval.dataset.label_file_list=["{eval_data}"] \\
    -o Global.save_model_dir={save_dir} \\
    -o Global.epoch_num={epochs} \\
    -o Train.loader.batch_size_per_card={batch_size} \\
    -o Optimizer.lr.learning_rate={learning_rate}
"""
        return cmd
```

### PaddleOCR Fine-tuning Configuration

```yaml
# config/paddleocr_config.yaml
# Configuration for fine-tuning PaddleOCR on hallmark data

Global:
  use_gpu: true
  epoch_num: 100
  save_model_dir: ./output/hallmark_rec
  save_epoch_step: 10
  eval_batch_step: [0, 500]
  pretrained_model: ./pretrain_models/en_PP-OCRv4_rec_train/best_accuracy

Optimizer:
  name: Adam
  lr:
    name: Cosine
    learning_rate: 0.0001  # Lower for fine-tuning
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 0.00001

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetV3
    scale: 0.95
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
    dims: 120
  Head:
    name: CTCHead
    fc_decay: 0.00001

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/
    label_file_list:
      - ./data/train_label.txt
  loader:
    shuffle: True
    batch_size_per_card: 64
    drop_last: True
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/
    label_file_list:
      - ./data/val_label.txt
  loader:
    shuffle: False
    batch_size_per_card: 64
    drop_last: False
    num_workers: 4
```

---

## 6. Stage 4: Validation Layer

Validates OCR results against known BIS hallmark patterns.

### src/validation/hallmark_validator.py

```python
"""
Hallmark Validation Module

Validates detected hallmarks against BIS (Bureau of Indian Standards) patterns.
Optionally verifies HUID against BIS database.

References:
- https://www.bis.gov.in/hallmarking-overview/
- https://www.bluestone.com/jewellery-education/certification-guide/bis-hallmark
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class HallmarkType(Enum):
    """Types of hallmark components."""
    BIS_LOGO = "bis_logo"
    PURITY_MARK = "purity_mark"
    HUID = "huid"
    JEWELER_MARK = "jeweler_mark"
    AHC_MARK = "ahc_mark"


@dataclass
class ValidationResult:
    """Result of hallmark validation."""
    is_valid: bool
    confidence: float
    hallmark_type: Optional[HallmarkType]
    details: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class HallmarkInfo:
    """Complete hallmark information."""
    purity_code: Optional[str] = None
    karat: Optional[str] = None
    purity_percentage: Optional[float] = None
    huid: Optional[str] = None
    bis_certified: bool = False
    jeweler_id: Optional[str] = None
    ahc_code: Optional[str] = None
    overall_confidence: float = 0.0
    validation_results: List[ValidationResult] = field(default_factory=list)


class BISHallmarkPatterns:
    """
    BIS Hallmark pattern definitions and validation rules.

    Based on IS 1417 (grades of gold and gold alloys).
    """

    # Valid gold purity grades (BIS recognized)
    GOLD_PURITY_GRADES = {
        "375": {"karat": "9K", "purity": 37.5, "min_fineness": 375},
        "585": {"karat": "14K", "purity": 58.5, "min_fineness": 583},
        "750": {"karat": "18K", "purity": 75.0, "min_fineness": 750},
        "875": {"karat": "21K", "purity": 87.5, "min_fineness": 875},
        "916": {"karat": "22K", "purity": 91.6, "min_fineness": 916},
        "958": {"karat": "23K", "purity": 95.8, "min_fineness": 958},
        "999": {"karat": "24K", "purity": 99.9, "min_fineness": 990},
    }

    # Silver purity grades
    SILVER_PURITY_GRADES = {
        "800": {"purity": 80.0, "grade": "Silver 800"},
        "835": {"purity": 83.5, "grade": "Silver 835"},
        "900": {"purity": 90.0, "grade": "Silver 900"},
        "925": {"purity": 92.5, "grade": "Sterling Silver"},
        "950": {"purity": 95.0, "grade": "Britannia Silver"},
        "999": {"purity": 99.9, "grade": "Fine Silver"},
    }

    # Alternative representations
    PURITY_ALIASES = {
        # Gold
        "22K916": "916",
        "22K": "916",
        "22KT": "916",
        "18K750": "750",
        "18K": "750",
        "18KT": "750",
        "14K585": "585",
        "14K": "585",
        "14KT": "585",
        "9K375": "375",
        "9K": "375",
        "9KT": "375",
        # Silver
        "STERLING": "925",
        "STG": "925",
        "SS925": "925",
    }

    # HUID format: 6 alphanumeric characters
    HUID_PATTERN = re.compile(r'^[A-Z0-9]{6}$')

    # Common OCR errors and corrections
    OCR_CORRECTIONS = {
        'O': '0',  # Letter O to zero
        'I': '1',  # Letter I to one
        'l': '1',  # Lowercase L to one
        'S': '5',  # S to 5
        'B': '8',  # B to 8
        'G': '6',  # G to 6
    }


class HallmarkValidator:
    """
    Validates hallmark information against BIS standards.
    """

    def __init__(self):
        self.patterns = BISHallmarkPatterns()

    def validate_purity_mark(self, raw_text: str) -> ValidationResult:
        """
        Validate purity mark (916, 750, 585, etc.)
        """
        # Clean and normalize
        text = raw_text.upper().strip()
        text = re.sub(r'[^A-Z0-9]', '', text)

        # Check aliases first
        if text in self.patterns.PURITY_ALIASES:
            code = self.patterns.PURITY_ALIASES[text]
        else:
            # Extract 3-digit code
            matches = re.findall(r'\d{3}', text)
            code = matches[0] if matches else None

        if code and code in self.patterns.GOLD_PURITY_GRADES:
            grade_info = self.patterns.GOLD_PURITY_GRADES[code]
            return ValidationResult(
                is_valid=True,
                confidence=0.95,
                hallmark_type=HallmarkType.PURITY_MARK,
                details={
                    "raw_text": raw_text,
                    "purity_code": code,
                    "karat": grade_info["karat"],
                    "purity_percentage": grade_info["purity"],
                    "metal": "gold"
                }
            )

        if code and code in self.patterns.SILVER_PURITY_GRADES:
            grade_info = self.patterns.SILVER_PURITY_GRADES[code]
            return ValidationResult(
                is_valid=True,
                confidence=0.95,
                hallmark_type=HallmarkType.PURITY_MARK,
                details={
                    "raw_text": raw_text,
                    "purity_code": code,
                    "purity_percentage": grade_info["purity"],
                    "grade": grade_info["grade"],
                    "metal": "silver"
                }
            )

        # Try OCR correction
        corrected = self._apply_ocr_corrections(text)
        corrected_matches = re.findall(r'\d{3}', corrected)

        if corrected_matches:
            code = corrected_matches[0]
            if code in self.patterns.GOLD_PURITY_GRADES:
                grade_info = self.patterns.GOLD_PURITY_GRADES[code]
                return ValidationResult(
                    is_valid=True,
                    confidence=0.75,  # Lower confidence due to correction
                    hallmark_type=HallmarkType.PURITY_MARK,
                    details={
                        "raw_text": raw_text,
                        "corrected_text": corrected,
                        "purity_code": code,
                        "karat": grade_info["karat"],
                        "purity_percentage": grade_info["purity"],
                        "metal": "gold"
                    },
                    warnings=["OCR correction applied"]
                )

        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            hallmark_type=HallmarkType.PURITY_MARK,
            details={"raw_text": raw_text},
            errors=[f"Unknown purity code: {raw_text}"]
        )

    def validate_huid(self, raw_text: str) -> ValidationResult:
        """
        Validate HUID (Hallmark Unique Identification).

        HUID is a 6-digit alphanumeric code mandatory since April 2023.
        """
        # Clean text
        text = raw_text.upper().strip()
        text = re.sub(r'[^A-Z0-9]', '', text)

        # Find 6-character sequences
        matches = re.findall(r'[A-Z0-9]{6}', text)

        if matches:
            huid = matches[0]
            if self.patterns.HUID_PATTERN.match(huid):
                return ValidationResult(
                    is_valid=True,
                    confidence=0.9,
                    hallmark_type=HallmarkType.HUID,
                    details={
                        "raw_text": raw_text,
                        "huid": huid,
                        "format_valid": True
                    }
                )

        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            hallmark_type=HallmarkType.HUID,
            details={"raw_text": raw_text},
            errors=["Invalid HUID format (expected 6 alphanumeric characters)"]
        )

    def validate_bis_logo(self, detected: bool, confidence: float) -> ValidationResult:
        """
        Validate BIS logo detection.
        """
        return ValidationResult(
            is_valid=detected,
            confidence=confidence,
            hallmark_type=HallmarkType.BIS_LOGO,
            details={"detected": detected}
        )

    def _apply_ocr_corrections(self, text: str) -> str:
        """Apply common OCR error corrections."""
        result = text
        for wrong, correct in self.patterns.OCR_CORRECTIONS.items():
            result = result.replace(wrong, correct)
        return result

    def validate_complete_hallmark(
        self,
        purity_text: Optional[str] = None,
        huid_text: Optional[str] = None,
        bis_logo_detected: bool = False,
        bis_logo_confidence: float = 0.0
    ) -> HallmarkInfo:
        """
        Validate complete hallmark information.

        A valid BIS hallmark should have:
        1. BIS logo (triangle)
        2. Purity/Fineness mark (916, 750, etc.)
        3. HUID (6-digit alphanumeric) - mandatory since April 2023
        """
        info = HallmarkInfo()

        # Validate BIS logo
        if bis_logo_detected:
            logo_result = self.validate_bis_logo(True, bis_logo_confidence)
            info.validation_results.append(logo_result)
            info.bis_certified = logo_result.is_valid and logo_result.confidence > 0.5

        # Validate purity mark
        if purity_text:
            purity_result = self.validate_purity_mark(purity_text)
            info.validation_results.append(purity_result)

            if purity_result.is_valid:
                info.purity_code = purity_result.details.get("purity_code")
                info.karat = purity_result.details.get("karat")
                info.purity_percentage = purity_result.details.get("purity_percentage")

        # Validate HUID
        if huid_text:
            huid_result = self.validate_huid(huid_text)
            info.validation_results.append(huid_result)

            if huid_result.is_valid:
                info.huid = huid_result.details.get("huid")

        # Calculate overall confidence
        valid_results = [r for r in info.validation_results if r.is_valid]
        if valid_results:
            info.overall_confidence = sum(r.confidence for r in valid_results) / len(valid_results)

        return info

    def to_dict(self, info: HallmarkInfo) -> Dict:
        """Convert HallmarkInfo to dictionary for API response."""
        return {
            "purity_code": info.purity_code,
            "karat": info.karat,
            "purity_percentage": info.purity_percentage,
            "huid": info.huid,
            "bis_certified": info.bis_certified,
            "overall_confidence": round(info.overall_confidence, 3),
            "validation_details": [
                {
                    "type": r.hallmark_type.value if r.hallmark_type else None,
                    "is_valid": r.is_valid,
                    "confidence": round(r.confidence, 3),
                    "details": r.details,
                    "warnings": r.warnings,
                    "errors": r.errors
                }
                for r in info.validation_results
            ]
        }


# Load patterns from JSON config
def load_patterns_from_config(config_path: str) -> BISHallmarkPatterns:
    """Load custom patterns from JSON configuration."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    patterns = BISHallmarkPatterns()

    if "gold_grades" in config:
        patterns.GOLD_PURITY_GRADES.update(config["gold_grades"])

    if "silver_grades" in config:
        patterns.SILVER_PURITY_GRADES.update(config["silver_grades"])

    if "aliases" in config:
        patterns.PURITY_ALIASES.update(config["aliases"])

    return patterns
```

### config/hallmark_patterns.json

```json
{
  "gold_grades": {
    "375": {"karat": "9K", "purity": 37.5, "min_fineness": 375},
    "585": {"karat": "14K", "purity": 58.5, "min_fineness": 583},
    "750": {"karat": "18K", "purity": 75.0, "min_fineness": 750},
    "875": {"karat": "21K", "purity": 87.5, "min_fineness": 875},
    "916": {"karat": "22K", "purity": 91.6, "min_fineness": 916},
    "958": {"karat": "23K", "purity": 95.8, "min_fineness": 958},
    "999": {"karat": "24K", "purity": 99.9, "min_fineness": 990}
  },
  "silver_grades": {
    "800": {"purity": 80.0, "grade": "Silver 800"},
    "925": {"purity": 92.5, "grade": "Sterling Silver"},
    "950": {"purity": 95.0, "grade": "Britannia Silver"},
    "999": {"purity": 99.9, "grade": "Fine Silver"}
  },
  "aliases": {
    "22K916": "916",
    "22K": "916",
    "18K750": "750",
    "18K": "750",
    "14K585": "585",
    "14K": "585",
    "STERLING": "925"
  },
  "ocr_corrections": {
    "O": "0",
    "I": "1",
    "l": "1",
    "S": "5"
  }
}
```

---

## 7. Dataset Preparation

### Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Detection (YOLO) | 500 annotated images | 2000+ images |
| Recognition (OCR) | 500 text samples | 5000+ samples |

### Collecting Images

```python
"""
Dataset collection guidelines for hallmark detection.
"""

# Image requirements:
# - Resolution: At least 1080p, preferably 4K for small hallmarks
# - Lighting: Varied (natural, studio, flash)
# - Angles: Multiple angles per hallmark
# - Surfaces: Different gold colors, finishes
# - Conditions: Clean, worn, partially visible

# Recommended sources:
# 1. Photograph your own jewelry collection
# 2. Partner with jewelers for access
# 3. Controlled studio setup for consistent quality
# 4. Synthetic data generation (as supplement)

CAPTURE_GUIDELINES = """
1. Camera Settings:
   - Use macro mode for close-ups
   - Minimum 12MP resolution
   - RAW format if possible
   - Disable flash (use diffused lighting)

2. Lighting:
   - Use ring light or light tent to minimize reflections
   - Capture same hallmark under different lighting
   - Include some challenging conditions for robustness

3. Composition:
   - Fill frame with hallmark area
   - Capture at multiple angles (0°, 15°, 30°)
   - Include context images (full jewelry + close-up)

4. Variety:
   - Different purity marks (916, 750, 585)
   - Various HUID codes
   - Different BIS logo stamps
   - Worn/aged hallmarks
   - Different metal colors (yellow, white, rose gold)
"""
```

### Annotation Tools

1. **For YOLO (Detection)**:
   - [Roboflow](https://roboflow.com) - Web-based, easy to use
   - [CVAT](https://cvat.org) - Open-source, feature-rich
   - [LabelImg](https://github.com/tzutalin/labelImg) - Simple, offline

2. **For PaddleOCR (Recognition)**:
   - [PPOCRLabel](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/PPOCRLabel/README.md) - Official tool

### YOLO Annotation Format

```
# data/annotations/image001.txt
# Format: class_id center_x center_y width height (normalized 0-1)

0 0.5 0.5 0.1 0.05    # hallmark_region
1 0.3 0.4 0.05 0.05   # bis_logo
2 0.5 0.5 0.08 0.04   # purity_mark (916)
3 0.7 0.5 0.12 0.04   # huid_code
```

### Data Augmentation

```python
"""
Augmentation strategies for hallmark dataset.
Critical for handling varied real-world conditions.
"""

import albumentations as A
import cv2
import numpy as np


def get_hallmark_augmentation():
    """
    Augmentation pipeline optimized for hallmark images.
    """
    return A.Compose([
        # Geometric transforms (simulate different angles)
        A.Rotate(limit=15, p=0.5),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        A.Affine(shear=(-5, 5), p=0.3),

        # Lighting variations (critical for metal surfaces)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),

        # Simulate reflections and glare
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 1),
            angle_lower=0,
            angle_upper=1,
            num_flare_circles_lower=1,
            num_flare_circles_upper=3,
            src_radius=100,
            src_color=(255, 255, 255),
            p=0.2
        ),

        # Color variations (different gold colors)
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

        # Blur (simulate motion/focus issues)
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1),
            A.GaussianBlur(blur_limit=5, p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.3),

        # Noise (simulate camera noise)
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1),
        ], p=0.3),

        # Quality degradation
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),

    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def augment_dataset(
    images_dir: str,
    annotations_dir: str,
    output_dir: str,
    augmentations_per_image: int = 5
):
    """
    Generate augmented dataset.
    """
    from pathlib import Path
    import shutil

    transform = get_hallmark_augmentation()

    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_images = output_path / "images"
    output_labels = output_path / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    for img_file in images_path.glob("*.jpg"):
        # Load image
        image = cv2.imread(str(img_file))

        # Load annotations
        ann_file = Path(annotations_dir) / f"{img_file.stem}.txt"
        bboxes = []
        class_labels = []

        if ann_file.exists():
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:5]])

        # Copy original
        shutil.copy(img_file, output_images / img_file.name)
        if ann_file.exists():
            shutil.copy(ann_file, output_labels / ann_file.name)

        # Generate augmentations
        for i in range(augmentations_per_image):
            augmented = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )

            # Save augmented image
            aug_name = f"{img_file.stem}_aug{i}{img_file.suffix}"
            cv2.imwrite(str(output_images / aug_name), augmented['image'])

            # Save augmented annotations
            aug_ann = output_labels / f"{img_file.stem}_aug{i}.txt"
            with open(aug_ann, 'w') as f:
                for cls, bbox in zip(augmented['class_labels'], augmented['bboxes']):
                    f.write(f"{cls} {' '.join(map(str, bbox))}\n")

    print(f"Augmented dataset saved to {output_dir}")
```

---

## 8. Training Pipeline

### Complete Training Script

```python
"""
training/train_pipeline.py

Complete training pipeline for hallmark detection system.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_yolo_detector(
    data_yaml: str = "config/yolo_config.yaml",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 8,
    base_model: str = "yolov8s.pt"
):
    """
    Train YOLO detection model for hallmark localization.
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("STAGE 1: Training YOLO Detection Model")
    print("=" * 60)

    model = YOLO(base_model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/hallmark_detection",
        name=f"train_{timestamp}",
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        # Small object optimizations
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    best_model = Path("runs/hallmark_detection") / f"train_{timestamp}" / "weights" / "best.pt"
    print(f"\nYOLO training complete!")
    print(f"Best model: {best_model}")

    return str(best_model)


def prepare_ocr_dataset(
    detection_model: str,
    images_dir: str,
    output_dir: str
):
    """
    Use trained YOLO model to crop hallmark regions for OCR training.
    """
    from src.detection.yolo_detector import HallmarkDetector, DetectorConfig
    import cv2
    from pathlib import Path

    print("=" * 60)
    print("STAGE 2: Preparing OCR Dataset")
    print("=" * 60)

    config = DetectorConfig(model_path=detection_model)
    detector = HallmarkDetector(config)

    output_path = Path(output_dir)
    (output_path / "purity_marks").mkdir(parents=True, exist_ok=True)
    (output_path / "huid_codes").mkdir(parents=True, exist_ok=True)

    labels = []

    for img_file in Path(images_dir).glob("*.jpg"):
        image = cv2.imread(str(img_file))
        detections = detector.detect(image, return_crops=True)

        for i, det in enumerate(detections):
            if det.cropped_image is not None:
                if det.class_name == "purity_mark":
                    save_dir = output_path / "purity_marks"
                elif det.class_name == "huid_code":
                    save_dir = output_path / "huid_codes"
                else:
                    continue

                crop_name = f"{img_file.stem}_{det.class_name}_{i}.jpg"
                cv2.imwrite(str(save_dir / crop_name), det.cropped_image)

                # You'll need to manually label these or use existing labels
                labels.append({
                    "image": crop_name,
                    "class": det.class_name,
                    "text": ""  # Fill in manually
                })

    print(f"Cropped {len(labels)} regions for OCR training")
    print(f"Output: {output_dir}")
    print("\nNOTE: You need to manually label the text for each cropped image")

    return output_dir


def train_paddleocr(
    data_dir: str,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.0001
):
    """
    Fine-tune PaddleOCR recognition model.

    Note: This generates the training command.
    Actual training requires PaddleOCR repository setup.
    """
    print("=" * 60)
    print("STAGE 3: Fine-tuning PaddleOCR")
    print("=" * 60)

    cmd = f"""
# Prerequisites:
# 1. Clone PaddleOCR: git clone https://github.com/PaddlePaddle/PaddleOCR.git
# 2. Install dependencies: pip install -r requirements.txt
# 3. Download pretrained model

# Training command:
cd PaddleOCR

python tools/train.py \\
    -c configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml \\
    -o Global.pretrained_model=./pretrain_models/en_PP-OCRv4_rec_train/best_accuracy \\
    -o Train.dataset.data_dir={data_dir} \\
    -o Train.dataset.label_file_list=["{data_dir}/train_label.txt"] \\
    -o Eval.dataset.data_dir={data_dir} \\
    -o Eval.dataset.label_file_list=["{data_dir}/val_label.txt"] \\
    -o Global.save_model_dir=./output/hallmark_ocr \\
    -o Global.epoch_num={epochs} \\
    -o Train.loader.batch_size_per_card={batch_size} \\
    -o Optimizer.lr.learning_rate={learning_rate}

# Export for inference:
python tools/export_model.py \\
    -c configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml \\
    -o Global.pretrained_model=./output/hallmark_ocr/best_accuracy \\
    -o Global.save_inference_dir=./inference/hallmark_rec
"""

    print(cmd)
    print("\nRun the above commands in your PaddleOCR directory")

    return cmd


def main():
    """
    Complete training pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train Hallmark OCR System")
    parser.add_argument("--stage", choices=["yolo", "ocr-prep", "ocr", "all"], default="all")
    parser.add_argument("--data-yaml", default="config/yolo_config.yaml")
    parser.add_argument("--images-dir", default="data/raw")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)

    args = parser.parse_args()

    if args.stage in ["yolo", "all"]:
        yolo_model = train_yolo_detector(
            data_yaml=args.data_yaml,
            epochs=args.epochs,
            batch=args.batch
        )

    if args.stage in ["ocr-prep", "all"]:
        if args.stage == "ocr-prep":
            yolo_model = "models/yolo/best.pt"  # Use existing model

        prepare_ocr_dataset(
            detection_model=yolo_model,
            images_dir=args.images_dir,
            output_dir="data/ocr_crops"
        )

    if args.stage in ["ocr", "all"]:
        train_paddleocr(
            data_dir="data/ocr_crops",
            epochs=args.epochs
        )

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 9. Deployment & API Integration

### Complete Pipeline

```python
"""
src/pipeline.py

Complete hallmark detection and recognition pipeline.
"""

import cv2
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from src.preprocessing.image_processor import HallmarkImageProcessor, PreprocessConfig
from src.detection.yolo_detector import HallmarkDetector, DetectorConfig
from src.recognition.ocr_engine import HallmarkOCR, OCRConfig
from src.validation.hallmark_validator import HallmarkValidator


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    yolo_model_path: str = "models/yolo/hallmark_detector.pt"
    ocr_backend: str = "paddleocr"
    ocr_model_dir: Optional[str] = None
    use_gpu: bool = True
    confidence_threshold: float = 0.5


class HallmarkPipeline:
    """
    Complete pipeline for hallmark detection and recognition.

    Usage:
        pipeline = HallmarkPipeline()
        result = pipeline.process("jewelry_image.jpg")
        print(result)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize components
        self.preprocessor = HallmarkImageProcessor()

        self.detector = HallmarkDetector(DetectorConfig(
            model_path=self.config.yolo_model_path,
            confidence_threshold=self.config.confidence_threshold
        ))

        self.ocr = HallmarkOCR(
            backend=self.config.ocr_backend,
            config=OCRConfig(
                model_dir=self.config.ocr_model_dir,
                use_gpu=self.config.use_gpu
            )
        )

        self.validator = HallmarkValidator()

    def process(self, image_input) -> Dict:
        """
        Process an image through the complete pipeline.

        Args:
            image_input: File path (str) or numpy array (BGR image)

        Returns:
            Dictionary with hallmark information and confidence scores
        """
        # Load image if path provided
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                return {"error": f"Could not load image: {image_input}"}
        else:
            image = image_input

        # Stage 1: Preprocess
        preprocessed = self.preprocessor.process(image)
        detection_input = preprocessed["for_detection"]

        # Stage 2: Detect hallmark regions
        detections = self.detector.detect(detection_input, return_crops=True)

        if not detections:
            return {
                "success": False,
                "error": "No hallmarks detected",
                "confidence": 0.0
            }

        # Stage 3: OCR on detected regions
        purity_text = None
        huid_text = None
        bis_logo_detected = False
        bis_logo_confidence = 0.0

        for det in detections:
            if det.cropped_image is None:
                continue

            # Preprocess crop for OCR
            crop_processed = self.preprocessor.process_for_ocr(det.cropped_image)

            if det.class_name == "purity_mark":
                result = self.ocr.read_purity_mark(crop_processed)
                if result["confidence"] > 0.5:
                    purity_text = result["raw"]

            elif det.class_name == "huid_code":
                result = self.ocr.read_huid(crop_processed)
                if result["confidence"] > 0.5:
                    huid_text = result["raw"]

            elif det.class_name == "bis_logo":
                bis_logo_detected = True
                bis_logo_confidence = det.confidence

        # Stage 4: Validate
        hallmark_info = self.validator.validate_complete_hallmark(
            purity_text=purity_text,
            huid_text=huid_text,
            bis_logo_detected=bis_logo_detected,
            bis_logo_confidence=bis_logo_confidence
        )

        # Build response
        response = {
            "success": True,
            "hallmark": self.validator.to_dict(hallmark_info),
            "detections": [
                {
                    "class": det.class_name,
                    "confidence": round(det.confidence, 3),
                    "bbox": det.bbox
                }
                for det in detections
            ]
        }

        return response

    def process_batch(self, image_inputs: list) -> list:
        """Process multiple images."""
        return [self.process(img) for img in image_inputs]


# Convenience function
def detect_hallmark(image_path: str, config: Optional[PipelineConfig] = None) -> Dict:
    """
    Quick function to detect hallmark in an image.

    Args:
        image_path: Path to the jewelry image
        config: Optional pipeline configuration

    Returns:
        Dictionary with hallmark information

    Example:
        >>> result = detect_hallmark("ring.jpg")
        >>> print(result["hallmark"]["purity_code"])  # "916"
        >>> print(result["hallmark"]["karat"])  # "22K"
    """
    pipeline = HallmarkPipeline(config)
    return pipeline.process(image_path)
```

### FastAPI REST API

```python
"""
api/app.py

REST API for hallmark detection service.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
import io
from PIL import Image

from src.pipeline import HallmarkPipeline, PipelineConfig


# Initialize FastAPI app
app = FastAPI(
    title="Hallmark OCR API",
    description="AI-powered hallmark detection and recognition for jewelry authentication",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (lazy loading)
pipeline: Optional[HallmarkPipeline] = None


def get_pipeline() -> HallmarkPipeline:
    """Get or create pipeline instance."""
    global pipeline
    if pipeline is None:
        pipeline = HallmarkPipeline()
    return pipeline


class HallmarkResponse(BaseModel):
    """Response model for hallmark detection."""
    success: bool
    purity_code: Optional[str] = None
    karat: Optional[str] = None
    purity_percentage: Optional[float] = None
    huid: Optional[str] = None
    bis_certified: bool = False
    confidence: float = 0.0
    error: Optional[str] = None


class DetectionDetail(BaseModel):
    """Detection detail model."""
    class_name: str
    confidence: float
    bbox: List[int]


class FullResponse(BaseModel):
    """Full response with detection details."""
    success: bool
    hallmark: Optional[dict] = None
    detections: List[DetectionDetail] = []
    error: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Hallmark OCR API"}


@app.post("/detect", response_model=FullResponse)
async def detect_hallmark(file: UploadFile = File(...)):
    """
    Detect and recognize hallmarks in an uploaded image.

    - **file**: Image file (JPEG, PNG)

    Returns hallmark information including purity, HUID, and BIS certification status.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Process through pipeline
        pipe = get_pipeline()
        result = pipe.process(image)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/simple", response_model=HallmarkResponse)
async def detect_hallmark_simple(file: UploadFile = File(...)):
    """
    Simplified endpoint returning only essential hallmark information.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        pipe = get_pipeline()
        result = pipe.process(image)

        if not result.get("success"):
            return HallmarkResponse(
                success=False,
                error=result.get("error", "Detection failed")
            )

        hallmark = result.get("hallmark", {})

        return HallmarkResponse(
            success=True,
            purity_code=hallmark.get("purity_code"),
            karat=hallmark.get("karat"),
            purity_percentage=hallmark.get("purity_percentage"),
            huid=hallmark.get("huid"),
            bis_certified=hallmark.get("bis_certified", False),
            confidence=hallmark.get("overall_confidence", 0.0)
        )

    except Exception as e:
        return HallmarkResponse(success=False, error=str(e))


@app.post("/validate/huid")
async def validate_huid(huid: str):
    """
    Validate HUID format (does not check against BIS database).
    """
    from src.validation.hallmark_validator import HallmarkValidator

    validator = HallmarkValidator()
    result = validator.validate_huid(huid)

    return {
        "huid": huid,
        "valid": result.is_valid,
        "formatted": result.details.get("huid"),
        "errors": result.errors
    }


# Run with: uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Running the API

```bash
# Development
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Production with Gunicorn
gunicorn api.app:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### API Usage Examples

```bash
# Detect hallmark
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@jewelry_image.jpg"

# Simple response
curl -X POST "http://localhost:8000/detect/simple" \
  -F "file=@ring.jpg"

# Validate HUID
curl -X POST "http://localhost:8000/validate/huid?huid=AB1234"
```

---

## 10. Performance Optimization

### Model Optimization

```python
"""
Export models for production deployment.
"""

# YOLO to ONNX
from ultralytics import YOLO

model = YOLO("models/yolo/hallmark_detector.pt")
model.export(format="onnx", imgsz=1280, dynamic=True)

# YOLO to TensorRT (NVIDIA GPUs)
model.export(format="engine", imgsz=1280, half=True)  # FP16 for speed

# PaddleOCR to ONNX
# Run in PaddleOCR directory:
# paddle2onnx --model_dir ./inference/hallmark_rec \
#   --model_filename inference.pdmodel \
#   --params_filename inference.pdiparams \
#   --save_file ./onnx/hallmark_rec.onnx
```

### Inference Optimization Tips

| Technique | Speedup | Memory | Notes |
|-----------|---------|--------|-------|
| TensorRT (FP16) | 2-4x | -30% | NVIDIA GPUs only |
| ONNX Runtime | 1.5-2x | Similar | Cross-platform |
| Batch processing | 2-3x | +50% | Process multiple images |
| Image resize | 1.5x | -40% | 640 vs 1280, accuracy tradeoff |
| CPU threads | 1.3x | Same | Set `OMP_NUM_THREADS` |

---

## 11. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Low detection accuracy | Small hallmarks | Increase `imgsz` to 1280 |
| Reflections causing errors | Metal surface | Enable reflection removal preprocessing |
| OCR reading wrong characters | Font mismatch | Fine-tune on your specific hallmarks |
| CUDA out of memory | Large batch/image | Reduce batch size or image size |
| HUID not detected | Small text | Use higher resolution images |

### Debugging Tips

```python
# Save intermediate results for debugging
def debug_pipeline(image_path: str):
    """Save all intermediate processing stages."""
    import cv2
    from pathlib import Path

    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)

    # Load and preprocess
    image = cv2.imread(image_path)
    preprocessor = HallmarkImageProcessor()
    results = preprocessor.process(image)

    # Save each stage
    for stage_name, stage_image in results.items():
        if isinstance(stage_image, np.ndarray):
            cv2.imwrite(str(debug_dir / f"{stage_name}.jpg"), stage_image)

    # Detect and visualize
    detector = HallmarkDetector()
    detections = detector.detect(results["for_detection"])

    viz = detector.visualize(image, detections)
    cv2.imwrite(str(debug_dir / "detections.jpg"), viz)

    print(f"Debug output saved to {debug_dir}")
```

---

## Quick Start Checklist

- [ ] Set up Python environment with dependencies
- [ ] Collect 500+ hallmark images
- [ ] Annotate images for YOLO (use Roboflow or CVAT)
- [ ] Train YOLO detection model
- [ ] Crop hallmark regions using trained model
- [ ] Label text for OCR training
- [ ] Fine-tune PaddleOCR (or use pretrained)
- [ ] Test complete pipeline
- [ ] Deploy API
- [ ] Monitor and improve with production data

---

## References

### Image Preprocessing
- [7 Steps of Image Pre-processing for OCR](https://nextgeninvent.com/blogs/7-steps-of-image-pre-processing-to-improve-ocr-using-python-2/)
- [PyImageSearch: Improving OCR with Image Processing](https://pyimagesearch.com/2021/11/22/improving-ocr-results-with-basic-image-processing/)

### YOLO Training
- [Roboflow: Train YOLOv8 on Custom Dataset](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/)
- [LearnOpenCV: YOLOv8 Custom Training](https://learnopencv.com/train-yolov8-on-custom-dataset/)

### PaddleOCR Fine-tuning
- [HackerNoon: OCR Fine-tuning Guide](https://hackernoon.com/ocr-fine-tuning-from-raw-data-to-custom-paddle-ocr-model)
- [Medium: Fine-tuning PaddleOCR](https://anushsom.medium.com/finetuning-paddleocrs-recognition-model-for-dummies-by-a-dummy-89ac7d7edcf6)
- [Official PaddleOCR Training Docs](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/training_en.md)

### BIS Hallmarking Standards
- [BIS Official Hallmarking Overview](https://www.bis.gov.in/hallmarking-overview/)
- [BlueStone: BIS Hallmark Guide](https://www.bluestone.com/jewellery-education/certification-guide/bis-hallmark)
- [Wikipedia: BIS Hallmark](https://en.wikipedia.org/wiki/BIS_hallmark)

---

*Document Version: 1.0*
*Last Updated: February 2026*
