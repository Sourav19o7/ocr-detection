"""Shared pytest configuration.

Two jobs:
1.  Put config/ and src/ on sys.path so helpers import cleanly.
2.  Stub out ocr_model_v2 BEFORE anything else can load it so the tests don't
    drag in PaddleOCR. The stub gives deterministic OCR results for the HUID
    upload path.
"""

import os
import sys
import types
from dataclasses import dataclass, field
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("config", "src"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)


# --- ocr_model_v2 stub (keeps PaddleOCR out of the test process) ---------

@dataclass
class _StubOCRResult:
    text: str = ""
    confidence: float = 0.0


@dataclass
class _StubHallmarkInfo:
    huid: Optional[str] = None
    purity_code: Optional[str] = None
    karat: Optional[str] = None
    purity_percentage: Optional[float] = None
    overall_confidence: float = 0.0
    all_results: List[_StubOCRResult] = field(default_factory=list)


class _StubOCREngine:
    """Returns a predictable hallmark info payload. Tests that need a
    different payload override ``api.ocr_engine_v2`` directly."""

    last_called_with = None

    def __init__(self, **_kwargs):
        pass

    def extract_with_hallmark_info(self, image):
        type(self).last_called_with = image
        return _StubHallmarkInfo(
            huid="YY6DUG",
            purity_code="916",
            karat="22K",
            purity_percentage=91.6,
            overall_confidence=0.92,
            all_results=[_StubOCRResult("22K 916 YY6DUG", 0.92)],
        )


if "ocr_model_v2" not in sys.modules:
    stub = types.ModuleType("ocr_model_v2")
    stub.OCREngineV2 = _StubOCREngine
    stub.OCRResultV2 = _StubOCRResult
    stub.HallmarkInfo = _StubHallmarkInfo

    class _HallmarkType:
        HUID = "huid"
        PURITY_MARK = "purity"
        BIS_LOGO = "bis"
        JEWELER_MARK = "jeweler"
        CHECK = "check"

    class _CheckInfo:
        pass

    stub.HallmarkType = _HallmarkType
    stub.CheckInfo = _CheckInfo
    sys.modules["ocr_model_v2"] = stub
