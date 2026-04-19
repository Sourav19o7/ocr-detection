"""Shared pytest configuration: put config/ on sys.path so tests can import
the parse helpers without booting the full FastAPI app."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("config", "src"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)
