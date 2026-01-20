#!/usr/bin/env python3.9
"""
Entry point for the OCR Dashboard.
Run with: python3.9 run.py
Or directly: python3.9 -m streamlit run src/dashboard.py
"""

import subprocess
import sys


def main():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard.py",
        "--server.headless", "true"
    ])


if __name__ == "__main__":
    main()
