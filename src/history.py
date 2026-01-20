"""
History management for OCR results.
Stores processed images and their extracted text locally.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, asdict


HISTORY_DIR = Path(__file__).parent.parent / "history"
HISTORY_FILE = HISTORY_DIR / "history.json"


@dataclass
class HistoryEntry:
    """A single history entry."""
    id: str
    timestamp: str
    image_path: str
    text: str
    avg_confidence: float
    thumbnail_path: str


def ensure_history_dir():
    """Ensure history directory exists."""
    HISTORY_DIR.mkdir(exist_ok=True)


def load_history() -> list[dict]:
    """Load history from JSON file."""
    ensure_history_dir()
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_history(history: list[dict]):
    """Save history to JSON file."""
    ensure_history_dir()
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def add_to_history(image: Image.Image, text: str, avg_confidence: float) -> HistoryEntry:
    """
    Add a new entry to history.

    Args:
        image: PIL Image that was processed
        text: Extracted text
        avg_confidence: Average confidence score

    Returns:
        The created HistoryEntry
    """
    ensure_history_dir()

    # Generate unique ID
    entry_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Save full image
    image_path = HISTORY_DIR / f"{entry_id}.png"
    image.save(image_path)

    # Create and save thumbnail
    thumbnail = image.copy()
    thumbnail.thumbnail((150, 150))
    thumbnail_path = HISTORY_DIR / f"{entry_id}_thumb.png"
    thumbnail.save(thumbnail_path)

    # Create entry
    entry = HistoryEntry(
        id=entry_id,
        timestamp=timestamp,
        image_path=str(image_path),
        text=text,
        avg_confidence=avg_confidence,
        thumbnail_path=str(thumbnail_path)
    )

    # Add to history
    history = load_history()
    history.insert(0, asdict(entry))  # Most recent first

    # Keep only last 20 entries
    if len(history) > 20:
        # Remove old entries and their images
        for old_entry in history[20:]:
            try:
                os.remove(old_entry["image_path"])
                os.remove(old_entry["thumbnail_path"])
            except OSError:
                pass
        history = history[:20]

    save_history(history)
    return entry


def get_history() -> list[dict]:
    """Get all history entries."""
    return load_history()


def clear_history():
    """Clear all history entries and delete images."""
    history = load_history()
    for entry in history:
        try:
            os.remove(entry["image_path"])
            os.remove(entry["thumbnail_path"])
        except OSError:
            pass
    save_history([])
