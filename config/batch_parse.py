"""Pure parsing + validation helpers for the /stage1/upload-batch endpoint.

Kept independent of FastAPI and PaddleOCR so the logic is cheap to unit test.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


TAG_ID_MAX_LEN = 64
EXPECTED_HUID_MAX_LEN = 32

# Known aliases for the two canonical column names. The value is the canonical
# name; every normalized header that matches a key on the left is mapped to it.
TAG_ID_ALIASES = {
    "tag_id", "tagid", "tag", "tag_no", "tag_number",
    "id", "item_id", "ahc_tag", "ahc_tag_id",
}
HUID_ALIASES = {
    "expected_huid", "huid", "huid_id", "huid_no",
    "expectedhuid", "laser_printing_mark", "laser_print", "laser_mark",
}


@dataclass
class ParsedRow:
    row_number: int        # 1-based, matches what the user sees in Excel
    tag_id: str
    expected_huid: str


@dataclass
class RejectedRow:
    row_number: int
    tag_id: str
    reason: str

    def to_dict(self) -> dict:
        return {"row": self.row_number, "tag_id": self.tag_id, "reason": self.reason}


@dataclass
class ParseResult:
    total_rows: int
    accepted: List[ParsedRow]
    rejected: List[RejectedRow]


class ParseError(Exception):
    """Raised for structural problems that prevent any row from being read."""

    def __init__(self, message: str, status_code: int = 400, **extra):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.extra = extra


def _normalize_header(h: Any) -> str:
    return re.sub(r"\s+", "_", str(h).strip().lower())


def pick_sheet(sheet_names: List[str]) -> str:
    """Prefer a sheet whose name starts with HUID or PRINT (case-insensitive)."""
    for name in sheet_names:
        if re.match(r"^(huid|print)", str(name).strip(), flags=re.IGNORECASE):
            return name
    return sheet_names[0]


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return a copy of df with canonical tag_id/expected_huid columns.

    Raises ParseError(400) if either required column is missing. The error
    carries the list of detected columns so the UI can surface it.
    """
    original = list(df.columns)
    normalized = [_normalize_header(h) for h in original]

    df = df.copy()
    df.columns = normalized

    mapping: Dict[str, str] = {}
    for alias in TAG_ID_ALIASES:
        if alias in df.columns:
            mapping["tag_id"] = alias
            break
    for alias in HUID_ALIASES:
        if alias in df.columns:
            mapping["expected_huid"] = alias
            break

    missing = [c for c in ("tag_id", "expected_huid") if c not in mapping]
    if missing:
        raise ParseError(
            f"Missing required column(s): {', '.join(missing)}",
            status_code=400,
            detected_columns=original,
            normalized_columns=normalized,
        )

    renames = {v: k for k, v in mapping.items() if v != k}
    if renames:
        df = df.rename(columns=renames)
    return df, mapping


# --- HUID validation (composite format) -------------------------------------
#
# expected_huid values in this project are composite strings like
# "22K91677WAX9" — karat + purity + 6-char HUID suffix. compare_huids()
# already treats them that way at OCR-match time, so batch validation does
# the same: the string is valid iff it contains at least one 6-char
# alphanumeric substring with a letter that is not a known purity alias.
_PURITY_ALIAS_WORDS = {"916000", "999000", "925000"}
_HUID_CANDIDATE_RE = re.compile(r"[A-Z0-9]{6}")


def _has_valid_huid_substring(value: str) -> bool:
    for candidate in _HUID_CANDIDATE_RE.findall(value):
        if candidate.isdigit():
            continue
        if candidate in _PURITY_ALIAS_WORDS:
            continue
        if re.search(r"[A-Z]", candidate):
            return True
    return False


def validate_row(row_number: int, raw_tag: Any, raw_huid: Any) -> Tuple[Optional[ParsedRow], Optional[RejectedRow]]:
    """Normalize and validate a single row.

    Returns ``(accepted, None)`` or ``(None, rejected)`` — never both.
    """
    tag_id = "" if raw_tag is None else str(raw_tag).strip()
    # pandas turns empty Excel cells into NaN which stringifies to 'nan'.
    if tag_id.lower() == "nan":
        tag_id = ""
    if not tag_id:
        return None, RejectedRow(row_number, "", "tag_id is required")
    if len(tag_id) > TAG_ID_MAX_LEN:
        return None, RejectedRow(row_number, tag_id, f"tag_id exceeds {TAG_ID_MAX_LEN} chars")

    huid = "" if raw_huid is None else str(raw_huid).strip().upper()
    if huid.lower() == "nan":
        huid = ""
    if not huid:
        return None, RejectedRow(row_number, tag_id, "expected_huid is required")
    if len(huid) > EXPECTED_HUID_MAX_LEN:
        return None, RejectedRow(
            row_number, tag_id, f"expected_huid exceeds {EXPECTED_HUID_MAX_LEN} chars"
        )
    if not _has_valid_huid_substring(huid):
        return None, RejectedRow(
            row_number,
            tag_id,
            "expected_huid must contain a 6-character alphanumeric code with at least one letter",
        )

    return ParsedRow(row_number, tag_id, huid), None


def parse_batch_file(contents: bytes, filename: str) -> ParseResult:
    """Parse an uploaded CSV/Excel file into accepted/rejected rows.

    Raises ParseError for structural problems (unreadable file, unknown
    extension, missing columns). Row-level problems come back as
    ``RejectedRow`` entries inside the result.
    """
    name = (filename or "").lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise ParseError(f"Could not parse CSV: {e}", status_code=400)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            excel = pd.ExcelFile(io.BytesIO(contents))
            sheet = pick_sheet(excel.sheet_names)
            df = pd.read_excel(excel, sheet_name=sheet)
        except Exception as e:
            raise ParseError(f"Could not parse Excel file: {e}", status_code=400)
    else:
        raise ParseError(
            f"Unsupported file type: {filename}. Use .csv, .xlsx, or .xls",
            status_code=415,
        )

    df, _ = normalize_columns(df)

    accepted: List[ParsedRow] = []
    rejected: List[RejectedRow] = []
    seen_tags: Dict[str, int] = {}

    # Row numbering: header is row 1 from the user's POV, so data starts at 2.
    for idx, row in enumerate(df.itertuples(index=False), start=2):
        raw_tag = getattr(row, "tag_id", None)
        raw_huid = getattr(row, "expected_huid", None)
        accepted_row, rejected_row = validate_row(idx, raw_tag, raw_huid)

        if rejected_row is not None:
            rejected.append(rejected_row)
            continue

        assert accepted_row is not None
        if accepted_row.tag_id in seen_tags:
            rejected.append(
                RejectedRow(
                    accepted_row.row_number,
                    accepted_row.tag_id,
                    f"duplicate tag_id (first seen on row {seen_tags[accepted_row.tag_id]})",
                )
            )
            continue

        seen_tags[accepted_row.tag_id] = accepted_row.row_number
        accepted.append(accepted_row)

    return ParseResult(total_rows=len(df), accepted=accepted, rejected=rejected)
