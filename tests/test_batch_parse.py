"""Tests for config/batch_parse.py — the Stage 1 upload parser."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from batch_parse import (
    ParseError,
    parse_batch_file,
    pick_sheet,
    validate_row,
)


# --- helpers ---------------------------------------------------------------

def _csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode()


def _xlsx_bytes(sheets: dict[str, list[dict]]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


# --- row-level validation --------------------------------------------------

class TestValidateRow:
    def test_accepts_composite_huid(self):
        accepted, rejected = validate_row(2, "TAG001", "22K916YY6DUG")
        assert rejected is None
        assert accepted.tag_id == "TAG001"
        assert accepted.expected_huid == "22K916YY6DUG"

    def test_accepts_bare_six_char_huid(self):
        accepted, rejected = validate_row(2, "TAG001", "YY6DUG")
        assert rejected is None
        assert accepted.expected_huid == "YY6DUG"

    def test_uppercases_and_strips_huid(self):
        accepted, _ = validate_row(2, " TAG001 ", "  yy6dug  ")
        assert accepted.tag_id == "TAG001"
        assert accepted.expected_huid == "YY6DUG"

    def test_empty_tag_rejected(self):
        _, rejected = validate_row(5, "", "22K916YY6DUG")
        assert rejected.reason == "tag_id is required"

    def test_nan_tag_rejected(self):
        # pandas read_excel turns blanks into float NaN
        _, rejected = validate_row(5, float("nan"), "22K916YY6DUG")
        assert "required" in rejected.reason

    def test_empty_huid_rejected(self):
        _, rejected = validate_row(5, "TAG001", "")
        assert rejected.reason == "expected_huid is required"

    def test_huid_all_digits_rejected(self):
        # 6 digits, no letter — indistinguishable from a purity code
        _, rejected = validate_row(5, "TAG001", "916000")
        assert "letter" in rejected.reason

    def test_huid_too_short_rejected(self):
        _, rejected = validate_row(5, "TAG001", "ABC12")
        assert rejected is not None

    def test_tag_over_max_rejected(self):
        _, rejected = validate_row(5, "T" * 65, "YY6DUG")
        assert "64" in rejected.reason


# --- sheet selection -------------------------------------------------------

class TestPickSheet:
    def test_prefers_huid_sheet(self):
        assert pick_sheet(["Cover", "HUID"]) == "HUID"

    def test_prefers_print_sheet(self):
        assert pick_sheet(["Cover", "Print Sheet"]) == "Print Sheet"

    def test_case_insensitive(self):
        assert pick_sheet(["cover", "huid_data"]) == "huid_data"

    def test_falls_back_to_first(self):
        assert pick_sheet(["Cover", "Data"]) == "Cover"


# --- full-file parsing -----------------------------------------------------

class TestParseBatchFile:
    def test_happy_path_csv(self):
        body = _csv_bytes([
            {"tag_id": "TAG001", "expected_huid": "22K916YY6DUG"},
            {"tag_id": "TAG002", "expected_huid": "AB1234"},
        ])
        result = parse_batch_file(body, "batch.csv")
        assert result.total_rows == 2
        assert len(result.accepted) == 2
        assert result.rejected == []
        assert result.accepted[0].tag_id == "TAG001"

    def test_excel_with_print_sheet(self):
        body = _xlsx_bytes({
            "Summary": [{"foo": 1}],
            "PRINT": [
                {"tag_id": "TAG001", "expected_huid": "22K916YY6DUG"},
                {"tag_id": "TAG002", "expected_huid": "YY6DUG"},
            ],
        })
        result = parse_batch_file(body, "batch.xlsx")
        assert len(result.accepted) == 2

    def test_header_variants_are_normalized(self):
        body = _csv_bytes([
            {"Tag No": "T01", "HUID ID": "AB1234"},
            {"Tag No": "T02", "HUID ID": "CD5678"},
        ])
        result = parse_batch_file(body, "variants.csv")
        assert len(result.accepted) == 2
        assert result.accepted[0].tag_id == "T01"

    def test_duplicate_tag_rejected(self):
        body = _csv_bytes([
            {"tag_id": "TAG001", "expected_huid": "AB1234"},
            {"tag_id": "TAG001", "expected_huid": "CD5678"},
        ])
        result = parse_batch_file(body, "dups.csv")
        assert len(result.accepted) == 1
        assert len(result.rejected) == 1
        assert "duplicate" in result.rejected[0].reason

    def test_invalid_huid_rejected(self):
        body = _csv_bytes([
            {"tag_id": "OK", "expected_huid": "AB1234"},
            {"tag_id": "BAD", "expected_huid": "916000"},   # pure digits
        ])
        result = parse_batch_file(body, "mix.csv")
        assert [r.tag_id for r in result.accepted] == ["OK"]
        assert result.rejected[0].tag_id == "BAD"

    def test_missing_column_raises_parse_error(self):
        body = _csv_bytes([{"tag_id": "T1", "description": "x"}])
        with pytest.raises(ParseError) as excinfo:
            parse_batch_file(body, "bad.csv")
        err = excinfo.value
        assert err.status_code == 400
        assert "expected_huid" in err.message
        assert err.extra["detected_columns"] == ["tag_id", "description"]

    def test_unsupported_extension_raises_415(self):
        with pytest.raises(ParseError) as excinfo:
            parse_batch_file(b"x", "batch.txt")
        assert excinfo.value.status_code == 415

    def test_row_numbering_is_spreadsheet_relative(self):
        # First data row is row 2 (row 1 is headers)
        body = _csv_bytes([
            {"tag_id": "OK", "expected_huid": "AB1234"},
            {"tag_id": "", "expected_huid": "AB1234"},
        ])
        result = parse_batch_file(body, "rows.csv")
        assert result.rejected[0].row_number == 3
