"""
Manakonline Excel Parser

Parses Excel files downloaded from the Manakonline portal for comparison with OCR results.

Expected format:
- BIS Job No: Extracted from filename (e.g., HUID_Print_125554301.xlsx -> 125554301)
- Columns: AHC Tag, Material Category, Item Category, Declared Purity, HUID

The file may be in .xlsx, .xls, or .numbers format (Apple Numbers).
"""

import os
import re
import io
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd


@dataclass
class ManakRow:
    """A row from the Manakonline Excel file."""
    row_number: int
    ahc_tag: str
    material_category: Optional[str]
    item_category: Optional[str]
    declared_purity: Optional[str]
    huid: Optional[str]


@dataclass
class ManakParseResult:
    """Result of parsing a Manakonline Excel file."""
    bis_job_no: str
    rows: List[ManakRow]
    total_rows: int
    errors: List[dict]


# Column name aliases for flexible header detection
COLUMN_ALIASES = {
    'ahc_tag': [
        'ahc tag', 'ahc_tag', 'tag', 'tag id', 'tag_id', 'tagid',
        'ahctag', 'article tag', 'article_tag', 's no'
    ],
    'material_category': [
        'material category', 'material_category', 'material', 'metal',
        'metal type', 'metal_type'
    ],
    'item_category': [
        'item category', 'item_category', 'item', 'category',
        'article', 'article type', 'article_type'
    ],
    'declared_purity': [
        'declared purity', 'declared_purity', 'purity', 'karat',
        'karatage', 'fineness', 'grade'
    ],
    'huid': [
        'huid', 'huid id', 'huid_id', 'hallmark uid', 'unique id',
        'hu id', 'h.u.i.d', 'huid no', 'huid number'
    ]
}


def extract_bis_job_no(filename: str) -> Optional[str]:
    """Extract BIS Job No from filename.

    Examples:
        HUID_Print_125554301.xlsx -> 125554301
        125554301_HUID.csv -> 125554301
    """
    # Remove extension
    name = os.path.splitext(filename)[0]

    # Try to find a sequence of digits that looks like a job number
    # Usually 9-12 digits
    matches = re.findall(r'\d{6,12}', name)
    if matches:
        # Return the longest match (most likely the job number)
        return max(matches, key=len)

    return None


def normalize_header(header: str) -> str:
    """Normalize a header string for comparison."""
    if not header:
        return ''
    return str(header).lower().strip().replace(' ', '_').replace('.', '').replace('-', '_')


def find_column(df: pd.DataFrame, field: str) -> Optional[str]:
    """Find a column in the DataFrame matching the field aliases."""
    aliases = COLUMN_ALIASES.get(field, [field])

    for col in df.columns:
        normalized = normalize_header(col)
        if normalized in [normalize_header(a) for a in aliases]:
            return col

    return None


def parse_numbers_file(file_bytes: bytes) -> Optional[pd.DataFrame]:
    """Attempt to parse an Apple Numbers file.

    Numbers files are ZIP archives containing protobuf data.
    We extract what we can using basic parsing.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            # Look for data in the archive
            all_text = []
            for name in zf.namelist():
                try:
                    content = zf.read(name)
                    # Extract printable strings
                    strings = re.findall(rb'[\x20-\x7e]{4,}', content)
                    all_text.extend(s.decode('utf-8', errors='ignore') for s in strings)
                except:
                    pass

            # Try to find HUID patterns (6 alphanumeric chars with at least one letter)
            huids = []
            for text in all_text:
                matches = re.findall(r'\b[A-Z0-9]{6}\b', text.upper())
                for m in matches:
                    if any(c.isalpha() for c in m) and not m.isdigit():
                        huids.append(m)

            # Try to find purity patterns
            purities = re.findall(r'\b(22K916|18K750|14K585|24K999|925|916|750|585)\b',
                                  ' '.join(all_text).upper())

            # Try to find tag patterns
            tags = []
            for text in all_text:
                # Look for patterns like a1, a2, TAG001, etc.
                matches = re.findall(r'\b[a-zA-Z]\d+\b|\b[A-Z]{2,}\d+\b', text)
                tags.extend(matches)

            if huids or tags:
                # Build a basic DataFrame
                max_len = max(len(huids), len(tags), len(purities))
                data = {
                    'AHC Tag': tags[:max_len] if tags else [''] * max_len,
                    'HUID': huids[:max_len] if huids else [''] * max_len,
                    'Declared Purity': purities[:max_len] if purities else [''] * max_len,
                }
                return pd.DataFrame(data)
    except Exception as e:
        print(f"Numbers parsing error: {e}")

    return None


def parse_manak_file(file_bytes: bytes, filename: str) -> ManakParseResult:
    """Parse a Manakonline Excel file.

    Args:
        file_bytes: Raw file content
        filename: Original filename (used to extract BIS Job No)

    Returns:
        ManakParseResult with parsed rows and metadata
    """
    errors = []
    rows = []

    # Extract BIS Job No from filename
    bis_job_no = extract_bis_job_no(filename) or 'unknown'

    # Detect file type and parse
    df = None

    # Check if it's a ZIP file (Numbers or XLSX)
    if file_bytes[:2] == b'PK':
        # Try as Numbers file first (Apple's format)
        df = parse_numbers_file(file_bytes)

        if df is None:
            # Try as regular Excel
            try:
                df = pd.read_excel(io.BytesIO(file_bytes))
            except Exception as e:
                errors.append({'error': f'Excel parse error: {str(e)}'})
    else:
        # Try CSV
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
                    break
                except:
                    continue
        except Exception as e:
            errors.append({'error': f'CSV parse error: {str(e)}'})

    if df is None or df.empty:
        return ManakParseResult(
            bis_job_no=bis_job_no,
            rows=[],
            total_rows=0,
            errors=errors or [{'error': 'Could not parse file'}]
        )

    # Find columns
    ahc_tag_col = find_column(df, 'ahc_tag')
    material_col = find_column(df, 'material_category')
    item_col = find_column(df, 'item_category')
    purity_col = find_column(df, 'declared_purity')
    huid_col = find_column(df, 'huid')

    if not ahc_tag_col:
        # Try using first column as tag
        ahc_tag_col = df.columns[0] if len(df.columns) > 0 else None

    if not huid_col:
        # Try to find HUID in any column by pattern
        for col in df.columns:
            sample = df[col].dropna().head(10).astype(str)
            for val in sample:
                if re.match(r'^[A-Z0-9]{6}$', str(val).upper()):
                    huid_col = col
                    break
            if huid_col:
                break

    # Parse rows
    for idx, row in df.iterrows():
        try:
            ahc_tag = str(row[ahc_tag_col]).strip() if ahc_tag_col and pd.notna(row.get(ahc_tag_col)) else ''

            if not ahc_tag or ahc_tag.lower() in ['nan', 'none', '']:
                continue

            manak_row = ManakRow(
                row_number=idx + 2,  # Excel rows are 1-indexed, +1 for header
                ahc_tag=ahc_tag,
                material_category=str(row.get(material_col, '')).strip() if material_col else None,
                item_category=str(row.get(item_col, '')).strip() if item_col else None,
                declared_purity=str(row.get(purity_col, '')).strip() if purity_col else None,
                huid=extract_huid(str(row.get(huid_col, ''))) if huid_col else None
            )
            rows.append(manak_row)

        except Exception as e:
            errors.append({
                'row': idx + 2,
                'error': str(e)
            })

    return ManakParseResult(
        bis_job_no=bis_job_no,
        rows=rows,
        total_rows=len(rows),
        errors=errors
    )


def extract_huid(value: str) -> Optional[str]:
    """Extract a 6-character HUID from a value.

    The value might be:
    - Just the HUID: "YY6DUG"
    - Combined with purity: "22K916YY6DUG"
    - With prefix/suffix: "#$YY6DUG"
    """
    if not value or pd.isna(value):
        return None

    value = str(value).upper().strip()

    # Remove common prefixes
    value = re.sub(r'^[#$\s]+', '', value)

    # If it's exactly 6 chars and alphanumeric with at least one letter
    if len(value) == 6 and value.isalnum() and any(c.isalpha() for c in value):
        return value

    # Try to extract from combined format (e.g., 22K916YY6DUG)
    # HUID is usually the last 6 chars
    if len(value) > 6:
        last6 = value[-6:]
        if last6.isalnum() and any(c.isalpha() for c in last6):
            return last6

    # Look for 6-char alphanumeric pattern anywhere
    matches = re.findall(r'[A-Z0-9]{6}', value)
    for m in matches:
        if any(c.isalpha() for c in m) and not m.isdigit():
            return m

    return None


def extract_purity(value: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract purity code and karat from a declared purity value.

    Returns:
        Tuple of (purity_code, karat)

    Examples:
        "22K916" -> ("916", "22K")
        "916" -> ("916", "22K")
        "18K750" -> ("750", "18K")
    """
    if not value or pd.isna(value):
        return None, None

    value = str(value).upper().strip()

    # Common purity codes and their karats
    PURITY_MAP = {
        '999': '24K',
        '958': '23K',
        '916': '22K',
        '875': '21K',
        '750': '18K',
        '585': '14K',
        '375': '9K',
        '925': 'Sterling',
    }

    # Look for purity code
    for code, karat in PURITY_MAP.items():
        if code in value:
            return code, karat

    # Try to extract karat
    karat_match = re.search(r'(\d{1,2})K', value)
    if karat_match:
        k = int(karat_match.group(1))
        karat = f"{k}K"
        # Map karat to purity
        karat_to_purity = {
            24: '999', 23: '958', 22: '916', 21: '875',
            18: '750', 14: '585', 9: '375'
        }
        purity = karat_to_purity.get(k)
        return purity, karat

    return None, None


def compare_with_ocr(manak_rows: List[ManakRow], ocr_results: dict) -> List[dict]:
    """Compare Manakonline data with OCR results.

    Args:
        manak_rows: Parsed rows from Manakonline Excel
        ocr_results: Dict mapping tag_id to OCR result

    Returns:
        List of comparison results
    """
    comparisons = []

    for row in manak_rows:
        ocr = ocr_results.get(row.ahc_tag)

        comparison = {
            'tag_id': row.ahc_tag,
            'manak_huid': row.huid,
            'manak_purity': row.declared_purity,
            'ocr_huid': ocr.get('actual_huid') if ocr else None,
            'ocr_purity': ocr.get('purity_code') if ocr else None,
            'huid_match': False,
            'purity_match': False,
            'status': 'missing_ocr'
        }

        if ocr:
            # Compare HUID
            if row.huid and ocr.get('actual_huid'):
                comparison['huid_match'] = (
                    row.huid.upper() == ocr['actual_huid'].upper() or
                    row.huid.upper() in ocr['actual_huid'].upper() or
                    ocr['actual_huid'].upper() in row.huid.upper()
                )

            # Compare purity
            if row.declared_purity and ocr.get('purity_code'):
                manak_purity, _ = extract_purity(row.declared_purity)
                comparison['purity_match'] = (
                    manak_purity == ocr['purity_code'] if manak_purity else False
                )

            # Determine status
            if comparison['huid_match'] and comparison['purity_match']:
                comparison['status'] = 'match'
            elif comparison['huid_match'] or comparison['purity_match']:
                comparison['status'] = 'partial_match'
            else:
                comparison['status'] = 'mismatch'

        comparisons.append(comparison)

    return comparisons
