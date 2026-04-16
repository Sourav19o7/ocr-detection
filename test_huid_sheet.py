#!/usr/bin/env python3
"""
Test script to verify HUID sheet detection in Excel files
"""
import pandas as pd

excel_file_path = "HUID_Print_125554301 (1).xlsx"

print("=" * 60)
print("HUID Sheet Detection Test")
print("=" * 60)

try:
    # Open the Excel file
    excel_file = pd.ExcelFile(excel_file_path)

    print(f"\n📁 File: {excel_file_path}")
    print(f"\n📊 Available sheets: {len(excel_file.sheet_names)}")
    for i, sheet in enumerate(excel_file.sheet_names, 1):
        print(f"  {i}. {sheet}")

    # Detect HUID sheet
    huid_sheet = None
    for sheet in excel_file.sheet_names:
        if 'HUID' in sheet.upper() or 'PRINT' in sheet.upper():
            huid_sheet = sheet
            break

    if huid_sheet:
        print(f"\n✓ HUID sheet detected: '{huid_sheet}'")

        # Read the sheet
        df = pd.read_excel(excel_file_path, sheet_name=huid_sheet)

        print(f"\n📋 Sheet preview:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        # Show first few rows
        print(f"\n🔍 First 5 rows:")
        print(df.head().to_string(index=False))

        # Check for required columns
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        print(f"\n🏷️  Normalized columns: {list(df.columns)}")

        # Check for tag_id and huid columns
        has_tag = any(col in df.columns for col in ["tag_id", "tagid", "tag", "id", "item_id"])
        has_huid = any(col in df.columns for col in ["expected_huid", "huid", "expected_id", "expectedhuid"])

        print(f"\n✓ Column validation:")
        print(f"  Tag ID column found: {has_tag}")
        print(f"  HUID column found: {has_huid}")

        if has_tag and has_huid:
            print(f"\n✅ File is valid for batch upload!")
        else:
            print(f"\n⚠️  Missing required columns")

    else:
        print(f"\n⚠️  No HUID sheet found. Will use first sheet: '{excel_file.sheet_names[0]}'")
        df = pd.read_excel(excel_file_path, sheet_name=0)
        print(f"  Columns: {list(df.columns)}")

except FileNotFoundError:
    print(f"\n❌ File not found: {excel_file_path}")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "=" * 60)
