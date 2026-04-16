#!/usr/bin/env python3
"""
Test script for live hallmark OCR API on EC2
"""
import requests
import os

# API Configuration
API_BASE = "http://65.2.187.3:8000"
UPLOAD_ENDPOINT = f"{API_BASE}/api/erp/upload-and-process"

def test_health():
    """Test API health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        print(f"✓ Health check: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_upload_and_process(image_path, tag_id, expected_huid=""):
    """Test ERP upload and process endpoint"""
    print(f"\nTesting upload-and-process with {os.path.basename(image_path)}...")
    print(f"Tag ID: {tag_id}")
    print(f"Expected HUID: {expected_huid or '(empty)'}")

    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            data = {
                "tag_id": tag_id,
                "expected_huid": expected_huid
            }

            response = requests.post(
                UPLOAD_ENDPOINT,
                files=files,
                data=data,
                timeout=60
            )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ SUCCESS!")
            print(f"  Decision: {result.get('decision', 'N/A')}")
            print(f"  Detected HUID: {result.get('actual_huid', 'N/A')}")
            print(f"  HUID Match: {result.get('huid_match', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 0)*100:.1f}%")
            print(f"  Status: {result.get('status', 'N/A')}")
            return True
        else:
            print(f"\n✗ FAILED!")
            print(f"  Response: {response.text}")
            return False

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False

def main():
    print("=" * 60)
    print("Hallmark OCR API - Live Test")
    print("=" * 60)

    # Test 1: Health check
    if not test_health():
        print("\n⚠ API is not responding. Exiting.")
        return

    # Test 2: Upload without expected HUID
    test_upload_and_process(
        image_path="test.jpeg",
        tag_id="TEST_001",
        expected_huid=""
    )

    # Test 3: Upload with expected HUID (if example exists)
    if os.path.exists("examples/TAG001.jpg"):
        test_upload_and_process(
            image_path="examples/TAG001.jpg",
            tag_id="TEST_002",
            expected_huid="ABC123"
        )

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
