"""Endpoint contract tests — Stage 2 upload routing + Stage 3 item shape.

These run against a fresh SQLite DB per test and use the stubbed OCR engine
from conftest.py so PaddleOCR never loads.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Fresh app instance per test: scratch DB + scratch uploads dir."""
    # Relocate SQLite + uploads to a tmp dir so tests are isolated.
    db_path = tmp_path / "qc.db"
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    # Nuke any cached singletons from previous tests so config is re-read.
    import importlib
    import sys
    for mod in ["api", "database", "storage_service", "aws_config"]:
        sys.modules.pop(mod, None)

    import api as api_module                 # noqa: F401  imports trigger setup
    importlib.reload(api_module)

    with TestClient(api_module.app) as c:
        yield c


# --- helpers ---------------------------------------------------------------

def _csv_payload(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode()


def _png_bytes(color: str = "red") -> bytes:
    img = Image.new("RGB", (16, 16), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _seed_batch(client: TestClient, rows: list[dict]) -> int:
    resp = client.post(
        "/stage1/upload-batch",
        files={"file": ("batch.csv", _csv_payload(rows), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["batch_id"]


# --- tests ---------------------------------------------------------------

class TestStage2Routing:
    def test_huid_upload_runs_ocr_and_writes_result(self, client):
        _seed_batch(client, [{"tag_id": "TAG001", "expected_huid": "22K916YY6DUG"}])

        resp = client.post(
            "/stage2/upload-image",
            data={"tag_id": "TAG001", "image_type": "huid", "slot": "0"},
            files={"file": ("TAG001.png", _png_bytes(), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["actual_huid"] == "YY6DUG"
        assert body["huid_match"] is True

        # The OCR-driven fields should now show up on the item endpoint too.
        item = client.get("/stage3/item/TAG001").json()
        assert item["actual_huid"] == "YY6DUG"
        assert item["confidence"] == 0.92
        assert item["images"]["huid"] is not None

    def test_artifact_upload_skips_ocr(self, client):
        _seed_batch(client, [{"tag_id": "TAG002", "expected_huid": "22K916YY6DUG"}])

        import api as api_module
        # Swap in a sentinel engine that fails the test if called.
        engine_before = api_module.ocr_engine_v2

        called = {"n": 0}
        class Guard:
            def extract_with_hallmark_info(self, image):
                called["n"] += 1
                raise AssertionError("OCR must not run for artifacts")
        api_module.ocr_engine_v2 = Guard()
        try:
            resp = client.post(
                "/stage2/upload-artifact",
                data={"tag_id": "TAG002", "slot": "2"},
                files={"file": ("art.png", _png_bytes("blue"), "image/png")},
            )
        finally:
            api_module.ocr_engine_v2 = engine_before

        assert resp.status_code == 200, resp.text
        assert called["n"] == 0
        assert resp.json()["slot"] == 2

    def test_artifact_slot_out_of_range_rejected(self, client):
        _seed_batch(client, [{"tag_id": "TAG003", "expected_huid": "22K916YY6DUG"}])
        resp = client.post(
            "/stage2/upload-artifact",
            data={"tag_id": "TAG003", "slot": "4"},
            files={"file": ("art.png", _png_bytes(), "image/png")},
        )
        assert resp.status_code == 400

    def test_upload_image_with_artifact_type_skips_ocr(self, client):
        _seed_batch(client, [{"tag_id": "TAG004", "expected_huid": "22K916YY6DUG"}])

        import api as api_module
        class Guard:
            def extract_with_hallmark_info(self, image):
                raise AssertionError("OCR must not run for artifact image_type")
        api_module.ocr_engine_v2 = Guard()

        resp = client.post(
            "/stage2/upload-image",
            data={"tag_id": "TAG004", "image_type": "artifact", "slot": "1"},
            files={"file": ("art.png", _png_bytes("green"), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["decision"] == "pending"
        assert body["actual_huid"] is None


class TestItemEndpoint:
    def test_returns_404_for_unknown_tag(self, client):
        resp = client.get("/stage3/item/NOPE")
        assert resp.status_code == 404

    def test_payload_includes_huid_and_artifacts(self, client):
        _seed_batch(client, [{"tag_id": "TAG005", "expected_huid": "22K916YY6DUG"}])

        client.post(
            "/stage2/upload-image",
            data={"tag_id": "TAG005", "image_type": "huid", "slot": "0"},
            files={"file": ("TAG005.png", _png_bytes(), "image/png")},
        )
        client.post(
            "/stage2/upload-artifact",
            data={"tag_id": "TAG005", "slot": "1"},
            files={"file": ("a1.png", _png_bytes("blue"), "image/png")},
        )
        client.post(
            "/stage2/upload-artifact",
            data={"tag_id": "TAG005", "slot": "3"},
            files={"file": ("a3.png", _png_bytes("green"), "image/png")},
        )

        body = client.get("/stage3/item/TAG005").json()
        assert body["tag_id"] == "TAG005"
        assert body["batch"]["name"]
        assert body["images"]["huid"]["url"].endswith(".png")

        slots = [a["slot"] for a in body["images"]["artifacts"]]
        assert slots == [1, 3]  # sorted, no slot 2
        for art in body["images"]["artifacts"]:
            assert art["url"]  # presigned (local fallback → /uploads/...)
            assert art["content_type"] == "image/png"

    def test_deleting_artifact_removes_it_from_payload(self, client):
        _seed_batch(client, [{"tag_id": "TAG006", "expected_huid": "22K916YY6DUG"}])
        client.post(
            "/stage2/upload-image",
            data={"tag_id": "TAG006", "image_type": "huid", "slot": "0"},
            files={"file": ("TAG006.png", _png_bytes(), "image/png")},
        )
        client.post(
            "/stage2/upload-artifact",
            data={"tag_id": "TAG006", "slot": "2"},
            files={"file": ("a2.png", _png_bytes(), "image/png")},
        )
        assert client.delete("/stage2/artifact/TAG006/2").status_code == 200

        body = client.get("/stage3/item/TAG006").json()
        assert body["images"]["artifacts"] == []


class TestBatchListing:
    def test_batch_results_include_thumbnail_url(self, client):
        batch_id = _seed_batch(client, [
            {"tag_id": "TAG007", "expected_huid": "22K916YY6DUG"},
            {"tag_id": "TAG008", "expected_huid": "22K916ZZ9PQR"},
        ])
        client.post(
            "/stage2/upload-image",
            data={"tag_id": "TAG007", "image_type": "huid", "slot": "0"},
            files={"file": ("x.png", _png_bytes(), "image/png")},
        )
        body = client.get(f"/stage3/batch/{batch_id}/results").json()
        by_tag = {r["tag_id"]: r for r in body["results"]}
        assert by_tag["TAG007"]["thumbnail_url"]
        assert by_tag["TAG008"]["thumbnail_url"] is None
