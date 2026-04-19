# Hallmark QC

AI-assisted quality control for BIS hallmarked jewellery. Upload an Excel of `tag_id` / `expected_huid` pairs, upload the corresponding HUID image for each tag, and the service OCR's the image, extracts the HUID + purity code, and compares against the expected value. Each tag can also carry up to three artifact images for reference. The whole thing is a single FastAPI app serving a lightweight SPA at `/`.

## Run locally

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env — at minimum set AWS_* + S3_BUCKET_NAME if you want S3 storage.
# Without S3, everything falls back to ./uploads/.

uvicorn api:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/> — the dashboard mounts on `/`. The OpenAPI explorer lives at `/docs`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers batch parsing (happy paths, header variants, duplicate / invalid rows, sheet selection) and the Stage 2 / Stage 3 contract (HUID vs artifact routing, item endpoint shape).

## Environment

| Variable | Purpose | Default |
| --- | --- | --- |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | S3 credentials | — |
| `AWS_REGION` | S3 region | `ap-south-1` |
| `S3_BUCKET_NAME` | Target bucket | `hallmark-qc-images` |
| `S3_PREFIX` | Root key prefix | `hallmark-images` |
| `DATABASE_URL` | SQLite path | `sqlite:///./hallmark_qc.db` |
| `SESSION_SECRET` | Cookie signing key | random per-boot |
| `OCR_AUTO_APPROVE` | Auto-approve threshold | `0.85` |
| `OCR_AUTO_REJECT` | Auto-reject threshold | `0.50` |
| `API_HOST`, `API_PORT` | Uvicorn bind | `0.0.0.0:8000` |

When S3 credentials are absent the storage layer silently falls back to `./uploads/`, mirroring the S3 key layout — handy for local dev without AWS access.

## S3 layout

Every image written by `/stage2/upload-image` or `/stage2/upload-artifact` lands at:

```
{S3_PREFIX}/batches/{batch_id}/{tag_id}/huid/{uuid}{ext}
{S3_PREFIX}/batches/{batch_id}/{tag_id}/artifact-{1|2|3}/{uuid}{ext}
```

S3 objects also carry `x-amz-meta-tag-id`, `x-amz-meta-batch-id`, `x-amz-meta-image-type`, and `x-amz-meta-slot` metadata for easy reverse-lookup.

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/` | Dashboard SPA |
| `GET`  | `/api/health` | Health check |
| `POST` | `/stage1/upload-batch?strict=<bool>` | Upload CSV/Excel of `tag_id` + `expected_huid` pairs |
| `GET`  | `/stage1/batches` | List all batches |
| `GET`  | `/stage1/batch/{id}` | Batch summary |
| `POST` | `/stage2/upload-image` | Upload a tag's HUID image (triggers OCR) or an artifact |
| `POST` | `/stage2/upload-image-bulk` | Bulk HUID uploads, auto-matched by filename |
| `POST` | `/stage2/upload-artifact` | Upload an artifact image (no OCR) |
| `DELETE` | `/stage2/artifact/{tag_id}/{slot}` | Delete an artifact (slot 1–3) |
| `GET`  | `/stage3/item/{tag_id}` | Full item payload: metadata + HUID + artifacts with signed URLs |
| `GET`  | `/stage3/batch/{id}/results` | All rows in a batch (includes `thumbnail_url`) |
| `GET`  | `/stage3/search` | Filter by tag / decision / batch |
| `GET`  | `/qc/rules` | BIS validation rules |
| `POST` | `/validate/huid` | Standalone HUID format check |
| `POST` | `/extract/v2` | Standalone OCR (hallmark-aware) |
| `POST` | `/api/get-upload-url`, `/api/get-ocr-upload-url` | Presigned PUT URLs |

## Repo layout

```
api.py                  FastAPI app — all routes live here
config/
  aws_config.py         Env-loaded AWS + DB config
  database.py           SQLite manager + dataclasses
  batch_parse.py        Pure CSV/Excel validator (Phase 3)
  qc_hallmark_config.py BIS rules + purity/HUID validation
src/
  ocr_model_v2.py       PaddleOCR wrapper with hallmark-aware extraction
  qc_service.py         QC orchestration around the OCR engine
  storage_service.py    S3 / local storage with structured key layout
migrations/             Idempotent schema migrations
static/                 Vanilla-JS SPA (light-mode dashboard)
tests/                  pytest — parser + endpoint contract tests
uploads/                Local-mode fallback for S3 objects
```

## Limitations

- Approve / Reject / Request re-capture buttons on the item preview are UI-only for now; wire them to a decision endpoint before rolling out.
- KPI "match rate" card on the batches landing is a placeholder — needs a cross-batch aggregate.
- SQLite only. Works fine up to a few hundred thousand rows; beyond that switch `DATABASE_URL` to Postgres and add an alembic-compatible migration runner (current runner is sqlite3-only).
