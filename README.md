# ROCT — Recognition of Characters using Tesseract

ROCT is a React + Vite OCR demonstration app. It keeps browser-based Tesseract.js as the default OCR engine, optionally improves recognized text with a local Express/Gemini correction backend, and includes a reproducible Python MLOps toolkit for dataset validation, preprocessing, deterministic splits, baseline training dry runs, prediction, and evaluation.

The repository does **not** include a real handwritten OCR benchmark dataset or a production-trained CNN/CRNN checkpoint. A tiny synthetic isolated-character fixture is included only for smoke tests. Accuracy claims in this repository are therefore limited to the generated reports under `reports/`.

## Features

- Browser OCR with Tesseract.js, image upload, drag-and-drop, preview, crop, preprocessing, confidence display, low-confidence word reporting, copy/download, and reset.
- Optional Express backend for Gemini OCR text correction with retries, timeout, explicit missing-key handling, and safe error responses.
- Python dataset audit for class distribution, duplicates, missing labels, invalid/corrupted files, dimensions, and split leakage.
- Deterministic train/validation/test manifest generation with a fixed random seed and duplicate grouping.
- Shared dependency-free preprocessing for sample PGM/PPM data: inversion detection, contrast stretch, median denoise, Otsu threshold, and deterministic resize.
- CPU-only template classifier for isolated-character dry runs and smoke tests. Tesseract remains the default engine unless a real held-out dataset proves a better model.
- Evaluation utilities for CER, WER, accuracy, macro/weighted F1, confusion matrix, and confused-character pairs.
- Unit, integration, smoke, dependency, and secret-scan scripts plus GitHub Actions CI.

## Architecture

```text
.
├── .github/workflows/ci.yml        # CI validation workflow
├── configs/train.sample.json       # Verified sample dry-run training config
├── data/sample/characters/         # Tiny synthetic isolated-character smoke dataset
├── ml_pipeline/                    # Python OCR MLOps toolkit
│   ├── dataset.py                  # Dataset audit and deterministic splits
│   ├── evaluation.py               # Metrics reports and model comparison helpers
│   ├── metrics.py                  # CER/WER edit-distance metrics
│   ├── models.py                   # CPU template classifier interface
│   ├── predict.py                  # Local prediction CLI
│   ├── preprocessing.py            # Shared preprocessing for dry runs/inference
│   ├── training.py                 # Template training and dry-run CLI
│   ├── validate.py                 # `python -m ml_pipeline.validate` alias
│   └── validation.py               # Manifest validation and leakage checks
├── reports/                        # Audit, baseline, and comparison reports
├── scripts/                        # Secret scan and report generation scripts
├── server/server.js                # Express Gemini correction backend
├── src/                            # React OCR application
└── tests/                          # Python tests
```

## Prerequisites

- Node.js 24.x was used for verification in this environment.
- npm 11.x was used for verification in this environment.
- Python 3.14 was used for verification in this environment; CI uses Python 3.12.
- Optional: a Gemini API key for backend text correction.
- Optional: local system Tesseract for external benchmarking. The web app uses browser Tesseract.js from CDN at runtime.

## Environment variables

Create a local `.env` from the placeholder file if you want Gemini correction:

```bash
cp .env.example .env
```

Then edit `.env` locally. Never commit real secrets.

Supported variables:

- `PORT` — backend port; default `8000`.
- `CLIENT_ORIGIN` — additional allowed frontend origin; default local Vite origins are already allowed.
- `GEMINI_API_KEY` — required only for Gemini correction.
- `MAX_CORRECTION_TEXT_LENGTH` — backend text limit; default `8000`.
- `GEMINI_TIMEOUT_MS` — provider request timeout; default `15000`.

## Install

```bash
npm ci
```

No Python packages are required for the included ML validation, preprocessing, training dry run, or tests.

## Run the application

Start frontend and backend together:

```bash
npm run dev
```

Or start separately:

```bash
npm run server
npm run client
```

Open the Vite URL, usually `http://localhost:5173`.

## Backend API

- `GET /` returns health and whether Gemini is configured.
- `POST /api/correct` accepts `{ "text": "..." }` and returns `{ correctedText, model, engine }` when `GEMINI_API_KEY` is configured.
- Missing Gemini configuration returns an explicit `503` for correction requests; browser-only OCR still works.

## Dataset format

For real datasets, use either class folders for isolated characters:

```text
dataset/
├── A/img-001.pgm
├── A/img-002.pgm
├── B/img-001.pgm
└── B/img-002.pgm
```

or CSV manifests with required columns:

```csv
image_path,text
samples/page-001-line-001.png,The quick brown fox
```

Optional manifest columns `writer_id` and `document_id` strengthen leakage detection for handwriting datasets.

## Dataset validation

Audit a dataset directory:

```bash
python3 -m ml_pipeline.validate --dataset data/sample/characters
```

Create deterministic splits:

```bash
python3 -m ml_pipeline.validate \
  --dataset data/sample/characters \
  --create-splits \
  --output-dir data/sample/characters_splits
```

Validate existing manifests:

```bash
python3 -m ml_pipeline.cli validate \
  --dataset-root data/sample/characters \
  --train data/sample/characters_splits/train.csv \
  --val data/sample/characters_splits/val.csv \
  --test data/sample/characters_splits/test.csv
```

## Preprocessing

Run the shared preprocessing pipeline and save debug stages:

```bash
python3 -m ml_pipeline.preprocessing \
  --input data/sample/characters/A/A_1.pgm \
  --output reports/A_1_preprocessed.pgm \
  --width 8 \
  --height 8 \
  --debug-dir reports/preprocess_debug
```

The original input is never overwritten.

## Training dry run and local baseline model

Run the verified dry run:

```bash
python3 -m ml_pipeline.training --config configs/train.sample.json --dry-run
```

Run the lightweight template baseline for the sample fixture:

```bash
python3 -m ml_pipeline.training \
  --config configs/train.sample.json \
  --output-dir reports/sample_template_model_full
```

This is a CPU-only isolated-character baseline, not a production handwritten line OCR model.

## Prediction

```bash
python3 -m ml_pipeline.predict \
  --model reports/sample_template_model_full/template_model.json \
  --image data/sample/characters/A/A_1.pgm
```

The output includes recognized text, confidence, engine, and latency.

## Evaluation and model comparison

Evaluate a prediction CSV with `prediction,reference` columns:

```bash
python3 -m ml_pipeline.evaluation --predictions predictions.csv --output-json reports/evaluation.json
```

Regenerate baseline and comparison reports after running sample training:

```bash
python3 scripts/generate_reports.py
```

Current reports:

- `reports/baseline_metrics.json`
- `reports/baseline_report.md`
- `reports/model_comparison.json`
- `reports/model_comparison.md`
- `reports/final_audit.md`

## Testing and verification

```bash
npm run lint
npm run test:all
npm run smoke
npm run build
npm run check:deps
npm run security:secrets
```

`npm run security:audit` is configured but returned HTTP 403 from the registry audit endpoint in this environment. Run it in an environment with audit endpoint access.

## Production build

```bash
npm run build
npm run preview
```

## Security guidance

- Never commit `.env`, real API keys, tokens, passwords, private URLs, private datasets, or private checkpoints.
- Upload limits are enforced in both frontend file handling and backend text correction payloads.
- Backend errors are sanitized and do not intentionally print secrets.
- CORS is restricted to local Vite origins plus optional `CLIENT_ORIGIN`.
- Run `npm run security:secrets` before every commit.

## Troubleshooting

- **Gemini correction returns 503**: set `GEMINI_API_KEY` in a local `.env`; browser OCR still works without it.
- **Tesseract.js fails to load**: check network/CDN access, or vendor Tesseract.js for offline deployments.
- **Dataset validation finds leakage**: remove duplicate image hashes from validation/test splits or split by writer/document groups.
- **No real accuracy improvement is reported**: add a real held-out OCR dataset first, then compare Tesseract against a suitable isolated-character CNN or CRNN/CTC sequence model.

## Known limitations

- The default app OCR engine is still browser Tesseract.js.
- The included sample data is synthetic and too small for real accuracy conclusions.
- No CNN/CRNN is selected because no real dataset is present to justify additional dependencies or architecture complexity.
- Tesseract.js is loaded from CDN unless you vendor it for offline use.

## Real OCR model integration workflow

The current repository data contains only synthetic isolated-character smoke fixtures. It does **not** contain a real handwritten word/line dataset, so this branch does not train or promote a production CNN/CRNN checkpoint. To integrate a real model safely:

1. Add a private or public real dataset under an ignored path such as `data/processed/handwriting/`.
2. Determine the task type:
   - isolated characters: train a CNN or transfer-learning classifier;
   - words/lines: train a CRNN with CTC decoding;
   - mixed pages: keep Tesseract or add segmentation before any classifier.
3. Create `image_path,text` manifests for train/validation/test splits and keep the test set untouched.
4. Validate and split data:

```bash
python3 -m ml_pipeline.validate --dataset data/processed/handwriting
python3 -m ml_pipeline.cli validate \
  --dataset-root data/processed/handwriting \
  --train data/processed/handwriting/train.csv \
  --val data/processed/handwriting/val.csv \
  --test data/processed/handwriting/test.csv
```

5. Train the selected model outside normal app startup and save the best checkpoint under `checkpoints/`.
6. Compare against Tesseract using an untouched test set and update `reports/model_comparison.json`.
7. Set `ML_MODEL_PATH` to the selected checkpoint and start the Python inference backend.

## Python ML inference backend

Start the backend after configuring a model checkpoint:

```bash
ML_MODEL_PATH=checkpoints/real_ocr_model/model.json npm run ml:server
```

Health check:

```bash
curl http://127.0.0.1:9000/
```

Prediction endpoint:

```bash
curl -F "file=@data/sample/characters/A/A_1.pgm" http://127.0.0.1:9000/api/ocr
```

The included lightweight backend accepts PGM/PPM images for the dependency-free template model. JPG/PNG handwritten production inference should be backed by a real trained CNN/CRNN service and the same `/api/ocr` response shape: `{ text, confidence, engine, model, latency_ms }`.

## OCR engine selector in the web app

The visible app now exposes four OCR engines:

- **Auto** — tries the ML backend first and falls back to Tesseract when the ML endpoint is unavailable or low confidence.
- **Tesseract** — browser Tesseract.js only.
- **ML model** — Python `/api/ocr` backend only.
- **Hybrid** — runs Tesseract and ML backend and selects the higher-confidence result.

The result panel shows the selected model/engine, confidence, low-confidence warnings, and an engine-comparison card when multiple engines run.
