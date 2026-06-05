# Image to Text OCR AI

Image to Text OCR AI is a React + Vite OCR application that runs Tesseract.js in the browser and optionally sends the OCR output to a local Express backend for Gemini-powered text correction. The repository also includes a lightweight, dependency-minimal machine-learning pipeline package for validating OCR datasets, detecting data leakage, planning training runs, and evaluating prediction files.

> Security note: do not commit `.env` files, API keys, access tokens, passwords, downloaded datasets, or model checkpoints. The included secret scan is intended as a last line of defense, not a replacement for careful review.

## Current architecture

```text
.
├── ml_pipeline/              # Tested Python ML validation/evaluation helpers
│   ├── cli.py                # `validate`, `evaluate`, and `train` CLI entry points
│   ├── metrics.py            # Pure-Python CER/WER and prediction CSV evaluation
│   ├── training.py           # Training dry-run planner with dataset gates
│   └── validation.py         # Manifest validation and leakage checks
├── scripts/
│   └── check-no-secrets.js   # Local credential-pattern scanner
├── server/
│   └── server.js             # Express Gemini correction API
├── src/
│   ├── App.jsx               # OCR UI and Tesseract workflow
│   ├── geminiCorrector.js    # Frontend client for `/api/correct`
│   ├── ocrTextUtils.js       # Rule-based OCR cleanup utilities
│   ├── components/
│   │   └── MlPipeline.jsx    # Visual ML pipeline documentation in the app
│   └── __tests__/            # Node test-runner unit tests
├── tests/                    # Python unittest coverage for ML helpers
├── package.json
└── vite.config.js
```

## Features

- Browser OCR through Tesseract.js with image preprocessing and selectable page-segmentation modes.
- Optional Gemini text correction through a local Express backend.
- Dataset manifest validation before model training.
- Data leakage safeguards for duplicate image hashes, repeated normalized labels, and overlapping `writer_id` / `document_id` groups.
- Pure-Python CER/WER metrics for model evaluation without requiring GPU dependencies.
- Unit tests for OCR text cleanup, Gemini client behavior, metrics, and dataset validation.
- Dependency and security scripts for `npm ls`, `npm audit`, and committed-secret scanning.

## Prerequisites

- Node.js 18+.
- npm 9+.
- Python 3.10+ for the ML pipeline utilities and Python tests.
- A Gemini API key only if you want backend auto-correction.

## Setup

Install JavaScript dependencies:

```bash
npm install
```

Optional backend configuration: create a local, untracked `.env` file only if you need Gemini correction, and add `GEMINI_API_KEY=<your key>` locally. Never commit real `.env` files.

The repository intentionally does not require Python packages for validation and metric tests. If you later add full TrOCR fine-tuning, install GPU-specific dependencies in a separate virtual environment and keep that environment outside version control.

## Run the app

Start the Vite frontend and Express backend together:

```bash
npm run dev
```

Or start them separately:

```bash
npm run client
npm run server
```

Open the Vite URL, usually `http://localhost:5173`.

## Backend API

The Express backend exposes:

- `GET /` — health check.
- `POST /api/correct` — accepts JSON `{ "text": "..." }` and returns corrected text.

Environment variables:

- `GEMINI_API_KEY` — required for Gemini correction.
- `PORT` — optional backend port; defaults to `8000`.
- `CLIENT_ORIGIN` — optional additional allowed CORS origin.

## Dataset manifest format

Training, validation, and test manifests are CSV files. Required columns:

```csv
image_path,text
samples/page-001-line-001.png,The quick brown fox
```

Recommended optional columns for stronger leakage checks:

```csv
image_path,text,writer_id,document_id
samples/page-001-line-001.png,The quick brown fox,writer-001,page-001
```

Rules enforced by `ml_pipeline.validation`:

- `image_path` and `text` must be present and non-empty.
- Image paths must stay inside `--dataset-root` unless absolute paths are explicitly enabled in code.
- File extensions and file headers must look like supported images.
- Labels cannot contain control characters and must stay below the configured maximum length.
- Identical image hashes across splits fail validation.
- Identical normalized text across splits is reported as possible leakage.
- Reused `writer_id` or `document_id` across splits fails validation by default.

## Validate a dataset

```bash
python3 -m ml_pipeline.cli validate \
  --dataset-root data/processed \
  --train data/train.csv \
  --val data/val.csv \
  --test data/test.csv
```

The command prints a JSON report and exits with status `1` if validation fails.

## Plan training

Use a dry run before launching any expensive model job:

```bash
python3 -m ml_pipeline.training \
  --dataset-root data/processed \
  --train-manifest data/train.csv \
  --val-manifest data/val.csv \
  --output-dir checkpoints/best_model \
  --batch-size 8 \
  --epochs 10 \
  --dry-run
```

The dry run validates the data, checks hyperparameters, calculates step counts, and writes `training_plan.json` under the output directory. The repository currently provides the tested validation/planning gate, but does not download or fine-tune large TrOCR models during CI. Add GPU-specific training code behind this validation gate if you deploy model training infrastructure.

## Evaluate predictions

Create a prediction CSV with `prediction` and `reference` columns:

```csv
prediction,reference
hello world,hello world
helo world,hello world
```

Run:

```bash
python3 -m ml_pipeline.cli evaluate --predictions predictions.csv
```

The output includes row count, character error rate (CER), and word error rate (WER).

## Testing, dependency checks, and security checks

Run frontend/unit tests:

```bash
npm test
```

Run Python ML tests:

```bash
npm run test:python
```

Run all unit tests:

```bash
npm run test:all
```

Check installed JavaScript dependency tree:

```bash
npm run check:deps
```

Scan committed files for credential-like content:

```bash
npm run security:secrets
```

Run npm's vulnerability audit:

```bash
npm run security:audit
```

Run both security checks:

```bash
npm run security
```

Build production assets:

```bash
npm run build
```

## Remaining known issues

- Tesseract.js is loaded from a CDN at runtime; offline usage requires vendoring or bundling that dependency.
- Full TrOCR fine-tuning is not implemented in this repository because it requires large model downloads and GPU-specific dependencies. The included pipeline focuses on the validation, leakage prevention, planning, and evaluation steps that should gate that training job.
- Gemini correction requires a local `GEMINI_API_KEY`; missing or invalid keys will make `/api/correct` fail while browser-only OCR remains usable.
