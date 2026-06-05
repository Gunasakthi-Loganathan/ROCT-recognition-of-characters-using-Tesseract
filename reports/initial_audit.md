# Initial Technical Audit

Generated: 2026-06-05T14:07:43Z

## Inspection commands executed before modification
- `git checkout -b audit/full-project-ml-improvement`
- `find . -maxdepth 4 -type f ...`
- `rg -n ...` for APIs, secrets, Tesseract, upload handling, unsafe patterns
- `sed -n ...` over package, server, frontend, ML pipeline, tests, and docs

## Current architecture
- React + Vite single-page OCR UI in `src/App.jsx`.
- Tesseract.js is loaded dynamically from a CDN in the browser; OCR happens client-side.
- Optional Express backend in `server/server.js` exposes Gemini OCR correction at `/api/correct`.
- `ml_pipeline/` contains dependency-light Python validation, metrics, and training planning utilities from the prior change.
- Tests exist for Python metrics/validation and frontend utility/API-client functions.

## Application execution flow
1. User uploads an image in the React UI.
2. Browser validates MIME type and size.
3. Optional canvas preprocessing runs in `App.jsx`.
4. Tesseract.js is loaded from CDN and invoked with selected PSM mode(s).
5. Rule-based cleanup runs locally.
6. Gemini correction is attempted through the Express backend; local cleanup remains as fallback.

## OCR and ML pipeline flow
- Browser OCR baseline is Tesseract.js only.
- Python ML utilities validate manifests and calculate text metrics.
- No real dataset, trained checkpoint, CNN, CRNN, or full training implementation is present.
- Prior training module only creates a plan and raises `NotImplementedError` for non-dry-run execution.

## Existing features
- Image upload, drag-and-drop, preview, crop, preprocessing, Tesseract OCR, confidence display, low-confidence words, copy/download/reset UI, dark mode.
- Gemini correction backend with model retry list.
- Python CER/WER metrics and split leakage checks.
- Unit tests and secret scan script.

## Findings by severity

### Critical
- No included dataset or trained model exists; measurable handwritten-character accuracy improvement cannot be proven on a project dataset.
- Full training path is incomplete for real model training; non-dry-run training raises `NotImplementedError`.
- Backend constructs a Gemini client even when `GEMINI_API_KEY` is missing, causing correction requests to fail later instead of returning an explicit configuration response.

### High
- Tesseract.js is loaded from CDN at runtime, creating offline/deployment fragility and supply-chain dependency on a remote script.
- No backend OCR endpoint exists; only Gemini correction is served. Frontend OCR cannot be integration-tested without a browser and network CDN.
- Dataset validation lacks single-dataset discovery/splitting, class distribution, dimensions, corrupted-image counts, and synthetic dry-run support.
- No preprocessing module exists in Python; preprocessing is duplicated only in browser canvas logic and cannot be shared with training/inference.
- No CI workflow exists.

### Medium
- README overstates training capabilities compared with implementation.
- `src/components/MlPipeline.jsx` contains large illustrative code snippets that reference FastAPI/TrOCR files not present in the repository.
- No linting/formatting commands exist.
- API requests have no client-side timeout.
- Backend logs full error objects, increasing risk of leaking provider internals.
- No `.env.example` exists.

### Low
- Large `App.jsx` remains difficult to maintain.
- No checked-in sample image/dataset for smoke testing.
- No changelog/contributing documentation.
- No machine-readable baseline/model comparison reports.

## Decision constraints
- Do not download large datasets or paid models.
- Do not claim recognition accuracy improvements without a real held-out dataset.
- Keep Tesseract as the default baseline/fallback.
- Implement lightweight, deterministic, testable scaffolding that supports future CNN/CRNN work when suitable data is added.
