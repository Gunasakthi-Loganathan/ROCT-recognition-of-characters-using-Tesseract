# Changelog

## Unreleased

- Added full audit reports and reproducible baseline/model-comparison artifacts.
- Added dependency-free OCR preprocessing, dataset auditing, deterministic splitting, template-model dry runs, evaluation, and prediction CLIs.
- Added a synthetic isolated-character sample dataset for smoke testing without external downloads.
- Hardened the Express Gemini correction backend for missing secrets, timeouts, safer errors, and testability.
- Added CI, environment example, contributor guidance, additional Python/backend tests, and documentation updates.

## OCR model integration branch

- Added a Python `/api/ocr` inference backend scaffold for configured OCR model checkpoints.
- Added a frontend ML OCR client and visible OCR engine selector with Auto, Tesseract, ML model, and Hybrid modes.
- Added confidence-based hybrid fallback and engine-comparison UI.
- Documented the real-dataset workflow required before training or promoting a CNN/CRNN model.
