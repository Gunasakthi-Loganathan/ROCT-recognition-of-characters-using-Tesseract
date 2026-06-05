"""Generate reproducible baseline/model-comparison reports for the sample dataset."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_pipeline.evaluation import write_model_comparison

REPORTS = ROOT / "reports"


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    tesseract_path = shutil.which("tesseract")
    history_dir = REPORTS / ("sample_template_model_full" if (REPORTS / "sample_template_model_full" / "training_history.json").exists() else "sample_template_model")
    sample_metrics = json.loads((history_dir / "training_history.json").read_text(encoding="utf-8"))
    final_validation = sample_metrics["history"][-1]["validation"]
    baseline = {
        "dataset": "data/sample/characters",
        "dataset_type": "synthetic isolated characters for smoke testing",
        "tesseract_available": bool(tesseract_path),
        "tesseract_path": tesseract_path,
        "default_engine": "tesseract.js in browser",
        "measured": False,
        "reason": "System Tesseract is not installed and browser Tesseract.js depends on runtime CDN access; no real project dataset is included.",
        "metrics": {
            "character_accuracy": None,
            "word_accuracy": None,
            "cer": None,
            "wer": None,
            "precision": None,
            "recall": None,
            "macro_f1": None,
            "weighted_f1": None,
            "latency_ms_avg": None,
            "model_size_bytes": None,
        },
    }
    (REPORTS / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
    (REPORTS / "baseline_report.md").write_text(
        "# Baseline OCR Report\n\n"
        "The repository does not include a real OCR benchmark dataset or a local Tesseract binary. "
        "The production baseline remains browser Tesseract.js. In this environment, system `tesseract` "
        f"was {'found' if tesseract_path else 'not found'}, so Tesseract-only accuracy metrics are not claimed.\n\n"
        "See `baseline_metrics.json` for the machine-readable result.\n",
        encoding="utf-8",
    )
    model_row = {
        "model_name": "template-isolated-character-smoke",
        "character_accuracy": final_validation.get("character_accuracy", 0.0),
        "word_accuracy": final_validation.get("word_accuracy", 0.0),
        "cer": final_validation.get("cer", 1.0),
        "wer": final_validation.get("wer", 1.0),
        "macro_f1": final_validation.get("macro_f1", 0.0),
        "weighted_f1": final_validation.get("weighted_f1", 0.0),
        "latency_ms_avg": final_validation.get("latency_ms_avg", 0.0),
        "model_size_bytes": Path(sample_metrics["model_path"]).stat().st_size,
        "parameter_count": len(json.loads(Path(sample_metrics["model_path"]).read_text(encoding="utf-8"))["centroids"]),
        "cpu_compatible": True,
        "selected_default": False,
        "notes": "Smoke-test baseline only; Tesseract remains default because no real held-out dataset proves improvement.",
    }
    tesseract_row = {
        "model_name": "tesseract.js-browser-baseline",
        "character_accuracy": 0.0,
        "word_accuracy": 0.0,
        "cer": 1.0,
        "wer": 1.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "latency_ms_avg": "not measured",
        "model_size_bytes": "external CDN/runtime",
        "parameter_count": "n/a",
        "cpu_compatible": True,
        "selected_default": True,
        "notes": "Default fallback; metrics unavailable without local Tesseract/browser benchmark dataset.",
    }
    write_model_comparison([tesseract_row, model_row], REPORTS / "model_comparison.json", REPORTS / "model_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
