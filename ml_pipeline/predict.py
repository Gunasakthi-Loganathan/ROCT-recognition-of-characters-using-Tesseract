"""Prediction CLI for local OCR model smoke tests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .models import TemplateClassifier
from .preprocessing import preprocess_file


def predict(model_path: str | Path, image_path: str | Path) -> dict[str, object]:
    model = TemplateClassifier.load(model_path)
    image = preprocess_file(image_path, size=model.image_size)
    started = time.perf_counter()
    result = model.predict(image)
    return {
        "text": result.text,
        "confidence": result.confidence,
        "engine": result.engine,
        "latency_ms": (time.perf_counter() - started) * 1000,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict an isolated character with a local template model")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(predict(args.model, args.image), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
