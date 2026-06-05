"""Lightweight OCR model interfaces used by dry runs and smoke tests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .preprocessing import GrayImage


@dataclass(slots=True)
class Prediction:
    text: str
    confidence: float
    engine: str


@dataclass(slots=True)
class TemplateClassifier:
    """Dependency-free nearest-centroid classifier for isolated-character datasets.

    This is not selected over Tesseract by default. It provides a measurable,
    deterministic local baseline for small isolated-character datasets and a
    safe CPU-only training dry run when PyTorch is unavailable.
    """

    image_size: tuple[int, int] = (32, 32)
    centroids: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _vector(image: GrayImage) -> list[float]:
        return [value / 255.0 for value in image.pixels]

    def partial_fit(self, image: GrayImage, label: str) -> None:
        vector = self._vector(image)
        if label not in self.centroids:
            self.centroids[label] = vector[:]
            self.counts[label] = 1
            return
        count = self.counts[label]
        self.centroids[label] = [((old * count) + new) / (count + 1) for old, new in zip(self.centroids[label], vector)]
        self.counts[label] = count + 1

    def predict(self, image: GrayImage) -> Prediction:
        if not self.centroids:
            raise ValueError("template classifier has not been trained")
        vector = self._vector(image)
        distances = {
            label: math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, centroid)))
            for label, centroid in self.centroids.items()
        }
        label, distance = min(distances.items(), key=lambda item: item[1])
        max_distance = math.sqrt(len(vector)) or 1.0
        confidence = max(0.0, min(1.0, 1.0 - distance / max_distance))
        return Prediction(text=label, confidence=confidence, engine="template")

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "model_type": "template_classifier",
                    "image_size": list(self.image_size),
                    "centroids": self.centroids,
                    "counts": self.counts,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TemplateClassifier":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("model_type") != "template_classifier":
            raise ValueError("unsupported model type")
        return cls(
            image_size=tuple(data.get("image_size", [32, 32])),
            centroids={str(label): [float(value) for value in values] for label, values in data.get("centroids", {}).items()},
            counts={str(label): int(value) for label, value in data.get("counts", {}).items()},
        )
