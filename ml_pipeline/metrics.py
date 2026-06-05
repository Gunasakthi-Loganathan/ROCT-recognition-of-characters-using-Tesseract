"""Pure-Python OCR evaluation metrics.

The functions intentionally avoid heavy ML dependencies so they can be used in
CI, dataset smoke tests, and small baseline evaluations before a GPU training
job is launched.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


def _edit_distance(source: Sequence[str], target: Sequence[str]) -> int:
    previous = list(range(len(target) + 1))
    for i, source_item in enumerate(source, start=1):
        current = [i] + [0] * len(target)
        for j, target_item in enumerate(target, start=1):
            cost = 0 if source_item == target_item else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[-1]


def _as_list(values: Iterable[str]) -> list[str]:
    return ["" if value is None else str(value) for value in values]


def character_error_rate(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Return total Levenshtein character errors divided by reference characters."""

    preds = _as_list(predictions)
    refs = _as_list(references)
    if len(preds) != len(refs):
        raise ValueError("predictions and references must have the same length")

    total_errors = sum(_edit_distance(list(pred), list(ref)) for pred, ref in zip(preds, refs))
    total_chars = sum(len(ref) for ref in refs)
    return 0.0 if total_chars == 0 else total_errors / total_chars


def word_error_rate(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Return total word-level edit errors divided by reference words."""

    preds = _as_list(predictions)
    refs = _as_list(references)
    if len(preds) != len(refs):
        raise ValueError("predictions and references must have the same length")

    total_errors = sum(
        _edit_distance(pred.split(), ref.split()) for pred, ref in zip(preds, refs)
    )
    total_words = sum(len(ref.split()) for ref in refs)
    return 0.0 if total_words == 0 else total_errors / total_words


def evaluate_prediction_csv(path: str | Path) -> dict[str, float | int]:
    """Evaluate a CSV containing `prediction` and `reference` columns."""

    predictions: list[str] = []
    references: list[str] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"prediction", "reference"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            predictions.append(row["prediction"])
            references.append(row["reference"])

    return {
        "rows": len(predictions),
        "cer": character_error_rate(predictions, references),
        "wer": word_error_rate(predictions, references),
    }
