"""Evaluation helpers and CLI for OCR predictions and isolated-character models."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .metrics import character_error_rate, word_error_rate


def classification_report(predictions: Iterable[str], references: Iterable[str]) -> dict[str, object]:
    preds = [str(item) for item in predictions]
    refs = [str(item) for item in references]
    if len(preds) != len(refs):
        raise ValueError("predictions and references must have the same length")
    labels = sorted(set(preds) | set(refs))
    matrix = {label: {inner: 0 for inner in labels} for label in labels}
    for pred, ref in zip(preds, refs):
        matrix[ref][pred] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    macro_f1 = 0.0
    weighted_f1 = 0.0
    total_support = len(refs)
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        support = sum(matrix[label].values())
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        macro_f1 += f1
        weighted_f1 += f1 * support
    macro_f1 = macro_f1 / len(labels) if labels else 0.0
    weighted_f1 = weighted_f1 / total_support if total_support else 0.0
    correct = sum(1 for pred, ref in zip(preds, refs) if pred == ref)
    return {
        "samples": len(refs),
        "character_accuracy": 0.0 if not refs else correct / len(refs),
        "word_accuracy": 0.0 if not refs else correct / len(refs),
        "cer": character_error_rate(preds, refs),
        "wer": word_error_rate(preds, refs),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "most_confused": most_confused_pairs(matrix),
    }


def most_confused_pairs(matrix: dict[str, dict[str, int]], limit: int = 10) -> list[dict[str, object]]:
    pairs: list[tuple[int, str, str]] = []
    for reference, row in matrix.items():
        for prediction, count in row.items():
            if reference != prediction and count:
                pairs.append((count, reference, prediction))
    pairs.sort(reverse=True)
    return [{"reference": ref, "prediction": pred, "count": count} for count, ref, pred in pairs[:limit]]


def evaluate_prediction_csv(path: str | Path) -> dict[str, object]:
    predictions: list[str] = []
    references: list[str] = []
    latencies: list[float] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"prediction", "reference"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            predictions.append(row["prediction"])
            references.append(row["reference"])
            if row.get("latency_ms"):
                latencies.append(float(row["latency_ms"]))
    report = classification_report(predictions, references)
    if latencies:
        report["latency_ms_avg"] = sum(latencies) / len(latencies)
        report["latency_ms_p95"] = sorted(latencies)[max(0, round(0.95 * len(latencies)) - 1)]
    return report


def write_model_comparison(rows: list[dict[str, object]], json_path: str | Path, md_path: str | Path) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps({"models": rows}, indent=2, sort_keys=True), encoding="utf-8")
    headers = ["Model", "Char Accuracy", "Word Accuracy", "CER", "WER", "Macro F1", "Latency ms", "Default"]
    lines = ["# Model Comparison", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| {name} | {char:.4f} | {word:.4f} | {cer:.4f} | {wer:.4f} | {f1:.4f} | {latency} | {default} |".format(
                name=row.get("model_name", ""),
                char=float(row.get("character_accuracy", 0.0)),
                word=float(row.get("word_accuracy", 0.0)),
                cer=float(row.get("cer", 0.0)),
                wer=float(row.get("wer", 0.0)),
                f1=float(row.get("macro_f1", 0.0)),
                latency=row.get("latency_ms_avg", "n/a"),
                default="yes" if row.get("selected_default") else "no",
            )
        )
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate OCR predictions")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    report = evaluate_prediction_csv(args.predictions)
    report["evaluation_runtime_ms"] = (time.perf_counter() - start) * 1000
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
