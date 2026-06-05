"""Training orchestration for reproducible OCR dry runs and template baselines."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .evaluation import classification_report
from .models import TemplateClassifier
from .preprocessing import preprocess_file
from .validation import validate_splits


@dataclass(slots=True)
class TrainingConfig:
    dataset_root: Path
    train_manifest: Path
    val_manifest: Path
    output_dir: Path = Path("checkpoints/template_model")
    model_name: str = "template-isolated-character"
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 0.0
    seed: int = 42
    image_width: int = 32
    image_height: int = 32
    patience: int = 3
    dry_run: bool = False

    def validate_hyperparameters(self) -> list[str]:
        errors: list[str] = []
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.epochs < 1:
            errors.append("epochs must be >= 1")
        if self.image_width < 1 or self.image_height < 1:
            errors.append("image dimensions must be positive")
        if self.patience < 1:
            errors.append("patience must be >= 1")
        return errors


def _load_config_file(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json", ""}:
        return json.loads(text)
    # Minimal YAML subset for key: value configs without adding PyYAML.
    data: dict[str, object] = {}
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"unsupported config line: {line!r}")
        key, value = [part.strip() for part in stripped.split(":", 1)]
        if value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
        else:
            try:
                data[key] = int(value)
            except ValueError:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value.strip('"\'')
    return data


def _read_manifest(manifest: Path, dataset_root: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"image_path", "text"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{manifest} must contain image_path,text columns")
        for row in reader:
            rows.append((dataset_root / (row.get("image_path") or ""), row.get("text") or ""))
    return rows


def build_training_plan(config: TrainingConfig) -> dict[str, object]:
    manifests = {"train": config.train_manifest, "val": config.val_manifest}
    report = validate_splits(manifests, config.dataset_root)
    report.errors.extend(config.validate_hyperparameters())
    train_rows = report.split_counts.get("train", 0)
    steps_per_epoch = (train_rows + config.batch_size - 1) // config.batch_size if config.batch_size else 0
    return {
        "ok": report.ok,
        "validation": report.to_dict(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "steps_per_epoch": steps_per_epoch,
        "total_steps": steps_per_epoch * config.epochs,
        "model_family": "isolated-character-template",
        "default_engine_after_training": "tesseract remains default unless validation metrics prove improvement",
    }


def _fit_template_classifier(config: TrainingConfig, *, max_batches: int | None = None) -> dict[str, object]:
    rng = random.Random(config.seed)
    train_rows = _read_manifest(config.train_manifest, config.dataset_root)
    val_rows = _read_manifest(config.val_manifest, config.dataset_root)
    rng.shuffle(train_rows)
    model = TemplateClassifier(image_size=(config.image_width, config.image_height))
    history: list[dict[str, object]] = []
    best_accuracy = -1.0
    stale_epochs = 0
    best_path = config.output_dir / "template_model.json"

    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        batches_seen = 0
        for start in range(0, len(train_rows), config.batch_size):
            batch = train_rows[start : start + config.batch_size]
            for image_path, label in batch:
                image = preprocess_file(image_path, size=(config.image_width, config.image_height))
                model.partial_fit(image, label)
            batches_seen += 1
            if max_batches is not None and batches_seen >= max_batches:
                break

        predictions: list[str] = []
        references: list[str] = []
        latencies: list[float] = []
        for image_path, label in val_rows:
            image = preprocess_file(image_path, size=(config.image_width, config.image_height))
            t0 = time.perf_counter()
            prediction = model.predict(image)
            latencies.append((time.perf_counter() - t0) * 1000)
            predictions.append(prediction.text)
            references.append(label)
        metrics = classification_report(predictions, references) if references else {"character_accuracy": 0.0}
        metrics["latency_ms_avg"] = sum(latencies) / len(latencies) if latencies else 0.0
        history.append({"epoch": epoch, "validation": metrics})
        accuracy = float(metrics.get("character_accuracy", 0.0))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            stale_epochs = 0
            model.save(best_path)
        else:
            stale_epochs += 1
        if config.dry_run or stale_epochs >= config.patience:
            break

    return {
        "model_path": str(best_path),
        "history": history,
        "best_validation_accuracy": best_accuracy,
        "runtime_ms": (time.perf_counter() - started) * 1000,
        "dry_run_completed": config.dry_run,
    }


def run_training(config: TrainingConfig) -> dict[str, object]:
    """Validate data and train or dry-run a CPU-only isolated-character template model."""

    plan = build_training_plan(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "training_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    if not plan["ok"]:
        raise SystemExit("dataset validation failed; see training_plan.json")
    result = _fit_template_classifier(config, max_batches=1 if config.dry_run else None)
    result["plan"] = plan
    (config.output_dir / "training_history.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train or dry-run an OCR isolated-character baseline")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    values = vars(args)
    config_path = values.pop("config")
    if config_path:
        file_values = _load_config_file(config_path)
        file_values.update({key: value for key, value in values.items() if value is not None and not (key == "dry_run" and value is False)})
        values = file_values
    missing = [key for key in ("dataset_root", "train_manifest", "val_manifest") if not values.get(key)]
    if missing:
        parser.error("missing required arguments: " + ", ".join(f"--{key.replace('_', '-')}" for key in missing))
    path_keys = {"dataset_root", "train_manifest", "val_manifest", "output_dir"}
    normalized = {key: Path(value) if key in path_keys else value for key, value in values.items()}
    return TrainingConfig(**normalized)


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    result = run_training(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
