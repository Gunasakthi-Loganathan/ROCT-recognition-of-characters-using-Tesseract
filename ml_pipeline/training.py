"""Training orchestration helpers for OCR model experiments.

This module keeps expensive ML imports out of normal validation/test runs. Use
`--dry-run` in CI to validate manifests, compute step counts, and verify output
paths before launching a GPU training job.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .validation import validate_splits


@dataclass(slots=True)
class TrainingConfig:
    dataset_root: Path
    train_manifest: Path
    val_manifest: Path
    output_dir: Path = Path("checkpoints/best_model")
    model_name: str = "microsoft/trocr-base-handwritten"
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 5e-5
    seed: int = 42
    max_target_length: int = 128
    num_workers: int = 2
    gradient_accumulation_steps: int = 1
    dry_run: bool = False

    def validate_hyperparameters(self) -> list[str]:
        errors: list[str] = []
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.epochs < 1:
            errors.append("epochs must be >= 1")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.max_target_length < 1:
            errors.append("max_target_length must be >= 1")
        if self.gradient_accumulation_steps < 1:
            errors.append("gradient_accumulation_steps must be >= 1")
        return errors


def build_training_plan(config: TrainingConfig) -> dict[str, object]:
    manifests = {"train": config.train_manifest, "val": config.val_manifest}
    report = validate_splits(manifests, config.dataset_root)
    hyperparameter_errors = config.validate_hyperparameters()
    if hyperparameter_errors:
        report.errors.extend(hyperparameter_errors)

    train_rows = report.split_counts.get("train", 0)
    steps_per_epoch = (train_rows + config.batch_size - 1) // config.batch_size if config.batch_size else 0
    optimizer_steps_per_epoch = (
        (steps_per_epoch + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps
        if config.gradient_accumulation_steps
        else 0
    )

    return {
        "ok": report.ok,
        "validation": report.to_dict(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": optimizer_steps_per_epoch * config.epochs,
    }


def run_training(config: TrainingConfig) -> dict[str, object]:
    """Validate the experiment and either emit a dry-run plan or start training.

    The concrete fine-tuning implementation is intentionally guarded behind a
    non-dry-run path so CI can exercise the safety checks without downloading
    multi-GB model artifacts. The raised error documents the integration point
    for teams that add GPU infrastructure.
    """

    plan = build_training_plan(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    if not plan["ok"]:
        raise SystemExit("dataset validation failed; see training_plan.json")
    if config.dry_run:
        return plan

    raise NotImplementedError(
        "Full TrOCR fine-tuning requires torch/transformers and project-specific GPU settings. "
        "Use ml_pipeline.validation and ml_pipeline.metrics as the tested gate before adding the training backend."
    )


def parse_args(argv: list[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Validate and plan OCR model training")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/best_model"))
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return TrainingConfig(**vars(args))


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    plan = run_training(config)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
