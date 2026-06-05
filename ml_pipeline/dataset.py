"""Dataset discovery, validation summaries, and deterministic split creation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .preprocessing import read_netpbm
from .validation import ALLOWED_EXTENSIONS, validate_splits

MANIFEST_NAMES = {"train.csv", "val.csv", "validation.csv", "test.csv", "manifest.csv"}


@dataclass(slots=True)
class DatasetSample:
    image_path: Path
    label: str
    split: str = ""
    sha256: str = ""
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    valid: bool = True
    error: str = ""


@dataclass(slots=True)
class DatasetAudit:
    dataset_path: Path
    samples: list[DatasetSample] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        labels = Counter(sample.label for sample in self.samples if sample.valid and sample.label)
        hashes = Counter(sample.sha256 for sample in self.samples if sample.sha256)
        duplicates = sum(count - 1 for count in hashes.values() if count > 1)
        invalid = [sample for sample in self.samples if not sample.valid]
        dims = Counter(
            f"{sample.width}x{sample.height}x{sample.channels}"
            for sample in self.samples
            if sample.valid and sample.width and sample.height and sample.channels
        )
        return {
            "dataset_path": str(self.dataset_path),
            "total_samples": len(self.samples),
            "valid_samples": len(self.samples) - len(invalid),
            "invalid_samples": len(invalid),
            "number_of_classes": len(labels),
            "samples_per_class": dict(sorted(labels.items())),
            "duplicate_count": duplicates,
            "corrupted_image_count": sum(1 for sample in invalid if "corrupt" in sample.error.lower() or "invalid" in sample.error.lower()),
            "missing_label_count": sum(1 for sample in self.samples if not sample.label),
            "image_dimensions": dict(sorted(dims.items())),
            "errors": self.errors + [f"{sample.image_path}: {sample.error}" for sample in invalid],
            "warnings": self.warnings,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_manifest_samples(manifest: Path, dataset_root: Path, split: str = "") -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"image_path", "text"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{manifest} must contain image_path,text columns")
        for row in reader:
            samples.append(DatasetSample(dataset_root / (row.get("image_path") or ""), row.get("text") or "", split=split))
    return samples


def discover_samples(dataset_path: str | Path) -> list[DatasetSample]:
    """Discover samples from manifests or class-directory layout.

    Supported layouts:
    - `manifest.csv` with `image_path,text` columns.
    - `train.csv`, `val.csv`, `test.csv` manifests.
    - Class folders such as `A/img001.pgm`, `B/img002.pgm`.
    """

    root = Path(dataset_path)
    manifests = [path for path in sorted(root.glob("*.csv")) if path.name in MANIFEST_NAMES]
    if manifests:
        samples: list[DatasetSample] = []
        for manifest in manifests:
            split = manifest.stem.replace("validation", "val") if manifest.stem != "manifest" else ""
            samples.extend(_read_manifest_samples(manifest, root, split=split))
        return samples

    samples = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS.union({".pgm", ".ppm"}):
            continue
        if path.name in MANIFEST_NAMES:
            continue
        label = path.parent.name if path.parent != root else path.stem.split("_")[0]
        samples.append(DatasetSample(path, label))
    return samples


def audit_dataset(dataset_path: str | Path) -> DatasetAudit:
    """Audit a dataset directory and return class, duplicate, and corruption statistics."""

    root = Path(dataset_path)
    audit = DatasetAudit(root)
    if not root.exists():
        audit.errors.append("dataset path does not exist")
        return audit
    if not root.is_dir():
        audit.errors.append("dataset path is not a directory")
        return audit

    samples = discover_samples(root)
    if not samples:
        audit.warnings.append("no samples found; expected manifests or class subdirectories")
    for sample in samples:
        try:
            if not sample.label:
                sample.valid = False
                sample.error = "missing label"
            if not sample.image_path.exists():
                sample.valid = False
                sample.error = "missing image file"
            elif sample.image_path.stat().st_size == 0:
                sample.valid = False
                sample.error = "empty image file"
            else:
                sample.sha256 = _hash_file(sample.image_path)
                if sample.image_path.suffix.lower() in {".pgm", ".ppm"}:
                    image = read_netpbm(sample.image_path)
                    sample.width = image.width
                    sample.height = image.height
                    sample.channels = 1
                else:
                    # Full image decoding is intentionally left to optional tools; validation.py checks headers.
                    sample.width = None
                    sample.height = None
                    sample.channels = None
        except Exception as exc:  # noqa: BLE001 - record all validation failures per sample.
            sample.valid = False
            sample.error = f"invalid or corrupt image: {exc}"
        audit.samples.append(sample)
    return audit


def create_splits(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, Path]:
    """Create deterministic stratified train/val/test manifests with duplicate grouping."""

    if round(train_ratio + val_ratio + test_ratio, 8) != 1.0:
        raise ValueError("split ratios must sum to 1.0")
    audit = audit_dataset(dataset_path)
    if audit.errors:
        raise ValueError("dataset audit failed: " + "; ".join(audit.errors))
    valid_samples = [sample for sample in audit.samples if sample.valid]
    if not valid_samples:
        raise ValueError("no valid samples available for splitting")

    grouped_by_label: dict[str, dict[str, list[DatasetSample]]] = defaultdict(lambda: defaultdict(list))
    for sample in valid_samples:
        grouped_by_label[sample.label][sample.sha256 or str(sample.image_path)].append(sample)

    rng = random.Random(seed)
    split_rows: dict[str, list[DatasetSample]] = {"train": [], "val": [], "test": []}
    for label, groups_by_hash in sorted(grouped_by_label.items()):
        groups = list(groups_by_hash.values())
        rng.shuffle(groups)
        total = len(groups)
        if total == 1:
            boundaries = (1, 1)
        elif total == 2:
            boundaries = (1, 2)
        else:
            train_count = max(1, min(total - 2, round(total * train_ratio)))
            val_count = max(1, min(total - train_count - 1, round(total * val_ratio)))
            boundaries = (train_count, train_count + val_count)
        train_end, val_end = boundaries
        for index, group in enumerate(groups):
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "val"
            else:
                split = "test"
            split_rows[split].extend(group)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Path] = {}
    root = Path(dataset_path).resolve()
    for split, rows in split_rows.items():
        manifest = out / f"{split}.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "text"])
            writer.writeheader()
            for sample in sorted(rows, key=lambda item: str(item.image_path)):
                writer.writerow({"image_path": str(sample.image_path.resolve().relative_to(root)), "text": sample.label})
        manifests[split] = manifest

    nonempty_manifests = {split: path for split, path in manifests.items() if split_rows[split]}
    leakage_report = validate_splits(nonempty_manifests, root, fail_on_group_overlap=True)
    if not leakage_report.ok:
        raise ValueError("created splits failed leakage validation: " + "; ".join(leakage_report.errors))
    (out / "dataset_audit.json").write_text(audit.to_json(), encoding="utf-8")
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit or split an OCR dataset")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--create-splits", action="store_true")
    args = parser.parse_args(argv)
    if args.create_splits:
        if not args.output_dir:
            parser.error("--output-dir is required with --create-splits")
        manifests = create_splits(args.dataset, args.output_dir, seed=args.seed)
        print(json.dumps({key: str(path) for key, path in manifests.items()}, indent=2, sort_keys=True))
    else:
        print(audit_dataset(args.dataset).to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
