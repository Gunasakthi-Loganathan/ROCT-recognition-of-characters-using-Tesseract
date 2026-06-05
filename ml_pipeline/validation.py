"""Dataset validation and data-leakage safeguards for OCR training manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pgm", ".ppm"}
IMAGE_MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"BM": "bmp",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
    b"RIFF": "webp",
    b"P2": "pgm",
    b"P3": "ppm",
    b"P5": "pgm",
    b"P6": "ppm",
}
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
TEXT_NORMALIZER = re.compile(r"\s+")


@dataclass(slots=True)
class ValidationConfig:
    dataset_root: Path
    max_file_size_mb: float = 25.0
    max_text_length: int = 512
    allow_absolute_paths: bool = False
    fail_on_group_overlap: bool = True


@dataclass(slots=True)
class DatasetRecord:
    split: str
    row_number: int
    image_path: str
    text: str
    resolved_path: Path | None = None
    sha256: str | None = None
    writer_id: str = ""
    document_id: str = ""


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    duplicate_images: list[dict[str, str]] = field(default_factory=list)
    duplicate_texts: list[dict[str, str]] = field(default_factory=list)
    overlapping_groups: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "split_counts": self.split_counts,
            "duplicate_images": self.duplicate_images,
            "duplicate_texts": self.duplicate_texts,
            "overlapping_groups": self.overlapping_groups,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def normalize_text(text: str) -> str:
    return TEXT_NORMALIZER.sub(" ", text.casefold()).strip()


def _is_probably_image(path: Path) -> bool:
    with path.open("rb") as handle:
        header = handle.read(16)
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return True
    return any(header.startswith(prefix) for prefix in IMAGE_MAGIC)


def _safe_resolve_image_path(raw_path: str, config: ValidationConfig) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and not config.allow_absolute_paths:
        raise ValueError("absolute image paths are disabled")
    resolved = (candidate if candidate.is_absolute() else config.dataset_root / candidate).resolve()
    root = config.dataset_root.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("image path escapes dataset_root")
    return resolved


def read_manifest(path: str | Path, split: str) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "text"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            records.append(
                DatasetRecord(
                    split=split,
                    row_number=row_number,
                    image_path=(row.get("image_path") or "").strip(),
                    text=row.get("text") or "",
                    writer_id=(row.get("writer_id") or "").strip(),
                    document_id=(row.get("document_id") or "").strip(),
                )
            )
    return records


def _validate_record(record: DatasetRecord, config: ValidationConfig, report: ValidationReport) -> None:
    prefix = f"{record.split}:row {record.row_number}"
    if not record.image_path:
        report.errors.append(f"{prefix}: image_path is empty")
        return
    if not record.text.strip():
        report.errors.append(f"{prefix}: text label is empty")
    if len(record.text) > config.max_text_length:
        report.errors.append(f"{prefix}: text exceeds {config.max_text_length} characters")
    if CONTROL_CHARS.search(record.text):
        report.errors.append(f"{prefix}: text contains control characters")

    try:
        image_path = _safe_resolve_image_path(record.image_path, config)
    except ValueError as exc:
        report.errors.append(f"{prefix}: {exc}")
        return

    record.resolved_path = image_path
    if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        report.errors.append(f"{prefix}: unsupported image extension {image_path.suffix!r}")
    if not image_path.exists():
        report.errors.append(f"{prefix}: image file does not exist: {record.image_path}")
        return
    if not image_path.is_file():
        report.errors.append(f"{prefix}: image path is not a file")
        return

    size = image_path.stat().st_size
    if size == 0:
        report.errors.append(f"{prefix}: image file is empty")
    if size > config.max_file_size_mb * 1024 * 1024:
        report.errors.append(f"{prefix}: image exceeds {config.max_file_size_mb:g} MB")
    try:
        if not _is_probably_image(image_path):
            report.errors.append(f"{prefix}: file header is not a supported image")
    except OSError as exc:
        report.errors.append(f"{prefix}: could not read image header: {exc}")
        return

    hasher = hashlib.sha256()
    with image_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    record.sha256 = hasher.hexdigest()


def _find_leakage(records: list[DatasetRecord], report: ValidationReport, config: ValidationConfig) -> None:
    by_hash: dict[str, DatasetRecord] = {}
    by_text: dict[str, DatasetRecord] = {}
    by_group: dict[tuple[str, str], str] = {}

    for record in records:
        if record.sha256:
            previous = by_hash.get(record.sha256)
            if previous and previous.split != record.split:
                item = {
                    "sha256": record.sha256,
                    "first": f"{previous.split}:row {previous.row_number}",
                    "second": f"{record.split}:row {record.row_number}",
                }
                report.duplicate_images.append(item)
                report.errors.append(
                    f"image leakage across splits: {item['first']} and {item['second']} share the same file hash"
                )
            else:
                by_hash[record.sha256] = record

        normalized = normalize_text(record.text)
        if normalized and len(normalized) > 2:
            previous = by_text.get(normalized)
            if previous and previous.split != record.split:
                item = {
                    "text": normalized[:80],
                    "first": f"{previous.split}:row {previous.row_number}",
                    "second": f"{record.split}:row {record.row_number}",
                }
                report.duplicate_texts.append(item)
                report.warnings.append(
                    f"possible label leakage: {item['first']} and {item['second']} have identical normalized text"
                )
            else:
                by_text[normalized] = record

        for column_name, value in (("writer_id", record.writer_id), ("document_id", record.document_id)):
            if not value:
                continue
            key = (column_name, value)
            previous_split = by_group.get(key)
            if previous_split and previous_split != record.split:
                item = {"column": column_name, "value": value, "splits": f"{previous_split},{record.split}"}
                report.overlapping_groups.append(item)
                message = f"{column_name}={value!r} appears in both {previous_split} and {record.split}"
                if config.fail_on_group_overlap:
                    report.errors.append(message)
                else:
                    report.warnings.append(message)
            else:
                by_group[key] = record.split


def validate_splits(
    manifests: Mapping[str, str | Path],
    dataset_root: str | Path,
    *,
    max_file_size_mb: float = 25.0,
    max_text_length: int = 512,
    allow_absolute_paths: bool = False,
    fail_on_group_overlap: bool = True,
) -> ValidationReport:
    """Validate train/validation/test manifests and detect cross-split leakage."""

    config = ValidationConfig(
        dataset_root=Path(dataset_root),
        max_file_size_mb=max_file_size_mb,
        max_text_length=max_text_length,
        allow_absolute_paths=allow_absolute_paths,
        fail_on_group_overlap=fail_on_group_overlap,
    )
    report = ValidationReport()
    all_records: list[DatasetRecord] = []

    for split, manifest_path in manifests.items():
        try:
            records = read_manifest(manifest_path, split)
        except (OSError, ValueError) as exc:
            report.errors.append(f"{split}: {exc}")
            continue
        report.split_counts[split] = len(records)
        if not records:
            report.errors.append(f"{split}: manifest has no rows")
        for record in records:
            _validate_record(record, config, report)
        all_records.extend(records)

    if len(report.split_counts) >= 2:
        _find_leakage(all_records, report, config)
    else:
        report.warnings.append("only one split was provided; leakage checks require at least two splits")

    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for manifest validation or dataset-directory auditing."""

    import argparse
    from .dataset import audit_dataset

    parser = argparse.ArgumentParser(description="Validate OCR manifests or audit a dataset directory")
    parser.add_argument("--dataset", type=Path, help="dataset directory to audit")
    parser.add_argument("--dataset-root", type=Path, help="root for manifest image paths")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--test", type=Path)
    args = parser.parse_args(argv)
    if args.dataset:
        audit = audit_dataset(args.dataset)
        print(audit.to_json())
        return 0 if not audit.to_dict()["errors"] else 1
    if not args.dataset_root:
        parser.error("--dataset or --dataset-root is required")
    manifests = {key: value for key, value in {"train": args.train, "val": args.val, "test": args.test}.items() if value}
    report = validate_splits(manifests, args.dataset_root)
    print(report.to_json())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
