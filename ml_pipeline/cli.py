"""Command-line interface for repository-local ML pipeline checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import audit_dataset, create_splits
from .evaluation import evaluate_prediction_csv
from .training import main as training_main
from .validation import validate_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate dataset manifests")
    validate_parser.add_argument("--dataset", type=Path, help="dataset directory to audit")
    validate_parser.add_argument("--dataset-root", type=Path)
    validate_parser.add_argument("--train", type=Path)
    validate_parser.add_argument("--val", type=Path)
    validate_parser.add_argument("--test", type=Path)
    validate_parser.add_argument("--allow-group-overlap", action="store_true")
    validate_parser.add_argument("--create-splits", action="store_true")
    validate_parser.add_argument("--output-dir", type=Path)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate prediction/reference CSV")
    evaluate_parser.add_argument("--predictions", required=True, type=Path)

    train_parser = subparsers.add_parser("train", help="validate and plan training")
    train_parser.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    if args.command == "validate":
        if args.dataset:
            if args.create_splits:
                if not args.output_dir:
                    parser.error("--output-dir is required with --create-splits")
                manifests = create_splits(args.dataset, args.output_dir)
                print(json.dumps({key: str(path) for key, path in manifests.items()}, indent=2, sort_keys=True))
                return 0
            audit = audit_dataset(args.dataset)
            print(audit.to_json())
            return 0 if not audit.to_dict()["errors"] else 1
        if not args.dataset_root:
            parser.error("--dataset-root is required when validating manifests")
        manifests = {k: v for k, v in {"train": args.train, "val": args.val, "test": args.test}.items() if v}
        report = validate_splits(manifests, args.dataset_root, fail_on_group_overlap=not args.allow_group_overlap)
        print(report.to_json())
        return 0 if report.ok else 1
    if args.command == "evaluate":
        print(json.dumps(evaluate_prediction_csv(args.predictions), indent=2, sort_keys=True))
        return 0
    if args.command == "train":
        return training_main(args.args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
