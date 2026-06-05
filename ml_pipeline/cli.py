"""Command-line interface for repository-local ML pipeline checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import evaluate_prediction_csv
from .training import main as training_main
from .validation import validate_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate dataset manifests")
    validate_parser.add_argument("--dataset-root", required=True, type=Path)
    validate_parser.add_argument("--train", type=Path)
    validate_parser.add_argument("--val", type=Path)
    validate_parser.add_argument("--test", type=Path)
    validate_parser.add_argument("--allow-group-overlap", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate prediction/reference CSV")
    evaluate_parser.add_argument("--predictions", required=True, type=Path)

    train_parser = subparsers.add_parser("train", help="validate and plan training")
    train_parser.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    if args.command == "validate":
        manifests = {k: v for k, v in {"train": args.train, "val": args.val, "test": args.test}.items() if v}
        report = validate_splits(
            manifests,
            args.dataset_root,
            fail_on_group_overlap=not args.allow_group_overlap,
        )
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
