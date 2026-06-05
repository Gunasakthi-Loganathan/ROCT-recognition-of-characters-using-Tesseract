import csv
import tempfile
import unittest
from pathlib import Path

from ml_pipeline.validation import validate_splits

PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"0" * 32


def write_manifest(path, rows):
    fieldnames = ["image_path", "text", "writer_id", "document_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ValidationTest(unittest.TestCase):
    def test_valid_split_manifests_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.png").write_bytes(PNG_BYTES + b"a")
            (root / "val.png").write_bytes(PNG_BYTES + b"b")
            train = root / "train.csv"
            val = root / "val.csv"
            write_manifest(train, [{"image_path": "train.png", "text": "hello", "writer_id": "w1", "document_id": "d1"}])
            write_manifest(val, [{"image_path": "val.png", "text": "world", "writer_id": "w2", "document_id": "d2"}])

            report = validate_splits({"train": train, "val": val}, root)

            self.assertTrue(report.ok, report.to_json())
            self.assertEqual(report.split_counts, {"train": 1, "val": 1})

    def test_duplicate_image_across_splits_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "same.png").write_bytes(PNG_BYTES)
            train = root / "train.csv"
            val = root / "val.csv"
            write_manifest(train, [{"image_path": "same.png", "text": "hello", "writer_id": "w1", "document_id": "d1"}])
            write_manifest(val, [{"image_path": "same.png", "text": "different", "writer_id": "w2", "document_id": "d2"}])

            report = validate_splits({"train": train, "val": val}, root)

            self.assertFalse(report.ok)
            self.assertTrue(report.duplicate_images)

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "train.csv"
            write_manifest(manifest, [{"image_path": "../secret.png", "text": "hello", "writer_id": "w1", "document_id": "d1"}])

            report = validate_splits({"train": manifest}, root)

            self.assertFalse(report.ok)
            self.assertIn("escapes dataset_root", "\n".join(report.errors))


if __name__ == "__main__":
    unittest.main()
