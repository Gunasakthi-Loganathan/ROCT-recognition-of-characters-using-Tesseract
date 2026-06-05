import csv
import tempfile
import unittest
from pathlib import Path

from ml_pipeline.dataset import audit_dataset, create_splits
from ml_pipeline.predict import predict
from ml_pipeline.preprocessing import GrayImage, write_pgm
from ml_pipeline.training import TrainingConfig, run_training


def write_sample_dataset(root: Path):
    for label, pixels in {"A": [0, 255, 255, 0], "B": [0, 0, 255, 255]}.items():
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for idx in range(3):
            write_pgm(GrayImage(2, 2, pixels if idx % 2 == 0 else list(reversed(pixels))), folder / f"{label}_{idx}.pgm")


class DatasetTrainingTest(unittest.TestCase):
    def test_audit_and_split_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            write_sample_dataset(root)
            audit = audit_dataset(root).to_dict()
            self.assertEqual(audit["valid_samples"], 6)
            self.assertEqual(audit["number_of_classes"], 2)

            manifests = create_splits(root, Path(tmp) / "splits")
            self.assertIn("train", manifests)
            self.assertTrue(manifests["val"].exists())

    def test_training_dry_run_and_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            splits = Path(tmp) / "splits"
            output = Path(tmp) / "model"
            write_sample_dataset(root)
            manifests = create_splits(root, splits)
            result = run_training(
                TrainingConfig(
                    dataset_root=root,
                    train_manifest=manifests["train"],
                    val_manifest=manifests["val"],
                    output_dir=output,
                    batch_size=1,
                    epochs=1,
                    image_width=2,
                    image_height=2,
                    dry_run=True,
                )
            )
            self.assertTrue(Path(result["model_path"]).exists())
            with manifests["train"].open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            prediction = predict(result["model_path"], root / row["image_path"])
            self.assertIn("text", prediction)
            self.assertIn("confidence", prediction)


if __name__ == "__main__":
    unittest.main()
