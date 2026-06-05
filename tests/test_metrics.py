import csv
import tempfile
import unittest
from pathlib import Path

from ml_pipeline.metrics import character_error_rate, evaluate_prediction_csv, word_error_rate


class MetricsTest(unittest.TestCase):
    def test_character_error_rate(self):
        self.assertAlmostEqual(character_error_rate(["helo"], ["hello"]), 0.2)

    def test_word_error_rate(self):
        self.assertAlmostEqual(word_error_rate(["hello there"], ["hello world"]), 0.5)

    def test_evaluate_prediction_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["prediction", "reference"])
                writer.writeheader()
                writer.writerow({"prediction": "hello", "reference": "hello"})
            self.assertEqual(evaluate_prediction_csv(path), {"rows": 1, "cer": 0.0, "wer": 0.0})


if __name__ == "__main__":
    unittest.main()
