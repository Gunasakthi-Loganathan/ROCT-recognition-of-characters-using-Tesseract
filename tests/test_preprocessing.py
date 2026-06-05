import tempfile
import unittest
from pathlib import Path

from ml_pipeline.preprocessing import GrayImage, autocontrast, preprocess_file, threshold_otsu, write_pgm


class PreprocessingTest(unittest.TestCase):
    def test_preprocess_file_preserves_original_and_writes_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.pgm"
            output = root / "output.pgm"
            debug = root / "debug"
            write_pgm(GrayImage(2, 2, [0, 255, 255, 0]), source)

            image = preprocess_file(source, output, size=(4, 4), debug_dir=debug)

            self.assertEqual((image.width, image.height), (4, 4))
            self.assertTrue(output.exists())
            self.assertTrue((debug / "input-06-resized.pgm").exists())
            self.assertIn("2 2", source.read_text(encoding="ascii"))

    def test_threshold_and_contrast_are_deterministic(self):
        image = autocontrast(GrayImage(2, 2, [10, 20, 30, 40]))
        binary = threshold_otsu(image)
        self.assertEqual(len(binary.pixels), 4)
        self.assertTrue(set(binary.pixels).issubset({0, 255}))


if __name__ == "__main__":
    unittest.main()
