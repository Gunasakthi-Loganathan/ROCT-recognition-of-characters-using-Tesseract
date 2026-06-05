import json
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path

from ml_pipeline.inference_server import InferenceError, _parse_multipart, predict_uploaded_image
from ml_pipeline.models import TemplateClassifier
from ml_pipeline.preprocessing import GrayImage, write_pgm


class InferenceServerTest(unittest.TestCase):
    def test_parse_multipart_file(self):
        body = (
            b"--abc\r\n"
            b"Content-Disposition: form-data; name=\"file\"; filename=\"x.pgm\"\r\n"
            b"Content-Type: image/x-portable-graymap\r\n\r\n"
            b"P2\n1 1\n255\n0\n\r\n"
            b"--abc--\r\n"
        )
        data, filename = _parse_multipart(body, "multipart/form-data; boundary=abc")
        self.assertEqual(filename, "x.pgm")
        self.assertIn(b"P2", data)

    def test_predict_uploaded_image_with_template_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.pgm"
            write_pgm(GrayImage(2, 2, [0, 255, 255, 0]), image_path)
            model = TemplateClassifier(image_size=(2, 2))
            model.partial_fit(GrayImage(2, 2, [0, 255, 255, 0]), "A")
            model_path = root / "model.json"
            model.save(model_path)

            result = predict_uploaded_image(image_path.read_bytes(), "sample.pgm", model_path)

            self.assertEqual(result["text"], "A")
            self.assertEqual(result["engine"], "ml")
            self.assertGreaterEqual(result["confidence"], 0)

    def test_predict_rejects_unsupported_upload_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TemplateClassifier(image_size=(1, 1))
            model.partial_fit(GrayImage(1, 1, [0]), "A")
            model_path = Path(tmp) / "model.json"
            model.save(model_path)
            with self.assertRaises(InferenceError) as ctx:
                predict_uploaded_image(b"not-png", "sample.png", model_path)
            self.assertEqual(ctx.exception.status, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)


if __name__ == "__main__":
    unittest.main()
