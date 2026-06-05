"""Minimal Python OCR inference backend.

This server intentionally has no third-party runtime dependencies so it can run
in constrained demo and CI environments. It exposes a production-shaped upload
endpoint that loads an already-trained model checkpoint from `ML_MODEL_PATH`.

The included template model supports isolated-character PGM/PPM inputs. Real
handwritten word/line deployments should train a CNN/CRNN with the tooling and
point `ML_MODEL_PATH` at the selected checkpoint. The endpoint returns clear
503/415/422 responses instead of silently falling back or claiming accuracy.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .models import TemplateClassifier
from .preprocessing import preprocess_file

MAX_UPLOAD_BYTES = int(os.environ.get("ML_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
DEFAULT_HOST = os.environ.get("ML_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("ML_PORT", "9000"))
DEFAULT_MODEL_PATH = os.environ.get("ML_MODEL_PATH", "")
ALLOWED_EXTENSIONS = {".pgm", ".ppm"}


class InferenceError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _parse_multipart(body: bytes, content_type: str) -> tuple[bytes, str]:
    marker = "boundary="
    if marker not in content_type:
        raise InferenceError(HTTPStatus.BAD_REQUEST, "multipart boundary is missing")
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    delimiter = ("--" + boundary).encode("utf-8")
    for part in body.split(delimiter):
        part = part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        headers_raw, data = part.split(b"\r\n\r\n", 1)
        headers = headers_raw.decode("utf-8", errors="replace").lower()
        if 'name="file"' not in headers:
            continue
        filename = "upload.pgm"
        for header_line in headers_raw.decode("utf-8", errors="replace").splitlines():
            if not header_line.lower().startswith("content-disposition:"):
                continue
            for item in header_line.split(";"):
                item = item.strip()
                if item.startswith("filename="):
                    filename = item.split("=", 1)[1].strip().strip('"') or filename
        return data.rstrip(b"\r\n"), Path(filename).name
    raise InferenceError(HTTPStatus.BAD_REQUEST, "multipart body does not contain a file field")


def predict_uploaded_image(data: bytes, filename: str, model_path: str | Path | None = None) -> dict[str, object]:
    checkpoint = Path(model_path or DEFAULT_MODEL_PATH)
    if not str(checkpoint):
        raise InferenceError(HTTPStatus.SERVICE_UNAVAILABLE, "ML_MODEL_PATH is not configured")
    if not checkpoint.exists():
        raise InferenceError(HTTPStatus.SERVICE_UNAVAILABLE, f"ML model checkpoint not found: {checkpoint}")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise InferenceError(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            "The configured lightweight backend currently accepts PGM/PPM images. Use Tesseract for JPG/PNG or configure a real CNN/CRNN backend.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    try:
        model = TemplateClassifier.load(checkpoint)
        image = preprocess_file(temp_path, size=model.image_size)
        started = time.perf_counter()
        prediction = model.predict(image)
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "text": prediction.text,
            "confidence": prediction.confidence,
            "engine": "ml",
            "model": checkpoint.name,
            "latency_ms": latency_ms,
        }
    except ValueError as exc:
        raise InferenceError(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


class OcrInferenceHandler(BaseHTTPRequestHandler):
    server_version = "ROCTML/1.0"

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        data = _json_bytes(payload)
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", os.environ.get("ML_CLIENT_ORIGIN", "http://localhost:5173"))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        self._send_json(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        if urlparse(self.path).path != "/":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        checkpoint = Path(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH else None
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "modelConfigured": bool(checkpoint and checkpoint.exists()),
                "modelPath": str(checkpoint) if checkpoint else "",
                "acceptedExtensions": sorted(ALLOWED_EXTENSIONS),
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        if urlparse(self.path).path != "/api/ocr":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise InferenceError(HTTPStatus.BAD_REQUEST, "request body is empty")
            if content_length > MAX_UPLOAD_BYTES:
                raise InferenceError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "uploaded image exceeds server size limit")
            content_type = self.headers.get("Content-Type", "")
            body = self.rfile.read(content_length)
            data, filename = _parse_multipart(body, content_type)
            result = predict_uploaded_image(data, filename)
            self._send_json(HTTPStatus.OK, result)
        except InferenceError as exc:
            self._send_json(exc.status, {"error": exc.message, "message": exc.message})
        except Exception as exc:  # noqa: BLE001 - isolate server errors from clients.
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "ML inference failed", "message": str(exc)})

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - inherited API name.
        if os.environ.get("ML_QUIET", "0") != "1":
            super().log_message(format, *args)


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), OcrInferenceHandler)
    print(f"ML OCR backend running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
