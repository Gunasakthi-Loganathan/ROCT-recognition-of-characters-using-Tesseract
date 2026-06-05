"""Deterministic OCR image preprocessing without heavyweight dependencies.

The production browser pipeline still uses Canvas/Tesseract.js. This module gives
Python dataset validation, dry runs, and smoke tests a shared preprocessing path
that works in constrained CI environments. It supports Netpbm PGM/PPM sample
images directly and reports clear errors for unsupported/corrupted inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import median


@dataclass(slots=True)
class GrayImage:
    """Small grayscale image container with 8-bit pixels."""

    width: int
    height: int
    pixels: list[int]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if len(self.pixels) != self.width * self.height:
            raise ValueError("pixel count does not match image dimensions")
        self.pixels = [max(0, min(255, int(value))) for value in self.pixels]

    def rows(self) -> list[list[int]]:
        return [self.pixels[i : i + self.width] for i in range(0, len(self.pixels), self.width)]


def _read_netpbm_tokens(data: bytes) -> list[bytes]:
    tokens: list[bytes] = []
    for line in data.splitlines():
        line = line.split(b"#", 1)[0]
        tokens.extend(line.split())
    return tokens


def read_netpbm(path: str | Path) -> GrayImage:
    """Read ASCII PGM/PPM (P2/P3) or binary PGM/PPM (P5/P6) into grayscale."""

    raw = Path(path).read_bytes()
    if not raw.startswith((b"P2", b"P3", b"P5", b"P6")):
        raise ValueError("only PGM/PPM Netpbm images are supported by this lightweight reader")
    magic = raw[:2]
    if magic in {b"P2", b"P3"}:
        tokens = _read_netpbm_tokens(raw)
        if len(tokens) < 4:
            raise ValueError("invalid Netpbm header")
        width, height, max_value = map(int, tokens[1:4])
        values = [int(value) for value in tokens[4:]]
    else:
        # Parse binary header manually while respecting comments.
        tokens: list[bytes] = []
        index = 0
        while len(tokens) < 4:
            while index < len(raw) and raw[index] in b" \t\r\n":
                index += 1
            if index < len(raw) and raw[index:index + 1] == b"#":
                while index < len(raw) and raw[index:index + 1] not in b"\r\n":
                    index += 1
                continue
            start = index
            while index < len(raw) and raw[index] not in b" \t\r\n":
                index += 1
            tokens.append(raw[start:index])
        while index < len(raw) and raw[index] in b" \t\r\n":
            index += 1
        width, height, max_value = map(int, tokens[1:4])
        values = list(raw[index:])
    if max_value <= 0 or max_value > 255:
        raise ValueError("only 8-bit Netpbm images are supported")
    expected = width * height * (3 if magic in {b"P3", b"P6"} else 1)
    if len(values) < expected:
        raise ValueError("image data is shorter than declared dimensions")
    values = values[:expected]
    if magic in {b"P3", b"P6"}:
        gray: list[int] = []
        for i in range(0, len(values), 3):
            gray.append(round(0.299 * values[i] + 0.587 * values[i + 1] + 0.114 * values[i + 2]))
        values = gray
    if max_value != 255:
        values = [round(value * 255 / max_value) for value in values]
    return GrayImage(width, height, values)


def write_pgm(image: GrayImage, path: str | Path) -> None:
    """Write an image as ASCII PGM for deterministic debug artifacts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [" ".join(str(value) for value in row) for row in image.rows()]
    target.write_text(f"P2\n{image.width} {image.height}\n255\n" + "\n".join(rows) + "\n", encoding="ascii")


def invert_if_needed(image: GrayImage) -> GrayImage:
    """Ensure dark foreground on light background using border brightness."""

    border: list[int] = []
    rows = image.rows()
    border.extend(rows[0])
    border.extend(rows[-1])
    for row in rows[1:-1]:
        border.append(row[0])
        border.append(row[-1])
    if border and sum(border) / len(border) < 128:
        return GrayImage(image.width, image.height, [255 - value for value in image.pixels])
    return image


def autocontrast(image: GrayImage) -> GrayImage:
    """Stretch image contrast to the full 0-255 range."""

    lo, hi = min(image.pixels), max(image.pixels)
    if hi <= lo:
        return image
    return GrayImage(image.width, image.height, [round((value - lo) * 255 / (hi - lo)) for value in image.pixels])


def median_filter(image: GrayImage, radius: int = 1) -> GrayImage:
    """Apply a small median filter to reduce salt-and-pepper noise."""

    if radius <= 0:
        return image
    rows = image.rows()
    out: list[int] = []
    for y in range(image.height):
        for x in range(image.width):
            window: list[int] = []
            for yy in range(max(0, y - radius), min(image.height, y + radius + 1)):
                for xx in range(max(0, x - radius), min(image.width, x + radius + 1)):
                    window.append(rows[yy][xx])
            out.append(int(median(window)))
    return GrayImage(image.width, image.height, out)


def threshold_otsu(image: GrayImage) -> GrayImage:
    """Binarize using Otsu's method."""

    hist = [0] * 256
    for value in image.pixels:
        hist[value] += 1
    total = len(image.pixels)
    sum_total = sum(i * count for i, count in enumerate(hist))
    sum_background = 0.0
    weight_background = 0
    best_variance = -1.0
    threshold = 127
    for i, count in enumerate(hist):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += i * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = i
    return GrayImage(image.width, image.height, [0 if value <= threshold else 255 for value in image.pixels])


def resize_nearest(image: GrayImage, width: int, height: int) -> GrayImage:
    """Resize using nearest-neighbor interpolation for deterministic dry runs."""

    rows = image.rows()
    out: list[int] = []
    for y in range(height):
        src_y = min(image.height - 1, round(y * image.height / height))
        for x in range(width):
            src_x = min(image.width - 1, round(x * image.width / width))
            out.append(rows[src_y][src_x])
    return GrayImage(width, height, out)


def preprocess_image(
    image: GrayImage,
    *,
    size: tuple[int, int] = (32, 32),
    denoise: bool = True,
    binarize: bool = True,
) -> GrayImage:
    """Apply the shared OCR preprocessing sequence used by dry-run training/inference."""

    result = invert_if_needed(image)
    result = autocontrast(result)
    if denoise:
        result = median_filter(result)
    if binarize:
        result = threshold_otsu(result)
    return resize_nearest(result, size[0], size[1])


def preprocess_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    size: tuple[int, int] = (32, 32),
    debug_dir: str | Path | None = None,
) -> GrayImage:
    """Read, preprocess, optionally write debug stages, and return an image."""

    original = read_netpbm(input_path)
    stages = {
        "01-original": original,
        "02-inverted": invert_if_needed(original),
    }
    stages["03-contrast"] = autocontrast(stages["02-inverted"])
    stages["04-denoise"] = median_filter(stages["03-contrast"])
    stages["05-binary"] = threshold_otsu(stages["04-denoise"])
    processed = resize_nearest(stages["05-binary"], size[0], size[1])
    stages["06-resized"] = processed
    if debug_dir:
        base = Path(input_path).stem
        for name, stage in stages.items():
            write_pgm(stage, Path(debug_dir) / f"{base}-{name}.pgm")
    if output_path:
        write_pgm(processed, output_path)
    return processed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess a PGM/PPM image for OCR dry runs")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--debug-dir", type=Path)
    args = parser.parse_args(argv)
    preprocess_file(args.input, args.output, size=(args.width, args.height), debug_dir=args.debug_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
