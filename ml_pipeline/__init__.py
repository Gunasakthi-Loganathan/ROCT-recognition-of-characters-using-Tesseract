"""Utilities for validating, preprocessing, training, and evaluating OCR datasets.

Imports are intentionally lazy to avoid side effects when modules are executed
with `python -m ml_pipeline.<module>`.
"""

__all__ = [
    "GrayImage",
    "audit_dataset",
    "character_error_rate",
    "create_splits",
    "preprocess_file",
    "preprocess_image",
    "validate_splits",
    "word_error_rate",
]


def __getattr__(name):
    if name in {"audit_dataset", "create_splits"}:
        from .dataset import audit_dataset, create_splits

        return {"audit_dataset": audit_dataset, "create_splits": create_splits}[name]
    if name in {"character_error_rate", "word_error_rate"}:
        from .metrics import character_error_rate, word_error_rate

        return {"character_error_rate": character_error_rate, "word_error_rate": word_error_rate}[name]
    if name in {"GrayImage", "preprocess_file", "preprocess_image"}:
        from .preprocessing import GrayImage, preprocess_file, preprocess_image

        return {"GrayImage": GrayImage, "preprocess_file": preprocess_file, "preprocess_image": preprocess_image}[name]
    if name == "validate_splits":
        from .validation import validate_splits

        return validate_splits
    raise AttributeError(name)
