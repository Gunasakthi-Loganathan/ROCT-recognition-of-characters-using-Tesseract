"""Utilities for validating and evaluating OCR training datasets."""

from .metrics import character_error_rate, word_error_rate
from .validation import validate_splits

__all__ = ["character_error_rate", "word_error_rate", "validate_splits"]
