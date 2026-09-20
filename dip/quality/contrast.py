"""
Contrast estimation.

Two measurements are exposed:
 * `std_contrast`  = standard deviation of grayscale intensities.
 * `rms_contrast`  = sqrt(mean((I - mean(I))^2)) — algebraically identical
                     to std for a full image; kept as a named function to
                     mirror the classical definition.
 * `michelson_contrast` for reference (uses min/max, sensitive to outliers).

Categorisation:
* std < 30  : very_low
* std < 50  : low
* std < 80  : moderate
* else      : good

Safe for faint handwriting: YES (measurement only).
"""
from __future__ import annotations

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def std_contrast(image: np.ndarray) -> float:
    return float(np.std(_gray(image)))


def rms_contrast(image: np.ndarray) -> float:
    g = _gray(image).astype(np.float64) / 255.0
    return float(np.sqrt(np.mean((g - g.mean()) ** 2)))


def michelson_contrast(image: np.ndarray) -> float:
    g = _gray(image).astype(np.float64)
    lo, hi = g.min(), g.max()
    if hi + lo == 0:
        return 0.0
    return float((hi - lo) / (hi + lo))


def contrast_category(std_value: float) -> str:
    if std_value < 30:
        return "very_low"
    if std_value < 50:
        return "low"
    if std_value < 80:
        return "moderate"
    return "good"
