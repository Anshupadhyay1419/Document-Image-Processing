"""
Blur estimation.

Algorithm: Variance of the Laplacian
------------------------------------
* Convolve grayscale image with the Laplacian kernel.
* Compute the variance of the resulting response map.
* Low variance -> smooth / blurry image; High variance -> sharp edges.

Why:
Fast, single-scalar, dependency-free. Widely used as a first-pass sharpness
proxy for document images.

Failure modes:
* Low-texture pages (blank sections) score low even when sharp.
* Absolute score is scene-dependent; use as a **relative** signal or with
  a dataset-tuned threshold.

Safe for faint handwriting: YES (measurement only, no modification).
"""
from __future__ import annotations

import cv2
import numpy as np


def variance_of_laplacian(image: np.ndarray) -> float:
    """Return the variance of the Laplacian of the grayscale image."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def blur_score(image: np.ndarray) -> float:
    """Alias returning the same score as `variance_of_laplacian`."""
    return variance_of_laplacian(image)


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Convenience predicate.
    Threshold of 100 works reasonably for A4 phone-scans; tune per dataset.
    """
    return blur_score(image) < threshold
