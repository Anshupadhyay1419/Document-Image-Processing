"""
Noise estimation.

Algorithm (Immerkaer, 1996): sigma_n ~= sqrt(pi/2) * 1/(6*(W-2)*(H-2)) *
                                         sum(|I * N|)
where N is a 3x3 kernel that vanishes on smooth surfaces:
        [[ 1, -2,  1],
         [-2,  4, -2],
         [ 1, -2,  1]]
This gives a fast per-image estimate of Gaussian noise standard deviation.

Why:
Simple, no reference image required, robust for document photos.

Failure modes:
* Heavy text/edges inflate the estimate. Consider computing on a smooth
  patch or accepting the estimate as an upper bound.

Safe for faint handwriting: YES (measurement only).
"""
from __future__ import annotations

import cv2
import numpy as np


_KERNEL = np.array(
    [[1, -2, 1],
     [-2, 4, -2],
     [1, -2, 1]],
    dtype=np.float32,
)


def estimate_noise(image: np.ndarray) -> float:
    """Return an estimate of Gaussian noise standard deviation."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    H, W = gray.shape
    if H < 3 or W < 3:
        return 0.0
    conv = cv2.filter2D(gray, cv2.CV_32F, _KERNEL)
    sigma = float(np.sum(np.abs(conv)))
    sigma *= np.sqrt(0.5 * np.pi) / (6.0 * (W - 2) * (H - 2))
    return sigma


def noise_category(sigma: float) -> str:
    if sigma < 2:
        return "low"
    if sigma < 6:
        return "moderate"
    return "high"
