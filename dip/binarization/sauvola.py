from __future__ import annotations

import cv2
import numpy as np

from ..utils.logging_utils import get_logger

_log = get_logger()


def sauvola(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.34,
    R: float = 128.0,
) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if window_size % 2 == 0:
        window_size += 1

    # Prefer skimage when available (matches classical Sauvola exactly).
    try:
        from skimage.filters import threshold_sauvola

        T = threshold_sauvola(gray, window_size=window_size, k=k, r=R)
        binary = (gray > T).astype(np.uint8) * 255
        return binary
    except Exception:
        _log.debug("scikit-image not available; falling back to boxFilter Sauvola.")

    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, cv2.CV_32F, (window_size, window_size))
    sqmean = cv2.boxFilter(g * g, cv2.CV_32F, (window_size, window_size))
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0))
    T = mean * (1.0 + k * (std / R - 1.0))
    binary = (g > T).astype(np.uint8) * 255
    return binary
