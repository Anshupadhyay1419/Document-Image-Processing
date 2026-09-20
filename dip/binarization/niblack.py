from __future__ import annotations

import cv2
import numpy as np

from ..utils.logging_utils import get_logger

_log = get_logger()


def niblack(image: np.ndarray, window_size: int = 25, k: float = -0.2) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if window_size % 2 == 0:
        window_size += 1

    try:
        from skimage.filters import threshold_niblack

        T = threshold_niblack(gray, window_size=window_size, k=k)
        return (gray > T).astype(np.uint8) * 255
    except Exception:
        _log.debug("scikit-image not available; falling back to boxFilter Niblack.")

    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, cv2.CV_32F, (window_size, window_size))
    sqmean = cv2.boxFilter(g * g, cv2.CV_32F, (window_size, window_size))
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0))
    T = mean + k * std
    return (g > T).astype(np.uint8) * 255
