from __future__ import annotations

import cv2
import numpy as np


def otsu(image: np.ndarray, invert: bool = False) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, out = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    return out
