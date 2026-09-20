"""
Brightness estimation.

Approach: mean grayscale intensity in [0, 255].
* < 60  : very dark
* 60-100: dark
* 100-180: normal
* 180-220: bright
* > 220 : washed out

Failure modes:
* Large white margins bias the mean upwards; consider cropping to content
  before measurement for cleaner numbers.

Safe for faint handwriting: YES.
"""
from __future__ import annotations

import cv2
import numpy as np


def mean_brightness(image: np.ndarray) -> float:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def brightness_category(value: float) -> str:
    if value < 60:
        return "very_dark"
    if value < 100:
        return "dark"
    if value < 180:
        return "normal"
    if value < 220:
        return "bright"
    return "washed_out"
