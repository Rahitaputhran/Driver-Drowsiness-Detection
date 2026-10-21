import numpy as np

from src.features import eye_aspect_ratio, mouth_aspect_ratio


def test_ear_positive():
    pts = np.array([
        [0, 0], [1, 1], [2, 1],
        [4, 0], [2, -1], [1, -1],
    ], dtype=float)
    assert eye_aspect_ratio(pts) > 0


def test_mar_positive():
    pts = np.array([
        [0, 0], [1, 1], [3, 1],
        [4, 0], [3, -1], [1, -1],
    ], dtype=float)
    assert mouth_aspect_ratio(pts) > 0
