from __future__ import annotations

import math
import numpy as np


def euclidean(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR for 6 ordered eye landmarks: p1,p2,p3,p4,p5,p6."""
    if len(points) != 6:
        raise ValueError("EAR expects exactly 6 points.")
    p1, p2, p3, p4, p5, p6 = points
    horizontal = euclidean(p1, p4)
    if horizontal <= 1e-8:
        return 0.0
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(points: np.ndarray) -> float:
    """MAR for six mouth points: left, upper-left, upper-right, right, lower-right, lower-left."""
    if len(points) != 6:
        raise ValueError("MAR expects exactly 6 points.")
    left, ul, ur, right, lr, ll = points
    width = euclidean(left, right)
    if width <= 1e-8:
        return 0.0
    return (euclidean(ul, ll) + euclidean(ur, lr)) / (2.0 * width)


def normalize_score(value, low, high) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))
