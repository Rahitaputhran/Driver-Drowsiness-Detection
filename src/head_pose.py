from __future__ import annotations

import cv2
import numpy as np


def estimate_head_pose(
    image_points: np.ndarray,
    frame_width: int,
    frame_height: int,
):
    """
    Estimate pitch/yaw/roll from selected 2D facial landmarks using solvePnP.

    image_points order:
    nose tip, chin, left eye outer, right eye outer, left mouth, right mouth
    """
    if len(image_points) != 6:
        raise ValueError("Expected six 2D facial landmarks.")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.0, -12.0),
        (-43.0, 32.0, -26.0),
        (43.0, 32.0, -26.0),
        (-30.0, -28.0, -18.0),
        (30.0, -28.0, -18.0),
    ], dtype=np.float64)

    focal = float(frame_width)
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        np.asarray(image_points, dtype=np.float64),
        camera_matrix,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math_sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        roll = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        roll = 0.0

    return {
        "pitch": float(pitch),
        "yaw": float(yaw),
        "roll": float(roll),
        "translation": translation_vector.reshape(-1),
    }


def math_sqrt(x):
    return float(np.sqrt(x))
