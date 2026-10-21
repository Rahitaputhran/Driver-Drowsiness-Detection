from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


class FaceLandmarker:
    """
    MediaPipe Tasks API wrapper.

    The task model must be placed at models/face_landmarker.task.
    """

    def __init__(self, model_path: str | Path):
        if mp is None:
            raise ImportError("Install mediapipe first.")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing MediaPipe model: {model_path}. "
                "Download face_landmarker.task and place it in models/."
            )

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.mp = mp

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect_for_video(image, timestamp_ms)

    def close(self):
        close = getattr(self.detector, "close", None)
        if close:
            close()
