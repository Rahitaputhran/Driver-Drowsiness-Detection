from __future__ import annotations

import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from src.config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FACE_MODEL_PATH,
    EAR_CLOSED_THRESHOLD,
    MAR_YAWN_THRESHOLD,
    WARNING_SCORE,
    DROWSY_SCORE,
)
from src.face_landmarks import FaceLandmarker
from src.features import eye_aspect_ratio, mouth_aspect_ratio
from src.temporal import TemporalFeatures
from src.head_pose import estimate_head_pose
from src.decision import HysteresisDecision
from src.alarm import beep


# =========================================================
# MediaPipe landmark indices
# These MUST match training/collect_features.py
# =========================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

MOUTH = [61, 291, 0, 17, 269, 405]

# Required order for estimate_head_pose():
# nose, chin, left eye outer, right eye outer,
# left mouth corner, right mouth corner
POSE_POINTS = [1, 152, 263, 33, 61, 291]


# =========================================================
# Paths
# =========================================================

MODEL_PATH = Path("models/drowsiness_xgb.joblib")


# =========================================================
# Helper
# =========================================================

def lm_xy(landmarks, indices, w, h):
    """Convert MediaPipe normalized landmarks to pixel coordinates."""

    return np.array(
        [
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in indices
        ],
        dtype=np.float64,
    )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # Check MediaPipe model
    # -----------------------------------------------------

    if not Path(FACE_MODEL_PATH).exists():
        raise SystemExit(
            f"Missing {FACE_MODEL_PATH}.\n"
            "Place the MediaPipe Face Landmarker model there."
        )

    # -----------------------------------------------------
    # Check XGBoost model
    # -----------------------------------------------------

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Missing trained model: {MODEL_PATH}\n"
            "Train the model first."
        )

    # -----------------------------------------------------
    # Load trained XGBoost model
    # -----------------------------------------------------

    print("Loading trained XGBoost model...")

    model_bundle = joblib.load(MODEL_PATH)

    # Your .joblib contains:
    # {
    #     "model": trained XGBoost model,
    #     "features": feature names
    # }

    if not isinstance(model_bundle, dict):
        raise SystemExit(
            "Unexpected model file format."
        )

    if "model" not in model_bundle:
        raise SystemExit(
            "Model bundle does not contain 'model'."
        )

    model = model_bundle["model"]

    feature_names = model_bundle.get(
        "features",
        [],
    )

    print("Model loaded successfully.")
    print("Features:", feature_names)

    # -----------------------------------------------------
    # Webcam
    # -----------------------------------------------------

    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    if not cap.isOpened():
        raise SystemExit(
            "Could not open webcam."
        )

    # -----------------------------------------------------
    # MediaPipe Face Landmarker
    # -----------------------------------------------------

    detector = FaceLandmarker(
        FACE_MODEL_PATH
    )

    # -----------------------------------------------------
    # Temporal feature tracker
    # -----------------------------------------------------

    temporal = TemporalFeatures(
        window_seconds=30.0
    )

    # -----------------------------------------------------
    # Decision layer
    # -----------------------------------------------------

    decision = HysteresisDecision(
        WARNING_SCORE,
        DROWSY_SCORE,
    )

    # -----------------------------------------------------
    # Timing
    # -----------------------------------------------------

    start = time.monotonic()

    last_alarm = -999.0

    # -----------------------------------------------------
    # Head pose history
    # -----------------------------------------------------

    pose_history = []

    # =====================================================
    # Main webcam loop
    # =====================================================

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                print("Could not read frame.")
                break

            h, w = frame.shape[:2]

            timestamp = (
                time.monotonic() - start
            )

            # -------------------------------------------------
            # MediaPipe face detection / landmarks
            # -------------------------------------------------

            result = detector.detect(
                frame,
                int(timestamp * 1000),
            )

            # -------------------------------------------------
            # Default values
            # -------------------------------------------------

            label = "NO FACE"

            score = 0.0

            ear = 0.0
            mar = 0.0

            perclos = 0.0

            eye_closed_ratio = 0.0
            max_eye_closure_s = 0.0

            blink_rate = 0.0
            yawn_rate = 0.0

            head_pitch = 0.0
            head_yaw = 0.0
            head_roll = 0.0
            head_motion = 0.0

            # =================================================
            # Face available
            # =================================================

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                # -------------------------------------------------
                # 1. Extract eyes and mouth
                # -------------------------------------------------

                left = lm_xy(
                    landmarks,
                    LEFT_EYE,
                    w,
                    h,
                )

                right = lm_xy(
                    landmarks,
                    RIGHT_EYE,
                    w,
                    h,
                )

                mouth = lm_xy(
                    landmarks,
                    MOUTH,
                    w,
                    h,
                )

                # -------------------------------------------------
                # 2. EAR
                # -------------------------------------------------

                left_ear = eye_aspect_ratio(
                    left
                )

                right_ear = eye_aspect_ratio(
                    right
                )

                ear = (
                    left_ear + right_ear
                ) / 2.0

                # -------------------------------------------------
                # 3. MAR
                # -------------------------------------------------

                mar = mouth_aspect_ratio(
                    mouth
                )

                # -------------------------------------------------
                # 4. Temporal features
                # -------------------------------------------------

                tf = temporal.update(
                    timestamp,
                    ear,
                    mar,
                    EAR_CLOSED_THRESHOLD,
                    MAR_YAWN_THRESHOLD,
                )

                perclos = tf["perclos"]

                closed_duration = tf[
                    "eye_closed_duration"
                ]

                # Same feature representation used
                # during training.
                eye_closed_ratio = perclos

                # Current continuous eye-closure duration.
                max_eye_closure_s = closed_duration

                # -------------------------------------------------
                # 5. Blink rate
                # -------------------------------------------------

                elapsed_minutes = max(
                    timestamp / 60.0,
                    1.0 / 60.0,
                )

                blink_rate = (
                    tf["blink_count"]
                    / elapsed_minutes
                )

                # -------------------------------------------------
                # 6. Yawn rate
                # -------------------------------------------------

                yawn_rate = (
                    tf["yawn_count"]
                    / elapsed_minutes
                )

                # -------------------------------------------------
                # 7. Head pose
                # -------------------------------------------------

                pose_points = lm_xy(
                    landmarks,
                    POSE_POINTS,
                    w,
                    h,
                )

                pose = estimate_head_pose(
                    pose_points,
                    w,
                    h,
                )

                if pose is not None:

                    head_pitch = pose[
                        "pitch"
                    ]

                    head_yaw = pose[
                        "yaw"
                    ]

                    head_roll = pose[
                        "roll"
                    ]

                # -------------------------------------------------
                # 8. Head motion
                # -------------------------------------------------

                pose_history.append(
                    [
                        head_pitch,
                        head_yaw,
                        head_roll,
                    ]
                )

                # Keep recent history bounded.
                if len(pose_history) > 300:
                    pose_history.pop(0)

                pose_array = np.asarray(
                    pose_history,
                    dtype=np.float64,
                )

                if len(pose_array) > 1:

                    head_motion = float(
                        np.std(
                            pose_array[:, 0]
                        )
                        + np.std(
                            pose_array[:, 1]
                        )
                        + np.std(
                            pose_array[:, 2]
                        )
                    )

                # =================================================
                # 9. Create EXACT 11 model features
                # =================================================

                feature_values = [
                    ear,
                    mar,
                    perclos,
                    eye_closed_ratio,
                    max_eye_closure_s,
                    blink_rate,
                    yawn_rate,
                    head_pitch,
                    head_yaw,
                    head_roll,
                    head_motion,
                ]

                # -------------------------------------------------
                # Use saved feature order if available
                # -------------------------------------------------

                if feature_names:

                    feature_map = {
                        "ear": ear,
                        "mar": mar,
                        "perclos": perclos,
                        "eye_closed_ratio": eye_closed_ratio,
                        "max_eye_closure_s": max_eye_closure_s,
                        "blink_rate": blink_rate,
                        "yawn_rate": yawn_rate,
                        "head_pitch": head_pitch,
                        "head_yaw": head_yaw,
                        "head_roll": head_roll,
                        "head_motion": head_motion,
                    }

                    feature_values = [
                        feature_map[name]
                        for name in feature_names
                    ]

                features = np.array(
                    [feature_values],
                    dtype=np.float32,
                )

                # =================================================
                # 10. XGBoost prediction
                # =================================================

                probability = (
                    model.predict_proba(
                        features
                    )[0, 1]
                )

                score = float(
                    probability
                )

                # =================================================
                # 11. Decision layer
                # =================================================

                d = decision.update(
                    score
                )

                label = d.label

                # =================================================
                # 12. Alarm
                # =================================================

                if label == "DROWSY":

                    if (
                        timestamp - last_alarm
                        > 2.0
                    ):

                        beep()

                        last_alarm = (
                            timestamp
                        )

                # =================================================
                # Draw landmarks
                # =================================================

                for point in np.vstack(
                    [
                        left,
                        right,
                        mouth,
                    ]
                ).astype(int):

                    cv2.circle(
                        frame,
                        tuple(point),
                        2,
                        (0, 255, 0),
                        -1,
                    )

            # =====================================================
            # Display information
            # =====================================================

            cv2.rectangle(
                frame,
                (10, 10),
                (500, 305),
                (20, 20, 20),
                -1,
            )

            cv2.putText(
                frame,
                f"STATUS: {label}",
                (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (25, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (25, 97),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                f"PERCLOS: {perclos:.1%}",
                (25, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                f"Drowsiness: {score:.1%}",
                (25, 147),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                f"Eye closure: {max_eye_closure_s:.2f}s",
                (25, 172),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                f"Blink rate: {blink_rate:.1f}/min",
                (25, 197),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                f"Yawn rate: {yawn_rate:.1f}/min",
                (25, 222),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                (
                    f"Head: "
                    f"P{head_pitch:.1f} "
                    f"Y{head_yaw:.1f} "
                    f"R{head_roll:.1f}"
                ),
                (25, 247),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                f"Head motion: {head_motion:.2f}",
                (25, 272),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (25, 297),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )

            # -----------------------------------------------------
            # Show webcam
            # -----------------------------------------------------

            cv2.imshow(
                "Driver Drowsiness Detection",
                frame,
            )

            # -----------------------------------------------------
            # Quit
            # -----------------------------------------------------

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

    finally:

        detector.close()

        cap.release()

        cv2.destroyAllWindows()


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()