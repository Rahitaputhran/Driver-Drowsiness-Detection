from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from src.config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FACE_MODEL_PATH,
    XGB_MODEL_PATH,
    EAR_CLOSED_THRESHOLD,
    MAR_YAWN_THRESHOLD,
    WARNING_SCORE,
    DROWSY_SCORE,
)
from src.face_landmarks import FaceLandmarker
from src.features import eye_aspect_ratio, mouth_aspect_ratio
from src.head_pose import estimate_head_pose
from src.decision import HysteresisDecision
from src.alarm import beep
from training.collect_features import (
    SAMPLE_INTERVAL,
    WINDOW_SECONDS,
    MIN_SAMPLES_FOR_FEATURES,
    YAWN_MIN_DURATION,
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 0, 17, 269, 405]
POSE_POINTS = [1, 152, 263, 33, 61, 291]
FEATURES = [
    "ear", "mar", "perclos", "eye_closed_ratio", "max_eye_closure_s",
    "blink_rate", "yawn_rate", "head_pitch", "head_yaw", "head_roll",
    "head_motion",
]


def lm_xy(landmarks, indices, w, h):
    return np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in indices],
        dtype=np.float64,
    )


def main():
    face_path = Path(FACE_MODEL_PATH)
    model_path = Path(XGB_MODEL_PATH)

    if not face_path.exists():
        raise SystemExit(
            f"Missing MediaPipe model: {face_path}\n"
            "Place face_landmarker.task in the models folder."
        )
    if not model_path.exists():
        raise SystemExit(f"Missing trained XGBoost model: {model_path}")

    print(f"Loading XGBoost model: {model_path}")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise SystemExit(
            "Unexpected model format. Expected a bundle with 'model' and 'features'."
        )

    model = bundle["model"]
    feature_names = bundle.get("features", FEATURES)
    if feature_names != FEATURES:
        raise SystemExit(
            "Model feature order does not match the live pipeline.\n"
            f"Expected: {FEATURES}\nFound: {feature_names}\n"
            "Retrain with training.train_xgboost before running the app."
        )
    if getattr(model, "n_features_in_", None) != len(FEATURES):
        raise SystemExit(
            f"Model expects {getattr(model, 'n_features_in_', 'unknown')} features; "
            f"this app supplies {len(FEATURES)}. Retrain the model first."
        )

    print("Model loaded. Features:", ", ".join(feature_names))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check CAMERA_INDEX and camera permissions.")

    detector = FaceLandmarker(str(face_path))
    decision = HysteresisDecision(WARNING_SCORE, DROWSY_SCORE)
    history = deque()
    pose_history = []

    start_time = time.monotonic()
    last_prediction_time = 0.0
    last_alarm = -999.0
    previous_eye_closed = False
    eye_closure_start = None
    blink_count = 0
    previous_yawn = False
    yawn_start = None
    yawn_count = 0
    max_eye_closure = 0.0
    last_label = "WARMING UP"
    last_score = 0.0
    latest_features = None

    print("ML-based drowsiness detection running. Press Q to quit.")
    print("Predictions are experimental; do not use this demo as a driving-safety device.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from webcam.")
                break

            h, w = frame.shape[:2]
            timestamp = time.monotonic() - start_time
            result = detector.detect(frame, int(timestamp * 1000))
            status_color = (0, 0, 255)
            ear = mar = 0.0

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                left = lm_xy(landmarks, LEFT_EYE, w, h)
                right = lm_xy(landmarks, RIGHT_EYE, w, h)
                mouth = lm_xy(landmarks, MOUTH, w, h)
                ear = (
                    eye_aspect_ratio(left) + eye_aspect_ratio(right)
                ) / 2.0
                mar = mouth_aspect_ratio(mouth)

                # Match the collection script's blink counting rules.
                eye_closed = ear < EAR_CLOSED_THRESHOLD
                if eye_closed and not previous_eye_closed:
                    eye_closure_start = timestamp
                if not eye_closed and previous_eye_closed and eye_closure_start is not None:
                    closure_duration = timestamp - eye_closure_start
                    max_eye_closure = max(max_eye_closure, closure_duration)
                    if 0.05 <= closure_duration <= 0.8:
                        blink_count += 1
                    eye_closure_start = None
                previous_eye_closed = eye_closed
                if eye_closed and eye_closure_start is not None:
                    max_eye_closure = max(
                        max_eye_closure, timestamp - eye_closure_start
                    )

                # Match the collection script's sustained-yawn counting rules.
                yawning = mar > MAR_YAWN_THRESHOLD
                if yawning and not previous_yawn:
                    yawn_start = timestamp
                if not yawning and previous_yawn and yawn_start is not None:
                    if timestamp - yawn_start >= YAWN_MIN_DURATION:
                        yawn_count += 1
                    yawn_start = None
                previous_yawn = yawning

                pose_points = lm_xy(landmarks, POSE_POINTS, w, h)
                pose = estimate_head_pose(pose_points, w, h)
                if pose is None:
                    pitch = yaw = roll = 0.0
                else:
                    pitch = pose["pitch"]
                    yaw = pose["yaw"]
                    roll = pose["roll"]

                pose_history.append((pitch, yaw, roll))
                if len(pose_history) > 3000:
                    pose_history.pop(0)

                # Store per-frame observations, then aggregate over the same
                # rolling window used by the collector.
                history.append((timestamp, ear, mar, eye_closed, pitch, yaw, roll))
                while history and timestamp - history[0][0] > WINDOW_SECONDS:
                    history.popleft()

                if timestamp - last_prediction_time >= SAMPLE_INTERVAL and len(history) >= MIN_SAMPLES_FOR_FEATURES:
                    arr = np.asarray([item[1:] for item in history], dtype=np.float64)
                    ears = arr[:, 0]
                    mars = arr[:, 1]
                    closed = arr[:, 2].astype(bool)
                    pitches = arr[:, 3]
                    yaws = arr[:, 4]
                    rolls = arr[:, 5]

                    elapsed_minutes = max(timestamp / 60.0, 1.0 / 60.0)
                    perclos = float(np.mean(closed))
                    eye_closed_ratio = perclos
                    blink_rate = blink_count / elapsed_minutes
                    yawn_rate = yawn_count / elapsed_minutes
                    head_motion = float(np.std(pitches) + np.std(yaws) + np.std(rolls))

                    feature_map = {
                        "ear": float(np.mean(ears)),
                        "mar": float(np.mean(mars)),
                        "perclos": perclos,
                        "eye_closed_ratio": eye_closed_ratio,
                        "max_eye_closure_s": float(max_eye_closure),
                        "blink_rate": float(blink_rate),
                        "yawn_rate": float(yawn_rate),
                        "head_pitch": float(np.mean(pitches)),
                        "head_yaw": float(np.mean(yaws)),
                        "head_roll": float(np.mean(rolls)),
                        "head_motion": head_motion,
                    }
                    x = np.asarray([[feature_map[name] for name in feature_names]], dtype=np.float32)
                    probability = float(model.predict_proba(x)[0, 1])
                    result_decision = decision.update(probability)
                    last_label = result_decision.label
                    last_score = probability
                    latest_features = feature_map
                    last_prediction_time = timestamp

                    if last_label == "DROWSY" and timestamp - last_alarm > 2.0:
                        beep()
                        last_alarm = timestamp

                for point in np.vstack([left, right, mouth]).astype(int):
                    cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

                status_color = (0, 0, 255) if last_label in ("WARNING", "DROWSY") else (0, 200, 0)
                display_label = last_label
            else:
                # Do not fabricate a prediction when no face is detected.
                display_label = "NO FACE"
                previous_eye_closed = False
                eye_closure_start = None
                previous_yawn = False
                yawn_start = None

            cv2.rectangle(frame, (10, 10), (520, 220), (20, 20, 20), -1)
            cv2.putText(frame, f"STATUS: {display_label}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)
            cv2.putText(frame, f"ML drowsiness score: {last_score:.1%}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            cv2.putText(frame, f"Live EAR: {ear:.3f}   MAR: {mar:.3f}", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)
            if latest_features:
                cv2.putText(frame, f"PERCLOS: {latest_features['perclos']:.1%}", (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)
                cv2.putText(frame, f"Eye closure max: {latest_features['max_eye_closure_s']:.2f}s", (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "Collecting temporal features...", (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)
            cv2.putText(frame, "Press Q to quit", (25, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
            cv2.imshow("Driver Drowsiness Detection - XGBoost", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
