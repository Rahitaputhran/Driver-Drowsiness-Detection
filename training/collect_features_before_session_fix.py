import csv
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from src.config import EAR_CLOSED_THRESHOLD, MAR_YAWN_THRESHOLD
from src.features import eye_aspect_ratio, mouth_aspect_ratio
from src.head_pose import estimate_head_pose


# =========================================================
# CONFIGURATION
# =========================================================

OUTPUT_FILE = "data/drowsiness_features.csv"

SAMPLE_INTERVAL = 1.0
WINDOW_SECONDS = 30
MIN_SAMPLES_FOR_FEATURES = 5

# A yawn must remain above the MAR threshold
# for at least this long.
YAWN_MIN_DURATION = 1.0

# Ignore extremely unlikely head-pose estimates.
MAX_PITCH = 60.0
MAX_YAW = 75.0
MAX_ROLL = 60.0

os.makedirs("data", exist_ok=True)


# =========================================================
# MEDIAPIPE
# =========================================================

MODEL_PATH = "models/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
)

landmarker = FaceLandmarker.create_from_options(options)


# =========================================================
# LANDMARK INDICES
# =========================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

MOUTH = [61, 291, 0, 17, 269, 405]

# Head pose:
# nose, chin, left eye outer, right eye outer,
# left mouth corner, right mouth corner
POSE_INDICES = [1, 152, 263, 33, 61, 291]


# =========================================================
# CSV
# =========================================================

FIELDNAMES = [
    "ear",
    "mar",
    "perclos",
    "eye_closed_ratio",
    "max_eye_closure_s",
    "blink_rate",
    "yawn_rate",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "head_motion",
    "drowsy",
]


def create_csv():

    if not os.path.exists(OUTPUT_FILE):

        with open(
            OUTPUT_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES
            )

            writer.writeheader()


# =========================================================
# MAIN
# =========================================================

def main():

    create_csv()

    print()
    print("======================================")
    print(" DRIVER DROWSINESS DATA COLLECTOR")
    print("======================================")
    print()

    print("Label:")
    print("0 = ALERT")
    print("1 = DROWSY-LIKE")
    print()

    label = input(
        "Enter label (0 or 1): "
    ).strip()

    if label not in ["0", "1"]:

        print("Invalid label.")
        return

    print()
    print("Starting camera...")
    print("Press Q to stop.")
    print()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")
        return

    # =====================================================
    # TEMPORAL STATE
    # =====================================================

    history = deque()

    last_sample_time = 0.0

    # Blink state
    previous_eye_closed = False
    eye_closure_start = None
    blink_count = 0

    # Yawn state
    yawn_start = None
    yawn_count = 0
    previous_yawn = False

    # Maximum eye closure
    max_eye_closure = 0.0

    # Head pose
    previous_head = None

    start_time = time.time()

    frame_timestamp_ms = 0

    # =====================================================
    # CAMERA LOOP
    # =====================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Could not read webcam frame.")
            break

        frame = cv2.flip(
            frame,
            1
        )

        height, width = frame.shape[:2]

        # -------------------------------------------------
        # MediaPipe
        # -------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        frame_timestamp_ms += 33

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        current_time = time.time()

        # =================================================
        # FACE DETECTED
        # =================================================

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            # -------------------------------------------------
            # Landmark helper
            # -------------------------------------------------

            def point(index):

                landmark = landmarks[index]

                return np.array([
                    landmark.x * width,
                    landmark.y * height
                ])

            # -------------------------------------------------
            # Eyes
            # -------------------------------------------------

            left_eye = np.array([
                point(i)
                for i in LEFT_EYE
            ])

            right_eye = np.array([
                point(i)
                for i in RIGHT_EYE
            ])

            # -------------------------------------------------
            # Mouth
            # -------------------------------------------------

            mouth = np.array([
                point(i)
                for i in MOUTH
            ])

            # =================================================
            # EAR
            # =================================================

            left_ear = eye_aspect_ratio(
                left_eye
            )

            right_ear = eye_aspect_ratio(
                right_eye
            )

            ear = (
                left_ear +
                right_ear
            ) / 2.0

            # =================================================
            # MAR
            # =================================================

            mar = mouth_aspect_ratio(
                mouth
            )

            # =================================================
            # EYE CLOSURE / BLINK
            # =================================================

            eye_closed = (
                ear < EAR_CLOSED_THRESHOLD
            )

            # Eye just closed
            if (
                eye_closed
                and not previous_eye_closed
            ):

                eye_closure_start = current_time

            # Eye just opened
            if (
                not eye_closed
                and previous_eye_closed
            ):

                if eye_closure_start is not None:

                    duration = (
                        current_time -
                        eye_closure_start
                    )

                    max_eye_closure = max(
                        max_eye_closure,
                        duration
                    )

                    # Genuine short blink
                    if (
                        0.05 <= duration <= 0.8
                    ):

                        blink_count += 1

                eye_closure_start = None

            previous_eye_closed = eye_closed

            # Current eye closure
            current_closure = 0.0

            if (
                eye_closed
                and eye_closure_start is not None
            ):

                current_closure = (
                    current_time -
                    eye_closure_start
                )

            max_eye_closure = max(
                max_eye_closure,
                current_closure
            )

            # =================================================
            # SUSTAINED YAWN DETECTION
            # =================================================

            yawn = (
                mar > MAR_YAWN_THRESHOLD
            )

            # Start potential yawn
            if (
                yawn
                and not previous_yawn
            ):

                yawn_start = current_time

            # End potential yawn
            if (
                not yawn
                and previous_yawn
            ):

                if yawn_start is not None:

                    duration = (
                        current_time -
                        yawn_start
                    )

                    # Only count sustained mouth opening
                    if duration >= YAWN_MIN_DURATION:

                        yawn_count += 1

                yawn_start = None

            previous_yawn = yawn

            # =================================================
            # HEAD POSE
            # =================================================

            pose_points = np.array([
                [
                    landmarks[i].x * width,
                    landmarks[i].y * height
                ]
                for i in POSE_INDICES
            ], dtype=np.float64)

            try:

                pose_result = estimate_head_pose(
                    pose_points,
                    width,
                    height
                )

                if pose_result is not None:

                    raw_pitch = pose_result["pitch"]
                    raw_yaw = pose_result["yaw"]
                    raw_roll = pose_result["roll"]

                    # Reject obviously unstable estimates
                    if (
                        abs(raw_pitch) <= MAX_PITCH
                        and abs(raw_yaw) <= MAX_YAW
                        and abs(raw_roll) <= MAX_ROLL
                    ):

                        pitch = raw_pitch
                        yaw = raw_yaw
                        roll = raw_roll

                    else:

                        pitch = 0.0
                        yaw = 0.0
                        roll = 0.0

                else:

                    pitch = 0.0
                    yaw = 0.0
                    roll = 0.0

            except Exception:

                pitch = 0.0
                yaw = 0.0
                roll = 0.0

            # =================================================
            # HEAD MOTION
            # =================================================

            head_motion = 0.0

            if previous_head is not None:

                current_head = np.array([
                    pitch,
                    yaw,
                    roll
                ])

                previous_head_array = np.array(
                    previous_head
                )

                head_motion = float(
                    np.linalg.norm(
                        current_head -
                        previous_head_array
                    )
                )

            previous_head = (
                pitch,
                yaw,
                roll
            )

            # =================================================
            # TEMPORAL HISTORY
            # =================================================

            history.append(
                (
                    current_time,
                    ear,
                    mar,
                    eye_closed,
                    pitch,
                    yaw,
                    roll,
                )
            )

            # Keep only recent window
            while (
                history
                and
                current_time - history[0][0]
                > WINDOW_SECONDS
            ):

                history.popleft()

            # =================================================
            # FEATURE GENERATION
            # =================================================

            if (
                current_time - last_sample_time
                >= SAMPLE_INTERVAL
                and
                len(history)
                >= MIN_SAMPLES_FOR_FEATURES
            ):

                ears = np.array([
                    item[1]
                    for item in history
                ])

                mars = np.array([
                    item[2]
                    for item in history
                ])

                closed = np.array([
                    item[3]
                    for item in history
                ])

                pitches = np.array([
                    item[4]
                    for item in history
                ])

                yaws = np.array([
                    item[5]
                    for item in history
                ])

                rolls = np.array([
                    item[6]
                    for item in history
                ])

                # -------------------------------------------------
                # PERCLOS
                # -------------------------------------------------

                perclos = float(
                    np.mean(closed)
                )

                eye_closed_ratio = perclos

                # -------------------------------------------------
                # Rates
                # -------------------------------------------------

                elapsed_minutes = max(
                    (
                        current_time -
                        start_time
                    ) / 60.0,
                    1.0 / 60.0
                )

                blink_rate = (
                    blink_count /
                    elapsed_minutes
                )

                yawn_rate = (
                    yawn_count /
                    elapsed_minutes
                )

                # -------------------------------------------------
                # Feature row
                # -------------------------------------------------

                row = {

                    "ear":
                        float(np.mean(ears)),

                    "mar":
                        float(np.mean(mars)),

                    "perclos":
                        perclos,

                    "eye_closed_ratio":
                        eye_closed_ratio,

                    "max_eye_closure_s":
                        float(max_eye_closure),

                    "blink_rate":
                        float(blink_rate),

                    "yawn_rate":
                        float(yawn_rate),

                    "head_pitch":
                        float(np.mean(pitches)),

                    "head_yaw":
                        float(np.mean(yaws)),

                    "head_roll":
                        float(np.mean(rolls)),

                    "head_motion":
                        float(
                            np.std(pitches)
                            + np.std(yaws)
                            + np.std(rolls)
                        ),

                    "drowsy":
                        int(label),
                }

                # -------------------------------------------------
                # Save
                # -------------------------------------------------

                with open(
                    OUTPUT_FILE,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.DictWriter(
                        file,
                        fieldnames=FIELDNAMES
                    )

                    writer.writerow(row)

                last_sample_time = current_time

                print(
                    f"Saved | "
                    f"EAR={row['ear']:.3f} | "
                    f"MAR={row['mar']:.3f} | "
                    f"PERCLOS={row['perclos']:.2f} | "
                    f"Blink/min={row['blink_rate']:.1f} | "
                    f"Yawn/min={row['yawn_rate']:.1f} | "
                    f"Head="
                    f"({row['head_pitch']:.1f}, "
                    f"{row['head_yaw']:.1f}, "
                    f"{row['head_roll']:.1f})"
                )

            # =================================================
            # DISPLAY
            # =================================================

            label_text = (
                "DROWSY-LIKE"
                if label == "1"
                else "ALERT"
            )

            cv2.putText(
                frame,
                f"Label: {label_text}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Pitch: {pitch:.1f}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Roll: {roll:.1f}",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {len(history)}",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # =================================================
        # DISPLAY
        # =================================================

        cv2.imshow(
            "Drowsiness Feature Collector",
            frame
        )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("Data collection stopped.")
    print(
        f"Dataset saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()