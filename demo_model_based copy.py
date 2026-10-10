import cv2
import time
import mediapipe as mp
from math import hypot, degrees, atan2

# ---------------- Configuration ----------------
CAMERA_INDEX = 0
EAR_THRESHOLD = 0.28
MAR_THRESHOLD = 0.75
TILT_THRESHOLD = 25.0
EYE_CLOSED_DURATION = 2.0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ---------------- Feature calculations ----------------
def euclidean(p1, p2):
    return hypot(p2[0] - p1[0], p2[1] - p1[1])


def get_ear(landmarks, eye_indices):
    vertical_1 = euclidean(
        landmarks[eye_indices[1]], landmarks[eye_indices[5]]
    )
    vertical_2 = euclidean(
        landmarks[eye_indices[2]], landmarks[eye_indices[4]]
    )
    horizontal = euclidean(
        landmarks[eye_indices[0]], landmarks[eye_indices[3]]
    )

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_mar(landmarks):
    mouth_height = euclidean(landmarks[13], landmarks[14])
    mouth_width = euclidean(landmarks[78], landmarks[308])

    if mouth_width == 0:
        return 0.0

    return mouth_height / mouth_width


def get_head_tilt(landmarks):
    # Approximate roll angle using the line between the eye corners.
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    if dx == 0 and dy == 0:
        return 0.0

    return abs(degrees(atan2(dy, dx)))


# ---------------- Main application ----------------
def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(
            "Error: Could not open webcam. Check camera permissions "
            "or change CAMERA_INDEX to 1."
        )
        face_mesh.close()
        return

    eye_closed_start = None
    print("Drowsiness demo running. Press Q in the video window to quit.")
    print("Note: This is a threshold-based demo, not the repository's ML model.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read a frame from the webcam.")
                break

            # Mirror the camera image for a more natural preview.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            status = "No face detected"
            color = (0, 0, 255)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [
                    (int(point.x * width), int(point.y * height))
                    for point in face_landmarks.landmark
                ]

                left_ear = get_ear(landmarks, LEFT_EYE)
                right_ear = get_ear(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0
                mar = get_mar(landmarks)
                tilt = get_head_tilt(landmarks)

                eyes_closed = ear < EAR_THRESHOLD
                mouth_open = mar > MAR_THRESHOLD
                head_tilted = tilt > TILT_THRESHOLD

                if eyes_closed:
                    if eye_closed_start is None:
                        eye_closed_start = time.time()
                    closed_duration = time.time() - eye_closed_start
                else:
                    eye_closed_start = None
                    closed_duration = 0.0

                if closed_duration >= EYE_CLOSED_DURATION:
                    status = "Drowsy: eyes closed"
                elif mouth_open:
                    status = "Possible yawn"
                elif eyes_closed and head_tilted:
                    status = "Possible drowsiness: head tilt"
                else:
                    status = "Alert"

                color = (0, 0, 255) if "Drowsy" in status or "Possible" in status else (0, 200, 0)

                cv2.putText(
                    frame, f"EAR: {ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
                )
                cv2.putText(
                    frame, f"MAR: {mar:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
                )
                cv2.putText(
                    frame, f"Head roll: {tilt:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
                )

            else:
                # Require a fresh continuous closure period after face loss.
                eye_closed_start = None

            cv2.putText(
                frame, status, (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
            )
            cv2.imshow("Driver Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    main()
