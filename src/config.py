from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
XGB_MODEL_PATH = MODEL_DIR / "drowsiness_xgb.joblib"
CNN_MODEL_PATH = MODEL_DIR / "eye_state_mobilenet.keras"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Geometry features
EAR_CLOSED_THRESHOLD = 0.21
MAR_YAWN_THRESHOLD = 0.55

# Temporal defaults
WINDOW_SECONDS = 30.0
DROWSY_CLOSED_SECONDS = 1.5
WARNING_CLOSED_SECONDS = 0.8

# Decision thresholds
WARNING_SCORE = 0.50
DROWSY_SCORE = 0.75
