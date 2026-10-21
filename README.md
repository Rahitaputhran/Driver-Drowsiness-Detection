# Driver Drowsiness Detection System

A real-time Python computer-vision + ML/DL project for detecting driver drowsiness.

## Architecture

Camera
→ MediaPipe Face Landmarker
→ EAR / MAR / head-pose / temporal features
→ optional CNN eye-state probability
→ XGBoost classifier
→ temporal smoothing + hysteresis
→ alert/dashboard

The implementation is designed to favor lightweight, real-time algorithms:
- MediaPipe Face Landmarker for facial landmarks
- EAR for eye openness
- MAR for yawning
- PERCLOS and temporal statistics
- solvePnP for head pose
- XGBoost for engineered tabular features
- MobileNetV3Small option for deep eye-state classification
- temporal hysteresis to reduce false alarms

## Project structure

```text
driver_drowsiness_detection/
├── app/
│   ├── realtime.py
│   └── streamlit_app.py
├── src/
│   ├── config.py
│   ├── features.py
│   ├── face_landmarks.py
│   ├── head_pose.py
│   ├── temporal.py
│   ├── decision.py
│   └── alarm.py
├── training/
│   ├── make_feature_dataset.py
│   ├── train_xgboost.py
│   └── train_eye_cnn.py
├── models/
│   └── README.md
├── data/
│   └── README.md
├── tests/
│   └── test_features.py
├── requirements.txt
├── requirements-cpu.txt
├── .gitignore
└── README.md
```

## 1. Create the environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
```

If your machine supports TensorFlow normally:

```powershell
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Then select `.venv` as the Python interpreter in VS Code.

## 2. Download the MediaPipe face model

The application expects a MediaPipe Face Landmarker task model at:

```text
models/face_landmarker.task
```

Download the official `face_landmarker.task` model from the MediaPipe documentation/model repository and place it there.

The code intentionally does not silently download model files at runtime.

## 3. Run real-time detection

```powershell
python -m app.realtime
```

Press `Q` to quit.

The camera window displays:
- Alert / Warning / Drowsy state
- EAR
- MAR
- PERCLOS
- blink/yawn timing
- head pose
- drowsiness score

## 4. Train the ML model

The intended workflow is:

1. Collect labeled sessions/features.
2. Save a CSV under `data/`.
3. Run:

```powershell
python training/train_xgboost.py --csv data/features.csv --target drowsy
```

The script:
- performs a stratified train/test split
- trains XGBoost
- reports precision, recall, F1, ROC-AUC where possible
- saves the model to `models/drowsiness_xgb.joblib`

### Important

Do not evaluate by randomly splitting individual frames if frames from the same video/person occur in both train and test. For a serious evaluation, split by driver/session to avoid data leakage.

## 5. Optional deep-learning eye classifier

The CNN trainer expects:

```text
data/eye_dataset/
├── train/
│   ├── open/
│   └── closed/
├── val/
│   ├── open/
│   └── closed/
└── test/
    ├── open/
    └── closed/
```

Run:

```powershell
python training/train_eye_cnn.py --data-dir data/eye_dataset
```

The default backbone is MobileNetV3Small with ImageNet initialization and a small classification head.

## 6. Streamlit dashboard

```powershell
streamlit run app/streamlit_app.py
```

The dashboard can show the architecture/status and can be extended to host the camera inference loop.

## Notes on the "optimal" design

This project deliberately avoids an unnecessarily heavy end-to-end model. For a laptop/webcam application, a lightweight landmark detector plus temporal features is usually much more practical than a large vision model.

The system uses a hybrid design:

- Geometry: EAR, MAR, head pose
- Temporal behavior: PERCLOS, closure duration, blink/yawn timing
- ML: XGBoost on engineered features
- DL: MobileNetV3Small eye-state classifier when a labeled eye dataset is available
- Decision layer: hysteresis/rolling windows rather than a single-frame threshold

This makes the system fast, interpretable, and interview-friendly.

## Safety

This is a research/portfolio project and must not be treated as a safety-certified automotive driver-monitoring system. Camera conditions, glasses, lighting, face angle, and individual differences can affect performance.
