# Data

For the XGBoost pipeline, create a CSV containing numerical features and a target column named `drowsy`.

Recommended feature columns include:

- ear
- mar
- perclos
- eye_closed_ratio
- max_eye_closure_s
- blink_rate
- yawn_rate
- head_pitch
- head_yaw
- head_roll
- head_motion

For the CNN, use:

```text
eye_dataset/
├── train/open
├── train/closed
├── val/open
├── val/closed
├── test/open
└── test/closed
```

For rigorous evaluation, keep drivers/sessions separated between train, validation, and test.
