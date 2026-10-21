"""
Starter utility for constructing an ML feature CSV from labeled session records.

A production dataset should be generated from videos/sessions with driver-level
splits. This template shows the expected schema; implement your video/session
reader here rather than accidentally leaking frames between train and test.
"""

from pathlib import Path
import pandas as pd


COLUMNS = [
    "session_id",
    "driver_id",
    "timestamp",
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


def create_empty_template(path="data/features_template.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
    print(f"Created {path}")


if __name__ == "__main__":
    create_empty_template()
