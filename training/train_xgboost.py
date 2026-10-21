from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


FEATURES = [
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
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="drowsy")
    parser.add_argument("--output", default="models/drowsiness_xgb.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURES + [args.target] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    X = df[FEATURES].replace([float("inf"), -float("inf")], pd.NA).fillna(0)
    y = df[args.target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print("\nClassification report:\n", classification_report(y_test, pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, output)
    print(f"\nSaved model to {output}")


if __name__ == "__main__":
    main()
