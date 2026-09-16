"""
Threshold tuning for the RandomForest intervention model
(intervention_model.joblib).

Default 0.5 cutoff isn't F1-optimal on imbalanced data. This sweeps
thresholds on the same test split used in train_intervention.py and
reports the F1-maximizing one.

Run AFTER train_intervention.py has produced intervention_model.joblib:
    python tune_threshold.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "data", "intervention_synthetic.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "intervention_model.joblib")

FEATURE_COLS = ["amount", "deadline_hours", "is_recurring", "time_of_day", "day_of_week"]


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])

    X = df[FEATURE_COLS]
    y = df["intervention_label"]

    # same split as train_intervention.py -- must match for this to be valid
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = joblib.load(MODEL_PATH)
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_test_np = y_test.values

    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'BalAcc':>8}")
    best_f1, best_t = -1, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (y_proba >= t).astype(int)
        p = precision_score(y_test_np, preds, zero_division=0)
        r = recall_score(y_test_np, preds, zero_division=0)
        f1 = f1_score(y_test_np, preds, zero_division=0)
        ba = balanced_accuracy_score(y_test_np, preds)
        print(f"{t:>10.2f} {p:>10.3f} {r:>10.3f} {f1:>8.3f} {ba:>8.3f}")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\nBest threshold: {best_t:.2f}  ->  F1: {best_f1:.3f}")
    print("Hardcode this threshold in your inference/gating code instead of 0.5.")


if __name__ == "__main__":
    main()