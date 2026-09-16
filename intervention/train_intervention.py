"""
Trains the intervention predictor using RandomForest on SYNTHETIC
domain-rule data (see datasets/generate_synthetic_data.py).

WHY RANDOMFOREST, NOT LSTM:
An LSTM version was tried first (see git history / train_lstm.py).
It underperformed and was unstable across runs (F1 0.18-0.39,
balanced accuracy 0.48-0.71 depending on random init -- no seed was
set, so results weren't even reproducible). Root cause: the LSTM
was fed artificial 5-row "sequences" built from independent,
randomly-generated events with no real temporal dependency between
them -- a sequence model was applied to non-sequential tabular data.
A flat RandomForest on the same features scored F1 0.73, balanced
accuracy 0.81 on the same data -- confirming the task has no genuine
sequential structure. LSTM code is kept in git history for reference
but is no longer the production path.

HONEST FRAMING FOR README/INTERVIEW:
This model learns to approximate the hand-coded rule function in
generate_synthetic_data.py's label_task(). It does NOT validate real
user intervention behavior. Report accuracy as "fit to generating
function", not as evidence the system understands real users.

Guards against the same two traps as before:
1. Label leakage: '_generating_rule' column dropped before training.
2. Class imbalance (~13% positive): class_weight='balanced' +
   balanced_accuracy/F1 reported instead of raw accuracy.

Run datasets/generate_synthetic_data.py FIRST.
pip install scikit-learn pandas numpy joblib
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "data", "intervention_synthetic.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "intervention_model.joblib")

FEATURE_COLS = ["amount", "deadline_hours", "is_recurring", "time_of_day", "day_of_week"]


def load_data():
    df = pd.read_csv(DATA_PATH)

    # TRAP 1 GUARD: drop the label-leak column before it ever touches training
    leak_cols = [c for c in df.columns if c.startswith("_")]
    if leak_cols:
        print(f"Dropping potential label-leak columns before training: {leak_cols}")
        df = df.drop(columns=leak_cols)

    return df


def train():
    df = load_data()

    X = df[FEATURE_COLS]
    y = df["intervention_label"]

    print(f"Total rows: {len(df)}  |  Positive (intervene) rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TRAP 2 GUARD: class_weight='balanced' so the model can't just predict 0 everywhere
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    always_majority_acc = max(y_test.mean(), 1 - y_test.mean())
    preds = clf.predict(X_test)

    print(f"\nAlways-predict-majority-class 'accuracy' would be: {always_majority_acc:.3f}  <- the trap")
    print(f"Raw accuracy (misleading on imbalanced data): {accuracy_score(y_test, preds):.4f}")
    print(f"Balanced accuracy (the honest number):        {balanced_accuracy_score(y_test, preds):.4f}")
    print(f"F1 score:                                     {f1_score(y_test, preds, zero_division=0):.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))
    print("\nReminder: fit-to-generating-function accuracy (synthetic labels),")
    print("not validated against real user intervention behavior.")

    # feature importance -- useful for the README / interview talking points
    importances = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1])
    print("\nFeature importances:")
    for name, imp in importances:
        print(f"  {name:<15} {imp:.3f}")

    joblib.dump(clf, MODEL_OUT)
    print(f"\nSaved trained model to {MODEL_OUT}")


if __name__ == "__main__":
    train()