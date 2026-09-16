"""
Trains the style classifier (Random Forest) on SYNTHETIC accept/reject
data (see datasets/generate_style_data.py).

Same two guards as intervention predictor:
1. Drops '_generating_rule' column (label leak)
2. Uses class_weight='balanced' + reports balanced_accuracy/F1, not raw accuracy

Run datasets/generate_style_data.py FIRST.
pip install scikit-learn pandas numpy joblib
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "data", "style_synthetic.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "style_model.joblib")

FEATURE_COLS = ["agent_confidence", "user_past_accept_rate",
                 "action_reversibility", "deviation_from_routine"]


def load_data():
    df = pd.read_csv(DATA_PATH)

    leak_cols = [c for c in df.columns if c.startswith("_")]
    if leak_cols:
        print(f"Dropping potential label-leak columns before training: {leak_cols}")
        df = df.drop(columns=leak_cols)

    le = LabelEncoder()
    df["action_type_enc"] = le.fit_transform(df["action_type"])
    return df


def train():
    df = load_data()

    X = df[FEATURE_COLS + ["action_type_enc"]]
    y = df["accept_label"]

    print(f"Total rows: {len(df)}  |  Accept rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    always_majority_acc = max(y_test.mean(), 1 - y_test.mean())
    preds = clf.predict(X_test)

    print(f"\nAlways-predict-majority-class 'accuracy' would be: {always_majority_acc:.3f}  <- the trap")
    print(f"Real model raw accuracy:      {accuracy_score(y_test, preds):.3f}")
    print(f"Real model balanced accuracy: {balanced_accuracy_score(y_test, preds):.3f}  <- report THIS one")
    print(f"Real model F1:                {f1_score(y_test, preds):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))
    print("\nReminder: fit-to-generating-function accuracy (synthetic labels),")
    print("not validated against real user acceptance behavior.")

    joblib.dump(clf, MODEL_OUT)
    print(f"\nSaved trained model to {MODEL_OUT}")


if __name__ == "__main__":
    train()