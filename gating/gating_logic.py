"""
Combines the trained RandomForest model with the old hardcoded rule.
Model is the primary decision-maker; the rule is a demoted safety-net
that only fires when the model is unsure (or not yet trained).

Every call logs which path decided and why, so this data becomes the
training set for the NEXT retrain -- this is your real feedback loop.

NOTE: previously wired to the LSTM (see git history / legacy_lstm/).
Switched to RandomForest after testing showed the LSTM was misapplied
to non-sequential data (F1 0.39 vs RandomForest's 0.73 on identical
features). See intervention/train_intervention.py.
"""

import os
import sys
import joblib
import numpy as np
# gating/gating_logic.py — top imports mein ye line add kar:
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "logging"))
from db import log_task_event  # noqa

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "intervention", "intervention_model.joblib")

# Must match datasets/generate_synthetic_data.py's AMOUNT_THRESHOLD (800),
# NOT an independently chosen number -- otherwise the rule fallback and
# the model disagree systematically because they were never trained on
# the same logic. Keep these two values in sync if either changes.
RULE_THRESHOLD = 800

# F1-optimal decision threshold from intervention/tune_threshold.py --
# NOT the default 0.5. Re-run tune_threshold.py after any retrain and
# update this constant.
DECISION_THRESHOLD = 0.65

# Confidence band: how far from DECISION_THRESHOLD the model's probability
# needs to be before we trust it over the rule. This is a SEPARATE concept
# from DECISION_THRESHOLD -- one says "where's the cutoff", the other says
# "how sure does the model need to be before we skip the safety net".
CONFIDENCE_MARGIN = 0.15  # model_prob must be outside [threshold-margin, threshold+margin]

FEATURE_COLS = ["amount", "deadline_hours", "is_recurring", "time_of_day", "day_of_week"]

_model = None


def _load_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        else:
            _model = False  # sentinel: tried and not found, don't retry every call
    return _model or None


def rule_based_decision(context: dict) -> bool:
    """The original hardcoded logic -- kept as an explicit, auditable fallback."""
    amount = context.get("amount", 0)
    return amount > RULE_THRESHOLD


def _build_features(context: dict):
    row = {col: context.get(col, 0) for col in FEATURE_COLS}
    return pd.DataFrame([row])  # pandas import karna hoga: import pandas as pd

def gated_decision(context: dict) -> dict:
    """
    Returns the real decision + which path made it + the model's
    probability (if available), and logs everything to Postgres.

    context: dict with keys matching FEATURE_COLS (amount, deadline_hours,
    is_recurring, time_of_day, day_of_week) plus task_type for logging.
    """
    model = _load_model()
    rule_result = rule_based_decision(context)

    model_prob = None
    decision_source = "rule"
    final_decision = rule_result

    if model is not None:
        X = _build_features(context)
        model_prob = float(model.predict_proba(X)[0, 1])

        lower = DECISION_THRESHOLD - CONFIDENCE_MARGIN
        upper = DECISION_THRESHOLD + CONFIDENCE_MARGIN
        if model_prob >= upper or model_prob <= lower:
            # model is confident either way -> let it decide, using the
            # F1-tuned threshold, not 0.5
            final_decision = model_prob >= DECISION_THRESHOLD
            decision_source = "model"
        else:
            # model unsure -> fall back to the rule, and log this as a
            # disagreement case worth relabeling for the next retrain
            final_decision = rule_result
            decision_source = "rule_fallback_low_confidence"

    task_id = log_task_event(
        task_type=context.get("task_type", "unknown"),
        context=context,
        rule_decision=rule_result,
        rule_threshold=RULE_THRESHOLD,
        model_prob=model_prob,
        final_decision=final_decision,
        decision_source=decision_source,
    )

    return {
        "task_id": task_id,
        "decision": final_decision,
        "source": decision_source,
        "model_prob": model_prob,
        "rule_decision": rule_result,
    }


if __name__ == "__main__":
    # Smoke test -- before intervention_model.joblib exists, this always
    # falls back to rule
    result = gated_decision({
        "task_type": "bill_payment",
        "amount": 900,
        "deadline_hours": 24,
        "is_recurring": 0,
        "time_of_day": 14,
        "day_of_week": 2,
    })
    print(result)