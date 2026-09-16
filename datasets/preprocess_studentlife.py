"""
Converts raw StudentLife EMA + sensing data into (context_features, label)
pairs for the intervention LSTM.

Task framing: predict whether an interruption (EMA prompt) was answered
promptly ("receptive") vs ignored/delayed ("not receptive") — this is our
real proxy label for "should the agent interrupt the user right now?"

Run download_studentlife.py FIRST.
pip install pandas numpy
"""

import os
import json
import glob
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "studentlife_raw")
OUT_PATH = os.path.join(os.path.dirname(__file__), "intervention_training_data.csv")


def load_ema_response_times():
    """
    EMA folder has per-user JSON files of prompted questions with response
    timestamps. A fast response = receptive to interruption (label=1),
    a very delayed / missing response = not receptive (label=0).
    """
    ema_files = glob.glob(os.path.join(RAW_DIR, "EMA", "response", "**", "*.json"), recursive=True)
    rows = []
    for fp in ema_files:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            continue
        uid = os.path.basename(fp).split("_")[-1].replace(".json", "")
        for entry in (data if isinstance(data, list) else []):
            resp_time = entry.get("resp_time")
            null_time = entry.get("null_time")  # timestamp of missed/expired prompt
            if resp_time:
                rows.append({"user": uid, "timestamp": resp_time, "label": 1})
            elif null_time:
                rows.append({"user": uid, "timestamp": null_time, "label": 0})
    return pd.DataFrame(rows)


def load_activity_context():
    """
    Sensing/activity folder has per-user activity inference (still/walking/
    running/unknown) logged over time — this becomes our context feature
    (what was the user doing when the interruption happened).
    """
    act_files = glob.glob(os.path.join(RAW_DIR, "sensing", "activity", "*.csv"))
    frames = []
    for fp in act_files:
        uid = os.path.basename(fp).replace("activity_", "").replace(".csv", "")
        try:
            df = pd.read_csv(fp)
            df["user"] = uid
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["user", "timestamp", "activity_inference"])
    return pd.concat(frames, ignore_index=True)


def build_training_set():
    ema = load_ema_response_times()
    activity = load_activity_context()

    if ema.empty:
        raise RuntimeError(
            "No EMA data found — check that download_studentlife.py completed "
            "and RAW_DIR points at the extracted folder."
        )

    # nearest-activity join: for each EMA event, find closest activity reading
    activity = activity.sort_values("timestamp")
    merged_rows = []
    for _, row in ema.iterrows():
        user_acts = activity[activity["user"] == row["user"]]
        if user_acts.empty:
            continue
        idx = (user_acts["timestamp"] - row["timestamp"]).abs().idxmin()
        act_val = user_acts.loc[idx, "activity_inference"] if "activity_inference" in user_acts.columns else -1
        hour_of_day = pd.to_datetime(row["timestamp"], unit="s", errors="coerce")
        merged_rows.append({
            "user": row["user"],
            "hour_of_day": hour_of_day.hour if pd.notnull(hour_of_day) else -1,
            "activity_inference": act_val,
            "label": row["label"],
        })

    df = pd.DataFrame(merged_rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df["label"].value_counts())
    return df


if __name__ == "__main__":
    build_training_set()
