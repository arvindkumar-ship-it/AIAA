"""
Synthetic data generator for AIAA intervention predictor.

IMPORTANT (read before using numbers in README/interview):
This generates labels from a hand-coded rule function (see `label_task`).
A model trained on this data learns to approximate THAT function, not
real user intervention behavior. Report it as:
  "Trained on N=<rows> synthetic samples generated from a domain-matched
   rule function (see generate_synthetic_data.py). Accuracy reflects fit
   to the generating function, not validated against real user behavior."

Run: python generate_synthetic_data.py --rows 3000 --out data/intervention_synthetic.csv
"""

import argparse
import csv
import random
from datetime import datetime, timedelta

random.seed(42)

TASK_TYPES = ["bill_payment", "appointment", "paperwork", "reminder"]

# ---- amount distribution -------------------------------------------------

def sample_amount():
    """Log-normal: most bills small, occasional large ones."""
    val = random.lognormvariate(mu=5.0, sigma=1.1)  # median ~ e^5 ~= 148
    return round(min(max(val, 10), 20000), 2)


def sample_deadline_hours():
    """Hours until deadline. Skewed toward 'not urgent'."""
    val = random.expovariate(1 / 48)  # mean ~48h
    return round(min(val, 24 * 14), 1)  # cap at 2 weeks


def sample_time_of_day():
    return random.randint(0, 23)


def sample_day_of_week():
    return random.randint(0, 6)  # 0=Mon


# ---- label rule (THIS is what the model will actually learn) -------------

AMOUNT_THRESHOLD = 800
URGENT_DEADLINE_HOURS = 6

def label_task(task_type, amount, deadline_hours, is_recurring):
    """
    Deterministic domain rule + explicit noise conditions.
    Returns (label, reason) -- reason kept for debugging/audit, not fed to model.
    """
    if task_type != "bill_payment":
        if deadline_hours < 2:
            return 1, "non_financial_but_imminent"
        return 0, "non_financial_low_risk"

    if is_recurring and amount < AMOUNT_THRESHOLD * 2:
        base = 0
        reason = "recurring_suppressed"
    elif amount >= AMOUNT_THRESHOLD:
        base = 1
        reason = "high_amount"
    else:
        base = 0
        reason = "low_amount"

    if base == 0 and deadline_hours <= URGENT_DEADLINE_HOURS:
        return 1, "deadline_override"

    return base, reason


NOISE_RATE = 0.05

def maybe_flip(label):
    if random.random() < NOISE_RATE:
        return 1 - label
    return label


def generate_rows(n):
    rows = []
    start = datetime(2026, 1, 1)
    for i in range(n):
        task_type = random.choices(
            TASK_TYPES, weights=[0.45, 0.25, 0.15, 0.15]
        )[0]
        amount = sample_amount() if task_type == "bill_payment" else 0.0
        deadline_hours = sample_deadline_hours()
        is_recurring = random.random() < 0.3 if task_type == "bill_payment" else False
        tod = sample_time_of_day()
        dow = sample_day_of_week()
        ts = start + timedelta(days=i // 20, hours=tod)

        clean_label, reason = label_task(task_type, amount, deadline_hours, is_recurring)
        final_label = maybe_flip(clean_label)

        rows.append({
            "task_id": f"synth_{i:05d}",
            "timestamp": ts.isoformat(),
            "task_type": task_type,
            "amount": amount,
            "deadline_hours": deadline_hours,
            "is_recurring": int(is_recurring),
            "time_of_day": tod,
            "day_of_week": dow,
            "intervention_label": final_label,
            "_generating_rule": reason,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--out", type=str, default="data/intervention_synthetic.csv")
    args = parser.parse_args()

    rows = generate_rows(args.rows)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    pos_rate = sum(r["intervention_label"] for r in rows) / len(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"Positive (intervene) rate: {pos_rate:.1%}")
    print("Reminder: labels come from a hand-coded rule (see label_task()).")
    print("Report results as fit-to-generating-function, not real user validation.")


if __name__ == "__main__":
    main()
