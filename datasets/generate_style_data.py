"""
Synthetic data generator for AIAA style classifier.

IMPORTANT (same rule as intervention generator):
Labels come from a hand-coded rule (see label_style()). Report as:
  "Trained on N=<rows> synthetic samples generated from a domain-matched
   rule function. Accuracy reflects fit to the generating function,
   not validated against real user acceptance behavior."

Task: predict whether a user will ACCEPT (1) or REJECT/EDIT (0) an
agent-proposed action, based on interaction context.

Run: python generate_style_data.py --rows 3000 --out data/style_synthetic.csv
"""

import argparse
import csv
import random

random.seed(42)

ACTION_TYPES = ["schedule_event", "pay_bill", "draft_message", "reorganize_files", "send_reminder"]


def sample_confidence():
    """Agent's own confidence score for the action it proposed."""
    return round(random.betavariate(5, 2), 3)  # skewed toward higher confidence


def sample_past_acceptance_rate():
    """Rolling acceptance rate for this user over their last N actions (0-1)."""
    return round(random.betavariate(4, 3), 3)


def sample_action_reversibility():
    """0 = irreversible (e.g. sent message), 1 = fully reversible (e.g. draft only)."""
    return round(random.random(), 3)


def sample_deviation_from_routine():
    """0 = matches user's usual pattern, 1 = totally novel action."""
    return round(random.betavariate(2, 5), 3)  # usually low deviation


# ---- label rule -----------------------------------------------------

ACCEPT_CONF_THRESHOLD = 0.7
HIGH_DEVIATION = 0.6

def label_style(confidence, past_accept_rate, reversibility, deviation):
    """
    Deterministic domain rule + explicit noise conditions.
    """
    # low reversibility (risky, hard to undo) makes users more cautious
    if reversibility < 0.3 and confidence < 0.85:
        return 0, "risky_irreversible_low_conf"

    # high deviation from routine gets rejected unless confidence very high
    if deviation > HIGH_DEVIATION and confidence < 0.9:
        return 0, "novel_action_distrust"

    # user's own history is a strong prior
    if past_accept_rate > 0.7 and confidence > ACCEPT_CONF_THRESHOLD:
        return 1, "high_trust_high_conf"

    if past_accept_rate < 0.3:
        return 0, "low_trust_user"

    if confidence > ACCEPT_CONF_THRESHOLD:
        return 1, "confident_default_accept"

    return 0, "low_confidence_default_reject"


NOISE_RATE = 0.07

def maybe_flip(label):
    if random.random() < NOISE_RATE:
        return 1 - label
    return label


def generate_rows(n):
    rows = []
    for i in range(n):
        action_type = random.choice(ACTION_TYPES)
        confidence = sample_confidence()
        past_accept = sample_past_acceptance_rate()
        reversibility = sample_action_reversibility()
        deviation = sample_deviation_from_routine()

        clean_label, reason = label_style(confidence, past_accept, reversibility, deviation)
        final_label = maybe_flip(clean_label)

        rows.append({
            "task_id": f"style_synth_{i:05d}",
            "action_type": action_type,
            "agent_confidence": confidence,
            "user_past_accept_rate": past_accept,
            "action_reversibility": reversibility,
            "deviation_from_routine": deviation,
            "accept_label": final_label,
            "_generating_rule": reason,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--out", type=str, default="data/style_synthetic.csv")
    args = parser.parse_args()

    rows = generate_rows(args.rows)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    pos_rate = sum(r["accept_label"] for r in rows) / len(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"Accept rate: {pos_rate:.1%}")
    print("Reminder: labels come from a hand-coded rule (see label_style()).")
    print("Report results as fit-to-generating-function, not real user validation.")


if __name__ == "__main__":
    main()