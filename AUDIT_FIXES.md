# Model availability and evidence

Repair branch: `fix/internship-audit-20261010`. Changes are scoped to audit findings; no deployment or live provider action is included.

## Changed

Missing or incompatible LSTM weights now prevent prediction and require human review. Calibration is off by default; it requires explicit held-out probabilities and labels instead of random labels. Metrics reject mismatched samples and do not invent AUC for one-class data.

## Verification

`pytest -q`

Commands are entry points, not a claim that every live integration was executed. See the repair bundle for actual results.

## Setup and remaining evidence

Model tests use a synthetic weight fixture to test loading and forwarding, not predictive performance. No human-labelled intervention benchmark, learned timing metric or calibration claim has been established. Collect consented examples, split by user/task before training, record dataset/artifact hashes and evaluate held-out calibration and failure costs. Calendar credentials and real portal execution require separate manual setup. Do not present synthetic rule fit as real user accuracy.
