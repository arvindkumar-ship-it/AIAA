# Autonomous Intervention-Aware Agent (AIAA)

Research prototype for memory handling, intervention decisions, collaboration-style classification and autonomy rules. The demo currently validates task inputs and uses rule-based decisions; it does not establish autonomous browser/portal execution.

The intervention LSTM only predicts when compatible trained weights are explicitly available. Missing weights require human review. Calibration requires supplied held-out probabilities and labels; random calibration is not used. Synthetic rule-generated datasets measure fit to that generator, not real user behavior.

## Reproducible checks

```sh
python -m pytest -q
```

Six offline regression checks cover missing/incompatible weights, explicit calibration inputs, model forwarding with synthetic weights, and metric validation. They establish no trained accuracy, timing accuracy, cost reduction or success-rate claim. Historical README percentages had inconsistent datasets/definitions and are not presented as validated results.

Real Google Calendar credentials and portal actions require separate setup and consent. Do not run live action scripts merely to reproduce a unit test.

See [AUDIT_FIXES.md](AUDIT_FIXES.md) for remaining data and evaluation work.
