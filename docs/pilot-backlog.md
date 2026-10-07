# Pilot backlog: honest follow-ups, scoped but not started

## 13. Held-out fraud-pattern test
Train with one fraud family removed (e.g. all mule rows), test only on it.
This measures generalization to unseen tactics instead of memorized patterns.
Expected: lower than headline metrics — report both, and state that
thresholds must be recalibrated on client data regardless.

## Label delay
Real fraud labels arrive days or weeks late (chargebacks, investigations).
Until then: treat analyst decisions as *review outcomes*, not ground truth;
evaluate on rolling labeled windows; never retrain on the same cases you
report. The decisions table already stores `model_version` + timestamp, so
delayed-label joins are a query, not a migration.

## 16. Challenger/rollback retraining loop
Nightly: train a challenger on confirmed outcomes (not raw clicks), compare
challenger vs champion on the last labeled window (AUC + precision at review
capacity), promote only on a clear win with fairness slices intact. Keep the
previous `artifacts/` directory versioned for one-command rollback; the
`model_version` on every score and decision makes the cutover auditable.

## Deferred infrastructure
- Multi-worker / Redis for rate buckets, live cases, and ops counters
  (today: single worker, documented).
- Service API keys alongside human JWT-style sessions.
- Per-segment thresholds once fairness slices show stable gaps.
- LightGBM challenger path (installed, metrics tied; not wired into training).
