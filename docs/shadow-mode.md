# Shadow-mode go-live plan (item 12)

The honest path from hackathon to a real client: score real traffic, take no
action, compare against analyst outcomes. No customer impact until the numbers
earn it.

## Phase 0 — readiness gate (before any client data)
- `pytest -q` green; `/ready` returns 200; eval metrics reproduced on demand.
- `VIGIL_SEED_DEMO=0`, real admin created, all analysts registered + approved.
- Thresholds reset to review-heavy (raise `bands.medium`, keep holds off:
  set `bands.high` above any observed score or route High to review).

## Phase 1 — historical backtest (days 1–7)
- Client provides 30–90 days of labeled history (labels may arrive late —
  see pilot-backlog on label delay).
- Run `eval.paysim_adapter`-style ingestion (client CSV → Vigil schema),
  train a candidate, report AUC / Precision@review-capacity / Recall@5%FPR
  against their rule baseline. Go/no-go: beat the baseline at *their*
  review capacity, not ours.

## Phase 2 — live shadow (weeks 2–6)
- Stream live transfers to `POST /score` (no `commit` until labels flow);
  actions stay `review`-only. Analysts work cases normally in the console.
- Nightly: compare model flags vs analyst outcomes. Track precision at the
  team's daily review capacity, false-hold rate, and fairness slices.
- Recalibrate `config/thresholds.yaml` weekly from observed data. Never
  tune on the same cases you report.

## Phase 3 — gradual enforcement
- Enable holds for the top band only, capped share of traffic, with instant
  rollback (previous `artifacts/` + thresholds in git; SQLite backup first).
- Expand bands only while measured precision holds. Any regression in
  fairness slices pauses expansion.

## Exit criteria for full production
Precision at capacity ≥ agreed target for 4 consecutive weeks, fairness gaps
documented with a mitigation plan, retraining loop (pilot-backlog) running,
runbook exercised in a game day.
