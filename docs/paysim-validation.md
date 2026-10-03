# Cross-dataset validation on PaySim (data we didn't create)

Our pipeline trains on our own generator — this checks it also works on
PaySim, a mobile-money fraud simulator calibrated on real MFS logs
([GitHub](https://github.com/EdgarLopezPhD/PaySim), Kaggle `ealaxi/paysim1`:
6.36M rows, fraud only in TRANSFER/CASH_OUT). Offline experiment only; the API
still trains on our richer schema (devices, locations, resets).

## How to reproduce
```bash
kaggle datasets download -d ealaxi/paysim1 -p /tmp/paysim && unzip -o /tmp/paysim/paysim1.zip -d /tmp/paysim
python -m eval.paysim_adapter --in /tmp/paysim/PS_20174392719_1491204439457_log.csv --out /tmp/paysim_vigil
python -m models.train --data /tmp/paysim_vigil --artifacts /tmp/paysim_art
python -m eval.evaluate --data /tmp/paysim_vigil --artifacts /tmp/paysim_art --out docs/paysim-eval.json
```
Same model code, same metrics, no tuning. Adapter contract: `tests/test_paysim.py`.

## Results (`docs/paysim-eval.json`, 108k adapted rows, all 8,213 frauds kept)
| Metric | Model | Rule baseline |
|---|---|---|
| AUC | **0.8985** | 0.5049 (≈ random) |
| Recall@5%FPR | **0.6286** | 0.0 |
| p95 latency | 23ms | — |

## Reading these numbers honestly
- **Lower than our 0.998 — and that's the point.** PaySim has no devices,
  locations, or password resets, so our best signals read 0 (see
  `unavailable_signals_zeroed` in the adapter report). The pipeline still
  ranks fraud well using only amounts, velocity, type, and graph shape.
- **Rules collapse where ML survives.** The baseline leans on device/location
  flags that don't exist here (AUC 0.50, zero recall) — the strongest evidence
  that the ensemble adds value beyond hand-written rules, on foreign data.
- **Precision-style numbers are optimistic here.** Keep-all-fraud sampling
  enriches the test window (52% fraud); trust the rank-based AUC/recall, not
  P@100. Fairness is single-bucket (all defaults) hence non-informative.
- **Leakage we refused:** `oldbalance*`/`newbalance*` perfectly reveal PaySim
  fraud (annulled rows) and `isFlaggedFraud` is a rule label — all excluded.
  Using them would print 1.000s and prove nothing.
