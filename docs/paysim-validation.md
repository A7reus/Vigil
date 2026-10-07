# Cross-dataset validation on PaySim (data we didn't create)

Our pipeline trains on our own generator. This page checks it also works on
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
Same model code, same metrics, no tuning — this is *pipeline portability*
(retrained weights), distinct from the frozen-model zero-shot above.

## Frozen-model zero-shot: does the *model* generalize, or only the pipeline?

Retraining answers "does our code work elsewhere". This answers the harder
question with zero refit — frozen weights + frozen anomaly calibration score
a feed the model never saw (`eval/zeroshot.py`; no `models.train` anywhere):

```bash
# portable weights (no device/location/reset signals)
python -m models.train --data data --artifacts artifacts_intersect --feature-set intersect
# foreign feed: fresh generator run, unseen seed + different fraud mix
python -m data_gen.generate --seed 7 --fraud-rate 0.06 --out /tmp/foreign
# frozen scoring (intersect AND full weights)
python -m eval.zeroshot --data /tmp/foreign --artifacts artifacts_intersect --out /tmp/zs-intersect.json
python -m eval.zeroshot --data /tmp/foreign --artifacts artifacts --out /tmp/zs-full.json
# foreign generator: adapted PaySim (needs the Kaggle CSV, same adapter)
python -m eval.zeroshot --data /tmp/paysim_vigil --artifacts artifacts_intersect --out docs/paysim-zeroshot.json
```

Executed 2026-10-07 (source: seed-42 50k run; foreign: seed-7 40k run, 6% fraud):

| Setup | AUC | Recall@5%FPR | Frozen 0.60 band (P/R) | Frozen 0.85 band (P/R) |
|---|---|---|---|---|
| In-domain (ours→ours, full) | 0.999 | 0.990 | — | — |
| Zero-shot full weights → foreign | **0.996** | **0.990** | 0.95 / 0.96 | 0.98 / 0.93 |
| Zero-shot intersect → foreign | **0.984** | **0.966** | 0.92 / 0.91 | 0.97 / 0.85 |
| Retrained → PaySim (below) | 0.899 | 0.629 | — | — |
| Zero-shot intersect → PaySim | pending CSV — command above | | | |

Read: frozen weights hold up across distributions, and the operating bands
transfer (a Medium flag still means ~0.92+ precision on unseen data). The
intersect variant exists for feeds missing our device/location/reset signals;
where those exist, full weights transfer best. Simulator-to-simulator is
mechanism-generality evidence, not real-world proof — stated as such.

## Results (`docs/paysim-eval.json`, 108k adapted rows, all 8,213 frauds kept)
| Metric | Model | Rule baseline |
|---|---|---|
| AUC | **0.8985** | 0.5049 (≈ random) |
| Recall@5%FPR | **0.6286** | 0.0 |
| p95 latency | 23ms | n/a (rules need no timing) |

## Reading these numbers honestly
- **Lower than our 0.998, and that is the point.** PaySim has no devices,
  locations, or password resets, so our best signals read 0 (see
  `unavailable_signals_zeroed` in the adapter report). The pipeline still
  ranks fraud well using only amounts, velocity, type, and graph shape.
- **Rules collapse where ML survives.** The baseline leans on device/location
  flags that don't exist here (AUC 0.50, zero recall), which is the strongest
  evidence that the ensemble adds value beyond hand-written rules, on foreign data.
- **Precision-style numbers are optimistic here.** Keep-all-fraud sampling
  enriches the test window (52% fraud); trust the rank-based AUC/recall, not
  P@100. Fairness is single-bucket (all defaults) hence non-informative.
- **Leakage we refused:** `oldbalance*`/`newbalance*` perfectly reveal PaySim
  fraud (annulled rows) and `isFlaggedFraud` is a rule label, so all are excluded.
  Using them would print 1.000s and prove nothing.
