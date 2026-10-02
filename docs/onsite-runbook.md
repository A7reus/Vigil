# On-site final runbook (7 Oct 2026, DIU)

New requirements land at the start of the day; the last 90 minutes are judging.
Optimize for fast, committable, demoable changes.

## 1. Before leaving home
- `python -m pytest -q` green; `artifacts/` + `data/` regenerated at 50k;
  `docs/eval-sample.json` fresh; all commits pushed (`git status` clean).
- Save an offline bundle: this repo + `data/` + `artifacts/` on disk, venue net
  may fail. The LLM falls back to deterministic EN/BN templates without a key.
- Note the CPU-only path: XGBoost trains in ~10s on CPU, no GPU needed.

## 2. When new requirements drop (first 30 min)
- Map to the cheapest layer:
  - threshold/band/weight change → `config/thresholds.yaml` only (<30 min, no retrain)
  - new scam text/pattern → `data_gen/generate.py` + `python -m models.train --sample 10000` (~60s) + `python -m eval.evaluate --sample 5000`
  - new narrative language/format → `api/llm.py` templates only
  - new queue filter/sort → `web/app.js` only
- Keep ML and rules separate (architecture requirement); never hide a policy
  decision inside an LLM prompt.

## 3. Implement + commit continuously (middle block)
- Small commits per change (`git add -p`), push often; judges verify history.
- Bound cold start on the venue machine: `VIGIL_ALERTS_LIMIT=100 uvicorn api.main:app --port 8000`.
- Verify after each change: `pytest -q`, one `/score`, one `/case`, open `/`.

## 4. Final 90 minutes (judging)
- Freeze code 15 min early. Keep running: API + `GET /` console.
- Demo path: high alert → mule-ring sample → EN/BN narrative → decision log →
  `/metrics` (requests, decisions, startup). State metrics from `eval-sample.json`.
- Q&A prep: calibration (`anomaly_calib.npy`), no-leakage features, fairness
  numbers, why ensemble beats rules, scale path in `scale-plan.md`.
