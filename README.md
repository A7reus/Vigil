# Vigil — Trust & Risk Intelligence for upay (Track 01)

> For upay users losing money to scams/ATO/mules and analysts drowning in opaque alerts, slow manual review causes loss + churn. We build a real-time risk scorer + graph + LLM investigator on synthetic transactions to score, explain, and recommend action — measured by Precision@100 + investigation time saved.

Answers the Track 01 test: **What happened? Why is it risky? What should upay do next?**

## Features
- **Real-time scoring API** — `POST /score` in → `risk_score 0-1, risk_level, top_3_reasons, recommended_action` out. Ensemble `0.7·classifier + 0.2·anomaly + 0.1·graph`. Bands in `config/thresholds.yaml`: `>0.85` hold+step-up+review, `0.6–0.85` review, `<0.6` allow. Never auto-blocks money.
- **AI engine (4x)** — XGBoost classifier (HGB fallback) + IsolationForest anomaly (train-calibrated percentile, no test leakage) + NetworkX 2-hop mule boost + grounded LLM investigator (EN/BN, offline fallback). Reasons combine auditable rules with per-row SHAP attributions.
- **Analyst queue** — `GET /alerts` (pre-scored, risk-sorted), `GET /case/:id` (timeline + narrative, reuses cached causal features), `POST /decision` (feedback loop for retrain).
- **Analyst console** — no-build static frontend in `/web` served at `GET /`: risk queue with level filter + search, case detail with EN/BN narrative + timeline + decision buttons, and a `POST /score` playground.
- **Evaluation** — `python -m eval.evaluate`: Precision@100, Recall@5%FPR, AUC vs rule baseline, p95 latency, fairness FPR by district/account-age, business simulation (loss prevented, analyst-minutes saved). Batched scoring (~2s for 8k rows vs ~100s before).
- **Cross-dataset check** — same pipeline on PaySim MFS data (`eval/paysim_adapter.py`, offline): AUC 0.90 vs rules 0.50 with our best signals unavailable. See `docs/paysim-validation.md`.

## Technology stack
Python 3.12+, Pandas, NumPy, Scikit-learn, NetworkX, FastAPI/Uvicorn, PyYAML, Joblib.
ML: XGBoost primary classifier (HGB fallback if absent), per-row SHAP attributions
(lazy, ~1.4ms, falls back to global importance), IsolationForest anomaly
(train-calibrated percentile), NetworkX 2-hop mule boost. LLM: any
OpenAI-compatible API (Groq default) with deterministic offline template fallback.

## Requirements
- Python 3.12+ with venv (pandas 3 requires it)
- 2 GB RAM, no GPU needed (CPU-only; `nvidia-nccl` wheels ship with XGBoost but are unused)
- Optional `LLM_API_KEY` for live narratives (works offline without it)

## Installation and setup
```bash
python3 -m venv ~/.venvs/vigil && source ~/.venvs/vigil/bin/activate
pip install -r requirements.txt
python -m data_gen.generate --n-customers 5000 --n-txns 50000 --out data
python -m models.train --data data --artifacts artifacts
```

## Environment variables
Copy `.env.example` to `.env` (git-ignored, loaded on startup); plain
environment variables take precedence over the file.

| Name | Purpose | Example |
|---|---|---|
| `LLM_API_KEY` | Live investigator narratives (leave unset for offline fallback) | `gsk_...` (placeholder — never commit secrets) |
| `LLM_BASE_URL` | OpenAI-compatible endpoint | `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | Chat model (default follows Groq's post-Aug-2026 replacement) | `openai/gpt-oss-20b` |
| `VIGIL_ALERTS_LIMIT` | Pre-scored queue size at startup (bounds cold start) | `200` |
| `VIGIL_RATE_LIMIT_PER_MIN` | Per-IP writes/min on `/score` + `/decision` (`0` disables) | `120` |

## Run and build commands
```bash
uvicorn api.main:app --reload --port 8000
# analyst console (no build step)
open http://localhost:8000/            # queue + case detail + scoring playground
# score one txn
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d \
 '{"sender_id":"C000001","receiver_id":"C000002","amount":45000,"channel":"app","device_id":"DX999","location":"Dhaka","timestamp":"2026-08-15T23:10:00","type":"P2P","lang":"bn"}'
# queue / case / feedback
curl 'localhost:8000/alerts?limit=5'; curl localhost:8000/case/T0000100
curl -X POST localhost:8000/decision -H 'Content-Type: application/json' -d '{"txn_id":"T0000100","decision":"step-up"}'
python -m eval.evaluate --data data --artifacts artifacts
python -m scripts.run_demo              # 4k-txn end-to-end: generate → train → normal vs scam
```

## Live deployment URL
`TBD — deploy API to Render/Railway and put URL here before T+72h` (judges require a live link; local fallback: follow Run commands + video).

## Testing instructions
```bash
pytest -q                                   # 43 tests: smoke + API + security + LLM (mocked) + intensive units (needs data/ + artifacts/)
python -m eval.evaluate --sample 20000      # offline metrics + fairness + business sim
# API verify: /health -> {"ok": true}; /score latency_ms should be <200 p95 locally
# Frontend verify: GET / -> 200 text/html; queue + case + playground in browser
```

## Other configuration
- `config/thresholds.yaml` — change bands/weights/graph thresholds on-site in <30 min, no ML retrain. Bound cold start with `VIGIL_ALERTS_LIMIT=100`.
- `api/llm.py: TEMPLATE_EN/BN` — prompt lives outside decision logic; toggle `lang: en|bn`.
- `data/` + `artifacts/` are regenerable and git-ignored. Clean test split: last 20% by timestamp, never trained on. `artifacts/anomaly_calib.npy` is the train-only anomaly calibration (regenerated on retrain).
- Synthetic-data assumptions documented in `data_gen/generate.py` header, including hardened noise (legit new-device/night/round-amount/reset/fan-in + fraud overlap). All amounts in BDT (৳).
- Ops: `GET /health` (queue/startup info), `GET /metrics` (requests, decisions, startup). Structured logs via stdlib `logging`.
- Security: open CORS is demo-only (see comment in `api/main.py`); restrict before prod. See `docs/security.md`.

## Docs
- `docs/logic-chain.md` — 9-step product logic + problem statement
- `docs/data-dictionary.md` — tables, patterns, features
- `docs/scale-plan.md` — readiness checklist + integration path + business math
- `docs/security.md` — privacy, explainability, fairness, prompt-injection, oversight
- `docs/onsite-runbook.md` — final-day playbook (triage → commit → demo)
- `docs/eval-sample.json` — reference 50k eval output (model vs baseline + fairness + business)
- `docs/paysim-validation.md` + `docs/paysim-eval.json` — independent check on foreign MFS data

## Project structure
```
/data_gen    synthetic customers/devices/transactions (hardened, overlapping)
/features    causal batch feature layer (same logic as online store)
/models      train.py (clf+anomaly+calibration), graph.py (2-hop), infer.py (ensemble+batch)
/api         FastAPI: main/store/rules/llm/schemas (+ static /web mount, /metrics)
/web         analyst console: index.html/app.js/styles.css + favicon (no build step)
/eval        metrics vs baseline + fairness + business sim (batched)
/docs        logic-chain, data-dictionary, scale-plan, security, runbook, eval-sample
/config      thresholds.yaml (on-site tunable)
/tests /scripts
```
