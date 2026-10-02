# ScamShield — Trust & Risk Intelligence for upay (Track 01)

> For upay users losing money to scams/ATO/mules and analysts drowning in opaque alerts, slow manual review causes loss + churn. We build a real-time risk scorer + graph + LLM investigator on synthetic transactions to score, explain, and recommend action — measured by Precision@100 + investigation time saved.

Answers the Track 01 test: **What happened? Why is it risky? What should upay do next?**

## Features
- **Real-time scoring API** — `POST /score` in → `risk_score 0-1, risk_level, top_3_reasons, recommended_action` out. Ensemble `0.7·classifier + 0.2·anomaly + 0.1·graph`. Bands in `config/thresholds.yaml`: `>0.85` hold+step-up+review, `0.6–0.85` review, `<0.6` allow. Never auto-blocks money.
- **AI engine (4x)** — XGBoost (HGB fallback) classifier + IsolationForest anomaly + NetworkX 2-hop mule boost + grounded LLM investigator (EN/BN, offline fallback).
- **Analyst queue** — `GET /alerts` (pre-scored, risk-sorted), `GET /case/:id` (timeline + narrative), `POST /decision` (feedback loop for retrain).
- **Evaluation** — `python -m eval.evaluate`: Precision@100, Recall@5%FPR, AUC vs rule baseline, p95 latency, fairness FPR by district/account-age, business simulation (loss prevented, analyst-minutes saved).

No UI in this repo by team choice — API-only prototype (UI was cut to hit backend depth in 72h).

## Technology stack
Python 3.10+, Pandas, NumPy, Scikit-learn, NetworkX, FastAPI/Uvicorn, PyYAML, Joblib. Optional: XGBoost, SHAP, LightGBM (graceful fallback if absent). LLM: any OpenAI-compatible API (Groq default) with deterministic offline template fallback.

## Requirements
- Python 3.10+ with venv
- 2 GB RAM, no GPU needed
- Optional `LLM_API_KEY` for live narratives (works offline without it)

## Installation and setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: pip install xgboost shap
python -m data_gen.generate --n-customers 5000 --n-txns 50000 --out data
python -m models.train --data data --artifacts artifacts
```

## Environment variables
| Name | Purpose | Example |
|---|---|---|
| `LLM_API_KEY` | Live investigator narratives (leave unset for offline fallback) | `gsk_...` (placeholder — never commit secrets) |
| `LLM_BASE_URL` | OpenAI-compatible endpoint | `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | Chat model | `llama-3.1-8b-instant` |

## Run and build commands
```bash
uvicorn api.main:app --reload --port 8000
# score one txn
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d \
 '{"sender_id":"C000001","receiver_id":"C000002","amount":45000,"channel":"app","device_id":"DX999","location":"Dhaka","timestamp":"2026-08-15T23:10:00","type":"P2P","lang":"bn"}'
# queue / case / feedback
curl 'localhost:8000/alerts?limit=5'; curl localhost:8000/case/T0000100
curl -X POST localhost:8000/decision -H 'Content-Type: application/json' -d '{"txn_id":"T0000100","decision":"step-up"}'
python -m eval.evaluate --data data --artifacts artifacts
```

## Live deployment URL
`TBD — deploy API to Render/Railway and put URL here before T+72h` (judges require a live link; local fallback: follow Run commands + video).

## Testing instructions
```bash
pip install pytest httpx
pytest -q                                   # rules/LLM-fallback/feature checks
python -m eval.evaluate --sample 20000      # offline metrics + fairness + business sim
# API verify: /health -> {"ok": true}; /score latency_ms should be <200 p95 locally
```

## Other configuration
- `config/thresholds.yaml` — change bands/weights/graph thresholds on-site in <30 min, no ML retrain.
- `api/llm.py: TEMPLATE_EN/BN` — prompt lives outside decision logic; toggle `lang: en|bn`.
- `data/` + `artifacts/` are regenerable and git-ignored. Clean test split: last 20% by timestamp, never trained on.
- Synthetic-data assumptions documented in `data_gen/generate.py` header. All amounts in BDT (৳).

## Project structure
```
/data_gen    synthetic customers/devices/transactions
/features    causal batch feature layer (same logic as online store)
/models      train.py (clf+anomaly), graph.py (2-hop), infer.py (ensemble+reasons)
/api         FastAPI: main/store/rules/llm/schemas
/eval        metrics vs baseline + fairness + business sim
/config      thresholds.yaml (on-site tunable)
/tests /scripts
```
