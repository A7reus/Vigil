# Vigil: Trust and Risk Intelligence for upay (Track 01)

> Upay users lose money to scams, takeovers, and mule rings while analysts drown in alerts they cannot explain. Slow manual review means losses mount and users leave. Vigil is a real-time risk scorer with a transaction graph and an LLM investigator, built on synthetic transactions to score each transfer, explain why it looks risky, and recommend what to do next. Success is measured by Precision@100 and investigation time saved.

Answers the Track 01 test: **What happened? Why is it risky? What should upay do next?**

## Features
- **Real-time scoring API**: `POST /score` takes a transfer and returns `risk_score` (0 to 1), `risk_level`, `top_3_reasons`, and `recommended_action`. The ensemble blends `0.7·classifier + 0.2·anomaly + 0.1·graph`. Bands in `config/thresholds.yaml`: above `0.85` means hold plus step-up plus review, `0.6–0.85` means review, below `0.6` means allow. Money is never blocked automatically.
- **AI engine (4 parts)**: XGBoost classifier (HGB fallback) plus IsolationForest anomaly scores calibrated to percentiles on training data with no test leakage, plus a NetworkX 2-hop mule boost, plus a grounded LLM investigator (English/Bangla, offline fallback). Reasons combine auditable rules with per-row SHAP attributions.
- **Analyst queue**: `GET /alerts` (pre-scored, risk-sorted), `GET /case/:id` (timeline plus narrative plus workflow state, reusing the exact cached causal features), `POST /decision` (feedback loop for retraining, attributed to the signed-in analyst), `POST /cases/:id/assign` + `/status` (triage workflow), `GET /cases` (board).
- **Accounts + admin**: `POST /auth/register|login`, `GET /auth/me`, `POST /auth/logout` (PBKDF2 passwords, opaque bearer tokens, analyst/admin roles, pending-until-approved); `GET /admin/users|decisions|cases`, approve/disable/enable; `POST /admin/retrain` runs `scripts/retrain.py` in the background with a SHIP/HOLD verdict (`GET /admin/retrain/status`), applying winners over the live model. Writes accept a token or the `VIGIL_API_KEY` service key; guests keep the open demo.
- **Analyst console**: a static frontend in `/web` with no build step, served at `GET /`: a risk queue with level filter and search, case detail with English/Bangla narrative plus timeline plus decision buttons, a `POST /score` playground, sign-in with guest mode, and an admin dashboard (approvals, users, reviews, one-click retraining).
- **Evaluation**: `python -m eval.evaluate` reports Precision@100, Recall@5%FPR, AUC against a rule baseline, p95 latency, fairness (FPR by district and account age), and a business simulation (loss prevented, analyst minutes saved). Batched scoring handles 8k rows in about 2s, down from about 100s.
- **Cross-dataset check**: the same pipeline retrained on PaySim mobile money data (`eval/paysim_adapter.py`, offline) reaches AUC 0.90 against 0.50 for rules — and **frozen weights transfer**: zero-shot scoring with no refit (`eval/zeroshot.py`) reaches AUC 0.88 on PaySim (portable 22-col weights) and 0.98+ on an unseen same-schema feed, against 0.70 for full weights leaning on signals PaySim lacks. Operating bands transfer on same-schema data; PaySim's shift needs per-deployment recalibration, as documented. See `docs/paysim-validation.md`.

## Technology stack
Python 3.12+, Pandas, NumPy, Scikit-learn, NetworkX, FastAPI/Uvicorn, PyYAML, Joblib, psycopg (Postgres audit log + accounts).
ML: XGBoost primary classifier (HGB fallback if absent), per-row SHAP attributions
(lazy, ~1.4ms, falls back to global importance), IsolationForest anomaly
(train-calibrated percentile), NetworkX 2-hop mule boost. Narratives: a small
local model via Ollama (`OLLAMA_MODEL`, default `qwen2.5:3b`) with
deterministic offline template fallback — case evidence never leaves the box.

## Requirements
- Python 3.12+ with venv (pandas 3 requires it)
- 2 GB RAM, no GPU needed (CPU-only; `nvidia-nccl` wheels ship with XGBoost but are unused)
- Optional local narratives: `ollama pull qwen2.5:3b` + `ollama serve` (works offline without it, via templates)

## Installation and setup
```bash
python3 -m venv ~/.venvs/vigil && source ~/.venvs/vigil/bin/activate
pip install -r requirements.txt
docker run -d -e POSTGRES_PASSWORD=vigil -e POSTGRES_USER=vigil -e POSTGRES_DB=vigil -p 5432:5432 postgres:16-alpine
export DATABASE_URL=postgresql://vigil:vigil@localhost:5432/vigil   # required — the API refuses to boot without it
python -m data_gen.generate --n-customers 5000 --n-txns 50000 --out data
python -m models.train --data data --artifacts artifacts
```

## Environment variables
Copy `.env.example` to `.env` (git-ignored, loaded on startup); plain
environment variables take precedence over the file.

| Name | Purpose | Example |
|---|---|---|
| `OLLAMA_HOST` | Local Ollama daemon (never a cloud URL — case data stays in-country) | `http://localhost:11434` |
| `OLLAMA_MODEL` | Small local instruction model with usable Bangla | `qwen2.5:3b` |
| `VIGIL_ALERTS_LIMIT` | Pre-scored queue size at startup (bounds cold start) | `200` |
| `VIGIL_API_KEY` | Shared secret for writes (empty = open demo; set = `X-API-Key` required; signed-in tokens also accepted) | `` (empty) |
| `VIGIL_SEED_DEMO` | Seed `admin/admin123` + `analyst/analyst123` on empty user tables (`0` disables — set it anywhere real) | `1` |
| `VIGIL_TOKEN_DAYS` | Bearer-token lifetime for signed-in analysts | `30` |
| `DATABASE_URL` | Postgres for the decision audit log — required, API refuses to boot without it | `postgresql://vigil:vigil@localhost:5432/vigil` |
| `VIGIL_ALERT_WINDOW` | Recent rows scored to fill the queue; top risks kept | `2000` |
| `VIGIL_RATE_LIMIT_PER_MIN` | Per-IP writes/min on `/score` + `/decision` (`0` disables) | `120` |

## Run and build commands
```bash
export DATABASE_URL=postgresql://vigil:vigil@localhost:5432/vigil   # every command below needs it
export TEST_POSTGRES_URL=$DATABASE_URL                              # PG-gated tests (auth, decisions, retrain)
uvicorn api.main:app --reload --port 8000
# analyst console (no build step)
open http://localhost:8000/            # queue + case detail + scoring playground (+ sign-in, admin dashboard)
# score one txn
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d \
 '{"sender_id":"C000001","receiver_id":"C000002","amount":45000,"channel":"app","device_id":"DX999","location":"Dhaka","timestamp":"2026-08-15T23:10:00","type":"P2P","lang":"bn"}'
# queue / case / feedback (sign in first for attributed decisions: POST /auth/login as admin/admin123)
curl 'localhost:8000/alerts?limit=5'; curl localhost:8000/case/T0000100
curl -X POST localhost:8000/decision -H 'Content-Type: application/json' -d '{"txn_id":"T0000100","decision":"step-up"}'
python -m eval.evaluate --data data --artifacts artifacts
python -m scripts.retrain --data data --artifacts artifacts --out-data /tmp/aug --out-artifacts /tmp/art2
python -m scripts.run_demo              # 4k-txn end-to-end: generate → train → normal vs scam
```

## Live deployment URL
`https://vigil-qna5.onrender.com/` (API plus analyst console at `/`; interactive docs at `/docs`). Free-tier hosting sleeps when idle, so the first visit after a pause takes about a minute to wake. Local fallback: follow Run commands + video.

## Testing instructions
```bash
pytest -q                                   # 66 tests: smoke, API, auth (PG-gated), security, LLM (mocked), intensive, paysim, zeroshot, decisions, retrain (all need data/ + artifacts/; PG tests need TEST_POSTGRES_URL; 3 live-model tests need VIGIL_LIVE_LLM_TEST=1 + local ollama)
python -m eval.evaluate --sample 20000      # offline metrics + fairness + business sim
# API verify: /health -> {"ok": true}; /score latency_ms should be <200 p95 locally
# Frontend verify: GET / -> 200 text/html; queue + case + playground (+ sign-in, admin) in browser
```

## Other configuration
- `config/thresholds.yaml`: change bands, weights, and graph thresholds on-site in under 30 min with no ML retrain. Bound cold start with `VIGIL_ALERTS_LIMIT=100`.
- `api/llm.py: TEMPLATE_EN/BN`: the prompt lives outside decision logic; toggle with `lang: en|bn`.
- `data/` + `artifacts/` are regenerable and git-ignored. Clean test split: last 20% by timestamp, never trained on. `artifacts/anomaly_calib.npy` is the train-only anomaly calibration (regenerated on retrain).
- Synthetic-data assumptions documented in `data_gen/generate.py` header, including hardened noise (legit new-device/night/round-amount/reset/fan-in + fraud overlap). All amounts in BDT (৳).
- Ops: `GET /health` (queue/startup info), `GET /metrics` (requests, decisions, startup). Structured logs via stdlib `logging`.
- Security: open CORS is demo-only (see comment in `api/main.py`); restrict before prod. See `docs/security.md`.

## Docs
- `docs/logic-chain.md`: 9-step product logic plus problem statement
- `docs/data-dictionary.md`: tables, patterns, features
- `docs/scale-plan.md`: readiness checklist plus integration path plus business math
- `docs/security.md`: privacy, explainability, fairness, prompt injection, oversight
- `docs/onsite-runbook.md`: final-day playbook (triage, then commit, then demo)
- `docs/eval-sample.json`: reference 50k eval output (model vs baseline plus fairness plus business)
- `docs/paysim-validation.md` + `docs/paysim-eval.json`: independent check on foreign MFS data
- `docs/zeroshot-*.json`: frozen-model transfer evidence (PaySim full/intersect, shifted-seed full/intersect)
- `scripts/retrain.py`: analyst decisions → weak labels → retrained artifacts + SHIP/HOLD verdict (`POST /admin/retrain` runs it from the dashboard)

## Project structure
Output of `tree -I '.git|__pycache__|data|artifacts|report.*'` (generated
`data/`, `artifacts/`, and TeX build files omitted):
```
.
├── api                  FastAPI service: main (routes/mounts), auth (PBKDF2 users, bearer tokens, RBAC, case workflow), decisions (Postgres audit log), store, rules, llm, schemas
├── config               thresholds.yaml, tunable on-site with no ML retrain
├── data_gen             synthetic customers/devices/transactions (hardened, overlapping)
├── docs                 logic-chain, data-dictionary, scale-plan, security, runbook,
│                        eval-sample, paysim-validation (plus their JSON evidence),
│                        zeroshot transfer evidence (PaySim + shifted-seed JSONs)
├── eval                 evaluate.py (metrics vs baseline, fairness, business sim),
│                        paysim_adapter.py (offline cross-dataset check),
│                        zeroshot.py (frozen-model transfer, full + intersect weights)
├── features             causal batch feature layer (same logic as online store)
├── models               train.py (classifier plus anomaly plus calibration, full/intersect sets),
│                        graph.py (2-hop mule boost), infer.py (ensemble plus batch)
├── presentation         slides.pptx (hackathon slide deck)
├── report               report.tex plus compiled report.pdf (project report)
├── scripts              run_demo.py: 4k-transaction end-to-end (generate, train, score two cases);
│                        retrain.py: decisions → weak labels → retrained artifacts + verdict;
│                        load_test.py: stdlib throughput probe for POST /score
├── tests                test_smoke (rules, fallback, features), test_api (endpoints incl. HEAD),
│                        test_auth (register/approve/login, roles, workflow, retrain gating, PG-gated),
│                        test_security (fuzz regressions), test_llm (mocked investigator),
│                        test_llm_live (local Ollama, gated), test_intensive (eval math,
│                        graph, causality), test_paysim (adapter contract),
│                        test_zeroshot (frozen transfer: intersect contract + shifted-seed e2e),
│                        test_decisions (audit-log contract, PG-gated),
│                        test_retrain (feedback loop end-to-end, PG-gated)
├── web                  analyst console: index.html, app.js, styles.css, favicon (no build step)
├── .env.example         copy to .env; all runtime variables with placeholders
├── .python-version      pins Python 3.14 for hosts like Render
├── render.yaml          one-click Render blueprint (build, train, serve)
├── requirements.txt     exact pinned dependencies including XGBoost, SHAP, LightGBM, psycopg
└── README.md            this file
```
