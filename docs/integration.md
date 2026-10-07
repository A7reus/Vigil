# Integration guide, runbook, and threat model (trimmed item 17)

## Auth flows
```bash
# 1. register (always pending until approved)
curl -X POST localhost:8000/auth/register -H 'Content-Type: application/json' \
  -d '{"username":"ops1","password":"a-strong-password"}'
# 2. admin approves (login as admin/admin123 first in demo)
TOKEN=$(curl -s -X POST localhost:8000/auth/login -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
UID=$(curl -s localhost:8000/admin/users -H "Authorization: Bearer $TOKEN" | python3 -c "import json,sys; print([u['id'] for u in json.load(sys.stdin)['users'] if u['username']=='ops1'][0])")
curl -X POST localhost:8000/admin/users/$UID/approve -H "Authorization: Bearer $TOKEN"
# 3. use the token: Authorization: Bearer $TOKEN on writes
```
Demo credentials (seeded into an empty DB; set `VIGIL_SEED_DEMO=0` in pilot):
`admin/admin123` (admin), `analyst/analyst123` (analyst).

## Scoring with attribution, idempotency, and committed ingestion
```bash
# authenticated score that joins committed history (velocity/seen-sets learn it)
curl -X POST 'localhost:8000/score?commit=true' -H "Authorization: Bearer $TOKEN" \
  -H 'Idempotency-Key: client-txn-9f3a' -H 'Content-Type: application/json' -d '{...}'
# replaying the same Idempotency-Key returns the stored response + "deduplicated": true
```

## SQLite schema contract (shared by queue, decisions, identity)
`api/db.py` owns it; teammate P0 work aligns here. Tables:
`users(id, username UNIQUE, pw_hash, role, status, created_at)`,
`sessions(token_sha PK, user_id, expires_at, created_at)`,
`cases(txn_id PK, payload, risk_score, risk_level, status, analyst, source,
created_at, updated_at)` with `status ∈ open|assigned|closed`,
`decisions(id, txn_id, analyst, decision, note, at, model_version,
UNIQUE(txn_id, analyst))`, `idempotency(key PK, response, created_at)`.
Back up `vigil.db` before upgrades; decisions carry `model_version` so every
review is attributable to an exact artifact set.

## Runbook
- **Restart**: `uvicorn api.main:app --workers 1` (single worker is
  load-bearing: rate buckets, live_cases and ops counters are per-process).
  `/ready` must return 200 before routing traffic; `/health` is liveness only.
- **Rollback**: previous `artifacts/` + `config/thresholds.yaml` are in git;
  `model_version` in every score/decision tells you what ran. Restore files,
  restart, confirm `/health`.
- **Degraded mode**: a failed component returns Medium/review with
  `"degraded": true`, never silent allow. Watch `vigil_errors_total` and
  the `llm fallback` counter in `/metrics/prom`.
- **What an alert means**: `risk_score` + `top_3_reasons` are the contract;
  the narrative is explanatory only. High = hold + step-up + human review.

## Threat model (sketch)
- Anonymous scoring abuse → rate limits, input bounds, LIVE traffic isolated
  from scoring state; `commit=true` requires auth.
- Credential theft → PBKDF2 hashes, opaque revocable tokens, logout revokes.
- Privilege creep → admin-only user/case management; analysts self-assign
  only; close requires assignee-or-admin; self-disable blocked.
- Model tampering → sha256 manifest verified before unpickle, fail closed.
- Data leakage → no training labels in responses; wallet IDs never logged;
  decisions attributable by design (audit, not secret).
- Prompt injection → evidence sanitized, template-fixed, faithfulness
  enforced with fallback.

## Compliance note
Check the client's local data-protection rules and central-bank
requirements before making compliance claims. The system is built to help:
synthetic-first development, PII-free logs, attributable decisions,
per-segment fairness reporting, and human oversight on every consequential
action — but none of that is a certification.
