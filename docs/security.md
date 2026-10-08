# Responsible AI & security (guideline Sec 14)

## Privacy
Synthetic-only during hackathon (`data_gen/`); no production data, no PII.
`data/` + `artifacts/` git-ignored; the analyst secret (if set) via env (`VIGIL_API_KEY`), placeholders in README.

Case evidence never leaves the building. Narratives are written by a local
model through Ollama (`OLLAMA_HOST` defaults to localhost; there is no remote
endpoint to configure and no key to leak). This is deliberate: MFS case data
cannot cross borders, so the design removes the crossing instead of guarding
it. Daemon down just means template narratives — the offline fallback judges
already saw, now as the everyday path rather than the exception.

## Explainability
Every score ships `top_3_reasons` (auditable rules first, then model importance)
plus the LLM narrative grounded in structured evidence only
(`api/llm.py` system prompt: use ONLY the provided values, fixed template,
no JSON in the output).
Predictions, assumptions, and generated text are separate fields.

## Fairness
`eval.evaluate` reports FPR by `district` and `account_age_bucket`.
Current synthetic gaps are small but non-zero (see `eval-sample.json`).
Mitigation path: per-segment threshold review in `config/thresholds.yaml`
before any enforcement; never auto-tune on unreviewed labels.

## Security
- CORS `*` is demo-only (see code comment in `api/main.py`); restrict to the
  deployed frontend domain for anything beyond the hackathon.
- Auth, two layers: (a) signed-in analysts/admins (PBKDF2 passwords, opaque
  `vigil_*` bearer tokens with expiry, roles analyst/admin, registration
  pending until approved; demo seeds `admin/admin123` + `analyst/analyst123`
  only on empty user tables unless `VIGIL_SEED_DEMO=0`); (b) the service key
  `VIGIL_API_KEY` (`X-API-Key` on writes) for scripts. Writes accept token
  or key; a signed-in analyst decides as themselves (no impersonation);
  admin routes (`/admin/*`, retraining) need an admin token — the service
  key is not admin. Reads stay open. Compensated by (a) LIVE isolation, which keeps
  unreviewed `/score` traffic is scoring-neutral (separate capped buffer,
  excluded from features/seen-sets/graph/history eviction), (b) per-IP rate
  limiting on `/score` + `/decision` + login (`VIGIL_RATE_LIMIT_PER_MIN`, default 120,
  0 disables; 429 JSON), (c) strict input bounds (lengths, timestamp range,
  finite JSON enforced by middleware). The console sends no key, so it pairs
  with the open default; keyed deployments use service clients.
- Ground truth: `/alerts` and `/case` never expose training labels.
- Decisions live in Postgres (`DATABASE_URL`, required — boot fails without
  it): unknown `txn_id` → 404; same analyst+txn upserts instead of
  duplicating; notes capped at 500 chars; the log survives sleeps, restarts,
  and redeploys, and feeds the retrain queue (`pending_retrain`). Local dev
  uses a throwaway container, CI a service, hosted Render the managed
  database — same contract everywhere.
- Prompt injection: raw fields are sanitized (`_safe()` strips control chars,
  caps at 120) before prompts/narratives, and the system instruction treats
  evidence values as untrusted data. High-impact actions still require analyst
  confirmation (`step-up + hold + review`, never auto-block).
- Frontend: every server value is HTML-escaped at render (`esc()` in
  `web/app.js`); narratives use `textContent`.
- Adversarial: exact-amount rules are dodged by near-round mule amounts in the
  generator; the model uses behavioral + graph signals, not amount alone.

## Human oversight
Bands in `config/thresholds.yaml`: High → hold + step-up + analyst review,
Medium → review, Low → allow. Money is never auto-blocked.
