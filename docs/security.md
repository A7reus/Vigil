# Responsible AI & security (guideline Sec 14)

## Privacy
Synthetic-only during hackathon (`data_gen/`); no production data, no PII.
`data/` + `artifacts/` git-ignored; secrets via env (`LLM_API_KEY`), placeholders in README.

## Explainability
Every score ships `top_3_reasons` (auditable rules first, then model importance)
plus the LLM narrative grounded in structured evidence only
(`api/llm.py` system prompt: use ONLY provided JSON, fixed template).
Groundedness is enforced, not just instructed: every number and ID in the
response must appear in the evidence or the template fallback is served.
Predictions, assumptions, and generated text are separate fields.

## Fairness
`eval.evaluate` reports FPR by `district`, `age_group`, and
`account_age_bucket`. Note the tension we disclose rather than hide:
`account_age_days` is both a model input and a fairness slice, so a gap
there could mean the model leans on account age. Current synthetic gaps are
small but non-zero (see `eval-sample.json`).
Mitigation path: per-segment threshold review in `config/thresholds.yaml`
before any enforcement; monitor-first in shadow mode; never auto-tune on
unreviewed labels.

## Security
- CORS `*` is demo-only (see code comment in `api/main.py`); restrict to the
  deployed frontend domain for anything beyond the hackathon.
- Auth: password login (PBKDF2) with opaque revocable bearer tokens;
  registration is pending until admin approval; roles analyst/admin enforced
  server-side on every privileged route. Anonymous scoring stays open for
  judges, compensated by (a) LIVE isolation (scoring-neutral buffer), (b)
  per-IP rate limiting (`VIGIL_RATE_LIMIT_PER_MIN`, 429 JSON), (c) strict
  input bounds. Service API keys are a pilot-backlog item.
- Ground truth: `/alerts` and `/case` never expose training labels.
- Decisions: unknown `txn_id` → 404; same analyst+txn upserts instead of
  duplicating; notes capped at 500 chars.
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
