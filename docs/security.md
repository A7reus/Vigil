# Responsible AI & security (guideline Sec 14)

## Privacy
Synthetic-only during hackathon (`data_gen/`); no production data, no PII.
`data/` + `artifacts/` git-ignored; secrets via env (`LLM_API_KEY`), placeholders in README.

## Explainability
Every score ships `top_3_reasons` (auditable rules first, then model importance)
plus the LLM narrative grounded in structured evidence only
(`api/llm.py` system prompt: use ONLY provided JSON, fixed template).
Predictions, assumptions, and generated text are separate fields.

## Fairness
`eval.evaluate` reports FPR by `district` and `account_age_bucket`.
Current synthetic gaps are small but non-zero (see `eval-sample.json`).
Mitigation path: per-segment threshold review in `config/thresholds.yaml`
before any enforcement; never auto-tune on unreviewed labels.

## Security
- CORS `*` is demo-only (see code comment in `api/main.py`); restrict to the
  deployed frontend domain for anything beyond the hackathon.
- No auth/rate-limit: acceptable single-tenant demo; add API key + throttle
  before pilot.
- Prompt injection: sender/receiver/device IDs are attacker-influenceable and
  flow into the LLM evidence JSON. The investigator is instructed to use values
  verbatim within the template and never invent actions; high-impact actions
  still require analyst confirmation (`step-up + hold + review`, never auto-block).
- Adversarial: exact-amount rules are dodged by near-round mule amounts in the
  generator; the model uses behavioral + graph signals, not amount alone.

## Human oversight
Bands in `config/thresholds.yaml`: High → hold + step-up + analyst review,
Medium → review, Low → allow. Money is never auto-blocked.
