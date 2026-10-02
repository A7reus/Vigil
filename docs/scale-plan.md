# From prototype to product (guideline Sec 13)

## Readiness checklist (current state)
- [x] Frequent, economically meaningful problem (scam loss + analyst hours)
- [x] AI beyond deterministic rules (ensemble beats rule baseline 0.99 vs 0.21 R@5%FPR)
- [x] Clear action after prediction (allow / review / step-up+hold, human-reviewed)
- [x] Measurable benefit (`eval.business`: loss prevented BDT + minutes saved per 1000 holds)
- [~] Validatable with real data (same API contract; needs governed labels — future)
- [x] Privacy/fairness/explainability addressed (see `security.md`)
- [~] Integratable (stateless FastAPI + `/metrics`; needs auth, queue,Idempotency for prod)

## Post-hackathon pathway
1. Competition → prototype + `docs/eval-sample.json` + console
2. Technical review → model quality, `anomaly_calib.npy` drift checks, load test
3. Business review → hold precision vs analyst capacity, threshold economics
4. Controlled validation → governed anonymized stream through same `/score` schema
5. POC → shadow-score live traffic, compare with analyst labels from `/decision`
6. Pilot → enforce step-up+hold on High, measure prevented loss + false-hold rate
7. Decision → integrate / incubate / close

## Integration sketch
Real stream → feature store (same 26 cols, past-only windows) → `POST /score`
→ case queue UI → analyst `/decision` → label store → weekly
`python -m models.train` → versioned `artifacts/` + `metrics.json` gate.
Thresholds stay in `config/thresholds.yaml` so ops tunes without ML deploys.

## Business assumptions (explicit)
`eval.business_sim`: top-1000 holds, 2 min/case with narrative vs 15 min manual
(13 min saved each). Replace 2/15 with measured times during pilot; multiply by
loaded analyst hourly cost for ROI. False holds cost customer friction — track
appeal rate alongside precision.
