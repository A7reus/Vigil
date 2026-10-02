# Vigil — one-page logic chain (guideline Sec 10)

| Step | Answer |
|---|---|
| 1. User | upay fraud analyst triaging scam/ATO/mule alerts; secondarily the upay sender about to pay a scammer |
| 2. Problem | Opaque risk scores + manual graph/timeline reconstruction → slow review, missed mule rings, user money lost + churn |
| 3. Why now | Synthetic behavioral + graph + grounded-LLM patterns are strong enough to prototype; API + batch features make it demoable in 72h |
| 4. Solution | Real-time `/score` ensemble + risk-sorted `/alerts` queue + `/case` with grounded EN/BN narrative + timeline + mule-ring sample + `/decision` feedback |
| 5. AI role | Prediction (XGBoost P(fraud)), detection (IsolationForest anomaly, NetworkX 2-hop boost), per-row SHAP explanations, generation (LLM investigator grounded in structured evidence only) |
| 6. Impact | Precision@100 and Recall@5%FPR vs rule baseline; p95 latency <200ms; business sim: loss prevented (BDT) + analyst-minutes saved per 1000 holds |
| 7. Data | Fully synthetic (`data_gen/generate.py`): customers, devices, transactions with scam/ATO/mule/agent patterns + realistic noise (legit new-device/night/round-amount/reset/fan-in, fraud overlap). Chronological 80/20 split, test never trained on. No PII, amounts in BDT |
| 8. Validation | Offline: `python -m eval.evaluate` (AUC, P@100, R@5%FPR, fairness FPR by district/account-age, business sim). Online: analyst decision log → pending-retrain counter; on-site: new requirement integrated via `config/thresholds.yaml` without ML retrain |
| 9. Scale | Same FastAPI contract against real stream; weekly retrain on governed labels; thresholds per segment if fairness gaps appear; see `scale-plan.md` |

**Problem statement (template filled):**
For upay risk analysts facing opaque fraud alerts, slow manual investigation causes delayed holds and lost funds.
We build ScamShield, an AI investigation console that uses synthetic behavioral + graph features to score,
explain, and recommend action, with success measured by Precision@100 ≥ 0.6 at p95 <200ms and positive analyst time saved.
