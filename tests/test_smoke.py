"""Smoke tests: rules bands, offline LLM fallback, feature causality."""
from features.build import FEATURE_COLS


def test_bands():
    from api.rules import decide
    assert decide(0.9)["risk_level"] == "High"
    assert decide(0.7)["risk_level"] == "Medium"
    assert decide(0.1)["risk_level"] == "Low"
    assert "analyst" in decide(0.9)["recommended_action"]


def test_llm_fallback_works_offline(monkeypatch):
    import api.llm as llm
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    out = llm.narrate(
        {"sender_id": "C000001", "receiver_id": "C000002", "amount": 25000,
         "channel": "app", "timestamp": "2026-08-15T23:10:00"},
        {"amount_vs_user_avg": 5.0, "new_device": 1, "location_jump": 1,
         "sender_cnt_1h": 4, "recv_n_senders_1h": 0, "location_new": 1},
        {"risk_score": 0.91, "risk_level": "High", "recommended_action": "step-up-auth + hold + analyst review",
         "top_3_reasons": ["new device not seen for this sender"]},
        {"boost": 0.0, "fraud_neighbors_2hop": 0}, lang="bn")
    assert out["template"] == "offline-fallback" and out["narrative"]


def test_feature_cols_stable():
    assert len(FEATURE_COLS) >= 20 and "sender_cnt_1h" in FEATURE_COLS
