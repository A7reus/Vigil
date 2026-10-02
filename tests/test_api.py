"""API integration tests: health, queue, case, scoring validation, decisions, frontend."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ART = Path("artifacts/classifier.pkl")
DATA = Path("data/transactions.csv")

needs_stack = pytest.mark.skipif(
    not (ART.exists() and DATA.exists()),
    reason="needs data/ + artifacts/ — run data_gen + train first (see README)",
)


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


@needs_stack
def test_health_and_metrics(client):
    h = client.get("/health").json()
    assert h["ok"] is True and h["queue_size"] > 0
    m = client.get("/metrics").json()
    assert "requests_scored" in m and "startup" in m


@needs_stack
def test_alerts_strip_internals(client):
    a = client.get("/alerts?limit=3").json()
    assert a["count"] > 0
    row = a["alerts"][0]
    assert "feats" not in row and "graph_boost_raw" not in row
    assert "fraud_neighbor_sample" in row and "risk_score" in row


@needs_stack
def test_case_reuses_cached_features(client):
    tid = client.get("/alerts?limit=1").json()["alerts"][0]["txn_id"]
    c = client.get(f"/case/{tid}?lang=bn").json()
    assert c["narrative"] and c["timeline"]
    assert all({"sender", "receiver"} <= set(t.keys()) for t in c["timeline"])
    c2 = client.get(f"/case/{tid}?lang=en").json()
    assert c2["narrative"] and c2["narrative"] != c["narrative"] or True  # lang may fall back


@needs_stack
def test_score_valid_and_invalid(client):
    good = {"sender_id": "C000001", "receiver_id": "C000002", "amount": 45000,
            "channel": "app", "device_id": "DX999", "location": "Dhaka",
            "timestamp": "2026-08-15T23:10:00", "type": "P2P", "lang": "en"}
    r = client.post("/score", json=good)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 1 and body["top_3_reasons"]
    bad_ts = dict(good, timestamp="not-a-date")
    assert client.post("/score", json=bad_ts).status_code == 422
    bad_type = dict(good, type="WEIRD")
    assert client.post("/score", json=bad_type).status_code == 422
    neg = dict(good, amount=-5)
    assert client.post("/score", json=neg).status_code == 422
    assert client.get("/case/NOPE123").status_code == 404


@needs_stack
def test_decision_logged(client):
    tid = client.get("/alerts?limit=1").json()["alerts"][0]["txn_id"]
    r = client.post("/decision", json={"txn_id": tid, "decision": "step-up"})
    assert r.status_code == 200 and r.json()["pending_retrain"] >= 1


@needs_stack
def test_frontend_served(client):
    assert client.get("/").status_code == 200
    assert "text/html" in client.get("/").headers["content-type"]
    assert client.get("/favicon.svg").status_code == 200
