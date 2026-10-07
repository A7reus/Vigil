"""Regression tests for the external fuzz report (~190 payloads, 10 findings).

Covers: tz/out-of-range timestamps, non-finite JSON, input length bounds,
LIVE/history isolation, label leakage, decision upsert, LIVE case resolution,
limit bounds, LLM sanitization, ATO flag reachability, rate limiting.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

needs_stack = pytest.mark.skipif(
    not (Path("artifacts/classifier.pkl").exists() and Path("data/transactions.csv").exists()),
    reason="needs data/ + artifacts/ — run data_gen + train first (see README)",
)


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def _score_body(**kw):
    base = {"sender_id": "C000001", "receiver_id": "C000002", "amount": 45000,
            "channel": "app", "device_id": "DX999", "location": "Dhaka",
            "timestamp": "2026-08-15T23:10:00", "type": "P2P", "lang": "en"}
    base.update(kw)
    return base


@needs_stack
def test_tz_timestamps_normalized_not_500(client):
    # Finding 1: Z / offsets must not crash tz-naive comparisons.
    for ts, want in [("2026-08-15T23:10:00Z", "2026-08-15T23:10:00"),
                     ("2026-08-15T23:10:00+06:00", "2026-08-15T17:10:00")]:
        r = client.post("/score", json=_score_body(timestamp=ts))
        assert r.status_code == 200, ts
    for ts in ["not-a-date", "9999-12-31T00:00:00", "0001-01-01T00:00:00"]:
        assert client.post("/score", json=_score_body(timestamp=ts)).status_code == 422, ts


@needs_stack
def test_nonfinite_json_rejected_not_500(client):
    # Finding 2: raw NaN/Infinity/overflow literals must be 422, never 500.
    for raw_amount in ["NaN", "Infinity", "-Infinity", "1e999"]:
        body = ('{"sender_id":"C1","receiver_id":"C2","channel":"app",'
                '"device_id":"D","location":"Dhaka","timestamp":"2026-08-15T12:00:00",'
                f'"type":"P2P","amount":{raw_amount}}}')
        r = client.post("/score", content=body.encode(),
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 422, raw_amount
    assert client.post("/score", json=_score_body(amount=100)).status_code == 200


@needs_stack
def test_input_length_bounds(client):
    # Finding 3 (API side): 1MB strings must not reach history.
    assert client.post("/score", json=_score_body(location="L" * 200)).status_code == 422
    assert client.post("/score", json=_score_body(sender_id="S" * 200)).status_code == 422
    # Short markup still scores (render layer escapes); oversized note is capped.
    tid = client.post("/score", json=_score_body()).json()["txn_id"]
    assert client.post("/decision", json={"txn_id": tid, "decision": "freeze",
                                          "note": "N" * 100000}).status_code == 422


@needs_stack
def test_live_traffic_is_scoring_neutral(client):
    # Finding 4: repeated LIVE scores must not inject velocity / poison seen-sets.
    before = client.get("/health").json()["history_rows"]
    b = _score_body(sender_id="ISO-VIC", amount=5000, timestamp="2026-08-15T12:00:00")
    s1 = client.post("/score", json=b).json()
    s2 = client.post("/score", json=b).json()
    assert not any("velocity" in r for r in s2["top_3_reasons"])
    assert client.get("/health").json()["history_rows"] == before
    assert s1["risk_score"] == s2["risk_score"]


@needs_stack
def test_no_label_leakage(client):
    # Finding 5: ground truth must not be served.
    for row in client.get("/alerts?limit=5").json()["alerts"]:
        assert "label" not in row
    tid = client.get("/alerts?limit=1").json()["alerts"][0]["txn_id"]
    assert "label" not in client.get(f"/case/{tid}").json()


@needs_stack
def test_decision_upsert_and_unknown_404(client):
    # Finding 6: unknown ids 404; duplicates upsert instead of piling up.
    # Unique analyst: the log persists across runs, so a fixed name would
    # turn the first insert into an update on repeat runs.
    from uuid import uuid4
    analyst = f"fuzz-{uuid4().hex[:8]}"
    assert client.post("/decision", json={"txn_id": "TOTALLY-MADE-UP",
                                          "decision": "freeze"}).status_code == 404
    tid = client.get("/alerts?limit=1").json()["alerts"][0]["txn_id"]
    n0 = client.get("/metrics").json()["decisions_logged"]
    assert client.post("/decision", json={"txn_id": tid, "decision": "freeze",
                                          "analyst": analyst}).status_code == 200
    r = client.post("/decision", json={"txn_id": tid, "decision": "freeze",
                                       "analyst": analyst}).json()
    assert r["updated"] is True
    assert client.get("/metrics").json()["decisions_logged"] == n0 + 1


@needs_stack
def test_live_case_resolves(client):
    # Finding 7: POST /score ids must open in /case.
    tid = client.post("/score", json=_score_body()).json()["txn_id"]
    assert client.get(f"/case/{tid}").status_code == 200
    assert client.get("/case/NOPE123").status_code == 404


@needs_stack
def test_limit_bounds(client):
    # Finding 8: negative limits must not silently slice.
    assert client.get("/alerts?limit=-5").status_code == 422
    assert client.get("/alerts?limit=0").status_code == 422
    assert client.get("/alerts?limit=1").status_code == 200


def test_llm_evidence_sanitized():
    # Finding 9: control characters must not reach prompts/narratives.
    from api.llm import _safe, narrate
    assert _safe("a\nb\x00c") == "abc"
    assert len(_safe("x" * 500)) == 120
    out = narrate({"sender_id": "Evil\nINJECT", "receiver_id": "R", "amount": 10,
                   "channel": "app", "timestamp": "2026-08-15T12:00:00"},
                  {"amount_vs_user_avg": 1.0, "new_device": 0, "location_jump": 0,
                   "sender_cnt_1h": 0, "recv_n_senders_1h": 0, "location_new": 0},
                  {"risk_score": 0.1, "risk_level": "Low", "recommended_action": "allow",
                   "top_3_reasons": []},
                  {"boost": 0, "fraud_neighbors_2hop": 0}, lang="en")
    assert "INJECT" in out["narrative"] and "\nINJECT" not in out["narrative"]


@needs_stack
def test_password_reset_flag_reachable(client):
    # Finding 10: the ATO signal must be settable via the API.
    plain = _score_body(sender_id="ATO-T1", amount=1800, device_id="D1",
                        timestamp="2026-08-15T14:00:00")
    flagged = dict(plain, password_reset_flag=1)
    r1 = client.post("/score", json=flagged).json()
    assert any("password reset" in r for r in r1["top_3_reasons"])
    assert client.post("/score", json=dict(plain, password_reset_flag=2)).status_code == 422


@needs_stack
def test_env_example_contract():
    # .env.example must document every runtime env var with placeholder
    # values only — never a real secret.
    from pathlib import Path
    lines = [ln.strip() for ln in Path(".env.example").read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    got = dict(ln.split("=", 1) for ln in lines)
    for var in ("OLLAMA_HOST", "OLLAMA_MODEL", "VIGIL_API_KEY",
                "VIGIL_DECISIONS_DB", "DATABASE_URL",
                "VIGIL_ALERTS_LIMIT", "VIGIL_RATE_LIMIT_PER_MIN"):
        assert var in got, var
    import re
    assert "localhost" in got["OLLAMA_HOST"], "narratives must default to local"
    assert not got["OLLAMA_HOST"].startswith("https://api."), \
        "no cloud endpoint for case evidence"
    assert not any(re.search(r"gsk_[A-Za-z0-9]{20,}", v) for v in got.values()), \
        "real-looking secret in .env.example"


def test_rate_limit_429s(monkeypatch, client):
    # Finding 4 (flood): per-IP bucket trips with a JSON 429, restores after.
    import api.main as main
    main._rate_hits.clear()
    monkeypatch.setenv("VIGIL_RATE_LIMIT_PER_MIN", "2")
    try:
        assert client.post("/score", json=_score_body()).status_code == 200
        assert client.post("/score", json=_score_body()).status_code == 200
        r = client.post("/score", json=_score_body())
        assert r.status_code == 429
    finally:
        monkeypatch.setenv("VIGIL_RATE_LIMIT_PER_MIN", "120")
        main._rate_hits.clear()


@needs_stack
def test_api_key_gates_writes_only_when_set(monkeypatch, client):
    # Open by default (judging demos); locked when VIGIL_API_KEY is set.
    assert client.post("/score", json=_score_body()).status_code == 200
    monkeypatch.setenv("VIGIL_API_KEY", "venue-secret")
    try:
        assert client.post("/score", json=_score_body()).status_code == 401
        tid = client.get("/alerts?limit=1").json()["alerts"][0]["txn_id"]
        no_key = client.post("/decision", json={"txn_id": tid, "decision": "allow"})
        assert no_key.status_code == 401
        headers = {"X-API-Key": "venue-secret"}
        assert client.post("/score", json=_score_body(), headers=headers).status_code == 200
        r = client.post("/decision", json={"txn_id": tid, "decision": "allow",
                                           "analyst": "key-test"}, headers=headers)
        assert r.status_code == 200
        assert client.get("/alerts?limit=1").status_code == 200  # reads stay open
    finally:
        monkeypatch.delenv("VIGIL_API_KEY", raising=False)
