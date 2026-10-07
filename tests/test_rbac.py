"""RBAC + persistence: register/approve/login, roles, commit flag, DB queue."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # DB isolation comes from the autouse fixture in conftest.py.
    from api.main import app
    with TestClient(app) as c:
        yield c


def _h(token):
    return {"Authorization": f"Bearer {token}"}


def _analyst_token(client):
    r = client.post("/auth/login", json={"username": "analyst", "password": "analyst123"})
    assert r.status_code == 200
    return r.json()["token"]


def _admin_token(client):
    r = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    return r.json()["token"]


def test_seed_and_login(client):
    r = client.get("/admin/users", headers=_h(_admin_token(client)))
    names = {u["username"]: u for u in r.json()["users"]}
    assert names["admin"]["role"] == "admin" and names["analyst"]["role"] == "analyst"
    assert all(u["status"] == "active" for u in names.values())


def test_register_pending_gate_and_approve(client):
    assert client.post("/auth/register",
                       json={"username": "newbie", "password": "password123"}).status_code == 201
    assert client.post("/auth/register",
                       json={"username": "newbie", "password": "password123"}).status_code == 409
    assert client.post("/auth/login",
                       json={"username": "newbie", "password": "password123"}).status_code == 403
    # analyst cannot approve; admin can
    users = client.get("/admin/users", headers=_h(_admin_token(client))).json()["users"]
    uid = next(u["id"] for u in users if u["username"] == "newbie")
    assert client.post(f"/admin/users/{uid}/approve").status_code == 401
    assert client.post(f"/admin/users/{uid}/approve",
                       headers=_h(_analyst_token(client))).status_code == 403
    assert client.post(f"/admin/users/{uid}/approve",
                       headers=_h(_admin_token(client))).status_code == 200
    r = client.post("/auth/login", json={"username": "newbie", "password": "password123"})
    assert r.status_code == 200 and r.json()["user"]["status"] == "active"


def test_login_rejects_and_me_and_logout(client):
    assert client.post("/auth/login",
                       json={"username": "analyst", "password": "nope"}).status_code == 401
    tok = _analyst_token(client)
    assert client.get("/auth/me", headers=_h(tok)).json()["user"]["username"] == "analyst"
    assert client.get("/auth/me").status_code == 401
    assert client.post("/auth/logout", headers=_h(tok)).status_code == 200
    assert client.get("/auth/me", headers=_h(tok)).status_code == 401


def test_disable_enable_cycle(client):
    admin = _admin_token(client)
    users = client.get("/admin/users", headers=_h(admin)).json()["users"]
    uid = next(u["id"] for u in users if u["username"] == "analyst")
    admin_id = next(u["id"] for u in users if u["username"] == "admin")
    assert client.post(f"/admin/users/{admin_id}/disable", headers=_h(admin)).status_code == 400
    client.post(f"/admin/users/{uid}/disable", headers=_h(admin))
    assert client.post("/auth/login",
                       json={"username": "analyst", "password": "analyst123"}).status_code == 403
    client.post(f"/admin/users/{uid}/enable", headers=_h(admin))
    assert client.post("/auth/login",
                       json={"username": "analyst", "password": "analyst123"}).status_code == 200


def test_commit_flag_updates_features(client):
    tok = _analyst_token(client)
    b = {"sender_id": "CMT", "receiver_id": "R", "amount": 5000, "channel": "app",
         "device_id": "D", "location": "Dhaka", "timestamp": "2026-08-15T12:00:00", "type": "P2P"}
    assert client.post("/score?commit=true", json=b).status_code == 401
    for _ in range(3):
        assert client.post("/score?commit=true", json=b, headers=_h(tok)).status_code == 200
    s = client.post("/score?commit=true", json=b, headers=_h(tok)).json()
    assert any("velocity" in r for r in s["top_3_reasons"])


def test_db_cases_join_queue_and_survive_restart(client):
    import api.main as main
    b = {"sender_id": "DBQ", "receiver_id": "R", "amount": 90000, "channel": "app",
         "device_id": "DX1", "location": "Dhaka", "timestamp": "2026-08-15T23:00:00", "type": "P2P"}
    tid = client.post("/score", json=b).json()["txn_id"]
    assert any(a["txn_id"] == tid for a in client.get("/alerts?limit=500").json()["alerts"])
    main.live_cases.clear()  # simulate restart: memory gone, DB remains
    assert client.get(f"/case/{tid}").status_code == 200


def test_decision_uses_token_identity_and_upserts(client):
    tok = _analyst_token(client)
    tid = client.get("/alerts?limit=1").json()["alerts"][0]["txn_id"]
    r = client.post("/decision", json={"txn_id": tid, "decision": "freeze", "analyst": "spoofed"},
                    headers=_h(tok))
    assert r.json()["logged"]["analyst"] == "analyst"  # token wins over body
    n0 = client.get("/metrics").json()["decisions_logged"]
    r2 = client.post("/decision", json={"txn_id": tid, "decision": "allow"}, headers=_h(tok))
    assert r2.json()["updated"] is True
    assert client.get("/metrics").json()["decisions_logged"] == n0


def test_case_assign_and_close_rules(client):
    admin, analyst = _admin_token(client), _analyst_token(client)
    tid = client.get("/alerts?limit=1").json()["alerts"][0]["txn_id"]
    assert client.post(f"/cases/{tid}/assign", json={"analyst": "analyst"}).status_code == 401
    assert client.post(f"/cases/{tid}/assign", json={"analyst": "ghost"},
                       headers=_h(admin)).status_code == 404
    assert client.post(f"/cases/{tid}/assign", json={"analyst": "admin"},
                       headers=_h(analyst)).status_code == 403
    assert client.post(f"/cases/{tid}/assign", json={"analyst": "analyst"},
                       headers=_h(analyst)).status_code == 200
    assert client.post(f"/cases/{tid}/status", json={"status": "closed"},
                       headers=_h(admin)).status_code == 200
    mine = client.post("/score", json={"sender_id": "CLS", "receiver_id": "R", "amount": 100,
                                       "channel": "app", "device_id": "D", "location": "Dhaka",
                                       "timestamp": "2026-08-15T12:00:00", "type": "P2P"}).json()["txn_id"]
    client.post(f"/cases/{mine}/assign", json={"analyst": "analyst"}, headers=_h(admin))
    assert client.post(f"/cases/{mine}/status", json={"status": "closed"},
                       headers=_h(analyst)).status_code == 200
    assert client.post(f"/cases/{mine}/status", json={"status": "bogus"},
                       headers=_h(admin)).status_code == 422


def test_admin_lists(client):
    admin = _admin_token(client)
    assert client.get("/admin/decisions", headers=_h(admin)).status_code == 200
    assert client.get("/admin/cases?status=open", headers=_h(admin)).status_code == 200
    assert client.get("/admin/cases?status=bogus", headers=_h(admin)).status_code == 422
    assert client.get("/admin/decisions").status_code == 401
    assert client.get("/admin/cases", headers=_h(_analyst_token(client))).status_code == 403
