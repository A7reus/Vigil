"""RBAC on Postgres: register/approve/login, roles, case workflow, attribution."""
import os

import pytest
from fastapi.testclient import TestClient

needs_pg = pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason="needs TEST_POSTGRES_URL pointing at a throwaway database",
)
needs_stack = pytest.mark.skipif(
    not (__import__("pathlib").Path("artifacts/classifier.pkl").exists()
         and __import__("pathlib").Path("data/transactions.csv").exists()),
    reason="needs data/ + artifacts/",
)


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def clean_users():
    """Fresh-boot state: empty tables, then the lifespan seed re-applied."""
    import psycopg
    from api import auth
    db = psycopg.connect(os.environ["TEST_POSTGRES_URL"], autocommit=True)
    db.execute("TRUNCATE users, sessions, case_state CASCADE")
    db.close()
    auth.seed_demo_users()


def _login(client, username, password):
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


@needs_pg
def test_password_hashing_roundtrip():
    from api import auth
    h = auth.hash_password("correct horse 123")
    assert auth.verify_password("correct horse 123", h)
    assert not auth.verify_password("wrong", h)
    assert not auth.verify_password("x", "not-a-hash")


@needs_pg
def test_register_login_approve_flow(client, clean_users):
    assert client.post("/auth/register",
                       json={"username": "ops1", "password": "a-strong-password"}).status_code == 201
    # duplicate
    assert client.post("/auth/register",
                       json={"username": "ops1", "password": "other-password"}).status_code == 409
    # pending cannot log in
    r = client.post("/auth/login", json={"username": "ops1", "password": "a-strong-password"})
    assert r.status_code == 403
    # seeded admin approves
    admin = _login(client, "admin", "admin123")
    users = client.get("/admin/users", headers={"Authorization": f"Bearer {admin}"}).json()["users"]
    uid = next(u["id"] for u in users if u["username"] == "ops1")
    assert client.post(f"/admin/users/{uid}/approve",
                       headers={"Authorization": f"Bearer {admin}"}).status_code == 200
    token = _login(client, "ops1", "a-strong-password")
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["user"]
    assert me["username"] == "ops1" and me["role"] == "analyst"
    # logout kills the token
    assert client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


@needs_pg
def test_wrong_password_and_disabled(client, clean_users):
    from api import auth
    auth.create_user("doomed", auth.hash_password("long-enough-pw"), "analyst", "active")
    bad = client.post("/auth/login", json={"username": "doomed", "password": "nope-nope-nope"})
    assert bad.status_code == 401
    admin = _login(client, "admin", "admin123")
    uid = next(u["id"] for u in client.get(
        "/admin/users", headers={"Authorization": f"Bearer {admin}"}).json()["users"]
        if u["username"] == "doomed")
    client.post(f"/admin/users/{uid}/disable", headers={"Authorization": f"Bearer {admin}"})
    r = client.post("/auth/login", json={"username": "doomed", "password": "long-enough-pw"})
    assert r.status_code == 403


@needs_pg
def test_admin_routes_forbid_analysts_and_anon(client, clean_users):
    analyst = _login(client, "analyst", "analyst123")
    assert client.get("/admin/users").status_code == 401
    assert client.get("/admin/users",
                       headers={"Authorization": "Bearer bogus"}).status_code == 401
    assert client.get("/admin/users",
                       headers={"Authorization": f"Bearer {analyst}"}).status_code == 403
    assert client.post("/admin/retrain",
                       headers={"Authorization": f"Bearer {analyst}"}).status_code == 403


@needs_pg
@needs_stack
def test_token_decision_attributes_token_user(client, clean_users):
    tid = client.get("/alerts?limit=1").json()["alerts"][0]["txn_id"]
    analyst = _login(client, "analyst", "analyst123")
    # body claims someone else; the token identity must win
    r = client.post("/decision", json={"txn_id": tid, "decision": "freeze",
                                       "analyst": "impostor"},
                    headers={"Authorization": f"Bearer {analyst}"})
    assert r.status_code == 200
    assert r.json()["logged"]["analyst"] == "analyst"


@needs_pg
@needs_stack
def test_case_assign_and_status_rules(client, clean_users):
    tid = client.get("/alerts?limit=1").json()["alerts"][0]["txn_id"]
    analyst = _login(client, "analyst", "analyst123")
    admin = _login(client, "admin", "admin123")
    h = {"Authorization": f"Bearer {analyst}"}
    # analysts assign only to themselves; unknown cases 404
    assert client.post("/cases/NOPE/assign", json={"analyst": "analyst"}, headers=h).status_code == 404
    assert client.post(f"/cases/{tid}/assign",
                       json={"analyst": "someone-else"}, headers=h).status_code == 403
    r = client.post(f"/cases/{tid}/assign",
                    json={"analyst": "analyst"}, headers=h)
    assert r.status_code == 200 and r.json()["status"] == "assigned"
    assert client.post(f"/cases/{tid}/status", json={"status": "closed"},
                       headers=h).status_code == 200
    assert client.post(f"/cases/{tid}/status", json={"status": "bogus"},
                       headers=h).status_code == 422
    board = client.get("/admin/cases",
                       headers={"Authorization": f"Bearer {admin}"}).json()
    assert any(c["txn_id"] == tid and c["status"] == "closed" for c in board["cases"])
    mine = client.get("/cases", headers=h).json()
    assert any(c["txn_id"] == tid for c in mine["cases"])


@needs_pg
def test_seed_demo_users_idempotent():
    from api import auth
    first = auth.seed_demo_users()
    again = auth.seed_demo_users()
    assert again == []
    assert {u["username"] for u in auth.list_users()} >= {"admin", "analyst"}
    assert first == [] or any("admin" in m for m in first)


@needs_pg
def test_retrain_endpoints_gated_and_tracked(client, clean_users, monkeypatch):
    import time

    import api.main as main
    analyst = _login(client, "analyst", "analyst123")
    admin = _login(client, "admin", "admin123")
    anon, ah = {}, {"Authorization": f"Bearer {admin}"}
    assert client.post("/admin/retrain", headers=anon).status_code == 401
    assert client.post("/admin/retrain",
                       headers={"Authorization": f"Bearer {analyst}"}).status_code == 403
    assert client.get("/admin/retrain/status", headers=anon).status_code == 401

    def _fake_worker(data_dir, artifacts_dir, apply):
        return {"verdict": "HOLD", "stub": True, "apply": apply}

    monkeypatch.setattr(main, "_retrain_worker", _fake_worker)
    assert client.post("/admin/retrain", headers=ah).status_code == 200
    for _ in range(100):
        s = client.get("/admin/retrain/status", headers=ah).json()
        if s["state"] == "done":
            break
        time.sleep(0.1)
    assert s["state"] == "done" and s["result"]["stub"] is True

    main._retrain_job.update(state="running", result=None)
    try:
        assert client.post("/admin/retrain", headers=ah).status_code == 409
    finally:
        main._retrain_job.update(state="idle", result=None)
