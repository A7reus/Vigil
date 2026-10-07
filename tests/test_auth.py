"""Auth + DB unit tests: hashing, validation, users, sessions, cases, decisions."""
import pytest

from api import auth, db


@pytest.fixture()
def tdb(tmp_path, monkeypatch):
    p = tmp_path / "t.db"
    monkeypatch.setenv("VIGIL_DB_PATH", str(p))
    monkeypatch.setenv("VIGIL_SEED_DEMO", "0")
    db.configure(str(p))
    db.reset_for_tests()
    yield p
    db.configure()  # restore default for other modules


def test_password_roundtrip_and_rejects():
    h = auth.hash_password("correct horse 123")
    assert auth.verify_password("correct horse 123", h)
    assert not auth.verify_password("wrong", h)
    assert not auth.verify_password("x", "garbage")


def test_username_password_rules():
    import fastapi
    for bad in ["ab", "x" * 33, "has space", "semi;colon", ""]:
        with pytest.raises(fastapi.HTTPException):
            auth.check_username(bad)
    assert auth.check_username("analyst_1") == "analyst_1"
    with pytest.raises(fastapi.HTTPException):
        auth.check_password("short")
    assert auth.check_password("long enough 123") == "long enough 123"


def test_register_pending_and_conflict(tdb):
    u = db.create_user("newbie", auth.hash_password("password123"))
    assert u["status"] == "pending" and u["role"] == "analyst"
    assert db.create_user("newbie", auth.hash_password("other123")) is None
    assert db.get_user_by_username("nope") is None


def test_status_flow(tdb):
    u = db.create_user("w", auth.hash_password("password123"))
    assert db.set_user_status(u["id"], "active")["status"] == "active"
    assert db.set_user_status(u["id"], "disabled")["status"] == "disabled"
    assert [x["username"] for x in db.list_users()] == ["w"]


def test_token_issue_lookup_revoke(tdb):
    u = db.create_user("t", auth.hash_password("password123"), status="active")
    raw, exp = auth.issue_token(u["id"])
    assert raw.startswith("vigil_") and exp
    import hashlib
    sess = db.get_session(hashlib.sha256(raw.encode()).hexdigest())
    assert sess and sess["user_id"] == u["id"]
    assert db.get_session("0" * 64) is None
    db.revoke_session(hashlib.sha256(raw.encode()).hexdigest())
    assert db.get_session(hashlib.sha256(raw.encode()).hexdigest()) is None


def test_cases_and_decisions(tdb):
    db.upsert_case("T1", {"risk_score": 0.9}, 0.9, "High")
    assert db.get_case("T1")["status"] == "open"
    assert db.set_case_status("T1", "assigned", "amy") is True
    assert db.set_case_status("NOPE", "closed") is False
    assert [c["txn_id"] for c in db.list_open_cases()] == ["T1"]
    e1, upd1 = db.upsert_decision("T1", "amy", "freeze", "looks bad")
    e2, upd2 = db.upsert_decision("T1", "amy", "allow", "changed mind")
    assert (upd1, upd2) == (False, True) and e2["decision"] == "allow"
    assert db.count_decisions() == 1
    assert len(db.list_decisions(analyst="amy")) == 1
    assert db.list_decisions(analyst="nobody") == []


def test_idempotency_store(tdb):
    assert db.get_idempotent("k") is None
    db.put_idempotent("k", {"ok": True})
    assert db.get_idempotent("k") == {"ok": True}


def test_seed_demo_users(tdb, monkeypatch):
    monkeypatch.setenv("VIGIL_SEED_DEMO", "1")
    made = auth.seed_demo_users()
    assert len(made) == 2
    admin = db.get_user_by_username("admin")
    assert admin["role"] == "admin" and admin["status"] == "active"
    assert auth.verify_password("analyst123", db.get_user_by_username("analyst")["pw_hash"])
    assert auth.seed_demo_users() == []  # second run is a no-op
