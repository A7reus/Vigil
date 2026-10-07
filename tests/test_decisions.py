"""Decision audit log: upsert contract + restart persistence (SQLite)."""
from api.decisions import DecisionLog


def _entry(txn="T1", analyst="a1", decision="step-up"):
    return {"txn_id": txn, "decision": decision, "analyst": analyst,
            "note": "", "at": "2026-08-15T12:00:00+00:00"}


def test_upsert_returns_updated_flag(tmp_path):
    log = DecisionLog(tmp_path / "d.db")
    assert log.upsert(_entry()) is False
    assert log.upsert(_entry()) is True  # same analyst+txn overwrites
    assert len(log) == 1
    assert log.upsert(_entry(analyst="a2")) is False  # another analyst: new row
    assert len(log) == 2


def test_log_survives_reopen(tmp_path):
    path = tmp_path / "d.db"
    DecisionLog(path).upsert(_entry())
    reopened = DecisionLog(path)
    assert len(reopened) == 1
    assert reopened.all()[0]["txn_id"] == "T1"


def test_default_path_follows_env(tmp_path, monkeypatch):
    from api import decisions
    monkeypatch.setenv("VIGIL_DECISIONS_DB", str(tmp_path / "custom.db"))
    assert decisions.default_db_path() == tmp_path / "custom.db"
