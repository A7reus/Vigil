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


def test_backend_selection(monkeypatch, tmp_path):
    from api.decisions import select_backend
    assert select_backend(tmp_path / "a.db", "") == "sqlite"
    assert select_backend(None, "postgresql://u:p@host/db") == "pg"
    # Explicit file still wins (offline venue over env leftovers).
    assert select_backend(tmp_path / "b.db", "postgresql://u:p@host/db") == "sqlite"


_live_pg = __import__("pytest").mark.skipif(
    not __import__("os").getenv("TEST_POSTGRES_URL"),
    reason="needs TEST_POSTGRES_URL pointing at a throwaway database",
)


@_live_pg
def test_postgres_parity():
    """Same contract, real Postgres: upsert flag, count, reopen persistence."""
    import os
    from api.decisions import DecisionLog
    url = os.environ["TEST_POSTGRES_URL"]
    log = DecisionLog(url=url)
    log._db.execute("TRUNCATE decisions")
    assert log.upsert(_entry()) is False
    assert log.upsert(_entry()) is True
    assert len(DecisionLog(url=url)) == 1
