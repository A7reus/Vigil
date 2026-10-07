"""Decision audit log: Postgres contract (upsert flag, count, persistence)."""
import os

import pytest

from api.decisions import DecisionLog, database_url

needs_pg = pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason="needs TEST_POSTGRES_URL pointing at a throwaway database",
)


def _entry(txn="T1", analyst="a1", decision="step-up"):
    return {"txn_id": txn, "decision": decision, "analyst": analyst,
            "note": "", "at": "2026-08-15T12:00:00+00:00"}


@pytest.fixture
def log():
    lg = DecisionLog(url=os.environ["TEST_POSTGRES_URL"])
    lg._db.execute("TRUNCATE decisions")
    return lg


def test_database_url_required(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        database_url()
    assert database_url("postgresql://x") == "postgresql://x"


@needs_pg
def test_upsert_returns_updated_flag(log):
    assert log.upsert(_entry()) is False
    assert log.upsert(_entry()) is True  # same analyst+txn overwrites
    assert len(log) == 1
    assert log.upsert(_entry(analyst="a2")) is False  # another analyst: new row
    assert len(log) == 2


@needs_pg
def test_log_survives_reopen():
    url = os.environ["TEST_POSTGRES_URL"]
    DecisionLog(url=url).upsert(_entry(txn="REOPEN-1"))
    reopened = DecisionLog(url=url)
    assert any(r["txn_id"] == "REOPEN-1" for r in reopened.all())
