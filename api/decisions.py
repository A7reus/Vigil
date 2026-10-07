"""Analyst decision audit log: Postgres, the only backend.

Decisions are the audit trail and the retrain queue, so they live in a
managed database that survives sleeps, restarts, and redeploys — hosted
free disks are wiped, which is where SQLite died. `DATABASE_URL` is
required; the API refuses to boot without it. Local dev points it at a
throwaway Postgres (`docker run ... postgres:16-alpine`); CI provides one
as a service; the test suite gates on `TEST_POSTGRES_URL`.
"""
from __future__ import annotations

import os
import threading

SCHEMA = """CREATE TABLE IF NOT EXISTS decisions (
  txn_id TEXT NOT NULL, decision TEXT NOT NULL,
  analyst TEXT NOT NULL DEFAULT 'analyst',
  note TEXT NOT NULL DEFAULT '',
  at TEXT NOT NULL,
  PRIMARY KEY (txn_id, analyst))"""


def database_url(url: str | None = None) -> str:
    url = url if url is not None else os.getenv("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Local: run a throwaway Postgres and "
            "export DATABASE_URL=postgresql://vigil:vigil@localhost:5432/vigil "
            "(see README). Hosted: link the Render PostgreSQL so Render injects it."
        )
    return url


class DecisionLog:
    def __init__(self, url: str | None = None):
        import psycopg
        self._lock = threading.Lock()
        self._db = psycopg.connect(database_url(url), autocommit=True)
        with self._lock:
            cur = self._db.cursor()
            cur.execute(SCHEMA)
            cur.close()

    def upsert(self, entry: dict) -> bool:
        """Insert or overwrite the same analyst+txn row. Returns updated."""
        with self._lock:
            cur = self._db.cursor()
            cur.execute("SELECT 1 FROM decisions WHERE txn_id=%s AND analyst=%s",
                        (entry["txn_id"], entry.get("analyst", "analyst")))
            updated = cur.fetchone() is not None
            cur.execute(
                """INSERT INTO decisions (txn_id, decision, analyst, note, at)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT(txn_id, analyst)
                   DO UPDATE SET decision=excluded.decision, note=excluded.note,
                                 at=excluded.at""",
                (entry["txn_id"], entry["decision"], entry.get("analyst", "analyst"),
                 entry.get("note", ""), entry.get("at", "")))
            cur.close()
        return updated

    def __len__(self) -> int:
        with self._lock:
            cur = self._db.cursor()
            cur.execute("SELECT COUNT(*) FROM decisions")
            n = cur.fetchone()[0]
            cur.close()
        return n

    def all(self) -> list[dict]:
        with self._lock:
            cur = self._db.cursor()
            cur.execute("SELECT txn_id, decision, analyst, note, at FROM decisions ORDER BY at")
            rows = cur.fetchall()
            cur.close()
        return [{"txn_id": t, "decision": d, "analyst": a, "note": n, "at": ts}
                for t, d, a, n, ts in rows]
