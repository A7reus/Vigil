"""Analyst decision audit log: SQLite locally, Postgres when hosted.

Phase 1 kept decisions in a process-local list, so every restart wiped the
audit trail. SQLite fixed the venue laptop (zero deps, offline, survives
restarts). But hosted-free disks are ephemeral — Render wipes them on every
sleep/redeploy — so when `DATABASE_URL` is set we speak Postgres instead
(Render free PG covers the judging window; paid covers the pilot).

Same upsert contract on both backends (same analyst + txn overwrites instead
of duplicating). `main.py` just builds `DecisionLog()` and lets the env pick.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

SCHEMA = """CREATE TABLE IF NOT EXISTS decisions (
  txn_id TEXT NOT NULL, decision TEXT NOT NULL,
  analyst TEXT NOT NULL DEFAULT 'analyst',
  note TEXT NOT NULL DEFAULT '',
  at TEXT NOT NULL,
  PRIMARY KEY (txn_id, analyst))"""


def default_db_path() -> Path:
    return Path(os.getenv("VIGIL_DECISIONS_DB", "data/decisions.db"))


def select_backend(path: str | Path | None, url: str) -> str:
    # An explicit file path always means SQLite (tests, offline venue).
    # Otherwise the environment picks: Postgres when hosted, SQLite at home.
    return "sqlite" if path is not None or not url else "pg"


class DecisionLog:
    def __init__(self, path: str | Path | None = None, url: str | None = None):
        url = url if url is not None else os.getenv("DATABASE_URL", "")
        self.backend = select_backend(path, url)
        self._lock = threading.Lock()
        if self.backend == "pg":
            import psycopg
            self._db = psycopg.connect(url, autocommit=True)
            self._ph = "%s"
        else:
            self.path = Path(path) if path is not None else default_db_path()
            if str(self.path) != ":memory:":
                self.path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: FastAPI serves requests on a thread pool.
            self._db = sqlite3.connect(str(self.path), check_same_thread=False)
            self._ph = "?"
        with self._lock:
            cur = self._db.cursor()
            cur.execute(SCHEMA)
            try:
                self._db.commit()
            except Exception:
                pass  # autocommit backends (psycopg) have nothing to commit
            cur.close()

    def upsert(self, entry: dict) -> bool:
        """Insert or overwrite the same analyst+txn row. Returns updated."""
        p = self._ph
        with self._lock:
            cur = self._db.cursor()
            cur.execute(f"SELECT 1 FROM decisions WHERE txn_id={p} AND analyst={p}",
                        (entry["txn_id"], entry.get("analyst", "analyst")))
            updated = cur.fetchone() is not None
            cur.execute(
                f"""INSERT INTO decisions (txn_id, decision, analyst, note, at)
                    VALUES ({p}, {p}, {p}, {p}, {p})
                    ON CONFLICT(txn_id, analyst)
                    DO UPDATE SET decision=excluded.decision, note=excluded.note,
                                  at=excluded.at""",
                (entry["txn_id"], entry["decision"], entry.get("analyst", "analyst"),
                 entry.get("note", ""), entry.get("at", "")))
            try:
                self._db.commit()
            except Exception:
                pass
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
