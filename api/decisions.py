"""Analyst decision audit log: SQLite, survives restarts.

Phase 1 kept decisions in a process-local list, so every restart wiped the
audit trail and the retrain queue — the least "real" part of the prototype.
This is the same upsert contract (same analyst + txn overwrites instead of
duplicating) backed by a file the next boot reopens.

Default lives next to the data it annotates (`data/decisions.db`, git-ignored
like everything regenerable); override with `VIGIL_DECISIONS_DB`.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def default_db_path() -> Path:
    return Path(os.getenv("VIGIL_DECISIONS_DB", "data/decisions.db"))


class DecisionLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_db_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI serves requests on a thread pool.
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS decisions (
                 txn_id TEXT NOT NULL, decision TEXT NOT NULL,
                 analyst TEXT NOT NULL DEFAULT 'analyst',
                 note TEXT NOT NULL DEFAULT '',
                 at TEXT NOT NULL,
                 PRIMARY KEY (txn_id, analyst))"""
        )
        self._db.commit()

    def upsert(self, entry: dict) -> bool:
        """Insert or overwrite the same analyst+txn row. Returns updated."""
        cur = self._db.execute("SELECT 1 FROM decisions WHERE txn_id=? AND analyst=?",
                               (entry["txn_id"], entry.get("analyst", "analyst")))
        updated = cur.fetchone() is not None
        self._db.execute(
            """INSERT INTO decisions (txn_id, decision, analyst, note, at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(txn_id, analyst)
               DO UPDATE SET decision=excluded.decision, note=excluded.note,
                             at=excluded.at""",
            (entry["txn_id"], entry["decision"], entry.get("analyst", "analyst"),
             entry.get("note", ""), entry.get("at", "")))
        self._db.commit()
        return updated

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

    def all(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT txn_id, decision, analyst, note, at FROM decisions ORDER BY at").fetchall()
        return [{"txn_id": t, "decision": d, "analyst": a, "note": n, "at": ts}
                for t, d, a, n, ts in rows]
