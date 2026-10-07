"""SQLite persistence: users, sessions, cases, decisions, idempotency keys.

Shared schema contract — analyst queue, decisions and analyst identity all
read and write these tables (teammate: align the P0 SQLite work here):

  users(id, username UNIQUE, pw_hash, role, status, created_at)
    role:   analyst | admin
    status: pending | active | disabled
  sessions(token_sha PK, user_id FK, expires_at, created_at)
  cases(txn_id PK, payload JSON, risk_score, risk_level, status,
        analyst, source, created_at, updated_at)
    status: open | assigned | closed ; source: live | queue
  decisions(id PK, txn_id, analyst, decision, note, at, UNIQUE(txn_id, analyst))
  idempotency(key PK, response JSON, created_at)

Single connection + lock (SQLite is file-local; TestClient threads share it).
Multi-worker deployments need one worker (documented) or an external DB.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  pw_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('analyst', 'admin')),
  status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'disabled')),
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  token_sha TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
  txn_id TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  risk_score REAL NOT NULL,
  risk_level TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'assigned', 'closed')),
  analyst TEXT,
  source TEXT NOT NULL DEFAULT 'live',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  txn_id TEXT NOT NULL,
  analyst TEXT NOT NULL,
  decision TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  at TEXT NOT NULL,
  UNIQUE (txn_id, analyst)
);
CREATE TABLE IF NOT EXISTS idempotency (
  key TEXT PRIMARY KEY,
  response TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_status_score ON cases(status, risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_txn ON decisions(txn_id);
"""

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_path: str | None = None


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure(path: str | Path | None = None) -> sqlite3.Connection:
    """(Re)connect to the DB file and ensure schema. Called in lifespan and tests."""
    global _conn, _path
    p = str(path or os.getenv("VIGIL_DB_PATH", "vigil.db"))
    with _lock:
        if _conn is not None and _path == p:
            return _conn
        if _conn is not None:
            _conn.close()
        _conn = sqlite3.connect(p, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        _path = p
        return _conn


def conn() -> sqlite3.Connection:
    if _conn is None:
        return configure()
    return _conn


def _row(r: sqlite3.Row | None) -> dict | None:
    return dict(r) if r is not None else None


# --- users ---------------------------------------------------------------
def create_user(username: str, pw_hash: str, role: str = "analyst",
                status: str = "pending") -> dict | None:
    c = conn()
    with _lock:
        try:
            cur = c.execute(
                "INSERT INTO users (username, pw_hash, role, status, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (username, pw_hash, role, status, utcnow()))
            c.commit()
        except sqlite3.IntegrityError:
            return None
        return get_user_by_id(cur.lastrowid)


def get_user_by_username(username: str) -> dict | None:
    c = conn()
    with _lock:
        return _row(c.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone())


def get_user_by_id(uid: int) -> dict | None:
    c = conn()
    with _lock:
        return _row(c.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone())


def set_user_status(uid: int, status: str) -> dict | None:
    c = conn()
    with _lock:
        c.execute("UPDATE users SET status = ? WHERE id = ?", (status, uid))
        c.commit()
        return get_user_by_id(uid)


def list_users() -> list[dict]:
    c = conn()
    with _lock:
        return [dict(r) for r in
                c.execute("SELECT id, username, role, status, created_at FROM users ORDER BY id").fetchall()]


def count_users() -> int:
    c = conn()
    with _lock:
        return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def public_user(u: dict) -> dict:
    return {k: u[k] for k in ("id", "username", "role", "status", "created_at") if k in u}


# --- sessions ------------------------------------------------------------
def create_session(token_sha: str, user_id: int, expires_at: str) -> None:
    c = conn()
    with _lock:
        c.execute("INSERT INTO sessions (token_sha, user_id, expires_at, created_at)"
                  " VALUES (?, ?, ?, ?)", (token_sha, user_id, expires_at, utcnow()))
        c.commit()


def get_session(token_sha: str) -> dict | None:
    c = conn()
    with _lock:
        return _row(c.execute("SELECT * FROM sessions WHERE token_sha = ?", (token_sha,)).fetchone())


def revoke_session(token_sha: str) -> None:
    c = conn()
    with _lock:
        c.execute("DELETE FROM sessions WHERE token_sha = ?", (token_sha,))
        c.commit()


# --- cases ---------------------------------------------------------------
def upsert_case(txn_id: str, payload: dict, risk_score: float, risk_level: str,
                source: str = "live") -> None:
    import json
    now = utcnow()
    c = conn()
    with _lock:
        c.execute(
            """INSERT INTO cases (txn_id, payload, risk_score, risk_level, status, source, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'open', ?, ?, ?)
               ON CONFLICT (txn_id) DO UPDATE SET payload=excluded.payload,
                 risk_score=excluded.risk_score, risk_level=excluded.risk_level,
                 updated_at=excluded.updated_at""",
            (txn_id, json.dumps(payload), float(risk_score), risk_level, source, now, now))
        c.commit()


def get_case(txn_id: str) -> dict | None:
    c = conn()
    with _lock:
        return _row(c.execute("SELECT * FROM cases WHERE txn_id = ?", (txn_id,)).fetchone())


def list_open_cases(limit: int = 500) -> list[dict]:
    c = conn()
    with _lock:
        return [dict(r) for r in c.execute(
            "SELECT * FROM cases WHERE status != 'closed'"
            " ORDER BY risk_score DESC LIMIT ?", (limit,)).fetchall()]


def list_cases(status: str | None = None, analyst: str | None = None,
               limit: int = 500) -> list[dict]:
    q = "SELECT * FROM cases"
    clauses, args = [], []
    if status:
        clauses.append("status = ?")
        args.append(status)
    if analyst:
        clauses.append("analyst = ?")
        args.append(analyst)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY updated_at DESC LIMIT ?"
    args.append(limit)
    c = conn()
    with _lock:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def set_case_status(txn_id: str, status: str, analyst: str | None = None) -> bool:
    c = conn()
    with _lock:
        if analyst is None:
            cur = c.execute("UPDATE cases SET status = ?, updated_at = ? WHERE txn_id = ?",
                            (status, utcnow(), txn_id))
        else:
            cur = c.execute("UPDATE cases SET status = ?, analyst = ?, updated_at = ? WHERE txn_id = ?",
                            (status, analyst, utcnow(), txn_id))
        c.commit()
        return cur.rowcount > 0


# --- decisions -----------------------------------------------------------
def upsert_decision(txn_id: str, analyst: str, decision: str, note: str = "") -> tuple[dict, bool]:
    """Returns (entry, updated?). Replaces the old in-memory upsert loop."""
    now = utcnow()
    c = conn()
    with _lock:
        cur = c.execute("SELECT id FROM decisions WHERE txn_id = ? AND analyst = ?",
                        (txn_id, analyst))
        row = cur.fetchone()
        if row:
            c.execute("UPDATE decisions SET decision = ?, note = ?, at = ? WHERE id = ?",
                      (decision, note, now, row["id"]))
            c.commit()
            updated = True
        else:
            c.execute("INSERT INTO decisions (txn_id, analyst, decision, note, at)"
                      " VALUES (?, ?, ?, ?, ?)", (txn_id, analyst, decision, note, now))
            c.commit()
            updated = False
        entry = _row(c.execute(
            "SELECT * FROM decisions WHERE txn_id = ? AND analyst = ?",
            (txn_id, analyst)).fetchone())
        assert entry is not None
        return entry, updated


def list_decisions(analyst: str | None = None, limit: int = 500) -> list[dict]:
    c = conn()
    with _lock:
        if analyst:
            rows = c.execute("SELECT * FROM decisions WHERE analyst = ?"
                             " ORDER BY at DESC LIMIT ?", (analyst, limit)).fetchall()
        else:
            rows = c.execute("SELECT * FROM decisions ORDER BY at DESC LIMIT ?",
                             (limit,)).fetchall()
        return [dict(r) for r in rows]


def count_decisions() -> int:
    c = conn()
    with _lock:
        return c.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]


# --- idempotency ---------------------------------------------------------
def get_idempotent(key: str) -> dict | None:
    import json
    c = conn()
    with _lock:
        row = c.execute("SELECT response FROM idempotency WHERE key = ?", (key,)).fetchone()
        return json.loads(row["response"]) if row else None


def put_idempotent(key: str, response: dict) -> None:
    import json
    c = conn()
    with _lock:
        c.execute("INSERT OR IGNORE INTO idempotency (key, response, created_at)"
                  " VALUES (?, ?, ?)", (key, json.dumps(response), utcnow()))
        c.commit()


def reset_for_tests() -> None:
    """Wipe all rows (test isolation only — never call in production paths)."""
    c = conn()
    with _lock:
        for t in ("idempotency", "decisions", "cases", "sessions", "users"):
            c.execute(f"DELETE FROM {t}")
        c.commit()


def row_to_public_case(row: dict) -> dict:
    """Shape a DB case row like an /alerts entry (no internals, no labels)."""
    import json
    payload = json.loads(row["payload"]) if isinstance(row.get("payload"), str) else {}
    out = {"txn_id": row["txn_id"], "status": row.get("status", "open")}
    if isinstance(payload, dict):
        out.update(payload)
    out["risk_score"] = float(row.get("risk_score", out.get("risk_score", 0.0)))
    out["risk_level"] = row.get("risk_level", out.get("risk_level", "Low"))
    if row.get("analyst"):
        out["assignee"] = row["analyst"]
    return out



