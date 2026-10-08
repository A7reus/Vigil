"""Authentication + RBAC on Postgres: PBKDF2 passwords (stdlib), opaque tokens.

No JWT library needed: tokens are `vigil_<hex>` stored sha256-hashed with an
expiry, which also makes logout/revocation trivial (delete the row).

Roles: analyst (reviews cases, scores, decides) vs admin (everything plus
user approvals, disable/enable, all reviews, retraining).
Registration is always pending until an admin approves it.

Endpoints stay open without a token (judging demos, service API key) —
`current_user` returns None and `need_user`/`need_admin` gate the routes
that must not be anonymous.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request

log = logging.getLogger("vigil")

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
PW_MIN, PW_MAX = 8, 128
PBKDF2_ITERS = 200_000
TOKEN_DAYS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY,
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
CREATE TABLE IF NOT EXISTS case_state (
  txn_id TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'open'
    CHECK (status IN ('open', 'assigned', 'closed')),
  assignee TEXT,
  updated_at TEXT NOT NULL
);
"""

_lock = threading.Lock()
_db = None


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_conn():
    global _db
    from api.decisions import database_url
    if _db is None or _db.closed:
        import psycopg
        _db = psycopg.connect(database_url(), autocommit=True)
    return _db


def ensure_schema() -> None:
    with _lock:
        cur = _db_conn().cursor()
        cur.execute(SCHEMA)
        cur.close()


def _row(cur, query: str, args: tuple = ()) -> dict | None:
    cur.execute(query, args)
    r = cur.fetchone()
    if r is None:
        return None
    return dict(zip([d[0] for d in cur.description], r))


# --- users ---------------------------------------------------------------
def create_user(username: str, pw_hash: str, role: str = "analyst",
                status: str = "pending") -> dict | None:
    with _lock:
        cur = _db_conn().cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, pw_hash, role, status, created_at)"
                " VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (username, pw_hash, role, status, utcnow()))
            uid = cur.fetchone()[0]
        except Exception:
            cur.close()
            return None  # duplicate username (or any other insert refusal)
        u = _row(cur, "SELECT * FROM users WHERE id = %s", (uid,))
        cur.close()
        return u


def get_user_by_username(username: str) -> dict | None:
    with _lock:
        cur = _db_conn().cursor()
        u = _row(cur, "SELECT * FROM users WHERE username = %s", (username,))
        cur.close()
        return u


def get_user_by_id(uid: int) -> dict | None:
    with _lock:
        cur = _db_conn().cursor()
        u = _row(cur, "SELECT * FROM users WHERE id = %s", (uid,))
        cur.close()
        return u


def set_user_status(uid: int, status: str) -> dict | None:
    with _lock:
        cur = _db_conn().cursor()
        cur.execute("UPDATE users SET status = %s WHERE id = %s", (status, uid))
        u = _row(cur, "SELECT * FROM users WHERE id = %s", (uid,))
        cur.close()
        return u


def list_users() -> list[dict]:
    with _lock:
        cur = _db_conn().cursor()
        cur.execute("SELECT id, username, role, status, created_at FROM users ORDER BY id")
        rows = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
        cur.close()
        return rows


def count_users() -> int:
    with _lock:
        cur = _db_conn().cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        n = cur.fetchone()[0]
        cur.close()
        return n


def public_user(u: dict) -> dict:
    return {k: u[k] for k in ("id", "username", "role", "status", "created_at") if k in u}


# --- sessions ------------------------------------------------------------
def create_session(token_sha: str, user_id: int, expires_at: str) -> None:
    with _lock:
        cur = _db_conn().cursor()
        cur.execute("INSERT INTO sessions (token_sha, user_id, expires_at, created_at)"
                    " VALUES (%s, %s, %s, %s)", (token_sha, user_id, expires_at, utcnow()))
        cur.close()


def get_session(token_sha: str) -> dict | None:
    with _lock:
        cur = _db_conn().cursor()
        s = _row(cur, "SELECT * FROM sessions WHERE token_sha = %s", (token_sha,))
        cur.close()
        return s


def revoke_session(token_sha: str) -> None:
    with _lock:
        cur = _db_conn().cursor()
        cur.execute("DELETE FROM sessions WHERE token_sha = %s", (token_sha,))
        cur.close()


# --- case workflow -------------------------------------------------------
def get_case_state(txn_id: str) -> dict:
    with _lock:
        cur = _db_conn().cursor()
        s = _row(cur, "SELECT * FROM case_state WHERE txn_id = %s", (txn_id,))
        cur.close()
        return s or {"txn_id": txn_id, "status": "open", "assignee": None}


def set_case_state(txn_id: str, status: str | None = None,
                   assignee: str | None = None) -> dict:
    """Assign and/or move a case; untouched sides keep their values."""
    with _lock:
        cur = _db_conn().cursor()
        cur_state = _row(cur, "SELECT * FROM case_state WHERE txn_id = %s", (txn_id,))
        new_status = status or (cur_state["status"] if cur_state else "open")
        new_assignee = assignee if assignee is not None else (cur_state["assignee"] if cur_state else None)
        cur.execute(
            """INSERT INTO case_state (txn_id, status, assignee, updated_at)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT(txn_id) DO UPDATE SET status=excluded.status,
                 assignee=excluded.assignee, updated_at=excluded.updated_at""",
            (txn_id, new_status, new_assignee, utcnow()))
        s = _row(cur, "SELECT * FROM case_state WHERE txn_id = %s", (txn_id,))
        cur.close()
        return s


def list_case_states(status: str | None = None, assignee: str | None = None,
                     limit: int = 500) -> list[dict]:
    q, args = "SELECT * FROM case_state", []
    clauses = []
    if status:
        clauses.append("status = %s")
        args.append(status)
    if assignee:
        clauses.append("assignee = %s")
        args.append(assignee)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY updated_at DESC LIMIT %s"
    args.append(limit)
    with _lock:
        cur = _db_conn().cursor()
        cur.execute(q, tuple(args))
        rows = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
        cur.close()
        return rows


# --- passwords / tokens --------------------------------------------------
def hash_password(pw: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), PBKDF2_ITERS)
    return f"pbkdf2_sha256${PBKDF2_ITERS}${salt}${dk.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, iters, salt, hexed = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), int(iters))
        return hmac.compare_digest(dk.hex(), hexed)
    except Exception:
        return False


def check_username(username: str) -> str:
    if not USERNAME_RE.match(username or ""):
        raise HTTPException(422, "username must be 3-32 chars: letters, digits, _ . -")
    return username


def check_password(password: str) -> str:
    if not (PW_MIN <= len(password or "") <= PW_MAX):
        raise HTTPException(422, f"password must be {PW_MIN}-{PW_MAX} chars")
    return password


def issue_token(user_id: int, days: int | None = None) -> tuple[str, str]:
    """Returns (raw_token, expires_at). Store only the sha256."""
    raw = "vigil_" + secrets.token_hex(24)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    exp = datetime.now(timezone.utc) + timedelta(
        days=days if days is not None else int(os.getenv("VIGIL_TOKEN_DAYS", str(TOKEN_DAYS))))
    create_session(digest, user_id, exp.isoformat())
    return raw, exp.isoformat()


def bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None


def current_user(request: Request) -> dict | None:
    """Authenticated active user, or None (endpoints stay open for judges)."""
    token = bearer_token(request)
    if not token:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    sess = get_session(digest)
    if not sess or sess["expires_at"] <= utcnow():
        return None
    u = get_user_by_id(sess["user_id"])
    if not u or u["status"] != "active":
        return None
    return public_user(u)


def need_user(request: Request) -> dict:
    u = current_user(request)
    if u is None:
        raise HTTPException(401, "authentication required (POST /auth/login)")
    return u


def need_admin(request: Request) -> dict:
    u = need_user(request)
    if u["role"] != "admin":
        raise HTTPException(403, "admin role required")
    return u


def seed_demo_users() -> list[str]:
    """Seed admin/admin123 + analyst/analyst123 on empty user tables.

    Runs at startup only when no users exist and VIGIL_SEED_DEMO is not "0".
    Convenience for demos and the venue laptop; set VIGIL_SEED_DEMO=0 anywhere
    real, and create real accounts instead.
    """
    if count_users() > 0 or os.getenv("VIGIL_SEED_DEMO", "1") == "0":
        return []
    made = []
    for username, password, role in (("admin", "admin123", "admin"),
                                     ("analyst", "analyst123", "analyst")):
        if create_user(username, hash_password(password), role, "active"):
            made.append(f"{username}/{password} ({role})")
    if made:
        log.warning("seeded demo accounts (set VIGIL_SEED_DEMO=0 in production): %s", made)
    return made
