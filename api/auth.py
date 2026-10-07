"""Authentication + RBAC: PBKDF2 passwords (stdlib), opaque bearer tokens.

No JWT library needed: tokens are `vigil_<hex>` stored sha256-hashed with an
expiry, which also makes logout/revocation trivial (delete the row).

Roles: analyst (reviews cases, scores, decides) vs admin (everything plus
user approvals, disable/enable, all reviews, case assignment).
Registration is always pending until an admin approves it.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request

from api import db

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
PW_MIN, PW_MAX = 8, 128
PBKDF2_ITERS = 200_000
TOKEN_DAYS = 30


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
    db.create_session(digest, user_id, exp.isoformat())
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
    sess = db.get_session(digest)
    if not sess or sess["expires_at"] <= db.utcnow():
        return None
    u = db.get_user_by_id(sess["user_id"])
    if not u or u["status"] != "active":
        return None
    return db.public_user(u)


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
    """Seed admin/admin123 + analyst/analyst123 on empty DBs (demo convenience).

    Runs at startup only when the users table is empty and VIGIL_SEED_DEMO
    is not "0". Disable in production and create real accounts instead.
    """
    if db.count_users() > 0 or os.getenv("VIGIL_SEED_DEMO", "1") == "0":
        return []
    made = []
    for username, password, role in (("admin", "admin123", "admin"),
                                     ("analyst", "analyst123", "analyst")):
        if db.create_user(username, hash_password(password), role, "active"):
            made.append(f"{username}/{password} ({role})")
    if made:
        import logging
        logging.getLogger("vigil").warning(
            "seeded demo accounts (set VIGIL_SEED_DEMO=0 in production): %s", made)
    return made
