"""Request/response schemas for POST /score etc."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ScoreRequest(BaseModel):
    sender_id: str = Field(description="Sender wallet id", max_length=64)
    receiver_id: str = Field(description="Receiver wallet id", max_length=64)
    amount: float = Field(gt=0, le=100_000_000, description="Amount in BDT, must be positive")
    channel: Literal["app", "ussd", "agent"] = "app"
    device_id: str = Field(default="unknown", max_length=64)
    location: str = Field(default="Dhaka", max_length=64)
    timestamp: str = Field(description="ISO-8601 timestamp")
    type: str = "P2P"
    lang: Literal["en", "bn"] = "en"
    # ATO signal: was silently dropped (extra fields ignored), so it could never
    # fire via the API. 0/1 only; featurize + reasons already handle it.
    password_reset_flag: int = Field(default=0, ge=0, le=1)

    @field_validator("timestamp")
    @classmethod
    def _must_be_iso8601(cls, v: str) -> str:
        # History is tz-naive; mixing aware/naive datetimes crashes comparisons
        # (500) and out-of-range years overflow pandas. Normalize here so every
        # downstream consumer sees a bounded naive ISO string.
        try:
            ts = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            raise ValueError("timestamp must be ISO-8601, e.g. 2026-08-15T23:10:00")
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        if not (datetime(2020, 1, 1) <= ts <= datetime(2030, 12, 31)):
            raise ValueError("timestamp out of supported range 2020-01-01..2030-12-31")
        return ts.isoformat()

    @field_validator("type")
    @classmethod
    def _must_be_known_type(cls, v: str) -> str:
        allowed = {"P2P", "merchant", "bill", "cash-out", "salary-in"}
        if v not in allowed:
            raise ValueError(f"type must be one of {sorted(allowed)}")
        return v

    @field_validator("amount")
    @classmethod
    def _must_be_finite(cls, v: float) -> float:
        # Defense in depth: the JSON middleware already rejects non-finite
        # literals over HTTP; this covers direct Python callers.
        if not math.isfinite(v):
            raise ValueError("amount must be a finite number")
        return v


class ScoreResponse(BaseModel):
    risk_score: float
    risk_level: str
    top_3_reasons: list[str]
    recommended_action: str
    components: dict
    narrative: str
    txn_id: str


class DecisionRequest(BaseModel):
    txn_id: str = Field(max_length=64)
    decision: Literal["allow", "step-up", "freeze"]
    analyst: str = Field(default="analyst-1", max_length=64)
    note: str = Field(default="", max_length=500)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class CaseAssignRequest(BaseModel):
    analyst: str = Field(min_length=3, max_length=32)


class CaseStatusRequest(BaseModel):
    status: Literal["open", "assigned", "closed"]
