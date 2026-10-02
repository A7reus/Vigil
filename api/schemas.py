"""Request/response schemas for POST /score etc."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ScoreRequest(BaseModel):
    sender_id: str = Field(description="Sender wallet id")
    receiver_id: str = Field(description="Receiver wallet id")
    amount: float = Field(gt=0, le=100_000_000, description="Amount in BDT, must be positive")
    channel: Literal["app", "ussd", "agent"] = "app"
    device_id: str = "unknown"
    location: str = "Dhaka"
    timestamp: str = Field(description="ISO-8601 timestamp")
    type: str = "P2P"
    lang: Literal["en", "bn"] = "en"

    @field_validator("timestamp")
    @classmethod
    def _must_be_iso8601(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except Exception:
            raise ValueError("timestamp must be ISO-8601, e.g. 2026-08-15T23:10:00")
        return v

    @field_validator("type")
    @classmethod
    def _must_be_known_type(cls, v: str) -> str:
        allowed = {"P2P", "merchant", "bill", "cash-out", "salary-in"}
        if v not in allowed:
            raise ValueError(f"type must be one of {sorted(allowed)}")
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
    txn_id: str
    decision: Literal["allow", "step-up", "freeze"]
    analyst: str = "analyst-1"
    note: str = ""
