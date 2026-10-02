"""Request/response schemas for POST /score etc."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    sender_id: str = Field(description="Sender wallet id")
    receiver_id: str = Field(description="Receiver wallet id")
    amount: float = Field(gt=0)
    channel: Literal["app", "ussd", "agent"] = "app"
    device_id: str = "unknown"
    location: str = "Dhaka"
    timestamp: str = Field(description="ISO-8601 timestamp")
    type: str = "P2P"
    lang: Literal["en", "bn"] = "en"


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
