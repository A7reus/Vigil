"""Business rules kept SEPARATE from ML (architecture requirement).

ML outputs a score; this module maps score -> level/action and owns the
human-oversight policy: never auto-block money.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_cfg = None


def load_config(path: str | Path = "config/thresholds.yaml") -> dict:
    global _cfg
    _cfg = yaml.safe_load(open(path))
    return _cfg


def get_config() -> dict:
    global _cfg
    if _cfg is None:
        load_config()
    return _cfg


def decide(risk_score: float) -> dict:
    cfg = get_config()
    high, med = cfg["bands"]["high"], cfg["bands"]["medium"]
    if risk_score > high:
        return {"risk_level": "High", "recommended_action": "step-up-auth + hold + analyst review"}
    if risk_score >= med:
        return {"risk_level": "Medium", "recommended_action": "review"}
    return {"risk_level": "Low", "recommended_action": "allow"}
