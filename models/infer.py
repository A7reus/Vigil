"""Inference: ensemble scorer + human-readable reasons.

Final score = w_clf * P(fraud) + w_anom * anomaly01 + w_graph * graph_boost-scaled.
Rules/action mapping lives in api/rules.py + config/thresholds.yaml (not here).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features.build import FEATURE_COLS

ART = Path("artifacts")
_clf = _iso = None
_feature_importance: dict[str, float] = {}


def load_artifacts(art_dir: str | Path = ART):
    global _clf, _iso, _feature_importance
    art_dir = Path(art_dir)
    _clf = joblib.load(art_dir / "classifier.pkl")
    _iso = joblib.load(art_dir / "anomaly.pkl")
    cols = json.loads((art_dir / "feature_cols.json").read_text())
    assert cols == FEATURE_COLS, "feature drift: retrain"
    fi = getattr(_clf, "feature_importances_", None)
    if fi is None and hasattr(_clf, "coef_"):
        fi = np.abs(np.asarray(_clf.coef_).ravel())
    if fi is not None:
        _feature_importance = dict(zip(FEATURE_COLS, [float(x) for x in fi]))
    return _clf, _iso


def _anomaly01(X: np.ndarray) -> np.ndarray:
    s = -_iso.score_samples(X)
    return pd.Series(s).rank(pct=True).to_numpy()


def explain_row(feat_row: dict, p_fraud: float) -> list[str]:
    """Top-3 reasons: rule triggers first (auditable), then model importance."""
    reasons = []
    if feat_row.get("new_device", 0) == 1:
        reasons.append("new device not seen for this sender")
    if feat_row.get("location_jump", 0) == 1:
        reasons.append(f"location jump away from home district ({feat_row.get('location_new', 0)} new loc)")
    if float(feat_row.get("amount_vs_user_avg", 0)) > 3:
        reasons.append(f"amount {feat_row['amount_vs_user_avg']:.1f}x user average")
    if int(feat_row.get("sender_cnt_1h", 0)) >= 3:
        reasons.append(f"velocity burst: {feat_row['sender_cnt_1h']} txns in last hour")
    if int(feat_row.get("recv_n_senders_1h", 0)) >= 5:
        reasons.append(f"fan-in: {feat_row['recv_n_senders_1h']} distinct senders to receiver in 1h")
    if int(feat_row.get("is_round_amount", 0)) == 1:
        reasons.append("round mule-like amount (e.g. 9,900/19,500)")
    if int(feat_row.get("password_reset_flag", 0)) == 1:
        reasons.append("password reset just before transfer (ATO signal)")
    if int(feat_row.get("unusual_hour", 0)) == 1:
        reasons.append("unusual hour transfer")
    # backfill with top model features if fewer than 3
    if len(reasons) < 3 and _feature_importance:
        for k, _ in sorted(_feature_importance.items(), key=lambda kv: -kv[1]):
            if k not in ("amount",) and not any(k.replace("_", " ")[:6] in r for r in reasons):
                reasons.append(f"elevated model signal: {k}={feat_row.get(k)}")
            if len(reasons) >= 3:
                break
    return reasons[:3] or [f"classifier P(fraud)={p_fraud:.2f} with no single dominant rule"]


def score_features(feat_row: dict, graph_boost: float = 0.0,
                   weights: dict | None = None) -> dict:
    if _clf is None or _iso is None:
        load_artifacts()
    weights = weights or {"classifier": 0.7, "anomaly": 0.2, "graph_boost": 0.1}
    X = np.array([[float(feat_row[c]) for c in FEATURE_COLS]])
    p = float(_clf.predict_proba(X)[0, 1])
    a = float(_anomaly01(X)[0])
    g = max(0.0, min(float(graph_boost), 0.30)) / 0.30  # normalize boost to 0..1
    final = weights["classifier"] * p + weights["anomaly"] * a + weights["graph_boost"] * g
    return {"p_fraud": round(p, 4), "anomaly": round(a, 4),
            "graph_boost": round(float(graph_boost), 4),
            "risk_score": round(min(max(final, 0.0), 1.0), 4),
            "top_3_reasons": explain_row(feat_row, p)}
