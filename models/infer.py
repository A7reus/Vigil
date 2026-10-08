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
_cols: list[str] = list(FEATURE_COLS)  # actual model inputs, from feature_cols.json
_calib: np.ndarray | None = None
_feature_importance: dict[str, float] = {}
_explainer = None  # lazy SHAP TreeExplainer (built on first explained row)
_explainer_broken = False  # set when shap/model combo fails — don't retry


def load_artifacts(art_dir: str | Path = ART):
    global _clf, _iso, _calib, _feature_importance, _explainer, _explainer_broken, _cols
    _explainer = None  # model changed — rebuild lazily
    _explainer_broken = False
    art_dir = Path(art_dir)
    _clf = joblib.load(art_dir / "classifier.pkl")
    _iso = joblib.load(art_dir / "anomaly.pkl")
    cols = json.loads((art_dir / "feature_cols.json").read_text())
    # Full artifacts match FEATURE_COLS exactly; portable ("intersect")
    # artifacts are a strict subset — both are served, unknown names are not.
    assert set(cols) <= set(FEATURE_COLS), f"unknown features in {art_dir}: retrain"
    _cols = list(cols)
    calib_path = art_dir / "anomaly_calib.npy"
    try:
        _calib = np.sort(np.load(calib_path)) if calib_path.exists() else None
    except Exception:
        _calib = None
    # Prefer train-time permutation importance (works for any model); fall back
    # to native importances for artifacts trained before that file existed.
    _feature_importance = {}
    try:
        imp_path = art_dir / "feature_importance.json"
        if imp_path.exists():
            _feature_importance = {k: float(v) for k, v in json.loads(imp_path.read_text()).items()
                                   if k in FEATURE_COLS}
    except Exception:
        _feature_importance = {}
    if not _feature_importance:
        fi = getattr(_clf, "feature_importances_", None)
        if fi is None and hasattr(_clf, "coef_"):
            fi = np.abs(np.asarray(_clf.coef_).ravel())
        if fi is not None:
            _feature_importance = dict(zip(_cols, [float(x) for x in fi]))
    return _clf, _iso


def _anomaly01(X: np.ndarray) -> np.ndarray:
    """Percentile-normalize via TRAIN calibration (no test leakage).

    Old code ranked a single row (always 1.0). Now uses searchsorted against
    the sorted train-normal scores saved at train time. Falls back to 0.5 if
    calibration is missing (e.g. artifacts trained before this fix).
    """
    assert _iso is not None, "call load_artifacts() first"
    s = -_iso.score_samples(X)
    if _calib is None or len(_calib) == 0:
        return np.full_like(s, 0.5, dtype=float)
    idx = np.searchsorted(_calib, np.asarray(s, dtype=float), side="left")
    return (idx / len(_calib)).astype(float)


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
    # backfill with per-row SHAP attributions if fewer than 3;
    # fall back to global importance when SHAP is unavailable
    if len(reasons) < 3:
        for r in _shap_backfill(feat_row, skip_substr=tuple(reasons)):
            reasons.append(r)
            if len(reasons) >= 3:
                break
    if len(reasons) < 3 and _feature_importance:
        for k, _ in sorted(_feature_importance.items(), key=lambda kv: -kv[1]):
            if k not in ("amount",) and not any(k.replace("_", " ")[:6] in r for r in reasons):
                reasons.append(f"elevated model signal: {k}={feat_row.get(k)}")
            if len(reasons) >= 3:
                break
    return reasons[:3] or [f"classifier P(fraud)={p_fraud:.2f} with no single dominant rule"]


def _get_explainer():
    """Lazily build a SHAP TreeExplainer (import ~1.4s, build ~0.04s, row ~1.4ms).

    Returns None when shap is not installed or the model type is unsupported —
    callers must fall back to global feature importance.
    """
    global _explainer, _explainer_broken
    if _explainer is None and not _explainer_broken and _clf is not None:
        try:
            import shap  # type: ignore[import-not-found]  # lazy: keeps cold start fast

            _explainer = shap.TreeExplainer(_clf)
        except Exception:
            _explainer_broken = True  # don't retry a broken combination
    return _explainer


def _shap_backfill(feat_row: dict, skip_substr: tuple = ()) -> list[str]:
    """Top risk-increasing features for THIS row via SHAP (~1.4ms)."""
    ex = _get_explainer()
    if ex is None:
        return []
    try:
        X = np.array([[float(feat_row[c]) for c in _cols]])
        sv = np.asarray(ex.shap_values(X)).ravel()
    except Exception:
        return []
    out = []
    for i in np.argsort(-sv):  # most risk-increasing first
        if sv[i] <= 0:
            break
        k = _cols[int(i)]
        if k in ("amount",):
            continue
        if any(k.replace("_", " ")[:6] in r for r in list(skip_substr) + out):
            continue
        out.append(f"model signal: {k}={feat_row.get(k)} (+{float(sv[i]):.2f})")
    return out


def score_batch(X: np.ndarray, graph_boosts: np.ndarray | list[float] | float = 0.0,
                weights: dict | None = None) -> np.ndarray:
    """Vectorized risk scores for N rows (eval + queue pre-scoring).

    Same formula as score_features but batched: one predict_proba call,
    one score_samples call. graph_boosts are raw boosts (0..0.30).
    """
    if _clf is None or _iso is None:
        load_artifacts()
    assert _clf is not None and _iso is not None
    weights = weights or {"classifier": 0.7, "anomaly": 0.2, "graph_boost": 0.1}
    X = np.asarray(X, dtype=float)
    p = _clf.predict_proba(X)[:, 1]
    a = _anomaly01(X)
    if np.isscalar(graph_boosts):
        gb = np.full(len(X), float(graph_boosts))  # type: ignore[arg-type]
    elif isinstance(graph_boosts, np.ndarray):
        gb = graph_boosts.astype(float)
    else:
        gb = np.asarray(graph_boosts, dtype=float)
    g = np.clip(gb, 0.0, 0.30) / 0.30
    final = weights["classifier"] * p + weights["anomaly"] * a + weights["graph_boost"] * g
    return np.clip(final, 0.0, 1.0)


def score_features(feat_row: dict, graph_boost: float = 0.0,
                   weights: dict | None = None) -> dict:
    if _clf is None or _iso is None:
        load_artifacts()
    assert _clf is not None
    weights = weights or {"classifier": 0.7, "anomaly": 0.2, "graph_boost": 0.1}
    X = np.array([[float(feat_row[c]) for c in _cols]])
    p = float(_clf.predict_proba(X)[0, 1])
    a = float(_anomaly01(X)[0])
    g = max(0.0, min(float(graph_boost), 0.30)) / 0.30  # normalize boost to 0..1
    final = weights["classifier"] * p + weights["anomaly"] * a + weights["graph_boost"] * g
    return {"p_fraud": round(p, 4), "anomaly": round(a, 4),
            "graph_boost": round(float(graph_boost), 4),
            "risk_score": round(min(max(final, 0.0), 1.0), 4),
            "top_3_reasons": explain_row(feat_row, p)}
