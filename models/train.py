"""Train classifier + anomaly detector. XGBoost if present, else sklearn HGB fallback.

Usage:
    python -m models.train --data data --artifacts artifacts --sample 50000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from features.build import FEATURE_COLS, INTERSECT_COLS, build_features

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 100) -> float:
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].mean()) if len(idx) else 0.0


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr: float = 0.05) -> float:
    from sklearn.metrics import roc_curve
    fprs, tprs, _ = roc_curve(y_true, scores)
    mask = fprs <= fpr
    return float(tprs[mask].max()) if mask.any() else 0.0


def calibrate_anomaly(scores_calib: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Percentile-normalize anomaly scores using TRAIN-only calibration.

    rank(pct) over the test set leaks test distribution and always returns 1.0
    for a single row at serve time. Instead, map each score to its percentile
    within the sorted train-normal scores via searchsorted (O(log N)).
    """
    calib = np.sort(np.asarray(scores_calib, dtype=float))
    s = np.asarray(scores, dtype=float)
    if len(calib) == 0:
        return np.full_like(s, 0.5, dtype=float)
    idx = np.searchsorted(calib, s, side="left")
    return (idx / len(calib)).astype(float)


def train(data_dir: str = "data", artifacts: str = "artifacts", sample: int | None = None,
          feature_set: str = "full"):
    """Train classifier + anomaly detector.

    feature_set="full" uses all FEATURE_COLS. "intersect" uses only the
    cross-schema INTERSECT_COLS (no device/location/reset signals) so the
    frozen artifact can score foreign feeds zero-shot.
    """
    assert feature_set in ("full", "intersect"), "feature_set must be full|intersect"
    cols = INTERSECT_COLS if feature_set == "intersect" else FEATURE_COLS
    data_dir, art = Path(data_dir), Path(artifacts)
    art.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open("config/thresholds.yaml"))

    customers = pd.read_csv(data_dir / "customers.csv")
    devices = pd.read_csv(data_dir / "devices.csv")
    txns = pd.read_csv(data_dir / "transactions.csv")
    if sample and len(txns) > sample:
        # stratified-ish: keep all fraud, sample normals
        fraud = txns[txns.is_fraud == 1]
        norm = txns[txns.is_fraud == 0].sample(n=sample - len(fraud), random_state=42)
        txns = pd.concat([fraud, norm]).sort_values("timestamp").reset_index(drop=True)

    feat = build_features(txns, customers, devices)
    X = feat[cols].to_numpy(dtype=float)
    y = feat["is_fraud"].to_numpy(dtype=int)
    tr = (feat["split"] == "train").to_numpy() if "split" in feat else np.arange(len(feat)) < int(0.8 * len(feat))
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[~tr], y[~tr]

    if HAS_XGB:
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.06,
                            subsample=0.9, colsample_bytree=0.8,
                            scale_pos_weight=max(1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())),
                            eval_metric="logloss", tree_method="hist", n_jobs=4)
    else:
        clf = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.06)
    clf.fit(Xtr, ytr)

    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    iso.fit(Xtr[ytr == 0][: min(20000, (ytr == 0).sum())])

    p_test = clf.predict_proba(Xte)[:, 1]
    # Calibrate on TRAIN normals only — no test leakage, same mapping as serving.
    calib_raw = -iso.score_samples(Xtr[ytr == 0][: min(20000, (ytr == 0).sum())])
    np.save(art / "anomaly_calib.npy", np.sort(np.asarray(calib_raw, dtype=float)))
    a_test = (-iso.score_samples(Xte))  # higher = more anomalous
    a_test_n = calibrate_anomaly(calib_raw, a_test)

    w = cfg["ensemble_weights"]
    final = w["classifier"] * p_test + w["anomaly"] * a_test_n  # graph added at serve time
    metrics = {
        "model": "xgboost" if HAS_XGB else "histgradientboosting",
        "feature_set": feature_set,
        "n_train": int(tr.sum()), "n_test": int((~tr).sum()),
        "auc": round(float(roc_auc_score(yte, p_test)) if len(np.unique(yte)) > 1 else 0.0, 4),
        "avg_precision": round(float(average_precision_score(yte, p_test)) if len(np.unique(yte)) > 1 else 0.0, 4),
        "precision@100": round(precision_at_k(yte, final, 100), 4),
        "recall@5%FPR": round(recall_at_fpr(yte, final, 0.05), 4),
        "fraud_rate_test": round(float(yte.mean()), 4),
    }
    joblib.dump(clf, art / "classifier.pkl")
    joblib.dump(iso, art / "anomaly.pkl")
    (art / "feature_cols.json").write_text(json.dumps(cols, indent=2))
    # Model-agnostic global importance (permutation on a small train sample) so
    # explanations always have a fallback — HGB exposes no feature_importances_
    # and SHAP may be absent on minimal installs.
    try:
        from sklearn.inspection import permutation_importance

        rng = np.random.default_rng(42)
        idx = rng.choice(len(Xtr), size=min(3000, len(Xtr)), replace=False)
        perm = permutation_importance(clf, Xtr[idx], ytr[idx], n_repeats=5,
                                      random_state=42, n_jobs=4)
        (art / "feature_importance.json").write_text(json.dumps(
            {c: round(float(v), 5) for c, v in zip(cols, perm.importances_mean)},
            indent=2))
    except Exception as e:
        print(f"warning: permutation importance skipped ({e})")
    # fraud address book for graph boost (from TRAIN only — no test leakage)
    fset = set(txns[tr & (txns.is_fraud == 1).to_numpy()][["sender", "receiver"]].stack().tolist())
    (art / "fraud_nodes.json").write_text(json.dumps(sorted(fset)))
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--feature-set", default="full", choices=["full", "intersect"])
    a = ap.parse_args()
    train(a.data, a.artifacts, a.sample, a.feature_set)
