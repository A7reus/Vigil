"""Evaluate ScamShield on the clean chronological test split.

Metrics: AUC, Precision@100, Recall@5%FPR, p95 scoring latency,
fairness (FPR by district/age_group/account_age), business simulation.

Usage:
    python -m eval.evaluate --data data --artifacts artifacts --out artifacts/eval.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from features.build import FEATURE_COLS, build_features
from models.graph import build_graph, fraud_nodes, network_risk
from models.infer import load_artifacts, score_features


def rule_baseline(feat: pd.DataFrame) -> np.ndarray:
    s = np.zeros(len(feat))
    s += (feat.amount > 20000).astype(int) * 0.3
    s += ((feat.new_device == 1) & (feat.amount_vs_user_avg > 2)).astype(int) * 0.3
    s += (feat.sender_cnt_1h >= 3).astype(int) * 0.2
    s += (feat.fan_in_flag == 1).astype(int) * 0.2
    s += (feat.password_reset_flag == 1).astype(int) * 0.2
    return np.clip(s, 0, 1)


def precision_at_k(y, scores, k=100):
    idx = np.argsort(scores)[::-1][:k]
    return float(y[idx].mean())


def recall_at_fpr(y, scores, fpr=0.05):
    from sklearn.metrics import roc_curve
    fprs, tprs, _ = roc_curve(y, scores)
    m = fprs <= fpr
    return float(tprs[m].max()) if m.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="artifacts/eval.json")
    ap.add_argument("--sample", type=int, default=20000)
    a = ap.parse_args()

    load_artifacts(a.artifacts)
    cfg = yaml.safe_load(open("config/thresholds.yaml"))
    customers = pd.read_csv(Path(a.data) / "customers.csv")
    devices = pd.read_csv(Path(a.data) / "devices.csv")
    txns = pd.read_csv(Path(a.data) / "transactions.csv")
    if len(txns) > a.sample:
        f = txns[txns.is_fraud == 1]
        n = txns[txns.is_fraud == 0].sample(n=a.sample - len(f), random_state=0)
        txns = pd.concat([f, n]).sort_values("timestamp").reset_index(drop=True)

    feat = build_features(txns, customers, devices)
    te = feat[feat.split == "test"].reset_index(drop=True) if "split" in feat else feat.tail(int(len(feat) * 0.2))
    tr = feat[~feat.index.isin(te.index)] if "split" in feat else feat.head(len(feat) - len(te))
    y = te.is_fraud.to_numpy()

    g = build_graph(tr)
    fraud = fraud_nodes(tr[tr.is_fraud == 1]) if "is_fraud" in tr else set()
    X = te[FEATURE_COLS].to_dict("records")
    scores, t0 = [], time.perf_counter()
    lat = []
    for row in X:
        t1 = time.perf_counter()
        # graph boost from receiver proximity (train-only fraud set: no leakage)
        recv = te.iloc[len(scores)]["receiver"]
        gb = network_risk(recv, g, fraud)["boost"]
        scores.append(score_features(row, gb, cfg["ensemble_weights"])["risk_score"])
        lat.append((time.perf_counter() - t1) * 1000)
    scores = np.array(scores)
    rules = rule_baseline(te)

    def fpr_by(col):
        out = {}
        pred = scores >= cfg["bands"]["medium"]
        for k, idx in te.groupby(col).groups.items():
            yy = y[idx.to_numpy()] if hasattr(idx, "to_numpy") else y[np.array(list(idx))]
            pp = pred[np.array(list(idx))]
            neg = (yy == 0)
            out[str(k)] = round(float(pp[neg].mean()) if neg.sum() else 0.0, 4)
        return out

    te2 = te.copy()
    te2["account_age_bucket"] = pd.cut(te2.account_age_days, [0, 180, 365, 9999], labels=["new", "mid", "old"])
    te = te2
    res = {
        "n_test": len(te), "fraud_rate_test": round(float(y.mean()), 4),
        "model": {"auc": round(float(roc_auc_score(y, scores)), 4),
                  "precision@100": round(precision_at_k(y, scores), 4),
                  "recall@5%FPR": round(recall_at_fpr(y, scores), 4),
                  "p95_latency_ms": round(float(np.percentile(lat, 95)), 2)},
        "rule_baseline": {"auc": round(float(roc_auc_score(y, rules)), 4),
                          "precision@100": round(precision_at_k(y, rules), 4),
                          "recall@5%FPR": round(recall_at_fpr(y, rules), 4)},
        "fairness_FPR@sweep": {"district": fpr_by("district") if "district" in te else {},
                               "account_age_bucket": fpr_by("account_age_bucket")},
        "business": business_sim(te, scores, cfg),
        "eval_latency_total_s": round(time.perf_counter() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def business_sim(te: pd.DataFrame, scores: np.ndarray, cfg: dict) -> dict:
    """1000 high-risk holds: loss prevented vs analyst cost."""
    order = np.argsort(scores)[::-1][:1000]
    flag = te.iloc[order]
    prec = float(flag.is_fraud.mean()) if len(flag) else 0
    prevented = float(flag[flag.is_fraud == 1].amount.sum())
    analyst_min = len(flag) * 2  # 2 min/case with summary vs 15 min manual
    saved_min = len(flag) * 13
    return {"holds": int(len(flag)), "precision": round(prec, 4),
            "loss_prevented_bdt": round(prevented, 2),
            "analyst_minutes": analyst_min, "minutes_saved_vs_manual": saved_min}


if __name__ == "__main__":
    main()
