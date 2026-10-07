"""Frozen-model zero-shot transfer: score a FOREIGN feed with SOURCE artifacts.

No training happens here — that is the point. The judge's question was
whether the *model* generalizes or only the *pipeline*: this freezes the
weights (classifier + anomaly calibration trained on the source feed) and
measures them untouched on foreign rows.

Two foreign feeds are supported:
  1. Adapted PaySim (needs the 500MB Kaggle CSV — manual download):
       python -m eval.paysim_adapter --in /tmp/paysim/PS_*.csv --out /tmp/paysim_vigil
       python -m eval.zeroshot --data /tmp/paysim_vigil \\
           --artifacts artifacts_intersect --out docs/paysim-zeroshot.json
  2. Fresh generator run, no download needed (different seed + fraud mix,
     i.e. a distribution the weights never saw):
       python -m data_gen.generate --seed 7 --fraud-rate 0.06 --out /tmp/foreign
       python -m eval.zeroshot --data /tmp/foreign \\
           --artifacts artifacts_intersect --out /tmp/foreign-zeroshot.json

Source artifacts must be portable ("intersect" feature set):
    python -m models.train --data data --artifacts artifacts_intersect \\
        --feature-set intersect

Honesty notes (also stated in docs/paysim-validation.md):
  - Graph boost uses the TARGET feed's own train labels (transductive helper,
    same as eval/evaluate.py). The frozen part is the ML weights + anomaly
    calibration; no refit happens anywhere in this module.
  - Simulator-to-simulator transfer evidences mechanism generality
    (bursts + fan-in + first-time large transfers), not real-world performance.
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

from eval.evaluate import precision_at_k, recall_at_fpr
from features.build import build_features
from models import infer
from models.graph import build_graph, fraud_nodes, network_risk


def threshold_transfer(y: np.ndarray, scores: np.ndarray, cfg: dict) -> dict:
    """Do the frozen operating bands still mean anything on foreign data?"""
    out = {}
    for name, thr in (("medium@0.60", cfg["bands"]["medium"]),
                      ("high@0.85", cfg["bands"]["high"])):
        flag = scores >= thr
        tp = int(((y == 1) & flag).sum())
        fp = int(((y == 0) & flag).sum())
        fn = int(((y == 1) & ~flag).sum())
        out[name] = {
            "flagged": int(flag.sum()),
            "precision": round(tp / max(1, tp + fp), 4),
            "recall": round(tp / max(1, tp + fn), 4),
            "fpr": round(fp / max(1, int((y == 0).sum())), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="FOREIGN feed dir (Vigil schema)")
    ap.add_argument("--artifacts", required=True, help="FROZEN source artifacts (read-only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=20000)
    a = ap.parse_args()

    infer.load_artifacts(a.artifacts)
    cols = list(infer._cols)
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
    t0 = time.perf_counter()
    X_mat = te[cols].to_numpy(dtype=float)
    cfg_graph = cfg.get("graph", {})
    boosts = np.array([
        network_risk(recv, g, fraud,
                     k_threshold=cfg_graph.get("two_hop_fraud_neighbors_threshold", 3),
                     boost_per_hit=cfg_graph.get("boost_per_hit", 0.15),
                     max_boost=cfg_graph.get("max_boost", 0.30))["boost"]
        for recv in te["receiver"].tolist()
    ], dtype=float)
    scores = infer.score_batch(X_mat, boosts, cfg["ensemble_weights"])

    res = {
        "protocol": "frozen-zero-shot (no training; weights + anomaly calibration from source)",
        "source_artifacts": str(a.artifacts),
        "source_feature_set": json.loads(Path(a.artifacts, "feature_cols.json").read_text()),
        "target_feed": str(a.data),
        "n_test": len(te), "fraud_rate_test": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y, scores)), 4),
        "precision@100": round(precision_at_k(y, scores), 4),
        "recall@5%FPR": round(recall_at_fpr(y, scores), 4),
        "threshold_transfer": threshold_transfer(y, scores, cfg),
        "eval_latency_total_s": round(time.perf_counter() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
