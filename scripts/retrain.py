"""Close the feedback loop: analyst decisions -> retrained model.

`POST /decision` rows are weak labels, stated plainly: freeze/step-up ~=
fraud, allow ~= legit. A 2-minute judgment is not ground truth, and reviewed
cases are all edge cases (selection bias) — so the full original data stays
as ballast and analyst rows are the top-up, never the whole diet. A random
review sample would balance it further; until then, ballast does the job.

Queued-alert decisions (`T...` ids) join `transactions.csv` directly; LIVE
playground scores rebuild from the `txn` snapshot column, since they never
entered committed history.

Usage:
    export DATABASE_URL=postgresql://vigil:vigil@localhost:5432/vigil
    python -m scripts.retrain --data data --artifacts artifacts \\
        --out-data /tmp/aug --out-artifacts /tmp/art2
    # prints old-vs-new metrics + SHIP/HOLD verdict; --apply copies winners over.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

LABEL = {"freeze": 1, "step-up": 1, "allow": 0}
TOL = 0.005  # metric regression tolerance for the SHIP verdict


def collect_labels(url: str | None = None) -> pd.DataFrame:
    from api.decisions import DecisionLog
    rows = DecisionLog(url=url).all()
    out = []
    for r in rows:
        if r.get("decision") not in LABEL:
            continue
        try:
            snap = json.loads(r.get("txn") or "{}")
        except (json.JSONDecodeError, TypeError):
            snap = {}
        out.append({"txn_id": r["txn_id"], "label": LABEL[r["decision"]], "snap": snap})
    return pd.DataFrame(out, columns=["txn_id", "label", "snap"])


def build_augmented(data_dir: str | Path, labels: pd.DataFrame, out_dir: str | Path) -> dict:
    """Original rows as ballast; analyst labels override/add as train rows."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txns = pd.read_csv(data_dir / "transactions.csv")
    for f in ("customers.csv", "devices.csv"):
        shutil.copy(data_dir / f, out_dir / f)
    joined, rebuilt, skipped = 0, 0, 0
    have = set(txns["txn_id"])
    new_rows = []
    for row in labels.itertuples():
        if row.txn_id in have:
            m = txns["txn_id"] == row.txn_id
            txns.loc[m, "is_fraud"] = row.label
            txns.loc[m, "split"] = "train"
            joined += 1
            continue
        s = row.snap if isinstance(row.snap, dict) else {}
        if not all(s.get(k) for k in ("sender_id", "receiver_id", "amount", "timestamp")):
            skipped += 1
            continue
        new_rows.append({
            "txn_id": row.txn_id, "sender": s["sender_id"], "receiver": s["receiver_id"],
            "amount": float(s["amount"]), "type": s.get("type", "P2P"),
            "timestamp": s["timestamp"], "device_id": s.get("device_id", "unknown"),
            "location": s.get("location", "Dhaka"), "channel": s.get("channel", "app"),
            "is_fraud": row.label, "fraud_type": "review",
            "password_reset_flag": int(s.get("password_reset_flag", 0) or 0),
            "split": "train"})
        rebuilt += 1
    if new_rows:
        txns = pd.concat([txns, pd.DataFrame(new_rows)], ignore_index=True)
    txns.to_csv(out_dir / "transactions.csv", index=False)
    return {"joined": joined, "rebuilt": rebuilt, "skipped": skipped,
            "n_labels": len(labels), "n_train_rows": int(len(txns))}


def compare(old: dict, new: dict) -> dict:
    keys = ("auc", "recall@5%FPR", "precision@100")
    delta = {k: round(new.get(k, 0) - old.get(k, 0), 4) for k in keys}
    ship = new.get("auc", 0) >= old.get("auc", 0) - TOL \
        and new.get("recall@5%FPR", 0) >= old.get("recall@5%FPR", 0) - TOL
    return {"verdict": "SHIP" if ship else "HOLD", "delta": delta,
            "old": {k: old.get(k) for k in keys}, "new": {k: new.get(k) for k in keys}}


def main(argv=None):
    from models.train import train
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out-data", default="/tmp/vigil_aug")
    ap.add_argument("--out-artifacts", default="/tmp/vigil_art2")
    ap.add_argument("--min-labels", type=int, default=1)
    ap.add_argument("--apply", action="store_true",
                    help="copy the new artifacts over --artifacts, SHIP verdict only")
    a = ap.parse_args(argv)
    labels = collect_labels()
    if len(labels) < a.min_labels:
        res = {"verdict": "NOTHING-TO-LEARN", "n_labels": len(labels)}
        print(json.dumps(res, indent=2))
        return res
    stats = build_augmented(a.data, labels, a.out_data)
    new_metrics = train(a.out_data, a.out_artifacts)
    old_metrics = json.loads(Path(a.artifacts, "metrics.json").read_text())
    res = {"n_labels": len(labels), "augment": stats, **compare(old_metrics, new_metrics)}
    if a.apply:
        if res["verdict"] != "SHIP":
            print(json.dumps(res, indent=2))
            print("refusing --apply on a HOLD verdict", file=sys.stderr)
            return res
        for f in Path(a.out_artifacts).glob("*"):
            shutil.copy(f, Path(a.artifacts) / f.name)
        res["applied"] = True
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
