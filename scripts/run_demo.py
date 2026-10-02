"""End-to-end demo: small data -> train -> score two txns (normal vs scam)."""
import json
import tempfile
from pathlib import Path

import pandas as pd

from data_gen.generate import generate
from models.graph import build_graph, fraud_nodes, network_risk
from models.infer import load_artifacts, score_features
from models.train import train


def main():
    tmp = Path(tempfile.mkdtemp(prefix="vigil_demo_"))
    data_dir = tmp / "data"
    art_dir = tmp / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[demo] working dir: {tmp}")

    customers, devices, txns = generate(n_customers=400, n_txns=4000, seed=7)
    print(f"[demo] data: {len(txns)} txns, fraud={int(txns.is_fraud.sum())}")
    print(txns.groupby("fraud_type").size().to_dict())

    ts = pd.to_datetime(txns.timestamp)
    cut = ts.quantile(0.8)
    txns["split"] = (ts > cut).map({True: "test", False: "train"})
    customers.to_csv(data_dir / "customers.csv", index=False)
    devices.to_csv(data_dir / "devices.csv", index=False)
    txns.to_csv(data_dir / "transactions.csv", index=False)

    metrics = train(str(data_dir), str(art_dir))
    print("[demo] train metrics:", json.dumps(metrics, indent=2))
    load_artifacts(art_dir)

    g = build_graph(txns[txns.split == "train"])
    fraud = fraud_nodes(txns[(txns.split == "train") & (txns.is_fraud == 1)])

    # Normal-looking txn vs scam-looking txn (same API shape as POST /score).
    normal_txn = {
        "sender_id": customers.customer_id.iloc[0],
        "receiver_id": customers.customer_id.iloc[1],
        "amount": 1800, "channel": "app",
        "device_id": devices[devices.customer_id == customers.customer_id.iloc[0]].device_id.iloc[0],
        "location": customers[customers.customer_id == customers.customer_id.iloc[0]].district.iloc[0],
        "timestamp": "2026-08-15T14:10:00", "type": "merchant",
    }
    scam_txn = {
        "sender_id": customers.customer_id.iloc[2],
        "receiver_id": customers.customer_id.iloc[3],
        "amount": 45000, "channel": "app",
        "device_id": "DX999999", "location": "Dhaka",
        "timestamp": "2026-08-15T23:10:00", "type": "P2P",
    }
    from api.store import HistoryStore

    store = HistoryStore(str(data_dir))
    for name, txn in (("normal", normal_txn), ("scam-like", scam_txn)):
        feats = store.featurize(txn)
        gb = network_risk(txn["receiver_id"], g, fraud)["boost"]
        out = score_features(feats, gb)
        print(f"[demo] {name}: risk={out['risk_score']} p={out['p_fraud']} "
              f"anom={out['anomaly']} graph_boost={gb} reasons={out['top_3_reasons']}")
    print(f"[demo] artifacts kept at: {art_dir} (retrain with: python -m models.train)")


if __name__ == "__main__":
    main()
