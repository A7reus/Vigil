"""Item 2: training/serving skew test.

Features exist twice (batch build_features vs online HistoryStore.featurize).
For sampled rows, rebuild online state from strictly-earlier transactions
only and assert every feature matches the batch value. Any drift here would
silently degrade real-world accuracy.
"""
import math

import pytest

from features.build import FEATURE_COLS


def _gen(tmp_path):
    from data_gen.generate import generate
    customers, devices, txns = generate(n_customers=400, n_txns=4000,
                                        fraud_rate=0.04, days=30, seed=11)
    txns = txns.sort_values("timestamp").reset_index(drop=True)
    return customers, devices, txns


def _write_prefix(tmp_path, customers, devices, past, tag):
    from api.store import HistoryStore
    d = tmp_path / f"pre{tag}"
    d.mkdir(exist_ok=True)
    customers.to_csv(d / "customers.csv", index=False)
    devices.to_csv(d / "devices.csv", index=False)
    past.to_csv(d / "transactions.csv", index=False)
    return HistoryStore(d)


@pytest.mark.slow
def test_batch_online_feature_parity(tmp_path):
    from features.build import build_features
    customers, devices, txns = _gen(tmp_path)
    feat = build_features(txns, customers, devices)
    # Batch order is authoritative (stable ties aside): match rows by txn_id
    # and define "past" exactly as the rows the batch processed before.
    order = feat["txn_id"].tolist()
    assert set(order) == set(txns["txn_id"])
    by_id = txns.set_index("txn_id")
    positions = list(range(200, len(txns), max(1, len(txns) // 40)))[:40]
    worst, checked = 0.0, 0
    for n, i in enumerate(positions):
        tid = order[i]
        row, brow = by_id.loc[tid], feat.iloc[i]
        assert brow["txn_id"] == tid
        past = txns[txns["txn_id"].isin(order[:i])]
        store = _write_prefix(tmp_path, customers, devices, past, n)
        online = store.featurize({
            "sender_id": row["sender"], "receiver_id": row["receiver"],
            "amount": float(row["amount"]), "channel": row["channel"],
            "device_id": row["device_id"], "location": row["location"],
            "timestamp": row["timestamp"], "type": row["type"],
            "password_reset_flag": int(row.get("password_reset_flag", 0))})
        for c in FEATURE_COLS:
            a, b = float(brow[c]), float(online[c])
            if isinstance(brow[c], (int,)) or c in (
                    "new_receiver", "new_device", "location_jump", "location_new",
                    "is_round_amount", "password_reset_flag", "fan_in_flag",
                    "cash_out_flag", "agent_channel_flag", "p2p_flag",
                    "unusual_hour", "is_night", "is_weekend"):
                assert a == b, (i, c, a, b)
            else:
                tol = 1e-6 * max(1.0, abs(a))
                assert abs(a - b) <= tol, (i, c, a, b)
                worst = max(worst, abs(a - b) / max(1.0, abs(a)))
            checked += 1
    assert checked == len(positions) * len(FEATURE_COLS) >= 30 * len(FEATURE_COLS)
    print(f"\nparity checked {checked} values, worst rel diff {worst:.2e}")


def test_batch_online_exact_on_handcrafted_stream(tmp_path):
    """Tiny deterministic stream where every window can be hand-verified."""
    import pandas as pd
    from features.build import build_features
    customers = pd.DataFrame([{"customer_id": "C1", "age_group": "26-35", "district": "Dhaka",
                               "account_age_days": 365, "avg_balance": 5000.0}])
    devices = pd.DataFrame([{"device_id": "D1", "customer_id": "C1",
                             "first_seen": "2024-01-01"}])
    base = "2026-08-10T10:00:00"
    txns = pd.DataFrame([
        {"txn_id": f"T{i:07d}", "sender": "C1", "receiver": f"R{i}", "amount": 1000.0 + i,
         "type": "P2P", "timestamp": f"2026-08-10T10:{i:02d}:00", "device_id": "D1",
         "location": "Dhaka", "channel": "app", "is_fraud": 0, "fraud_type": "none",
         "password_reset_flag": 0, "split": "train"}
        for i in range(5)
    ])
    feat = build_features(txns, customers, devices)
    last = feat.iloc[4]
    # 4 strictly-earlier same-sender txns inside 1h/24h windows
    assert last["sender_cnt_1h"] == 4 and last["sender_cnt_24h"] == 4
    assert last["sender_sum_1h"] == 1000 + 1001 + 1002 + 1003
    store = _write_prefix(tmp_path, customers, devices, txns.iloc[:4], "hand")
    online = store.featurize({"sender_id": "C1", "receiver_id": "R4", "amount": 1004.0,
                              "channel": "app", "device_id": "D1", "location": "Dhaka",
                              "timestamp": base.replace("10:00:00", "10:04:00"), "type": "P2P"})
    assert online["sender_cnt_1h"] == 4 and online["sender_sum_1h"] == 4006.0
    assert online["new_receiver"] == 1 and online["new_device"] == 0
