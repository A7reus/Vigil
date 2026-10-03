"""Intensive unit coverage for load-bearing logic tested only indirectly so far.

Eval math, graph thresholds, feature causality (no future leakage), data-gen
statistics, store caps, rate-limiter buckets, middleware edges, API latency.
"""
from datetime import datetime

import numpy as np


# --- eval math ---------------------------------------------------------------

def test_precision_and_recall_helpers():
    from models.train import precision_at_k, recall_at_fpr
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precision_at_k(y, s, k=2) == 1.0
    assert precision_at_k(y, s, k=4) == 0.5
    y2 = np.array([0] * 90 + [1] * 10)
    s2 = np.array(list(np.linspace(0.01, 0.5, 90)) + [0.9] * 10)
    assert recall_at_fpr(y2, s2, fpr=0.05) == 1.0


def test_rule_baseline_bounded():
    import pandas as pd
    from eval.evaluate import rule_baseline
    n = 50
    feat = pd.DataFrame({
        "amount": np.linspace(1000, 100000, n),
        "new_device": [0, 1] * 25,
        "amount_vs_user_avg": [1.0, 5.0] * 25,
        "sender_cnt_1h": [0, 4] * 25,
        "fan_in_flag": [0, 1] * 25,
        "password_reset_flag": [0, 0] * 25,
    })
    s = rule_baseline(feat)
    assert len(s) == n and bool(((s >= 0) & (s <= 1)).all())
    assert s[1] > s[0]  # stacked signals outscore a plain txn


def test_business_sim_arithmetic():
    import pandas as pd
    from eval.evaluate import business_sim
    te = pd.DataFrame({"is_fraud": [1, 1, 0, 0], "amount": [100.0, 200.0, 50.0, 50.0]})
    out = business_sim(te, np.array([0.9, 0.8, 0.7, 0.1]), {})
    assert out["holds"] == 4 and out["precision"] == 0.5
    assert out["loss_prevented_bdt"] == 300.0
    assert out["analyst_minutes"] == 8 and out["minutes_saved_vs_manual"] == 52


# --- graph thresholds ---------------------------------------------------------

def _graph_case(n_fraud):
    """R - A - {F_i} ring where intermediaries stay clean: F_i touch A through
    clean edges and are tainted only by separate fraud edges F_i -> X_i.
    (Any fraud edge would taint A too — production semantics, tested below.)"""
    import pandas as pd
    from models.graph import build_graph, network_risk
    rows = [{"sender": "A", "receiver": "R", "amount": 100.0, "is_fraud": 0}]
    for i in range(n_fraud):
        f = f"F{i}"
        rows.append({"sender": f, "receiver": "A", "amount": 100.0, "is_fraud": 0})
        rows.append({"sender": f, "receiver": f"X{i}", "amount": 100.0, "is_fraud": 1})
    txns = pd.DataFrame(rows)
    g = build_graph(txns)
    fraud = set(txns[txns.is_fraud == 1][["sender", "receiver"]].stack())
    return network_risk("R", g, fraud)


def test_graph_threshold_and_cap():
    assert _graph_case(1)["boost"] == 0.0  # below K=3, not direct
    mid = _graph_case(2)
    assert mid["fraud_neighbors_2hop"] == 2 and mid["boost"] == 0.0
    full = _graph_case(5)
    assert full["fraud_neighbors_2hop"] == 5
    assert full["boost"] == 0.30  # capped at max_boost
    assert set(full["fraud_neighbor_sample"]) <= {f"F{i}" for i in range(5)}


def test_graph_intermediary_taint_is_intended():
    """A middle node touched by a fraud edge is itself fraud-proximate and
    earns partial credit — documents why collectors light up the ring."""
    import pandas as pd
    from models.graph import build_graph, network_risk
    txns = pd.DataFrame([{"sender": "F1", "receiver": "A", "amount": 1.0, "is_fraud": 1},
                         {"sender": "A", "receiver": "R", "amount": 1.0, "is_fraud": 0}])
    fraud = set(txns[txns.is_fraud == 1][["sender", "receiver"]].stack())
    out = network_risk("R", build_graph(txns), fraud)
    assert out["is_direct_fraud_neighbor"] is True and out["boost"] == 0.075


def test_graph_direct_neighbor_partial_credit():
    import pandas as pd
    from models.graph import build_graph, network_risk
    txns = pd.DataFrame([{"sender": "F1", "receiver": "R", "amount": 1.0, "is_fraud": 1}])
    out = network_risk("R", build_graph(txns), {"F1"})
    assert out["is_direct_fraud_neighbor"] is True
    assert out["boost"] == 0.075  # boost_per_hit / 2


def test_graph_unknown_receiver():
    import networkx as nx
    from models.graph import network_risk
    assert network_risk("GHOST", nx.DiGraph(), set())["boost"] == 0.0


# --- data-gen statistics + feature causality ----------------------------------

def test_data_gen_statistics():
    from data_gen.generate import generate
    _, _, txns = generate(n_customers=200, n_txns=2000, seed=11)
    rate = txns.is_fraud.mean()
    assert 0.02 <= rate <= 0.08, rate
    assert set(txns.fraud_type.unique()) >= {"none", "scam", "ato", "mule"}
    assert txns.txn_id.is_unique and txns.timestamp.is_monotonic_increasing


def test_features_use_strictly_past_only():
    """No-leakage property: a row's 1h velocity must equal a manual recount of
    strictly-earlier same-sender txns inside the window."""
    from data_gen.generate import generate
    from features.build import FEATURE_COLS, build_features
    customers, devices, txns = generate(n_customers=200, n_txns=2000, seed=11)
    feat = build_features(txns, customers, devices)
    assert set(FEATURE_COLS) <= set(feat.columns)
    for i in [100, 500, 1000, 1500]:
        row = feat.iloc[i]
        ts = datetime.fromisoformat(row["timestamp"])
        prior = txns[(txns.sender == row["sender"]) &
                     (txns.timestamp < row["timestamp"])]
        expect = sum(1 for t in prior["timestamp"]
                     if (ts - datetime.fromisoformat(t)).total_seconds() <= 3600)
        assert row["sender_cnt_1h"] == expect, (i, row["sender_cnt_1h"], expect)


# --- store caps + rate limiter units ------------------------------------------

def test_live_buffer_capped(tmp_path):
    import api.store as store_mod
    from api.store import HistoryStore
    monkey_max = store_mod.MAX_LIVE
    store_mod.MAX_LIVE = 5
    try:
        s = HistoryStore(tmp_path)  # empty history, no CSVs needed
        for i in range(9):
            s.append_live({"txn_id": f"L{i}", "sender": "A", "receiver": "B",
                           "amount": 1.0, "timestamp": "2026-08-15T12:00:00",
                           "location": "Dhaka"})
        assert len(s.live_rows) == 5 and s.live_rows[0]["txn_id"] == "L4"
        assert len(s.txns) == 0  # scoring state untouched
    finally:
        store_mod.MAX_LIVE = monkey_max


def test_timeline_merges_and_sorts(tmp_path):
    from api.store import HistoryStore
    s = HistoryStore(tmp_path)
    s.append_live({"txn_id": "L1", "sender": "A", "receiver": "B", "amount": 2.0,
                   "timestamp": "2026-08-15T13:00:00", "location": "Dhaka"})
    s.append_live({"txn_id": "L2", "sender": "C", "receiver": "A", "amount": 3.0,
                   "timestamp": "2026-08-15T11:00:00", "location": "Dhaka"})
    tl = s.timeline_for("A", n=10)
    assert [t["txn_id"] for t in tl] == ["L2", "L1"]
    assert all(set(t) >= {"txn_id", "sender", "receiver", "amount", "timestamp", "location"}
               for t in tl)


def test_rate_limiter_bucket(monkeypatch):
    import api.main as main
    main._rate_hits.clear()
    monkeypatch.setenv("VIGIL_RATE_LIMIT_PER_MIN", "2")
    assert main._rate_limit_ok("ip") and main._rate_limit_ok("ip")
    assert not main._rate_limit_ok("ip")
    monkeypatch.setenv("VIGIL_RATE_LIMIT_PER_MIN", "0")
    assert main._rate_limit_ok("ip")  # 0 disables
    main._rate_hits.clear()


# --- middleware + latency edges -------------------------------------------------

def test_middleware_edges():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        # Non-JSON content type never reaches validation as JSON.
        r = c.post("/score", content=b"hello",
                   headers={"Content-Type": "text/plain"})
        assert r.status_code == 422
        # Empty JSON body is a clean 422, not a 500.
        r = c.post("/score", content=b"",
                   headers={"Content-Type": "application/json"})
        assert r.status_code == 422


def test_sequential_scoring_latency():
    import time
    from fastapi.testclient import TestClient
    from api.main import app
    body = {"sender_id": "LAT", "receiver_id": "R", "amount": 5000, "channel": "app",
            "device_id": "D", "location": "Dhaka", "timestamp": "2026-08-15T12:00:00",
            "type": "P2P"}
    with TestClient(app, raise_server_exceptions=False) as c:
        lats = []
        for _ in range(10):
            t0 = time.perf_counter()
            r = c.post("/score", json=body)
            lats.append((time.perf_counter() - t0) * 1000)
            assert r.status_code == 200
        assert float(np.percentile(lats, 95)) < 5000  # generous incl. TestClient overhead
