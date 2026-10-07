"""Zero-shot transfer contract: frozen weights score foreign feeds, no refit.

Uses two tiny generator runs (different seeds + fraud mix) as source/foreign
feeds — no 500MB download. Proves the machinery: intersect training, subset
artifacts loading, and frozen scoring with threshold transfer.
"""
import json
import sys

import pandas as pd


def _tiny_feed(path, seed, fraud_rate, n_customers=120, n_txns=800):
    from data_gen.generate import generate

    path.mkdir(parents=True, exist_ok=True)
    customers, devices, txns = generate(n_customers, n_txns, fraud_rate, seed=seed)
    ts = pd.to_datetime(txns.timestamp)
    txns["split"] = (ts > ts.quantile(0.8)).map({True: "test", False: "train"})
    customers.to_csv(path / "customers.csv", index=False)
    devices.to_csv(path / "devices.csv", index=False)
    txns.to_csv(path / "transactions.csv", index=False)


def test_intersect_excludes_unavailable_signals():
    from features.build import FEATURE_COLS, INTERSECT_COLS, PORTABLE_EXCLUDED

    assert PORTABLE_EXCLUDED == {"new_device", "location_jump", "location_new", "password_reset_flag"}
    assert all(c not in INTERSECT_COLS for c in PORTABLE_EXCLUDED)
    assert set(INTERSECT_COLS) < set(FEATURE_COLS)
    assert len(INTERSECT_COLS) >= 15  # still a real model, not a stub


def test_frozen_weights_score_foreign_feed(tmp_path, monkeypatch):
    from models import infer
    from models.train import train

    src, foreign = tmp_path / "src", tmp_path / "foreign"
    _tiny_feed(src, seed=11, fraud_rate=0.08)
    _tiny_feed(foreign, seed=77, fraud_rate=0.05)
    art = tmp_path / "art"
    train(str(src), str(art), feature_set="intersect")
    assert json.loads((art / "feature_cols.json").read_text()) == __import__(
        "features.build", fromlist=["INTERSECT_COLS"]).INTERSECT_COLS

    # subset artifacts must load (old assert demanded exact FEATURE_COLS match)
    infer.load_artifacts(art)

    monkeypatch.setattr(sys, "argv", ["zeroshot", "--data", str(foreign),
                                      "--artifacts", str(art),
                                      "--out", str(tmp_path / "zs.json"),
                                      "--sample", "800"])
    from eval.zeroshot import main
    main()
    res = json.loads((tmp_path / "zs.json").read_text())
    assert res["protocol"].startswith("frozen-zero-shot")
    assert 0.0 <= res["auc"] <= 1.0
    assert res["auc"] > 0.5  # frozen weights beat chance on a foreign feed
    assert set(res["threshold_transfer"]) == {"medium@0.60", "high@0.85"}
    # artifacts untouched by scoring: still the source feature set
    assert json.loads((art / "feature_cols.json").read_text()) != \
        __import__("features.build", fromlist=["FEATURE_COLS"]).FEATURE_COLS
