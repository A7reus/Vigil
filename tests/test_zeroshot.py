"""Zero-shot transfer contract: frozen weights score foreign feeds, no refit.

Two coverages in one file: the intersect contract on tiny in-process feeds
(portable feature set, subset artifacts, threshold transfer) and the
full-artifact end-to-end run (frozen home model must beat rules on shifted data).
"""
import json
import subprocess
import sys
from pathlib import Path

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
    assert res["model"]["auc"] == res["auc"]
    assert "rule_baseline" in res  # model-vs-rules pairing stays in one artifact
    # artifacts untouched by scoring: still the source feature set
    assert json.loads((art / "feature_cols.json").read_text()) != \
        __import__("features.build", fromlist=["FEATURE_COLS"]).FEATURE_COLS


def _run(*args):
    subprocess.run([sys.executable, *args], check=True,
                   capture_output=True, text=True, timeout=600)


def test_zeroshot_beats_rules_on_shifted_seed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    art = tmp_path / "art"
    _run("-m", "data_gen.generate", "--seed", "42",
         "--n-customers", "400", "--n-txns", "4000", "--out", str(a))
    _run("-m", "data_gen.generate", "--seed", "7",
         "--n-customers", "400", "--n-txns", "4000", "--out", str(b))
    _run("-m", "models.train", "--data", str(a), "--artifacts", str(art))
    out = tmp_path / "zero.json"
    _run("-m", "eval.zeroshot", "--artifacts", str(art),
         "--data", str(b), "--out", str(out))
    res = json.loads(Path(out).read_text())
    assert res["mode"].startswith("zero-shot")
    assert res["model"]["auc"] > 0.7, res
    assert res["model"]["auc"] > res["rule_baseline"]["auc"], res
