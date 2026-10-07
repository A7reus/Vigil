"""Feedback loop: decisions become training rows, retrain, ship/hold verdict."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

needs_pg = pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason="needs TEST_POSTGRES_URL pointing at a throwaway database",
)


def _run(*args):
    subprocess.run([sys.executable, *args], check=True,
                   capture_output=True, text=True, timeout=600)


@needs_pg
def test_retrain_loop_end_to_end(tmp_path):
    import pandas as pd
    from api.decisions import DecisionLog
    from scripts.retrain import build_augmented, collect_labels, compare

    d, art = tmp_path / "d", tmp_path / "art"
    _run("-m", "data_gen.generate", "--seed", "42",
         "--n-customers", "400", "--n-txns", "4000", "--out", str(d))
    _run("-m", "models.train", "--data", str(d), "--artifacts", str(art))

    url = os.environ["TEST_POSTGRES_URL"]
    log = DecisionLog(url=url)
    log._db.execute("TRUNCATE decisions")
    txns = pd.read_csv(d / "transactions.csv")
    t1, t2 = txns["txn_id"].iloc[:2].tolist()
    snap = {"sender_id": "CX", "receiver_id": "CY", "amount": 100.0,
            "timestamp": "2026-08-15T12:00:00", "type": "P2P"}
    log.upsert({"txn_id": t1, "decision": "freeze", "analyst": "rt",
                "note": "", "at": "2026-08-15T12:00:00+00:00",
                "txn": json.dumps(snap)})
    log.upsert({"txn_id": t2, "decision": "allow", "analyst": "rt",
                "note": "", "at": "2026-08-15T12:00:00+00:00",
                "txn": json.dumps(snap)})
    log.upsert({"txn_id": "LIVE-abc", "decision": "freeze", "analyst": "rt",
                "note": "", "at": "2026-08-15T12:00:00+00:00",
                "txn": json.dumps({**snap, "channel": "app"})})

    labels = collect_labels(url)
    assert len(labels) == 3
    stats = build_augmented(d, labels, tmp_path / "aug")
    assert (stats["joined"], stats["rebuilt"]) == (2, 1), stats
    aug = pd.read_csv(tmp_path / "aug" / "transactions.csv")
    assert int(aug[aug.txn_id == t1].iloc[0].is_fraud) == 1
    assert set(aug[aug.txn_id == t1].split) == {"train"}
    assert "LIVE-abc" in set(aug["txn_id"])

    from models.train import train
    new_metrics = train(tmp_path / "aug", tmp_path / "art2")
    old_metrics = json.loads(Path(art, "metrics.json").read_text())
    verdict = compare(old_metrics, new_metrics)
    assert verdict["verdict"] in ("SHIP", "HOLD") and "delta" in verdict
