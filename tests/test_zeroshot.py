"""Zero-shot harness contract: frozen model scores shifted data, no retrain."""
import json
import subprocess
import sys
from pathlib import Path


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
