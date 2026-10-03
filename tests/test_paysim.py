"""PaySim adapter contract: mapping, leakage exclusions, split integrity.

Uses a tiny synthetic PaySim-schema fixture (NOT real PaySim) so no 500MB
download is needed. Proves the adapter never touches balance columns and
preserves fraud through the mapping.
"""
import pandas as pd


def _fixture(path, n=300, seed=3):
    rng = __import__("numpy").random.default_rng(seed)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], size=n, p=[0.4, 0.4, 0.2])
    fraud = (types != "PAYMENT") & (rng.random(n) < 0.1)
    df = pd.DataFrame({
        "step": rng.integers(1, 100, size=n),
        "type": types,
        "amount": rng.uniform(100, 50000, size=n).round(2),
        "nameOrig": [f"C{i % 50}" for i in range(n)],
        "nameDest": [f"C{(i + 7) % 50}" for i in range(n)],
        # Leakage trap values: perfect fraud predictors the adapter must ignore.
        "oldbalanceOrg": [1000.0 if f else 5000.0 for f in fraud],
        "newbalanceOrig": [0.0 if f else 4000.0 for f in fraud],
        "oldbalanceDest": [0.0] * n,
        "newbalanceDest": [1000.0 if f else 0.0 for f in fraud],
        "isFraud": fraud.astype(int),
        "isFlaggedFraud": [0] * n,
    })
    p = path / "mini_paysim.csv"
    df.to_csv(p, index=False)
    return df


def test_adapter_mapping_and_exclusions(tmp_path):
    from eval.paysim_adapter import adapt
    raw = _fixture(tmp_path)
    rep = adapt(tmp_path / "mini_paysim.csv", tmp_path / "out", max_normals=1000)
    txns = pd.read_csv(tmp_path / "out" / "transactions.csv")
    # No balance/rule-label column survives, whatever the fixture contains.
    assert not any("balance" in c.lower() or "flagged" in c.lower() for c in txns.columns)
    # All fraud preserved; types mapped; chronological split on both sides.
    assert txns.is_fraud.sum() == raw.isFraud.sum()
    assert set(txns.type.unique()) <= {"P2P", "cash-out", "merchant"}
    assert set(txns["split"].unique()) == {"train", "test"}
    assert set(rep["excluded_leakage_cols"]) >= {"oldbalanceOrg", "isFlaggedFraud"}
    # Unavailable signals are constants, not noise.
    assert txns.device_id.str.startswith("D_").all()
    assert (txns.location == "Dhaka").all()
    assert (txns.password_reset_flag == 0).all()
