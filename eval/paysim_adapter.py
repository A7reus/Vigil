"""PaySim -> Vigil schema adapter for OFFLINE cross-dataset validation.

The Vigil pipeline trains on our own generator; this answers "does it also work
on data we didn't create?" using PaySim (mobile-money simulator calibrated on
real MFS logs: https://github.com/EdgarLopezPhD/PaySim, Kaggle ealaxi/paysim1).

Deliberately NOT wired into the API: PaySim lacks our richest signals (devices,
locations, password resets), so it validates generalization instead of
replacing training data.

Leakage discipline (the famous PaySim trap): fraud rows are annulled, so
oldbalance*/newbalance* perfectly reveal the label and isFlaggedFraud is a
deterministic rule label. ALL of them are excluded — using them would be
cheating, and would prove nothing.

Mapping:
  TRANSFER -> P2P, CASH_OUT -> cash-out, PAYMENT -> merchant,
  CASH_IN -> salary-in, DEBIT -> bill. CASH_* channel -> agent, else app.
  step (hours) -> timestamp from 2026-08-01. Devices/locations are synthesized
  as constants, so device/location signals read 0 (documented as unavailable).

Usage:
  kaggle datasets download -d ealaxi/paysim1 -p /tmp/paysim && unzip -o /tmp/paysim/paysim1.zip -d /tmp/paysim
  python -m eval.paysim_adapter --in /tmp/paysim/PS_20174392719_1491204439457_log.csv --out /tmp/paysim_vigil
  python -m models.train --data /tmp/paysim_vigil --artifacts /tmp/paysim_art
  python -m eval.evaluate --data /tmp/paysim_vigil --artifacts /tmp/paysim_art --out docs/paysim-eval.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

TYPE_MAP = {"TRANSFER": "P2P", "CASH_OUT": "cash-out", "PAYMENT": "merchant",
            "CASH_IN": "salary-in", "DEBIT": "bill"}
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]  # PaySim fraud lives only here
BASE = datetime(2026, 8, 1)


def adapt(src: str | Path, out: str | Path, max_normals: int = 100_000,
          test_frac: float = 0.2, seed: int = 7) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src, usecols=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"])
    n_raw = len(df)
    df = df[df.type.isin(FRAUD_TYPES)].copy()
    fraud = df[df.isFraud == 1]
    norm = df[df.isFraud == 0].sample(n=min(max_normals, (df.isFraud == 0).sum()),
                                      random_state=seed)
    df = pd.concat([fraud, norm]).sort_values(["step", "amount"]).reset_index(drop=True)

    ts = [BASE + timedelta(hours=int(s)) for s in df["step"]]
    txns = pd.DataFrame({
        "txn_id": [f"P{i:07d}" for i in range(len(df))],
        "sender": df["nameOrig"], "receiver": df["nameDest"],
        "amount": df["amount"].round(2),
        "type": df["type"].map(TYPE_MAP),
        "timestamp": [t.isoformat() for t in ts],
        # No per-device/location/reset data in PaySim: stable constants so the
        # corresponding signals read 0 instead of noise.
        "device_id": "D_" + df["nameOrig"],
        "location": "Dhaka",
        "channel": df["type"].map(lambda t: "agent" if t.startswith("CASH") else "app"),
        "is_fraud": df["isFraud"].astype(int),
        "fraud_type": df["type"].map(lambda t: "mule" if t == "TRANSFER" else "agent"),
        "password_reset_flag": 0,
    })
    cut = pd.to_datetime(txns.timestamp).quantile(1 - test_frac)
    txns["split"] = (pd.to_datetime(txns.timestamp) > cut).map({True: "test", False: "train"})

    customers = pd.DataFrame({"customer_id": pd.unique(txns[["sender", "receiver"]].stack())})
    customers["age_group"] = "26-35"
    customers["district"] = "Dhaka"
    customers["account_age_days"] = 365
    customers["avg_balance"] = 5000.0
    devices = pd.DataFrame({"device_id": "D_" + customers["customer_id"],
                            "customer_id": customers["customer_id"]})

    customers.to_csv(out / "customers.csv", index=False)
    devices.to_csv(out / "devices.csv", index=False)
    txns.to_csv(out / "transactions.csv", index=False)
    report = {"raw_rows": n_raw, "kept_rows": len(txns),
              "fraud_kept": int(txns.is_fraud.sum()),
              "fraud_rate": round(float(txns.is_fraud.mean()), 5),
              "excluded_leakage_cols": ["oldbalanceOrg", "newbalanceOrig",
                                        "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"],
              "unavailable_signals_zeroed": ["new_device", "location_jump",
                                             "location_new", "password_reset_flag"],
              "type_map": TYPE_MAP}
    print(report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-normals", type=int, default=100_000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    adapt(a.src, a.out, a.max_normals, a.test_frac, a.seed)


if __name__ == "__main__":
    main()
