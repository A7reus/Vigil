"""Generate synthetic upay-like MFS data with known injected fraud patterns.

Tables:
  customers(customer_id, age_group, district, account_age_days, avg_balance)
  devices(device_id, customer_id, first_seen)
  transactions(txn_id, sender, receiver, amount, type, timestamp,
               device_id, location, channel, is_fraud, fraud_type, password_reset_flag)

Patterns (documented assumptions):
  Normal: salary-in monthly -> small P2P/merchant/bill out, same device/location,
    daytime (7-22) mostly. With realistic noise: ~5% legit new-device (new phone),
    ~8% legit night txns, ~3% legit round amounts, ~1% legit password resets,
    ~5% legit large cash-outs, ~2% travel location changes, plus popular-merchant
    fan-in bursts (legit) so fan-in alone is not perfectly separable.
  Scam (~40% of fraud): small test credit then urgent large P2P out to NEW receiver,
    night (22-5) or pressure-hour, new device, new receiver. Hardened: ~20% use a
    known device, ~25% occur in daytime, ~30% use smaller overlapping amounts
    (3k-12k) to mimic normal spending.
  ATO (~25%): location jump (Dhaka<->Chattogram ~250km) within 10-60min + device
    change + velocity burst + usually password_reset_flag=1. Hardened: ~20% have
    no reset flag, ~15% use normal-sized amounts, ~10% are short 2-txn bursts.
  Mule (~20%): fan-in from ~10 wallets -> fan-out to 1 collector within 1h,
    usually round amounts (9900, 19500, 49500, ...). Hardened: ~25% use near-round
    off-by amounts to dodge exact-match rules.
  Agent anomaly (~15%): channel=cash-out via agent, volume 3x peer median.
    Hardened: ~30% use smaller overlapping amounts (25k-45k).

Split: chronological — last `test_frac` by timestamp is the clean test set (never train on it).
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DISTRICTS = ["Dhaka", "Chattogram", "Sylhet", "Khulna", "Rajshahi", "Barishal", "Rangpur", "Mymensingh"]
# Approx coords for location-jump distance (lat, lon) — coarse, synthetic only.
DISTRICT_COORDS = {
    "Dhaka": (23.81, 90.41),
    "Chattogram": (22.35, 91.83),
    "Sylhet": (24.90, 91.87),
    "Khulna": (22.82, 89.55),
    "Rajshahi": (24.37, 88.60),
    "Barishal": (22.70, 90.37),
    "Rangpur": (25.75, 89.27),
    "Mymensingh": (24.75, 90.40),
}
AGE_GROUPS = ["18-25", "26-35", "36-50", "50+"]
TXN_TYPES = ["P2P", "merchant", "bill", "cash-out", "salary-in"]
CHANNELS = ["app", "ussd", "agent"]
ROUND_MULE_AMOUNTS = [9900, 19500, 19900, 29500, 49500, 49900]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gen_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    districts = rng.choice(DISTRICTS, size=n, p=[0.35, 0.15, 0.1, 0.1, 0.1, 0.07, 0.07, 0.06])
    ages = rng.choice(AGE_GROUPS, size=n, p=[0.3, 0.35, 0.25, 0.1])
    account_age = rng.integers(30, 1500, size=n)
    avg_balance = np.round(rng.lognormal(mean=8.5, sigma=1.0, size=n), 2)  # ~5k median
    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n)],
        "age_group": ages,
        "district": districts,
        "account_age_days": account_age,
        "avg_balance": avg_balance,
    })


def gen_devices(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    dev_n = 0
    base = datetime(2024, 1, 1)
    for _, c in customers.iterrows():
        for _ in range(int(rng.choice([1, 1, 1, 2]))):
            rows.append({
                "device_id": f"D{dev_n:06d}",
                "customer_id": c["customer_id"],
                "first_seen": (base + timedelta(days=int(rng.integers(0, 600)))).isoformat(),
            })
            dev_n += 1
    return pd.DataFrame(rows)


def _daytime_hour(rng) -> int:
    return int(rng.choice(list(range(7, 23)), p=np.ones(16) / 16))


def _night_hour(rng) -> int:
    return int(rng.choice([22, 23, 0, 1, 2, 3, 4, 5]))


def generate(n_customers: int = 5000, n_txns: int = 50000, fraud_rate: float = 0.04,
             days: int = 30, seed: int = 42):
    rng = _rng(seed)
    random.seed(seed)
    customers = gen_customers(n_customers, rng)
    devices = gen_devices(customers, rng)
    dev_by_cust = devices.groupby("customer_id")["device_id"].apply(list).to_dict()
    cust_district = dict(zip(customers.customer_id, customers.district))

    start = datetime(2026, 8, 1, 0, 0, 0)
    n_fraud = int(n_txns * fraud_rate)
    n_normal = n_txns - n_fraud

    txns = []
    tid = 0

    def add(sender, receiver, amount, ttype, ts, device, loc, channel, is_fraud=0, ftype="none", pwd=0):
        nonlocal tid
        txns.append({
            "txn_id": f"T{tid:07d}", "sender": sender, "receiver": receiver,
            "amount": round(float(amount), 2), "type": ttype,
            "timestamp": ts.isoformat(), "device_id": device, "location": loc,
            "channel": channel, "is_fraud": is_fraud, "fraud_type": ftype,
            "password_reset_flag": pwd,
        })
        tid += 1

    cust_ids = customers.customer_id.tolist()

    # ---- Normal traffic (with realistic noise so fraud is not trivially separable) ----
    for _ in range(n_normal):
        s = random.choice(cust_ids)
        r = random.choice(cust_ids)
        if r == s:
            continue
        ttype = random.choices(["P2P", "merchant", "bill", "cash-out", "salary-in"],
                               weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]
        if ttype == "salary-in":
            amount = float(rng.choice([20000, 30000, 50000, 80000]) + rng.integers(-2000, 2000))
        elif ttype == "cash-out" and rng.random() < 0.05:
            # legit large business cash-out: overlaps agent-fraud amounts
            amount = round(float(rng.choice([30000, 40000, 50000, 60000]) + rng.integers(-3000, 3000)), 2)
        else:
            amount = round(float(rng.lognormal(7.5, 0.9)), 2)  # ~1.8k median
            amount = min(amount, 20000)
            if rng.random() < 0.03:
                # legit round payment (e.g. 9,900 merchant bill): overlaps mule rule
                amount = float(random.choice(ROUND_MULE_AMOUNTS))
        day = int(rng.integers(0, days))
        if rng.random() < 0.08:
            hour = _night_hour(rng)  # legit late-night txn
        else:
            hour = _daytime_hour(rng)
        ts = start + timedelta(days=day, hours=hour, minutes=int(rng.integers(0, 60)))
        if rng.random() < 0.05:
            dev = f"DXL{tid:06d}"  # legit new phone, not fraud
        else:
            dev = random.choice(dev_by_cust[s])
        if rng.random() < 0.02:
            # legit travel: different district but normal behavior
            loc = random.choice([d for d in DISTRICTS if d != cust_district[s]])
        else:
            loc = cust_district[s]
        ch = random.choices(["app", "ussd", "agent"], weights=[0.7, 0.2, 0.1])[0]
        pwd = 1 if rng.random() < 0.01 else 0  # legit password reset, no fraud
        add(s, r, amount, ttype, ts, dev, loc, ch, pwd=pwd)

    # ---- Legit popular-merchant fan-in (non-fraud bursts so fan-in is not perfect) ----
    n_merchants = max(3, n_normal // 10000)
    for _ in range(n_merchants):
        merchant = random.choice(cust_ids)
        burst_day = int(rng.integers(0, days))
        burst_t = start + timedelta(days=burst_day, hours=int(rng.integers(11, 20)))
        for _ in range(int(rng.integers(15, 30))):
            s = random.choice(cust_ids)
            if s == merchant:
                continue
            amount = round(float(rng.lognormal(7.5, 0.9)), 2)
            amount = min(amount, 20000)
            ts = burst_t + timedelta(minutes=int(rng.integers(0, 60)))
            dev = random.choice(dev_by_cust[s])
            add(s, merchant, amount, "merchant", ts, dev, cust_district[s], "app")

    # ---- Injected fraud (budget by TXN count so mule fan-out doesn't inflate rate) ----
    target_fraud_txns = n_fraud
    fraud_txns = 0
    # type mix per fraud *event*
    type_cycle = ["scam", "ato", "mule", "agent"]
    weights = [0.4, 0.25, 0.2, 0.15]

    # Pre-pick mule collector rings: groups of 10 senders -> 1 collector
    while fraud_txns < target_fraud_txns:
        ft = random.choices(type_cycle, weights=weights)[0]
        day = int(rng.integers(0, days))
        if ft == "scam":
            s = random.choice(cust_ids)
            r = random.choice(cust_ids)
            # Hardened: 30% small overlapping amounts, 25% daytime, 20% known device
            if rng.random() < 0.30:
                amount = round(float(rng.lognormal(8.0, 0.7)), 2)  # 3k-12k overlaps normal
                amount = min(max(amount, 2000), 15000)
            else:
                amount = float(rng.choice([15000, 25000, 45000, 80000]))
            if rng.random() < 0.25:
                ts = start + timedelta(days=day, hours=_daytime_hour(rng),
                                       minutes=int(rng.integers(0, 60)))
            else:
                ts = start + timedelta(days=day, hours=_night_hour(rng), minutes=int(rng.integers(0, 60)))
            if rng.random() < 0.20:
                new_dev = random.choice(dev_by_cust[s])  # compromised known device
            else:
                new_dev = f"DX{tid:06d}"  # brand-new device not in registry
            add(s, r, amount, "P2P", ts, new_dev, cust_district[s], "app", 1, "scam")
            fraud_txns += 1
        elif ft == "ato":
            s = random.choice(cust_ids)
            home = cust_district[s]
            jump = "Chattogram" if home == "Dhaka" else "Dhaka"
            # Hardened: 10% short 2-txn burst, else 3-5 rapid txns
            burst_len = 2 if rng.random() < 0.10 else int(rng.integers(3, 6))
            no_pwd = rng.random() < 0.20  # 20% without reset flag
            small_amt = rng.random() < 0.15  # 15% normal-sized amounts
            burst_t = start + timedelta(days=day, hours=int(rng.integers(0, 24)), minutes=0)
            for k in range(burst_len):
                r = random.choice(cust_ids)
                if small_amt:
                    amount = round(float(rng.lognormal(7.5, 0.9)), 2)
                else:
                    amount = float(rng.lognormal(9.0, 0.5))  # ~8k, 5x normal velocity/size
                ts = burst_t + timedelta(minutes=int(k * rng.integers(2, 10)))
                add(s, r, amount, "P2P", ts, f"DX{tid:06d}", jump if k > 0 else home,
                    "app", 1, "ato", pwd=0 if no_pwd else (1 if k == 0 else 0))
                fraud_txns += 1
        elif ft == "mule":
            collectors = random.sample(cust_ids, 1)
            fanin = random.sample([c for c in cust_ids if c not in collectors], 10)
            collector = collectors[0]
            burst_t = start + timedelta(days=day, hours=int(rng.integers(10, 20)))
            for s in fanin:
                if rng.random() < 0.25:
                    # near-round off-by amount to dodge exact-match rules
                    base = float(random.choice(ROUND_MULE_AMOUNTS))
                    amount = round(base + float(rng.integers(-600, 600)), 2)
                    if int(round(amount)) in set(ROUND_MULE_AMOUNTS):
                        amount += 37.0
                else:
                    amount = float(random.choice(ROUND_MULE_AMOUNTS))
                ts = burst_t + timedelta(minutes=int(rng.integers(0, 60)))
                add(s, collector, amount, "P2P", ts, random.choice(dev_by_cust[s]),
                    cust_district[s], "app", 1, "mule")
                fraud_txns += 1
                if fraud_txns >= target_fraud_txns:
                    break
        else:  # agent anomaly: big cash-outs 3x peer median (hardened: 30% smaller overlap)
            s = random.choice(cust_ids)
            r = random.choice(cust_ids)
            if rng.random() < 0.30:
                amount = round(float(rng.choice([25000, 35000, 45000]) + rng.integers(-3000, 3000)), 2)
            else:
                amount = float(rng.choice([60000, 90000, 120000]))
            ts = start + timedelta(days=day, hours=_daytime_hour(rng))
            add(s, r, amount, "cash-out", ts, random.choice(dev_by_cust[s]),
                cust_district[s], "agent", 1, "agent")
            fraud_txns += 1

    df = pd.DataFrame(txns).sort_values("timestamp").reset_index(drop=True)
    # reassign txn ids in time order
    df["txn_id"] = [f"T{i:07d}" for i in range(len(df))]
    return customers, devices, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-customers", type=int, default=5000)
    ap.add_argument("--n-txns", type=int, default=50000)
    ap.add_argument("--fraud-rate", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    customers, devices, txns = generate(args.n_customers, args.n_txns, args.fraud_rate, seed=args.seed)
    # chronological split markers (parse ISO strings first)
    ts = pd.to_datetime(txns.timestamp)
    cut = ts.quantile(1 - args.test_frac)
    txns["split"] = (ts > cut).map({True: "test", False: "train"})
    customers.to_csv(out / "customers.csv", index=False)
    devices.to_csv(out / "devices.csv", index=False)
    txns.to_csv(out / "transactions.csv", index=False)
    print(f"wrote {len(customers)} customers, {len(devices)} devices, {len(txns)} txns -> {out}")
    print(txns.groupby(["split", "is_fraud"]).size())
    print(txns.groupby("fraud_type").size())


if __name__ == "__main__":
    main()
