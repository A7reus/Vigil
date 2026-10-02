"""In-memory history store: preloaded CSV history + newly scored txns.

Featurizes a single incoming txn by scanning history with vectorized pandas
filters (1h/24h windows). History capped to last N rows for p95 latency.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from features.build import ROUND_AMOUNTS

MAX_HISTORY = 60_000


class HistoryStore:
    def __init__(self, data_dir: str | Path = "data"):
        data_dir = Path(data_dir)
        self.customers = pd.read_csv(data_dir / "customers.csv") if (data_dir / "customers.csv").exists() else pd.DataFrame()
        self.devices = pd.read_csv(data_dir / "devices.csv") if (data_dir / "devices.csv").exists() else pd.DataFrame()
        if (data_dir / "transactions.csv").exists():
            t = pd.read_csv(data_dir / "transactions.csv")
            t["__ts"] = pd.to_datetime(t["timestamp"])
            t = t.sort_values("__ts").tail(MAX_HISTORY).reset_index(drop=True)
        else:
            t = pd.DataFrame(columns=["txn_id", "sender", "receiver", "amount", "type",
                                      "timestamp", "device_id", "location", "channel",
                                      "is_fraud", "fraud_type", "password_reset_flag", "__ts"])
        self.txns = t
        self.cust = self.customers.set_index("customer_id").to_dict("index") if len(self.customers) else {}
        self.registry: dict[str, set] = defaultdict(set)
        for _, r in self.devices.iterrows():
            self.registry[r["customer_id"]].add(r["device_id"])
        self.seen_recv: dict[str, set] = defaultdict(set)
        self.seen_dev: dict[str, set] = defaultdict(set)
        self.seen_loc: dict[str, set] = defaultdict(set)
        for _, r in self.txns.iterrows():
            self.seen_recv[r["sender"]].add(r["receiver"])
            self.seen_dev[r["sender"]].add(r["device_id"])
            self.seen_loc[r["sender"]].add(r["location"])
        self.decisions: list[dict] = []

    def featurize(self, txn: dict) -> dict:
        ts = datetime.fromisoformat(txn["timestamp"])
        s = txn["sender_id"]
        hist = self.txns[self.txns.sender == s]
        if len(hist):
            h_ts = hist["__ts"]
            m1 = hist[h_ts > (ts - pd.Timedelta(hours=1))]
            m24 = hist[h_ts > (ts - pd.Timedelta(days=1))]
            cnt_1h, sum_1h = len(m1), float(m1.amount.sum())
            cnt_24, sum_24 = len(m24), float(m24.amount.sum())
            last = h_ts.max()
            tsl = (ts - last.to_pydatetime()).total_seconds() / 60 if pd.notna(last) else 24 * 60.0
            avg = float(hist.amount.mean())
        else:
            cnt_1h = cnt_24 = 0
            sum_1h = sum_24 = 0.0
            tsl = 24 * 60.0
            avg = float(txn["amount"])
        amt = float(txn["amount"])
        hour = ts.hour
        rh = self.txns[(self.txns.receiver == txn["receiver_id"]) & (self.txns.__ts > (ts - pd.Timedelta(hours=1)))]
        recv_cnt = len(rh)
        recv_senders = int(rh.sender.nunique()) if len(rh) else 0
        reg = self.registry.get(s, set())
        cinfo = self.cust.get(s, {})
        home = cinfo.get("district", txn.get("location", "Dhaka"))
        return {
            "amount": amt, "amount_log": float(np.log1p(amt)),
            "hour": hour, "unusual_hour": 1 if hour < 6 or hour >= 23 else 0,
            "is_night": 1 if hour >= 22 or hour <= 5 else 0,
            "is_weekend": 1 if ts.weekday() >= 5 else 0,
            "sender_cnt_1h": cnt_1h, "sender_sum_1h": sum_1h,
            "sender_cnt_24h": cnt_24, "sender_sum_24h": sum_24,
            "time_since_last_min": min(tsl, 24 * 60),
            "amount_vs_user_avg": min(amt / max(avg, 1.0), 20.0),
            "new_receiver": 1 if (txn["receiver_id"] not in self.seen_recv[s] and len(self.seen_recv[s]) > 0) else 0,
            "new_device": 1 if (txn["device_id"] not in reg and txn["device_id"] not in self.seen_dev[s]) else 0,
            "location_jump": 1 if (txn.get("location") != home and len(self.seen_loc[s]) > 0) else 0,
            "location_new": 1 if (txn.get("location") not in self.seen_loc[s] and len(self.seen_loc[s]) > 0) else 0,
            "is_round_amount": 1 if int(round(amt)) in ROUND_AMOUNTS else 0,
            "password_reset_flag": int(txn.get("password_reset_flag", 0)),
            "recv_cnt_1h": recv_cnt, "recv_n_senders_1h": recv_senders,
            "fan_in_flag": 1 if recv_senders >= 5 else 0,
            "cash_out_flag": 1 if txn.get("type") == "cash-out" else 0,
            "agent_channel_flag": 1 if txn.get("channel") == "agent" else 0,
            "p2p_flag": 1 if txn.get("type", "P2P") == "P2P" else 0,
            "account_age_days": int(cinfo.get("account_age_days", 365)),
            "avg_balance_log": float(np.log1p(float(cinfo.get("avg_balance", 5000)))),
        }

    def append(self, txn_row: dict):
        row = pd.DataFrame([txn_row])
        row["__ts"] = pd.to_datetime(row["timestamp"])
        self.txns = pd.concat([self.txns, row], ignore_index=True).tail(MAX_HISTORY)
        self.seen_recv[txn_row["sender"]].add(txn_row["receiver"])
        self.seen_dev[txn_row["sender"]].add(txn_row["device_id"])
        self.seen_loc[txn_row["sender"]].add(txn_row["location"])
