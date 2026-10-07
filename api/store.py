"""In-memory history store: preloaded CSV history + newly scored txns.

Featurize runs on per-sender / per-receiver rolling event lists, so cost is
proportional to one wallet's activity, not history size (the old full-frame
scan grew with all 60k rows). Semantics match features.build exactly —
see tests/test_parity.py, which must stay green after any change here.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from features.build import ROUND_AMOUNTS

MAX_HISTORY = 60_000
# Unreviewed LIVE scores live here — never in txns/seen-sets — so unauthenticated
# /score traffic cannot move anyone else's features (velocity, seen-device,
# seen-location) nor evict committed history.
MAX_LIVE = 2000


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
        self.live_rows: list[dict] = []
        # Rolling scoring state (item 3): per-sender (ts, amount) events plus
        # all-time sum/count/max for the average and idle-time features;
        # per-receiver (ts, sender) events for the fan-in features.
        self.send_evts: dict[str, list] = {}
        self.send_total: dict[str, list] = {}
        self.send_max: dict[str, datetime] = {}
        self.recv_evts: dict[str, list] = {}
        for _, r in self.txns.iterrows():
            self._ingest(r["sender"], r["receiver"], r["__ts"], float(r["amount"]))

    def _ingest(self, sender, receiver, ts, amount: float):
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        ev = self.send_evts.setdefault(sender, [])
        ev.append((ts, amount))
        tot = self.send_total.setdefault(sender, [0.0, 0])
        tot[0] += amount
        tot[1] += 1
        if sender not in self.send_max or ts > self.send_max[sender]:
            self.send_max[sender] = ts
        self.recv_evts.setdefault(receiver, []).append((ts, sender))

    def featurize(self, txn: dict) -> dict:
        ts = datetime.fromisoformat(txn["timestamp"])
        s = txn["sender_id"]
        if s in self.send_total and self.send_total[s][1] > 0:
            cnt_1h = cnt_24 = 0
            sum_1h = sum_24 = 0.0
            for t, a in self.send_evts[s]:
                dt = (ts - t).total_seconds()
                if dt < 3600:
                    cnt_1h += 1
                    sum_1h += a
                if dt < 24 * 3600:
                    cnt_24 += 1
                    sum_24 += a
            last = self.send_max[s]
            tsl = (ts - last).total_seconds() / 60
            tsum, tcnt = self.send_total[s]
            avg = tsum / tcnt
        else:
            cnt_1h = cnt_24 = 0
            sum_1h = sum_24 = 0.0
            tsl = 24 * 60.0
            avg = float(txn["amount"])
        amt = float(txn["amount"])
        hour = ts.hour
        recv_cnt, senders_1h = 0, set()
        for t, snd in self.recv_evts.get(txn["receiver_id"], ()):
            if (ts - t).total_seconds() < 3600:
                recv_cnt += 1
                senders_1h.add(snd)
        recv_senders = len(senders_1h)
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
        """Committed history ingestion (updates scoring state)."""
        row = pd.DataFrame([txn_row])
        row["__ts"] = pd.to_datetime(row["timestamp"])
        self.txns = pd.concat([self.txns, row], ignore_index=True)
        self._ingest(txn_row["sender"], txn_row["receiver"],
                     row["__ts"].iloc[0], float(txn_row["amount"]))
        if len(self.txns) > MAX_HISTORY:
            # Trim rarely (not per row) and rebuild rolling lists from the
            # surviving window so evicted events stop influencing features.
            self.txns = self.txns.tail(MAX_HISTORY).reset_index(drop=True)
            self._rebuild_rolling()
        self.seen_recv[txn_row["sender"]].add(txn_row["receiver"])
        self.seen_dev[txn_row["sender"]].add(txn_row["device_id"])
        self.seen_loc[txn_row["sender"]].add(txn_row["location"])

    def _rebuild_rolling(self):
        self.send_evts, self.send_total, self.send_max, self.recv_evts = {}, {}, {}, {}
        for _, r in self.txns.iterrows():
            self._ingest(r["sender"], r["receiver"], r["__ts"], float(r["amount"]))

    def append_live(self, txn_row: dict):
        """Unreviewed /score traffic: visible in timelines only, scoring-neutral."""
        self.live_rows.append(txn_row)
        if len(self.live_rows) > MAX_LIVE:
            self.live_rows = self.live_rows[-MAX_LIVE:]

    def timeline_for(self, wallet_id: str, n: int = 10) -> list[dict]:
        """Last sends+receives from committed history merged with live rows."""
        hist = self.txns[(self.txns.sender == wallet_id) | (self.txns.receiver == wallet_id)].tail(n)
        rows = hist[["txn_id", "sender", "receiver", "amount", "timestamp", "location"]].to_dict("records")
        for r in self.live_rows:
            if r.get("sender") == wallet_id or r.get("receiver") == wallet_id:
                rows.append({k: r.get(k) for k in ("txn_id", "sender", "receiver", "amount", "timestamp", "location")})
        rows.sort(key=lambda r: str(r.get("timestamp", "")))
        return rows[-n:]
