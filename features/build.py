"""Batch + online feature builder. No leakage: only past txns (strictly earlier timestamp)."""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "amount", "amount_log",
    "hour", "unusual_hour", "is_night", "is_weekend",
    "sender_cnt_1h", "sender_sum_1h", "sender_cnt_24h", "sender_sum_24h",
    "time_since_last_min", "amount_vs_user_avg",
    "new_receiver", "new_device", "location_jump", "location_new",
    "is_round_amount", "password_reset_flag",
    "recv_cnt_1h", "recv_n_senders_1h", "fan_in_flag",
    "cash_out_flag", "agent_channel_flag", "p2p_flag",
    "account_age_days", "avg_balance_log",
]

ROUND_AMOUNTS = {9900, 19500, 19900, 29500, 49500, 49900, 99000}

# Signals unavailable outside our schema. PaySim (and any foreign MFS feed)
# has no per-device registry, no district trail, and no reset flags, so these
# read 0 there. Everything else is computable in any schema with sender,
# receiver, amount, type/channel, and timestamp — that subset is what portable
# (zero-shot) models train on.
PORTABLE_EXCLUDED = {"new_device", "location_jump", "location_new", "password_reset_flag"}
INTERSECT_COLS = [c for c in FEATURE_COLS if c not in PORTABLE_EXCLUDED]


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def build_features(txns: pd.DataFrame, customers: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    """Add FEATURE_COLS to a copy of txns sorted by timestamp."""
    df = txns.copy()
    df["__ts"] = df["timestamp"].apply(_parse_ts)
    df = df.sort_values("__ts").reset_index(drop=True)

    cust = customers.set_index("customer_id").to_dict("index")
    known_devices: dict[str, set] = defaultdict(set)
    for _, r in devices.iterrows():
        known_devices[r["customer_id"]].add(r["device_id"])

    sender_hist: dict[str, deque] = defaultdict(deque)   # (ts, amount)
    sender_last_ts: dict[str, datetime] = {}
    sender_seen_recv: dict[str, set] = defaultdict(set)
    sender_seen_dev: dict[str, set] = defaultdict(set)
    sender_seen_loc: dict[str, set] = defaultdict(set)
    sender_amounts: dict[str, list] = defaultdict(list)
    recv_hist: dict[str, deque] = defaultdict(deque)     # (ts, sender)

    out_rows = []
    for _, r in df.iterrows():
        s, ts, amt = r["sender"], r["__ts"], float(r["amount"])
        # prune windows
        sh = sender_hist[s]
        while sh and (ts - sh[0][0]).total_seconds() > 24 * 3600:
            sh.popleft()
        cnt_24 = len(sh)
        sum_24 = sum(a for _, a in sh)
        cnt_1h = sum(1 for t, _ in sh if (ts - t).total_seconds() <= 3600)
        sum_1h = sum(a for t, a in sh if (ts - t).total_seconds() <= 3600)
        if s in sender_last_ts:
            tsl = (ts - sender_last_ts[s]).total_seconds() / 60.0
        else:
            tsl = 24 * 60.0
        past = sender_amounts[s]
        avg = sum(past) / len(past) if past else amt
        amount_vs = amt / max(avg, 1.0)

        new_recv = 1 if r["receiver"] not in sender_seen_recv[s] and len(sender_seen_recv[s]) > 0 else 0
        # new device = not in registry and not seen before in stream
        reg = known_devices.get(s, set())
        new_dev = 1 if (r["device_id"] not in reg and r["device_id"] not in sender_seen_dev[s]) else 0
        home = cust.get(s, {}).get("district", r["location"])
        loc_jump = 1 if (r["location"] != home and len(sender_seen_loc[s]) > 0) else 0
        loc_new = 1 if (r["location"] not in sender_seen_loc[s] and len(sender_seen_loc[s]) > 0) else 0

        hour = ts.hour
        unusual = 1 if hour < 6 or hour >= 23 else 0
        night = 1 if hour >= 22 or hour <= 5 else 0
        weekend = 1 if ts.weekday() >= 5 else 0
        is_round = 1 if int(round(amt)) in ROUND_AMOUNTS else 0

        rh = recv_hist[r["receiver"]]
        while rh and (ts - rh[0][0]).total_seconds() > 3600:
            rh.popleft()
        recv_cnt = len(rh)
        recv_senders = len({x for _, x in rh})
        fan_in = 1 if recv_senders >= 5 else 0

        cinfo = cust.get(s, {})
        out_rows.append({
            "district": cinfo.get("district", "Dhaka"),
            "age_group": cinfo.get("age_group", "26-35"),
            "amount": amt, "amount_log": float(np.log1p(amt)),
            "hour": hour, "unusual_hour": unusual, "is_night": night, "is_weekend": weekend,
            "sender_cnt_1h": cnt_1h, "sender_sum_1h": sum_1h,
            "sender_cnt_24h": cnt_24, "sender_sum_24h": sum_24,
            "time_since_last_min": min(tsl, 24 * 60),
            "amount_vs_user_avg": min(amount_vs, 20.0),
            "new_receiver": new_recv, "new_device": new_dev,
            "location_jump": loc_jump, "location_new": loc_new,
            "is_round_amount": is_round,
            "password_reset_flag": int(r.get("password_reset_flag", 0)),
            "recv_cnt_1h": recv_cnt, "recv_n_senders_1h": recv_senders, "fan_in_flag": fan_in,
            "cash_out_flag": 1 if r["type"] == "cash-out" else 0,
            "agent_channel_flag": 1 if r["channel"] == "agent" else 0,
            "p2p_flag": 1 if r["type"] == "P2P" else 0,
            "account_age_days": int(cinfo.get("account_age_days", 365)),
            "avg_balance_log": float(np.log1p(float(cinfo.get("avg_balance", 5000)))),
        })

        # update state AFTER computing (no leakage)
        sh.append((ts, amt))
        sender_last_ts[s] = ts
        sender_seen_recv[s].add(r["receiver"])
        sender_seen_dev[s].add(r["device_id"])
        sender_seen_loc[s].add(r["location"])
        sender_amounts[s].append(amt)
        rh.append((ts, s))

    feat = pd.DataFrame(out_rows)
    # feat carries canonical `amount` / `password_reset_flag` (same values); drop originals to avoid dup cols
    df = pd.concat([df.drop(columns=["__ts", "amount", "password_reset_flag"]), feat], axis=1)
    return df
