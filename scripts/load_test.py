"""Throughput probe for POST /score. Stdlib only, no extra deps.

Starts nothing: point it at a running API (default port 8009 here to stay
clear of a dev server on 8000) and it reports requests/s plus p50/p95.
Set VIGIL_API_KEY on the server first and pass --key to measure the
authenticated path; without it the open demo path is measured.

Usage:
    uvicorn api.main:app --port 8009 &
    python -m scripts.load_test --n 500 --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BODY = {"sender_id": "C000001", "receiver_id": "C000002", "amount": 45000,
        "channel": "app", "device_id": "DX999", "location": "Dhaka",
        "timestamp": "2026-08-15T23:10:00", "type": "P2P", "lang": "en"}


def one(base: str, key: str) -> tuple[float, bool]:
    data = json.dumps(BODY).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    req = urllib.request.Request(f"{base}/score", data=data, headers=headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            ok = r.status == 200
    except Exception:
        ok = False
    return (time.perf_counter() - t0) * 1000, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8009")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--key", default="")
    a = ap.parse_args()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        t0 = time.perf_counter()
        res = list(ex.map(lambda _: one(a.base, a.key), range(a.n)))
        total = time.perf_counter() - t0
    lat = sorted(ms for ms, _ in res)
    ok = sum(1 for _, o in res if o)
    out = {"n": a.n, "concurrency": a.concurrency, "ok": ok,
           "errors": a.n - ok, "rps": round(a.n / total, 1),
           "p50_ms": round(lat[len(lat) // 2], 1),
           "p95_ms": round(lat[int(len(lat) * 0.95)], 1)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
