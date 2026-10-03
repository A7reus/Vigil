"""Vigil FastAPI: POST /score, GET /alerts, GET /case/:id, POST /decision."""
from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import networkx as nx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.llm import narrate
from api.rules import decide, get_config
from api.schemas import DecisionRequest, ScoreRequest
from api.store import HistoryStore
from models import infer
from models.graph import build_graph, fraud_nodes, network_risk

try:
    from dotenv import load_dotenv

    load_dotenv()  # so a local `.env` (see .env.example) feeds os.getenv below
except ImportError:
    pass  # minimal installs without python-dotenv: plain environment only

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vigil")

store: HistoryStore | None = None
graph: nx.DiGraph | None = None
fraud_set: set = set()
alert_cache: list[dict] = []
startup_info: dict = {}
request_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, graph, fraud_set, alert_cache, startup_info
    t0 = time.perf_counter()
    get_config()
    try:
        infer.load_artifacts()
    except FileNotFoundError as e:
        raise RuntimeError(
            f"model artifacts missing ({e}). Run in order: "
            "python -m data_gen.generate --out data && "
            "python -m models.train --data data --artifacts artifacts, "
            "then start the API."
        ) from e
    store = HistoryStore()
    if len(store.txns) == 0:
        log.warning("no transaction history found in data/ — queue will be empty until data is generated")
    tx = store.txns
    graph = build_graph(tx) if len(tx) else nx.DiGraph()
    try:
        fraud_set = set(json.loads(Path("artifacts/fraud_nodes.json").read_text()))
    except Exception:
        fraud_set = fraud_nodes(tx) if len(tx) and "is_fraud" in tx else set()
    # Pre-score recent txns as the analyst queue. Bound via env so large
    # histories don't blow up cold-start time on the on-site machine.
    pre_limit = int(os.getenv("VIGIL_ALERTS_LIMIT", "200"))
    alert_cache = _build_alerts(limit=pre_limit)
    startup_info = {
        "history_rows": len(tx), "graph_nodes": graph.number_of_nodes(),
        "queue_size": len(alert_cache),
        "startup_s": round(time.perf_counter() - t0, 1),
        "pre_limit": pre_limit,
    }
    log.info("startup complete: %s", startup_info)
    yield


def _build_alerts(limit: int = 500) -> list[dict]:
    """Causal rescoring: rebuild features on the slice in time order (no future leakage)."""
    assert store is not None
    from features.build import FEATURE_COLS, build_features
    cfg = get_config()
    recent = store.txns.tail(limit).copy()
    # build causal features over history + recent so velocity/seen-flags are past-only
    hist = store.txns.head(max(0, len(store.txns) - limit))
    combined = pd.concat([hist, recent], ignore_index=True)
    feat_all = build_features(combined, store.customers, store.devices)
    feat_recent = feat_all.tail(len(recent)).reset_index(drop=True)
    recent = recent.reset_index(drop=True)
    out = []
    assert graph is not None, "lifespan must build graph first"
    for i, r in recent.iterrows():
        frow = feat_recent.iloc[i]
        # Convert to plain Python floats so the cache is JSON-serializable
        # and reusable by /case without re-featurizing with dummy values.
        feats = {c: float(frow[c]) for c in FEATURE_COLS}
        # expose a few raw fields the LLM needs
        feats["location_new"] = float(frow.get("location_new", 0))
        feats["amount_vs_user_avg"] = float(feats.get("amount_vs_user_avg", 1.0))
        txn = {"sender_id": r["sender"], "receiver_id": r["receiver"], "amount": float(r["amount"]),
               "channel": r["channel"], "device_id": r["device_id"], "location": r["location"],
               "timestamp": r["timestamp"], "type": r["type"]}
        gf = network_risk(txn["receiver_id"], graph, fraud_set,
                          k_threshold=cfg["graph"]["two_hop_fraud_neighbors_threshold"],
                          boost_per_hit=cfg["graph"]["boost_per_hit"], max_boost=cfg["graph"]["max_boost"])
        s = infer.score_features(feats, gf["boost"], cfg["ensemble_weights"])
        d = decide(s["risk_score"])
        out.append({"txn_id": r["txn_id"], **txn, **s, **d,
                    "fraud_neighbors_2hop": gf["fraud_neighbors_2hop"],
                    "fraud_neighbor_sample": gf.get("fraud_neighbor_sample", []),
                    "label": int(r.get("is_fraud", 0)),
                    "feats": feats,
                    "graph_boost_raw": float(gf["boost"])})
    return sorted(out, key=lambda x: -x["risk_score"])


app = FastAPI(title="Vigil Risk API", version="0.1.0", lifespan=lifespan)
# NOTE (security): open CORS is intentional for the hackathon demo (judges hit
# the API from any origin). Restrict allow_origins to the deployed frontend
# domain before any production use.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _reject_nonfinite_constant(v: str):
    raise ValueError(f"non-finite JSON literal rejected: {v}")


def _ensure_finite(obj) -> None:
    """Walk parsed JSON; stdlib json silently accepts NaN/Infinity (and 1e999
    overflows to inf), which then crash error serialization with a 500."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("non-finite number")
    elif isinstance(obj, list):
        for v in obj:
            _ensure_finite(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            _ensure_finite(v)


@app.middleware("http")
async def _reject_nonfinite_json(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH") and "application/json" in request.headers.get("content-type", ""):
        try:
            raw = (await request.body()).decode("utf-8")
            _ensure_finite(json.loads(raw, parse_constant=_reject_nonfinite_constant))
        except ValueError as e:
            log.warning("rejected non-JSON/non-finite body: %s", e)
            return JSONResponse({"detail": f"invalid JSON body: {e}"}, status_code=422)
        except Exception as e:
            log.warning("rejected unreadable body: %s", e)
            return JSONResponse({"detail": "body must be UTF-8 JSON"}, status_code=422)
    return await call_next(request)


@app.get("/health")
def health():
    return {"ok": True, "history_rows": 0 if store is None else len(store.txns),
            "graph_nodes": 0 if graph is None else graph.number_of_nodes(),
            "queue_size": len(alert_cache), "startup": startup_info}


@app.get("/metrics")
def metrics():
    """Lightweight ops counters (monitoring expectation in the guideline)."""
    return {"requests_scored": request_count, "queue_size": len(alert_cache),
            "decisions_logged": 0 if store is None else len(store.decisions),
            "startup": startup_info}


# Exact score-time state for LIVE txns (feats + graph facts), so /case/{live_id}
# resolves without recomputing. Capped; evicts oldest first.
live_cases: dict[str, dict] = {}
MAX_LIVE_CASES = 500

# Per-IP token buckets for write endpoints (demo-grade flood protection; the
# scoring state itself is already isolated from LIVE traffic in store.py).
_rate_hits: dict[str, list[float]] = {}


def _rate_limit_ok(ip: str) -> bool:
    try:
        limit = int(os.getenv("VIGIL_RATE_LIMIT_PER_MIN", "120"))
    except ValueError:
        limit = 120
    if limit <= 0:
        return True
    now = time.monotonic()
    hits = [t for t in _rate_hits.get(ip, []) if now - t < 60.0]
    if len(hits) >= limit:
        _rate_hits[ip] = hits
        return False
    _rate_hits[ip] = hits + [now]
    return True


def _known_txn_ids() -> set:
    ids = {r["txn_id"] for r in alert_cache} | set(live_cases)
    if store is not None and len(store.txns):
        ids |= set(store.txns["txn_id"].tolist())
    return ids


@app.post("/score")
def score(req: ScoreRequest, request: Request):
    global request_count
    assert store is not None and graph is not None
    if not _rate_limit_ok(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "rate limit exceeded, retry in a minute")
    t0 = time.perf_counter()
    txn = req.model_dump()
    feats = store.featurize(txn)
    cfg = get_config()
    gf = network_risk(txn["receiver_id"], graph, fraud_set,
                      k_threshold=cfg["graph"]["two_hop_fraud_neighbors_threshold"],
                      boost_per_hit=cfg["graph"]["boost_per_hit"], max_boost=cfg["graph"]["max_boost"])
    s = infer.score_features(feats, gf["boost"], cfg["ensemble_weights"])
    d = decide(s["risk_score"])
    nar = narrate(txn, feats, {**s, **d}, gf, lang=req.lang)
    txn_id = f"LIVE-{uuid4().hex[:8]}"
    # Scoring-neutral: visible in timelines only, never moves anyone's features.
    store.append_live({"txn_id": txn_id, "sender": txn["sender_id"], "receiver": txn["receiver_id"],
                       "amount": txn["amount"], "type": txn.get("type", "P2P"), "timestamp": txn["timestamp"],
                       "device_id": txn["device_id"], "location": txn["location"], "channel": txn["channel"]})
    live_cases[txn_id] = {"txn": txn, "feats": feats, "score": s, "decision": d,
                          "graph": {"boost": gf["boost"],
                                    "fraud_neighbors_2hop": gf["fraud_neighbors_2hop"],
                                    "fraud_neighbor_sample": gf.get("fraud_neighbor_sample", [])}}
    while len(live_cases) > MAX_LIVE_CASES:
        live_cases.pop(next(iter(live_cases)))
    latency_ms = (time.perf_counter() - t0) * 1000
    request_count += 1
    log.info("score %s -> %.3f %s (%.1fms)", txn_id, s["risk_score"], d["risk_level"], latency_ms)
    return {"txn_id": txn_id, "risk_score": s["risk_score"], "risk_level": d["risk_level"],
            "top_3_reasons": s["top_3_reasons"], "recommended_action": d["recommended_action"],
            "components": {**{k: s[k] for k in ("p_fraud", "anomaly", "graph_boost")},
                            "fraud_neighbors_2hop": gf["fraud_neighbors_2hop"],
                            "fraud_neighbor_sample": gf.get("fraud_neighbor_sample", []),
                            "latency_ms": round(latency_ms, 1)},
            "narrative": nar["narrative"]}


@app.get("/alerts")
def alerts(limit: int = Query(100, ge=1, le=500), level: str | None = None):
    rows = alert_cache[:limit] if alert_cache else []
    if level:
        rows = [r for r in rows if r["risk_level"].lower() == level.lower()]
    # Public view: no internals (feats), no ground-truth labels.
    public = [{k: v for k, v in r.items() if k not in ("feats", "graph_boost_raw", "label")}
              for r in rows[:limit]]
    return {"alerts": public, "count": len(public)}


@app.get("/case/{txn_id}")
def case(txn_id: str, lang: str = "en"):
    assert store is not None
    live = live_cases.get(txn_id)
    rows = [r for r in alert_cache if r["txn_id"] == txn_id]
    if live is None and not rows:
        raise HTTPException(404, "case not found; score it first via POST /score or pick a queued alert")
    if live is not None:
        txn, feats = live["txn"], live["feats"]
        c = {**live["score"], **live["decision"],
             "fraud_neighbors_2hop": live["graph"]["fraud_neighbors_2hop"],
             "fraud_neighbor_sample": live["graph"]["fraud_neighbor_sample"],
             "txn_id": txn_id, **{k: txn.get(k) for k in
                                  ("sender_id", "receiver_id", "amount", "channel", "device_id", "location", "timestamp")}}
        gfacts = live["graph"]
    else:
        c = rows[0]
        txn = {"sender_id": c.get("sender_id", c.get("sender")),
               "receiver_id": c.get("receiver_id", c.get("receiver")), "amount": c["amount"],
               "channel": c["channel"], "device_id": c.get("device_id", "unknown"),
               "location": c.get("location", "Dhaka"),
               "timestamp": c.get("timestamp", "")}
        # Reuse the exact causal features computed at queue build time.
        # Fallback to live featurization only for caches built before this fix.
        feats = c.get("feats")
        if not feats:
            feats = store.featurize({
                "sender_id": txn["sender_id"], "receiver_id": txn["receiver_id"],
                "amount": txn["amount"], "channel": txn["channel"],
                "device_id": txn.get("device_id", "unknown"),
                "location": txn.get("location", "Dhaka"),
                "timestamp": txn["timestamp"] or "2026-08-15T12:00:00"})
        gfacts = {"boost": c.get("graph_boost_raw", c.get("graph_boost", 0)),
                  "fraud_neighbors_2hop": c.get("fraud_neighbors_2hop", 0)}
    nar = narrate(txn, feats, c, gfacts, lang=lang)
    timeline = store.timeline_for(txn.get("sender_id", txn.get("sender", "")), n=10)
    public = {k: v for k, v in c.items() if k not in ("feats", "graph_boost_raw", "label")}
    log.info("case %s viewed (lang=%s)", txn_id, lang)
    return {**public, "narrative": nar["narrative"], "timeline": timeline}


@app.post("/decision")
def decision(req: DecisionRequest, request: Request):
    assert store is not None
    if not _rate_limit_ok(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "rate limit exceeded, retry in a minute")
    if req.txn_id not in _known_txn_ids():
        raise HTTPException(404, "unknown txn_id; score it or pick a queued alert first")
    entry = {**req.model_dump(), "at": pd.Timestamp.now("UTC").isoformat()}
    updated = False
    for i, d in enumerate(store.decisions):
        if d.get("txn_id") == req.txn_id and d.get("analyst") == req.analyst:
            store.decisions[i] = entry
            updated = True
            break
    if not updated:
        store.decisions.append(entry)
    log.info("decision %s -> %s by %s (updated=%s)", req.txn_id, req.decision, req.analyst, updated)
    return {"ok": True, "updated": updated, "logged": req.model_dump(),
            "pending_retrain": len(store.decisions)}


# Analyst console (no build step): served from /web. API routes above take
# precedence; this mount only handles / and static files.
try:
    from fastapi.staticfiles import StaticFiles

    _WEB_DIR = Path(__file__).resolve().parent.parent / "web"
    if _WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
except Exception:
    pass
