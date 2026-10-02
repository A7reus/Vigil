"""ScamShield FastAPI: POST /score, GET /alerts, GET /case/:id, POST /decision."""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import networkx as nx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.llm import narrate
from api.rules import decide, get_config
from api.schemas import DecisionRequest, ScoreRequest
from api.store import HistoryStore
from models import infer
from models.graph import build_graph, fraud_nodes, network_risk

store: HistoryStore | None = None
graph: nx.DiGraph | None = None
fraud_set: set = set()
alert_cache: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, graph, fraud_set, alert_cache
    get_config()
    infer.load_artifacts()
    store = HistoryStore()
    tx = store.txns
    graph = build_graph(tx) if len(tx) else nx.DiGraph()
    try:
        fraud_set = set(json.loads(Path("artifacts/fraud_nodes.json").read_text()))
    except Exception:
        fraud_set = fraud_nodes(tx) if len(tx) and "is_fraud" in tx else set()
    # pre-score recent test txns as the analyst queue (top-200 by risk, cached)
    alert_cache = _build_alerts(limit=200)
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
    for i, r in recent.iterrows():
        frow = feat_recent.iloc[i]
        feats = {c: frow[c] for c in FEATURE_COLS}
        # expose a few raw fields the LLM needs
        feats["location_new"] = frow.get("location_new", 0)
        txn = {"sender_id": r["sender"], "receiver_id": r["receiver"], "amount": float(r["amount"]),
               "channel": r["channel"], "device_id": r["device_id"], "location": r["location"],
               "timestamp": r["timestamp"], "type": r["type"]}
        gf = network_risk(txn["receiver_id"], graph, fraud_set,
                          k_threshold=cfg["graph"]["two_hop_fraud_neighbors_threshold"],
                          boost_per_hit=cfg["graph"]["boost_per_hit"], max_boost=cfg["graph"]["max_boost"])
        s = infer.score_features(feats, gf["boost"], cfg["ensemble_weights"])
        d = decide(s["risk_score"])
        out.append({"txn_id": r["txn_id"], **txn, **s, **d,
                    "fraud_neighbors_2hop": gf["fraud_neighbors_2hop"], "label": int(r.get("is_fraud", 0))})
    return sorted(out, key=lambda x: -x["risk_score"])


app = FastAPI(title="ScamShield Risk API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"ok": True, "history_rows": 0 if store is None else len(store.txns),
            "graph_nodes": 0 if graph is None else graph.number_of_nodes()}


@app.post("/score")
def score(req: ScoreRequest):
    assert store is not None
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
    store.append({"txn_id": txn_id, "sender": txn["sender_id"], "receiver": txn["receiver_id"],
                  "amount": txn["amount"], "type": txn.get("type", "P2P"), "timestamp": txn["timestamp"],
                  "device_id": txn["device_id"], "location": txn["location"], "channel": txn["channel"],
                  "is_fraud": 0, "fraud_type": "none", "password_reset_flag": 0})
    latency_ms = (time.perf_counter() - t0) * 1000
    return {"txn_id": txn_id, "risk_score": s["risk_score"], "risk_level": d["risk_level"],
            "top_3_reasons": s["top_3_reasons"], "recommended_action": d["recommended_action"],
            "components": {**{k: s[k] for k in ("p_fraud", "anomaly", "graph_boost")},
                           "fraud_neighbors_2hop": gf["fraud_neighbors_2hop"],
                           "latency_ms": round(latency_ms, 1)},
            "narrative": nar["narrative"]}


@app.get("/alerts")
def alerts(limit: int = Query(100, le=500), level: str | None = None):
    rows = alert_cache[:limit] if alert_cache else []
    if level:
        rows = [r for r in rows if r["risk_level"].lower() == level.lower()]
    return {"alerts": rows[:limit], "count": len(rows[:limit])}


@app.get("/case/{txn_id}")
def case(txn_id: str, lang: str = "en"):
    rows = [r for r in alert_cache if r["txn_id"] == txn_id]
    if not rows:
        raise HTTPException(404, "case not found in pre-scored queue; POST /score first")
    c = rows[0]
    txn = {"sender_id": c["sender_id"] if "sender_id" in c else c.get("sender"),
           "receiver_id": c.get("receiver_id", c.get("receiver")), "amount": c["amount"],
           "channel": c["channel"], "timestamp": c.get("timestamp", "")}
    feats = store.featurize({"sender_id": txn["sender_id"], "receiver_id": txn["receiver_id"],
                             "amount": txn["amount"], "channel": txn["channel"], "device_id": "unknown",
                             "location": "Dhaka", "timestamp": txn["timestamp"] or "2026-08-15T12:00:00"})
    nar = narrate(txn, feats, c, {"boost": c.get("graph_boost", 0),
                                  "fraud_neighbors_2hop": c.get("fraud_neighbors_2hop", 0)}, lang=lang)
    timeline = store.txns[store.txns.sender == txn["sender_id"]].tail(10)[
        ["txn_id", "receiver", "amount", "timestamp", "location"]].to_dict("records")
    return {**c, "narrative": nar["narrative"], "timeline": timeline}


@app.post("/decision")
def decision(req: DecisionRequest):
    assert store is not None
    store.decisions.append({**req.model_dump(), "at": pd.Timestamp.utcnow().isoformat()})
    return {"ok": True, "logged": req.model_dump(), "pending_retrain": len(store.decisions)}
