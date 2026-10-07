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

from api import auth, db
from api.llm import narrate
from api.rules import decide, get_config
from api.schemas import (CaseAssignRequest, CaseStatusRequest, DecisionRequest,
                         LoginRequest, RegisterRequest, ScoreRequest)
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
    db.configure()
    auth.seed_demo_users()
    for r in alert_cache:
        db.upsert_case(r["txn_id"], _blob_for_alert(r), r["risk_score"],
                       r["risk_level"], source="queue")
    startup_info = {
        "history_rows": len(tx), "graph_nodes": graph.number_of_nodes(),
        "queue_size": len(alert_cache),
        "startup_s": round(time.perf_counter() - t0, 1),
        "pre_limit": pre_limit,
    }
    log.info("startup complete: %s", startup_info)
    yield


def _build_alerts(limit: int = 500) -> list[dict]:
    """Analyst triage queue: score a wide recent window, keep the top `limit`
    by risk. Features stay causal (past-only, no future leakage); only the
    *selection* changed — latest-N showed whatever fraud density the tail
    window happened to have, which demos poorly and triages worse."""
    assert store is not None
    import numpy as np

    from features.build import FEATURE_COLS, build_features
    cfg = get_config()
    window = int(os.getenv("VIGIL_ALERT_WINDOW", "2000"))
    recent = store.txns.tail(window).copy()
    # build causal features over history + recent so velocity/seen-flags are past-only
    hist = store.txns.head(max(0, len(store.txns) - window))
    combined = pd.concat([hist, recent], ignore_index=True)
    feat_all = build_features(combined, store.customers, store.devices)
    feat_recent = feat_all.tail(len(recent)).reset_index(drop=True)
    recent = recent.reset_index(drop=True)
    assert graph is not None, "lifespan must build graph first"
    gcfg = cfg["graph"]
    # Batched ML scores for the whole window (one predict call), then graph
    # boosts, then full detail (reasons need per-row SHAP) only for the top K.
    X = feat_recent[FEATURE_COLS].to_numpy(dtype=float)
    boosts = np.array([network_risk(
        r, graph, fraud_set, k_threshold=gcfg["two_hop_fraud_neighbors_threshold"],
        boost_per_hit=gcfg["boost_per_hit"], max_boost=gcfg["max_boost"])["boost"]
        for r in recent["receiver"].tolist()], dtype=float)
    scores = infer.score_batch(X, boosts, cfg["ensemble_weights"])
    out = []
    for i in np.argsort(-scores)[:limit]:
        i = int(i)
        r, frow = recent.iloc[i], feat_recent.iloc[i]
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
                          k_threshold=gcfg["two_hop_fraud_neighbors_threshold"],
                          boost_per_hit=gcfg["boost_per_hit"], max_boost=gcfg["max_boost"])
        s = infer.score_features(feats, gf["boost"], cfg["ensemble_weights"])
        d = decide(s["risk_score"])
        out.append({"txn_id": r["txn_id"], **txn, **s, **d,
                    "fraud_neighbors_2hop": gf["fraud_neighbors_2hop"],
                    "fraud_neighbor_sample": gf.get("fraud_neighbor_sample", []),
                    "label": int(r.get("is_fraud", 0)),
                    "feats": feats,
                    "graph_boost_raw": float(gf["boost"])})
    return sorted(out, key=lambda x: -x["risk_score"])


def _blob_for_alert(r: dict) -> dict:
    """Persisted case shape mirroring live_cases (txn + feats + score + graph)."""
    txn = {k: r.get(k) for k in ("sender_id", "receiver_id", "amount", "channel",
                                 "device_id", "location", "timestamp", "type")}
    return {"txn": txn, "feats": r.get("feats", {}),
            "score": {k: r.get(k) for k in ("p_fraud", "anomaly", "graph_boost",
                                            "risk_score", "top_3_reasons")},
            "decision": {"risk_level": r.get("risk_level"),
                         "recommended_action": r.get("recommended_action")},
            "graph": {"boost": r.get("graph_boost_raw", 0.0),
                      "fraud_neighbors_2hop": r.get("fraud_neighbors_2hop", 0),
                      "fraud_neighbor_sample": r.get("fraud_neighbor_sample", [])}}


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
            "queue_size": len(alert_cache), "model_version": infer.MODEL_VERSION,
            "startup": startup_info}


@app.get("/ready")
def ready():
    """Readiness (can serve real scores?) vs /health liveness (is it up?).

    Returns 503 with per-check detail when anything required is missing, so
    orchestrators and shadow-mode harnesses can gate traffic correctly.
    """
    from fastapi.responses import JSONResponse
    checks: dict[str, bool | str] = {}
    try:
        infer.load_artifacts()
        checks["artifacts"] = True
    except Exception as e:
        checks["artifacts"] = f"missing: {e}"
    checks["history"] = bool(store is not None and len(store.txns) > 0)
    checks["graph"] = bool(graph is not None and graph.number_of_nodes() > 0)
    try:
        assert store is not None
        probe = store.featurize({"sender_id": "__probe__", "receiver_id": "__probe__",
                                 "amount": 100.0, "channel": "app", "device_id": "x",
                                 "location": "Dhaka", "timestamp": "2026-08-15T12:00:00",
                                 "type": "P2P"})
        infer.score_features(probe, 0.0)
        checks["canary_score"] = True
    except Exception as e:
        checks["canary_score"] = f"failed: {type(e).__name__}"
    ok = all(v is True for v in checks.values())
    return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)


@app.get("/metrics")
def metrics():
    """Lightweight ops counters (monitoring expectation in the guideline)."""
    try:
        logged, open_cases = db.count_decisions(), len(db.list_open_cases(limit=100000))
    except Exception:
        logged, open_cases = 0, len(alert_cache)
    return {"requests_scored": request_count, "queue_size": len(alert_cache),
            "decisions_logged": logged, "open_cases": open_cases,
            "startup": startup_info}


# --- auth: registration is pending until an admin approves -----------------
@app.post("/auth/register", status_code=201)
def register(req: RegisterRequest):
    auth.check_username(req.username)
    auth.check_password(req.password)
    u = db.create_user(req.username, auth.hash_password(req.password))
    if u is None:
        raise HTTPException(409, "username already taken")
    log.info("registered %s (pending approval)", req.username)
    return {"user": db.public_user(u),
            "message": "registered; awaiting admin approval before login"}


@app.post("/auth/login")
def login(req: LoginRequest, request: Request):
    if not _rate_limit_ok("login:" + (request.client.host if request.client else "?")):
        raise HTTPException(429, "too many login attempts, retry in a minute")
    u = db.get_user_by_username(req.username)
    if not u or not auth.verify_password(req.password, u["pw_hash"]):
        raise HTTPException(401, "invalid credentials")
    if u["status"] == "pending":
        raise HTTPException(403, "account pending admin approval")
    if u["status"] != "active":
        raise HTTPException(403, "account disabled")
    token, exp = auth.issue_token(u["id"])
    log.info("login %s", u["username"])
    return {"token": token, "expires_at": exp, "user": db.public_user(u)}


@app.get("/auth/me")
def me(request: Request):
    return {"user": auth.need_user(request)}


@app.post("/auth/logout")
def logout(request: Request):
    from api.auth import bearer_token
    import hashlib
    t = bearer_token(request)
    if t:
        db.revoke_session(hashlib.sha256(t.encode()).hexdigest())
    return {"ok": True}


# --- admin: users, reviews, case workflow ----------------------------------
@app.get("/admin/users")
def admin_users(request: Request):
    auth.need_admin(request)
    return {"users": db.list_users()}


@app.post("/admin/users/{uid}/approve")
def admin_approve(uid: int, request: Request):
    auth.need_admin(request)
    u = db.set_user_status(uid, "active")
    if u is None:
        raise HTTPException(404, "unknown user")
    log.info("admin approved user %s", u["username"])
    return {"user": db.public_user(u)}


@app.post("/admin/users/{uid}/disable")
def admin_disable(uid: int, request: Request):
    me = auth.need_admin(request)
    if me["id"] == uid:
        raise HTTPException(400, "cannot disable your own admin account")
    u = db.set_user_status(uid, "disabled")
    if u is None:
        raise HTTPException(404, "unknown user")
    return {"user": db.public_user(u)}


@app.post("/admin/users/{uid}/enable")
def admin_enable(uid: int, request: Request):
    auth.need_admin(request)
    u = db.set_user_status(uid, "active")
    if u is None:
        raise HTTPException(404, "unknown user")
    return {"user": db.public_user(u)}


@app.get("/admin/decisions")
def admin_decisions(request: Request, analyst: str | None = None, limit: int = Query(200, ge=1, le=2000)):
    auth.need_admin(request)
    return {"decisions": db.list_decisions(analyst=analyst, limit=limit)}


@app.get("/admin/cases")
def admin_cases(request: Request, status: str | None = None, analyst: str | None = None,
                limit: int = Query(200, ge=1, le=2000)):
    auth.need_admin(request)
    if status and status not in ("open", "assigned", "closed"):
        raise HTTPException(422, "status must be open|assigned|closed")
    return {"cases": [db.row_to_public_case(r) for r in db.list_cases(status, analyst, limit)]}


@app.post("/cases/{txn_id}/assign")
def assign_case(txn_id: str, req: CaseAssignRequest, request: Request):
    me = auth.need_user(request)
    target = db.get_user_by_username(req.analyst)
    if not target or target["status"] != "active":
        raise HTTPException(404, "unknown or inactive analyst")
    if me["role"] != "admin" and req.analyst != me["username"]:
        raise HTTPException(403, "analysts may only self-assign; admins may assign anyone")
    if not db.get_case(txn_id) and txn_id not in _known_txn_ids():
        raise HTTPException(404, "unknown txn_id")
    db.upsert_case(txn_id, _case_blob_or_empty(txn_id), *db_case_score(txn_id))
    if not db.set_case_status(txn_id, "assigned", req.analyst):
        raise HTTPException(404, "unknown txn_id")
    log.info("case %s assigned to %s by %s", txn_id, req.analyst, me["username"])
    return {"ok": True, "txn_id": txn_id, "analyst": req.analyst}


@app.post("/cases/{txn_id}/status")
def case_status(txn_id: str, req: CaseStatusRequest, request: Request):
    me = auth.need_user(request)
    row = db.get_case(txn_id)
    if row is None and txn_id not in _known_txn_ids():
        raise HTTPException(404, "unknown txn_id")
    if req.status == "closed" and me["role"] != "admin" and (row or {}).get("analyst") != me["username"]:
        raise HTTPException(403, "only the assignee or an admin may close a case")
    if row is None:
        db.upsert_case(txn_id, _case_blob_or_empty(txn_id), *db_case_score(txn_id))
    db.set_case_status(txn_id, req.status)
    return {"ok": True, "txn_id": txn_id, "status": req.status}


def _case_blob_or_empty(txn_id: str) -> dict:
    row = db.get_case(txn_id)
    if row:
        import json
        try:
            return json.loads(row["payload"])
        except Exception:
            pass
    live = live_cases.get(txn_id)
    if live:
        return {"txn": live["txn"], "feats": live["feats"], "score": live["score"],
                "decision": live["decision"], "graph": live["graph"]}
    for r in alert_cache:
        if r["txn_id"] == txn_id:
            return _blob_for_alert(r)
    return {}


def db_case_score(txn_id: str) -> tuple[float, str]:
    row = db.get_case(txn_id)
    if row:
        return float(row["risk_score"]), row["risk_level"]
    live = live_cases.get(txn_id)
    if live:
        return float(live["score"]["risk_score"]), live["decision"]["risk_level"]
    for r in alert_cache:
        if r["txn_id"] == txn_id:
            return float(r["risk_score"]), r["risk_level"]
    return 0.0, "Low"


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
    try:
        ids |= {r["txn_id"] for r in db.list_cases(limit=5000)}
    except Exception:
        pass
    return ids


@app.post("/score")
def score(req: ScoreRequest, request: Request, commit: bool = False):
    global request_count
    assert store is not None and graph is not None
    if not _rate_limit_ok(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "rate limit exceeded, retry in a minute")
    committer = None
    if commit:
        # Authenticated ingestion (item 1): reviewed/confirmed traffic joins
        # committed history so velocity and seen-sets reflect real behavior.
        committer = auth.need_user(request)
    idem = (request.headers.get("Idempotency-Key") or "").strip()
    if len(idem) > 128:
        raise HTTPException(422, "Idempotency-Key must be at most 128 chars")
    if idem:
        hit = db.get_idempotent(idem)
        if hit:
            return {**hit, "deduplicated": True}
    t0 = time.perf_counter()
    txn = req.model_dump()
    degraded, derr = False, ""
    try:
        feats = store.featurize(txn)
        cfg = get_config()
        gf = network_risk(txn["receiver_id"], graph, fraud_set,
                          k_threshold=cfg["graph"]["two_hop_fraud_neighbors_threshold"],
                          boost_per_hit=cfg["graph"]["boost_per_hit"], max_boost=cfg["graph"]["max_boost"])
        s = infer.score_features(feats, gf["boost"], cfg["ensemble_weights"])
        d = decide(s["risk_score"])
        nar = narrate(txn, feats, {**s, **d}, gf, lang=req.lang)
    except Exception as e:
        # Item 6: fail closed to human review, never silently allow.
        log.exception("score pipeline failed, degrading to review")
        degraded, derr = True, f"{type(e).__name__}"
        s = {"p_fraud": 0.0, "anomaly": 0.0, "graph_boost": 0.0, "risk_score": 0.60,
             "top_3_reasons": [f"scoring degraded ({derr}); human review required"]}
        d = {"risk_level": "Medium", "recommended_action": "review"}
        gf = {"boost": 0.0, "fraud_neighbors_2hop": 0, "fraud_neighbor_sample": []}
        feats = {}
        nar = {"narrative": (f"Scoring is degraded ({derr}); this transfer could not be "
                            "evaluated automatically. Hold for human review."), "llm_used": False,
                 "faithful": True, "template": "degraded-fallback", "lang": req.lang}
    txn_id = f"LIVE-{uuid4().hex[:8]}"
    row = {"txn_id": txn_id, "sender": txn["sender_id"], "receiver": txn["receiver_id"],
           "amount": txn["amount"], "type": txn.get("type", "P2P"), "timestamp": txn["timestamp"],
           "device_id": txn["device_id"], "location": txn["location"], "channel": txn["channel"],
           "is_fraud": 0, "fraud_type": "none", "password_reset_flag": int(txn.get("password_reset_flag", 0))}
    if committer:
        store.append(row)  # committed: moves future features
    else:
        # Scoring-neutral: visible in timelines only, never moves anyone's features.
        store.append_live(row)
    blob = {"txn": txn, "feats": feats, "score": s, "decision": d,
            "graph": {"boost": gf["boost"],
                      "fraud_neighbors_2hop": gf["fraud_neighbors_2hop"],
                      "fraud_neighbor_sample": gf.get("fraud_neighbor_sample", [])}}
    live_cases[txn_id] = blob
    db.upsert_case(txn_id, blob, s["risk_score"], d["risk_level"], source="live")
    while len(live_cases) > MAX_LIVE_CASES:
        live_cases.pop(next(iter(live_cases)))
    latency_ms = (time.perf_counter() - t0) * 1000
    request_count += 1
    log.info("score %s -> %.3f %s (%.1fms degraded=%s)", txn_id, s["risk_score"],
             d["risk_level"], latency_ms, degraded)
    resp = {"txn_id": txn_id, "risk_score": s["risk_score"], "risk_level": d["risk_level"],
            "top_3_reasons": s["top_3_reasons"], "recommended_action": d["recommended_action"],
            "degraded": degraded, "deduplicated": False,
            "model_version": infer.MODEL_VERSION,
            "components": {**{k: s[k] for k in ("p_fraud", "anomaly", "graph_boost")},
                            "fraud_neighbors_2hop": gf["fraud_neighbors_2hop"],
                            "fraud_neighbor_sample": gf.get("fraud_neighbor_sample", []),
                            "latency_ms": round(latency_ms, 1)},
            "narrative": nar["narrative"]}
    if idem:
        db.put_idempotent(idem, resp)
    return resp


@app.get("/alerts")
def alerts(limit: int = Query(100, ge=1, le=500), level: str | None = None):
    # Persistent queue: CSV pre-score merged with DB cases (live scores and
    # restarts survive here), deduped, top-risk-first.
    seen, merged = set(), []
    for r in alert_cache:
        seen.add(r["txn_id"])
        merged.append({k: v for k, v in r.items()
                       if k not in ("feats", "graph_boost_raw", "label")})
    try:
        for row in db.list_open_cases(limit=500):
            if row["txn_id"] not in seen:
                seen.add(row["txn_id"])
                merged.append(db.row_to_public_case(row))
    except Exception as e:
        log.warning("alerts DB merge skipped: %s", e)
    merged.sort(key=lambda r: -float(r.get("risk_score", 0.0)))
    if level:
        merged = [r for r in merged if str(r.get("risk_level", "")).lower() == level.lower()]
    return {"alerts": merged[:limit], "count": len(merged[:limit])}


def _blob_to_case(txn_id: str, blob: dict) -> tuple[dict, dict, dict, dict]:
    """Normalize a persisted score blob to (txn, feats, case, graph_facts)."""
    txn, feats = blob["txn"], blob["feats"]
    graph = blob.get("graph", {})
    c = {**blob.get("score", {}), **blob.get("decision", {}),
         "fraud_neighbors_2hop": graph.get("fraud_neighbors_2hop", 0),
         "fraud_neighbor_sample": graph.get("fraud_neighbor_sample", []),
         "txn_id": txn_id,
         **{k: txn.get(k) for k in ("sender_id", "receiver_id", "amount", "channel",
                                    "device_id", "location", "timestamp")}}
    return txn, feats, c, graph


@app.get("/case/{txn_id}")
def case(txn_id: str, lang: str = "en"):
    assert store is not None
    status = "open"
    live = live_cases.get(txn_id)
    rows = [r for r in alert_cache if r["txn_id"] == txn_id]
    db_row = None if (live is not None or rows) else db.get_case(txn_id)
    if live is None and not rows and db_row is None:
        raise HTTPException(404, "case not found; score it first via POST /score or pick a queued alert")
    if live is not None:
        txn, feats, c, gfacts = _blob_to_case(
            txn_id, {"txn": live["txn"], "feats": live["feats"],
                     "score": live["score"], "decision": live["decision"],
                     "graph": live["graph"]})
    elif db_row is not None:
        import json
        try:
            blob = json.loads(db_row["payload"])
        except Exception:
            raise HTTPException(500, "stored case payload is corrupt")
        txn, feats, c, gfacts = _blob_to_case(txn_id, blob)
        status = db_row.get("status", "open")
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
    public["status"] = status
    log.info("case %s viewed (lang=%s)", txn_id, lang)
    return {**public, "narrative": nar["narrative"], "timeline": timeline}


@app.post("/decision")
def decision(req: DecisionRequest, request: Request):
    assert store is not None
    if not _rate_limit_ok(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "rate limit exceeded, retry in a minute")
    if req.txn_id not in _known_txn_ids():
        raise HTTPException(404, "unknown txn_id; score it or pick a queued alert first")
    me = auth.current_user(request)
    analyst = me["username"] if me else req.analyst
    entry, updated = db.upsert_decision(req.txn_id, analyst, req.decision, req.note,
                                        model_version=infer.MODEL_VERSION)
    log.info("decision %s -> %s by %s (updated=%s)", req.txn_id, req.decision, analyst, updated)
    return {"ok": True, "updated": updated,
            "logged": {"txn_id": req.txn_id, "decision": req.decision,
                       "analyst": analyst, "note": req.note},
            "pending_retrain": db.count_decisions()}


# Analyst console (no build step): served from /web. API routes above take
# precedence; this mount only handles / and static files.
try:
    from fastapi.staticfiles import StaticFiles

    class _NoCacheStatic(StaticFiles):
        """Same-origin console redeploys often; with no cache headers browsers
        keep old HTML alongside old JS, which yields dead buttons and stuck
        queues. The files are tiny, so skip the cache entirely."""

        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            if resp.status_code == 200:
                resp.headers["Cache-Control"] = "no-store, must-revalidate"
            return resp

    _WEB_DIR = Path(__file__).resolve().parent.parent / "web"
    if _WEB_DIR.exists():
        app.mount("/", _NoCacheStatic(directory=str(_WEB_DIR), html=True), name="web")
except Exception:
    pass
