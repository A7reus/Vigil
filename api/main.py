"""Vigil FastAPI: POST /score, GET /alerts, GET /case/:id, POST /decision."""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import networkx as nx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import auth
from api.decisions import DecisionLog
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
decision_log: DecisionLog | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, graph, fraud_set, alert_cache, startup_info, decision_log
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
    from features.build import FEATURE_COLS
    assert set(infer._cols) == set(FEATURE_COLS), \
        "API serves full-feature artifacts only (intersect sets are eval-only)"
    store = HistoryStore()
    # Audit trail lives in Postgres (DATABASE_URL required — fail fast here,
    # not mid-demo). Replaces the phase-1 process-local list.
    decision_log = DecisionLog()
    # Accounts + case workflow tables share the same database; demo seeds
    # only fire on empty user tables (VIGIL_SEED_DEMO=0 disables).
    auth.ensure_schema()
    auth.seed_demo_users()
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


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    # HEAD exists for uptime monitors (UptimeRobot pings /health that way);
    # Starlette strips the body automatically, same 200 either way.
    return {"ok": True, "history_rows": 0 if store is None else len(store.txns),
            "graph_nodes": 0 if graph is None else graph.number_of_nodes(),
            "queue_size": len(alert_cache), "startup": startup_info}


@app.get("/metrics")
def metrics():
    """Lightweight ops counters (monitoring expectation in the guideline)."""
    return {"requests_scored": request_count, "queue_size": len(alert_cache),
            "decisions_logged": 0 if decision_log is None else len(decision_log),
            "startup": startup_info}


# Exact score-time state for LIVE txns (feats + graph facts), so /case/{live_id}
# resolves without recomputing. Capped; evicts oldest first.
live_cases: dict[str, dict] = {}
MAX_LIVE_CASES = 500

# Per-IP token buckets for write endpoints (demo-grade flood protection; the
# scoring state itself is already isolated from LIVE traffic in store.py).
_rate_hits: dict[str, list[float]] = {}


def _require_key(request: Request) -> None:
    """Shared-secret auth for writes. Empty key = open (judging demos);
    set VIGIL_API_KEY anywhere exposed — clients send it as X-API-Key."""
    want = os.getenv("VIGIL_API_KEY", "")
    if not want:
        return
    if request.headers.get("x-api-key") != want:
        raise HTTPException(401, "missing or wrong X-API-Key")


def _writer_ok(request: Request) -> dict | None:
    """Writes accept a signed-in analyst/admin (Bearer token) or the service
    API key. Returns the user when a token was used, else None (open/key)."""
    u = auth.current_user(request)
    if u is not None:
        return u
    _require_key(request)
    return None


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


_TXN_KEYS = ("sender_id", "receiver_id", "amount", "channel", "device_id",
             "location", "timestamp", "type")


def _txn_snapshot(txn_id: str) -> dict:
    """Scored-transfer fields for the decision audit row, so retraining can
    rebuild the example later — including LIVE ids that never entered
    committed history (their only trace is this snapshot)."""
    for r in alert_cache:
        if r["txn_id"] == txn_id:
            return {k: r.get(k) for k in _TXN_KEYS}
    live = live_cases.get(txn_id)
    if live is not None:
        return {k: live["txn"].get(k) for k in _TXN_KEYS}
    return {}


@app.post("/score")
def score(req: ScoreRequest, request: Request):
    global request_count
    assert store is not None and graph is not None
    _writer_ok(request)
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
    # Scoring path: template narrative, answered in ms (see narrate docstring).
    nar = narrate(txn, feats, {**s, **d}, gf, lang=req.lang, live=False)
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
    return {**public, "narrative": nar["narrative"], "timeline": timeline,
            "workflow": auth.get_case_state(txn_id)}


@app.post("/decision")
def decision(req: DecisionRequest, request: Request):
    assert store is not None and decision_log is not None
    user = _writer_ok(request)
    if not _rate_limit_ok(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "rate limit exceeded, retry in a minute")
    if req.txn_id not in _known_txn_ids():
        raise HTTPException(404, "unknown txn_id; score it or pick a queued alert first")
    body = req.model_dump()
    if user is not None:
        # A signed-in analyst decides as themselves — no impersonation.
        body["analyst"] = user["username"]
    entry = {**body, "at": pd.Timestamp.now("UTC").isoformat(),
             "txn": json.dumps(_txn_snapshot(req.txn_id), default=str)}
    updated = decision_log.upsert(entry)
    log.info("decision %s -> %s by %s (updated=%s)", req.txn_id, req.decision, body["analyst"], updated)
    logged = {k: body[k] for k in ("txn_id", "decision", "analyst", "note")}
    return {"ok": True, "updated": updated, "logged": logged,
            "pending_retrain": len(decision_log)}


# ---------------------------------------------------------------------------
# Accounts + RBAC (Postgres; no SQLite anywhere). Guests keep the open demo;
# signed-in analysts decide as themselves; admins approve users and retrain.
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@app.post("/auth/register", status_code=201)
def register(req: RegisterRequest):
    auth.check_username(req.username)
    auth.check_password(req.password)
    u = auth.create_user(req.username, auth.hash_password(req.password))
    if u is None:
        raise HTTPException(409, "username already taken")
    log.info("registered %s (pending approval)", req.username)
    return {"user": auth.public_user(u),
            "message": "registered; awaiting admin approval before login"}


@app.post("/auth/login")
def login(req: LoginRequest, request: Request):
    if not _rate_limit_ok("login:" + _client_ip(request)):
        raise HTTPException(429, "too many login attempts, retry in a minute")
    u = auth.get_user_by_username(req.username)
    if not u or not auth.verify_password(req.password, u["pw_hash"]):
        raise HTTPException(401, "invalid credentials")
    if u["status"] == "pending":
        raise HTTPException(403, "account pending admin approval")
    if u["status"] != "active":
        raise HTTPException(403, "account disabled")
    token, exp = auth.issue_token(u["id"])
    log.info("login %s", u["username"])
    return {"token": token, "expires_at": exp, "user": auth.public_user(u)}


@app.get("/auth/me")
def me(request: Request):
    return {"user": auth.need_user(request)}


@app.post("/auth/logout")
def logout(request: Request):
    t = auth.bearer_token(request)
    if t:
        import hashlib
        auth.revoke_session(hashlib.sha256(t.encode()).hexdigest())
    return {"ok": True}


@app.get("/admin/users")
def admin_users(request: Request):
    auth.need_admin(request)
    return {"users": auth.list_users()}


@app.post("/admin/users/{uid}/approve")
def admin_approve(uid: int, request: Request):
    auth.need_admin(request)
    u = auth.set_user_status(uid, "active")
    if u is None:
        raise HTTPException(404, "unknown user")
    return {"user": auth.public_user(u)}


@app.post("/admin/users/{uid}/disable")
def admin_disable(uid: int, request: Request):
    auth.need_admin(request)
    u = auth.set_user_status(uid, "disabled")
    if u is None:
        raise HTTPException(404, "unknown user")
    return {"user": auth.public_user(u)}


@app.post("/admin/users/{uid}/enable")
def admin_enable(uid: int, request: Request):
    return admin_approve(uid, request)


@app.get("/admin/decisions")
def admin_decisions(request: Request, analyst: str | None = None,
                    limit: int = Query(200, ge=1, le=2000)):
    auth.need_admin(request)
    assert decision_log is not None
    rows = decision_log.all()
    if analyst:
        rows = [r for r in rows if r.get("analyst") == analyst]
    return {"decisions": rows[:limit], "count": min(len(rows), limit)}


@app.get("/admin/cases")
def admin_cases(request: Request, status: str | None = None,
                analyst: str | None = None,
                limit: int = Query(200, ge=1, le=2000)):
    """Case workflow board: state rows joined with whatever the queue knows."""
    auth.need_admin(request)
    states = auth.list_case_states(status=status, assignee=analyst, limit=limit)
    known = {r["txn_id"]: r for r in alert_cache}
    known.update({tid: {"txn_id": tid, **live["txn"],
                        "risk_score": live["score"]["risk_score"],
                        "risk_level": live["decision"]["risk_level"]}
                for tid, live in live_cases.items()})
    out = []
    for s in states:
        row = {"txn_id": s["txn_id"], "status": s["status"],
               "assignee": s.get("assignee"), "updated_at": s.get("updated_at")}
        k = known.get(s["txn_id"], {})
        for f in ("risk_score", "risk_level", "amount", "sender_id", "receiver_id"):
            if f in k:
                row[f] = k[f]
        out.append(row)
    return {"cases": out, "count": len(out)}


@app.post("/cases/{txn_id}/assign")
def assign_case(txn_id: str, req: CaseAssignRequest, request: Request):
    u = auth.need_user(request)
    if txn_id not in _known_txn_ids():
        raise HTTPException(404, "unknown txn_id; score it or pick a queued alert first")
    if u["role"] != "admin" and req.analyst != u["username"]:
        raise HTTPException(403, "analysts can only assign cases to themselves")
    s = auth.set_case_state(txn_id, status="assigned", assignee=req.analyst)
    log.info("case %s assigned to %s by %s", txn_id, req.analyst, u["username"])
    return s


@app.post("/cases/{txn_id}/status")
def case_status(txn_id: str, req: CaseStatusRequest, request: Request):
    u = auth.need_user(request)
    if txn_id not in _known_txn_ids():
        raise HTTPException(404, "unknown txn_id; score it or pick a queued alert first")
    s = auth.set_case_state(txn_id, status=req.status)
    log.info("case %s -> %s by %s", txn_id, req.status, u["username"])
    return s


@app.get("/cases")
def list_cases(request: Request, status: str | None = None,
               analyst: str | None = None,
               limit: int = Query(200, ge=1, le=2000)):
    auth.need_user(request)
    return {"cases": auth.list_case_states(status=status, assignee=analyst, limit=limit)}


# ---------------------------------------------------------------------------
# Admin retraining: POST starts scripts/retrain in a background worker,
# GET polls it. Same SHIP/HOLD gate as the CLI; --apply equivalent copies
# the winner over the live artifacts and reloads in-process. On ephemeral
# hosting (Render free) the new model lasts until the next redeploy —
# the verdict says so plainly.
# ---------------------------------------------------------------------------
_retrain_executor = None
_retrain_job: dict = {"state": "idle", "result": None}


def _retrain_worker(data_dir: str, artifacts_dir: str, apply: bool) -> dict:
    import tempfile
    from scripts import retrain
    tmp = tempfile.mkdtemp(prefix="vigil_retrain_")
    try:
        res = retrain.main(["--data", data_dir, "--artifacts", artifacts_dir,
                            "--out-data", str(Path(tmp) / "aug"),
                            "--out-artifacts", str(Path(tmp) / "art2")] +
                           (["--apply"] if apply else []))
        if apply and res.get("applied"):
            infer.load_artifacts(artifacts_dir)  # serve the winner now
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/admin/retrain")
def admin_retrain(request: Request, apply: bool = False,
                  data_dir: str = "data", artifacts_dir: str = "artifacts"):
    auth.need_admin(request)
    global _retrain_executor
    if _retrain_job["state"] == "running":
        raise HTTPException(409, "a retrain job is already running")
    import concurrent.futures
    if _retrain_executor is None:
        _retrain_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _retrain_job.update(state="running", result=None)
    fut = _retrain_executor.submit(_retrain_worker, data_dir, artifacts_dir, apply)

    def _done(f):
        try:
            _retrain_job.update(state="done", result=f.result())
        except Exception as e:  # never leave the status stuck on running
            _retrain_job.update(state="error", result={"error": str(e)})
            log.exception("retrain job failed")

    fut.add_done_callback(_done)
    log.info("retrain started by admin (apply=%s)", apply)
    return {"state": "running"}


@app.get("/admin/retrain/status")
def admin_retrain_status(request: Request):
    auth.need_admin(request)
    return _retrain_job


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
