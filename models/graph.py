"""Graph risk: NetworkX directed money-flow graph + 2-hop fraud proximity.

Rule (kept separate from ML): if receiver is within 2 hops of >= K known-fraud
nodes, add boost. Visualized on the analyst UI; here we return facts for the
LLM investigator + score boost.
"""
from __future__ import annotations

from collections import deque

import networkx as nx
import pandas as pd


def build_graph(txns: pd.DataFrame, max_edges: int = 200_000) -> nx.DiGraph:
    g = nx.DiGraph()
    df = txns.tail(max_edges) if len(txns) > max_edges else txns
    for _, r in df.iterrows():
        s, t = r["sender"], r["receiver"]
        if g.has_edge(s, t):
            g[s][t]["weight"] += 1
            g[s][t]["amount"] += float(r["amount"])
        else:
            g.add_edge(s, t, weight=1, amount=float(r["amount"]))
    return g


def fraud_nodes(txns: pd.DataFrame) -> set:
    f = txns[txns["is_fraud"] == 1]
    return set(f["sender"].tolist()) | set(f["receiver"].tolist())


def network_risk(receiver: str, g: nx.DiGraph, fraud: set,
                 k_threshold: int = 3, boost_per_hit: float = 0.15,
                 max_boost: float = 0.30, max_sample: int = 10,
                 max_visits: int = 20000) -> dict:
    """BFS 2 hops on the undirected view from receiver; count fraud neighbors.

    Returns a small sample of fraud neighbor IDs so the analyst console can
    render the mule-ring without a separate graph query. max_visits bounds
    work against super-nodes (popular merchants) so one request cannot
    stall the worker.
    """
    if receiver not in g:
        return {"boost": 0.0, "fraud_neighbors_2hop": 0,
                "is_direct_fraud_neighbor": False, "fraud_neighbor_sample": []}
    ug = g.to_undirected(as_view=True)
    seen = {receiver}
    q = deque([(receiver, 0)])
    hits = 0
    direct = False
    visits = 0
    sample: list[str] = []
    while q:
        node, d = q.popleft()
        if d >= 2:
            continue
        for nb in ug.neighbors(node):
            visits += 1
            if visits > max_visits:
                break
            if nb in seen:
                continue
            seen.add(nb)
            if nb in fraud:
                hits += 1
                if d == 0:
                    direct = True
                if len(sample) < max_sample:
                    sample.append(str(nb))
            q.append((nb, d + 1))
        if visits > max_visits:
            break
    boost = min(hits * boost_per_hit, max_boost) if hits >= k_threshold else 0.0
    # small partial credit for direct proximity even below threshold
    if hits > 0 and hits < k_threshold and direct:
        boost = min(boost_per_hit / 2, max_boost)
    return {"boost": round(boost, 4), "fraud_neighbors_2hop": hits,
            "is_direct_fraud_neighbor": direct, "fraud_neighbor_sample": sample}
